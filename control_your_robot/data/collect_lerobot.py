"""
直接生成 LeRobot 格式的数据收集类
不需要中间 HDF5 格式，直接使用 LeRobot API 保存数据
"""
import sys
sys.path.append("./")

import os
import numpy as np
import torch
import cv2
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot.utils.base.data_handler import debug_print

KEY_BANNED = ["timestamp"]


class CollectLeRobot:
    """直接收集并保存为 LeRobot 格式的数据收集器"""

    def __init__(
        self,
        repo_id: str,
        output_dir: str,
        task_name: str,
        fps: int = 10,
        robot_type: str = "piper",
        state_dim: int = 7,  # 单臂：6关节+1夹爪
        action_dim: int = 7,
        image_size: tuple = (480, 640),  # (height, width)
        camera_keys: dict = None,  # {"cam_head": "image", "cam_wrist": "wrist_image"}
        move_check: bool = True,
        tolerance: float = 0.0001,
    ):
        """
        Args:
            repo_id: LeRobot 数据集 ID
            output_dir: 输出目录
            task_name: 任务名称
            fps: 采集频率
            robot_type: 机器人类型
            state_dim: 状态维度
            action_dim: 动作维度
            image_size: 图像尺寸 (height, width)
            camera_keys: 相机映射，例如 {"cam_head": "image", "cam_wrist": "wrist_image"}
            move_check: 是否检查机器人是否移动
            tolerance: 移动检测容差
        """
        self.repo_id = repo_id
        self.output_dir = Path(output_dir)
        self.task_name = task_name
        self.fps = fps
        self.robot_type = robot_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.image_size = image_size
        self.camera_keys = camera_keys or {}
        self.move_check = move_check
        self.tolerance = tolerance

        # 当前 episode 的数据缓存
        self.episode_buffer = []
        self.last_controller_data = None

        # LeRobot 数据集
        self.dataset = None
        self.dataset_created = False

    def _create_dataset(self):
        """创建或加载 LeRobot 数据集（第一次收集时调用）"""
        if self.dataset_created:
            return

        output_path = self.output_dir / self.repo_id

        # 检查数据集是否已存在
        if output_path.exists() and (output_path / "meta").exists():
            debug_print("collect_lerobot", f"检测到现有 LeRobot 数据集: {output_path}", "INFO")

            # 【优化方案】Monkey patch 跳过耗时的 timestamp 检查
            # 这是加载慢的根本原因：LeRobotDataset.__init__ 会加载所有 timestamp 并验证
            import lerobot.common.datasets.lerobot_dataset as lrd_module

            original_check_timestamps_sync = lrd_module.check_timestamps_sync

            # 临时禁用 timestamp 检查
            def noop_check(*args, **kwargs):
                pass

            lrd_module.check_timestamps_sync = noop_check

            # Patch torch.stack to fix compatibility issue
            original_stack = torch.stack
            def safe_stack(tensors, *args, **kwargs):
                if not isinstance(tensors, (list, tuple)):
                    try:
                        tensors = list(tensors)
                    except Exception:
                        pass
                if isinstance(tensors, list) and len(tensors) > 0 and isinstance(tensors[0], (int, float, np.number)):
                    return torch.tensor(tensors)
                return original_stack(tensors, *args, **kwargs)

            torch.stack = safe_stack

            try:
                debug_print("collect_lerobot", f"开始快速加载（跳过 timestamp 验证）...", "INFO")

                # 加载现有数据集（由于禁用了检查，会快很多）
                self.dataset = LeRobotDataset(
                    repo_id=self.repo_id,
                    root=output_path,
                )

                # 启动 image writer
                self.dataset.start_image_writer(
                    num_processes=5,
                    num_threads=10,
                )

                debug_print("collect_lerobot", f"✓ 数据集加载成功（快速模式，当前有 {self.dataset.num_episodes} 个 episodes）", "INFO")

            except Exception as e:
                debug_print("collect_lerobot", f"加载数据集失败: {e}", "ERROR")
                import traceback
                debug_print("collect_lerobot", f"详细错误:\n{traceback.format_exc()}", "ERROR")
                raise
            finally:
                # 恢复原始函数
                lrd_module.check_timestamps_sync = original_check_timestamps_sync
                torch.stack = original_stack
        else:
            debug_print("collect_lerobot", f"创建新的 LeRobot 数据集: {output_path}", "INFO")

            # 构建 features 配置
            # 注意：不在 features 中包含 task，而是通过 add_frame 的 task 参数传递
            features = {
                "state": {
                    "dtype": "float32",
                    "shape": (self.state_dim,),
                    "names": self._get_state_names(),
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (self.action_dim,),
                    "names": self._get_action_names(),
                },
            }

            # 添加图像特征
            for camera_name, lerobot_key in self.camera_keys.items():
                features[lerobot_key] = {
                    "dtype": "image",
                    "shape": (3, self.image_size[0], self.image_size[1]),
                    "names": ["channels", "height", "width"],
                }

            # 创建数据集
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                root=output_path,
                robot_type=self.robot_type,
                fps=self.fps,
                features=features,
                image_writer_threads=10,
                image_writer_processes=5,
            )
            debug_print("collect_lerobot", f"LeRobot 数据集创建成功", "INFO")

        self.dataset_created = True

    def _get_state_names(self):
        """生成 state 字段名"""
        if self.state_dim == 7:
            return ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]
        elif self.state_dim == 14:
            return [
                "left_joint_1", "left_joint_2", "left_joint_3", "left_joint_4", "left_joint_5", "left_joint_6", "left_gripper",
                "right_joint_1", "right_joint_2", "right_joint_3", "right_joint_4", "right_joint_5", "right_joint_6", "right_gripper"
            ]
        else:
            return [f"state_{i}" for i in range(self.state_dim)]

    def _get_action_names(self):
        """生成 action 字段名"""
        return self._get_state_names()  # action 和 state 结构相同

    def _build_features(self):
        """构建 features 配置（用于创建数据集）"""
        features = {
            "state": {
                "dtype": "float32",
                "shape": (self.state_dim,),
                "names": self._get_state_names(),
            },
            "actions": {
                "dtype": "float32",
                "shape": (self.action_dim,),
                "names": self._get_action_names(),
            },
        }

        # 添加图像特征
        for camera_name, lerobot_key in self.camera_keys.items():
            features[lerobot_key] = {
                "dtype": "image",
                "shape": (3, self.image_size[0], self.image_size[1]),
                "names": ["channels", "height", "width"],
            }

        return features

    def collect(self, controllers_data, sensors_data):
        """
        收集一帧数据到缓存

        Args:
            controllers_data: 控制器数据，例如 {"left_arm": {"joint": [...], "gripper": [...]}}
            sensors_data: 传感器数据，例如 {"cam_head": {"color": image_array}, "cam_wrist": {...}}
        """
        # 第一次收集时创建数据集
        if not self.dataset_created:
            self._create_dataset()

        # 检查机器人是否移动
        if self.move_check:
            if self.last_controller_data is None:
                self.last_controller_data = controllers_data
            else:
                if not self._move_check_success(controllers_data):
                    debug_print("collect_lerobot", "机器人未移动，跳过此帧", "INFO")
                    self.last_controller_data = controllers_data
                    return
                self.last_controller_data = controllers_data

        # 合并数据
        frame_data = {
            "controllers": controllers_data,
            "sensors": sensors_data,
        }

        self.episode_buffer.append(frame_data)

    def save_episode(self):
        """保存当前 episode 到 LeRobot 数据集"""
        if len(self.episode_buffer) == 0:
            debug_print("collect_lerobot", "Episode 为空，跳过保存", "WARNING")
            return

        if not self.dataset_created:
            debug_print("collect_lerobot", "数据集未创建，无法保存", "ERROR")
            return

        debug_print("collect_lerobot", f"开始保存 episode ({len(self.episode_buffer)} 帧)", "INFO")

        # 提取 state 和 action
        states = []
        for frame in self.episode_buffer:
            state = self._extract_state(frame["controllers"])
            states.append(state)

        # action = 下一帧的 state
        actions = []
        for i in range(len(states)):
            if i < len(states) - 1:
                actions.append(states[i + 1])
            else:
                actions.append(states[-1])  # 最后一帧重复

        # 逐帧添加到 LeRobot 数据集
        for i, frame in enumerate(self.episode_buffer):
            lerobot_frame = {
                "state": states[i],
                "actions": actions[i],
            }

            # 添加图像
            images = self._extract_images(frame["sensors"])
            lerobot_frame.update(images)

            self._add_frame(lerobot_frame)

        # 保存 episode
        self.dataset.save_episode()
        debug_print("collect_lerobot", f"Episode 保存成功 ({len(self.episode_buffer)} 帧)", "INFO")

        # 清空缓存
        self.episode_buffer = []

    def clear_current_episode(self):
        """清空当前 episode 缓存（用于放弃不满意的数据）"""
        frame_count = len(self.episode_buffer)
        self.episode_buffer = []
        self.last_controller_data = None
        debug_print("collect_lerobot", f"已清空当前 episode 缓存 ({frame_count} 帧)", "INFO")

    def _add_frame(self, lerobot_frame):
        """添加帧到 LeRobot 数据集（兼容不同版本 task 处理方式）"""
        frame_with_task = dict(lerobot_frame)
        frame_with_task["task"] = self.task_name
        frame_without_task = dict(lerobot_frame)
        frame_without_task.pop("task", None)

        try:
            self.dataset.add_frame(frame_with_task)
            return
        except TypeError:
            # 版本需要 add_frame(frame, task=...)
            self.dataset.add_frame(frame_without_task, task=self.task_name)
            return
        except ValueError:
            # task 不在 features 中，回退不带 task
            try:
                self.dataset.add_frame(frame_without_task)
                return
            except TypeError:
                self.dataset.add_frame(frame_without_task, task=self.task_name)
                return

    def _extract_state(self, controllers_data):
        """
        从控制器数据中提取 state

        单臂示例:
            controllers_data = {"left_arm": {"joint": [6维], "gripper": [1维]}}
            -> state = [7维]

        双臂示例:
            controllers_data = {
                "left_arm": {"joint": [6维], "gripper": [1维]},
                "right_arm": {"joint": [6维], "gripper": [1维]}
            }
            -> state = [14维]
        """
        state_list = []

        # 按固定顺序提取（left_arm -> right_arm）
        arm_order = ["left_arm", "right_arm"]

        for arm_name in arm_order:
            if arm_name in controllers_data:
                arm_data = controllers_data[arm_name]

                # 提取 joint
                if "joint" in arm_data:
                    joint = np.array(arm_data["joint"]).flatten()
                    state_list.append(joint)

                # 提取 gripper
                if "gripper" in arm_data:
                    gripper = np.array(arm_data["gripper"]).flatten()
                    state_list.append(gripper)

        state = np.concatenate(state_list).astype(np.float32)
        return torch.from_numpy(state)

    def _extract_images(self, sensors_data):
        """
        从传感器数据中提取图像

        Args:
            sensors_data: {"cam_head": {"color": image}, "cam_wrist": {"color": image}}

        Returns:
            {"image": torch.Tensor, "wrist_image": torch.Tensor}
        """
        images = {}

        for camera_name, lerobot_key in self.camera_keys.items():
            if camera_name in sensors_data:
                camera_data = sensors_data[camera_name]

                # 提取图像
                if "color" in camera_data:
                    img = camera_data["color"]
                    img = self._decode_image(img)
                    images[lerobot_key] = img

        return images

    def _decode_image(self, img_data):
        """解码图像数据并转换为 torch.Tensor"""
        # 如果是编码的图像（bytes），先解码
        if isinstance(img_data, (bytes, bytearray)):
            data = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(img_data, np.ndarray):
            img = img_data
        else:
            raise ValueError(f"不支持的图像数据类型: {type(img_data)}")

        # 确保是 RGB，uint8，(H, W, C)
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)

        return img

    def _move_check_success(self, controller_data: dict) -> bool:
        """
        判断机器人是否移动（任一关节变化超过容差）

        Args:
            controller_data: 当前控制器数据

        Returns:
            bool: True 表示移动，False 表示静止
        """
        for part, current_subdata in controller_data.items():
            previous_subdata = self.last_controller_data.get(part)
            if previous_subdata is None:
                return True  # 没有历史数据，视为移动

            if isinstance(current_subdata, dict):
                for key, current_value in current_subdata.items():
                    if key in KEY_BANNED:
                        continue

                    previous_value = previous_subdata.get(key)
                    if previous_value is None:
                        return True

                    current_arr = np.atleast_1d(current_value)
                    previous_arr = np.atleast_1d(previous_value)

                    if current_arr.shape != previous_arr.shape:
                        return True

                    if np.any(np.abs(current_arr - previous_arr) > self.tolerance):
                        return True
            else:
                current_arr = np.atleast_1d(current_subdata)
                previous_arr = np.atleast_1d(previous_subdata)

                if current_arr.shape != previous_arr.shape:
                    return True

                if np.any(np.abs(current_arr - previous_arr) > self.tolerance):
                    return True

        return False  # 所有值都在容差内，视为静止

    def get_dataset_path(self):
        """获取数据集保存路径"""
        return self.output_dir / self.repo_id
