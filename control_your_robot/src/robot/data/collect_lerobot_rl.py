"""
扩展的 LeRobot 数据收集类，支持强化学习所需的额外字段：
1. Intervention Flag: 标记人工干预 (1) 或自主操作 (0)
2. Value Labels: 奖励标签，用于强化学习训练
3. Reward / Reward Labels: episode 级回报监督
4. adv_ind: 优势指示字符串
"""
import sys
sys.path.append("./")

import gc
import numpy as np
import torch
import cv2
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from robot.utils.base.data_handler import debug_print

KEY_BANNED = ["timestamp"]
VALUE_LABEL_KEY = "value_label"
LEGACY_VALUE_LABEL_KEYS = ("value_lable", "value")
REWARD_KEY = "reward"
REWARD_LABEL_KEY = "reward_label"
ADV_IND_KEY = "adv_ind"
INTERVENTION_KEY = "intervention"


class CollectLeRobotRL:
    """支持强化学习标签的 LeRobot 数据收集器。"""

    def __init__(
        self,
        repo_id: str,
        output_dir: str,
        task_name: str,
        fps: int = 10,
        robot_type: str = "piper",
        state_dim: int = 7,
        action_dim: int = 7,
        image_size: tuple = (720,1280),
        camera_keys: dict = None,
        move_check: bool = True,
        tolerance: float = 0.002,  
        penalty_value: float = -1.0,  # 失败时的惩罚值 (-c)
        image_writer_processes: int = 5,
        image_writer_threads: int = 10,
        image_format: str = "png",
        release_dataset_after_save: bool = False,
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
            camera_keys: 相机映射
            move_check: 是否检查机器人移动
            tolerance: 移动检测容差
            penalty_value: 失败时的惩罚值 (默认 -1.0)
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
        self.penalty_value = penalty_value
        self.image_writer_processes = int(max(0, image_writer_processes))
        self.image_writer_threads = int(max(0, image_writer_threads))
        self.image_format = self._normalize_image_format(image_format)
        self.release_dataset_after_save = bool(release_dataset_after_save)

        # 当前 episode 的数据缓存
        self.episode_buffer = []
        self.last_controller_data = None

        # 干预标记缓存
        self.intervention_flags = []

        # LeRobot 数据集
        self.dataset = None
        self.dataset_created = False
        self.value_label_key = VALUE_LABEL_KEY

    @staticmethod
    def _normalize_image_format(image_format: str) -> str:
        normalized = str(image_format).strip().lower().lstrip(".")
        if normalized == "jpeg":
            normalized = "jpg"
        if normalized not in {"png", "jpg"}:
            raise ValueError(f"Unsupported image_format={image_format!r}; expected png or jpg.")
        return normalized

    def _patch_lerobot_image_format(self) -> None:
        """Patch LeRobot temporary frame extension before image writer starts."""
        if self.image_format == "png":
            return
        image_path_template = (
            "images/{image_key}/episode_{episode_index:06d}/"
            f"frame_{{frame_index:06d}}.{self.image_format}"
        )
        try:
            import lerobot.common.datasets.utils as dataset_utils
            import lerobot.common.datasets.lerobot_dataset as dataset_module

            dataset_utils.DEFAULT_IMAGE_PATH = image_path_template
            dataset_module.DEFAULT_IMAGE_PATH = image_path_template
        except Exception as exc:
            debug_print(
                "collect_lerobot_rl",
                f"Failed to patch LeRobot image format to {self.image_format}: {exc}",
                "WARNING",
            )

    def _release_dataset_after_episode(self) -> None:
        """Optionally release LeRobotDataset internals to avoid long-session RAM growth."""
        if not self.release_dataset_after_save or self.dataset is None:
            return
        try:
            if hasattr(self.dataset, "stop_image_writer"):
                self.dataset.stop_image_writer()
        except Exception as exc:
            debug_print("collect_lerobot_rl", f"stop_image_writer failed: {exc}", "WARNING")
        self.dataset = None
        self.dataset_created = False
        gc.collect()

    def _get_dataset_feature_keys(self) -> set[str]:
        """返回当前数据集 schema 中的 feature key 集合。"""
        if self.dataset is None:
            return set()

        feature_keys = None
        for attr_name in ("features", "hf_features"):
            feature_obj = getattr(self.dataset, attr_name, None)
            if feature_obj is None:
                continue
            if hasattr(feature_obj, "keys"):
                feature_keys = set(feature_obj.keys())
                break

        return feature_keys or set()

    def _sync_value_label_key_from_dataset(self):
        """根据已加载数据集的 schema 决定价值标签字段名。"""
        feature_keys = self._get_dataset_feature_keys()
        if not feature_keys:
            return

        if VALUE_LABEL_KEY in feature_keys:
            self.value_label_key = VALUE_LABEL_KEY
        else:
            for legacy_key in LEGACY_VALUE_LABEL_KEYS:
                if legacy_key in feature_keys:
                    self.value_label_key = legacy_key
                    break

    def _get_required_feature_keys(self) -> set[str]:
        """返回写入当前收集器所需的字段集合。"""
        return {
            INTERVENTION_KEY,
            self.value_label_key,
            REWARD_KEY,
            REWARD_LABEL_KEY,
            ADV_IND_KEY,
        }

    def _get_missing_required_feature_keys(self) -> list[str]:
        """检查当前数据集是否缺少本收集器写入所需字段。"""
        feature_keys = self._get_dataset_feature_keys()
        if not feature_keys:
            return []
        return sorted(self._get_required_feature_keys() - feature_keys)

    def _ensure_dataset_schema_for_writes(self):
        """在写入前验证数据集 schema，避免静默写入不兼容数据。"""
        missing_features = self._get_missing_required_feature_keys()
        if not missing_features:
            return

        output_path = self.output_dir / self.repo_id
        missing_str = ", ".join(missing_features)
        raise RuntimeError(
            "LeRobot dataset schema is missing required fields for RL collection: "
            f"{missing_str}. Dataset path: {output_path}. "
            "Create a new dataset directory with the updated schema before collecting."
        )

    def _create_dataset(self):
        """创建或加载 LeRobot 数据集（第一次收集时调用）"""
        if self.dataset_created:
            return

        self._patch_lerobot_image_format()
        output_path = self.output_dir / self.repo_id

        # 检查数据集是否已存在（必须有 meta 目录才算完整数据集）
        if output_path.exists() and (output_path / "meta").exists():
            debug_print("collect_lerobot_rl", f"检测到现有 LeRobot 数据集: {output_path}", "INFO")

            # 【优化方案】Monkey patch 跳过耗时的 timestamp 检查
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
                debug_print("collect_lerobot_rl", f"开始快速加载（跳过 timestamp 验证）...", "INFO")

                # 加载现有数据集（由于禁用了检查，会快很多）
                self.dataset = LeRobotDataset(
                    repo_id=self.repo_id,
                    root=output_path,
                )
                self._sync_value_label_key_from_dataset()

                # 启动 image writer
                self.dataset.start_image_writer(
                    num_processes=self.image_writer_processes,
                    num_threads=self.image_writer_threads,
                )

                debug_print("collect_lerobot_rl", f"✓ 数据集加载成功（快速模式，当前有 {self.dataset.num_episodes} 个 episodes）", "INFO")

            except Exception as e:
                debug_print("collect_lerobot_rl", f"加载数据集失败: {e}", "ERROR")
                import traceback
                debug_print("collect_lerobot_rl", f"详细错误:\n{traceback.format_exc()}", "ERROR")
                raise
            finally:
                # 恢复原始函数
                lrd_module.check_timestamps_sync = original_check_timestamps_sync
                torch.stack = original_stack
        else:
            debug_print("collect_lerobot_rl", f"创建新的 LeRobot 数据集: {output_path}", "INFO")

            # 构建 features 配置
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
                # 添加干预标记
                INTERVENTION_KEY: {
                    "dtype": "int64",
                    "shape": (1,),
                    "names": ["intervention_flag"],
                },
                # 添加价值标签
                VALUE_LABEL_KEY: {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": [VALUE_LABEL_KEY],
                },
                REWARD_KEY: {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": [REWARD_KEY],
                },
                REWARD_LABEL_KEY: {
                    "dtype": "float32",
                    "shape": (1,),
                    "names": [REWARD_LABEL_KEY],
                },
                ADV_IND_KEY: {
                    "dtype": "string",
                    "shape": (1,),
                    "names": [ADV_IND_KEY],
                },
            }

            # 添加图像特征
            for camera_name, lerobot_key in self.camera_keys.items():
                features[lerobot_key] = {
                    "dtype": "image",
                    "shape": (3, self.image_size[0], self.image_size[1]),
                    "names": ["channels", "height","width"],
                }

            # 创建数据集
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                root=output_path,
                robot_type=self.robot_type,
                fps=self.fps,
                features=features,
                image_writer_threads=self.image_writer_threads,
                image_writer_processes=self.image_writer_processes,
            )
            self.value_label_key = VALUE_LABEL_KEY
            debug_print("collect_lerobot_rl", f"LeRobot 数据集创建成功", "INFO")

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
        return self._get_state_names()

    def collect(self, controllers_data, sensors_data, is_intervention: bool = False):
        """
        收集一帧数据到缓存

        Args:
            controllers_data: 控制器数据
            sensors_data: 传感器数据
            is_intervention: 是否为人工干预模式 (True=1, False=0)
        """
        # 第一次收集时创建数据集
        if not self.dataset_created:
            self._create_dataset()
        self._ensure_dataset_schema_for_writes()

        # 检查机器人是否移动
        if self.move_check:
            if self.last_controller_data is None:
                self.last_controller_data = controllers_data
            else:
                if not self._move_check_success(controllers_data):
                    debug_print("collect_lerobot_rl", "机器人未移动，跳过此帧", "DEBUG")
                    self.last_controller_data = controllers_data
                    return
                self.last_controller_data = controllers_data

        # 合并数据
        frame_data = {
            "controllers": controllers_data,
            "sensors": sensors_data,
        }

        self.episode_buffer.append(frame_data)
        self.intervention_flags.append(1 if is_intervention else 0)

        debug_print("collect_lerobot_rl",
                   f"收集帧 {len(self.episode_buffer)} (干预: {is_intervention})",
                   "DEBUG")

    def save_episode(
        self,
        success: bool = True,
        adv_ind_value: str = "positive",
        failure_terminal_reward_label: float = -1.0,
    ):
        """
        保存当前 episode 到 LeRobot 数据集

        Args:
            success: 任务是否成功
            adv_ind_value: 为该 episode 每一帧写入的 adv_ind 字符串
            failure_terminal_reward_label: 失败 episode 最后一帧的 reward_label
        """
        if len(self.episode_buffer) == 0:
            debug_print("collect_lerobot_rl", "Episode 为空，跳过保存", "WARNING")
            return

        if not self.dataset_created:
            debug_print("collect_lerobot_rl", "数据集未创建，无法保存", "ERROR")
            return

        self._ensure_dataset_schema_for_writes()

        debug_print("collect_lerobot_rl",
                   f"开始保存 episode ({len(self.episode_buffer)} 帧, 成功: {success}, adv_ind: {adv_ind_value})",
                   "INFO")

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

        # 计算价值标签
        episode_length = len(self.episode_buffer)
        value_labels = self._compute_value_labels(episode_length, success)
        rewards = self._compute_rewards(episode_length, success)
        reward_labels = self._compute_reward_labels(
            episode_length,
            success,
            failure_terminal_reward_label=failure_terminal_reward_label,
        )
        adv_ind_labels = self._compute_adv_ind(episode_length, adv_ind_value)

        # 逐帧添加到 LeRobot 数据集
        for i, frame in enumerate(self.episode_buffer):
            lerobot_frame = {
                "state": states[i],
                "actions": actions[i],
                INTERVENTION_KEY: torch.tensor([self.intervention_flags[i]], dtype=torch.int64),
                self.value_label_key: torch.tensor([value_labels[i]], dtype=torch.float32),
                REWARD_KEY: torch.tensor([rewards[i]], dtype=torch.float32),
                REWARD_LABEL_KEY: torch.tensor([reward_labels[i]], dtype=torch.float32),
                ADV_IND_KEY: adv_ind_labels[i],
            }

            # 添加图像
            images = self._extract_images(frame["sensors"])
            lerobot_frame.update(images)

            self._add_frame(lerobot_frame)

        # 保存 episode
        self.dataset.save_episode()
        debug_print("collect_lerobot_rl",
                   f"Episode 保存成功 ({len(self.episode_buffer)} 帧)",
                   "INFO")

        # 清空缓存
        self.episode_buffer = []
        self.intervention_flags = []
        self.last_controller_data = None
        self._release_dataset_after_episode()

    def clear_current_episode(self):
        """清空当前 episode 缓存（用于放弃不满意的数据）"""
        frame_count = len(self.episode_buffer)
        self.episode_buffer = []
        self.intervention_flags = []
        self.last_controller_data = None
        debug_print("collect_lerobot_rl", f"已清空当前 episode 缓存 ({frame_count} 帧)", "INFO")

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
            self.dataset.add_frame(frame_without_task, task=self.task_name)
            return
        except ValueError:
            try:
                self.dataset.add_frame(frame_without_task)
                return
            except TypeError:
                self.dataset.add_frame(frame_without_task, task=self.task_name)
                return

    def _compute_value_labels(self, episode_length: int, success: bool) -> np.ndarray:
        """
        计算价值标签

        成功轨迹 (success=True):
            - 中间帧: value = -(T - t) / T  (范围 [-1, 0])
            - 最后一帧: value = 0

        失败轨迹 (success=False):
            - 所有帧: value = penalty_value (默认 -1.0)

        Args:
            episode_length: episode 总帧数 T
            success: 任务是否成功

        Returns:
            value_labels: [T], 价值标签数组
        """
        T = episode_length

        if success:
            # 成功轨迹: 使用归一化公式
            t = np.arange(T)
            value_labels = -(T - t) / T
            value_labels[-1] = 0.0  # 最后一帧给 0
        else:
            # 失败轨迹: 所有帧都是 penalty_value
            value_labels = np.full(T, self.penalty_value, dtype=np.float32)

        return value_labels.astype(np.float32)

    def _compute_rewards(self, episode_length: int, success: bool) -> np.ndarray:
        """
        计算逐帧 reward。

        成功轨迹:
            - 最后一帧 reward = 1
            - 其余帧 reward = 0

        失败轨迹:
            - 所有帧 reward = 0
        """
        rewards = np.zeros(episode_length, dtype=np.float32)
        if success:
            rewards[-1] = 1.0
        return rewards

    def _compute_reward_labels(
        self,
        episode_length: int,
        success: bool,
        failure_terminal_reward_label: float = -1.0,
    ) -> np.ndarray:
        """
        计算逐帧 reward_label。

        所有非终帧:
            - reward_label = -1 / T

        成功轨迹终帧:
            - reward_label = 0

        失败轨迹终帧:
            - reward_label = failure_terminal_reward_label
        """
        reward_labels = np.full(
            episode_length,
            -1.0 / float(episode_length),
            dtype=np.float32,
        )
        reward_labels[-1] = 0.0 if success else failure_terminal_reward_label
        return reward_labels

    def _compute_adv_ind(self, episode_length: int, adv_ind_value: str) -> list[str]:
        """生成每一帧的 adv_ind 字符串。"""
        return [adv_ind_value] * episode_length

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
