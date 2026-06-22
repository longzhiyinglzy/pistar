"""
Piper 单臂机器人 - 直接生成 LeRobot 格式
"""
import sys
sys.path.append("./")

import numpy as np

from my_robot.base_robot_lerobot import RobotLeRobot
from my_robot.camera_config import get_piper_camera_serials

from robot.controller.Piper_controller import PiperController
from robot.sensor.Realsense_sensor import RealsenseSensor

# 默认起始位置（弧度）
DEFAULT_RESET_JOINT_POSITION_LEFT_ARM = [
    0.04092797,   # Joint 1
    1.34207091,   # Joint 2
    -0.73867569,  # Joint 3
    0.02720968,   # Joint 4
    1.13512722,   # Joint 5
    0.29129545,   # Joint 6
]


class PiperSingleLeRobot(RobotLeRobot):
    """Piper 单臂机器人 - 直接生成 LeRobot Libero 格式"""

    def __init__(
        self,
        repo_id: str = "piper_single_task",
        output_dir: str = "./datasets/lerobot_datasets",
        task_name: str = "complete the task",
        fps: int = 10,
        move_check: bool = True,
        arm_can: str = "can0",
        reset_joint_position: list[float] | None = None,
    ):
        self.camera_serials = get_piper_camera_serials("single")
        # 相机映射：从 sensor 名称到 LeRobot 字段名
        camera_keys = {
            "cam_head": "image",          # Libero 格式使用 "image"
            "cam_wrist": "wrist_image",   # Libero 格式使用 "wrist_image"
        }
        if "side" in self.camera_serials:
            camera_keys["cam_side"] = "side_image"

        super().__init__(
            repo_id=repo_id,
            output_dir=output_dir,
            task_name=task_name,
            fps=fps,
            robot_type="piper",
            state_dim=7,  # 单臂：6关节 + 1夹爪
            action_dim=7,
            image_size=(720,1280),  # 更新为 1280x720 分辨率
            camera_keys=camera_keys,
            move_check=move_check,
            tolerance=0.0005,
        )

        self.name = "piper_single_lerobot"
        self.arm_can = arm_can
        self.reset_joint_position = np.array(
            reset_joint_position if reset_joint_position is not None else DEFAULT_RESET_JOINT_POSITION_LEFT_ARM,
            dtype=float,
        )

        # 初始化控制器和传感器
        self.controllers = {
            "arm": {
                "left_arm": PiperController("left_arm"),
            },
        }
        self.sensors = {
            "image": {
                "cam_head": RealsenseSensor("cam_head"),
                "cam_wrist": RealsenseSensor("cam_wrist"),
            },
        }
        if "side" in self.camera_serials:
            self.sensors["image"]["cam_side"] = RealsenseSensor("cam_side")

    def reset(self):
        """重置机器人到初始位置"""
        self.controllers["arm"]["left_arm"].reset(self.reset_joint_position)

    def set_up(self):
        """初始化机器人硬件"""
        super().set_up()

        # 初始化控制器
        self.controllers["arm"]["left_arm"].set_up(self.arm_can)

        # 初始化传感器
        self.sensors["image"]["cam_head"].set_up(self.camera_serials["head"])
        self.sensors["image"]["cam_wrist"].set_up(self.camera_serials["wrist"])
        if "side" in self.camera_serials:
            self.sensors["image"]["cam_side"].set_up(self.camera_serials["side"])

        # 设置需要收集的数据类型
        self.set_collect_type({
            "arm": ["joint", "qpos", "gripper"],
            "image": ["color"]
        })

        print(f"✓ {self.name} 初始化成功！")
        print(f"  - 数据将直接保存为 LeRobot Libero 格式")
        print(f"  - 输出路径: {self.collection.get_dataset_path()}")


if __name__ == "__main__":
    import time

    # 测试示例
    robot = PiperSingleLeRobot(
        repo_id="test_piper_single",
        output_dir="./datasets/lerobot_datasets",
        task_name="test task",
        fps=10,
    )

    robot.set_up()

    # 收集测试
    robot.reset()
    for i in range(100):
        print(f"收集第 {i} 帧")
        data = robot.get()
        robot.collect(data)
        time.sleep(0.1)

    robot.finish()
    print("✓ Episode 保存成功！")
