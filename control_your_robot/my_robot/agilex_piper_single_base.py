import sys
sys.path.append("./")

import numpy as np

from my_robot.base_robot import Robot
from my_robot.camera_config import get_piper_camera_serials

from robot.controller.Piper_controller import PiperController
from robot.sensor.Realsense_sensor import RealsenseSensor

from robot.data.collect_any import CollectAny

# Define start position (in radians)
START_POSITION_ANGLE_LEFT_ARM = [
    0.04092797,   # Joint 1
    1.34207091,   # Joint 2
    -0.73867569,  # Joint 3
    0.02720968,   # Joint 4
    1.13512722,   # Joint 5
    0.29129545,   # Joint 6
]

# Define start position (in radians)
START_POSITION_ANGLE_RIGHT_ARM = [
    0,   # Joint 1
    0,    # Joint 2
    0,  # Joint 3
    0,   # Joint 4
    0,  # Joint 5
    0,    # Joint 6
]

condition = {
    "robot":"piper_single",
    "save_path": "/home/chaihoa/Desktop/experiment/",
    "task_name": "Plug the black plug into the three-hole socket",
    "save_format": "hdf5",
    "save_freq": 10,
}


class PiperSingle(Robot):
    def __init__(
        self,
        condition=condition,
        move_check=True,
        start_episode=0,
        arm_can="can0",
        motion_speed_percent=10,
        reset_speed_percent=10,
    ):
        super().__init__(condition=condition, move_check=move_check, start_episode=start_episode)

        self.condition = condition
        self.arm_can = arm_can
        self.motion_speed_percent = int(motion_speed_percent)
        self.reset_speed_percent = int(reset_speed_percent)
        self.camera_serials = get_piper_camera_serials("single")
        self.controllers = {
            "arm":{
                "left_arm": PiperController("left_arm"),
            },
        }
        self.sensors = {
            "image":{
                "cam_head": RealsenseSensor("cam_head"),
                "cam_wrist": RealsenseSensor("cam_wrist"),
            },
        }
        if "side" in self.camera_serials:
            self.sensors["image"]["cam_side"] = RealsenseSensor("cam_side")

    # ============== init ==============
    def reset(self):
        self.controllers["arm"]["left_arm"].reset(
            np.array(START_POSITION_ANGLE_LEFT_ARM),
            speed_percent=self.reset_speed_percent,
        )

    def set_up(self):
        super().set_up()

        arm_controller = self.controllers["arm"]["left_arm"]
        arm_controller.set_up(self.arm_can)
        arm_controller.set_motion_speed_percent(self.motion_speed_percent)
        self.sensors["image"]["cam_head"].set_up(self.camera_serials["head"])
        self.sensors["image"]["cam_wrist"].set_up(self.camera_serials["wrist"])
        if "side" in self.camera_serials:
            self.sensors["image"]["cam_side"].set_up(self.camera_serials["side"])

        self.set_collect_type({"arm": ["joint","qpos","gripper"],
                               "image": ["color"]
                               })
        
        print("set up success!")
    
if __name__=="__main__":
    import time
    robot = PiperSingle()
    robot.set_up()
    # collection test
    robot.reset()
    data_list = []
    for i in range(100):
        print(i)
        data = robot.get()
        robot.collect(data)
        time.sleep(0.1)
    robot.finish()
    
    # moving test
    move_data = {
        "arm":{
            "left_arm":{
            "qpos":[0.057, 0.0, 0.216, 0.0, 0.085, 0.0],
            "gripper":0.2,
            },
        },
    }
    robot.move(move_data)
    time.sleep(1)
    move_data = {
        "arm":{
            "left_arm":{
            "joint":[0.00, 0.0, 0.0, 0.0, 0.0, 0.0],
            "gripper":0.2,
            },
        },
    }
    robot.move(move_data)
