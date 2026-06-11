import sys
sys.path.append("./")

from my_robot.agilex_piper_single_base import PiperSingle

import time
import argparse
import numpy as np
import math
from robot.policy.openpi import PI0_SINGLE

from robot.utils.base.data_handler import is_enter_pressed

joint_limits_rad = [
    (math.radians(-150), math.radians(150)),   # joint1
    (math.radians(0), math.radians(180)),      # joint2
    (math.radians(-170), math.radians(0)),     # joint3
    (math.radians(-100), math.radians(100)),   # joint4
    (math.radians(-70), math.radians(70)),     # joint5
    (math.radians(-120), math.radians(120))    # joint6
]
gripper_limit = [(0.0, 1.0)]
GRIPPER_CLOSE_THRESHOLD = 0.5
GRIPPER_CLOSE_OFFSET = 0.1

def input_transform(data):
    """将机器人数据转换为模型输入格式"""
    # State: [6个关节 + 1个夹爪] = 7维
    state_7d = np.concatenate([
        np.array(data[0]["left_arm"]["joint"]).reshape(-1),
        np.array(data[0]["left_arm"]["gripper"]).reshape(-1)
    ])

    # 填充为14维（因为训练时用的是14维双臂格式）
    state_14d = np.pad(state_7d, (0, 7), mode='constant', constant_values=0)

    sensors = data[1]

    def get_color_image(cam_keys):
        for cam_key in cam_keys:
            cam_data = sensors.get(cam_key)
            if cam_data is not None and "color" in cam_data:
                return cam_data["color"]
        return None

    # 兼容不同数据键名；单相机部署时，head 缺失则补零图像。
    img_wrist = get_color_image(["cam_wrist", "wrist_image"])
    if img_wrist is None:
        raise KeyError(
            f"未找到 wrist 相机图像，当前可用键: {list(sensors.keys())}。"
            "期望其中之一: ['cam_wrist', 'wrist_image']"
        )

    img_head = get_color_image(["cam_head", "image"])
    if img_head is None:
        img_head = np.zeros_like(img_wrist)

    img_arr = (img_head, img_wrist)  # PI0_SINGLE 仍然按双图像接口输入

    return img_arr, state_7d  # 返回14维状态

def output_transform(action):
    """将模型输出转换为机器人控制指令"""
    def clamp(value, min_val, max_val):
        """将值限制在[min_val, max_val]范围内"""
        return max(min_val, min(value, max_val))

    # ⚠️ 关键：模型输出14维，只取前7维
    action_7d = action[:7]
  

    # 限制关节角度在安全范围内
    joints = [
        clamp(action_7d[i], joint_limits_rad[i][0], joint_limits_rad[i][1])
        for i in range(6)
    ]

    # 限制夹爪在安全范围内。Piper 中 0 表示闭合，1 表示张开。
    gripper = clamp(action_7d[6], gripper_limit[0][0], gripper_limit[0][1])
    if gripper <= GRIPPER_CLOSE_THRESHOLD:
        gripper = clamp(gripper - GRIPPER_CLOSE_OFFSET, gripper_limit[0][0], gripper_limit[0][1])

    # 构建控制指令
    move_data = {
        "arm": {
            "left_arm": {
                "joint": joints,
                "gripper": gripper
            }
        }
    }
    return move_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PI0 单臂部署脚本")

    #toy
    # parser.add_argument("--model-path", type=str, default="/app/checkpoint/toy_merge3/21000", help="checkpoint 根目录")
    # parser.add_argument("--task-name", type=str, default="Put these toys into the box", help="任务名称")
    # parser.add_argument("--train-config", type=str, default="toy_419_all_positive_infer", help="训练配置名")


    #plug
    parser.add_argument("--model-path", type=str, default="/app/checkpoint/white_426_100_r1_r3_r3new/24000", help="checkpoint 根目录")
    parser.add_argument("--task-name", type=str, default="put the white plug into the two-hole socket", help="任务名称")
    parser.add_argument("--train-config", type=str, default="pi05_star_white_plug_infer", help="训练配置名")
    #parser.add_argument("--train-config", type=str, default="piper_plug_task_teleop", help="训练配置名")
    parser.add_argument("--max-step", type=int, default=300, help="单个 episode 最大步数")
    parser.add_argument("--num-episode", type=int, default=50, help="episode 数量")
    parser.add_argument("--adv-ind", type=str, default=None, help="PiStar 配置使用的 adv_ind，例如 positive/negative；普通 pi05 会忽略")
    parser.add_argument("--gripper-effort", type=int, default=500, help="夹爪力矩，范围 0-5000")
    parser.add_argument("--gripper-close-threshold", type=float, default=0.5, help="模型夹爪输出小于等于该值时视为闭合")
    parser.add_argument("--gripper-close-offset", type=float, default=0.1, help="闭合增强时从模型夹爪输出中减去的开度偏置")
    args = parser.parse_args()

    GRIPPER_CLOSE_THRESHOLD = max(0.0, min(args.gripper_close_threshold, 1.0))
    GRIPPER_CLOSE_OFFSET = max(0.0, min(args.gripper_close_offset, 1.0))

    # ========== 配置参数 ==========
    MODEL_PATH = args.model_path  # checkpoint 根目录（应包含 params/ 与 assets/）
    TASK_NAME = args.task_name
    TRAIN_CONFIG_NAME = args.train_config  # 你训练时使用的config
    MAX_STEP = args.max_step
    NUM_EPISODE = args.num_episode

    print("=" * 50)
    print("PI0 单臂部署脚本")
    print("=" * 50)
    print(f"模型路径: {MODEL_PATH}")
    print(f"任务名称: {TASK_NAME}")
    print(f"配置名称: {TRAIN_CONFIG_NAME}")
    print("=" * 50)

    # 初始化机器人
    print("\n[1/3] 初始化机器人...")
    robot = PiperSingle()
    robot.set_up()
    robot.controllers["arm"]["left_arm"].set_gripper_effort(args.gripper_effort)
    print(f"夹爪力矩设置为: {robot.controllers['arm']['left_arm'].gripper_effort}")
    print(f"夹爪闭合增强: output <= {GRIPPER_CLOSE_THRESHOLD} 时减小 {GRIPPER_CLOSE_OFFSET}")
    print("✓ 机器人初始化完成")

    # 加载模型
    print(f"\n[2/3] 加载 PI0 模型...")
    model = PI0_SINGLE(TASK_NAME, TRAIN_CONFIG_NAME, "model", MODEL_PATH, adv_ind=args.adv_ind)
    print("✓ 模型加载完成")

    # 开始推理
    print(f"\n[3/3] 准备执行 {NUM_EPISODE} 个 episode")
    print("-" * 50)

    for episode_idx in range(NUM_EPISODE):
        step = 0
        print(f"\n{'='*20} Episode {episode_idx + 1}/{NUM_EPISODE} {'='*20}")

        # 重置机器人和模型
        robot.reset()  
        model.reset_obsrvationwindows()
        model.random_set_language()

        # 等待开始信号
        print("\n按 Enter 键开始推理...")
        is_start = False
        while not is_start:
            if is_enter_pressed():
                is_start = True
                print("✓ 开始执行任务...")
            else:
                time.sleep(0.1)

        # 执行推理循环
        while step < MAX_STEP:
            # 获取当前状态
            data = robot.get()
            img_arr, state = input_transform(data)

            # 更新观察窗口并获取动作
            model.update_observation_window(img_arr, state)
            action_chunk = model.get_action()

            # 执行动作序列（取前10步）
            action_chunk = action_chunk[:10]
            for action in action_chunk:
                move_data = output_transform(action)
                robot.move(move_data)
                step += 1
                time.sleep(1/10)  # 10Hz控制频率

                # 如果按Enter可以中断
                if is_enter_pressed():
                    print("\n⚠ 用户中断执行")
                    is_start = False
                    break

            if not is_start:
                break

        print(f"✓ Episode {episode_idx + 1} 完成 (总步数: {step})")

    print("\n" + "=" * 50)
    print("全部 episode 执行完成！")
    print("=" * 50)
