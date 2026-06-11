"""
DAgger deployment with rollout + intervention + data collection.

Controls:
  - Space: toggle autonomous/intervention
  - Enter: start/stop episode
"""
import sys
sys.path.append("./")

import math
import threading
import time
import select
import tty
import termios
import signal
import argparse
import numpy as np

import my_robot.piper_dagger as piper_dagger_module
from my_robot.piper_dagger import PiperDAgger
from robot.policy.openpi import PI0_SINGLE
from robot.data.collect_lerobot_rl import CollectLeRobotRL

joint_limits_rad = [
    (math.radians(-150), math.radians(150)),
    (math.radians(0), math.radians(180)),
    (math.radians(-170), math.radians(0)),
    (math.radians(-100), math.radians(100)),
    (math.radians(-70), math.radians(70)),
    (math.radians(-120), math.radians(120)),
]
gripper_limit = [(0.0, 1.0)]
GRIPPER_CLOSE_THRESHOLD = 0.5
GRIPPER_CLOSE_OFFSET = 0.1
GRIPPER_INTERVENTION_DEADBAND = 0.15  # 15% - must overcome to release gripper hold on intervention entry
FIXED_MASTER_RESET_JOINT = [
    0.05902703429555556,
    -0.03265510974777778,
    0.01178097225,
    0.04764748776666667,
    -0.21891664434333333,
    0.030892327233333332,
]
FIXED_FOLLOWER_RESET_JOINT = [
    0.0,
    -0.4208,
    0.0324,
    0.0780,
    0.3558,
    0.0078,
]


class InterventionController:
    def __init__(self):
        self.is_intervention = False
        self.enter_pressed = False
        self.lock = threading.Lock()
        self.running = True
        self.paused = False
        self.old_settings = None

    def toggle_intervention(self):
        with self.lock:
            self.is_intervention = not self.is_intervention
            mode = "intervention" if self.is_intervention else "autonomous"
            print(f"\n[mode] {mode}")

    def set_intervention(self, enabled: bool):
        with self.lock:
            self.is_intervention = enabled

    def get_state(self):
        with self.lock:
            return self.is_intervention

    def is_enter_pressed(self):
        with self.lock:
            if self.enter_pressed:
                self.enter_pressed = False
                return True
            return False

    def pause_listener(self):
        with self.lock:
            self.paused = True

    def resume_listener(self):
        with self.lock:
            self.paused = False

    def _listen_keyboard(self):
        while self.running:
            with self.lock:
                paused = self.paused
            if paused:
                time.sleep(0.05)
                continue
            if select.select([sys.stdin], [], [], 0.1)[0]:
                char = sys.stdin.read(1)
                if char == " ":
                    self.toggle_intervention()
                elif char == "\r" or char == "\n":
                    with self.lock:
                        self.enter_pressed = True
                    print("\n[enter pressed]")
            time.sleep(0.01)

    def start_listener(self):
        self.running = True
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        listener_thread = threading.Thread(target=self._listen_keyboard, daemon=True)
        listener_thread.start()
        print("keyboard listener started (space: toggle, enter: start/stop)")

    def stop(self):
        self.running = False
        if self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


class MirrorController:
    def __init__(self, robot: PiperDAgger, intervention_ctrl: InterventionController, mirror_fps: int):
        self.robot = robot
        self.intervention_ctrl = intervention_ctrl
        self.mirror_fps = mirror_fps
        self.running = True
        self.last_mode = None
        self._pause_event = threading.Event()

    def start(self):
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()

    def stop(self):
        self.running = False

    def pause(self):
        self._pause_event.set()

    def resume(self):
        self._pause_event.clear()

    def _apply_mode(self, is_intervention: bool):
        if is_intervention:
            self.robot.set_policy_enabled(False)
            self.robot.hold_follower_position()
            self.robot.enable_master_drag_mode()
        else:
            self.robot.set_policy_enabled(False)
            self.robot.disable_master_drag_mode()
            self.robot.hold_follower_position()
            self.robot.set_policy_enabled(True)

    def _run(self):
        period = 1.0 / max(1, self.mirror_fps)
        while self.running:
            try:
                if self._pause_event.is_set():
                    time.sleep(period)
                    continue
                is_intervention = self.intervention_ctrl.get_state()
                if is_intervention != self.last_mode:
                    self._apply_mode(is_intervention)
                    self.last_mode = is_intervention

                # 只在自主模式下镜像从臂到主臂（观察用）
                # 干预模式下的镜像由主循环控制，避免冲突
                if not is_intervention:
                    self.robot.mirror_follower_to_master()
            except Exception as exc:
                print(f"[mirror] error: {exc}")
            time.sleep(period)


def input_transform(data):
    state_7d = np.concatenate([
        np.array(data[0]["left_arm"]["joint"]).reshape(-1),
        np.array(data[0]["left_arm"]["gripper"]).reshape(-1),
    ])

    img_head = data[1]["cam_head"]["color"]
    img_wrist = data[1]["cam_wrist"]["color"]
    img_arr = (img_head, img_wrist)

    return img_arr, state_7d


def output_transform(action):
    def clamp(value, min_val, max_val):
        return max(min_val, min(value, max_val))

    action_7d = action[:7]
    joints = [
        clamp(action_7d[i], joint_limits_rad[i][0], joint_limits_rad[i][1])
        for i in range(6)
    ]
    gripper = clamp(action_7d[6], gripper_limit[0][0], gripper_limit[0][1])
    if gripper <= GRIPPER_CLOSE_THRESHOLD:
        gripper = clamp(gripper - GRIPPER_CLOSE_OFFSET, gripper_limit[0][0], gripper_limit[0][1])

    return {
        "joint": joints,
        "gripper": gripper,
    }


def _teleop_transform(joint, gripper, teleop_kwargs):
    """应用遥操作变换（符号、偏移、缩放）"""
    joint = np.array(joint, dtype=float)
    gripper = float(gripper)

    joint_sign = teleop_kwargs.get("joint_sign", [1, 1, 1, 1, 1, 1])
    joint_offset = teleop_kwargs.get("joint_offset", [0, 0, 0, 0, 0, 0])
    joint_scale = teleop_kwargs.get("joint_scale", [1, 1, 1, 1, 1, 1])
    gripper_scale = teleop_kwargs.get("gripper_scale", 1.0)

    joint = joint * np.array(joint_sign) + np.array(joint_offset)
    joint = joint * np.array(joint_scale)
    gripper = gripper * gripper_scale

    return {"joint": joint.tolist(), "gripper": gripper}


def _read_master_action(master, teleop_kwargs):
    """
    读取主臂动作（参考 collect_lerobot_master_slave_teleop.py）
    - 优先读取控制帧（master 模式通常只发控制帧）
    - 可选回退到反馈状态
    """
    use_ctrl_frame = teleop_kwargs.get("use_ctrl_frame", True)
    fallback_to_feedback = teleop_kwargs.get("fallback_to_feedback", False)

    if use_ctrl_frame:
        try:
            ctrl = master.controller.GetArmJointCtrl()
            if getattr(ctrl, "Hz", 0) > 0:
                joint_ctrl = ctrl.joint_ctrl
                joint_cmd = np.array([
                    joint_ctrl.joint_1,
                    joint_ctrl.joint_2,
                    joint_ctrl.joint_3,
                    joint_ctrl.joint_4,
                    joint_ctrl.joint_5,
                    joint_ctrl.joint_6,
                ], dtype=float)
                # 控制帧单位 0.001 度 -> 弧度
                joint = joint_cmd / 57295.7795
                gripper_cmd = master.controller.GetArmGripperCtrl().gripper_ctrl.grippers_angle
                gripper = float(gripper_cmd) / (70 * 1000)
                return _teleop_transform(joint, gripper, teleop_kwargs)
        except Exception:
            pass

        if not fallback_to_feedback:
            return None

    state = master.get_state()
    return _teleop_transform(state["joint"], state["gripper"], teleop_kwargs)


def _filter_and_limit_action(move_data, last_cmd, last_send_time, teleop_kwargs):
    """对动作做死区与限频，避免主臂抖动导致从臂乱动"""
    if move_data is None:
        return None, last_cmd, last_send_time

    joint = np.array(move_data["joint"], dtype=float)
    gripper = float(move_data["gripper"])

    # 发送最小间隔
    min_interval = float(teleop_kwargs.get("min_send_interval", 0.02))
    now = time.time()
    if min_interval > 0 and now - last_send_time < min_interval:
        return None, last_cmd, last_send_time

    # 死区过滤
    joint_deadband = float(teleop_kwargs.get("joint_deadband", 0.005))
    gripper_deadband = float(teleop_kwargs.get("gripper_deadband", 0.01))

    if last_cmd is not None:
        last_joint = np.array(last_cmd["joint"], dtype=float)
        last_gripper = float(last_cmd["gripper"])

        joint_diff = np.abs(joint - last_joint)
        gripper_diff = abs(gripper - last_gripper)

        if np.all(joint_diff < joint_deadband) and gripper_diff < gripper_deadband:
            return None, last_cmd, last_send_time

    # 可选：低通滤波
    filter_alpha = teleop_kwargs.get("filter_alpha", None)
    if filter_alpha is not None and last_cmd is not None:
        last_joint = np.array(last_cmd["joint"], dtype=float)
        last_gripper = float(last_cmd["gripper"])
        joint = filter_alpha * joint + (1 - filter_alpha) * last_joint
        gripper = filter_alpha * gripper + (1 - filter_alpha) * last_gripper

    filtered_data = {"joint": joint.tolist(), "gripper": gripper}
    return filtered_data, filtered_data, now


if __name__ == "__main__":
    # 全局变量用于信号处理
    robot = None
    mirror_ctrl = None
    intervention_ctrl = None

    def apply_fixed_reset_poses():
        """把 DAgger 的主臂/从臂复位位姿固定为预先记录的关节角。"""
        piper_dagger_module.START_POSITION_ANGLE_MASTER_ARM = FIXED_MASTER_RESET_JOINT.copy()
        piper_dagger_module.START_POSITION_ANGLE_FOLLOWER_ARM = FIXED_FOLLOWER_RESET_JOINT.copy()
        print(
            "[reset] fixed master reset pose: "
            f"{piper_dagger_module.START_POSITION_ANGLE_MASTER_ARM}"
        )
        print(
            "[reset] fixed follower reset pose: "
            f"{piper_dagger_module.START_POSITION_ANGLE_FOLLOWER_ARM}"
        )

    def cleanup_on_exit(signum=None, frame=None):
        """清理函数：在程序退出时调用"""
        print("\n[cleanup] received exit signal, cleaning up...")
        try:
            # 先停止所有线程，避免命令冲突
            if mirror_ctrl:
                mirror_ctrl.stop()
            if intervention_ctrl:
                intervention_ctrl.stop()

            # 等待线程停止
            time.sleep(0.5)

            # 再清理机械臂状态
            if robot:
                robot.disable_master_drag_mode()
                robot.reset()

            print("[cleanup] cleanup complete")
        except Exception as e:
            print(f"[warn] cleanup error: {e}")
        sys.exit(0)

    def prepare_robot_for_reset(stop_listener: bool = False):
        """Make the episode-end reset path match the Ctrl+C cleanup path as closely as possible."""
        if mirror_ctrl:
            mirror_ctrl.pause()
        if stop_listener and intervention_ctrl:
            intervention_ctrl.stop()

        # Give the mirror thread and mode switch time to settle before resetting.
        time.sleep(0.5)

        if robot:
            intervention_ctrl.set_intervention(False)
            try:
                robot.disable_master_drag_mode()
            except Exception as exc:
                print(f"[warn] failed to disable master drag mode before reset: {exc}")

    # 注册信号处理器
    signal.signal(signal.SIGINT, cleanup_on_exit)  # Ctrl+C
    signal.signal(signal.SIGTERM, cleanup_on_exit)  # kill

    parser = argparse.ArgumentParser(
        description="PI0 DAgger 部署脚本（默认 pi05，传 adv_ind 时兼容 PiStar）"
    )






    parser.add_argument("--model-path", type=str, default="/app/checkpoint/white_426_100_r1_r3_r3new/24000", help="checkpoint 根目录")

    #plug
    parser.add_argument("--task-name", type=str, default="put the white plug into the two-hole socket", help="任务名称")
    parser.add_argument("--train-config", type=str, default="pi05_star_white_plug_infer", help="训练配置名")


    #toy
    
    # parser.add_argument("--task-name", type=str, default="Put these toys into the box", help="任务名称")
    # parser.add_argument("--train-config", type=str, default="toy_419_all_positive_infer", help="训练配置名")




    parser.add_argument("--num-episode", type=int, default=100, help="episode 数量")
    parser.add_argument("--repo-id", type=str, default="white_rollout3_new_test", help="数据集 repo_id")
    parser.add_argument("--output-dir", type=str, default="/app/dataset/rollout", help="数据输出目录")
    parser.add_argument("--fps", type=int, default=10, help="数据采集频率")
    parser.add_argument("--mirror-fps", type=int, default=200, help="镜像线程频率")
    parser.add_argument("--penalty-value", type=float, default=-1.0, help="失败惩罚值")
    parser.add_argument("--adv-ind", type=str, default=None, help="PiStar 配置使用的 adv_ind，例如 positive/negative；普通 pi05 会忽略")
    parser.add_argument("--gripper-close-threshold", type=float, default=0.5, help="模型夹爪输出小于等于该值时应用闭合增强")
    parser.add_argument("--gripper-close-offset", type=float, default=0.1, help="闭合增强时从模型夹爪输出中减去的开度偏置")
    args = parser.parse_args()

    GRIPPER_CLOSE_THRESHOLD = max(0.0, min(args.gripper_close_threshold, 1.0))
    GRIPPER_CLOSE_OFFSET = max(0.0, min(args.gripper_close_offset, 1.0))

    MODEL_PATH = args.model_path
    TASK_NAME = args.task_name
    TRAIN_CONFIG_NAME = args.train_config
    MAX_STEP = None  # 不限制最大步数，按 Enter 结束
    NUM_EPISODE = args.num_episode

    REPO_ID = args.repo_id
    OUTPUT_DIR = args.output_dir
    FPS = args.fps
    MIRROR_FPS = args.mirror_fps
    PENALTY_VALUE = args.penalty_value
    ADV_IND = args.adv_ind

    # 遥操作配置（参考 collect_lerobot_master_slave_teleop.py）
    TEACHING_FRICTION = 1  # 拖动示教摩擦力（1=最轻）
    USE_CTRL_FRAME = True  # 使用控制帧
    FALLBACK_TO_FEEDBACK = False  # 不回退到反馈状态
    JOINT_DEADBAND = 0.005  # 关节死区
    GRIPPER_DEADBAND = 0.01  # 夹爪死区
    MIN_SEND_INTERVAL = 0.01  # 最小发送间隔（100Hz）
    FILTER_ALPHA = None  # 低通滤波系数（None=不滤波）
    JOINT_SIGN = [1, 1, 1, 1, 1, 1]  # 关节符号
    JOINT_OFFSET = [0, 0, 0, 0, 0, 0]  # 关节偏移
    JOINT_SCALE = [1, 1, 1, 1, 1, 1]  # 关节缩放
    GRIPPER_SCALE = 1.0  # 夹爪缩放

    teleop_kwargs = {
        "teaching_friction": TEACHING_FRICTION,
        "use_ctrl_frame": USE_CTRL_FRAME,
        "fallback_to_feedback": FALLBACK_TO_FEEDBACK,
        "joint_deadband": JOINT_DEADBAND,
        "gripper_deadband": GRIPPER_DEADBAND,
        "min_send_interval": MIN_SEND_INTERVAL,
        "filter_alpha": FILTER_ALPHA,
        "joint_sign": JOINT_SIGN,
        "joint_offset": JOINT_OFFSET,
        "joint_scale": JOINT_SCALE,
        "gripper_scale": GRIPPER_SCALE,
    }

    print("=" * 60)
    print("PI0 DAgger deployment")
    print("=" * 60)
    print(f"model: {MODEL_PATH}")
    print(f"task: {TASK_NAME}")
    print(f"train config: {TRAIN_CONFIG_NAME}")
    print(f"dataset: {REPO_ID}")
    print(f"output: {OUTPUT_DIR}")
    print(f"adv_ind: {ADV_IND if ADV_IND is not None else 'None (pi05 default)'}")
    print(f"gripper close boost: output <= {GRIPPER_CLOSE_THRESHOLD}, offset {GRIPPER_CLOSE_OFFSET}")
    print("=" * 60)

    print("\n[1/4] init robot")
    apply_fixed_reset_poses()
    robot = PiperDAgger()
    robot.set_up()
    print("[ok] robot ready")

    print("\n[2/4] load model")
    model = PI0_SINGLE(TASK_NAME, TRAIN_CONFIG_NAME, "model", MODEL_PATH, adv_ind=ADV_IND)
    print("[ok] model loaded")

    print("\n[3/4] init collector")
    collector = CollectLeRobotRL(
        repo_id=REPO_ID,
        output_dir=OUTPUT_DIR,
        task_name=TASK_NAME,
        fps=FPS,
        robot_type="piper",
        state_dim=7,
        action_dim=7,
        image_size=(720,1280),
        camera_keys={
            "cam_head": "image",
            "cam_wrist": "wrist_image",
        },
        move_check=True,
        tolerance=0.002,  
        penalty_value=PENALTY_VALUE,
    )
    print("[ok] collector ready")

    intervention_ctrl = InterventionController()
    intervention_ctrl.start_listener()

    mirror_ctrl = MirrorController(robot, intervention_ctrl, MIRROR_FPS)
    # Keep mirroring paused until the first explicit reset completes,
    # otherwise the background thread can move the master arm before episode start.
    mirror_ctrl.pause()
    mirror_ctrl.start()

    print("\n[4/4] start episodes")
    prev_intervention = False
    intervention_gripper_hold = None

    for episode_idx in range(NUM_EPISODE):
        step = 0
        print(f"\n=== Episode {episode_idx + 1}/{NUM_EPISODE} ===")

        # 重置机器人和模型。这里和 episode 结束后的 reset 使用同一套准备流程，
        # 避免程序重新开始时主臂落到和 cleanup 不一致的位置。
        print("[reset] preparing robot for episode-start reset...")
        prepare_robot_for_reset(stop_listener=False)
        robot.reset()
        robot.set_mirror_sync_baseline()
        model.reset_obsrvationwindows()
        model.random_set_language()

        intervention_ctrl.set_intervention(False)

        print("press Enter to start")
        is_start = False
        while not is_start:
            if intervention_ctrl.is_enter_pressed():
                is_start = True
                mirror_ctrl.resume()
                print("[ok] episode started")
            else:
                time.sleep(0.1)

        last_time = time.time()
        last_cmd = None
        last_send_time = 0.0

        while is_start:  # 移除 MAX_STEP 限制，只通过 Enter 结束
            data = robot.get()
            is_intervention = intervention_ctrl.get_state()

            img_arr, state = input_transform(data)
            model.update_observation_window(img_arr, state)

            if is_intervention:
                if not prev_intervention:
                    # 刚进入干预模式：保持从臂当前位置，不要跳转到主臂位姿
                    prev_intervention = True
                    follower_state = robot.get_follower_state()
                    intervention_gripper_hold = float(follower_state["gripper"])
                    robot.hold_follower_position()
                    last_cmd = {"joint": follower_state["joint"].tolist(), "gripper": intervention_gripper_hold}
                    last_send_time = time.time()
                    time.sleep(0.01)
                    continue

                # 干预模式：使用 collect_lerobot_master_slave_teleop.py 的控制方式
                # 读取主臂动作（主臂是 right_arm）
                master_controller = robot.controllers["arm"]["right_arm"]
                move_data = _read_master_action(master_controller, teleop_kwargs)

                # 调试：检查是否读取到数据
                if move_data is None and step == 0:
                    print("[debug] move_data is None - 未读取到主臂动作")

                # 夹爪保持：防止进入干预时从臂夹爪跳变导致物体滑落
                if move_data is not None and intervention_gripper_hold is not None:
                    gripper_change = abs(move_data["gripper"] - intervention_gripper_hold)
                    if gripper_change < GRIPPER_INTERVENTION_DEADBAND:
                        move_data["gripper"] = intervention_gripper_hold
                    else:
                        intervention_gripper_hold = None

                # 过滤和限频
                move_data, last_cmd, last_send_time = _filter_and_limit_action(
                    move_data, last_cmd, last_send_time, teleop_kwargs
                )

                # 发送到从臂
                if move_data is not None:
                    robot.move_follower(move_data, bypass_policy=True)
                    if step == 0:
                        print(f"[debug] 首次发送动作到从臂: joint={move_data['joint'][:2]}, gripper={move_data['gripper']}")

                # 收集数据（仍然按 10Hz 收集）
                current_time = time.time()
                if current_time - last_time >= 1 / FPS:
                    follower_controller = {"left_arm": data[0]["left_arm"]}
                    collector.collect(follower_controller, data[1], is_intervention=True)
                    step += 1
                    last_time = current_time

                if intervention_ctrl.is_enter_pressed():
                    print("[warn] episode stop requested")
                    is_start = False

                time.sleep(0.01)  # 100Hz 镜像频率
                continue

            # 退出干预模式：重置追踪状态
            if prev_intervention:
                prev_intervention = False
                intervention_gripper_hold = None
                last_cmd = None
                last_send_time = 0.0

            action_chunk = model.get_action()
            action_chunk = action_chunk[:10]

            for action in action_chunk:
                if intervention_ctrl.get_state():
                    print("[warn] switched to intervention, stop policy actions")
                    break
                move_data = output_transform(action)
                robot.move_follower(move_data)
                data = robot.get()
                follower_controller = {"left_arm": data[0]["left_arm"]}
                collector.collect(follower_controller, data[1], is_intervention=False)
                step += 1

                current_time = time.time()
                sleep_time = max(0, 1 / FPS - (current_time - last_time))
                time.sleep(sleep_time)
                last_time = time.time()

                if intervention_ctrl.is_enter_pressed():
                    print("[warn] episode stop requested")
                    is_start = False
                    break

        print("=" * 60)
        print(f"episode done (steps: {step}, frames: {len(collector.episode_buffer)})")
        print("=" * 60)

        print("[reset] preparing robot for episode-end reset...")
        try:
            robot.hold_follower_position()
        except Exception as exc:
            print(f"[warn] failed to hold follower before reset: {exc}")
        prepare_robot_for_reset(stop_listener=False)

        if len(collector.episode_buffer) > 0:
            print("label episode: 1=success, 0=failure")
            intervention_ctrl.pause_listener()
            success_input = input("enter 1/0: ").strip()
            intervention_ctrl.resume_listener()
            success = (success_input == "1")
            collector.save_episode(success=success, adv_ind_value="none")
            print(f"[ok] episode saved (success: {success})")
        else:
            print("[warn] empty episode, nothing saved")

        # Episode 结束后复位到起始位置
        print("[reset] resetting robot to start position...")
        robot.reset()
        robot.set_mirror_sync_baseline()
        print("[ok] reset complete")

    # 程序结束时清理机械臂状态
    print("\n[cleanup] cleaning up robot state...")
    try:
        # 确保退出干预模式
        intervention_ctrl.set_intervention(False)
        robot.disable_master_drag_mode()
        robot.reset()
        print("[cleanup] robot state cleaned up successfully")
    except Exception as e:
        print(f"[warn] cleanup failed: {e}")

    mirror_ctrl.stop()
    intervention_ctrl.stop()

    print("\n" + "=" * 60)
    print("all episodes finished")
    print(f"dataset path: {collector.get_dataset_path()}")
    print("=" * 60)
