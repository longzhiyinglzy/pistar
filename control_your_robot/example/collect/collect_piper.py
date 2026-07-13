#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SpaceMouse -> Piper single arm teleop
方案一：控制高频(200Hz)，采样低频(30Hz)，最终按30Hz保存训练数据

设计目标：
1. 控制线程高频运行，保证遥操作跟手性
2. 采集线程低频运行，保证图像/状态/动作按30Hz对齐保存
3. 每条样本以“采集时刻 t_sample”为主时钟：
   - 图像/传感器：来自 robot.get()
   - state：从高频 state buffer 中取离 t_sample 最近的一条
   - action：从高频 action buffer 中取 t <= t_sample 的最近一条
4. 夹爪保持 0/1 控制
"""

import os
import time
import copy
import sys
import tty
import termios
import select
import argparse
import numpy as np
from collections import deque
from threading import Lock

# Allow running this script from any cwd (e.g. example/collect).
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from robot.utils.base.data_handler import debug_print
from robot.data.collect_any import CollectAny
from robot.controller.SpaceMouse_controller import SpaceMouseController
from my_robot.agilex_piper_single_base import PiperSingle
from robot.utils.node.node import TaskNode
from robot.utils.node.scheduler import Scheduler


# ============================================================
# 参数区
# ============================================================
CONTROL_HZ = 200
COLLECT_HZ = 30
ENABLE_JPEG_SENSOR = True

condition = {
    "save_path": "./piper_datasets",
    "task_name": "piper_spacemouse_demo",
    "save_format": "hdf5",
    "save_freq": COLLECT_HZ,   # 最终数据集频率
    "collect_type": "teleop",
}

USE_WORKSPACE_LIMIT = False

XYZ_MIN = np.array([0.10, -0.40, 0.0], dtype=np.float64)
XYZ_MAX = np.array([0.60,  0.40, 0.80], dtype=np.float64)

RPY_MIN = np.array([-3.14, -1.57, -3.14], dtype=np.float64)
RPY_MAX = np.array([ 3.14,  1.57,  3.14], dtype=np.float64)

DEADZONE = 30
MAX_VALUE = 350.0
TRANS_MOTION = 0.002
ROT_SCALE = 0.2

RESET_GRIPPER = 0.0
OPEN_GRIPPER = 0.6
CLOSE_GRIPPER = 0.0

STATE_BUFFER_SIZE = 4000    # 4000 / 200Hz = 20s
ACTION_BUFFER_SIZE = 4000   # 4000 / 200Hz = 20s


# ============================================================
# 工具函数
# ============================================================
def clip_qpos(qpos: np.ndarray) -> np.ndarray:
    qpos = np.array(qpos, dtype=np.float64).copy()

    if USE_WORKSPACE_LIMIT:
        qpos[:3] = np.clip(qpos[:3], XYZ_MIN, XYZ_MAX)
        qpos[3:] = np.clip(qpos[3:], RPY_MIN, RPY_MAX)

    return qpos


def map_spacemouse_to_distance(
    percentage: float,
    deadzone: float = DEADZONE,
    motion: float = TRANS_MOTION
) -> float:
    if abs(percentage) <= deadzone:
        return 0.0
    elif percentage > 0:
        normalized = (percentage - deadzone) / (100.0 - deadzone)
        return -normalized * motion
    else:
        normalized = (abs(percentage) - deadzone) / (100.0 - deadzone)
        return normalized * motion


def raw_to_delta_qpos(raw: np.ndarray) -> np.ndarray:
    """
    raw = [x, y, z, rx, ry, rz]

    映射关系：
    distance_x  <- raw_y
    distance_y  <- raw_x
    distance_z  <- raw_z

    distance_ry <- raw_rx
    distance_rx <- raw_ry
    distance_rz <- raw_rz
    """
    percentages = (raw / MAX_VALUE) * 100.0

    distance_x = map_spacemouse_to_distance(percentages[1])   # raw_y -> qpos_x
    distance_y = map_spacemouse_to_distance(percentages[0])   # raw_x -> qpos_y
    distance_z = map_spacemouse_to_distance(percentages[2])   # raw_z -> qpos_z

    distance_ry = map_spacemouse_to_distance(percentages[3]) * ROT_SCALE
    distance_rx = map_spacemouse_to_distance(percentages[4]) * ROT_SCALE
    distance_rz = map_spacemouse_to_distance(percentages[5]) * ROT_SCALE

    delta_qpos = np.array(
        [distance_x, distance_y, distance_z, distance_rx, distance_ry, distance_rz],
        dtype=np.float64
    )
    return delta_qpos


def setup_keyboard_cbreak_mode():
    fd = None
    owns_fd = False

    # 优先直接读控制终端，避免 IDE/重定向导致 stdin 行缓冲
    try:
        fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
        owns_fd = True
    except Exception:
        fd = None

    if fd is None and sys.stdin.isatty():
        try:
            fd = sys.stdin.fileno()
        except Exception:
            fd = None

    if fd is None:
        print("[keyboard] no tty input backend available", flush=True)
        return None, None, False

    try:
        old_attrs = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        return fd, old_attrs, owns_fd
    except Exception as e:
        print(f"[keyboard] set cbreak mode failed: {e}", flush=True)
        if owns_fd:
            try:
                os.close(fd)
            except Exception:
                pass
        return None, None, False


def restore_keyboard_mode(fd, old_attrs, owns_fd):
    if fd is None or old_attrs is None:
        return
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
    except Exception as e:
        print(f"[keyboard] restore mode failed: {e}", flush=True)
    if owns_fd:
        try:
            os.close(fd)
        except Exception:
            pass


def drain_keyboard_buffer(kb_fd):
    if kb_fd is None:
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if not ready:
                return
            ch = sys.stdin.read(1)
            if ch in ("\n", "\r", ""):
                return
        return
    while True:
        ready, _, _ = select.select([kb_fd], [], [], 0)
        if not ready:
            return
        try:
            data = os.read(kb_fd, 64)
        except BlockingIOError:
            return
        except Exception:
            return
        if not data:
            return


def poll_key(kb_fd):
    if kb_fd is None:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
        if not ready:
            return None
        ch = sys.stdin.read(1)
        return ch if ch else None

    ready, _, _ = select.select([kb_fd], [], [], 0)
    if not ready:
        return None

    try:
        data = os.read(kb_fd, 1)
    except BlockingIOError:
        return None
    except Exception:
        return None

    if not data:
        return None

    try:
        ch = data.decode("utf-8", errors="ignore")
    except Exception:
        return None

    return ch if ch else None


def wait_enter_press(kb_fd):
    while True:
        ch = poll_key(kb_fd)
        if ch is not None:
            if ch in ("\n", "\r"):
                time.sleep(0.1)
                return
            # 丢掉非 Enter 的剩余输入，避免脏按键穿透到后续逻辑
            drain_keyboard_buffer(kb_fd)
        time.sleep(0.01)


def poll_episode_command(kb_fd):
    """
    非阻塞按键命令：
      Enter -> 保存并结束本轮
      r     -> 丢弃本轮并重录同一 episode
      h     -> 机械臂复位（继续本轮）
    """
    ch = poll_key(kb_fd)
    if not ch:
        return None

    if ch in ("\n", "\r"):
        return "save"

    key = ch.lower()
    if key == "r":
        drain_keyboard_buffer(kb_fd)
        return "redo"
    if key == "h":
        drain_keyboard_buffer(kb_fd)
        return "home"

    drain_keyboard_buffer(kb_fd)
    return None


def get_arm_init_state(robot):
    while True:
        try:
            arm_state = robot.controllers["arm"]["left_arm"].get_state()
            qpos = arm_state.get("qpos")
            gripper = arm_state.get("gripper")
            if qpos is not None:
                init_qpos = np.array(qpos, dtype=np.float64)
                init_gripper = float(gripper) if gripper is not None else 0.0
                return init_qpos, init_gripper
        except Exception as e:
            print("[init] waiting arm state:", e, flush=True)
        time.sleep(0.1)


def enable_sensor_jpeg(robot):
    """
    Enable JPEG encoding inside VisionSensor.get_information().
    This significantly reduces per-frame memory footprint and prevents OOM
    when collecting multi-camera trajectories for long episodes.
    """
    if not ENABLE_JPEG_SENSOR:
        return

    image_sensors = robot.sensors.get("image", {})
    for sensor_name, sensor in image_sensors.items():
        if hasattr(sensor, "is_jpeg"):
            sensor.is_jpeg = True
            debug_print("main", f"sensor {sensor_name}: enable jpeg encoding", "INFO")


def build_controller_sensor_pack(obs, master_data, cmd_qpos, cmd_gripper, delta_qpos, sync_info):
    controller_data = {}
    sensor_data = {}

    # slave 实际状态
    for key, value in obs[0].items():
        controller_data["slave_" + key] = value

    # slave 相机 / 传感器
    for key, value in obs[1].items():
        sensor_data["slave_" + key] = value

    # master 输入
    controller_data["master_spacemouse"] = {
        "raw": np.array(master_data.get("raw", np.zeros(6)), dtype=np.float64),
        "gripper": float(master_data.get("gripper", 0.0)),
        "left_button": int(master_data.get("left_button", 0)),
        "right_button": int(master_data.get("right_button", 0)),
        "delta_qpos": np.array(delta_qpos, dtype=np.float64),
    }

    # action 标签
    controller_data["action_left_arm"] = {
        "qpos": np.array(cmd_qpos, dtype=np.float64),
        "gripper": float(cmd_gripper),
        "delta_qpos": np.array(delta_qpos, dtype=np.float64),
    }

    # 同步信息
    controller_data["sync_info"] = sync_info

    return controller_data, sensor_data


# ============================================================
# 高频缓存
# ============================================================
class TeleopBuffers:
    def __init__(self, state_maxlen=STATE_BUFFER_SIZE, action_maxlen=ACTION_BUFFER_SIZE):
        self.lock = Lock()
        self.state_buffer = deque(maxlen=state_maxlen)
        self.action_buffer = deque(maxlen=action_maxlen)

        self.prev_left_button = 0
        self.prev_right_button = 0
        self.cmd_gripper = RESET_GRIPPER
        self.last_sent_gripper = RESET_GRIPPER

    def reset(self, left_button_init=0, right_button_init=0):
        with self.lock:
            self.state_buffer.clear()
            self.action_buffer.clear()
            self.prev_left_button = int(left_button_init)
            self.prev_right_button = int(right_button_init)
            self.cmd_gripper = RESET_GRIPPER
            self.last_sent_gripper = RESET_GRIPPER

    def append_state(self, sample: dict):
        with self.lock:
            self.state_buffer.append(sample)

    def append_action(self, sample: dict):
        with self.lock:
            self.action_buffer.append(sample)

    def get_button_and_gripper_state(self):
        with self.lock:
            return (
                self.prev_left_button,
                self.prev_right_button,
                self.cmd_gripper,
                self.last_sent_gripper,
            )

    def update_button_and_gripper_state(self, prev_left_button, prev_right_button, cmd_gripper, last_sent_gripper=None):
        with self.lock:
            self.prev_left_button = int(prev_left_button)
            self.prev_right_button = int(prev_right_button)
            self.cmd_gripper = float(cmd_gripper)
            if last_sent_gripper is not None:
                self.last_sent_gripper = float(last_sent_gripper)

    def get_aligned_data(self, t_sample: float):
        with self.lock:
            if len(self.state_buffer) == 0 or len(self.action_buffer) == 0:
                return None

            # 1) 找离 t_sample 最近的 state
            nearest_state = min(
                self.state_buffer,
                key=lambda x: abs(x["t"] - t_sample)
            )

            # 2) 找 t <= t_sample 的最近 action；如果没有，就退化为最近 action
            past_actions = [a for a in self.action_buffer if a["t"] <= t_sample]
            if len(past_actions) > 0:
                aligned_action = past_actions[-1]
            else:
                aligned_action = min(
                    self.action_buffer,
                    key=lambda x: abs(x["t"] - t_sample)
                )

            return {
                "state": copy.deepcopy(nearest_state),
                "action": copy.deepcopy(aligned_action),
            }


# ============================================================
# 控制节点：200 Hz
# ============================================================
class SpaceMouseControlNode(TaskNode):
    def task_init(self, master, robot, buffers: TeleopBuffers):
        self.master = master
        self.robot = robot
        self.buffers = buffers

    def task_step(self):
        t_loop = time.monotonic()

        # 1) 读 SpaceMouse
        master_data = self.master.get()
        raw = np.array(master_data.get("raw", np.zeros(6)), dtype=np.float64)
        left_button = int(master_data.get("left_button", 0))
        right_button = int(master_data.get("right_button", 0))

        # 2) 取共享夹爪/按钮状态
        prev_left_button, prev_right_button, cmd_gripper, last_sent_gripper = \
            self.buffers.get_button_and_gripper_state()

        # 3) 夹爪 0/1 边沿控制
        if left_button == 1 and prev_left_button == 0:
            cmd_gripper = OPEN_GRIPPER
        if right_button == 1 and prev_right_button == 0:
            cmd_gripper = CLOSE_GRIPPER

        self.buffers.update_button_and_gripper_state(
            prev_left_button=left_button,
            prev_right_button=right_button,
            cmd_gripper=cmd_gripper,
            last_sent_gripper=None,
        )

        # 4) 读取机器人当前真实状态
        arm_state = self.robot.controllers["arm"]["left_arm"].get_state()

        actual_qpos = np.array(arm_state["qpos"], dtype=np.float64)
        actual_joint = np.array(arm_state["joint"], dtype=np.float64)
        actual_gripper = float(arm_state.get("gripper", 0.0))

        # 5) 写入高频 state buffer
        state_sample = {
            "t": t_loop,
            "qpos": actual_qpos.copy(),
            "joint": actual_joint.copy(),
            "gripper": actual_gripper,
        }
        self.buffers.append_state(state_sample)

        # 6) 计算控制命令
        delta_qpos = raw_to_delta_qpos(raw)
        cmd_qpos = clip_qpos(actual_qpos + delta_qpos)

        need_move = (
            np.any(np.abs(delta_qpos) > 0.0) or
            (abs(cmd_gripper - last_sent_gripper) > 1e-6)
        )

        sent = False
        if need_move:
            cmd = {
                "arm": {
                    "left_arm": {
                        "qpos": cmd_qpos.copy(),
                        "gripper": cmd_gripper,
                    }
                }
            }
            self.robot.move(cmd)
            sent = True
            self.buffers.update_button_and_gripper_state(
                prev_left_button=left_button,
                prev_right_button=right_button,
                cmd_gripper=cmd_gripper,
                last_sent_gripper=cmd_gripper,
            )

        # 7) 无论发没发，都写入高频 action buffer
        master_data_for_save = {
            "raw": raw.copy(),
            "left_button": left_button,
            "right_button": right_button,
            "gripper": cmd_gripper,
        }

        action_sample = {
            "t": t_loop,
            "cmd_qpos": cmd_qpos.copy(),
            "cmd_gripper": float(cmd_gripper),
            "delta_qpos": delta_qpos.copy(),
            "master_data": master_data_for_save,
            "sent": sent,
        }
        self.buffers.append_action(action_sample)


# ============================================================
# 采集节点：30 Hz
# ============================================================
class CameraAlignedCollectNode(TaskNode):
    def task_init(self, robot, collector: CollectAny, buffers: TeleopBuffers):
        self.robot = robot
        self.collector = collector
        self.buffers = buffers

    def task_step(self):
        # 采集时刻，作为当前样本的主时钟
        t_before_get = time.monotonic()

        try:
            obs = self.robot.get()
        except Exception as e:
            print("[collect] robot.get failed:", e, flush=True)
            obs = [
                {"left_arm": self.robot.controllers["arm"]["left_arm"].get()},
                {}
            ]

        t_after_get = time.monotonic()

        # 这里用 get() 返回后的时间作为采样时刻近似
        # 若以后相机驱动能返回硬件时间戳，可以替换成真实 t_cam
        t_sample = t_after_get

        aligned = self.buffers.get_aligned_data(t_sample)
        if aligned is None:
            return

        state_aligned = aligned["state"]
        action_aligned = aligned["action"]

        sync_info = {
            "t_sample": float(t_sample),
            "t_before_get": float(t_before_get),
            "t_after_get": float(t_after_get),
            "t_state": float(state_aligned["t"]),
            "t_action": float(action_aligned["t"]),
            "dt_state_to_sample": float(t_sample - state_aligned["t"]),
            "dt_action_to_sample": float(t_sample - action_aligned["t"]),
            "action_sent": bool(action_aligned.get("sent", False)),
        }

        controller_data, sensor_data = build_controller_sensor_pack(
            obs=obs,
            master_data=action_aligned["master_data"],
            cmd_qpos=action_aligned["cmd_qpos"],
            cmd_gripper=action_aligned["cmd_gripper"],
            delta_qpos=action_aligned["delta_qpos"],
            sync_info=sync_info,
        )

        # 额外保存对齐后的高频 state
        controller_data["aligned_state"] = {
            "qpos": state_aligned["qpos"],
            "joint": state_aligned["joint"],
            "gripper": state_aligned["gripper"],
        }

        self.collector.collect(controller_data, sensor_data)


# ============================================================
# 初始化节点与调度器
# ============================================================
def init_nodes(master, robot, collector, buffers):
    control_node = SpaceMouseControlNode(
        "SPACEMOUSE_CONTROL",
        master=master,
        robot=robot,
        buffers=buffers,
    )
    control_node.start()

    collect_node = CameraAlignedCollectNode(
        "CAMERA_ALIGNED_COLLECT",
        robot=robot,
        collector=collector,
        buffers=buffers,
    )
    collect_node.start()

    return control_node, collect_node


def build_schedulers(control_node, collect_node):
    control_scheduler = Scheduler(
        entry_nodes=[control_node],
        all_nodes=[control_node],
        final_nodes=[control_node],
        hz=CONTROL_HZ,
    )

    collect_scheduler = Scheduler(
        entry_nodes=[collect_node],
        all_nodes=[collect_node],
        final_nodes=[collect_node],
        hz=COLLECT_HZ,
    )

    return control_scheduler, collect_scheduler


# ============================================================
# 主函数
# ============================================================
def build_arg_parser():
    parser = argparse.ArgumentParser(description="Collect Piper HDF5 demos with SpaceMouse teleoperation.")
    parser.add_argument("--save-path", default=condition["save_path"])
    parser.add_argument("--task-name", default=condition["task_name"])
    parser.add_argument("--num-episode", type=int, default=100)
    parser.add_argument("--spacemouse-device-path", default=None)
    return parser


def main():
    global condition
    args = build_arg_parser().parse_args()
    condition = dict(condition)
    condition["save_path"] = args.save_path
    condition["task_name"] = args.task_name
    condition["save_freq"] = COLLECT_HZ

    os.environ["INFO_LEVEL"] = "INFO"

    num_episode = int(args.num_episode)
    device_path = args.spacemouse_device_path

    kb_fd, kb_old_attrs, kb_owns_fd = setup_keyboard_cbreak_mode()
    debug_print("main", f"running script: {os.path.abspath(__file__)}", "INFO")
    if kb_fd is None:
        raise RuntimeError(
            "Realtime keyboard backend unavailable. "
            "Please run this script in a real terminal, or in IDE Run Configuration enable "
            "'Emulate terminal in output console'."
        )

    debug_print("main", "keyboard backend: /dev/tty (realtime single-key)", "INFO")
    master = None
    try:
        # 1) 初始化 SpaceMouse
        master = SpaceMouseController(
            name="spacemouse",
            initial_gripper=0.0,
        )
        master.set_collect_info(["raw", "left_button", "right_button"])
        master.set_up(device_path=device_path, grab_device=True)

        # 2) 初始化 Piper
        robot = PiperSingle()
        robot.set_up()
        enable_sensor_jpeg(robot)

        for episode_id in range(num_episode):
            retry_same_episode = True
            skip_reset_before_retry = False
            while retry_same_episode:
                retry_same_episode = False

                if not skip_reset_before_retry:
                    debug_print("main", f"episode {episode_id}: reset robot...", "INFO")
                    robot.reset()
                    time.sleep(2.0)

                    # 强制硬件夹爪归零
                    try:
                        robot.controllers["arm"]["left_arm"].set_gripper(RESET_GRIPPER)
                        time.sleep(0.3)
                    except Exception as e:
                        print(f"[episode {episode_id}] set_gripper reset failed: {e}", flush=True)
                else:
                    skip_reset_before_retry = False
                    debug_print(
                        "main",
                        f"episode {episode_id}: restart same episode after immediate reset",
                        "INFO"
                    )

                init_qpos, init_gripper = get_arm_init_state(robot)
                debug_print(
                    "main",
                    f"episode {episode_id}: initial qpos = {init_qpos.tolist()}, init_gripper = {init_gripper}",
                    "INFO"
                )

                # 每轮新建 collector
                collector = CollectAny(
                    condition=condition,
                    start_episode=episode_id,
                    move_check=True,
                    resume=True,
                )

                # 每轮新建 buffer
                buffers = TeleopBuffers()

                # 清一次按钮初值，避免刚开始误触发
                init_master_data = master.get()
                buffers.reset(
                    left_button_init=int(init_master_data.get("left_button", 0)),
                    right_button_init=int(init_master_data.get("right_button", 0)),
                )

                debug_print("main", f"episode {episode_id}: press Enter to start", "INFO")
                wait_enter_press(kb_fd)

                control_node, collect_node = init_nodes(
                    master=master,
                    robot=robot,
                    collector=collector,
                    buffers=buffers,
                )
                control_scheduler, collect_scheduler = build_schedulers(control_node, collect_node)

                debug_print(
                    "main",
                    (
                        f"episode {episode_id}: start teleop | control={CONTROL_HZ}Hz | collect={COLLECT_HZ}Hz | "
                        "Enter=save&stop, r=redo episode, h=home arm"
                    ),
                    "INFO"
                )

                control_scheduler.start()
                collect_scheduler.start()

                stop_action = "save"
                while True:
                    cmd = poll_episode_command(kb_fd)
                    if cmd is None:
                        time.sleep(0.01)
                        continue

                    if cmd == "save":
                        stop_action = "save"
                        break

                    if cmd == "redo":
                        debug_print(
                            "main",
                            f"episode {episode_id}: keyboard 'r' -> discard this round and retry same episode",
                            "INFO"
                        )
                        stop_action = "redo"
                        break

                    if cmd == "home":
                        debug_print(
                            "main",
                            f"episode {episode_id}: keyboard 'h' -> arm reset to home and continue recording current trajectory",
                            "INFO"
                        )
                        robot.reset()
                        time.sleep(1.0)
                        try:
                            robot.controllers["arm"]["left_arm"].set_gripper(RESET_GRIPPER)
                            time.sleep(0.3)
                        except Exception as e:
                            print(f"[episode {episode_id}] keyboard home reset gripper failed: {e}", flush=True)

                        # 仅重置按钮边沿/夹爪状态，保留本轮轨迹数据连续录制
                        home_master_data = master.get()
                        buffers.update_button_and_gripper_state(
                            prev_left_button=int(home_master_data.get("left_button", 0)),
                            prev_right_button=int(home_master_data.get("right_button", 0)),
                            cmd_gripper=RESET_GRIPPER,
                            last_sent_gripper=RESET_GRIPPER,
                        )

                if stop_action == "save":
                    time.sleep(0.2)
                control_scheduler.stop()
                collect_scheduler.stop()

                if stop_action == "redo":
                    debug_print("main", f"episode {episode_id}: keyboard 'r' -> reset now and restart this episode", "INFO")
                    robot.reset()
                    time.sleep(1.0)
                    try:
                        robot.controllers["arm"]["left_arm"].set_gripper(RESET_GRIPPER)
                        time.sleep(0.3)
                    except Exception as e:
                        print(f"[episode {episode_id}] retry reset gripper failed: {e}", flush=True)
                    retry_same_episode = True
                    skip_reset_before_retry = True
                    continue

                collector.write()
                debug_print("main", f"episode {episode_id} finished and saved", "INFO")

                debug_print("main", f"episode {episode_id}: reset robot after save...", "INFO")
                robot.reset()
                time.sleep(1.0)

                try:
                    robot.controllers["arm"]["left_arm"].set_gripper(RESET_GRIPPER)
                    time.sleep(0.3)
                except Exception as e:
                    print(f"[episode {episode_id}] end reset gripper failed: {e}", flush=True)
    finally:
        restore_keyboard_mode(kb_fd, kb_old_attrs, kb_owns_fd)
        if master is not None:
            master.close()


if __name__ == "__main__":
    main()
