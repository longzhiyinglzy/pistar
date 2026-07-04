#!/usr/bin/env python3
"""Collect pi0.5/PiStar RTC rollouts as LeRobot RL episodes.

Typical use:
  1. Start an OpenPI websocket policy server in another terminal.
  2. Run this script.
  3. Press Enter to start each episode.
  4. During rollout press:
       s / Right arrow: save success
       f: save failure
       r / Left arrow: discard
       Enter: discard by default, configurable with --enter-label
       q / Esc: quit

By default the saved frames are autonomous policy frames. If SpaceMouse
intervention is enabled, the first non-deadzone SpaceMouse command or gripper
button switches the rest of the episode into human intervention mode and saves
``intervention=1`` for those subsequent frames.
"""

from __future__ import annotations

import argparse
import enum
import importlib.util
import logging
import math
import os
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from pathlib import Path

import numpy as np


LOCAL_CONTROL_YOUR_ROBOT_ROOT = Path(__file__).resolve().parents[2]
OPENPI_ROOT = LOCAL_CONTROL_YOUR_ROBOT_ROOT / "src" / "robot" / "policy" / "openpi"
for path in (
    OPENPI_ROOT,
    OPENPI_ROOT / "src",
    OPENPI_ROOT / "packages" / "openpi-client" / "src",
):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from openpi_client import image_tools, websocket_client_policy  # noqa: E402


JOINT_LIMITS_RAD = [
    (math.radians(-150), math.radians(150)),
    (math.radians(0), math.radians(180)),
    (math.radians(-170), math.radians(0)),
    (math.radians(-100), math.radians(100)),
    (math.radians(-70), math.radians(70)),
    (math.radians(-120), math.radians(120)),
]
GRIPPER_LIMIT = (0.0, 1.0)
DEFAULT_CONTROL_REPO_PATH = "/home/user/code/control_your_robot"
DEFAULT_RESET_JOINT = [
    0.04092797,
    1.34207091,
    -0.73867569,
    0.02720968,
    1.13512722,
    0.29129545,
]
DEFAULT_CAM_HEAD_SERIAL = "323522063521"
DEFAULT_CAM_SIDE_SERIAL = "349222061138"
DEFAULT_CAM_WRIST_SERIAL = "409122272461"
DEFAULT_SPACEMOUSE_DEADZONE = 30.0
DEFAULT_SPACEMOUSE_MAX_VALUE = 350.0
DEFAULT_SPACEMOUSE_TRANS_MOTION = 0.002
DEFAULT_SPACEMOUSE_ROT_SCALE = 0.2
DEFAULT_SPACEMOUSE_OPEN_GRIPPER = 0.6
DEFAULT_SPACEMOUSE_CLOSE_GRIPPER = 0.0


class RTCAttentionSchedule(enum.Enum):
    LINEAR = "linear"
    EXP = "exp"
    ONES = "ones"
    ZEROS = "zeros"


@dataclass
class RTCConfig:
    enabled: bool = True
    execution_horizon: int = 10
    max_guidance_weight: float = 10.0
    prefix_attention_schedule: RTCAttentionSchedule = RTCAttentionSchedule.EXP
    inference_delay_steps: int = 4
    measure_inference_delay: bool = False
    prefetch_threshold: int = 20
    worker_sleep: float = 0.005
    hold_last_action_on_underflow: bool = True
    debug: bool = False


@dataclass
class ObservationSnapshot:
    state7: np.ndarray
    cam_head: np.ndarray
    cam_wrist: np.ndarray
    prompt: str
    cam_side: np.ndarray | None = None
    adv_ind: str | None = None
    captured_at: float = 0.0


@dataclass
class RolloutStats:
    success: int = 0
    failure: int = 0
    discarded: int = 0

    @property
    def total_saved(self) -> int:
        return self.success + self.failure

    @property
    def success_rate(self) -> float:
        if self.total_saved == 0:
            return 0.0
        return 100.0 * float(self.success) / float(self.total_saved)

    def add_saved(self, *, success: bool) -> None:
        if success:
            self.success += 1
        else:
            self.failure += 1

    def add_discarded(self) -> None:
        self.discarded += 1

    def format_line(self, prefix: str = "[summary]") -> str:
        return (
            f"{prefix} saved_total={self.total_saved} "
            f"success={self.success} failure={self.failure} "
            f"success_rate={self.success_rate:.1f}% discarded={self.discarded}"
        )


@dataclass
class ExternalPiperRuntime:
    controller: object
    cam_head: object
    cam_wrist: object
    cam_side: object | None


@dataclass
class SpaceMouseInterventionRuntime:
    controller: object
    active: bool = False
    prev_left_button: int = 0
    prev_right_button: int = 0
    manual_gripper: float = DEFAULT_SPACEMOUSE_CLOSE_GRIPPER
    last_sent_gripper: float = DEFAULT_SPACEMOUSE_CLOSE_GRIPPER

    def reset_episode(self, *, current_gripper: float, keep_active: bool = False) -> None:
        self.active = bool(keep_active)
        self.manual_gripper = float(current_gripper)
        self.last_sent_gripper = float(current_gripper)
        try:
            data = self.controller.get()
            self.prev_left_button = int(data.get("left_button", 0))
            self.prev_right_button = int(data.get("right_button", 0))
        except Exception:
            self.prev_left_button = 0
            self.prev_right_button = 0


class KeyboardPoller:
    """Small non-blocking keyboard reader for episode commands."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._old_attrs: list | None = None
        self._owns_fd = False

        try:
            self._fd = os.open("/dev/tty", os.O_RDONLY | os.O_NONBLOCK)
            self._owns_fd = True
        except Exception:
            if sys.stdin.isatty():
                self._fd = sys.stdin.fileno()

        if self._fd is None:
            return

        try:
            self._old_attrs = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
        except Exception:
            self.close()

    @property
    def available(self) -> bool:
        return self._fd is not None

    def read_key(self) -> str | None:
        if self._fd is None:
            return None

        ready, _, _ = select.select([self._fd], [], [], 0)
        if not ready:
            return None

        try:
            data = os.read(self._fd, 1)
        except BlockingIOError:
            return None
        if not data:
            return None

        if data == b"\x1b":
            seq = b""
            deadline = time.monotonic() + 0.02
            while time.monotonic() < deadline and len(seq) < 2:
                ready, _, _ = select.select([self._fd], [], [], max(0.0, deadline - time.monotonic()))
                if not ready:
                    break
                try:
                    seq += os.read(self._fd, 2 - len(seq))
                except BlockingIOError:
                    break

            arrows = {
                b"[A": "up",
                b"[B": "down",
                b"[C": "right",
                b"[D": "left",
            }
            if seq in arrows:
                return arrows[seq]
            return "esc"

        return data.decode("utf-8", errors="ignore").lower() or None

    def drain(self) -> None:
        if self._fd is None:
            return
        while True:
            ready, _, _ = select.select([self._fd], [], [], 0)
            if not ready:
                return
            try:
                data = os.read(self._fd, 64)
            except BlockingIOError:
                return
            if not data:
                return

    def close(self) -> None:
        if self._fd is not None and self._old_attrs is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)
            except Exception:
                pass
        if self._fd is not None and self._owns_fd:
            try:
                os.close(self._fd)
            except Exception:
                pass
        self._fd = None
        self._old_attrs = None
        self._owns_fd = False


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean value, got: {value!r}")


def _parse_csv_floats(value: str, *, expected: int, name: str) -> list[float]:
    try:
        parsed = [float(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--{name} must be comma-separated floats") from exc
    if len(parsed) != expected:
        raise argparse.ArgumentTypeError(f"--{name} expects {expected} values, got {len(parsed)}")
    return parsed


def _reset_joint_arg(value: str) -> list[float]:
    return _parse_csv_floats(value, expected=6, name="reset-joint")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--control-repo-path", default=DEFAULT_CONTROL_REPO_PATH)
    parser.add_argument("--server-host", default="localhost")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--task-name", default="Pick up the block1 and assemble it.")
    parser.add_argument("--instruction", default=None, help="Explicit prompt. Defaults to --task-name.")
    parser.add_argument("--adv-ind", default=None, help="Use positive/negative when connected to a PiStar server.")

    parser.add_argument("--repo-id", default="assemble_block1_pi05_rtc_rollout")
    parser.add_argument("--output-dir", default="/home/user/.cache/huggingface/lerobot/piper")
    parser.add_argument("--num-episode", type=int, default=100)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--control-dt", type=float, default=0.033333)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--max-step", type=int, default=450)
    parser.add_argument("--timeout-label", choices=["success", "failure", "discard"], default="failure")
    parser.add_argument("--enter-label", choices=["success", "failure", "discard"], default="discard")
    parser.add_argument(
        "--min-save-frames",
        type=int,
        default=30,
        help="Discard any non-empty episode shorter than this many frames. Set to 1 to disable.",
    )
    parser.add_argument("--save-adv-ind", default="none")
    parser.add_argument("--penalty-value", type=float, default=-1.0)
    parser.add_argument("--failure-terminal-reward-label", type=float, default=-1.0)
    parser.add_argument("--image-writer-processes", type=int, default=1)
    parser.add_argument("--image-writer-threads", type=int, default=2)
    parser.add_argument("--image-format", choices=["png", "jpg", "jpeg"], default="jpg")
    parser.add_argument("--release-dataset-after-save", type=_bool_arg, default=True)

    parser.add_argument("--arm-can", default="can0")
    parser.add_argument("--arm-name", default="left_arm")
    parser.add_argument("--state-source", choices=["joint", "qpos"], default="joint")
    parser.add_argument("--cam-head-serial", default=DEFAULT_CAM_HEAD_SERIAL)
    parser.add_argument("--cam-side-serial", default=DEFAULT_CAM_SIDE_SERIAL)
    parser.add_argument("--cam-wrist-serial", default=DEFAULT_CAM_WRIST_SERIAL)
    parser.add_argument("--no-reset-before-episode", action="store_true")
    parser.add_argument("--reset-joint", type=_reset_joint_arg, default=list(DEFAULT_RESET_JOINT))
    parser.add_argument("--post-reset-sleep", type=float, default=2.0)
    parser.add_argument("--reset-gripper", type=float, default=0.0)
    parser.add_argument("--gripper-effort", type=int, default=1000)
    parser.add_argument("--gripper-close-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-close-offset", type=float, default=0.0)
    parser.add_argument("--clip-joint-action", action="store_true")
    parser.add_argument(
        "--move-check",
        action="store_true",
        help="Skip nearly-stationary frames when saving. Default keeps all rollout frames.",
    )
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--skip-non-finite", type=_bool_arg, default=True)
    parser.add_argument("--status-interval-s", type=float, default=1.0)

    parser.add_argument(
        "--enable-spacemouse-intervention",
        action="store_true",
        help="Allow SpaceMouse takeover. Once triggered, the rest of the episode is saved as intervention=1.",
    )
    parser.add_argument("--spacemouse-device-path", default=None)
    parser.add_argument("--spacemouse-grab-device", type=_bool_arg, default=True)
    parser.add_argument(
        "--sticky-intervention",
        type=_bool_arg,
        default=True,
        help="Keep intervention=1 after the first SpaceMouse/home intervention in an episode.",
    )
    parser.add_argument("--spacemouse-deadzone", type=float, default=DEFAULT_SPACEMOUSE_DEADZONE)
    parser.add_argument("--spacemouse-max-value", type=float, default=DEFAULT_SPACEMOUSE_MAX_VALUE)
    parser.add_argument("--spacemouse-trans-motion", type=float, default=DEFAULT_SPACEMOUSE_TRANS_MOTION)
    parser.add_argument("--spacemouse-rot-scale", type=float, default=DEFAULT_SPACEMOUSE_ROT_SCALE)
    parser.add_argument("--spacemouse-open-gripper", type=float, default=DEFAULT_SPACEMOUSE_OPEN_GRIPPER)
    parser.add_argument("--spacemouse-close-gripper", type=float, default=DEFAULT_SPACEMOUSE_CLOSE_GRIPPER)

    parser.add_argument("--rtc-enabled", type=_bool_arg, default=True)
    parser.add_argument("--rtc-execution-horizon", type=int, default=10)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        choices=[item.value for item in RTCAttentionSchedule],
        default=RTCAttentionSchedule.EXP.value,
    )
    parser.add_argument("--rtc-inference-delay-steps", type=int, default=4)
    parser.add_argument("--rtc-measure-inference-delay", type=_bool_arg, default=False)
    parser.add_argument("--rtc-prefetch-threshold", type=int, default=20)
    parser.add_argument("--rtc-worker-sleep", type=float, default=0.005)
    parser.add_argument("--rtc-hold-last-action-on-underflow", type=_bool_arg, default=True)
    parser.add_argument("--rtc-debug", type=_bool_arg, default=False)

    return parser


def build_rtc_config(args: argparse.Namespace) -> RTCConfig:
    return RTCConfig(
        enabled=bool(args.rtc_enabled),
        execution_horizon=int(args.rtc_execution_horizon),
        max_guidance_weight=float(args.rtc_max_guidance_weight),
        prefix_attention_schedule=RTCAttentionSchedule(args.rtc_prefix_attention_schedule),
        inference_delay_steps=int(args.rtc_inference_delay_steps),
        measure_inference_delay=bool(args.rtc_measure_inference_delay),
        prefetch_threshold=int(args.rtc_prefetch_threshold),
        worker_sleep=float(args.rtc_worker_sleep),
        hold_last_action_on_underflow=bool(args.rtc_hold_last_action_on_underflow),
        debug=bool(args.rtc_debug),
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.control_dt <= 0:
        raise ValueError("--control-dt must be positive.")
    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive.")
    if args.max_step < 0:
        raise ValueError("--max-step must be non-negative.")
    if args.min_save_frames < 1:
        raise ValueError("--min-save-frames must be at least 1.")
    if args.post_reset_sleep < 0:
        raise ValueError("--post-reset-sleep must be non-negative.")
    if args.image_writer_processes < 0:
        raise ValueError("--image-writer-processes must be non-negative.")
    if args.image_writer_threads < 0:
        raise ValueError("--image-writer-threads must be non-negative.")
    if args.rtc_execution_horizon < 0:
        raise ValueError("--rtc-execution-horizon must be non-negative.")
    if args.rtc_prefetch_threshold < 1:
        raise ValueError("--rtc-prefetch-threshold must be at least 1.")
    if args.rtc_inference_delay_steps < 0:
        raise ValueError("--rtc-inference-delay-steps must be non-negative.")
    if args.spacemouse_deadzone < 0 or args.spacemouse_deadzone >= 100:
        raise ValueError("--spacemouse-deadzone must be in [0, 100).")
    if args.spacemouse_max_value <= 0:
        raise ValueError("--spacemouse-max-value must be positive.")


def setup_control_repo(control_repo_path: str) -> Path:
    repo_root = Path(control_repo_path).expanduser().resolve()
    src_root = repo_root / "src"
    if not repo_root.exists():
        raise FileNotFoundError(f"control repo not found: {repo_root}")
    if not src_root.exists():
        raise FileNotFoundError(f"control repo src not found: {src_root}")

    for path in (src_root, repo_root):
        path_str = str(path)
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)
    return repo_root


def load_local_rl_collector_class():
    collector_path = LOCAL_CONTROL_YOUR_ROBOT_ROOT / "src" / "robot" / "data" / "collect_lerobot_rl.py"
    spec = importlib.util.spec_from_file_location("pistar_collect_lerobot_rl", collector_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load RL collector from: {collector_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CollectLeRobotRL


def create_rl_collector(args: argparse.Namespace):
    collect_lerobot_rl = load_local_rl_collector_class()
    return collect_lerobot_rl(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        task_name=args.task_name,
        fps=args.fps,
        robot_type="piper",
        state_dim=7,
        action_dim=7,
        image_size=(480, 640),
        camera_keys={
            "cam_head": "image",
            "cam_wrist": "wrist_image",
            "cam_side": "side_image",
        },
        move_check=bool(args.move_check),
        tolerance=0.0005,
        penalty_value=args.penalty_value,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
        image_format=args.image_format,
        release_dataset_after_save=args.release_dataset_after_save,
    )


def create_external_runtime(args: argparse.Namespace) -> ExternalPiperRuntime:
    setup_control_repo(args.control_repo_path)

    try:
        from robot.controller.Piper_controller import PiperController
        from robot.sensor.Realsense_sensor import RealsenseSensor
    except Exception as exc:
        raise ImportError(
            f"Failed to import Piper runtime modules from {args.control_repo_path}. "
            "Check piper_sdk, pyrealsense2, and control_your_robot installation."
        ) from exc

    controller = PiperController(args.arm_name)
    controller.set_up(args.arm_can)
    if hasattr(controller, "set_gripper_effort"):
        controller.set_gripper_effort(args.gripper_effort)

    cam_head = RealsenseSensor("cam_head")
    cam_wrist = RealsenseSensor("cam_wrist")
    cam_side = RealsenseSensor("cam_side") if args.cam_side_serial else None

    cam_head.set_up(args.cam_head_serial)
    cam_wrist.set_up(args.cam_wrist_serial)
    if cam_side is not None:
        cam_side.set_up(args.cam_side_serial)

    cam_head.set_collect_info(["color"])
    cam_wrist.set_collect_info(["color"])
    if cam_side is not None:
        cam_side.set_collect_info(["color"])

    return ExternalPiperRuntime(
        controller=controller,
        cam_head=cam_head,
        cam_wrist=cam_wrist,
        cam_side=cam_side,
    )


def create_spacemouse_runtime(args: argparse.Namespace) -> SpaceMouseInterventionRuntime:
    setup_control_repo(args.control_repo_path)
    try:
        from robot.controller.SpaceMouse_controller import SpaceMouseController
    except Exception as exc:
        raise ImportError(
            f"Failed to import SpaceMouseController from {args.control_repo_path}. "
            "Check the control_your_robot installation and SpaceMouse dependencies."
        ) from exc

    controller = SpaceMouseController(
        name="spacemouse",
        initial_gripper=float(args.spacemouse_close_gripper),
    )
    if hasattr(controller, "set_collect_info"):
        controller.set_collect_info(["raw", "left_button", "right_button"])
    controller.set_up(device_path=args.spacemouse_device_path, grab_device=bool(args.spacemouse_grab_device))
    return SpaceMouseInterventionRuntime(
        controller=controller,
        manual_gripper=float(args.spacemouse_close_gripper),
    )


def cleanup_external_runtime(runtime: ExternalPiperRuntime | None) -> None:
    if runtime is None:
        return
    for sensor in (runtime.cam_head, runtime.cam_wrist, runtime.cam_side):
        if sensor is not None and hasattr(sensor, "cleanup"):
            try:
                sensor.cleanup()
            except Exception:
                pass


def cleanup_spacemouse_runtime(runtime: SpaceMouseInterventionRuntime | None) -> None:
    if runtime is None:
        return
    if hasattr(runtime.controller, "close"):
        try:
            runtime.controller.close()
        except Exception:
            pass


def _to_scalar(value):
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        return _to_scalar(value.reshape(-1)[0])
    if isinstance(value, (list, tuple)):
        if not value:
            return 0.0
        return _to_scalar(value[0])
    if hasattr(value, "item"):
        return value.item()
    return value


def load_existing_rollout_stats(dataset_dir: Path) -> RolloutStats:
    stats = RolloutStats()
    data_dir = dataset_dir / "data"
    if not data_dir.exists():
        return stats

    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        print(f"[warn] could not import pyarrow to count existing episodes: {exc}", flush=True)
        return stats

    for parquet_path in sorted(data_dir.glob("chunk-*/episode_*.parquet")):
        try:
            table = pq.read_table(parquet_path, columns=["reward"])
            if table.num_rows == 0:
                continue
            final_reward = float(_to_scalar(table["reward"].to_pylist()[-1]))
        except Exception as exc:
            print(f"[warn] skip existing episode count for {parquet_path}: {exc}", flush=True)
            continue

        stats.add_saved(success=final_reward > 0.5)

    return stats


def _get_color_image(sensors: dict, *keys: str) -> np.ndarray | None:
    for key in keys:
        cam_data = sensors.get(key)
        if cam_data is not None and "color" in cam_data:
            return cam_data["color"]
    return None


def make_observation(
    robot_data: list[dict],
    *,
    state_source: str,
    prompt: str,
    adv_ind: str | None,
) -> ObservationSnapshot:
    controllers, sensors = robot_data
    arm_state = controllers["left_arm"]
    if state_source not in arm_state:
        raise KeyError(f"state_source={state_source!r} not found in left_arm keys: {list(arm_state.keys())}")
    if "gripper" not in arm_state:
        raise KeyError(f"left_arm state missing gripper key: {list(arm_state.keys())}")

    state6 = np.asarray(arm_state[state_source], dtype=np.float64).reshape(-1)
    if state6.shape != (6,):
        raise ValueError(f"expected 6D {state_source} state, got shape={state6.shape}")
    gripper = np.asarray([float(np.asarray(arm_state["gripper"]).reshape(-1)[0])], dtype=np.float64)
    state7 = np.concatenate([state6, gripper], axis=0)

    img_wrist = _get_color_image(sensors, "cam_wrist", "wrist_image")
    if img_wrist is None:
        raise KeyError(f"Missing wrist camera image. Available sensor keys: {list(sensors.keys())}")
    img_head = _get_color_image(sensors, "cam_head", "image")
    if img_head is None:
        img_head = np.zeros_like(img_wrist)
    img_side = _get_color_image(sensors, "cam_side", "side_image")
    if img_side is None:
        img_side = np.zeros_like(img_head)

    return ObservationSnapshot(
        state7=state7,
        cam_head=np.asarray(img_head),
        cam_wrist=np.asarray(img_wrist),
        cam_side=np.asarray(img_side),
        prompt=prompt,
        adv_ind=adv_ind,
        captured_at=time.monotonic(),
    )


def action_to_targets(action: np.ndarray, args: argparse.Namespace) -> tuple[np.ndarray, float]:
    action7 = np.asarray(action, dtype=np.float64).reshape(-1)[:7]
    if action7.shape[0] < 7:
        raise ValueError(f"expected at least 7D action, got shape={action7.shape}")

    target6 = action7[:6]
    if args.state_source == "joint" and args.clip_joint_action:
        target6 = np.asarray(
            [
                np.clip(float(target6[i]), JOINT_LIMITS_RAD[i][0], JOINT_LIMITS_RAD[i][1])
                for i in range(6)
            ],
            dtype=np.float64,
        )
    gripper = float(np.clip(action7[6], GRIPPER_LIMIT[0], GRIPPER_LIMIT[1]))
    if args.gripper_close_offset > 0 and gripper <= args.gripper_close_threshold:
        gripper = float(np.clip(gripper - args.gripper_close_offset, GRIPPER_LIMIT[0], GRIPPER_LIMIT[1]))

    return target6, gripper


def _map_spacemouse_axis_to_delta(percentage: float, *, deadzone: float, motion: float) -> float:
    if abs(percentage) <= deadzone:
        return 0.0
    if percentage > 0:
        normalized = (percentage - deadzone) / (100.0 - deadzone)
        return -normalized * motion
    normalized = (abs(percentage) - deadzone) / (100.0 - deadzone)
    return normalized * motion


def spacemouse_raw_to_delta_qpos(raw: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    if raw.shape[0] < 6:
        padded = np.zeros((6,), dtype=np.float64)
        padded[: raw.shape[0]] = raw
        raw = padded
    raw = raw[:6]
    percentages = (raw / float(args.spacemouse_max_value)) * 100.0
    deadzone = float(args.spacemouse_deadzone)
    trans_motion = float(args.spacemouse_trans_motion)
    rot_motion = trans_motion * float(args.spacemouse_rot_scale)

    # Match the user's existing SpaceMouse -> Piper qpos mapping.
    distance_x = _map_spacemouse_axis_to_delta(percentages[1], deadzone=deadzone, motion=trans_motion)
    distance_y = _map_spacemouse_axis_to_delta(percentages[0], deadzone=deadzone, motion=trans_motion)
    distance_z = _map_spacemouse_axis_to_delta(percentages[2], deadzone=deadzone, motion=trans_motion)
    distance_ry = _map_spacemouse_axis_to_delta(percentages[3], deadzone=deadzone, motion=rot_motion)
    distance_rx = _map_spacemouse_axis_to_delta(percentages[4], deadzone=deadzone, motion=rot_motion)
    distance_rz = _map_spacemouse_axis_to_delta(percentages[5], deadzone=deadzone, motion=rot_motion)
    return np.asarray([distance_x, distance_y, distance_z, distance_rx, distance_ry, distance_rz], dtype=np.float64)


def get_spacemouse_action(
    spacemouse: SpaceMouseInterventionRuntime,
    robot_data: list[dict],
    args: argparse.Namespace,
) -> tuple[np.ndarray | None, bool, bool, dict]:
    master_data = spacemouse.controller.get()
    raw = np.asarray(master_data.get("raw", np.zeros((6,), dtype=np.float64)), dtype=np.float64).reshape(-1)[:6]
    left_button = int(master_data.get("left_button", 0))
    right_button = int(master_data.get("right_button", 0))
    left_edge = left_button == 1 and spacemouse.prev_left_button == 0
    right_edge = right_button == 1 and spacemouse.prev_right_button == 0
    spacemouse.prev_left_button = left_button
    spacemouse.prev_right_button = right_button

    controllers = robot_data[0]
    arm_state = controllers["left_arm"]
    current_qpos = np.asarray(arm_state["qpos"], dtype=np.float64).reshape(-1)
    if current_qpos.shape != (6,):
        raise ValueError(f"expected 6D qpos for SpaceMouse intervention, got {current_qpos.shape}")
    current_gripper = float(np.asarray(arm_state["gripper"]).reshape(-1)[0])

    if not spacemouse.active:
        spacemouse.manual_gripper = current_gripper

    if left_edge:
        spacemouse.manual_gripper = float(args.spacemouse_open_gripper)
    if right_edge:
        spacemouse.manual_gripper = float(args.spacemouse_close_gripper)

    delta_qpos = spacemouse_raw_to_delta_qpos(raw, args)
    has_motion = bool(np.any(np.abs(delta_qpos) > 0.0))
    gripper_changed = bool(abs(spacemouse.manual_gripper - spacemouse.last_sent_gripper) > 1e-6)
    should_activate = has_motion or left_edge or right_edge
    if should_activate:
        spacemouse.active = True
    elif not spacemouse.active:
        return None, False, False, master_data

    should_send = has_motion or gripper_changed
    if not should_send:
        return None, True, False, master_data

    target_qpos = current_qpos + delta_qpos
    target_gripper = float(np.clip(spacemouse.manual_gripper, GRIPPER_LIMIT[0], GRIPPER_LIMIT[1]))
    spacemouse.last_sent_gripper = target_gripper
    action = np.concatenate([target_qpos, np.asarray([target_gripper], dtype=np.float64)], axis=0)
    return action, True, True, master_data


def execute_action(runtime: ExternalPiperRuntime, action: np.ndarray, args: argparse.Namespace) -> None:
    target6, target_gripper = action_to_targets(action, args)
    if args.state_source == "joint":
        runtime.controller.set_joint(target6)
    else:
        runtime.controller.set_position(target6)
    runtime.controller.set_gripper(target_gripper)


def execute_spacemouse_action(runtime: ExternalPiperRuntime, action: np.ndarray) -> None:
    action7 = np.asarray(action, dtype=np.float64).reshape(-1)[:7]
    if action7.shape[0] < 7:
        raise ValueError(f"expected 7D SpaceMouse action, got shape={action7.shape}")
    runtime.controller.set_position(action7[:6])
    runtime.controller.set_gripper(float(np.clip(action7[6], GRIPPER_LIMIT[0], GRIPPER_LIMIT[1])))


def reset_runtime(runtime: ExternalPiperRuntime, args: argparse.Namespace) -> None:
    reset_joint = np.asarray(args.reset_joint, dtype=np.float64)
    if reset_joint.shape != (6,):
        raise ValueError(f"--reset-joint must be length 6, got {reset_joint.shape}")
    runtime.controller.reset(reset_joint, gripper=float(args.reset_gripper))
    if args.post_reset_sleep > 0:
        time.sleep(args.post_reset_sleep)


def get_runtime_data(runtime: ExternalPiperRuntime) -> list[dict]:
    arm_state = runtime.controller.get_state()
    controller_data = {
        "left_arm": {
            "joint": np.asarray(arm_state["joint"], dtype=np.float64),
            "qpos": np.asarray(arm_state["qpos"], dtype=np.float64),
            "gripper": float(arm_state["gripper"]),
        }
    }
    sensor_data = {
        "cam_head": runtime.cam_head.get_image(),
        "cam_wrist": runtime.cam_wrist.get_image(),
    }
    if runtime.cam_side is not None:
        sensor_data["cam_side"] = runtime.cam_side.get_image()
    return [controller_data, sensor_data]


def wait_for_start(keyboard: KeyboardPoller, episode_idx: int, total: int) -> str:
    print(f"\nEpisode {episode_idx}/{total}: Enter=start, q/Esc=quit", flush=True)
    if not keyboard.available:
        value = input("Press Enter to start, or q to quit: ").strip().lower()
        return "quit" if value in {"q", "quit", "esc"} else "start"

    while True:
        key = keyboard.read_key()
        if key in {"q", "esc"}:
            return "quit"
        if key in {"\n", "\r"}:
            return "start"
        time.sleep(0.05)


def outcome_from_key(key: str | None, *, enter_label: str) -> str | None:
    if key in {"s", "right"}:
        return "success"
    if key == "f":
        return "failure"
    if key in {"r", "left"}:
        return "discard"
    if key in {"q", "esc"}:
        return "quit"
    if key in {"\n", "\r"}:
        return enter_label
    return None


def save_or_discard(
    collector,
    outcome: str,
    args: argparse.Namespace,
    stats: RolloutStats,
) -> bool:
    frame_count = len(collector.episode_buffer)
    if frame_count == 0:
        print("[warn] empty episode, nothing to save.", flush=True)
        return outcome != "quit"

    if outcome != "discard" and frame_count < args.min_save_frames:
        collector.clear_current_episode()
        stats.add_discarded()
        print(
            f"[warn] episode has only {frame_count} frames, below --min-save-frames={args.min_save_frames}; "
            "discarded instead of saving.",
            flush=True,
        )
        print(stats.format_line(), flush=True)
        return True

    if outcome == "discard":
        collector.clear_current_episode()
        stats.add_discarded()
        print(f"[episode] discarded {frame_count} frames", flush=True)
        print(stats.format_line(), flush=True)
        return True

    success = outcome == "success"
    collector.save_episode(
        success=success,
        adv_ind_value=args.save_adv_ind,
        failure_terminal_reward_label=args.failure_terminal_reward_label,
    )
    print(
        f"[episode] saved {frame_count} frames, success={success}, "
        f"intervention={int(np.sum(getattr(collector, 'intervention_flags', [])))}, adv_ind={args.save_adv_ind}",
        flush=True,
    )
    stats.add_saved(success=success)
    print(stats.format_line(), flush=True)
    return True


class OpenPiChunkClient:
    def __init__(self, host: str, port: int, resize_size: int, expected_horizon: int):
        self.policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.resize_size = int(resize_size)
        self.expected_horizon = int(expected_horizon)
        self._infer_lock = threading.Lock()

    def get_server_metadata(self) -> dict:
        with self._infer_lock:
            return self.policy.get_server_metadata()

    def infer_chunk(self, obs: ObservationSnapshot) -> np.ndarray:
        head = image_tools.resize_with_pad(obs.cam_head, self.resize_size, self.resize_size)
        wrist = image_tools.resize_with_pad(obs.cam_wrist, self.resize_size, self.resize_size)
        side = image_tools.resize_with_pad(obs.cam_side, self.resize_size, self.resize_size)

        payload = {
            "observation/state": obs.state7.astype(np.float64),
            "images": {
                "observation/images/cam_head": image_tools.convert_to_uint8(head),
                "observation/images/cam_wrist": image_tools.convert_to_uint8(wrist),
                "observation/images/cam_side": image_tools.convert_to_uint8(side),
            },
            "prompt": obs.prompt,
        }
        if obs.adv_ind is not None:
            payload["adv_ind"] = obs.adv_ind

        start_t = time.perf_counter()
        with self._infer_lock:
            response = self.policy.infer(payload)
        latency_ms = (time.perf_counter() - start_t) * 1000.0
        logging.debug("websocket infer latency: %.2f ms", latency_ms)

        if "actions" not in response:
            raise KeyError(f"policy response missing 'actions': keys={list(response.keys())}")
        actions = np.asarray(response["actions"], dtype=np.float64)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2:
            raise ValueError(f"expected action chunk [T, A], got {actions.shape}")
        if self.expected_horizon > 0:
            if actions.shape[0] != self.expected_horizon:
                logging.warning("Policy returned chunk length %d, expected %d.", actions.shape[0], self.expected_horizon)
            if actions.shape[0] > self.expected_horizon:
                actions = actions[: self.expected_horizon]
        return actions

    def reset(self) -> None:
        with self._infer_lock:
            self.policy.reset()


class ActionQueue:
    def __init__(self, action_dim: int = 7):
        self._lock = threading.Lock()
        self._queue = np.empty((0, action_dim), dtype=np.float64)
        self._action_dim = action_dim

    def clear(self) -> None:
        with self._lock:
            self._queue = np.empty((0, self._action_dim), dtype=np.float64)

    def size(self) -> int:
        with self._lock:
            return int(self._queue.shape[0])

    def is_empty(self) -> bool:
        return self.size() == 0

    def bootstrap(self, new_chunk: np.ndarray) -> None:
        new_chunk = self._normalize_chunk(new_chunk)
        with self._lock:
            self._queue = new_chunk.copy()
            self._action_dim = int(new_chunk.shape[1])

    def pop(self) -> np.ndarray | None:
        with self._lock:
            if self._queue.shape[0] == 0:
                return None
            action = self._queue[0].copy()
            self._queue = self._queue[1:].copy()
            return action

    def merge(self, new_chunk: np.ndarray, *, inference_delay: int, rtc_cfg: RTCConfig) -> dict[str, float | int]:
        new_chunk = self._normalize_chunk(new_chunk)
        with self._lock:
            if self._queue.shape[0] == 0:
                self._queue = new_chunk.copy()
                self._action_dim = int(new_chunk.shape[1])
                return {
                    "old_queue": 0,
                    "new_chunk": int(new_chunk.shape[0]),
                    "aligned_new": int(new_chunk.shape[0]),
                    "stale_trimmed": 0,
                    "guided": 0,
                    "merged_queue": int(self._queue.shape[0]),
                    "guidance_scale": 0.0,
                }

            old_queue = self._queue.copy()
            if old_queue.shape[1] != new_chunk.shape[1]:
                raise ValueError(f"action dim mismatch during merge: old={old_queue.shape}, new={new_chunk.shape}")

            stale_trimmed = min(max(inference_delay, 0), new_chunk.shape[0])
            aligned_new = new_chunk[stale_trimmed:]
            if aligned_new.shape[0] == 0:
                return {
                    "old_queue": int(old_queue.shape[0]),
                    "new_chunk": int(new_chunk.shape[0]),
                    "aligned_new": 0,
                    "stale_trimmed": int(stale_trimmed),
                    "guided": 0,
                    "merged_queue": int(old_queue.shape[0]),
                    "guidance_scale": 0.0,
                }

            overlap_total = min(max(rtc_cfg.execution_horizon, 0), old_queue.shape[0], aligned_new.shape[0])
            guided_len = max(0, overlap_total)

            parts: list[np.ndarray] = []
            guidance_scale = self._guidance_scale(rtc_cfg.max_guidance_weight)
            if guided_len > 0:
                old_overlap = old_queue[:guided_len]
                new_overlap = aligned_new[:guided_len]
                consistency_mask = self._consistency_mask(
                    guided_len,
                    rtc_cfg.prefix_attention_schedule,
                ).reshape(-1, 1)
                old_weight = np.clip(guidance_scale * consistency_mask, 0.0, 1.0)
                blended = old_weight * old_overlap + (1.0 - old_weight) * new_overlap
                parts.append(blended)

            if aligned_new.shape[0] > overlap_total:
                parts.append(aligned_new[overlap_total:].copy())
            elif old_queue.shape[0] > overlap_total:
                parts.append(old_queue[overlap_total:].copy())

            self._queue = (
                np.concatenate(parts, axis=0)
                if parts
                else np.empty((0, aligned_new.shape[1]), dtype=np.float64)
            )

            return {
                "old_queue": int(old_queue.shape[0]),
                "new_chunk": int(new_chunk.shape[0]),
                "aligned_new": int(aligned_new.shape[0]),
                "stale_trimmed": int(stale_trimmed),
                "guided": int(guided_len),
                "merged_queue": int(self._queue.shape[0]),
                "guidance_scale": float(guidance_scale),
            }

    @staticmethod
    def _normalize_chunk(chunk: np.ndarray) -> np.ndarray:
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim == 1:
            chunk = chunk[None, :]
        if chunk.ndim != 2:
            raise ValueError(f"expected action chunk [T, A], got {chunk.shape}")
        return chunk

    @staticmethod
    def _guidance_scale(max_guidance_weight: float) -> float:
        if max_guidance_weight <= 0:
            return 0.0
        return float(np.clip(max_guidance_weight / 10.0, 0.0, 1.0))

    @staticmethod
    def _consistency_mask(length: int, schedule: RTCAttentionSchedule) -> np.ndarray:
        if length <= 0:
            return np.empty((0,), dtype=np.float64)
        if schedule == RTCAttentionSchedule.ONES:
            return np.ones((length,), dtype=np.float64)
        if schedule == RTCAttentionSchedule.ZEROS:
            return np.zeros((length,), dtype=np.float64)
        if schedule == RTCAttentionSchedule.LINEAR:
            return np.linspace(1.0, 0.0, num=length + 2, dtype=np.float64)[1:-1]
        if schedule == RTCAttentionSchedule.EXP:
            x = np.linspace(0.0, 1.0, num=length + 2, dtype=np.float64)[1:-1]
            alpha = 4.0
            decay = np.exp(-alpha * x)
            floor = np.exp(-alpha)
            return (decay - floor) / (1.0 - floor)
        raise ValueError(f"unsupported RTCAttentionSchedule: {schedule}")


class AsyncRTCPolicy:
    def __init__(self, client: OpenPiChunkClient, rtc_cfg: RTCConfig, control_dt: float):
        self.client = client
        self.rtc_cfg = rtc_cfg
        self.control_dt = float(control_dt)
        self.queue = ActionQueue(action_dim=7)

        self._latest_obs: ObservationSnapshot | None = None
        self._obs_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._wakeup_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._inflight_lock = threading.Lock()
        self._inflight = False
        self._started = False

        self.last_latency_s: float | None = None
        self.last_inference_delay_steps = max(0, int(rtc_cfg.inference_delay_steps))
        self.last_action: np.ndarray | None = None

    def start(self) -> None:
        if not self.rtc_cfg.enabled or self._started:
            return
        self._worker = threading.Thread(target=self._worker_loop, daemon=True, name="rtc-worker")
        self._worker.start()
        self._started = True

    def stop(self) -> None:
        self._stop_event.set()
        self._wakeup_event.set()
        if self._worker is not None:
            self._worker.join(timeout=1.0)

    def reset(self) -> None:
        self.queue.clear()
        self.last_action = None
        self.client.reset()

    def update_observation(self, obs: ObservationSnapshot) -> None:
        snapshot = ObservationSnapshot(
            state7=np.asarray(obs.state7, dtype=np.float64).copy(),
            cam_head=np.asarray(obs.cam_head).copy(),
            cam_wrist=np.asarray(obs.cam_wrist).copy(),
            cam_side=None if obs.cam_side is None else np.asarray(obs.cam_side).copy(),
            prompt=str(obs.prompt),
            adv_ind=obs.adv_ind,
            captured_at=float(obs.captured_at),
        )
        with self._obs_lock:
            self._latest_obs = snapshot
        if self.rtc_cfg.enabled and self.queue.size() <= self.rtc_cfg.prefetch_threshold:
            self._wakeup_event.set()

    def bootstrap(self, obs: ObservationSnapshot) -> None:
        self.update_observation(obs)
        if not self.queue.is_empty():
            return
        start_t = time.monotonic()
        chunk = self.client.infer_chunk(obs)
        self.last_latency_s = time.monotonic() - start_t
        measured_steps = max(0, int(np.ceil(self.last_latency_s / self.control_dt)))
        self.last_inference_delay_steps = (
            measured_steps
            if self.rtc_cfg.measure_inference_delay
            else max(0, int(self.rtc_cfg.inference_delay_steps))
        )
        self.queue.bootstrap(chunk)
        if self.rtc_cfg.debug:
            logging.info(
                "RTC bootstrap chunk=%d latency=%.4fs delay_steps=%d",
                chunk.shape[0],
                self.last_latency_s,
                self.last_inference_delay_steps,
            )

    def pop_action(self, hold_action: np.ndarray | None = None) -> np.ndarray | None:
        action = self.queue.pop()
        if action is not None:
            self.last_action = action.copy()
            if self.rtc_cfg.enabled and self.queue.size() <= self.rtc_cfg.prefetch_threshold:
                self._wakeup_event.set()
            return action

        if self.rtc_cfg.hold_last_action_on_underflow:
            if self.last_action is not None:
                return self.last_action.copy()
            if hold_action is not None:
                hold = np.asarray(hold_action, dtype=np.float64).copy()
                self.last_action = hold.copy()
                return hold
        return None

    def _get_latest_obs_copy(self) -> ObservationSnapshot | None:
        with self._obs_lock:
            obs = self._latest_obs
            if obs is None:
                return None
            return ObservationSnapshot(
                state7=obs.state7.copy(),
                cam_head=obs.cam_head.copy(),
                cam_wrist=obs.cam_wrist.copy(),
                cam_side=None if obs.cam_side is None else obs.cam_side.copy(),
                prompt=obs.prompt,
                adv_ind=obs.adv_ind,
                captured_at=obs.captured_at,
            )

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            self._wakeup_event.wait(timeout=self.rtc_cfg.worker_sleep)
            self._wakeup_event.clear()
            if self._stop_event.is_set():
                break
            if self.queue.size() > self.rtc_cfg.prefetch_threshold:
                continue
            with self._inflight_lock:
                if self._inflight:
                    continue
                self._inflight = True

            try:
                obs = self._get_latest_obs_copy()
                if obs is None:
                    continue

                start_t = time.monotonic()
                chunk = self.client.infer_chunk(obs)
                latency_s = time.monotonic() - start_t
                self.last_latency_s = latency_s
                delay_steps = (
                    max(0, int(np.ceil(latency_s / self.control_dt)))
                    if self.rtc_cfg.measure_inference_delay
                    else max(0, int(self.rtc_cfg.inference_delay_steps))
                )
                self.last_inference_delay_steps = delay_steps

                merge_info = self.queue.merge(chunk, inference_delay=delay_steps, rtc_cfg=self.rtc_cfg)
                if self.rtc_cfg.debug:
                    logging.info(
                        "RTC merge latency=%.4fs delay_steps=%d info=%s",
                        latency_s,
                        delay_steps,
                        merge_info,
                    )
            except Exception:
                logging.exception("RTC background inference failed")
                time.sleep(max(self.control_dt, 0.05))
            finally:
                with self._inflight_lock:
                    self._inflight = False


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    logging.basicConfig(level=logging.INFO if args.rtc_debug else logging.WARNING, force=True)

    args.gripper_close_threshold = float(np.clip(args.gripper_close_threshold, 0.0, 1.0))
    args.gripper_close_offset = float(np.clip(args.gripper_close_offset, 0.0, 1.0))
    prompt = args.instruction if args.instruction else args.task_name
    rtc_cfg = build_rtc_config(args)
    dataset_path = Path(args.output_dir) / args.repo_id
    rollout_stats = load_existing_rollout_stats(dataset_path)

    collector = None
    runtime: ExternalPiperRuntime | None = None
    spacemouse: SpaceMouseInterventionRuntime | None = None
    keyboard = KeyboardPoller()

    chunk_client = OpenPiChunkClient(
        host=args.server_host,
        port=args.server_port,
        resize_size=args.resize_size,
        expected_horizon=args.action_horizon,
    )

    print("=" * 72, flush=True)
    print("Autonomous pi0.5/PiStar RTC rollout collection", flush=True)
    print(f"control repo: {Path(args.control_repo_path).expanduser().resolve()}", flush=True)
    print(f"server: ws://{args.server_host}:{args.server_port}", flush=True)
    print(f"prompt: {prompt}", flush=True)
    print(f"inference adv_ind: {args.adv_ind if args.adv_ind is not None else 'None'}", flush=True)
    print(f"save dataset: {dataset_path}", flush=True)
    print(f"fps/control_dt: {args.fps} / {args.control_dt}", flush=True)
    print(
        f"cameras: head={args.cam_head_serial}, side={args.cam_side_serial or 'None'}, wrist={args.cam_wrist_serial}",
        flush=True,
    )
    print(
        f"save mode: image_format={args.image_format}, "
        f"writer={args.image_writer_processes}p/{args.image_writer_threads}t, "
        f"release_after_save={args.release_dataset_after_save}",
        flush=True,
    )
    print(f"RTC: enabled={rtc_cfg.enabled}, exec_horizon={rtc_cfg.execution_horizon}, delay={rtc_cfg.inference_delay_steps}", flush=True)
    print(
        "SpaceMouse intervention: "
        f"enabled={args.enable_spacemouse_intervention}, sticky={args.sticky_intervention}",
        flush=True,
    )
    print("keys during rollout: s=success, f=failure, r=discard, h=home/intervene, q=quit", flush=True)
    print(rollout_stats.format_line(prefix="[existing]"), flush=True)
    print("=" * 72, flush=True)

    try:
        print("[1/3] init robot", flush=True)
        runtime = create_external_runtime(args)
        if args.enable_spacemouse_intervention:
            spacemouse = create_spacemouse_runtime(args)
        collector = create_rl_collector(args)

        print("[2/3] connect policy server", flush=True)
        server_metadata = chunk_client.get_server_metadata()
        print(f"server metadata: {server_metadata}", flush=True)
        if server_metadata.get("requires_adv_ind", False) and not args.adv_ind:
            raise ValueError("Connected policy requires adv_ind. Re-run with --adv-ind positive.")

        print("[3/3] start episodes", flush=True)
        for episode_idx in range(1, args.num_episode + 1):
            if not args.no_reset_before_episode:
                print("[reset] moving to reset joint position", flush=True)
                reset_runtime(runtime, args)
                keyboard.drain()

            command = wait_for_start(keyboard, episode_idx, args.num_episode)
            if command == "quit":
                break

            rtc_policy = AsyncRTCPolicy(chunk_client, rtc_cfg, control_dt=args.control_dt)
            bootstrapped = False
            worker_started = False
            policy_stopped_for_intervention = False
            outcome = None
            next_tick = time.perf_counter()
            last_status_t = time.monotonic()
            frames_since_status = 0
            policy_frames = 0
            intervention_frames = 0
            if spacemouse is not None:
                current_state = runtime.controller.get_state()
                current_gripper = float(np.asarray(current_state.get("gripper", args.reset_gripper)).reshape(-1)[0])
                spacemouse.reset_episode(current_gripper=current_gripper)

            try:
                while outcome is None:
                    key = keyboard.read_key()
                    if key == "h":
                        print("\n[home] keyboard h -> home/reset and mark subsequent frames as intervention", flush=True)
                        if spacemouse is not None:
                            spacemouse.active = True
                        if not policy_stopped_for_intervention:
                            rtc_policy.queue.clear()
                            rtc_policy.stop()
                            policy_stopped_for_intervention = True
                        reset_runtime(runtime, args)
                        if spacemouse is not None:
                            spacemouse.reset_episode(current_gripper=float(args.reset_gripper), keep_active=True)
                        keyboard.drain()
                        next_tick = time.perf_counter()
                        continue

                    outcome = outcome_from_key(key, enter_label=args.enter_label)
                    if outcome is not None:
                        break

                    robot_data = get_runtime_data(runtime)
                    obs = make_observation(
                        robot_data,
                        state_source=args.state_source,
                        prompt=prompt,
                        adv_ind=args.adv_ind,
                    )
                    if not np.isfinite(obs.state7).all():
                        msg = f"Non-finite state detected: {obs.state7}"
                        if args.skip_non_finite:
                            logging.error(msg)
                            continue
                        raise ValueError(msg)

                    action_source = "policy"
                    is_intervention = False
                    should_execute_manual = False
                    manual_action = None
                    if spacemouse is not None:
                        manual_action, is_intervention, should_execute_manual, _ = get_spacemouse_action(
                            spacemouse,
                            robot_data,
                            args,
                        )
                        if is_intervention and not bool(args.sticky_intervention):
                            spacemouse.active = False

                    if is_intervention:
                        action_source = "spacemouse"
                        if not policy_stopped_for_intervention:
                            print(
                                f"\n[intervention] SpaceMouse takeover at frame {len(collector.episode_buffer)}; "
                                "subsequent frames will be saved with intervention=1",
                                flush=True,
                            )
                            rtc_policy.queue.clear()
                            rtc_policy.stop()
                            policy_stopped_for_intervention = True
                        if manual_action is None or not should_execute_manual:
                            pass
                        elif not np.isfinite(manual_action).all():
                            msg = f"SpaceMouse returned non-finite action: {manual_action}"
                            if args.skip_non_finite:
                                logging.error(msg)
                            else:
                                raise ValueError(msg)
                        else:
                            execute_spacemouse_action(runtime, manual_action)
                    else:
                        if rtc_cfg.enabled:
                            rtc_policy.update_observation(obs)
                            if not bootstrapped:
                                rtc_policy.bootstrap(obs)
                                bootstrapped = True
                                if not worker_started:
                                    rtc_policy.start()
                                    worker_started = True
                        elif rtc_policy.queue.is_empty():
                            rtc_policy.bootstrap(obs)

                        action = rtc_policy.pop_action(hold_action=obs.state7)
                        if action is None:
                            logging.warning("RTC queue underflow with no hold action available; skipping step.")
                        elif not np.isfinite(action).all():
                            msg = f"Policy returned non-finite action: {action}"
                            if args.skip_non_finite:
                                logging.error(msg)
                            else:
                                raise ValueError(msg)
                        else:
                            execute_action(runtime, action, args)

                    collector.collect(robot_data[0], robot_data[1], is_intervention=is_intervention)
                    if action_source == "spacemouse":
                        intervention_frames += 1
                    else:
                        policy_frames += 1
                    frames_since_status += 1

                    if args.max_step > 0 and len(collector.episode_buffer) >= args.max_step:
                        outcome = args.timeout_label
                        print(f"\n[episode] max step reached -> {outcome}", flush=True)
                        break

                    now = time.monotonic()
                    if args.status_interval_s > 0 and now - last_status_t >= args.status_interval_s:
                        elapsed = max(now - last_status_t, 1e-6)
                        latency = rtc_policy.last_latency_s
                        latency_str = "None" if latency is None else f"{latency * 1000.0:.1f}ms"
                        print(
                            f"[rollout] frames={len(collector.episode_buffer)} "
                            f"policy={policy_frames} intervention={intervention_frames} "
                            f"rate={frames_since_status / elapsed:.1f}Hz "
                            f"queue={rtc_policy.queue.size()} "
                            f"latency={latency_str} delay={rtc_policy.last_inference_delay_steps}",
                            flush=True,
                        )
                        frames_since_status = 0
                        last_status_t = now

                    next_tick += args.control_dt
                    sleep_s = next_tick - time.perf_counter()
                    if sleep_s > 0:
                        time.sleep(sleep_s)
                    else:
                        next_tick = time.perf_counter()
            finally:
                rtc_policy.stop()
                rtc_policy.reset()

            if outcome == "quit":
                break
            print(
                f"[episode] collected policy frames={policy_frames}, intervention frames={intervention_frames}",
                flush=True,
            )
            if not save_or_discard(collector, outcome, args, rollout_stats):
                break

        print(rollout_stats.format_line(prefix="[done summary]"), flush=True)
        if collector is not None:
            print(f"\n[done] dataset path: {collector.get_dataset_path()}", flush=True)
    finally:
        keyboard.close()
        cleanup_spacemouse_runtime(spacemouse)
        cleanup_external_runtime(runtime)


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
