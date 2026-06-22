"""SpaceMouse helpers shared by PiPER teleop and rollout examples."""

from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_DEADZONE = 30.0
DEFAULT_MAX_VALUE = 350.0
DEFAULT_TRANS_MOTION = 0.003
DEFAULT_ROT_SCALE = 0.5
DEFAULT_OPEN_GRIPPER = 1.0
DEFAULT_CLOSE_GRIPPER = 0.0
DEFAULT_RESET_GRIPPER = 0.0

DEFAULT_QPOS_XYZ_MIN = np.array([0.10, -0.40, 0.0], dtype=np.float64)
DEFAULT_QPOS_XYZ_MAX = np.array([0.60, 0.40, 0.80], dtype=np.float64)
DEFAULT_QPOS_RPY_MIN = np.array([-3.14, -1.57, -3.14], dtype=np.float64)
DEFAULT_QPOS_RPY_MAX = np.array([3.14, 1.57, 3.14], dtype=np.float64)


class SpaceMouseReader:
    """Read 3Dconnexion SpaceMouse events from Linux evdev."""

    def __init__(self, device_path: str | None = None, *, grab_device: bool = True) -> None:
        try:
            from evdev import InputDevice, ecodes, list_devices
        except ImportError as exc:
            raise ImportError("Missing SpaceMouse dependency. Install it with: pip install evdev") from exc

        self._input_device_cls = InputDevice
        self._ecodes = ecodes
        self._list_devices = list_devices
        self._device = self._connect(device_path)
        self._grabbed = False
        self._axis = [0, 0, 0, 0, 0, 0]
        self._buttons = [0, 0]

        if grab_device:
            try:
                self._device.grab()
                self._grabbed = True
            except Exception:
                print("[spacemouse] could not grab device exclusively; continuing.", flush=True)

    @property
    def name(self) -> str:
        return getattr(self._device, "name", "SpaceMouse")

    @property
    def path(self) -> str:
        return getattr(self._device, "path", "")

    def _connect(self, device_path: str | None) -> Any:
        if device_path is not None:
            return self._input_device_cls(device_path)

        devices = []
        for path in self._list_devices():
            try:
                devices.append(self._input_device_cls(path))
            except Exception:
                pass

        for dev in devices:
            name = getattr(dev, "name", "")
            if "3Dconnexion" in name or "SpaceMouse" in name:
                return dev

        raise RuntimeError(
            "SpaceMouse device not found. Check USB connection and /dev/input/event* permissions."
        )

    def read(self) -> dict[str, Any]:
        ecodes = self._ecodes
        frame_axis = [0, 0, 0, 0, 0, 0]
        got_rel_event = False

        rel_to_idx = {
            ecodes.REL_X: 0,
            ecodes.REL_Y: 1,
            ecodes.REL_Z: 2,
            ecodes.REL_RX: 3,
            ecodes.REL_RY: 4,
            ecodes.REL_RZ: 5,
        }
        abs_to_idx = {
            ecodes.ABS_X: 0,
            ecodes.ABS_Y: 1,
            ecodes.ABS_Z: 2,
            ecodes.ABS_RX: 3,
            ecodes.ABS_RY: 4,
            ecodes.ABS_RZ: 5,
        }

        while True:
            try:
                event = self._device.read_one()
            except Exception:
                break
            if event is None:
                break
            if event.type == ecodes.EV_REL and event.code in rel_to_idx:
                frame_axis[rel_to_idx[event.code]] += event.value
                got_rel_event = True
            elif event.type == ecodes.EV_ABS and event.code in abs_to_idx:
                self._axis[abs_to_idx[event.code]] = event.value
            elif event.type == ecodes.EV_KEY:
                if event.code in (256, getattr(ecodes, "BTN_0", 256)):
                    self._buttons[0] = int(event.value)
                elif event.code in (257, getattr(ecodes, "BTN_1", 257)):
                    self._buttons[1] = int(event.value)

        raw = frame_axis if got_rel_event else self._axis
        return {
            "raw": np.asarray(raw, dtype=np.float64),
            "left_button": self._buttons[0],
            "right_button": self._buttons[1],
        }

    def close(self) -> None:
        if self._grabbed:
            try:
                self._device.ungrab()
            except Exception:
                pass
        self._grabbed = False


class KeyboardPoller:
    """Non-blocking keyboard reader that works while stdin is in cbreak mode."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._old_attrs: list[Any] | None = None
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


@dataclass
class SpaceMouseTeleopState:
    prev_left_button: int = 0
    prev_right_button: int = 0
    gripper: float = DEFAULT_RESET_GRIPPER
    last_sent_gripper: float = DEFAULT_RESET_GRIPPER


def map_spacemouse_to_distance(percentage: float, *, deadzone: float, motion: float) -> float:
    if abs(percentage) <= deadzone:
        return 0.0
    if percentage > 0:
        normalized = (percentage - deadzone) / (100.0 - deadzone)
        return -normalized * motion
    normalized = (abs(percentage) - deadzone) / (100.0 - deadzone)
    return normalized * motion


def raw_to_delta_qpos(
    raw: np.ndarray,
    *,
    deadzone: float = DEFAULT_DEADZONE,
    max_value: float = DEFAULT_MAX_VALUE,
    trans_motion: float = DEFAULT_TRANS_MOTION,
    rot_scale: float = DEFAULT_ROT_SCALE,
) -> np.ndarray:
    """Map [x, y, z, rx, ry, rz] SpaceMouse input to PiPER end-pose deltas."""
    percentages = (np.asarray(raw, dtype=np.float64) / max_value) * 100.0

    distance_x = map_spacemouse_to_distance(percentages[1], deadzone=deadzone, motion=trans_motion)
    distance_y = map_spacemouse_to_distance(percentages[0], deadzone=deadzone, motion=trans_motion)
    distance_z = map_spacemouse_to_distance(percentages[2], deadzone=deadzone, motion=trans_motion)
    distance_ry = (
        map_spacemouse_to_distance(percentages[3], deadzone=deadzone, motion=trans_motion) * rot_scale
    )
    distance_rx = (
        map_spacemouse_to_distance(percentages[4], deadzone=deadzone, motion=trans_motion) * rot_scale
    )
    distance_rz = (
        map_spacemouse_to_distance(percentages[5], deadzone=deadzone, motion=trans_motion) * rot_scale
    )

    return np.array(
        [distance_x, distance_y, distance_z, distance_rx, distance_ry, distance_rz],
        dtype=np.float64,
    )


def update_gripper_from_buttons(
    state: SpaceMouseTeleopState,
    *,
    left_button: int,
    right_button: int,
    open_gripper: float = DEFAULT_OPEN_GRIPPER,
    close_gripper: float = DEFAULT_CLOSE_GRIPPER,
) -> bool:
    left_edge = left_button == 1 and state.prev_left_button == 0
    right_edge = right_button == 1 and state.prev_right_button == 0
    if left_edge:
        state.gripper = float(open_gripper)
    if right_edge:
        state.gripper = float(close_gripper)
    state.prev_left_button = int(left_button)
    state.prev_right_button = int(right_button)
    return left_edge or right_edge


def has_motion(delta_qpos: np.ndarray) -> bool:
    return bool(np.any(np.abs(delta_qpos) > 0.0))


def clip_qpos(
    qpos: np.ndarray,
    *,
    use_workspace_limit: bool,
    xyz_min: np.ndarray = DEFAULT_QPOS_XYZ_MIN,
    xyz_max: np.ndarray = DEFAULT_QPOS_XYZ_MAX,
    rpy_min: np.ndarray = DEFAULT_QPOS_RPY_MIN,
    rpy_max: np.ndarray = DEFAULT_QPOS_RPY_MAX,
) -> np.ndarray:
    clipped = np.asarray(qpos, dtype=np.float64).copy()
    if use_workspace_limit:
        clipped[:3] = np.clip(clipped[:3], xyz_min, xyz_max)
        clipped[3:] = np.clip(clipped[3:], rpy_min, rpy_max)
    return clipped


def parse_csv_floats(value: str, *, expected: int, name: str) -> np.ndarray:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if len(items) != expected:
        raise ValueError(f"{name} must contain {expected} comma-separated values.")
    try:
        return np.asarray([float(item) for item in items], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{name} must contain only numbers.") from exc
