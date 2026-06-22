#!/usr/bin/env python3
"""Collect PiPER LeRobot RL episodes with a SpaceMouse.

Controls:
  Enter: start an episode
  SpaceMouse motion: move the PiPER end pose
  SpaceMouse left button: open gripper
  SpaceMouse right button: close gripper
  s / Right arrow: save as success
  f: save as failure
  r / Left arrow: discard episode
  q / Esc: quit
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

CONTROL_YOUR_ROBOT_ROOT = Path(__file__).resolve().parents[2]
for path in (CONTROL_YOUR_ROBOT_ROOT, CONTROL_YOUR_ROBOT_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from my_robot.piper_single_lerobot import PiperSingleLeRobot  # noqa: E402
from robot.controller.spacemouse_piper_utils import (  # noqa: E402
    DEFAULT_CLOSE_GRIPPER,
    DEFAULT_DEADZONE,
    DEFAULT_MAX_VALUE,
    DEFAULT_OPEN_GRIPPER,
    DEFAULT_QPOS_RPY_MAX,
    DEFAULT_QPOS_RPY_MIN,
    DEFAULT_QPOS_XYZ_MAX,
    DEFAULT_QPOS_XYZ_MIN,
    DEFAULT_RESET_GRIPPER,
    DEFAULT_ROT_SCALE,
    DEFAULT_TRANS_MOTION,
    KeyboardPoller,
    SpaceMouseReader,
    SpaceMouseTeleopState,
    clip_qpos,
    has_motion,
    parse_csv_floats,
    raw_to_delta_qpos,
    update_gripper_from_buttons,
)
from robot.data.collect_lerobot_rl import CollectLeRobotRL  # noqa: E402


def _csv3(name: str):
    def parse(value: str) -> np.ndarray:
        try:
            return parse_csv_floats(value, expected=3, name=name)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(str(exc)) from exc

    return parse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="piper_assemble_blocks_spacemouse")
    parser.add_argument("--output-dir", default="/home/user/piper_datasets")
    parser.add_argument("--task-name", default="assemble the blocks")
    parser.add_argument("--num-episode", type=int, default=100)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--control-hz", type=float, default=200.0)
    parser.add_argument("--arm-can", default="can0")
    parser.add_argument("--max-step", type=int, default=0, help="0 means no frame limit.")
    parser.add_argument("--penalty-value", type=float, default=-1.0)
    parser.add_argument("--success-adv-ind", default="positive")
    parser.add_argument("--failure-adv-ind", default="negative")
    parser.add_argument("--failure-terminal-reward-label", type=float, default=-1.0)
    parser.add_argument("--enter-label", choices=["success", "failure", "discard"], default="success")
    parser.add_argument("--no-reset-before-episode", action="store_true")
    parser.add_argument("--no-move-check", action="store_true")
    parser.add_argument("--gripper-effort", type=int, default=1000)

    parser.add_argument("--spacemouse", default=None, help="Optional /dev/input/eventX path.")
    parser.add_argument("--no-grab", action="store_true")
    parser.add_argument("--deadzone", type=float, default=DEFAULT_DEADZONE)
    parser.add_argument("--max-value", type=float, default=DEFAULT_MAX_VALUE)
    parser.add_argument("--trans-motion", type=float, default=DEFAULT_TRANS_MOTION)
    parser.add_argument("--rot-scale", type=float, default=DEFAULT_ROT_SCALE)
    parser.add_argument("--open-gripper", type=float, default=DEFAULT_OPEN_GRIPPER)
    parser.add_argument("--close-gripper", type=float, default=DEFAULT_CLOSE_GRIPPER)
    parser.add_argument("--reset-gripper", type=float, default=DEFAULT_RESET_GRIPPER)

    parser.add_argument("--use-workspace-limit", action="store_true")
    parser.add_argument("--xyz-min", type=_csv3("xyz-min"), default=DEFAULT_QPOS_XYZ_MIN)
    parser.add_argument("--xyz-max", type=_csv3("xyz-max"), default=DEFAULT_QPOS_XYZ_MAX)
    parser.add_argument("--rpy-min", type=_csv3("rpy-min"), default=DEFAULT_QPOS_RPY_MIN)
    parser.add_argument("--rpy-max", type=_csv3("rpy-max"), default=DEFAULT_QPOS_RPY_MAX)
    parser.add_argument("--status-interval-s", type=float, default=1.0)
    return parser


def install_rl_collector(robot: PiperSingleLeRobot, args: argparse.Namespace) -> None:
    base_collection = robot.collection
    robot.collection = CollectLeRobotRL(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        task_name=args.task_name,
        fps=args.fps,
        robot_type=base_collection.robot_type,
        state_dim=base_collection.state_dim,
        action_dim=base_collection.action_dim,
        image_size=base_collection.image_size,
        camera_keys=base_collection.camera_keys,
        move_check=not args.no_move_check,
        tolerance=base_collection.tolerance,
        penalty_value=args.penalty_value,
    )


def wait_for_start(keyboard: KeyboardPoller, episode_idx: int, total: int) -> str:
    print(f"\nEpisode {episode_idx}/{total}: Enter=start, q/Esc=quit", flush=True)
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


def move_with_spacemouse(
    robot: PiperSingleLeRobot,
    spacemouse: SpaceMouseReader,
    state: SpaceMouseTeleopState,
    args: argparse.Namespace,
) -> bool:
    master = spacemouse.read()
    raw = np.asarray(master["raw"], dtype=np.float64)
    button_changed = update_gripper_from_buttons(
        state,
        left_button=int(master["left_button"]),
        right_button=int(master["right_button"]),
        open_gripper=args.open_gripper,
        close_gripper=args.close_gripper,
    )
    delta_qpos = raw_to_delta_qpos(
        raw,
        deadzone=args.deadzone,
        max_value=args.max_value,
        trans_motion=args.trans_motion,
        rot_scale=args.rot_scale,
    )

    sent = False
    left_arm = robot.controllers["arm"]["left_arm"]
    if has_motion(delta_qpos):
        current_qpos = np.asarray(left_arm.get_state()["qpos"], dtype=np.float64)
        target_qpos = clip_qpos(
            current_qpos + delta_qpos,
            use_workspace_limit=args.use_workspace_limit,
            xyz_min=args.xyz_min,
            xyz_max=args.xyz_max,
            rpy_min=args.rpy_min,
            rpy_max=args.rpy_max,
        )
        robot.move({"arm": {"left_arm": {"qpos": target_qpos.tolist()}}})
        sent = True

    gripper_changed = button_changed or abs(state.gripper - state.last_sent_gripper) > 1e-6
    if gripper_changed:
        robot.move({"arm": {"left_arm": {"gripper": float(state.gripper)}}})
        state.last_sent_gripper = float(state.gripper)
        sent = True

    return sent


def save_or_discard(robot: PiperSingleLeRobot, outcome: str, args: argparse.Namespace) -> bool:
    frame_count = len(robot.collection.episode_buffer)
    if frame_count == 0:
        print("[warn] empty episode, nothing to save.", flush=True)
        return outcome != "quit"

    if outcome == "discard":
        robot.collection.clear_current_episode()
        print(f"[episode] discarded {frame_count} frames", flush=True)
        return True

    success = outcome == "success"
    adv_ind = args.success_adv_ind if success else args.failure_adv_ind
    robot.collection.save_episode(
        success=success,
        adv_ind_value=adv_ind,
        failure_terminal_reward_label=args.failure_terminal_reward_label,
    )
    print(f"[episode] saved {frame_count} frames, success={success}, adv_ind={adv_ind}", flush=True)
    return True


def run(args: argparse.Namespace) -> None:
    if args.control_hz <= 0:
        raise ValueError("--control-hz must be positive.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")

    robot = PiperSingleLeRobot(
        repo_id=args.repo_id,
        output_dir=args.output_dir,
        task_name=args.task_name,
        fps=args.fps,
        move_check=not args.no_move_check,
        arm_can=args.arm_can,
    )
    install_rl_collector(robot, args)

    spacemouse = SpaceMouseReader(args.spacemouse, grab_device=not args.no_grab)
    keyboard = KeyboardPoller()
    teleop_state = SpaceMouseTeleopState(
        gripper=float(args.reset_gripper),
        last_sent_gripper=float(args.reset_gripper),
    )

    print("=" * 60, flush=True)
    print("SpaceMouse PiPER LeRobot RL collection", flush=True)
    print(f"dataset: {Path(args.output_dir) / args.repo_id}", flush=True)
    print(f"task: {args.task_name}", flush=True)
    print(f"spacemouse: {spacemouse.name} {spacemouse.path}", flush=True)
    print("=" * 60, flush=True)

    try:
        robot.set_up()
        robot.controllers["arm"]["left_arm"].set_gripper_effort(args.gripper_effort)
        robot.move({"arm": {"left_arm": {"gripper": float(args.reset_gripper)}}})
        initial_input = spacemouse.read()
        teleop_state.prev_left_button = int(initial_input["left_button"])
        teleop_state.prev_right_button = int(initial_input["right_button"])

        for episode_idx in range(1, args.num_episode + 1):
            if not args.no_reset_before_episode:
                print("[reset] moving to reset joint position", flush=True)
                robot.reset()
                time.sleep(1.0)
                robot.move({"arm": {"left_arm": {"gripper": float(args.reset_gripper)}}})
                teleop_state.gripper = float(args.reset_gripper)
                teleop_state.last_sent_gripper = float(args.reset_gripper)

            command = wait_for_start(keyboard, episode_idx, args.num_episode)
            if command == "quit":
                break

            print(
                "[record] SpaceMouse=move, left=open, right=close, "
                "s=success, f=failure, r=discard, q=quit",
                flush=True,
            )
            period_s = 1.0 / float(args.control_hz)
            sample_period_s = 1.0 / float(args.fps)
            next_t = time.perf_counter()
            next_sample_t = time.perf_counter()
            last_status_t = time.monotonic()
            loops_since_status = 0
            commands_since_status = 0
            outcome = None

            while outcome is None:
                key = keyboard.read_key()
                outcome = outcome_from_key(key, enter_label=args.enter_label)
                if outcome is not None:
                    break

                if move_with_spacemouse(robot, spacemouse, teleop_state, args):
                    commands_since_status += 1

                now_perf = time.perf_counter()
                if now_perf >= next_sample_t:
                    data = robot.get()
                    robot.collection.collect(data[0], data[1], is_intervention=True)
                    next_sample_t += sample_period_s
                    if args.max_step > 0 and len(robot.collection.episode_buffer) >= args.max_step:
                        outcome = args.enter_label
                        break

                loops_since_status += 1
                now = time.monotonic()
                if args.status_interval_s > 0 and now - last_status_t >= args.status_interval_s:
                    elapsed = now - last_status_t
                    print(
                        f"[record] loop={loops_since_status / elapsed:.1f}Hz, "
                        f"commands={commands_since_status / elapsed:.1f}Hz, "
                        f"frames={len(robot.collection.episode_buffer)}, "
                        f"gripper={teleop_state.gripper:.3f}",
                        flush=True,
                    )
                    loops_since_status = 0
                    commands_since_status = 0
                    last_status_t = now

                next_t += period_s
                sleep_s = next_t - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
                else:
                    next_t = time.perf_counter()

            if outcome == "quit":
                break
            if not save_or_discard(robot, outcome, args):
                break

        print(f"\n[done] dataset path: {robot.collection.get_dataset_path()}", flush=True)
    finally:
        keyboard.close()
        spacemouse.close()


def main() -> None:
    parser = build_arg_parser()
    run(parser.parse_args())


if __name__ == "__main__":
    main()
