#!/usr/bin/env python3
"""Evaluate pi0.5/PiStar RTC success rate without saving rollout data.

The script runs a fixed evaluation plan and only records keyboard labels in
memory. No LeRobot dataset is created or written.

Default plan:
  - 30 in-distribution trials
  - 10 position-OOD trials
  - 10 angle-OOD trials

Keys:
  - Enter before a trial: start
  - s / Right arrow during a trial: success
  - f during a trial: failure
  - r / Left arrow during a trial: discard and retry current trial
  - q / Esc: quit
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import logging
import time
from pathlib import Path

import numpy as np

import piper_pi05_rtc_rollout_collect as rollout


@dataclass(frozen=True)
class TrialSpec:
    condition: str
    index_in_condition: int
    total_in_condition: int


@dataclass
class ConditionStats:
    success: int = 0
    failure: int = 0

    @property
    def total(self) -> int:
        return self.success + self.failure

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return 100.0 * self.success / self.total

    def add(self, outcome: str) -> None:
        if outcome == "success":
            self.success += 1
        elif outcome == "failure":
            self.failure += 1
        else:
            raise ValueError(f"Unsupported outcome: {outcome}")


def _bool_arg(value: str | bool) -> bool:
    return rollout._bool_arg(value)


def _reset_joint_arg(value: str) -> list[float]:
    return rollout._reset_joint_arg(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--control-repo-path", default=rollout.DEFAULT_CONTROL_REPO_PATH)
    parser.add_argument("--server-host", default="localhost")
    parser.add_argument("--server-port", type=int, default=8000)
    parser.add_argument("--task-name", default="Pick up the block1 and assemble it.")
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--adv-ind", default=None, help="Use positive/negative when connected to a PiStar server.")

    parser.add_argument("--num-id", type=int, default=30)
    parser.add_argument("--num-position-ood", type=int, default=10)
    parser.add_argument("--num-angle-ood", type=int, default=10)
    parser.add_argument("--max-step", type=int, default=1500)
    parser.add_argument("--timeout-label", choices=["success", "failure", "discard"], default="failure")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--control-dt", type=float, default=0.033333)
    parser.add_argument("--action-horizon", type=int, default=50)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--skip-non-finite", type=_bool_arg, default=True)
    parser.add_argument("--status-interval-s", type=float, default=1.0)

    parser.add_argument("--arm-can", default="can0")
    parser.add_argument("--arm-name", default="left_arm")
    parser.add_argument("--state-source", choices=["joint", "qpos"], default="joint")
    parser.add_argument("--cam-head-serial", default=rollout.DEFAULT_CAM_HEAD_SERIAL)
    parser.add_argument("--cam-side-serial", default=rollout.DEFAULT_CAM_SIDE_SERIAL)
    parser.add_argument("--cam-wrist-serial", default=rollout.DEFAULT_CAM_WRIST_SERIAL)
    parser.add_argument("--no-reset-before-trial", action="store_true")
    parser.add_argument("--reset-joint", type=_reset_joint_arg, default=list(rollout.DEFAULT_RESET_JOINT))
    parser.add_argument("--post-reset-sleep", type=float, default=2.0)
    parser.add_argument("--reset-gripper", type=float, default=0.0)
    parser.add_argument("--gripper-effort", type=int, default=1000)
    parser.add_argument("--gripper-close-threshold", type=float, default=0.5)
    parser.add_argument("--gripper-close-offset", type=float, default=0.0)
    parser.add_argument("--clip-joint-action", action="store_true")

    parser.add_argument("--rtc-enabled", type=_bool_arg, default=True)
    parser.add_argument("--rtc-execution-horizon", type=int, default=10)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    parser.add_argument(
        "--rtc-prefix-attention-schedule",
        choices=[item.value for item in rollout.RTCAttentionSchedule],
        default=rollout.RTCAttentionSchedule.EXP.value,
    )
    parser.add_argument("--rtc-inference-delay-steps", type=int, default=4)
    parser.add_argument("--rtc-measure-inference-delay", type=_bool_arg, default=False)
    parser.add_argument("--rtc-prefetch-threshold", type=int, default=20)
    parser.add_argument("--rtc-worker-sleep", type=float, default=0.005)
    parser.add_argument("--rtc-hold-last-action-on-underflow", type=_bool_arg, default=True)
    parser.add_argument("--rtc-debug", type=_bool_arg, default=False)

    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.num_id < 0 or args.num_position_ood < 0 or args.num_angle_ood < 0:
        raise ValueError("Evaluation counts must be non-negative.")
    if args.num_id + args.num_position_ood + args.num_angle_ood <= 0:
        raise ValueError("At least one evaluation trial is required.")
    if args.fps <= 0:
        raise ValueError("--fps must be positive.")
    if args.control_dt <= 0:
        raise ValueError("--control-dt must be positive.")
    if args.action_horizon <= 0:
        raise ValueError("--action-horizon must be positive.")
    if args.max_step < 0:
        raise ValueError("--max-step must be non-negative.")
    if args.rtc_execution_horizon < 0:
        raise ValueError("--rtc-execution-horizon must be non-negative.")
    if args.rtc_prefetch_threshold < 1:
        raise ValueError("--rtc-prefetch-threshold must be at least 1.")
    if args.rtc_inference_delay_steps < 0:
        raise ValueError("--rtc-inference-delay-steps must be non-negative.")


def build_eval_plan(args: argparse.Namespace) -> list[TrialSpec]:
    plan: list[TrialSpec] = []
    for condition, count in (
        ("id", args.num_id),
        ("position_ood", args.num_position_ood),
        ("angle_ood", args.num_angle_ood),
    ):
        for idx in range(1, count + 1):
            plan.append(TrialSpec(condition=condition, index_in_condition=idx, total_in_condition=count))
    return plan


def condition_instruction(condition: str) -> str:
    if condition == "id":
        return "Set the block to an in-distribution pose."
    if condition == "position_ood":
        return "Set the block to a position-OOD location."
    if condition == "angle_ood":
        return "Set the block to an angle-OOD orientation."
    return f"Set up condition: {condition}."


def wait_for_start(keyboard: rollout.KeyboardPoller, spec: TrialSpec, global_index: int, total: int) -> str:
    print(
        f"\nTrial {global_index}/{total} | {spec.condition} "
        f"{spec.index_in_condition}/{spec.total_in_condition}",
        flush=True,
    )
    print(f"[setup] {condition_instruction(spec.condition)}", flush=True)
    print("Enter=start, q/Esc=quit", flush=True)
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


def outcome_from_key(key: str | None) -> str | None:
    if key in {"s", "right"}:
        return "success"
    if key == "f":
        return "failure"
    if key in {"r", "left"}:
        return "discard"
    if key in {"q", "esc"}:
        return "quit"
    return None


def print_summary(stats: dict[str, ConditionStats], discarded: int, *, prefix: str = "[summary]") -> None:
    total_success = sum(item.success for item in stats.values())
    total_failure = sum(item.failure for item in stats.values())
    total = total_success + total_failure
    total_rate = 0.0 if total == 0 else 100.0 * total_success / total
    print(
        f"{prefix} total={total} success={total_success} failure={total_failure} "
        f"success_rate={total_rate:.1f}% discarded={discarded}",
        flush=True,
    )
    for condition, item in stats.items():
        print(
            f"  {condition}: total={item.total} success={item.success} failure={item.failure} "
            f"success_rate={item.success_rate:.1f}%",
            flush=True,
        )


def run_trial(
    *,
    args: argparse.Namespace,
    runtime: rollout.ExternalPiperRuntime,
    keyboard: rollout.KeyboardPoller,
    chunk_client: rollout.OpenPiChunkClient,
    rtc_cfg: rollout.RTCConfig,
    prompt: str,
) -> str:
    rtc_policy = rollout.AsyncRTCPolicy(chunk_client, rtc_cfg, control_dt=args.control_dt)
    bootstrapped = False
    worker_started = False
    outcome = None
    next_tick = time.perf_counter()
    last_status_t = time.monotonic()
    frames_since_status = 0
    frames = 0

    try:
        while outcome is None:
            key = keyboard.read_key()
            outcome = outcome_from_key(key)
            if outcome is not None:
                break

            robot_data = rollout.get_runtime_data(runtime)
            obs = rollout.make_observation(
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
                rollout.execute_action(runtime, action, args)

            frames += 1
            frames_since_status += 1

            if args.max_step > 0 and frames >= args.max_step:
                outcome = args.timeout_label
                print(f"\n[trial] max step reached -> {outcome}", flush=True)
                break

            now = time.monotonic()
            if args.status_interval_s > 0 and now - last_status_t >= args.status_interval_s:
                elapsed = max(now - last_status_t, 1e-6)
                latency = rtc_policy.last_latency_s
                latency_str = "None" if latency is None else f"{latency * 1000.0:.1f}ms"
                print(
                    f"[eval] frames={frames} rate={frames_since_status / elapsed:.1f}Hz "
                    f"queue={rtc_policy.queue.size()} latency={latency_str} "
                    f"delay={rtc_policy.last_inference_delay_steps}",
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

    assert outcome is not None
    print(f"[trial] outcome={outcome} frames={frames}", flush=True)
    return outcome


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    logging.basicConfig(level=logging.INFO if args.rtc_debug else logging.WARNING, force=True)

    args.gripper_close_threshold = float(np.clip(args.gripper_close_threshold, 0.0, 1.0))
    args.gripper_close_offset = float(np.clip(args.gripper_close_offset, 0.0, 1.0))
    prompt = args.instruction if args.instruction else args.task_name
    rtc_cfg = rollout.build_rtc_config(args)
    plan = build_eval_plan(args)
    stats = {
        "id": ConditionStats(),
        "position_ood": ConditionStats(),
        "angle_ood": ConditionStats(),
    }
    discarded = 0

    keyboard = rollout.KeyboardPoller()
    runtime: rollout.ExternalPiperRuntime | None = None
    chunk_client = rollout.OpenPiChunkClient(
        host=args.server_host,
        port=args.server_port,
        resize_size=args.resize_size,
        expected_horizon=args.action_horizon,
    )

    print("=" * 72, flush=True)
    print("pi0.5/PiStar RTC success-rate evaluation (no dataset saving)", flush=True)
    print(f"control repo: {Path(args.control_repo_path).expanduser().resolve()}", flush=True)
    print(f"server: ws://{args.server_host}:{args.server_port}", flush=True)
    print(f"prompt: {prompt}", flush=True)
    print(f"inference adv_ind: {args.adv_ind if args.adv_ind is not None else 'None'}", flush=True)
    print(
        f"plan: id={args.num_id}, position_ood={args.num_position_ood}, "
        f"angle_ood={args.num_angle_ood}, total={len(plan)}",
        flush=True,
    )
    print(f"RTC: enabled={rtc_cfg.enabled}, exec_horizon={rtc_cfg.execution_horizon}, delay={rtc_cfg.inference_delay_steps}", flush=True)
    print("keys during trial: s=success, f=failure, r=discard/retry, q=quit", flush=True)
    print("=" * 72, flush=True)

    try:
        print("[1/2] init robot", flush=True)
        runtime = rollout.create_external_runtime(args)

        print("[2/2] connect policy server", flush=True)
        server_metadata = chunk_client.get_server_metadata()
        print(f"server metadata: {server_metadata}", flush=True)
        if server_metadata.get("requires_adv_ind", False) and not args.adv_ind:
            raise ValueError("Connected policy requires adv_ind. Re-run with --adv-ind positive.")

        plan_pos = 0
        while plan_pos < len(plan):
            spec = plan[plan_pos]
            global_index = plan_pos + 1
            if not args.no_reset_before_trial:
                print("[reset] moving to reset joint position", flush=True)
                rollout.reset_runtime(runtime, args)
                keyboard.drain()

            command = wait_for_start(keyboard, spec, global_index, len(plan))
            if command == "quit":
                break

            outcome = run_trial(
                args=args,
                runtime=runtime,
                keyboard=keyboard,
                chunk_client=chunk_client,
                rtc_cfg=rtc_cfg,
                prompt=prompt,
            )
            if outcome == "quit":
                break
            if outcome == "discard":
                discarded += 1
                print("[trial] discarded; retrying the same trial slot.", flush=True)
                print_summary(stats, discarded)
                continue

            stats[spec.condition].add(outcome)
            print_summary(stats, discarded)
            plan_pos += 1

        print_summary(stats, discarded, prefix="[done summary]")
    finally:
        keyboard.close()
        rollout.cleanup_external_runtime(runtime)


def main() -> None:
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
