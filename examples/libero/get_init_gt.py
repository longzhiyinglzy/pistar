import argparse
import importlib
import json
import logging
import math
import os
import pathlib
import sys

import numpy as np
from openpi_client import image_tools

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
DEFAULT_SUITE_WHITELIST = ["libero_10", "libero_goal", "libero_spatial", "libero_object"]


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool from: {value}")


def _import_libero_modules():
    try:
        benchmark_mod = importlib.import_module("libero.libero.benchmark")
        libero_mod = importlib.import_module("libero.libero")
        env_mod = importlib.import_module("libero.libero.envs")
        return benchmark_mod, libero_mod.get_libero_path, env_mod.OffScreenRenderEnv
    except ModuleNotFoundError:
        pass

    env_root_raw = os.environ.get("LIBERO_REPO_ROOT", "").strip()
    candidate_roots = [
        # preferred local vendor path in this workspace
        pathlib.Path("/public/home/chenyuyao1/code/pistar/third_party/libero"),
        pathlib.Path(__file__).resolve().parents[2] / "third_party" / "libero",
    ]
    if env_root_raw:
        candidate_roots.insert(1, pathlib.Path(env_root_raw).expanduser())
    for root in candidate_roots:
        if root.exists() and str(root) not in sys.path:
            sys.path.insert(0, str(root))

    benchmark_mod = importlib.import_module("libero.libero.benchmark")
    libero_mod = importlib.import_module("libero.libero")
    env_mod = importlib.import_module("libero.libero.envs")
    return benchmark_mod, libero_mod.get_libero_path, env_mod.OffScreenRenderEnv


benchmark, get_libero_path, OffScreenRenderEnv = _import_libero_modules()


def _quat2axisangle(quat):
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _get_libero_env(task, resolution: int, seed: int):
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env


def _extract_obs_images(obs: dict) -> tuple[np.ndarray, np.ndarray]:
    # rotate 180 degrees to match train/inference preprocessing
    image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]).astype(np.uint8)
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]).astype(np.uint8)
    return image, wrist


def _extract_policy_state8(obs: dict) -> np.ndarray:
    """Extract 8D policy state: 7 joint positions + 1 gripper."""
    joint_keys = ["robot0_joint_pos", "robot0_joint_positions", "joint_pos", "joint_positions"]
    joints = None
    for key in joint_keys:
        if key in obs:
            joints = np.asarray(obs[key], dtype=np.float32).reshape(-1)
            break

    if joints is None:
        # Fallback if joint state is unavailable.
        logging.warning("Joint state key missing; fallback to EEF-derived 7D state.")
        joints = np.concatenate(
            [
                np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
                _quat2axisangle(obs["robot0_eef_quat"]),
            ],
            axis=0,
        )

    if joints.shape[0] >= 7:
        joints7 = joints[:7]
    else:
        joints7 = np.pad(joints, (0, 7 - joints.shape[0]))

    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    gripper = float(np.mean(gripper_qpos)) if gripper_qpos.size > 0 else 0.0
    return np.concatenate([joints7, np.asarray([gripper], dtype=np.float32)], axis=0).astype(np.float32)


def _extract_eef_pose7(obs: dict) -> np.ndarray:
    """Extract absolute EEF Cartesian pose: xyz + axis-angle + gripper."""
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1)
    gripper = float(np.mean(gripper_qpos)) if gripper_qpos.size > 0 else 0.0
    return np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
            _quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray([gripper], dtype=np.float32),
        ),
        axis=0,
    ).astype(np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export initial LIBERO GT for WM rollout")
    parser.add_argument("--task_suite_name", type=str, default="libero_spatial")
    parser.add_argument("--task_suite_names", type=str, nargs="*", default=None)
    parser.add_argument("--all_task_suites", type=str2bool, default=False)
    parser.add_argument("--task_ids", type=int, nargs="*", default=None)
    parser.add_argument("--num_init_gt_per_task", type=int, default=1)
    parser.add_argument("--export_all_init_states", type=str2bool, default=False)
    parser.add_argument("--num_steps_wait", type=int, default=10)
    parser.add_argument("--camera_resolution", type=int, default=256)
    parser.add_argument("--wm_height", type=int, default=192)
    parser.add_argument("--wm_width", type=int, default=320)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--manifest_name", type=str, default="init_gt_manifest.json")
    parser.add_argument("--overwrite", type=str2bool, default=False)
    return parser


def main():
    args = build_parser().parse_args()

    out_dir = pathlib.Path(args.output_dir)
    if args.overwrite and out_dir.exists():
        import shutil

        logging.warning("Removing existing output dir because --overwrite=true: %s", out_dir)
        shutil.rmtree(out_dir)
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()

    if args.all_task_suites:
        # Keep only practical suites for this pipeline and skip libero_90/libero_100.
        suite_names = [name for name in DEFAULT_SUITE_WHITELIST if name in benchmark_dict]
        if not suite_names:
            raise ValueError(
                f"No default suites found in benchmark_dict. "
                f"Expected one of {DEFAULT_SUITE_WHITELIST}, got keys={sorted(list(benchmark_dict.keys()))}"
            )
    elif args.task_suite_names:
        suite_names = list(args.task_suite_names)
    else:
        suite_names = [args.task_suite_name]

    unknown_suites = [name for name in suite_names if name not in benchmark_dict]
    if unknown_suites:
        raise ValueError(f"Unknown task suite names: {unknown_suites}")

    entries = []

    for suite_name in suite_names:
        try:
            task_suite = benchmark_dict[suite_name]()
        except Exception as exc:
            logging.warning("Skip suite %s because initialization failed: %s", suite_name, exc)
            continue
        num_tasks_in_suite = task_suite.n_tasks

        if args.task_ids is None:
            task_ids = list(range(num_tasks_in_suite))
        else:
            task_ids = [task_id for task_id in args.task_ids if 0 <= task_id < num_tasks_in_suite]
            invalid_task_ids = [task_id for task_id in args.task_ids if task_id < 0 or task_id >= num_tasks_in_suite]
            if invalid_task_ids:
                logging.warning("[%s] Ignoring out-of-range task ids: %s", suite_name, invalid_task_ids)
            if not task_ids:
                logging.warning("[%s] No valid task ids left after filtering, skip suite", suite_name)
                continue

        for task_id in task_ids:
            task = task_suite.get_task(task_id)
            task_description = str(task.language)
            env = _get_libero_env(task, args.camera_resolution, args.seed)
            initial_states = task_suite.get_task_init_states(task_id)

            if args.export_all_init_states or args.num_init_gt_per_task <= 0:
                num_export = len(initial_states)
            else:
                num_export = min(args.num_init_gt_per_task, len(initial_states))

            for init_state_idx in range(num_export):
                env.reset()
                obs = env.set_init_state(initial_states[init_state_idx])
                for _ in range(args.num_steps_wait):
                    obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)

                base_img, wrist_img = _extract_obs_images(obs)
                state8 = _extract_policy_state8(obs)
                pose7 = _extract_eef_pose7(obs)

                suite_tag = suite_name.replace("/", "_")
                episode_file = f"{suite_tag}_task{task_id:03d}_init{init_state_idx:04d}.npz"
                episode_path = episodes_dir / episode_file
                np.savez_compressed(
                    episode_path,
                    # Keep initial GT aligned with LIBERO raw camera output semantics:
                    # flipped but still original camera resolution (default 256x256).
                    base_img=base_img,
                    wrist_img=wrist_img,
                    state8=state8,
                    pose7=pose7,
                    task_description=np.asarray(task_description),
                    task_id=np.asarray(task_id, dtype=np.int32),
                    init_state_idx=np.asarray(init_state_idx, dtype=np.int32),
                    task_suite_name=np.asarray(suite_name),
                )

                entries.append(
                    {
                        "task_suite_name": suite_name,
                        "task_id": int(task_id),
                        "init_state_idx": int(init_state_idx),
                        "task_description": task_description,
                        "npz": str(pathlib.Path("episodes") / episode_file),
                    }
                )
                logging.info("Saved init GT: %s", episode_path)

            if hasattr(env, "close"):
                try:
                    env.close()
                except Exception:
                    pass

    manifest = {
        "format_version": 1,
        "task_suite_names": suite_names,
        "num_steps_wait": int(args.num_steps_wait),
        "camera_resolution": int(args.camera_resolution),
        "seed": int(args.seed),
        "entries": entries,
    }

    manifest_path = out_dir / args.manifest_name
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Saved manifest: {manifest_path}")
    print(f"Total episodes: {len(entries)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
