"""Create minimal canonical meta info and stats for WM / Dynamics.

Input:
    <dataset_output_path>/annotation/{train,val}/*.json

Output:
    <meta_output_root>/<dataset_name>/train_sample.json
    <meta_output_root>/<dataset_name>/val_sample.json
    <meta_output_root>/<dataset_name>/stat.json

Stat naming convention:
    - wm_state_* : computed from abs_pose (7D), used by world model conditioning
    - dyn_action_* : computed from action_raw[:6], used by dynamics action normalization
    - dyn_pose_* : computed from abs_pose_raw[:6], used by dynamics pose normalization
"""

from __future__ import annotations

import json
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm


def load_and_process_ann_file(data_root: Path, ann_file: str, start_interval: int = 1, num_frames: int = 5):
    samples: list[dict] = []
    stat_data = {
        "wm_state": [],
        "dyn_action": [],
        "dyn_pose": [],
    }
    try:
        with open(data_root / ann_file, "r") as f:
            ann = json.load(f)
    except Exception:
        print(f"[WARN] skip {ann_file}")
        return samples, stat_data

    # Samples are indexed on the downsampled 5Hz timeline.
    # Keep history-insufficient samples (history will clip to oldest),
    # but strictly filter out samples with insufficient future window.
    # Valid t must satisfy: t + (num_frames - 1) < n_frames.
    n_frames = int(ann.get("video_length", 0))
    max_start = n_frames - int(num_frames)
    if max_start >= 0:
        candidate_ts = range(0, max_start + 1, max(1, int(start_interval)))
    else:
        candidate_ts = []

    for start_frame in candidate_ts:
        sample = {
            "episode_id": int(ann["episode_id"]),
            "frame_ids": [int(start_frame)],
        }
        samples.append(sample)

    abs_pose = np.asarray(ann["abs_pose"], dtype=np.float32).reshape(-1, 7)
    abs_pose_raw = np.asarray(ann["abs_pose_raw"], dtype=np.float32).reshape(-1, 7)
    action_raw = np.asarray(ann["action_raw"], dtype=np.float32).reshape(-1, 7)

    action_raw = action_raw.reshape(-1, 7)

    if abs_pose.shape[0] > 0:
        stat_data["wm_state"].append(abs_pose)
    if action_raw.shape[0] > 0:
        stat_data["dyn_action"].append(action_raw[:, :6])
    if abs_pose_raw.shape[0] > 0:
        stat_data["dyn_pose"].append(abs_pose_raw[:, :6])
    return samples, stat_data


def init_anns(dataset_root: Path, ann_dir: str):
    final_path = dataset_root / ann_dir
    return [os.path.join(ann_dir, f) for f in os.listdir(final_path) if f.endswith(".json")]


def init_sequences(data_root: Path, ann_files, start_interval: int, num_frames: int):
    samples = []
    wm_states = []
    dyn_actions = []
    dyn_poses = []

    with ThreadPoolExecutor(32) as executor:
        futures = {
            executor.submit(
                load_and_process_ann_file,
                data_root,
                ann_file,
                start_interval,
                num_frames,
            ): ann_file
            for ann_file in ann_files
        }
        for future in tqdm(as_completed(futures), total=len(ann_files), desc="Parsing annotations"):
            ann_samples, stat_data = future.result()
            samples.extend(ann_samples)
            wm_states.extend(stat_data["wm_state"])
            dyn_actions.extend(stat_data["dyn_action"])
            dyn_poses.extend(stat_data["dyn_pose"])

        return samples, wm_states, dyn_actions, dyn_poses


def _percentile_stats(arr: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> tuple[np.ndarray, np.ndarray]:
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for stats, got shape={arr.shape}")
    return np.percentile(arr, q_low, axis=0), np.percentile(arr, q_high, axis=0)


def main(
    dataset_output_path: str,
    dataset_name: str = "libero_wm",
    meta_output_root: str = "dataset_meta_info",
    start_interval: int = 1,
    num_frames: int = 5,
):
    data_root = Path(dataset_output_path)
    out_root = Path(meta_output_root) / dataset_name

    if out_root.exists():
        print(f"[INFO] overwriting meta output directory: {out_root.resolve()}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    wm_state_all = []
    dyn_action_all = []
    dyn_pose_all = []

    for data_type in ["val", "train"]:
        ann_dir = f"annotation/{data_type}"
        ann_files = init_anns(data_root, ann_dir)
        samples, wm_states, dyn_actions, dyn_poses = init_sequences(
            data_root, ann_files, start_interval, num_frames
        )
        print(f"{data_root} {data_type} samples: {len(samples)}")

        wm_state_all.extend(wm_states)
        dyn_action_all.extend(dyn_actions)
        dyn_pose_all.extend(dyn_poses)

        random.shuffle(samples)

        with open(out_root / f"{data_type}_sample.json", "w") as f:
            json.dump(samples, f, indent=4)

    if len(wm_state_all) == 0:
        raise RuntimeError("No abs_pose found. Cannot compute WM stats.")
    if len(dyn_action_all) == 0 or len(dyn_pose_all) == 0:
        raise RuntimeError("No dynamics arrays found. Cannot compute dynamics stats.")

    wm_state = np.concatenate(wm_state_all, axis=0).astype(np.float32)
    dyn_action = np.concatenate(dyn_action_all, axis=0).astype(np.float32)
    dyn_pose = np.concatenate(dyn_pose_all, axis=0).astype(np.float32)

    wm_state_01, wm_state_99 = _percentile_stats(wm_state)
    dyn_action_01, dyn_action_99 = _percentile_stats(dyn_action)
    dyn_pose_01, dyn_pose_99 = _percentile_stats(dyn_pose)

    stat = {
        "wm_state_01": wm_state_01.tolist(),
        "wm_state_99": wm_state_99.tolist(),
        "dyn_action_01": dyn_action_01.tolist(),
        "dyn_action_99": dyn_action_99.tolist(),
        "dyn_pose_01": dyn_pose_01.tolist(),
        "dyn_pose_99": dyn_pose_99.tolist(),
    }

    with open(out_root / "stat.json", "w") as f:
        json.dump(stat, f, indent=2)

    print(f"[DONE] meta and stats written to {out_root}")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--dataset_output_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="libero_wm")
    parser.add_argument("--meta_output_root", type=str, default="dataset_meta_info")
    parser.add_argument("--start_interval", type=int, default=1)
    parser.add_argument("--num_frames", type=int, default=5)
    args = parser.parse_args()

    main(
        dataset_output_path=args.dataset_output_path,
        dataset_name=args.dataset_name,
        meta_output_root=args.meta_output_root,
        start_interval=args.start_interval,
        num_frames=args.num_frames,
    )
