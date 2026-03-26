"""Convert LIBERO RLDS datasets to canonical LeRobot schema for WM pipeline.

Canonical LeRobot schema (single source of truth across this project):
    - image: main RGB image, shape (256, 256, 3)
    - wrist_image: wrist RGB image, shape (256, 256, 3)
    - state: policy state, shape (8,) = abs_pose6 + gripper2
    - joint_state: joint positions, shape (7,)
    - actions: policy delta action, shape (7,) = delta pose6 + gripper command
    - abs_pose: canonical WM/Dynamics absolute pose, shape (7,)
    - task: language instruction string

Default behavior requires RLDS `abs_pose` to exist and uses it directly.
Optional fallback (`--allow_abs_pose_fallback`) can construct abs_pose from
`state[:6] + actions[6]`, but is disabled by default.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def _to_str(x) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _to_state8(step: dict) -> np.ndarray:
    # Canonical RLDS field: observation.state -> policy state (8D).
    state = np.asarray(step["observation"]["state"], dtype=np.float32).reshape(-1)
    if state.shape[0] < 8:
        state = np.pad(state, (0, 8 - state.shape[0]))
    return state[:8]


def _to_joint7(step: dict) -> np.ndarray:
    # Canonical RLDS field: observation.joint_state -> joint_state (7D).
    joint = np.asarray(step["observation"]["joint_state"], dtype=np.float32).reshape(-1)
    if joint.shape[0] < 7:
        joint = np.pad(joint, (0, 7 - joint.shape[0]))
    return joint[:7]


def _to_action7(step: dict) -> np.ndarray:
    # Canonical RLDS field: action -> actions (7D delta EEF action).
    action = np.asarray(step["action"], dtype=np.float32).reshape(-1)
    if action.shape[0] < 7:
        action = np.pad(action, (0, 7 - action.shape[0]))
    return action[:7]


def _to_abs_pose7(step: dict, *, allow_abs_pose_fallback: bool) -> np.ndarray:
    if "abs_pose" in step:
        abs_pose = np.asarray(step["abs_pose"], dtype=np.float32).reshape(-1)
        if abs_pose.shape[0] < 7:
            abs_pose = np.pad(abs_pose, (0, 7 - abs_pose.shape[0]))
        return abs_pose[:7]

    if not allow_abs_pose_fallback:
        raise KeyError(
            "RLDS step is missing required key `abs_pose`. "
            "Enable --allow_abs_pose_fallback to use fallback abs_pose=[state[:6], action[6]]."
        )

    state8 = _to_state8(step)
    action7 = _to_action7(step)
    return np.concatenate([state8[:6], np.asarray([float(action7[6])], dtype=np.float32)], axis=0).astype(np.float32)


def main(
    data_dir: str,
    repo_id: str = "ybpy/libero_pistar_wm",
    raw_dataset_names: tuple[str, ...] = (
        "libero_10",
        "libero_goal",
        "libero_object",
        "libero_spatial",
        "libero_90",
    ),
    output_root: str | None = None,
    overwrite: bool = False,
    allow_abs_pose_fallback: bool = False,
):
    output_path = (Path(output_root) / repo_id) if output_root else (HF_LEROBOT_HOME / repo_id)

    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}. Use --overwrite true to recreate.")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output_path,
        robot_type="panda",
        fps=15,
        features={
            "image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "joint_state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
            "abs_pose": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["abs_pose"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    episode_count = 0
    for raw_name in raw_dataset_names:
        print(f"[INFO] Loading RLDS split: {raw_name}")
        raw_dataset = tfds.load(raw_name, data_dir=data_dir, split="train")

        for episode in raw_dataset:
            for step in episode["steps"].as_numpy_iterator():
                state8 = _to_state8(step)
                action7 = _to_action7(step)
                task = _to_str(step["language_instruction"])
                dataset.add_frame(
                    {
                        "image": step["observation"]["image"],
                        "wrist_image": step["observation"]["wrist_image"],
                        "state": state8,
                        "joint_state": _to_joint7(step),
                        "actions": action7,
                        "abs_pose": _to_abs_pose7(step, allow_abs_pose_fallback=allow_abs_pose_fallback),
                        "task": task,
                    }
                )
            dataset.save_episode()
            episode_count += 1
            if episode_count % 100 == 0:
                print(f"[INFO] Saved {episode_count} episodes")

    print(f"[DONE] LeRobot dataset saved at: {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
