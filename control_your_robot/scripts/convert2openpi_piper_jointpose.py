import dataclasses
from pathlib import Path
import shutil
from typing import Literal
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import torch
import tqdm
import tyro
import json
import os
import fnmatch


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()
TARGET_CAMERAS = ["cam_head", "cam_side", "cam_wrist"]


def create_empty_dataset(
    repo_id: str,
    robot_type: str,
    cameras: list[str],
    mode: Literal["video", "image"] = "image",
    *,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> LeRobotDataset:
    motors = [
        "joint_1",
        "joint_2",
        "joint_3",
        "joint_4",
        "joint_5",
        "joint_6",
        "gripper",
    ]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": [motors],
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": [motors],
        },
    }

    for cam in cameras:
        features[f"observation.images.{cam}"] = {
            "dtype": mode,
            "shape": (3, 480, 640),
            "names": ["channels", "height", "width"],
        }

    if Path(HF_LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(HF_LEROBOT_HOME / repo_id)

    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def _read_required_first(ep: h5py.File, candidates: list[str], name: str) -> np.ndarray:
    for key in candidates:
        if key in ep:
            return ep[key][:]
    raise KeyError(f"{name} not found, candidates={candidates}")


def _read_optional_first(ep: h5py.File, candidates: list[str]) -> np.ndarray | None:
    for key in candidates:
        if key in ep:
            return ep[key][:]
    return None


def _load_camera_frames(ep: h5py.File, src_path: str, ep_path: str) -> np.ndarray:
    """
    Support both camera storage formats:
    1) uncompressed frames, shape=(T, H, W, C)
    2) JPEG-compressed bytes, shape=(T,)
    """
    dset = ep[src_path]

    if dset.ndim == 4:
        return dset[:]

    import cv2

    imgs_array: list[np.ndarray] = []
    for i, data in enumerate(dset):
        if isinstance(data, np.ndarray):
            encoded = data.tobytes()
        else:
            encoded = bytes(data)

        encoded = encoded.rstrip(b"\0")
        if len(encoded) == 0:
            raise ValueError(f"{ep_path} {src_path}[{i}] is empty after zero-padding strip")

        img = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"{ep_path} failed to decode JPEG at {src_path}[{i}]")

        imgs_array.append(img)

    return np.array(imgs_array)


def _detect_camera_sources(
    ep: h5py.File,
    *,
    emit_warning: bool = False,
    allow_fallback: bool = False,
) -> dict[str, str]:
    """
    输出统一为三视角：
      cam_head, cam_side, cam_wrist
    默认严格模式：必须有真实 side+wrist。
    可选兼容策略(allow_fallback=True)：
      - 只有 side 没有 wrist: wrist 复用 side
      - 只有 wrist 没有 side: side 复用 wrist
    """
    if "slave_cam_head/color" not in ep:
        raise KeyError("Missing required camera stream: slave_cam_head/color")

    side_src = "slave_cam_side/color" if "slave_cam_side/color" in ep else None
    wrist_src = "slave_cam_wrist/color" if "slave_cam_wrist/color" in ep else None

    if side_src is None and wrist_src is None:
        raise KeyError("Missing both slave_cam_side/color and slave_cam_wrist/color")

    if (side_src is None or wrist_src is None) and not allow_fallback:
        raise KeyError(
            "Require both slave_cam_side/color and slave_cam_wrist/color for true 3-view export. "
            "This usually means wrist camera was not recorded in collection stage. "
            "If you accept duplicated fallback views, set --allow-camera-fallback true."
        )

    if side_src is None:
        side_src = wrist_src
        if emit_warning:
            print("[warn] slave_cam_side/color missing, use slave_cam_wrist/color for cam_side", flush=True)

    if wrist_src is None:
        wrist_src = side_src
        if emit_warning:
            print("[warn] slave_cam_wrist/color missing, use slave_cam_side/color for cam_wrist", flush=True)

    return {
        "cam_head": "slave_cam_head/color",
        "cam_side": side_src,
        "cam_wrist": wrist_src,
    }


def detect_cameras_from_first_episode(
    hdf5_files: list[str],
    *,
    allow_camera_fallback: bool = False,
) -> list[str]:
    with h5py.File(hdf5_files[0], "r") as ep:
        _detect_camera_sources(ep, emit_warning=True, allow_fallback=allow_camera_fallback)
    return TARGET_CAMERAS.copy()


def load_raw_episode_data(
    ep_path: str,
    expected_cameras: list[str],
    action_mode: Literal["next_state", "same_state"] = "next_state",
    allow_camera_fallback: bool = False,
):
    with h5py.File(ep_path, "r") as ep:
        state_joint = _read_required_first(
            ep,
            ["aligned_state/joint", "slave_left_arm/joint"],
            name="state_joint",
        )
        state_gripper = _read_required_first(
            ep,
            ["aligned_state/gripper", "slave_left_arm/gripper"],
            name="state_gripper",
        )

        # 先对齐 state 的 joint / gripper，避免后续 concatenate 维度不一致
        state_pair_len = min(len(state_joint), len(state_gripper))
        if len(state_joint) != len(state_gripper):
            print(
                (
                    f"[warn] {ep_path} state length mismatch before concat: "
                    f"joint={len(state_joint)}, gripper={len(state_gripper)}, "
                    f"trim to {state_pair_len}"
                ),
                flush=True,
            )
            state_joint = state_joint[:state_pair_len]
            state_gripper = state_gripper[:state_pair_len]

        state = np.concatenate(
            [state_joint, state_gripper[:, None]], axis=1
        ).astype(np.float32)

        if action_mode == "next_state":
            action_joint = np.concatenate([state_joint[1:], state_joint[-1:]], axis=0)
            action_gripper = np.concatenate([state_gripper[1:], state_gripper[-1:]], axis=0)
        elif action_mode == "same_state":
            action_joint = state_joint
            action_gripper = state_gripper
        else:
            raise ValueError(f"Unsupported action_mode: {action_mode}")

        action = np.concatenate(
            [action_joint, action_gripper[:, None]], axis=1
        ).astype(np.float32)

        camera_sources = _detect_camera_sources(
            ep,
            emit_warning=False,
            allow_fallback=allow_camera_fallback,
        )
        imgs_per_cam: dict[str, np.ndarray] = {}
        cam_ts: dict[str, np.ndarray] = {}
        for out_name, src_path in camera_sources.items():
            imgs_per_cam[out_name] = _load_camera_frames(ep, src_path, ep_path)
            ts_path = src_path.replace("/color", "/timestamp")
            if ts_path in ep:
                cam_ts[f"{out_name}_ts"] = ep[ts_path][:]

        missing = [cam for cam in expected_cameras if cam not in imgs_per_cam]
        if missing:
            raise KeyError(f"{ep_path} missing cameras {missing}, available={list(imgs_per_cam.keys())}")

        # 用于对齐截断的时间戳（如果存在）
        state_ts = _read_optional_first(ep, ["aligned_state/timestamp", "sync_info/t_state", "slave_left_arm/timestamp"])
        action_ts = _read_optional_first(ep, ["action_left_arm/timestamp", "sync_info/t_action"])
        sample_ts = _read_optional_first(ep, ["sync_info/t_sample"])

    lengths: dict[str, int] = {
        "state": len(state),
        "action": len(action),
    }
    for cam in expected_cameras:
        lengths[cam] = len(imgs_per_cam[cam])
    if state_ts is not None:
        lengths["state_ts"] = len(state_ts)
    if action_ts is not None:
        lengths["action_ts"] = len(action_ts)
    if sample_ts is not None:
        lengths["sample_ts"] = len(sample_ts)
    for k, arr in cam_ts.items():
        lengths[k] = len(arr)

    min_len = min(lengths.values())
    if len(set(lengths.values())) != 1:
        print(f"[warn] {ep_path} length mismatch: {lengths}, trim to {min_len}", flush=True)

    state = state[:min_len]
    action = action[:min_len]
    imgs_per_cam = {cam: imgs_per_cam[cam][:min_len] for cam in expected_cameras}

    return torch.from_numpy(state), torch.from_numpy(action), imgs_per_cam


def populate_dataset(
    dataset: LeRobotDataset,
    hdf5_files: list[str],
    cameras: list[str],
    task: str,
    *,
    action_mode: Literal["next_state", "same_state"] = "next_state",
    allow_camera_fallback: bool = False,
    episodes: list[int] | None = None,
) -> LeRobotDataset:
    if episodes is None:
        episodes = range(len(hdf5_files))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        state, action, imgs_per_cam = load_raw_episode_data(
            ep_path,
            expected_cameras=cameras,
            action_mode=action_mode,
            allow_camera_fallback=allow_camera_fallback,
        )
        num_frames = state.shape[0]

        dir_path = os.path.dirname(ep_path)
        json_path = f"{dir_path}/instructions.json"

        if os.path.exists(json_path):
            with open(json_path, "r") as f_instr:
                instruction_dict = json.load(f_instr)
                instructions = instruction_dict.get("instructions", [task])
                instruction = np.random.choice(instructions)
        else:
            instruction = task

        for i in range(num_frames):
            frame = {
                "observation.state": state[i],
                "action": action[i],
                "task": instruction,
            }
            for cam in cameras:
                frame[f"observation.images.{cam}"] = imgs_per_cam[cam][i]
            dataset.add_frame(frame)

        dataset.save_episode()

    return dataset


def port_piper_jointpose(
    raw_dir: Path,
    repo_id: str,
    task: str = "teleoperate piper arm",
    *,
    action_mode: Literal["next_state", "same_state"] = "next_state",
    allow_camera_fallback: bool = False,
    episodes: list[int] | None = None,
    push_to_hub: bool = False,
    mode: Literal["video", "image"] = "image",
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
):
    if not raw_dir.exists():
        raise ValueError(f"raw_dir does not exist: {raw_dir}")

    hdf5_files: list[str] = []
    for root, _, files in os.walk(raw_dir):
        for filename in fnmatch.filter(files, "*.hdf5"):
            hdf5_files.append(os.path.join(root, filename))

    hdf5_files = sorted(hdf5_files)
    if len(hdf5_files) == 0:
        raise ValueError(f"No .hdf5 files found under {raw_dir}")

    cameras = detect_cameras_from_first_episode(
        hdf5_files,
        allow_camera_fallback=allow_camera_fallback,
    )
    print(f"[info] detected cameras: {cameras}", flush=True)
    print(f"[info] action_mode={action_mode}", flush=True)
    print(f"[info] allow_camera_fallback={allow_camera_fallback}", flush=True)

    dataset = create_empty_dataset(
        repo_id=repo_id,
        robot_type="piper",
        cameras=cameras,
        mode=mode,
        dataset_config=dataset_config,
    )

    populate_dataset(
        dataset,
        hdf5_files,
        cameras=cameras,
        task=task,
        action_mode=action_mode,
        allow_camera_fallback=allow_camera_fallback,
        episodes=episodes,
    )

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(port_piper_jointpose)
