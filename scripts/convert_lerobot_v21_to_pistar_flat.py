#!/usr/bin/env python3
"""Convert a LeRobot v2.1 video dataset to the flat PiStar dataset schema.

The source dataset is never modified. The output dataset stores decoded camera
frames as image bytes and uses these frame keys:

    image, wrist_image, side_image, state, actions, intervention,
    value_label, reward, reward_label, adv_ind, timestamp, frame_index,
    episode_index, index, task_index
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm


CAMERA_TO_OUTPUT = {
    "observation.images.cam_head": "image",
    "observation.images.cam_wrist": "wrist_image",
    "observation.images.cam_side": "side_image",
}

JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "gripper"]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _select_indices(num_rows: int, source_fps: float, target_fps: float) -> np.ndarray:
    if target_fps <= 0 or source_fps <= 0:
        raise ValueError("source_fps and target_fps must be positive")
    if target_fps > source_fps:
        raise ValueError(f"target_fps={target_fps} is higher than source_fps={source_fps}")

    step = source_fps / target_fps
    indices = np.floor(np.arange(0, num_rows, step)).astype(np.int64)
    indices = np.unique(np.clip(indices, 0, max(num_rows - 1, 0)))
    if len(indices) == 0 and num_rows > 0:
        return np.array([0], dtype=np.int64)
    return indices


def _video_path(source: Path, chunk_idx: int, video_key: str, episode_idx: int) -> Path:
    return source / "videos" / f"chunk-{chunk_idx:03d}" / video_key / f"episode_{episode_idx:06d}.mp4"


def _encode_frame(frame_rgb: np.ndarray, *, image_format: str, jpeg_quality: int) -> bytes:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    if image_format == "jpg":
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    elif image_format == "png":
        ok, encoded = cv2.imencode(".png", frame_bgr)
    else:
        raise ValueError(f"Unsupported image_format: {image_format}")
    if not ok:
        raise RuntimeError("Failed to encode image frame")
    return encoded.tobytes()


def _read_selected_video_frames(
    video_path: Path,
    selected_indices: np.ndarray,
    *,
    image_format: str,
    jpeg_quality: int,
) -> tuple[list[dict[str, bytes | None]], tuple[int, int]]:
    if not video_path.exists():
        raise FileNotFoundError(f"Missing video: {video_path}")
    return _read_selected_video_frames_pyav(
        video_path,
        selected_indices,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
    )


def _read_selected_video_frames_pyav(
    video_path: Path,
    selected_indices: np.ndarray,
    *,
    image_format: str,
    jpeg_quality: int,
) -> tuple[list[dict[str, bytes | None]], tuple[int, int]]:
    import imageio.v3 as iio

    selected = set(int(i) for i in selected_indices.tolist())
    max_index = int(selected_indices[-1]) if len(selected_indices) else -1
    frames: dict[int, dict[str, bytes | None]] = {}
    frame_size: tuple[int, int] | None = None

    for frame_idx, frame_rgb in enumerate(iio.imiter(video_path, plugin="pyav")):
        if frame_idx > max_index:
            break
        if frame_idx not in selected:
            continue
        frame_rgb = np.asarray(frame_rgb)
        if frame_size is None:
            frame_size = (int(frame_rgb.shape[0]), int(frame_rgb.shape[1]))
        frames[frame_idx] = {
            "bytes": _encode_frame(frame_rgb, image_format=image_format, jpeg_quality=jpeg_quality),
            "path": None,
        }

    missing = [int(i) for i in selected_indices.tolist() if int(i) not in frames]
    if missing:
        raise RuntimeError(f"{video_path} is missing selected frames, first missing: {missing[:10]}")
    if frame_size is None:
        raise RuntimeError(f"No frames decoded from: {video_path}")

    return [frames[int(i)] for i in selected_indices.tolist()], frame_size


def _feature(dtype: str, shape: list[int], names: Any = None, description: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"dtype": dtype, "shape": shape, "names": names}
    if description is not None:
        result["description"] = description
    return result


def _image_feature(height: int, width: int) -> dict[str, Any]:
    return {
        "dtype": "image",
        "shape": [3, height, width],
        "names": ["channels", "height", "width"],
    }


def _make_table(rows: dict[str, Any]) -> pa.Table:
    image_type = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
    vector7 = pa.list_(pa.float32(), 7)

    return pa.table(
        {
            "image": pa.array(rows["image"], type=image_type),
            "wrist_image": pa.array(rows["wrist_image"], type=image_type),
            "side_image": pa.array(rows["side_image"], type=image_type),
            "state": pa.array(rows["state"], type=vector7),
            "actions": pa.array(rows["actions"], type=vector7),
            "intervention": pa.array(rows["intervention"], type=pa.int64()),
            "value_label": pa.array(rows["value_label"], type=pa.float32()),
            "reward": pa.array(rows["reward"], type=pa.float32()),
            "reward_label": pa.array(rows["reward_label"], type=pa.float32()),
            "adv_ind": pa.array(rows["adv_ind"], type=pa.string()),
            "timestamp": pa.array(rows["timestamp"], type=pa.float32()),
            "frame_index": pa.array(rows["frame_index"], type=pa.int64()),
            "episode_index": pa.array(rows["episode_index"], type=pa.int64()),
            "index": pa.array(rows["index"], type=pa.int64()),
            "task_index": pa.array(rows["task_index"], type=pa.int64()),
        }
    )


def _convert_episode(
    source: Path,
    output: Path,
    *,
    episode_idx: int,
    new_episode_idx: int,
    chunk_idx: int,
    out_chunk_idx: int,
    source_fps: float,
    target_fps: float,
    image_format: str,
    jpeg_quality: int,
    global_frame_offset: int,
    chunks_size: int,
    adv_ind: str,
    intervention: int,
) -> tuple[int, tuple[int, int]]:
    parquet_path = source / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_idx:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Missing parquet: {parquet_path}")

    table = pq.read_table(parquet_path)
    data = table.to_pydict()
    num_rows = table.num_rows
    selected = _select_indices(num_rows, source_fps=source_fps, target_fps=target_fps)
    out_len = int(len(selected))

    camera_frames: dict[str, list[dict[str, bytes | None]]] = {}
    image_size: tuple[int, int] | None = None
    for video_key, output_key in CAMERA_TO_OUTPUT.items():
        frames, frame_size = _read_selected_video_frames(
            _video_path(source, chunk_idx, video_key, episode_idx),
            selected,
            image_format=image_format,
            jpeg_quality=jpeg_quality,
        )
        camera_frames[output_key] = frames
        if image_size is None:
            image_size = frame_size

    state_rows = [data["observation.state"][int(i)] for i in selected]
    action_rows = [data["action"][int(i)] for i in selected]
    task_rows = [int(data["task_index"][int(i)]) for i in selected]
    timestamps = [float(pos) / float(target_fps) for pos in range(out_len)]

    t = np.arange(out_len, dtype=np.float32)
    value_labels = (-(out_len - t) / float(out_len)).astype(np.float32)
    value_labels[-1] = 0.0
    rewards = np.zeros(out_len, dtype=np.float32)
    rewards[-1] = 1.0
    reward_labels = np.full(out_len, -1.0 / float(out_len), dtype=np.float32)
    reward_labels[-1] = 0.0

    rows = {
        "image": camera_frames["image"],
        "wrist_image": camera_frames["wrist_image"],
        "side_image": camera_frames["side_image"],
        "state": state_rows,
        "actions": action_rows,
        "intervention": [int(intervention)] * out_len,
        "value_label": value_labels.tolist(),
        "reward": rewards.tolist(),
        "reward_label": reward_labels.tolist(),
        "adv_ind": [adv_ind] * out_len,
        "timestamp": timestamps,
        "frame_index": list(range(out_len)),
        "episode_index": [new_episode_idx] * out_len,
        "index": list(range(global_frame_offset, global_frame_offset + out_len)),
        "task_index": task_rows,
    }

    out_table = _make_table(rows)
    out_file = output / "data" / f"chunk-{out_chunk_idx:03d}" / f"episode_{new_episode_idx:06d}.parquet"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, out_file)

    if image_size is None:
        raise RuntimeError(f"Could not infer image size for episode {episode_idx}")
    return out_len, image_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LeRobot v2.1 Piper videos to flat PiStar LeRobot data.")
    parser.add_argument("--source", required=True, help="Source LeRobot v2.1 dataset root.")
    parser.add_argument("--output", required=True, help="Output dataset root. Source is not modified.")
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--source-fps", type=float, default=None, help="Defaults to meta/info.json fps.")
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--max-episodes", type=int, default=None, help="Optional debug cap.")
    parser.add_argument("--image-format", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--adv-ind", default="positive")
    parser.add_argument("--intervention", type=int, default=1, help="Demo frames should usually be 1.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists: {output}. Use --overwrite.")
        shutil.rmtree(output)

    info = _load_json(source / "meta" / "info.json")
    source_fps = float(info.get("fps", 30) if args.source_fps is None else args.source_fps)
    target_fps = float(args.target_fps)
    chunks_size = int(args.chunks_size)
    source_chunks_size = int(info.get("chunks_size", 1000))

    tasks_rows = _load_jsonl(source / "meta" / "tasks.jsonl")
    episodes_rows = _load_jsonl(source / "meta" / "episodes.jsonl")
    episodes_rows = sorted(episodes_rows, key=lambda row: int(row["episode_index"]))
    if args.max_episodes is not None:
        episodes_rows = episodes_rows[: int(args.max_episodes)]

    output.mkdir(parents=True, exist_ok=True)
    (output / "meta").mkdir(parents=True, exist_ok=True)

    out_episodes: list[dict[str, Any]] = []
    out_episode_stats: list[dict[str, Any]] = []
    total_frames = 0
    image_size: tuple[int, int] | None = None

    progress = tqdm.tqdm(episodes_rows, desc="Converting episodes", unit="ep")
    for new_episode_idx, episode_row in enumerate(progress):
        old_episode_idx = int(episode_row["episode_index"])
        chunk_idx = old_episode_idx // source_chunks_size
        out_chunk_idx = new_episode_idx // chunks_size
        length, ep_image_size = _convert_episode(
            source,
            output,
            episode_idx=old_episode_idx,
            new_episode_idx=new_episode_idx,
            chunk_idx=chunk_idx,
            out_chunk_idx=out_chunk_idx,
            source_fps=source_fps,
            target_fps=target_fps,
            image_format=args.image_format,
            jpeg_quality=int(args.jpeg_quality),
            global_frame_offset=total_frames,
            chunks_size=chunks_size,
            adv_ind=args.adv_ind,
            intervention=int(args.intervention),
        )
        image_size = image_size or ep_image_size
        tasks = list(episode_row.get("tasks", []))
        out_episodes.append({"episode_index": new_episode_idx, "tasks": tasks, "length": length})
        out_episode_stats.append({"episode_index": new_episode_idx, "stats": {}})
        total_frames += length
        progress.set_postfix_str(f"old_ep={old_episode_idx} frames={length}")

    if image_size is None:
        raise RuntimeError("No episodes converted")
    height, width = image_size
    out_info = {
        "codebase_version": info.get("codebase_version", "v2.1"),
        "robot_type": info.get("robot_type", "piper"),
        "total_episodes": len(out_episodes),
        "total_frames": int(total_frames),
        "total_tasks": len(tasks_rows),
        "total_videos": 0,
        "total_chunks": int((len(out_episodes) + chunks_size - 1) // chunks_size),
        "chunks_size": chunks_size,
        "fps": int(target_fps) if float(target_fps).is_integer() else target_fps,
        "splits": {"train": f"0:{len(out_episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": None,
        "features": {
            "image": _image_feature(height, width),
            "wrist_image": _image_feature(height, width),
            "side_image": _image_feature(height, width),
            "state": _feature("float32", [7], [JOINT_NAMES]),
            "actions": _feature("float32", [7], [JOINT_NAMES]),
            "intervention": _feature("int64", [1], ["intervention_flag"]),
            "value_label": _feature("float32", [1], ["value_label"]),
            "reward": _feature("float32", [1], ["reward"]),
            "reward_label": _feature("float32", [1], ["reward_label"]),
            "adv_ind": _feature("string", [1], ["adv_ind"]),
            "timestamp": _feature("float32", [1]),
            "frame_index": _feature("int64", [1]),
            "episode_index": _feature("int64", [1]),
            "index": _feature("int64", [1]),
            "task_index": _feature("int64", [1]),
        },
    }

    _write_json(output / "meta" / "info.json", out_info)
    _write_jsonl(output / "meta" / "tasks.jsonl", tasks_rows)
    _write_jsonl(output / "meta" / "episodes.jsonl", out_episodes)
    _write_jsonl(output / "meta" / "episodes_stats.jsonl", out_episode_stats)

    print(f"Done: {output}")
    print(f"Episodes: {len(out_episodes)}, frames: {total_frames}, fps: {target_fps}")
    print("Keys: image, wrist_image, side_image, state, actions, intervention, value_label, reward, reward_label, adv_ind")


if __name__ == "__main__":
    main()
