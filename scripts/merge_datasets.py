#!/usr/bin/env python3
"""Merge multiple already-aligned LeRobot datasets into one dataset.

中文说明:
- 这是“纯合并脚本”。
- 前置假设:
  - 真机转换脚本 / 仿真转换脚本已经把数据统一成同一套 LeRobot 字段。
  - 本脚本不负责补字段、不负责修字段、不负责区分真机还是仿真。
- 本脚本只做合并，按下面这些键输出 parquet:
  - `image`
  - `wrist_image`
  - `side_image`（可选；任一源数据集包含时保留）
  - `state`
  - `actions`
  - `intervention`
  - `value_label`
  - `reward`
  - `reward_label`
  - `adv_ind`
  - `timestamp`
  - `frame_index`
  - `episode_index`
  - `index`
  - `task_index`

不会做的事情:
- 不补默认 `adv_ind`
- 不补默认 `intervention`
- 不补默认 `reward`
- 不补默认 `reward_label`
- 不补默认 `value_label`
- 不缩放图像
- 不转换 CHW/HWC
- 不推断 demo / rollout

处理方式:
- 直接按 parquet + meta 重写输出数据集
- 保留每一帧的必需键；如果源数据集包含 `side_image`，也会保留
- 重新生成全局一致的:
  - `task_index`
  - `episode_index`
  - `frame_index`
  - `index`
- 默认自动估算当前机器可用 CPU 线程数，并并行读取 / 预处理 episode，
  以尽量提高合并速度；也可以通过 `--num-workers` 手动指定。

数据合并示意:
    +------------------------------------------------------------------+
    | Dataset A                                                        |
    | keys: image, wrist_image, optional side_image, state, actions,   |
    |       value_label, reward, reward_label, adv_ind, timestamp,     |
    |       frame_index, episode_index, index, task_index              |
    +------------------------------------------------------------------+
                              |
                              |---------> 合并
                              |
    +------------------------------------------------------------------+
    | Dataset B                                                        |
    | keys: image, wrist_image, optional side_image, state, actions,   |
    |       value_label, reward, reward_label, adv_ind, timestamp,     |
    |       frame_index, episode_index, index, task_index              |
    +------------------------------------------------------------------+
                              |
                              |---------> 合并
                              v
    +------------------------------------------------------------------+
    | Output Dataset                                                   |
    | keys: image, wrist_image, optional side_image, state, actions,   |
    |       value_label, reward, reward_label, adv_ind, timestamp,     |
    |       frame_index, episode_index, index, task_index              |
    +------------------------------------------------------------------+

    字段变化:
    Dataset A/B -----> Output : image, wrist_image, optional side_image, state, actions
    Dataset A/B -----> Output : intervention, value_label, reward, reward_label, adv_ind
    Dataset A/B -----> Output : timestamp
    Dataset A/B -----> Output : frame_index, episode_index, index, task_index
    Dataset A/B - - -> Output : 仅重排全局 task_index / episode_index / frame_index / index
    Dataset A/B - - - -> Output : 去掉不在 14 键列表内的额外字段

English summary:
- This is a merge-only script.
- It assumes source datasets are already normalized to the same LeRobot schema.
- It keeps required keys plus optional `side_image` and rewrites parquet/meta directly.
- It does not fill missing fields or alter image layout/size.

Usage:
    sudo /.venv/bin/python scripts/merge_datasets.py \
      --sources \
        /public/home/user/lerobot_datasets/dataset_a \
        /public/home/user/lerobot_datasets/dataset_b \
      --output /public/home/user/lerobot_datasets/dataset_merged \
      --overwrite
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm


PARQUET_RE = re.compile(r"episode_(\d+)\.parquet$")
REQUIRED_KEEP_COLUMNS = [
    "image",
    "wrist_image",
    "state",
    "actions",
    "intervention",
    "value_label",
    "reward",
    "reward_label",
    "adv_ind",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
]
OPTIONAL_KEEP_COLUMNS = [
    "side_image",
]
CANDIDATE_KEEP_COLUMNS = REQUIRED_KEEP_COLUMNS + OPTIONAL_KEEP_COLUMNS
CANDIDATE_KEEP_COLUMNS_SET = set(CANDIDATE_KEEP_COLUMNS)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def _discover_episode_files(root: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for fpath in sorted((root / "data").glob("chunk-*/episode_*.parquet")):
        match = PARQUET_RE.match(fpath.name)
        if match:
            files[int(match.group(1))] = fpath
    return files


def _feature_entry(name: str, feature: dict[str, Any]) -> dict[str, Any]:
    copied = dict(feature)
    copied["names"] = [name]
    return copied


def _build_task_mapping(tasks_rows: list[dict[str, Any]]) -> dict[int, str]:
    return {int(row["task_index"]): str(row["task"]) for row in tasks_rows}


def _load_sources(sources: list[Path]) -> list[dict[str, Any]]:
    loaded: list[dict[str, Any]] = []
    for root in sources:
        info_path = root / "meta" / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Missing info.json: {info_path}")

        info = _load_json(info_path)
        tasks_rows = _load_jsonl(root / "meta" / "tasks.jsonl")
        episodes_rows = _load_jsonl(root / "meta" / "episodes.jsonl")
        episode_files = _discover_episode_files(root)

        selected_eps = []
        for row in sorted(episodes_rows, key=lambda x: int(x["episode_index"])):
            episode_index = int(row["episode_index"])
            if episode_index in episode_files:
                selected_eps.append(episode_index)

        loaded.append(
            {
                "root": root,
                "info": info,
                "tasks_rows": tasks_rows,
                "tasks_map": _build_task_mapping(tasks_rows),
                "episodes_rows_map": {int(r["episode_index"]): r for r in episodes_rows},
                "episode_files": episode_files,
                "selected_eps": selected_eps,
            }
        )
    return loaded


def _ensure_schema(
    sources: list[dict[str, Any]],
) -> tuple[dict[str, pa.DataType], dict[str, Any], list[str]]:
    global_types: dict[str, pa.DataType] = {}
    merged_features: dict[str, Any] = {}

    for src in sources:
        features = dict(src["info"].get("features", {}))
        for key in CANDIDATE_KEEP_COLUMNS:
            if key in features and key not in merged_features:
                merged_features[key] = _feature_entry(key, features[key])

        for episode_idx in src["selected_eps"][:1]:
            table = pq.read_table(src["episode_files"][episode_idx])
            for field in table.schema:
                if field.name in CANDIDATE_KEEP_COLUMNS_SET and field.name not in global_types:
                    global_types[field.name] = field.type

    optional_columns = [
        key for key in OPTIONAL_KEEP_COLUMNS
        if key in merged_features and key in global_types
    ]
    keep_columns = REQUIRED_KEEP_COLUMNS + optional_columns

    missing_features = [key for key in keep_columns if key not in merged_features]
    if missing_features:
        raise ValueError(f"Missing feature metadata for required keys: {missing_features}")

    missing_types = [key for key in keep_columns if key not in global_types]
    if missing_types:
        raise ValueError(f"Missing parquet schema types for required keys: {missing_types}")

    return global_types, {key: merged_features[key] for key in keep_columns}, keep_columns


def _set_or_add_column(table: pa.Table, name: str, arr: pa.Array) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx < 0:
        return table.append_column(name, arr)
    return table.set_column(idx, name, arr)


def _select_required_columns(table: pa.Table, keep_columns: list[str], global_types: dict[str, pa.DataType]) -> pa.Table:
    missing = [key for key in REQUIRED_KEEP_COLUMNS if table.schema.get_field_index(key) < 0]
    if missing:
        raise ValueError(f"Source episode is missing required keys: {missing}")
    for key in OPTIONAL_KEEP_COLUMNS:
        if key in keep_columns and table.schema.get_field_index(key) < 0:
            table = table.append_column(key, pa.nulls(table.num_rows, type=global_types[key]))
    return table.select(keep_columns)


def _available_cpu_threads() -> int:
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except Exception:
        cpu_count = os.cpu_count() or 1

    try:
        load1, _, _ = os.getloadavg()
        idle_estimate = cpu_count - int(math.floor(load1))
    except OSError:
        idle_estimate = cpu_count

    return max(1, min(cpu_count, idle_estimate))


def _resolve_num_workers(user_value: int | None, total_episodes: int) -> int:
    if total_episodes <= 0:
        return 1
    if user_value is not None:
        return max(1, min(int(user_value), total_episodes))
    return max(1, min(_available_cpu_threads(), total_episodes))


def _remap_task_indices(
    table: pa.Table,
    src_tasks_map: dict[int, str],
    global_task_to_index: dict[str, int],
    episode_tasks_hint: list[str],
) -> tuple[pa.Array, list[str]]:
    num_rows = table.num_rows
    old_task_idx = np.asarray(table.column("task_index").combine_chunks(), dtype=np.int64)
    mapped = np.zeros(num_rows, dtype=np.int64)
    used_tasks: list[str] = []

    for i, old_idx in enumerate(old_task_idx.tolist()):
        task_text = src_tasks_map.get(int(old_idx))
        if task_text is None:
            if episode_tasks_hint:
                task_text = episode_tasks_hint[0]
            else:
                task_text = f"task_{old_idx}"
        if task_text not in global_task_to_index:
            global_task_to_index[task_text] = len(global_task_to_index)
        mapped[i] = global_task_to_index[task_text]
        used_tasks.append(task_text)

    return pa.array(mapped, type=pa.int64()), sorted(set(used_tasks))


def _prepare_episode_table(
    episode_file: Path,
    old_episode_idx: int,
    episode_meta_row: dict[str, Any],
    keep_columns: list[str],
    global_types: dict[str, pa.DataType],
) -> tuple[pa.Table, int, list[str]]:
    table = pq.read_table(episode_file)
    table = _select_required_columns(table, keep_columns, global_types)
    ep_tasks = list(episode_meta_row.get("tasks", []))
    return table, old_episode_idx, ep_tasks


def _prepare_episode_item(item: tuple[Path, int, dict[str, Any], list[str], dict[str, pa.DataType]]) -> tuple[pa.Table, int, list[str]]:
    return _prepare_episode_table(*item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge already-aligned LeRobot datasets.")
    parser.add_argument("--sources", nargs="+", required=True, help="Source dataset roots.")
    parser.add_argument("--output", required=True, help="Output merged dataset root.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    parser.add_argument("--chunks-size", type=int, default=1000, help="Episode chunk size for output parquet layout.")
    parser.add_argument("--fps", type=int, default=None, help="Override output fps. Defaults to first source fps.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel worker threads for episode loading. Defaults to auto-detected available CPU threads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sources = [Path(s).expanduser().resolve() for s in args.sources]
    output_root = Path(args.output).expanduser().resolve()
    if not sources:
        raise ValueError("No sources provided.")

    if output_root.exists():
        if args.overwrite:
            shutil.rmtree(output_root)
        else:
            raise FileExistsError(f"Output exists: {output_root}. Use --overwrite.")

    (output_root / "data").mkdir(parents=True, exist_ok=True)
    (output_root / "meta").mkdir(parents=True, exist_ok=True)

    loaded = _load_sources(sources)
    global_types, merged_features, keep_columns = _ensure_schema(loaded)

    global_task_to_index: dict[str, int] = {}
    out_episodes: list[dict[str, Any]] = []
    out_episode_stats: list[dict[str, Any]] = []

    first_info = loaded[0]["info"]
    robot_type = first_info.get("robot_type", "unknown")
    fps = int(first_info.get("fps", 1)) if args.fps is None else int(args.fps)
    data_path = first_info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")

    total_source_episodes = sum(len(src["selected_eps"]) for src in loaded)
    num_workers = _resolve_num_workers(args.num_workers, total_source_episodes)
    progress = tqdm.tqdm(total=total_source_episodes, desc="Merging datasets", unit="ep")

    new_episode_idx = 0
    global_frame_offset = 0
    print(f"Using worker threads: {num_workers}", flush=True)
    try:
        for src in loaded:
            src_root: Path = src["root"]
            tasks_map: dict[int, str] = src["tasks_map"]
            episodes_rows_map: dict[int, dict[str, Any]] = src["episodes_rows_map"]

            work_items = [
                (
                    src["episode_files"][old_episode_idx],
                    old_episode_idx,
                    episodes_rows_map.get(old_episode_idx, {}),
                    keep_columns,
                    global_types,
                )
                for old_episode_idx in src["selected_eps"]
            ]

            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                prepared_iter = executor.map(_prepare_episode_item, work_items)
                for table, old_episode_idx, ep_tasks in prepared_iter:
                    num_rows = table.num_rows

                    task_arr, used_tasks = _remap_task_indices(table, tasks_map, global_task_to_index, ep_tasks)
                    table = _set_or_add_column(table, "task_index", task_arr)

                    episode_index_arr = pa.array(np.full(num_rows, new_episode_idx, dtype=np.int64), type=pa.int64())
                    frame_index_arr = pa.array(np.arange(num_rows, dtype=np.int64), type=pa.int64())
                    index_arr = pa.array(
                        np.arange(global_frame_offset, global_frame_offset + num_rows, dtype=np.int64),
                        type=pa.int64(),
                    )
                    table = _set_or_add_column(table, "episode_index", episode_index_arr)
                    table = _set_or_add_column(table, "frame_index", frame_index_arr)
                    table = _set_or_add_column(table, "index", index_arr)

                    out_chunk_idx = new_episode_idx // args.chunks_size
                    out_parquet = (
                        output_root / "data" / f"chunk-{out_chunk_idx:03d}" / f"episode_{new_episode_idx:06d}.parquet"
                    )
                    out_parquet.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(table.select(keep_columns), out_parquet)

                    out_episodes.append(
                        {
                            "episode_index": new_episode_idx,
                            "tasks": used_tasks if used_tasks else ep_tasks,
                            "length": int(num_rows),
                        }
                    )
                    out_episode_stats.append({"episode_index": new_episode_idx, "stats": {}})

                    global_frame_offset += num_rows
                    new_episode_idx += 1
                    progress.update(1)
                    progress.set_postfix_str(f"source={src_root.name}")
    finally:
        progress.close()

    out_tasks = [
        {"task_index": idx, "task": task}
        for task, idx in sorted(global_task_to_index.items(), key=lambda item: item[1])
    ]

    total_episodes = len(out_episodes)
    total_frames = int(sum(ep["length"] for ep in out_episodes))
    total_chunks = math.ceil(total_episodes / args.chunks_size) if total_episodes > 0 else 0

    out_info = {
        "codebase_version": first_info.get("codebase_version", "unknown"),
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(out_tasks),
        "total_videos": int(first_info.get("total_videos", 0)),
        "total_chunks": total_chunks,
        "chunks_size": int(args.chunks_size),
        "fps": int(fps),
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": data_path,
        "video_path": first_info.get("video_path"),
        "features": merged_features,
    }

    _write_json(output_root / "meta" / "info.json", out_info)
    _write_jsonl(output_root / "meta" / "tasks.jsonl", out_tasks)
    _write_jsonl(output_root / "meta" / "episodes.jsonl", out_episodes)
    _write_jsonl(output_root / "meta" / "episodes_stats.jsonl", out_episode_stats)

    print("Done.", flush=True)
    print(f"Output: {output_root}", flush=True)
    print(f"Episodes: {total_episodes}, Frames: {total_frames}, Tasks: {len(out_tasks)}", flush=True)
    print(f"Keys: {keep_columns}", flush=True)


if __name__ == "__main__":
    main()
