#!/usr/bin/env python3
"""Create an episode-level subset of a LeRobot dataset.

The source dataset is never modified. The output parquet files are rewritten
with contiguous episode/frame/global indices, while videos can be symlinked,
copied, or skipped. The default sampling strategy is stratified over the
episode order so a 250/500 split covers the whole collection instead of taking
only the first half.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import re
import shutil
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


EPISODE_RE = re.compile(r"episode_(\d+)\.(parquet|mp4)$")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _discover_episode_files(root: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in sorted((root / "data").glob("chunk-*/episode_*.parquet")):
        match = EPISODE_RE.match(path.name)
        if match:
            files[int(match.group(1))] = path
    return files


def _video_feature_keys(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    return [
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    ]


def _select_stratified(indices: list[int], count: int, rng: random.Random) -> list[int]:
    """Sample one episode from each equal-width bin over sorted episode order."""
    if count > len(indices):
        raise ValueError(f"Requested {count} episodes, but only {len(indices)} are available")
    if count == len(indices):
        return list(indices)

    selected: list[int] = []
    used: set[int] = set()
    n = len(indices)
    for bin_idx in range(count):
        start = int(np.floor(bin_idx * n / count))
        end = int(np.floor((bin_idx + 1) * n / count))
        end = max(end, start + 1)
        candidates = [value for value in indices[start:end] if value not in used]
        if not candidates:
            candidates = [value for value in indices if value not in used]
        choice = rng.choice(candidates)
        selected.append(choice)
        used.add(choice)
    return sorted(selected)


def _select_even(indices: list[int], count: int) -> list[int]:
    if count > len(indices):
        raise ValueError(f"Requested {count} episodes, but only {len(indices)} are available")
    if count == len(indices):
        return list(indices)
    positions = np.linspace(0, len(indices) - 1, count)
    selected_positions = sorted({int(round(pos)) for pos in positions})
    cursor = 0
    while len(selected_positions) < count:
        if cursor not in selected_positions:
            selected_positions.append(cursor)
        cursor += 1
    return sorted(indices[pos] for pos in sorted(selected_positions[:count]))


def _select_episodes(indices: list[int], count: int, strategy: str, seed: int) -> list[int]:
    rng = random.Random(seed)
    if count < 0:
        raise ValueError(f"Episode count must be non-negative, got {count}")
    if strategy == "stratified":
        return _select_stratified(indices, count, rng)
    if strategy == "even":
        return _select_even(indices, count)
    if strategy == "random":
        if count > len(indices):
            raise ValueError(f"Requested {count} episodes, but only {len(indices)} are available")
        return sorted(rng.sample(indices, count))
    if strategy == "first":
        return indices[:count]
    raise ValueError(f"Unknown sampling strategy: {strategy}")


def _replace_or_append_column(table: pa.Table, name: str, array: pa.Array) -> pa.Table:
    field_index = table.schema.get_field_index(name)
    if field_index < 0:
        return table.append_column(name, array)
    return table.set_column(field_index, name, array)


def _map_task_indices(
    table: pa.Table,
    old_task_to_text: dict[int, str],
    new_task_to_index: dict[str, int],
    episode_tasks: list[str],
) -> tuple[pa.Array, list[str]]:
    if table.schema.get_field_index("task_index") < 0:
        task_text = episode_tasks[0] if episode_tasks else "default"
        if task_text not in new_task_to_index:
            new_task_to_index[task_text] = len(new_task_to_index)
        mapped = np.full(table.num_rows, new_task_to_index[task_text], dtype=np.int64)
        return pa.array(mapped, type=pa.int64()), [task_text]

    old_task_indices = np.asarray(table.column("task_index").combine_chunks(), dtype=np.int64)
    mapped = np.zeros(table.num_rows, dtype=np.int64)
    used_tasks: list[str] = []
    for row_idx, old_task_index in enumerate(old_task_indices.tolist()):
        task_text = old_task_to_text.get(int(old_task_index))
        if task_text is None:
            task_text = episode_tasks[0] if episode_tasks else f"task_{old_task_index}"
        if task_text not in new_task_to_index:
            new_task_to_index[task_text] = len(new_task_to_index)
        mapped[row_idx] = new_task_to_index[task_text]
        used_tasks.append(task_text)
    return pa.array(mapped, type=pa.int64()), sorted(set(used_tasks))


def _rewrite_episode(
    *,
    source_file: Path,
    output_file: Path,
    new_episode_index: int,
    global_frame_offset: int,
    old_task_to_text: dict[int, str],
    new_task_to_index: dict[str, int],
    episode_tasks: list[str],
) -> tuple[int, list[str]]:
    table = pq.read_table(source_file)
    num_rows = table.num_rows

    table = _replace_or_append_column(
        table,
        "episode_index",
        pa.array(np.full(num_rows, new_episode_index, dtype=np.int64), type=pa.int64()),
    )
    table = _replace_or_append_column(
        table,
        "frame_index",
        pa.array(np.arange(num_rows, dtype=np.int64), type=pa.int64()),
    )
    table = _replace_or_append_column(
        table,
        "index",
        pa.array(np.arange(global_frame_offset, global_frame_offset + num_rows, dtype=np.int64), type=pa.int64()),
    )
    task_index_array, used_tasks = _map_task_indices(
        table,
        old_task_to_text=old_task_to_text,
        new_task_to_index=new_task_to_index,
        episode_tasks=episode_tasks,
    )
    table = _replace_or_append_column(table, "task_index", task_index_array)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_file)
    return num_rows, used_tasks


def _link_or_copy_video(source: Path, destination: Path, mode: str) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "skip":
        return
    if mode == "copy":
        shutil.copy2(source, destination)
        return
    destination.symlink_to(source)


def _copy_selected_videos(
    *,
    source_root: Path,
    output_root: Path,
    info: dict[str, Any],
    old_episode_index: int,
    new_episode_index: int,
    old_chunk_index: int,
    new_chunk_index: int,
    mode: str,
) -> int:
    if mode == "skip" or not info.get("video_path"):
        return 0

    copied = 0
    for video_key in _video_feature_keys(info):
        source = (
            source_root
            / "videos"
            / f"chunk-{old_chunk_index:03d}"
            / video_key
            / f"episode_{old_episode_index:06d}.mp4"
        )
        destination = (
            output_root
            / "videos"
            / f"chunk-{new_chunk_index:03d}"
            / video_key
            / f"episode_{new_episode_index:06d}.mp4"
        )
        if source.exists():
            _link_or_copy_video(source, destination, mode)
            copied += 1
    return copied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Input LeRobot dataset root.")
    parser.add_argument("--output", required=True, help="Output subset dataset root.")
    parser.add_argument("--count", type=int, required=True, help="Number of episodes to select.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed for stratified/random strategies.")
    parser.add_argument(
        "--strategy",
        choices=["stratified", "even", "random", "first"],
        default="stratified",
        help="Episode sampling strategy. Default: stratified over episode order.",
    )
    parser.add_argument(
        "--video-mode",
        choices=["symlink", "copy", "skip"],
        default="symlink",
        help="How to expose selected videos in the output dataset.",
    )
    parser.add_argument("--chunks-size", type=int, default=None, help="Output episodes per chunk. Defaults to source value.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source_root}")
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output_root}. Use --overwrite.")
        shutil.rmtree(output_root)

    info = _load_json(source_root / "meta" / "info.json")
    episodes = _load_jsonl(source_root / "meta" / "episodes.jsonl")
    episode_stats = _load_jsonl(source_root / "meta" / "episodes_stats.jsonl")
    tasks = _load_jsonl(source_root / "meta" / "tasks.jsonl")
    episode_files = _discover_episode_files(source_root)
    if not episode_files:
        raise FileNotFoundError(f"No parquet episodes found under {source_root / 'data'}")

    available = [
        int(row["episode_index"])
        for row in sorted(episodes, key=lambda item: int(item["episode_index"]))
        if int(row["episode_index"]) in episode_files
    ]
    selected = _select_episodes(available, args.count, args.strategy, args.seed)
    selected_set = set(selected)

    old_episode_rows = {int(row["episode_index"]): row for row in episodes}
    old_episode_stats = {int(row["episode_index"]): row for row in episode_stats}
    old_task_to_text = {int(row["task_index"]): str(row["task"]) for row in tasks}
    new_task_to_index: dict[str, int] = {}

    chunks_size = int(args.chunks_size if args.chunks_size is not None else info.get("chunks_size", 1000))
    output_episodes: list[dict[str, Any]] = []
    output_episode_stats: list[dict[str, Any]] = []
    global_frame_offset = 0
    linked_or_copied_videos = 0

    for new_episode_index, old_episode_index in enumerate(selected):
        old_chunk_index = old_episode_index // int(info.get("chunks_size", 1000))
        new_chunk_index = new_episode_index // chunks_size
        old_row = old_episode_rows.get(old_episode_index, {})
        old_tasks = list(old_row.get("tasks", []))

        output_file = (
            output_root
            / "data"
            / f"chunk-{new_chunk_index:03d}"
            / f"episode_{new_episode_index:06d}.parquet"
        )
        num_rows, used_tasks = _rewrite_episode(
            source_file=episode_files[old_episode_index],
            output_file=output_file,
            new_episode_index=new_episode_index,
            global_frame_offset=global_frame_offset,
            old_task_to_text=old_task_to_text,
            new_task_to_index=new_task_to_index,
            episode_tasks=old_tasks,
        )

        output_episodes.append(
            {
                "episode_index": new_episode_index,
                "tasks": used_tasks if used_tasks else old_tasks,
                "length": int(num_rows),
            }
        )
        stats_row = dict(old_episode_stats.get(old_episode_index, {"stats": {}}))
        stats_row["episode_index"] = new_episode_index
        output_episode_stats.append(stats_row)

        linked_or_copied_videos += _copy_selected_videos(
            source_root=source_root,
            output_root=output_root,
            info=info,
            old_episode_index=old_episode_index,
            new_episode_index=new_episode_index,
            old_chunk_index=old_chunk_index,
            new_chunk_index=new_chunk_index,
            mode=args.video_mode,
        )
        global_frame_offset += num_rows

    output_tasks = [
        {"task_index": task_index, "task": task}
        for task, task_index in sorted(new_task_to_index.items(), key=lambda item: item[1])
    ]
    total_chunks = int(np.ceil(len(selected) / chunks_size)) if selected else 0
    video_count = 0 if args.video_mode == "skip" else linked_or_copied_videos

    output_info = dict(info)
    output_info.update(
        {
            "total_episodes": len(selected),
            "total_frames": int(global_frame_offset),
            "total_tasks": len(output_tasks),
            "total_videos": int(video_count),
            "total_chunks": total_chunks,
            "chunks_size": chunks_size,
            "splits": {"train": f"0:{len(selected)}"},
        }
    )
    if args.video_mode == "skip":
        output_info["video_path"] = None

    _write_json(output_root / "meta" / "info.json", output_info)
    _write_jsonl(output_root / "meta" / "tasks.jsonl", output_tasks)
    _write_jsonl(output_root / "meta" / "episodes.jsonl", output_episodes)
    _write_jsonl(output_root / "meta" / "episodes_stats.jsonl", output_episode_stats)
    _write_json(
        output_root / "meta" / "selection_manifest.json",
        {
            "source": str(source_root),
            "count": len(selected),
            "seed": args.seed,
            "strategy": args.strategy,
            "video_mode": args.video_mode,
            "selected_old_episode_indices": selected,
            "old_to_new_episode_index": {str(old): new for new, old in enumerate(selected)},
        },
    )

    print(f"Done: {output_root}")
    print(f"Selected episodes: {len(selected)} / {len(available)}")
    print(f"Frames: {global_frame_offset}")
    print(f"Videos exposed: {video_count} ({args.video_mode})")
    print(f"First selected old episodes: {selected[:20]}")
    print(f"Manifest: {output_root / 'meta' / 'selection_manifest.json'}")


if __name__ == "__main__":
    main()
