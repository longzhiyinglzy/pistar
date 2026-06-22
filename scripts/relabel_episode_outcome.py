"""Relabel one LeRobot episode as success or failure.

This updates the per-frame supervision columns used by the PiStar value
pipeline:

- success: value_label ramps from -1 to 0, final reward is 1
- failure: value_label is all -1, reward is all 0

The script also updates the matching row in meta/episodes_stats.jsonl when
those statistics exist.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


VALUE_LABEL_COLUMN = "value_label"
LEGACY_VALUE_LABEL_COLUMN = "value_lable"
REWARD_COLUMN = "reward"
REWARD_LABEL_COLUMN = "reward_label"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _dump_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def _resolve_episode_file(dataset_dir: Path, episode_index: int) -> Path:
    info_path = dataset_dir / "meta" / "info.json"
    info = _load_json(info_path) if info_path.exists() else {}
    chunks_size = int(info.get("chunks_size", 1000))
    data_path = info.get(
        "data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    )
    episode_chunk = episode_index // chunks_size
    episode_path = dataset_dir / data_path.format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )
    if not episode_path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {episode_path}")
    return episode_path


def _compute_labels(
    length: int,
    outcome: str,
    *,
    penalty_value: float,
    failure_terminal_reward_label: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if length <= 0:
        raise ValueError("Episode has no frames.")

    if outcome == "success":
        t = np.arange(length, dtype=np.float32)
        value_labels = -((float(length) - t) / float(length))
        value_labels[-1] = 0.0
        rewards = np.zeros(length, dtype=np.float32)
        rewards[-1] = 1.0
        reward_labels = np.full(length, -1.0 / float(length), dtype=np.float32)
        reward_labels[-1] = 0.0
        return value_labels, rewards, reward_labels

    if outcome == "failure":
        value_labels = np.full(length, penalty_value, dtype=np.float32)
        rewards = np.zeros(length, dtype=np.float32)
        reward_labels = np.full(length, -1.0 / float(length), dtype=np.float32)
        reward_labels[-1] = failure_terminal_reward_label
        return value_labels, rewards, reward_labels

    raise ValueError(f"Unsupported outcome: {outcome}")


def _set_or_add_float_column(table: pa.Table, column_name: str, values: np.ndarray) -> pa.Table:
    array = pa.array(values.astype(np.float32, copy=False), type=pa.float32())
    column_index = table.schema.get_field_index(column_name)
    if column_index >= 0:
        return table.set_column(column_index, column_name, array)
    return table.append_column(column_name, array)


def _rewrite_episode_parquet(
    episode_path: Path,
    outcome: str,
    *,
    penalty_value: float,
    failure_terminal_reward_label: float,
    dry_run: bool,
) -> tuple[int, dict[str, np.ndarray]]:
    table = pq.read_table(episode_path)
    length = table.num_rows
    value_labels, rewards, reward_labels = _compute_labels(
        length,
        outcome,
        penalty_value=penalty_value,
        failure_terminal_reward_label=failure_terminal_reward_label,
    )

    value_column = VALUE_LABEL_COLUMN
    if table.schema.get_field_index(value_column) < 0 and table.schema.get_field_index(LEGACY_VALUE_LABEL_COLUMN) >= 0:
        value_column = LEGACY_VALUE_LABEL_COLUMN

    updated = table
    updated = _set_or_add_float_column(updated, value_column, value_labels)
    updated = _set_or_add_float_column(updated, REWARD_COLUMN, rewards)
    updated = _set_or_add_float_column(updated, REWARD_LABEL_COLUMN, reward_labels)

    if not dry_run:
        tmp_path = episode_path.with_suffix(episode_path.suffix + ".tmp")
        pq.write_table(updated, tmp_path)
        os.replace(tmp_path, episode_path)

    return length, {
        value_column: value_labels,
        REWARD_COLUMN: rewards,
        REWARD_LABEL_COLUMN: reward_labels,
    }


def _stats_for_array(values: np.ndarray) -> dict[str, list[float | int]]:
    values64 = values.astype(np.float64, copy=False)
    return {
        "min": [float(np.min(values64))],
        "max": [float(np.max(values64))],
        "mean": [float(np.mean(values64))],
        "std": [float(np.std(values64))],
        "count": [int(values64.shape[0])],
    }


def _update_episode_stats(
    dataset_dir: Path,
    episode_index: int,
    labels: dict[str, np.ndarray],
    *,
    dry_run: bool,
) -> bool:
    stats_path = dataset_dir / "meta" / "episodes_stats.jsonl"
    if not stats_path.exists():
        return False

    rows = _load_jsonl(stats_path)
    updated = False
    for row in rows:
        if int(row.get("episode_index", -1)) != episode_index:
            continue
        stats = row.setdefault("stats", {})
        for column_name, values in labels.items():
            stats[column_name] = _stats_for_array(values)
        updated = True
        break

    if updated and not dry_run:
        _dump_jsonl_atomic(stats_path, rows)
    return updated


def _summarize(labels: dict[str, np.ndarray]) -> str:
    parts = []
    for column_name, values in labels.items():
        parts.append(
            f"{column_name}: first={float(values[0]):.6g}, "
            f"last={float(values[-1]):.6g}, min={float(np.min(values)):.6g}, "
            f"max={float(np.max(values)):.6g}"
        )
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel one LeRobot episode as success or failure.")
    parser.add_argument("--dataset", required=True, type=Path, help="Local LeRobot dataset directory")
    parser.add_argument("--episode-index", required=True, type=int, help="Episode index to relabel")
    parser.add_argument("--outcome", required=True, choices=("success", "failure"))
    parser.add_argument("--penalty-value", type=float, default=-1.0)
    parser.add_argument("--failure-terminal-reward-label", type=float, default=-1.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    dataset_dir = args.dataset.expanduser().resolve()
    episode_path = _resolve_episode_file(dataset_dir, args.episode_index)
    length, labels = _rewrite_episode_parquet(
        episode_path,
        args.outcome,
        penalty_value=args.penalty_value,
        failure_terminal_reward_label=args.failure_terminal_reward_label,
        dry_run=args.dry_run,
    )
    stats_updated = _update_episode_stats(
        dataset_dir,
        args.episode_index,
        labels,
        dry_run=args.dry_run,
    )

    mode = "DRY RUN" if args.dry_run else "UPDATED"
    print(f"{mode}: {episode_path}")
    print(f"episode_index={args.episode_index} outcome={args.outcome} frames={length}")
    print(_summarize(labels))
    print(f"episodes_stats.jsonl updated: {stats_updated}")


if __name__ == "__main__":
    main()
