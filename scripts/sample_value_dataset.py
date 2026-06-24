#!/usr/bin/env python3
"""Build a reproducible outcome-balanced value-training dataset.

The source datasets are never modified. Selected episodes are exposed through
temporary symlinks and then rewritten by ``merge_datasets.py`` so the output
has contiguous episode/frame indices and regular LeRobot metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any

import pyarrow.parquet as pq


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


def _episode_files(root: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in sorted((root / "data").glob("chunk-*/episode_*.parquet")):
        files[int(path.stem.rsplit("_", 1)[1])] = path
    return files


def _terminal_value(path: Path) -> float:
    table = pq.read_table(path, columns=["value_label"])
    values = table.column("value_label").combine_chunks()
    if len(values) == 0:
        raise ValueError(f"Episode contains no frames: {path}")
    value = values[-1].as_py()
    if isinstance(value, (list, tuple)):
        value = value[0]
    return float(value)


def _classify_episodes(root: Path, threshold: float) -> tuple[list[int], list[int], dict[int, Path]]:
    files = _episode_files(root)
    if not files:
        raise FileNotFoundError(f"No episode parquet files found under {root / 'data'}")

    success: list[int] = []
    failure: list[int] = []
    for episode_index, path in sorted(files.items()):
        if _terminal_value(path) > threshold:
            success.append(episode_index)
        else:
            failure.append(episode_index)
    return success, failure, files


def _sample(indices: list[int], count: int, rng: random.Random, description: str) -> list[int]:
    if count < 0:
        raise ValueError(f"{description} count must be non-negative, got {count}")
    if count > len(indices):
        raise ValueError(f"Requested {count} {description} episodes, but only {len(indices)} are available")
    if count == len(indices):
        return sorted(indices)
    return sorted(rng.sample(indices, count))


def _create_selected_source(source: Path, destination: Path, selected: list[int], files: dict[int, Path]) -> None:
    info = _load_json(source / "meta" / "info.json")
    episode_rows = _load_jsonl(source / "meta" / "episodes.jsonl")
    episode_stats_rows = _load_jsonl(source / "meta" / "episodes_stats.jsonl")
    selected_set = set(selected)

    (destination / "data").mkdir(parents=True, exist_ok=True)
    (destination / "meta").mkdir(parents=True, exist_ok=True)

    for episode_index in selected:
        source_file = files[episode_index]
        relative_path = source_file.relative_to(source)
        destination_file = destination / relative_path
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        destination_file.symlink_to(source_file)

    info["total_episodes"] = len(selected)
    info["total_frames"] = sum(
        int(row.get("length", 0)) for row in episode_rows if int(row["episode_index"]) in selected_set
    )
    _write_json(destination / "meta" / "info.json", info)
    _write_jsonl(destination / "meta" / "tasks.jsonl", _load_jsonl(source / "meta" / "tasks.jsonl"))
    _write_jsonl(
        destination / "meta" / "episodes.jsonl",
        [row for row in episode_rows if int(row["episode_index"]) in selected_set],
    )
    _write_jsonl(
        destination / "meta" / "episodes_stats.jsonl",
        [row for row in episode_stats_rows if int(row["episode_index"]) in selected_set],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", nargs="+", required=True, help="Aligned LeRobot source dataset roots.")
    parser.add_argument(
        "--success-counts",
        nargs="+",
        required=True,
        type=int,
        help="Successful episodes to sample from each source, aligned with --sources.",
    )
    parser.add_argument(
        "--failure-counts",
        nargs="+",
        required=True,
        type=int,
        help="Failed episodes to sample from each source, aligned with --sources.",
    )
    parser.add_argument("--output", required=True, help="Output dataset root; source datasets are not modified.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed.")
    parser.add_argument(
        "--terminal-threshold",
        type=float,
        default=-0.5,
        help="Terminal value_label above this value is success; otherwise failure.",
    )
    parser.add_argument("--fps", type=int, default=None, help="Optional output FPS override.")
    parser.add_argument("--num-workers", type=int, default=0, help="Worker count passed to merge_datasets.py.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [Path(value).expanduser().resolve() for value in args.sources]
    output = Path(args.output).expanduser().resolve()
    if len(args.success_counts) != len(sources) or len(args.failure_counts) != len(sources):
        raise ValueError("--success-counts and --failure-counts must each contain one value per source")

    rng = random.Random(args.seed)
    selections: list[dict[str, Any]] = []
    source_files: list[dict[int, Path]] = []
    for source, success_count, failure_count in zip(
        sources, args.success_counts, args.failure_counts, strict=True
    ):
        success, failure, files = _classify_episodes(source, args.terminal_threshold)
        selected_success = _sample(success, success_count, rng, f"success episodes from {source.name}")
        selected_failure = _sample(failure, failure_count, rng, f"failure episodes from {source.name}")
        selections.append(
            {
                "source": str(source),
                "available_success": len(success),
                "available_failure": len(failure),
                "selected_success": selected_success,
                "selected_failure": selected_failure,
            }
        )
        source_files.append(files)
        print(
            f"{source.name}: selected success={len(selected_success)}/{len(success)}, "
            f"failure={len(selected_failure)}/{len(failure)}",
            flush=True,
        )

    total_success = sum(len(item["selected_success"]) for item in selections)
    total_failure = sum(len(item["selected_failure"]) for item in selections)
    total = total_success + total_failure
    print(
        f"Selection total: success={total_success}, failure={total_failure}, "
        f"failure_ratio={total_failure / total:.1%}",
        flush=True,
    )

    script_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="pistar_value_subset_") as temp_value:
        temp_root = Path(temp_value)
        selected_sources: list[Path] = []
        for source_index, (source, selection, files) in enumerate(
            zip(sources, selections, source_files, strict=True)
        ):
            selected = sorted(selection["selected_success"] + selection["selected_failure"])
            selected_source = temp_root / f"source_{source_index:02d}"
            _create_selected_source(source, selected_source, selected, files)
            selected_sources.append(selected_source)

        command = [
            sys.executable,
            str(script_dir / "merge_datasets.py"),
            "--sources",
            *(str(path) for path in selected_sources),
            "--output",
            str(output),
            "--num-workers",
            str(args.num_workers),
        ]
        if args.fps is not None:
            command.extend(["--fps", str(args.fps)])
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)

    manifest = {
        "seed": args.seed,
        "terminal_threshold": args.terminal_threshold,
        "total_success": total_success,
        "total_failure": total_failure,
        "failure_ratio": total_failure / total,
        "sources": selections,
    }
    _write_json(output / "meta" / "selection_manifest.json", manifest)
    print(f"Selection manifest: {output / 'meta' / 'selection_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
