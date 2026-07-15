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
    if isinstance(value, list | tuple):
        value = value[0]
    return float(value)


def _episode_has_intervention(path: Path) -> bool:
    table = pq.read_table(path, columns=["intervention"])
    values = table.column("intervention").combine_chunks().to_pylist()
    for value in values:
        if isinstance(value, list | tuple):
            if any(bool(item) for item in value):
                return True
        elif bool(value):
            return True
    return False


def _classify_episodes(root: Path, threshold: float) -> tuple[dict[str, list[int]], dict[int, Path]]:
    files = _episode_files(root)
    if not files:
        raise FileNotFoundError(f"No episode parquet files found under {root / 'data'}")

    groups: dict[str, list[int]] = {
        "success_intervention": [],
        "success_autonomous": [],
        "failure": [],
    }
    for episode_index, path in sorted(files.items()):
        if _terminal_value(path) > threshold:
            group = "success_intervention" if _episode_has_intervention(path) else "success_autonomous"
            groups[group].append(episode_index)
        else:
            groups["failure"].append(episode_index)
    return groups, files


def _sample(
    indices: list[int],
    count: int,
    rng: random.Random,
    description: str,
    strategy: str,
) -> list[int]:
    if count < 0:
        raise ValueError(f"{description} count must be non-negative, got {count}")
    if count > len(indices):
        raise ValueError(f"Requested {count} {description} episodes, but only {len(indices)} are available")
    if count == len(indices):
        return sorted(indices)
    if strategy == "random":
        return sorted(rng.sample(indices, count))
    if strategy == "stratified":
        selected: list[int] = []
        for bin_index in range(count):
            start = bin_index * len(indices) // count
            end = max((bin_index + 1) * len(indices) // count, start + 1)
            selected.append(rng.choice(indices[start:end]))
        return sorted(selected)
    raise ValueError(f"Unsupported sampling strategy: {strategy}")


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
        type=int,
        help="Successful episodes to sample from each source without intervention stratification.",
    )
    parser.add_argument(
        "--intervention-success-counts",
        nargs="+",
        type=int,
        help="Successful intervention episodes to sample from each source.",
    )
    parser.add_argument(
        "--autonomous-success-counts",
        nargs="+",
        type=int,
        help="Successful autonomous episodes to sample from each source.",
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
        "--sampling-strategy",
        choices=("random", "stratified"),
        default="random",
        help="Episode selection strategy within each outcome/intervention group.",
    )
    parser.add_argument(
        "--terminal-threshold",
        type=float,
        default=-0.5,
        help="Terminal value_label above this value is success; otherwise failure.",
    )
    parser.add_argument("--fps", type=int, default=None, help="Optional output FPS override.")
    parser.add_argument("--num-workers", type=int, default=0, help="Worker count passed to merge_datasets.py.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing output dataset.")
    parser.add_argument("--dry-run", action="store_true", help="Print the selection without creating a dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = [Path(value).expanduser().resolve() for value in args.sources]
    output = Path(args.output).expanduser().resolve()
    grouped_success_mode = args.intervention_success_counts is not None or args.autonomous_success_counts is not None
    if grouped_success_mode:
        if args.success_counts is not None:
            raise ValueError("Use either --success-counts or the two stratified success-count arguments, not both")
        if args.intervention_success_counts is None or args.autonomous_success_counts is None:
            raise ValueError(
                "--intervention-success-counts and --autonomous-success-counts must be provided together"
            )
        success_count_groups = (args.intervention_success_counts, args.autonomous_success_counts)
    else:
        if args.success_counts is None:
            raise ValueError(
                "Provide --success-counts, or provide both intervention/autonomous success counts"
            )
        success_count_groups = (args.success_counts,)

    count_groups = (*success_count_groups, args.failure_counts)
    if any(len(counts) != len(sources) for counts in count_groups):
        raise ValueError("Every count argument must contain one value per --sources entry")

    rng = random.Random(args.seed)
    selections: list[dict[str, Any]] = []
    source_files: list[dict[int, Path]] = []
    for source_index, source in enumerate(sources):
        groups, files = _classify_episodes(source, args.terminal_threshold)
        success_intervention = groups["success_intervention"]
        success_autonomous = groups["success_autonomous"]
        failure = groups["failure"]

        if grouped_success_mode:
            selected_success_intervention = _sample(
                success_intervention,
                args.intervention_success_counts[source_index],
                rng,
                f"successful intervention episodes from {source.name}",
                args.sampling_strategy,
            )
            selected_success_autonomous = _sample(
                success_autonomous,
                args.autonomous_success_counts[source_index],
                rng,
                f"successful autonomous episodes from {source.name}",
                args.sampling_strategy,
            )
        else:
            success = sorted(success_intervention + success_autonomous)
            selected_success = _sample(
                success,
                args.success_counts[source_index],
                rng,
                f"success episodes from {source.name}",
                args.sampling_strategy,
            )
            intervention_set = set(success_intervention)
            selected_success_intervention = [value for value in selected_success if value in intervention_set]
            selected_success_autonomous = [value for value in selected_success if value not in intervention_set]

        selected_success = sorted(selected_success_intervention + selected_success_autonomous)
        selected_failure = _sample(
            failure,
            args.failure_counts[source_index],
            rng,
            f"failure episodes from {source.name}",
            args.sampling_strategy,
        )
        selections.append(
            {
                "source": str(source),
                "available_success": len(success_intervention) + len(success_autonomous),
                "available_success_intervention": len(success_intervention),
                "available_success_autonomous": len(success_autonomous),
                "available_failure": len(failure),
                "selected_success": selected_success,
                "selected_success_intervention": selected_success_intervention,
                "selected_success_autonomous": selected_success_autonomous,
                "selected_failure": selected_failure,
            }
        )
        source_files.append(files)
        print(
            f"{source.name}: selected success={len(selected_success)}/"
            f"{len(success_intervention) + len(success_autonomous)} "
            f"(intervention={len(selected_success_intervention)}/{len(success_intervention)}, "
            f"autonomous={len(selected_success_autonomous)}/{len(success_autonomous)}), "
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

    if args.dry_run:
        print("Dry run: no output dataset was created.", flush=True)
        return

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
        "sampling_strategy": args.sampling_strategy,
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
