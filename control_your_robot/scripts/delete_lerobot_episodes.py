#!/usr/bin/env python
"""
删除本地 LeRobot 数据集中的指定 episodes，并保持数据集结构一致。

这个脚本会同时处理：
- data/chunk-*/episode_*.parquet
- meta/episodes.jsonl
- meta/episodes_stats.jsonl
- meta/info.json
- videos/chunk-*/*/episode_*.mp4（如果数据集使用视频）

和简单删除文件不同，它会对保留下来的 episodes 重新连续编号，并同步修复
parquet 里的 `episode_index` 与全局 `index` 列。

示例：
1. 直接指定数据集目录:
   python scripts/delete_lerobot_episodes.py \
       --dataset-root /app/dataset/rollout/white_rollout3_new \
       --episodes 12 16

2. 兼容旧参数:
   python scripts/delete_lerobot_episodes.py \
       --repo-id plug_rollout1 \
       --output-dir /path/to/rollout \
       --episode 73 74 75
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        f.write("\n")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="删除 LeRobot 数据集中的 episodes，并自动修复索引")
    parser.add_argument("--dataset-root", type=str, help="数据集根目录，例如 /path/to/plug_rollout1")
    parser.add_argument("--repo-id", type=str, help="数据集 ID，例如 plug_rollout1")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/media/chaihoa/software1/project_wang/dataset/lerobot",
        help="数据集父目录，最终根目录会拼成 output_dir/repo_id",
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode", type=int, help="删除单个 episode")
    group.add_argument("--episodes", type=int, nargs="+", help="删除多个 episodes，例如 46 50")
    group.add_argument("--range", type=int, nargs=2, metavar=("START", "END"), help="删除闭区间 START-END")
    group.add_argument("--delete-all", action="store_true", help="删除整个数据集目录")

    parser.add_argument("--dry-run", action="store_true", help="只预览，不真正写入")
    return parser.parse_args()


def resolve_dataset_root(args: argparse.Namespace) -> Path:
    if args.dataset_root:
        return Path(args.dataset_root).expanduser().resolve()
    if args.repo_id:
        return (Path(args.output_dir).expanduser().resolve() / args.repo_id).resolve()
    raise ValueError("请提供 --dataset-root，或同时提供 --repo-id 和 --output-dir")


def get_episode_indices(args: argparse.Namespace) -> list[int]:
    if args.delete_all:
        return []
    if args.episode is not None:
        return [args.episode]
    if args.episodes is not None:
        return args.episodes
    if args.range is not None:
        start, end = args.range
        if end < start:
            raise ValueError("--range 的 END 必须大于等于 START")
        return list(range(start, end + 1))
    raise ValueError("未提供要删除的 episodes")


def get_video_keys(info: dict) -> list[str]:
    return [key for key, ft in info["features"].items() if ft["dtype"] == "video"]


def get_chunk_size(info: dict) -> int:
    return int(info.get("chunks_size", 1000))


def get_data_file_path(dataset_root: Path, info: dict, episode_index: int) -> Path:
    chunk_size = get_chunk_size(info)
    episode_chunk = episode_index // chunk_size
    rel = info["data_path"].format(episode_chunk=episode_chunk, episode_index=episode_index)
    return dataset_root / rel


def get_video_file_path(dataset_root: Path, info: dict, episode_index: int, video_key: str) -> Path:
    chunk_size = get_chunk_size(info)
    episode_chunk = episode_index // chunk_size
    rel = info["video_path"].format(
        episode_chunk=episode_chunk,
        video_key=video_key,
        episode_index=episode_index,
    )
    return dataset_root / rel


def replace_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    idx = table.schema.get_field_index(name)
    if idx < 0:
        return table
    return table.set_column(idx, name, values)


def patch_episode_stats(stats: dict, new_episode_index: int, new_index_start: int, episode_length: int) -> None:
    if "episode_index" in stats:
        stats["episode_index"]["min"] = [new_episode_index]
        stats["episode_index"]["max"] = [new_episode_index]
        stats["episode_index"]["mean"] = [float(new_episode_index)]
        stats["episode_index"]["std"] = [0.0]
        stats["episode_index"]["count"] = [episode_length]

    if "index" in stats:
        end_index = new_index_start + episode_length - 1
        stats["index"]["min"] = [new_index_start]
        stats["index"]["max"] = [end_index]
        stats["index"]["mean"] = [float(new_index_start + end_index) / 2.0]
        stats["index"]["count"] = [episode_length]


def backup_meta(dataset_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = dataset_root / "meta" / "backups" / f"delete_episodes_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=False)

    for name in ["info.json", "episodes.jsonl", "episodes_stats.jsonl", "tasks.jsonl"]:
        src = dataset_root / "meta" / name
        if src.exists():
            shutil.copy2(src, backup_dir / name)

    return backup_dir


def write_table_atomically(table: pa.Table, dest: Path) -> None:
    tmp_dest = dest.with_name(dest.name + ".tmpwrite")
    if tmp_dest.exists():
        tmp_dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, tmp_dest)
    tmp_dest.replace(dest)


def rewrite_parquet_episode(src: Path, dest: Path, new_episode_index: int, new_index_start: int, expected_length: int) -> None:
    table = pq.read_table(src)
    if table.num_rows != expected_length:
        raise ValueError(
            f"Parquet 行数与元数据不一致: {src} 行数={table.num_rows}, metadata length={expected_length}"
        )

    if "episode_index" in table.column_names:
        ep_type = table.schema.field("episode_index").type
        table = replace_column(
            table,
            "episode_index",
            pa.array([new_episode_index] * table.num_rows, type=ep_type),
        )

    if "index" in table.column_names:
        index_type = table.schema.field("index").type
        table = replace_column(
            table,
            "index",
            pa.array(range(new_index_start, new_index_start + table.num_rows), type=index_type),
        )

    if dest.exists() and dest != src:
        raise FileExistsError(f"目标文件已存在，拒绝覆盖: {dest}")

    write_table_atomically(table, dest)
    if src != dest and src.exists():
        src.unlink()


def rename_video_episode(src: Path, dest: Path) -> None:
    if dest.exists() and dest != src:
        raise FileExistsError(f"目标视频已存在，拒绝覆盖: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dest)


def cleanup_empty_chunk_dirs(dataset_root: Path) -> None:
    for parent in [dataset_root / "data", dataset_root / "videos"]:
        if not parent.exists():
            continue
        for child in sorted(parent.rglob("*"), reverse=True):
            if child.is_dir():
                try:
                    child.rmdir()
                except OSError:
                    pass


def summarize_changes(kept_plan: list[dict], removed_indices: list[int]) -> None:
    print("\n删除的 episodes:", removed_indices)
    print(f"保留 episodes 数量: {len(kept_plan)}")
    moved = [item for item in kept_plan if item["old_index"] != item["new_index"]]
    if moved:
        print(f"需要重编号的 episodes 数量: {len(moved)}")
        head = moved[:5]
        tail = moved[-5:] if len(moved) > 5 else []
        print("前几个重编号映射:")
        for item in head:
            print(f"  {item['old_index']} -> {item['new_index']}")
        if tail:
            print("后几个重编号映射:")
            for item in tail:
                print(f"  {item['old_index']} -> {item['new_index']}")
    else:
        print("没有需要重编号的 episode")


def delete_all(dataset_root: Path, dry_run: bool) -> None:
    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集不存在: {dataset_root}")

    if dry_run:
        print(f"[dry-run] 将删除整个数据集: {dataset_root}")
        return

    confirm = input(f"⚠️  确定要删除整个数据集 '{dataset_root}' 吗？(yes/no): ")
    if confirm.lower() != "yes":
        print("已取消删除")
        return

    shutil.rmtree(dataset_root)
    print(f"✅ 已删除整个数据集: {dataset_root}")


def main() -> None:
    args = parse_args()
    dataset_root = resolve_dataset_root(args)

    if args.delete_all:
        delete_all(dataset_root, args.dry_run)
        return

    if not dataset_root.exists():
        raise FileNotFoundError(f"数据集不存在: {dataset_root}")

    info_path = dataset_root / "meta" / "info.json"
    episodes_path = dataset_root / "meta" / "episodes.jsonl"
    stats_path = dataset_root / "meta" / "episodes_stats.jsonl"

    info = load_json(info_path)
    episodes_rows = load_jsonl(episodes_path)
    stats_rows = load_jsonl(stats_path) if stats_path.exists() else []

    episodes_rows = sorted(episodes_rows, key=lambda x: x["episode_index"])
    stats_rows = sorted(stats_rows, key=lambda x: x["episode_index"])

    episode_by_old_index = {row["episode_index"]: row for row in episodes_rows}
    stats_by_old_index = {row["episode_index"]: row for row in stats_rows}
    delete_indices = sorted(set(get_episode_indices(args)))

    print(f"数据集目录: {dataset_root}")
    print(f"当前 metadata episodes 数量: {len(episodes_rows)}")
    print(f"请求删除: {delete_indices}")

    invalid_indices = [idx for idx in delete_indices if idx not in episode_by_old_index]
    if invalid_indices:
        raise ValueError(f"以下 episode 不存在于 metadata 中: {invalid_indices}")

    if not stats_rows:
        print("提示: 未找到 episodes_stats.jsonl，将只修复 episodes/info/parquet")

    kept_plan = []
    new_episodes_rows = []
    new_stats_rows = []
    new_total_frames = 0

    for old_index in sorted(episode_by_old_index):
        old_row = episode_by_old_index[old_index]
        if old_index in delete_indices:
            continue

        new_index = len(new_episodes_rows)
        new_row = copy.deepcopy(old_row)
        new_row["episode_index"] = new_index
        new_episodes_rows.append(new_row)

        length = int(new_row["length"])
        plan_item = {
            "old_index": old_index,
            "new_index": new_index,
            "length": length,
            "new_index_start": new_total_frames,
        }
        kept_plan.append(plan_item)

        if stats_rows:
            if old_index not in stats_by_old_index:
                raise ValueError(f"episodes_stats.jsonl 缺少 episode {old_index}")
            stat_row = copy.deepcopy(stats_by_old_index[old_index])
            stat_row["episode_index"] = new_index
            patch_episode_stats(stat_row["stats"], new_index, new_total_frames, length)
            new_stats_rows.append(stat_row)

        new_total_frames += length

    expected_total_frames = sum(
        int(row["length"]) for row in episodes_rows if row["episode_index"] not in delete_indices
    )
    if new_total_frames != expected_total_frames:
        raise ValueError("内部错误: 计算得到的 total_frames 不一致")

    video_keys = get_video_keys(info)
    summarize_changes(kept_plan, delete_indices)

    missing_removed_files = []
    for old_index in delete_indices:
        data_file = get_data_file_path(dataset_root, info, old_index)
        if not data_file.exists():
            missing_removed_files.append(str(data_file))

    if missing_removed_files:
        print("\n提示: 以下待删除 parquet 已经不存在，脚本会按“修复缺失后的合法状态”继续执行:")
        for path in missing_removed_files:
            print(f"  {path}")

    for item in kept_plan:
        data_file = get_data_file_path(dataset_root, info, item["old_index"])
        if not data_file.exists():
            raise FileNotFoundError(f"保留的 episode 文件不存在，无法修复: {data_file}")
        for video_key in video_keys:
            video_file = get_video_file_path(dataset_root, info, item["old_index"], video_key)
            if not video_file.exists():
                raise FileNotFoundError(f"保留的 video 文件不存在，无法修复: {video_file}")

    if args.dry_run:
        print("\n[dry-run] 删除后统计:")
        print(f"  total_episodes: {len(new_episodes_rows)}")
        print(f"  total_frames: {new_total_frames}")
        print(f"  total_videos: {len(new_episodes_rows) * len(video_keys)}")
        print(f"  total_chunks: {math.ceil(len(new_episodes_rows) / get_chunk_size(info)) if new_episodes_rows else 0}")
        return

    backup_dir = backup_meta(dataset_root)
    print(f"\n已备份 meta 到: {backup_dir}")

    for old_index in delete_indices:
        data_file = get_data_file_path(dataset_root, info, old_index)
        if data_file.exists():
            print(f"删除 parquet: {data_file}")
            data_file.unlink()
        for video_key in video_keys:
            video_file = get_video_file_path(dataset_root, info, old_index, video_key)
            if video_file.exists():
                print(f"删除 video: {video_file}")
                video_file.unlink()

    for item in kept_plan:
        old_index = item["old_index"]
        new_index = item["new_index"]
        if old_index == new_index:
            continue

        src = get_data_file_path(dataset_root, info, old_index)
        dest = get_data_file_path(dataset_root, info, new_index)
        print(f"重写 parquet: episode {old_index} -> {new_index}")
        rewrite_parquet_episode(
            src=src,
            dest=dest,
            new_episode_index=new_index,
            new_index_start=item["new_index_start"],
            expected_length=item["length"],
        )

        for video_key in video_keys:
            video_src = get_video_file_path(dataset_root, info, old_index, video_key)
            video_dest = get_video_file_path(dataset_root, info, new_index, video_key)
            print(f"重命名 video: {video_src.name} -> {video_dest.name}")
            rename_video_episode(video_src, video_dest)

    new_info = copy.deepcopy(info)
    new_total_episodes = len(new_episodes_rows)
    new_info["total_episodes"] = new_total_episodes
    new_info["total_frames"] = new_total_frames
    new_info["total_chunks"] = math.ceil(new_total_episodes / get_chunk_size(info)) if new_total_episodes else 0
    new_info["total_videos"] = new_total_episodes * len(video_keys)
    new_info["splits"] = {"train": f"0:{new_total_episodes}"}

    save_jsonl(episodes_path, new_episodes_rows)
    if stats_rows:
        save_jsonl(stats_path, new_stats_rows)
    save_json(info_path, new_info)

    cleanup_empty_chunk_dirs(dataset_root)

    print("\n✅ 修复完成")
    print(f"  total_episodes: {new_total_episodes}")
    print(f"  total_frames: {new_total_frames}")
    print(f"  total_videos: {new_info['total_videos']}")
    print(f"  total_chunks: {new_info['total_chunks']}")


if __name__ == "__main__":
    main()
