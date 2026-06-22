"""Export VLM value predictions for plotting and diagnostics.

This script runs the trained VLM value model over a LeRobot dataset and writes a
frame-level table containing predicted values plus common metadata columns. It
does not modify the dataset parquet files.

Example:
  python scripts/export_vlm_values.py \
    --data_dir /path/to/lerobot_dataset \
    --checkpoint_dir /path/to/value_checkpoints \
    --output_path /tmp/vlm_values.parquet \
    --tokenizer_path /path/to/tokenizer.model
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from label_advantage_from_vlm import IndexedDataset
from label_advantage_from_vlm import _build_inference_dataset
from label_advantage_from_vlm import _compute_values_with_dataloader
from label_advantage_from_vlm import _extract_human_mask
from label_advantage_from_vlm import _load_checkpoint_params
from label_advantage_from_vlm import _load_tasks_map
from label_advantage_from_vlm import _lookup_inferred_value
from label_advantage_from_vlm import _read_parquet_columns
from label_advantage_from_vlm import _resolve_checkpoint_path
from label_advantage_from_vlm import _resolve_dataset_row_indices
from label_advantage_from_vlm import _resolve_local_gemma_tokenizer_path
from label_advantage_from_vlm import _resolve_step_indices
from label_advantage_from_vlm import _select_reward_column
from label_advantage_from_vlm import _series_to_scalar_float_array
from label_advantage_from_vlm import _series_to_scalar_int_array
from label_advantage_from_vlm import _to_scalar_float
from label_advantage_from_vlm import _to_scalar_int
from label_advantage_from_vlm import _to_scalar_str
from label_advantage_from_vlm import _validate_gemma_tokenizer
from openpi.shared import console


LOG = logging.getLogger("openpi")

VALUE_COLUMN_CANDIDATES = ("value_label", "value_lable")
OPTIONAL_SCALAR_COLUMNS = (
    "episode_index",
    "frame_index",
    "index",
    "task_index",
    "reward",
    "reward_label",
    "value_label",
    "value_lable",
    "intervention",
    "adv_ind",
)


@dataclasses.dataclass
class EpisodeExportRequest:
    parquet_path: Path
    episode_id: int
    step_indices: np.ndarray
    dataset_row_indices: np.ndarray
    df: pd.DataFrame


def _extract_episode_id(df: pd.DataFrame, fallback: int) -> int:
    if "episode_index" not in df.columns or df.empty:
        return fallback
    return _to_scalar_int(df["episode_index"].iloc[0])


def _select_value_column(df: pd.DataFrame, specified: str | None) -> str | None:
    if specified:
        if specified not in df.columns:
            raise ValueError(f"Value column '{specified}' not found. Available columns: {list(df.columns)}")
        return specified

    for candidate in VALUE_COLUMN_CANDIDATES:
        if candidate in df.columns:
            return candidate
    return None


def _optional_float_array(df: pd.DataFrame, column: str | None) -> np.ndarray:
    if not column or column not in df.columns:
        return np.full((len(df),), np.nan, dtype=np.float32)
    return _series_to_scalar_float_array(df[column], name=column)


def _optional_int_array(df: pd.DataFrame, column: str | None) -> np.ndarray:
    if not column or column not in df.columns:
        return np.full((len(df),), -1, dtype=np.int64)
    return _series_to_scalar_int_array(df[column], name=column)


def _optional_str_array(df: pd.DataFrame, column: str | None) -> list[str]:
    if not column or column not in df.columns:
        return [""] * len(df)
    return [_to_scalar_str(value, name=column) for value in df[column].tolist()]


def _compute_mc_return(reward_labels: np.ndarray) -> np.ndarray:
    if reward_labels.size == 0 or np.all(np.isnan(reward_labels)):
        return np.full_like(reward_labels, np.nan, dtype=np.float32)
    return np.cumsum(reward_labels[::-1], dtype=np.float32)[::-1]


def _compute_oracle_advantage(values: np.ndarray, reward_labels: np.ndarray, lookahead: int) -> np.ndarray:
    if lookahead <= 0 or values.size == 0 or np.all(np.isnan(values)) or np.all(np.isnan(reward_labels)):
        return np.full_like(values, np.nan, dtype=np.float32)

    advantages = np.zeros((len(values),), dtype=np.float32)
    for t in range(len(values)):
        reward_end = min(t + lookahead, len(reward_labels))
        reward_sum = float(np.nansum(reward_labels[t:reward_end]))
        future_value = float(values[t + lookahead]) if t + lookahead < len(values) else 0.0
        advantages[t] = reward_sum + future_value - float(values[t])
    return advantages


def _build_export_requests(
    parquet_files: list[Path],
    *,
    reward_col: str | None,
    value_col: str | None,
    human_col: str | None,
    adv_col: str | None,
) -> tuple[list[EpisodeExportRequest], list[int], int]:
    requests: list[EpisodeExportRequest] = []
    selected_dataset_indices: list[int] = []
    flat_offset = 0
    total_rows = 0

    for fallback_episode_id, parquet_path in enumerate(
        tqdm(parquet_files, desc="扫描全部帧", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
    ):
        requested_columns = list(
            dict.fromkeys(
                [
                    *OPTIONAL_SCALAR_COLUMNS,
                    reward_col,
                    value_col,
                    human_col,
                    adv_col,
                ]
            )
        )
        df = _read_parquet_columns(parquet_path, requested_columns)
        total_rows += len(df)

        episode_id = _extract_episode_id(df, fallback_episode_id)
        step_indices = _resolve_step_indices(df)
        dataset_row_indices = _resolve_dataset_row_indices(df, flat_offset)

        requests.append(
            EpisodeExportRequest(
                parquet_path=parquet_path,
                episode_id=episode_id,
                step_indices=step_indices,
                dataset_row_indices=dataset_row_indices,
                df=df,
            )
        )
        selected_dataset_indices.extend(dataset_row_indices.tolist())
        flat_offset += len(df)

    return requests, selected_dataset_indices, total_rows


def _resolve_values_for_request(
    *,
    request: EpisodeExportRequest,
    value_cache,
    flat_values_by_dataset_index: dict[int, float],
) -> np.ndarray:
    values = np.zeros((len(request.df),), dtype=np.float32)
    missing_by_frame = False
    for i, step_index in enumerate(request.step_indices.tolist()):
        try:
            values[i] = np.float32(
                _lookup_inferred_value(value_cache, episode_id=request.episode_id, step_index=int(step_index))
            )
        except KeyError:
            missing_by_frame = True
            break

    if not missing_by_frame:
        return values

    for i, dataset_index in enumerate(request.dataset_row_indices.tolist()):
        if int(dataset_index) not in flat_values_by_dataset_index:
            raise KeyError(
                f"Missing inferred value for dataset index {dataset_index}; "
                "check that metadata indices match the transformed inference dataset."
            )
        values[i] = np.float32(flat_values_by_dataset_index[int(dataset_index)])
    return values


def _write_output(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(rows)

    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        output.to_csv(output_path, index=False)
    elif suffix in (".jsonl", ".json"):
        output.to_json(output_path, orient="records", lines=True)
    else:
        output.to_parquet(output_path, index=False)


def main() -> None:
    import jax
    import jax.numpy as jnp
    from openpi.models.value_model_config import ValueModelConfig

    parser = argparse.ArgumentParser(description="Export VLM value predictions for LeRobot frames")
    parser.add_argument("--data_dir", type=str, required=True, help="LeRobot dataset path")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Checkpoint directory containing step_*")
    parser.add_argument("--output_path", type=str, required=True, help="Output .parquet, .csv, or .jsonl path")
    parser.add_argument("--checkpoint_name", type=str, default=None, help="Specific checkpoint folder name")
    parser.add_argument("--use_ema", action=argparse.BooleanOptionalAction, default=True, help="Use EMA params if present")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Optional local Gemma3 tokenizer.model path")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for value inference")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader worker count")
    parser.add_argument("--decode_workers", type=int, default=None, help="Legacy alias for --num_workers")
    parser.add_argument("--seed", type=int, default=42, help="DataLoader seed")
    parser.add_argument("--reward_col", type=str, default="reward_label", help="Reward-label column")
    parser.add_argument("--value_col", type=str, default=None, help="Value-label column; auto-detects value_label/value_lable")
    parser.add_argument("--human_col", type=str, default="intervention", help="Intervention column")
    parser.add_argument("--adv_col", type=str, default="adv_ind", help="Advantage label column")
    parser.add_argument("--instruction_col", type=str, default=None, help="Instruction/prompt column")
    parser.add_argument("--base_image_col", type=str, default="image", help="Base image column")
    parser.add_argument("--wrist_image_col", type=str, default="wrist_image", help="Wrist image column")
    parser.add_argument(
        "--right_wrist_image_col",
        type=str,
        default="side_image",
        help="Third-view image column (default: side_image)",
    )
    parser.add_argument("--copy_wrist_to_right", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--lookahead", type=int, default=50, help="Optional oracle advantage horizon for exported metadata")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional debug cap; omit it to export every frame")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if cache_dir:
        jax.config.update("jax_compilation_cache_dir", cache_dir)

    data_dir = Path(args.data_dir)
    parquet_dir = data_dir / "data"
    if not parquet_dir.exists():
        raise ValueError(f"Cannot find data directory: {parquet_dir}")
    parquet_files = sorted(parquet_dir.rglob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found under: {parquet_dir}")

    requests, selected_dataset_indices, total_rows = _build_export_requests(
        parquet_files,
        reward_col=args.reward_col,
        value_col=args.value_col,
        human_col=args.human_col,
        adv_col=args.adv_col,
    )
    if args.max_frames is not None:
        if args.max_frames <= 0:
            raise ValueError("--max_frames must be positive")
        selected_dataset_indices = selected_dataset_indices[: args.max_frames]
        kept = set(selected_dataset_indices)
        trimmed_requests = []
        for request in requests:
            mask = np.asarray([int(idx) in kept for idx in request.dataset_row_indices], dtype=bool)
            if not np.any(mask):
                continue
            trimmed_requests.append(
                EpisodeExportRequest(
                    parquet_path=request.parquet_path,
                    episode_id=request.episode_id,
                    step_indices=request.step_indices[mask],
                    dataset_row_indices=request.dataset_row_indices[mask],
                    df=request.df.iloc[np.flatnonzero(mask)].reset_index(drop=True),
                )
            )
        requests = trimmed_requests

    LOG.info("Dataset frames: %d; frames scheduled for inference/export: %d", total_rows, len(selected_dataset_indices))

    checkpoint_path = _resolve_checkpoint_path(Path(args.checkpoint_dir), args.checkpoint_name)
    LOG.info("Using checkpoint: %s", checkpoint_path)

    config = ValueModelConfig()
    params = _load_checkpoint_params(checkpoint_path, use_ema=args.use_ema)
    model = config.load(params, remove_extra_params=True)
    supports = jnp.linspace(-1.0, 0.0, 201, dtype=jnp.float32)

    max_workers = args.num_workers
    if max_workers is None:
        max_workers = args.decode_workers if args.decode_workers is not None else 2
    max_workers = max(0, max_workers)

    resolved_tokenizer_path = _resolve_local_gemma_tokenizer_path(args.tokenizer_path)
    if resolved_tokenizer_path is not None:
        LOG.info(console.info(f"Using local Gemma3 tokenizer: {resolved_tokenizer_path}"))
    elif max_workers > 0:
        LOG.warning(console.warn("No local Gemma3 tokenizer found; setting num_workers=0."))
        max_workers = 0
    _validate_gemma_tokenizer(resolved_tokenizer_path)

    tasks_map = _load_tasks_map(data_dir) if (data_dir / "meta" / "tasks.jsonl").exists() else None
    dataset = _build_inference_dataset(
        data_dir=data_dir,
        model_config=config,
        tokenizer_path=resolved_tokenizer_path,
        instruction_col=args.instruction_col,
        base_image_col=args.base_image_col,
        wrist_image_col=args.wrist_image_col,
        right_wrist_image_col=args.right_wrist_image_col,
        copy_wrist_to_right=args.copy_wrist_to_right,
        tasks_map=tasks_map,
    )
    dataset = IndexedDataset(dataset, selected_dataset_indices)

    value_cache = _compute_values_with_dataloader(
        dataset=dataset,
        model=model,
        supports=supports,
        batch_size=args.batch_size,
        num_workers=max_workers,
        seed=args.seed,
    )
    flat_values_by_dataset_index = {
        int(index): float(value)
        for index, value in zip(selected_dataset_indices, value_cache.flat_values, strict=True)
    }

    rows: list[dict[str, Any]] = []
    for request in tqdm(
        requests,
        desc="整理导出表",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        df = request.df
        pred_values = _resolve_values_for_request(
            request=request,
            value_cache=value_cache,
            flat_values_by_dataset_index=flat_values_by_dataset_index,
        )

        reward_col = _select_reward_column(df, args.reward_col) if args.reward_col in df.columns else None
        value_col = _select_value_column(df, args.value_col)
        reward_labels = _optional_float_array(df, reward_col)
        value_labels = _optional_float_array(df, value_col)
        rewards = _optional_float_array(df, "reward")
        interventions = _extract_human_mask(df, args.human_col).astype(np.int64)
        task_indices = _optional_int_array(df, "task_index")
        frame_indices = request.step_indices.astype(np.int64)
        dataset_indices = request.dataset_row_indices.astype(np.int64)
        adv_labels = _optional_str_array(df, args.adv_col)
        mc_returns = _compute_mc_return(reward_labels)
        oracle_advantages = _compute_oracle_advantage(value_labels, reward_labels, args.lookahead)

        for i in range(len(df)):
            task_index = int(task_indices[i])
            rows.append(
                {
                    "dataset_index": int(dataset_indices[i]),
                    "episode_index": int(request.episode_id),
                    "frame_index": int(frame_indices[i]),
                    "task_index": task_index,
                    "task": tasks_map.get(task_index, "") if tasks_map else "",
                    "intervention": int(interventions[i]),
                    "adv_ind": adv_labels[i],
                    "reward": float(rewards[i]),
                    "reward_label": float(reward_labels[i]),
                    "value_label": float(value_labels[i]),
                    "vlm_value": float(pred_values[i]),
                    "mc_return": float(mc_returns[i]),
                    "value_error": float(pred_values[i] - value_labels[i]),
                    "abs_value_error": float(abs(pred_values[i] - value_labels[i])),
                    f"oracle_adv_h{args.lookahead}": float(oracle_advantages[i]),
                    "parquet_path": str(request.parquet_path),
                }
            )

    output_path = Path(args.output_path)
    _write_output(rows, output_path)
    print(console.ok(f"完成: 导出 {len(rows)} 帧 VLM value 到 {output_path}"))


if __name__ == "__main__":
    main()
