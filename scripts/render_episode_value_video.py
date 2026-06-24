#!/usr/bin/env python3
"""Render side-camera episode videos with predicted and target Value curves."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


CANVAS_WIDTH = 1400
CANVAS_HEIGHT = 720
IMAGE_PANEL = (30, 80, 790, 650)
PLOT_PANEL = (850, 110, 1360, 570)
PREDICTED_COLOR = (210, 120, 40)
TARGET_COLOR = (70, 170, 70)
CURRENT_COLOR = (45, 45, 220)
GRID_COLOR = (220, 220, 220)
TEXT_COLOR = (35, 35, 35)


def _decode_image(value: Any, dataset_root: Path) -> np.ndarray:
    if isinstance(value, dict):
        encoded = value.get("bytes")
        path = value.get("path")
        if encoded is None and path:
            encoded = (dataset_root / path).read_bytes()
    elif isinstance(value, (bytes, bytearray, memoryview)):
        encoded = bytes(value)
    else:
        encoded = None

    if encoded is None:
        raise ValueError(f"Unsupported camera value: {type(value)!r}")
    image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode camera image")
    return image


def _fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
        interpolation=cv2.INTER_AREA,
    )
    panel = np.full((height, width, 3), 245, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    panel[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return panel


def _plot_point(frame_index: int, value: float, max_frame: int) -> tuple[int, int]:
    left, top, right, bottom = PLOT_PANEL
    x = left + round(frame_index / max(max_frame, 1) * (right - left))
    clipped = float(np.clip(value, -1.0, 0.0))
    y = bottom - round((clipped + 1.0) * (bottom - top))
    return x, y


def _draw_dashed_polyline(canvas: np.ndarray, points: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    for index in range(len(points) - 1):
        if index % 2 == 0:
            cv2.line(canvas, points[index], points[index + 1], color, 2, cv2.LINE_AA)


def _draw_curve_panel(
    canvas: np.ndarray,
    frame_indices: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
    current_position: int,
) -> None:
    left, top, right, bottom = PLOT_PANEL
    cv2.rectangle(canvas, (left, top), (right, bottom), (250, 250, 250), thickness=-1)
    max_frame = int(frame_indices[-1])

    for value in np.linspace(-1.0, 0.0, 6):
        _, y = _plot_point(0, float(value), max_frame)
        cv2.line(canvas, (left, y), (right, y), GRID_COLOR, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{value:.1f}", (left - 48, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_COLOR, 1)
    for ratio in np.linspace(0.0, 1.0, 5):
        x = left + round(ratio * (right - left))
        cv2.line(canvas, (x, top), (x, bottom), GRID_COLOR, 1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"{ratio * max_frame:.0f}",
            (x - 14, bottom + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            TEXT_COLOR,
            1,
        )

    target_points = [
        _plot_point(int(frame), float(value), max_frame)
        for frame, value in zip(frame_indices, targets, strict=True)
    ]
    predicted_points = [
        _plot_point(int(frame), float(value), max_frame)
        for frame, value in zip(frame_indices, predictions, strict=True)
    ]
    _draw_dashed_polyline(canvas, target_points, TARGET_COLOR)
    if current_position > 0:
        cv2.polylines(
            canvas,
            [np.asarray(predicted_points[: current_position + 1], dtype=np.int32)],
            isClosed=False,
            color=PREDICTED_COLOR,
            thickness=3,
            lineType=cv2.LINE_AA,
        )
    current_point = predicted_points[current_position]
    cv2.circle(canvas, current_point, 7, CURRENT_COLOR, thickness=-1, lineType=cv2.LINE_AA)
    cv2.line(canvas, (current_point[0], top), (current_point[0], bottom), (175, 175, 175), 1, cv2.LINE_AA)

    cv2.rectangle(canvas, (left, top), (right, bottom), (90, 90, 90), 1)
    cv2.putText(canvas, "Frame", (left + 225, bottom + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
    cv2.putText(canvas, "Value", (left - 58, top - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)
    cv2.line(canvas, (left + 12, top - 40), (left + 48, top - 40), PREDICTED_COLOR, 3, cv2.LINE_AA)
    cv2.putText(canvas, "predicted", (left + 56, top - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_COLOR, 1)
    cv2.line(canvas, (left + 170, top - 40), (left + 206, top - 40), TARGET_COLOR, 2, cv2.LINE_AA)
    cv2.putText(canvas, "target", (left + 214, top - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.48, TEXT_COLOR, 1)


def _dataset_fps(parquet_path: Path) -> float:
    dataset_root = parquet_path.parents[2]
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return 30.0
    with info_path.open("r", encoding="utf-8") as file:
        return float(json.load(file).get("fps", 30.0))


def _render_episode(
    values: pd.DataFrame,
    *,
    camera_column: str,
    output_path: Path,
    fps: float | None,
    stride: int,
) -> None:
    values = values.sort_values("frame_index").reset_index(drop=True)
    episode_index = int(values["episode_index"].iloc[0])
    parquet_path = Path(values["parquet_path"].iloc[0])
    dataset_root = parquet_path.parents[2]
    table = pq.read_table(parquet_path, columns=["frame_index", camera_column])
    source_frames = np.asarray(table.column("frame_index").combine_chunks(), dtype=np.int64)
    camera_values = table.column(camera_column).combine_chunks()
    source_row_by_frame = {int(frame): index for index, frame in enumerate(source_frames.tolist())}

    missing = sorted(set(values["frame_index"].astype(int)) - set(source_row_by_frame))
    if missing:
        raise ValueError(f"Episode {episode_index} is missing camera frames: {missing[:10]}")

    frame_indices = values["frame_index"].to_numpy(dtype=np.int64)
    predictions = values["vlm_value"].to_numpy(dtype=np.float32)
    targets = values["value_label"].to_numpy(dtype=np.float32)
    outcome = "success" if float(targets[-1]) > -0.5 else "failure"
    output_fps = float(fps if fps is not None else _dataset_fps(parquet_path)) / stride
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (CANVAS_WIDTH, CANVAS_HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video: {output_path}")

    try:
        for position in range(0, len(values), stride):
            frame_index = int(frame_indices[position])
            camera_value = camera_values[source_row_by_frame[frame_index]].as_py()
            image = _decode_image(camera_value, dataset_root)
            canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), 255, dtype=np.uint8)
            x0, y0, x1, y1 = IMAGE_PANEL
            canvas[y0:y1, x0:x1] = _fit_image(image, x1 - x0, y1 - y0)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (90, 90, 90), 1)
            _draw_curve_panel(canvas, frame_indices, predictions, targets, position)

            cv2.putText(
                canvas,
                f"Episode {episode_index} | {outcome} | {camera_column}",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"frame {frame_index}/{int(frame_indices[-1])}   predicted {predictions[position]:.3f}   target {targets[position]:.3f}",
                (850, 670),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
            writer.write(canvas)
    finally:
        writer.release()

    print(f"[ok] episode={episode_index} outcome={outcome} frames={len(values)} output={output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--values_path", type=Path, required=True, help="Frame-level export from export_vlm_values.py.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for one MP4 per episode.")
    parser.add_argument("--camera_column", default="side_image", help="Camera column to render.")
    parser.add_argument("--fps", type=float, default=None, help="Input FPS override; defaults to dataset metadata.")
    parser.add_argument("--stride", type=int, default=1, help="Render every Nth frame and adjust output FPS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    values = pd.read_parquet(args.values_path)
    required = {"episode_index", "frame_index", "value_label", "vlm_value", "parquet_path"}
    missing = required - set(values.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    for episode_index, episode_values in values.groupby("episode_index", sort=True):
        outcome = "success" if float(episode_values.sort_values("frame_index")["value_label"].iloc[-1]) > -0.5 else "failure"
        output_path = args.output_dir / f"episode_{int(episode_index):06d}_{outcome}_{args.camera_column}_value.mp4"
        _render_episode(
            episode_values,
            camera_column=args.camera_column,
            output_path=output_path,
            fps=args.fps,
            stride=args.stride,
        )


if __name__ == "__main__":
    main()
