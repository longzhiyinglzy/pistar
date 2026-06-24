"""Plot diagnostics for frame-level VLM value exports."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GROUPS = (
    ("demo", "success", "Demo success", "#2878b5"),
    ("rollout", "success", "Rollout success", "#2a9d62"),
    ("rollout", "failure", "Rollout failure", "#d1495b"),
)


def _metrics(frame: pd.DataFrame) -> pd.Series:
    error = frame["vlm_value"] - frame["value_label"]
    return pd.Series(
        {
            "frames": len(frame),
            "mae": error.abs().mean(),
            "rmse": np.sqrt(np.mean(error**2)),
            "bias": error.mean(),
            "corr": frame["vlm_value"].corr(frame["value_label"]),
        }
    )


def _binary_auc(labels: pd.Series, scores: pd.Series) -> float:
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = scores.rank(method="average")
    rank_sum = float(ranks[labels].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _prepare(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "episode_index",
        "frame_index",
        "intervention",
        "value_label",
        "vlm_value",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = data.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    grouped = data.groupby("episode_index", sort=False)
    terminal = grouped.tail(1).copy().set_index("episode_index")
    terminal["outcome"] = np.where(terminal["value_label"] > -0.5, "success", "failure")

    episode_source = grouped["intervention"].all().map({True: "demo", False: "rollout"})
    terminal["source"] = episode_source
    data["source"] = data["episode_index"].map(episode_source)
    data["outcome"] = data["episode_index"].map(terminal["outcome"])

    episode_sizes = grouped.size()
    denominator = data["episode_index"].map((episode_sizes - 1).clip(lower=1))
    data["progress"] = data.groupby("episode_index").cumcount() / denominator
    return data, terminal.reset_index()


def _print_summary(data: pd.DataFrame, terminal: pd.DataFrame) -> None:
    print("\nFrame metrics")
    print(_metrics(data).to_frame("global").T.to_string(float_format=lambda value: f"{value:.4f}"))
    grouped_metrics = data.groupby(["source", "outcome"]).apply(_metrics, include_groups=False)
    print("\nMetrics by source/outcome")
    print(grouped_metrics.to_string(float_format=lambda value: f"{value:.4f}"))

    rollout_terminal = terminal[terminal["source"] == "rollout"]
    auc = _binary_auc(rollout_terminal["outcome"] == "success", rollout_terminal["vlm_value"])
    print(f"\nRollout terminal success AUC: {auc:.4f}")
    print("Rollout terminal predictions")
    print(
        rollout_terminal.groupby("outcome")["vlm_value"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .to_string(float_format=lambda value: f"{value:.4f}")
    )


def _plot(data: pd.DataFrame, terminal: pd.DataFrame, output_path: Path, progress_bins: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    ax = axes[0, 0]
    bins = np.minimum((data["progress"] * progress_bins).astype(int), progress_bins - 1)
    plot_data = data.assign(progress_bin=bins)
    for source, outcome, label, color in GROUPS:
        subset = plot_data[(plot_data["source"] == source) & (plot_data["outcome"] == outcome)]
        if subset.empty:
            continue
        trajectory = subset.groupby("progress_bin")[["vlm_value", "value_label"]].mean()
        x = (trajectory.index.to_numpy() + 0.5) / progress_bins
        ax.plot(x, trajectory["vlm_value"], color=color, linewidth=2.2, label=f"{label} predicted")
        ax.plot(x, trajectory["value_label"], color=color, linestyle="--", alpha=0.65, label=f"{label} target")
    ax.set(title="Mean value over episode progress", xlabel="Relative episode progress", ylabel="Value")
    ax.set_ylim(-1.05, 0.05)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[0, 1]
    hb = ax.hexbin(
        data["value_label"],
        data["vlm_value"],
        gridsize=55,
        extent=(-1, 0, -1, 0),
        mincnt=1,
        bins="log",
        cmap="viridis",
    )
    ax.plot([-1, 0], [-1, 0], color="#d1495b", linestyle="--", linewidth=1.5)
    ax.set(title="Frame predictions vs targets", xlabel="Value label", ylabel="Predicted value")
    fig.colorbar(hb, ax=ax, label="Log frame count")

    ax = axes[1, 0]
    rollout_terminal = terminal[terminal["source"] == "rollout"]
    hist_bins = np.linspace(-1.0, 0.0, 31)
    for outcome, label, color in (
        ("success", "Rollout success", "#2a9d62"),
        ("failure", "Rollout failure", "#d1495b"),
    ):
        values = rollout_terminal.loc[rollout_terminal["outcome"] == outcome, "vlm_value"]
        ax.hist(values, bins=hist_bins, density=True, alpha=0.55, color=color, label=label)
    ax.set(title="Rollout terminal value separation", xlabel="Terminal predicted value", ylabel="Density")
    ax.legend()
    ax.grid(alpha=0.2)

    ax = axes[1, 1]
    episode_mae = (
        data.assign(abs_error=(data["vlm_value"] - data["value_label"]).abs())
        .groupby(["episode_index", "source", "outcome"], as_index=False)["abs_error"]
        .mean()
    )
    box_values = []
    box_labels = []
    box_colors = []
    for source, outcome, label, color in GROUPS:
        values = episode_mae.loc[
            (episode_mae["source"] == source) & (episode_mae["outcome"] == outcome),
            "abs_error",
        ].to_numpy()
        if values.size:
            box_values.append(values)
            box_labels.append(label.replace(" ", "\n"))
            box_colors.append(color)
    boxes = ax.boxplot(box_values, tick_labels=box_labels, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], box_colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set(title="Episode-level mean absolute error", ylabel="Episode MAE")
    ax.grid(axis="y", alpha=0.2)

    fig.suptitle("VLM Value Function Diagnostics", fontsize=16)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot VLM value-function diagnostics")
    parser.add_argument("--input_path", type=Path, required=True, help="Parquet exported by export_vlm_values.py")
    parser.add_argument("--output_path", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--progress_bins", type=int, default=20)
    args = parser.parse_args()
    if args.progress_bins < 2:
        parser.error("--progress_bins must be at least 2")

    data, terminal = _prepare(pd.read_parquet(args.input_path))
    _print_summary(data, terminal)
    _plot(data, terminal, args.output_path, args.progress_bins)
    print(f"\nSaved diagnostic plot: {args.output_path}")


if __name__ == "__main__":
    main()
