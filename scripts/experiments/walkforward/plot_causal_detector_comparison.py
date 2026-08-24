"""Plot frozen-vs-selected and Q99-vs-causal walk-forward comparisons.

Usage:
    python scripts/experiments/walkforward/plot_causal_detector_comparison.py
    python scripts/experiments/walkforward/plot_causal_detector_comparison.py --comparison selection
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments._plot import (  # noqa: E402
    BACKGROUND,
    BLUE,
    INK,
    MUTED,
    SECONDARY_INK,
    save_figure,
    style_axis,
)


DATA_DIR = Path(r"D:\financial-prior-research-paper\journey\08_walkforward_multiseed\data")
OUT_DIR = DATA_DIR.parent / "figures"
FOLD_LABELS = {
    "A": "Fold A\n(test 2021)",
    "B": "Fold B\n(test 2022)",
    "C": "Fold C\n(test 2023-24, current)",
}


def fold_stats(frame, metric):
    return frame.groupby("fold")[metric].agg(["mean", "std"]).reindex(FOLD_LABELS)


def plot_comparison(left, right, labels, title, output):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor(BACKGROUND)
    x = np.arange(len(FOLD_LABELS))
    width = 0.32
    for offset, stats, color, label in (
        (-width / 2, left, MUTED, labels[0]),
        (width / 2, right, BLUE, labels[1]),
    ):
        ax.bar(
            x + offset,
            stats["mean"],
            width,
            yerr=stats["std"],
            color=color,
            label=label,
            capsize=4,
            error_kw={"ecolor": INK, "elinewidth": 1.3},
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(FOLD_LABELS.values(), fontsize=9, color=SECONDARY_INK)
    ax.set_ylabel("Event-F1 (point = mean of 3 seeds, error bar = std)", color=SECONDARY_INK)
    ax.set_title(title, color=INK, fontsize=10.5, loc="left")
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    style_axis(ax, "y")
    fig.tight_layout()
    save_figure(fig, OUT_DIR / output)


def plot_threshold_comparison():
    q99 = pd.DataFrame(json.loads((DATA_DIR / "walkforward_multiseed_metrics.json").read_text()))
    causal = pd.read_csv(DATA_DIR / "causal_detector_by_fold_seed.csv")
    q99_stats = fold_stats(q99, "event_f1")
    causal_stats = fold_stats(causal[causal["method"] == "behavior_filtered"], "f1")
    plot_comparison(
        q99_stats,
        causal_stats,
        ("Naive Q99 threshold", "Causal local-peak + behavior detector"),
        "Detection method matters more than the apparent regime gap",
        "q99_vs_causal_detector_by_fold.png",
    )
    summary = pd.DataFrame({
        "fold": list(FOLD_LABELS),
        "q99_event_f1_mean": q99_stats["mean"].values,
        "q99_event_f1_std": q99_stats["std"].values,
        "causal_f1_mean": causal_stats["mean"].values,
        "causal_f1_std": causal_stats["std"].values,
    })
    summary_path = DATA_DIR / "q99_vs_causal_by_fold_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")


def plot_selection_comparison():
    reused = pd.read_csv(DATA_DIR / "causal_detector_by_fold_seed.csv")
    selected = pd.read_csv(DATA_DIR / "causal_detector_reselected_per_run.csv")
    plot_comparison(
        fold_stats(reused[reused["method"] == "behavior_filtered"], "f1"),
        fold_stats(selected[selected["method"] == "behavior_filtered"], "f1"),
        ("Reused frozen parameters", "Selected on each run's validation set"),
        "Effect of selecting detector parameters separately for each run",
        "reused_vs_reselected_params_by_fold.png",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", choices=("all", "threshold", "selection"), default="all")
    args = parser.parse_args()
    if args.comparison in ("all", "threshold"):
        plot_threshold_comparison()
    if args.comparison in ("all", "selection"):
        plot_selection_comparison()


if __name__ == "__main__":
    main()
