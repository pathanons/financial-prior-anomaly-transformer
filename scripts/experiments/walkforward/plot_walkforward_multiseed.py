"""Plot AT-Time PR-AUC and Event-F1 across the 3 walk-forward folds, with
error bars = standard deviation across the 3 seeds trained per fold. Answers:
is the canonical config's performance stable across different training/test
regimes, or does it swing depending on which years happen to be used?

Requires ``run_model_grid.py`` with the default AT-Time grid to finish first.

Usage:
    python scripts/experiments/walkforward/plot_walkforward_multiseed.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments._plot import (  # noqa: E402
    BACKGROUND,
    BLUE,
    INK,
    SECONDARY_INK,
    save_figure,
    style_axis,
)

DATA_PATH = Path(
    r"D:\financial-prior-research-paper\journey\08_walkforward_multiseed\data\walkforward_multiseed_metrics.json"
)
OUT_DIR = Path(r"D:\financial-prior-research-paper\journey\08_walkforward_multiseed\figures")

FOLD_LABELS = {
    "A": "Fold A\n(test 2021)",
    "B": "Fold B\n(test 2022)",
    "C": "Fold C\n(test 2023-24, current)",
}
METRICS = [("auc_pr", "PR-AUC"), ("event_f1", "Event-F1")]


def main():
    if not DATA_PATH.exists():
        raise SystemExit(f"{DATA_PATH} not found yet — run scripts/experiments/run_model_grid.py first")

    with open(DATA_PATH) as f:
        rows = json.load(f)
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(11, 5))
    fig.patch.set_facecolor(BACKGROUND)

    for ax, (col, title) in zip(axes, METRICS):
        grouped = df.groupby("fold")[col].agg(["mean", "std", "count"]).reindex(FOLD_LABELS.keys())

        x = range(len(grouped))
        ax.errorbar(
            x, grouped["mean"], yerr=grouped["std"],
            fmt="o", color=BLUE, ecolor=INK, elinewidth=1.5, capsize=5,
            markersize=8, zorder=3,
        )
        for xi, (mean, std, n) in enumerate(zip(grouped["mean"], grouped["std"], grouped["count"])):
            ax.text(xi + 0.08, mean, f"{mean:.3f} ± {std:.3f}\n(n={int(n)} seeds)",
                    va="center", fontsize=8.5, color=SECONDARY_INK)

        ax.set_xticks(list(x))
        ax.set_xticklabels([FOLD_LABELS[k] for k in grouped.index], fontsize=9, color=SECONDARY_INK)
        ax.set_title(title, color=INK, fontsize=11)
        style_axis(ax, "y")
        ax.set_xlim(-0.5, len(grouped) - 0.15)

    fig.suptitle("AT-Time across walk-forward folds (point = mean of 3 seeds, error bar = std)\n"
                 "Fold A: train 2018-19, Fold B: train 2018-20, Fold C: train 2018-21 (current canonical setup)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.90])

    out_path = OUT_DIR / "walkforward_multiseed_stability.png"
    save_figure(fig, out_path)

    summary_csv = OUT_DIR.parent / "data" / "walkforward_multiseed_summary.csv"
    df.groupby("fold")[[c for c, _ in METRICS]].agg(["mean", "std"]).to_csv(summary_csv)
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
