"""PR-AUC comparison across all models, with a 95% bootstrap CI error bar
wherever one exists (only the canonical AT-Time ranker has one so far, from
scripts/bootstrap_confidence.py). Models without a computed CI show a plain
point with a note in the caption, rather than a fake/zero error bar.

Usage:
    python scripts/experiments/plots/plot_model_comparison.py
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
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

JOURNEY = Path(r"D:\financial-prior-research-paper\journey")
OUT_DIR = JOURNEY / "07_statistical_confidence" / "figures"


def main():
    comparison = pd.read_csv(JOURNEY / "01_model_and_baseline_search" / "data" / "model_comparison_metrics.csv")
    ci = pd.read_csv(JOURNEY / "07_statistical_confidence" / "data" / "bootstrap_ci_metrics.csv")

    at_time_ci = ci[ci["metric"] == "03_canonical_ranker.auc_pr"].iloc[0]

    comparison = comparison.sort_values("auc_pr", ascending=True).reset_index(drop=True)
    has_ci = comparison["model"] == "AT-Time"
    err_lower = (comparison["auc_pr"] - at_time_ci["ci_low_2.5pct"]).where(has_ci, 0)
    err_upper = (at_time_ci["ci_high_97.5pct"] - comparison["auc_pr"]).where(has_ci, 0)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    fig.patch.set_facecolor(BACKGROUND)

    y = range(len(comparison))
    colors = [BLUE if m == "AT-Time" else MUTED for m in comparison["model"]]
    ax.barh(y, comparison["auc_pr"], color=colors, height=0.6, zorder=3)
    ax.errorbar(
        comparison["auc_pr"], y,
        xerr=[err_lower, err_upper],
        fmt="none", ecolor=INK, elinewidth=1.5, capsize=4, zorder=4,
    )

    ax.set_yticks(list(y))
    ax.set_yticklabels(comparison["model"], color=INK, fontsize=10)
    ax.set_xlabel("PR-AUC (test, 2023-2024)", color=SECONDARY_INK)
    ax.set_title(
        "Model comparison: PR-AUC on the canonical evaluation\n"
        "(error bar = 95% ticker-clustered bootstrap CI, AT-Time only)",
        color=INK,
        fontsize=11,
        loc="left",
    )
    style_axis(ax, "x")

    for i, (v, hi) in enumerate(zip(comparison["auc_pr"], err_upper)):
        label_x = v + hi + 0.012 if hi > 0 else v + 0.012
        ax.text(label_x, i, f"{v:.3f}", va="center", fontsize=9, color=SECONDARY_INK)

    ax.text(0.98, 0.03,
            "Only AT-Time has a computed bootstrap CI so far — other bars are\n"
            "single point estimates, not yet uncertainty-quantified.",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=MUTED, style="italic")

    fig.tight_layout()
    out_path = OUT_DIR / "model_comparison_with_ci.png"
    save_figure(fig, out_path)


if __name__ == "__main__":
    main()
