"""Audit the canonical chronological split and its evaluation labels."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
CONFIG = Path(r"D:\multi-prior-at-run-recon3-contextual\AT-Time\config.json")
CACHE = ROOT / "SP500_features_vw60_lw60"
OUT = ROOT / "research_paper/results/training_split_audit"


def main():
    config = json.loads(CONFIG.read_text())
    frame = pd.concat(
        [pd.read_csv(path, usecols=["ticker", "date", "contextual_label"], parse_dates=["date"])
         for path in sorted(CACHE.glob("*_features.csv"))],
        ignore_index=True,
    )
    specs = [
        ("Train", config["train_start"], config["train_end"], "#4c78a8"),
        ("Validation", config["val_start"], config["val_end"], "#f58518"),
        ("Test", config["test_start"], config["test_end"], "#54a24b"),
    ]
    summaries = []
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1.6, 1]})
    daily = frame.groupby("date").size()
    axes[0].plot(daily.index, daily.values, color="0.25", linewidth=0.8, label="available ticker rows/day")
    for name, start, end, color in specs:
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        subset = frame[frame["date"].between(start, end)]
        sizes = subset.groupby("ticker").size()
        windows = (sizes - config["win_size"] + 1).clip(lower=0).sum()
        summaries.append({
            "split": name.lower(), "start": start.date(), "end": end.date(),
            "rows": len(subset), "tickers": subset["ticker"].nunique(),
            "windows": int(windows), "contextual_positive_rows": int(subset["contextual_label"].sum()),
            "contextual_positive_rate": subset["contextual_label"].mean(),
        })
        axes[0].axvspan(start, end, color=color, alpha=0.18, label=f"{name}: {start.year}–{end.year}")
    axes[0].set(ylabel="Ticker rows per trading day", title="Canonical chronological split audit (111 tickers)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncols=4, fontsize=9)

    summary = pd.DataFrame(summaries)
    bars = axes[1].bar(summary["split"], summary["windows"], color=[item[3] for item in specs])
    for bar, row in zip(bars, summary.itertuples()):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{row.windows:,} windows\ncontextual label={row.contextual_positive_rate:.2%}",
                     ha="center", va="bottom", fontsize=9)
    axes[1].set(ylabel="60-day windows", title="Labels are stored for evaluation; training loss does not consume them")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "canonical_training_split_audit.png", dpi=180, facecolor="white")
    plt.close(fig)
    summary.to_csv(OUT / "canonical_training_split_summary.csv", index=False)


if __name__ == "__main__":
    main()
