"""Plot the canonical test distribution without transforming the AT scores."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path(r"D:\multi-prior-at-run-recon3-contextual\AT-Time\test_timeline_scores.csv")
OUT = ROOT / "research_paper/results/canonical_score_distribution"


def main():
    frame = pd.read_csv(SOURCE, usecols=["ticker", "date", "score"])
    scores = frame["score"].dropna().astype(float)
    positive = scores[scores > 0]
    quantiles = scores.quantile([0.5, 0.95, 0.99])

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.geomspace(positive.min(), positive.max(), 100)
    ax.hist(positive, bins=bins, color="#4c78a8", alpha=0.85, edgecolor="white", linewidth=0.3)
    for q, color in zip((0.5, 0.95, 0.99), ("#54a24b", "#f58518", "#e45756")):
        value = quantiles[q]
        ax.axvline(value, color=color, linestyle="--", linewidth=1.5, label=f"Q{q * 100:g} = {value:.6g}")
    ax.set_xscale("log")
    ax.set(
        xlabel="Raw AT anomaly score (log-scaled axis)",
        ylabel="Ticker-day count",
        title=f"Canonical AT-Time raw test-score distribution\n{len(scores):,} ticker-days, {frame['ticker'].nunique()} tickers, 2023–2024",
    )
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "raw_score_distribution_all_tickers.png", dpi=180, facecolor="white")
    plt.close(fig)

    pd.DataFrame({
        "rows": [len(scores)], "tickers": [frame["ticker"].nunique()],
        "minimum": [scores.min()], "median": [quantiles[0.5]],
        "q95": [quantiles[0.95]], "q99": [quantiles[0.99]], "maximum": [scores.max()],
    }).to_csv(OUT / "raw_score_distribution_summary.csv", index=False)


if __name__ == "__main__":
    main()
