import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def safe_name(text):
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text))


def read_features(feature_dir, tickers, start, end):
    frames = []
    for ticker in tickers:
        path = Path(feature_dir) / f"{safe_name(ticker)}_features.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        frame["ticker"] = frame["ticker"].astype(str)
        frame = frame[(frame["date"] >= start) & (frame["date"] <= end)]
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No feature files found for selected tickers")
    return pd.concat(frames, ignore_index=True)


def add_by_ticker_predictions(scores, threshold_quantile):
    rows = scores.copy()
    rows["threshold"] = rows.groupby("ticker")["score"].transform(lambda s: s.quantile(threshold_quantile))
    rows["pred"] = (rows["score"] >= rows["threshold"]).astype(int)
    rows = rows.sort_values(["ticker", "date"]).copy()
    rows["raw_pred"] = rows["pred"]
    rows["pred"] = 0
    for _, group in rows.groupby("ticker", sort=False):
        raw = group["raw_pred"].astype(bool).to_numpy()
        starts = np.flatnonzero(raw & np.r_[True, ~raw[:-1]])
        ends = np.flatnonzero(raw & np.r_[~raw[1:], True])
        for start, end in zip(starts, ends):
            keep = group.iloc[start : end + 1]["score"].idxmax()
            rows.loc[keep, "pred"] = 1
    return rows


def choose_tickers(rows, n):
    by_label = (
        rows.groupby("ticker")
        .agg(labels=("event_label", "sum"), tp=("pred", lambda s: 0), max_score=("score", "max"))
        .reset_index()
    )
    tp_counts = rows[(rows["event_label"] == 1) & (rows["pred"] == 1)].groupby("ticker").size()
    fn_counts = rows[(rows["event_label"] == 1) & (rows["pred"] == 0)].groupby("ticker").size()
    fp_counts = rows[(rows["event_label"] == 0) & (rows["pred"] == 1)].groupby("ticker").size()
    by_label["tp"] = by_label["ticker"].map(tp_counts).fillna(0).astype(int)
    by_label["fn"] = by_label["ticker"].map(fn_counts).fillna(0).astype(int)
    by_label["fp"] = by_label["ticker"].map(fp_counts).fillna(0).astype(int)
    label_tickers = by_label.sort_values(["labels", "max_score"], ascending=False).head(n)["ticker"].tolist()
    score_tickers = rows.groupby("ticker")["score"].max().sort_values(ascending=False).head(n).index.tolist()
    return list(dict.fromkeys(label_tickers + score_tickers))


def add_extreme_labels(rows, std_multiplier, label_features):
    rows = rows.copy()
    for name, column in (("return", "log_return_1d"), ("volume", "volume_z"), ("gap", "gap")):
        rows[f"{name}_std"] = rows.groupby("ticker")[column].transform("std")
        rows[f"{name}_extreme"] = (rows[column].abs() > std_multiplier * rows[f"{name}_std"]).astype(int)
    rows["event_label"] = rows[[f"{name}_extreme" for name in label_features]].max(axis=1)
    return rows


def plot_ticker(frame, out_path, title, std_multiplier, label_features):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    dates = frame["date"]
    threshold = float(frame["threshold"].iloc[0])
    pred = frame["pred"].astype(bool)
    return_extreme = frame["return_extreme"].astype(bool)
    volume_extreme = frame["volume_extreme"].astype(bool)
    gap_extreme = frame["gap_extreme"].astype(bool)
    event = frame["event_label"].astype(bool)
    hit = pred & event
    score_only = pred & ~event
    missed_event = ~pred & event
    return_bound = float(std_multiplier * frame["return_std"].iloc[0])
    volume_bound = float(std_multiplier * frame["volume_std"].iloc[0])
    gap_bound = float(std_multiplier * frame["gap_std"].iloc[0])

    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [1.2, 1, 1, 1, 1]})
    ax = axes[0]
    ax.plot(dates, frame["close"], color="0.2", linewidth=1.1, label="close")
    ax.scatter(dates[hit], frame.loc[hit, "close"], color="#2ca02c", s=34, label="score+extreme")
    ax.scatter(dates[score_only], frame.loc[score_only, "close"], color="#d62728", marker="x", s=42, label="score only")
    ax.scatter(dates[missed_event], frame.loc[missed_event, "close"], facecolors="none", edgecolors="#ff7f0e", s=44, label="extreme only")
    ax.set_ylabel("close")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncols=4, fontsize=8)

    ax = axes[1]
    ax.plot(dates, frame["log_return_1d"], color="#1f77b4", linewidth=1.0, label="log_return_1d")
    ax.axhline(return_bound, color="#ff7f0e", linestyle="--", linewidth=1.0, label=f"+{std_multiplier:g}std")
    ax.axhline(-return_bound, color="#ff7f0e", linestyle="--", linewidth=1.0, label=f"-{std_multiplier:g}std")
    ax.scatter(dates[return_extreme], frame.loc[return_extreme, "log_return_1d"], color="#ff7f0e", s=28, label="return extreme")
    ax.set_ylabel("log return")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[2]
    ax.plot(dates, frame["volume_z"], color="#8c564b", linewidth=1.0, label="volume_z")
    if "volume" in label_features:
        ax.axhline(volume_bound, color="#d62728", linestyle="--", linewidth=1.0, label=f"+{std_multiplier:g}std")
        ax.axhline(-volume_bound, color="#d62728", linestyle="--", linewidth=1.0, label=f"-{std_multiplier:g}std")
        ax.scatter(dates[volume_extreme], frame.loc[volume_extreme, "volume_z"], color="#d62728", s=28, label="volume extreme")
    ax.set_ylabel("volume_z")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[3]
    ax.plot(dates, frame["gap"], color="#17becf", linewidth=1.0, label="gap")
    if "gap" in label_features:
        ax.axhline(gap_bound, color="#17becf", linestyle="--", linewidth=1.0, label=f"+{std_multiplier:g}std")
        ax.axhline(-gap_bound, color="#17becf", linestyle="--", linewidth=1.0, label=f"-{std_multiplier:g}std")
        ax.scatter(dates[gap_extreme], frame.loc[gap_extreme, "gap"], color="#17becf", s=28, label="gap extreme")
    ax.set_ylabel("gap")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[4]
    ax.plot(dates, frame["score"], color="#6f4e7c", linewidth=1.0, label="AT score")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.0, label="threshold")
    ax.scatter(dates[hit], frame.loc[hit, "score"], color="#2ca02c", s=34, label="score+extreme")
    ax.scatter(dates[score_only], frame.loc[score_only, "score"], color="#d62728", marker="x", s=42, label="score only")
    ax.scatter(dates[missed_event], frame.loc[missed_event, "score"], facecolors="none", edgecolors="#ff7f0e", s=44, label="extreme only")
    ax.set_ylabel("score")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncols=4, fontsize=8)

    for ax in axes:
        locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--feature_dir", default="SP500_features_vw60_lw60")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--top_n", type=int, default=12)
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--threshold_quantile", type=float, default=0.99)
    parser.add_argument("--include_tickers", default="")
    parser.add_argument("--title_prefix", default="AT-Time")
    parser.add_argument("--std_multiplier", type=float, default=3.0)
    parser.add_argument("--label_features", default="return,volume,gap",
                        help="Comma-separated extreme features used as ground truth: return,volume,gap")
    args = parser.parse_args()

    label_features = [name.strip() for name in args.label_features.split(",") if name.strip()]
    unknown = set(label_features) - {"return", "volume", "gap"}
    if not label_features or unknown:
        parser.error("--label_features must contain return, volume, and/or gap")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = pd.read_csv(args.scores, parse_dates=["date"])
    scores["ticker"] = scores["ticker"].astype(str)
    threshold_quantile = args.threshold_quantile

    scores = add_by_ticker_predictions(scores, threshold_quantile)
    features = read_features(args.feature_dir, scores["ticker"].unique(), args.start, args.end)
    merged = features.merge(scores, on=["ticker", "date"], how="inner")
    merged = add_extreme_labels(merged, args.std_multiplier, label_features)

    tickers = choose_tickers(merged, args.top_n)
    include_tickers = [ticker.strip() for ticker in args.include_tickers.split(",") if ticker.strip()]
    tickers = list(dict.fromkeys(tickers + include_tickers))
    features = read_features(args.feature_dir, tickers, args.start, args.end)
    merged = features.merge(scores, on=["ticker", "date"], how="inner")
    merged = add_extreme_labels(merged, args.std_multiplier, label_features)

    summary = (
        merged.groupby("ticker")
        .agg(
            rows=("date", "size"),
            labels=("event_label", "sum"),
            return_extremes=("return_extreme", "sum"),
            volume_extremes=("volume_extreme", "sum"),
            gap_extremes=("gap_extreme", "sum"),
            preds=("pred", "sum"),
            threshold=("threshold", "first"),
            max_score=("score", "max"),
            tp=("pred", lambda s: 0),
        )
        .reset_index()
    )
    summary["tp"] = summary["ticker"].map(merged[(merged["event_label"] == 1) & (merged["pred"] == 1)].groupby("ticker").size()).fillna(0).astype(int)
    summary["fp"] = summary["ticker"].map(merged[(merged["event_label"] == 0) & (merged["pred"] == 1)].groupby("ticker").size()).fillna(0).astype(int)
    summary["fn"] = summary["ticker"].map(merged[(merged["event_label"] == 1) & (merged["pred"] == 0)].groupby("ticker").size()).fillna(0).astype(int)
    summary["return_std"] = summary["ticker"].map(merged.groupby("ticker")["return_std"].first())
    summary["volume_std"] = summary["ticker"].map(merged.groupby("ticker")["volume_std"].first())
    summary["gap_std"] = summary["ticker"].map(merged.groupby("ticker")["gap_std"].first())
    summary.to_csv(out_dir / "plot_ticker_summary.csv", index=False)

    for ticker, group in merged.groupby("ticker"):
        group = group.sort_values("date")
        row = summary[summary["ticker"].eq(ticker)].iloc[0]
        title = (
            f"{args.title_prefix} | {ticker} | labels={int(row.labels)} "
            f"preds={int(row.preds)} tp={int(row.tp)} fp={int(row.fp)} fn={int(row.fn)} "
            f"by_ticker_q={threshold_quantile:.5f} label={'+'.join(label_features)}"
        )
        plot_ticker(group, out_dir / f"{safe_name(ticker)}_score_close.png", title, args.std_multiplier, label_features)

    print(f"Saved {len(tickers)} ticker plots to {out_dir}")
    print(summary.sort_values(["labels", "max_score"], ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
