import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


LABELS = {
    "positive": ("positive_label", "score_positive"),
    "negative": ("negative_label", "score_negative"),
    "absolute": ("absolute_label", "score_absolute"),
    "contextual": ("contextual_label", "score_contextual"),
    "overall_max": ("absolute_label", "score_overall_max"),
}


def percentile_by_ticker(frame, raw):
    out = pd.Series(index=frame.index, dtype=float)
    tmp = pd.DataFrame({"ticker": frame["ticker"], "raw": raw}, index=frame.index)
    for _, group in tmp.groupby("ticker"):
        out.loc[group.index] = group["raw"].rank(method="average", pct=True)
    return out.fillna(0.0)


def load_features(feature_dir):
    cols = [
        "ticker",
        "date",
        "z_return",
        "mad_z_return",
        "volume_z",
        "vol_ratio_5_20",
        "positive_label",
        "negative_label",
        "absolute_label",
        "contextual_label",
    ]
    frames = []
    for path in sorted(Path(feature_dir).glob("*_features.csv")):
        frame = pd.read_csv(path, parse_dates=["date"])
        frames.append(frame[cols])
    return (
        pd.concat(frames, ignore_index=True)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["z_return", "mad_z_return", "volume_z", "vol_ratio_5_20"])
    )


def add_scores(frame):
    pos_tail = np.maximum(frame["z_return"], frame["mad_z_return"])
    neg_tail = np.maximum(-frame["z_return"], -frame["mad_z_return"])
    abs_tail = np.maximum(frame["z_return"].abs(), frame["mad_z_return"].abs())
    volume = frame["volume_z"]
    vol_ratio = frame["vol_ratio_5_20"]

    frame = frame.copy()
    frame["score_positive"] = percentile_by_ticker(frame, pos_tail)
    frame["score_negative"] = percentile_by_ticker(frame, neg_tail)
    frame["score_absolute"] = percentile_by_ticker(frame, abs_tail)
    frame["score_volume"] = percentile_by_ticker(frame, volume)
    frame["score_vol_ratio"] = percentile_by_ticker(frame, vol_ratio)
    frame["score_contextual"] = np.minimum(
        frame["score_absolute"], np.maximum(frame["score_volume"], frame["score_vol_ratio"])
    )
    frame["score_overall_max"] = frame[
        ["score_positive", "score_negative", "score_contextual"]
    ].max(axis=1)
    return frame


def metrics_for(frame, label_col, score_col, threshold):
    y = frame[label_col].astype(int).to_numpy()
    score = frame[score_col].astype(float).to_numpy()
    pred = score >= threshold
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return {
        "positives": int(y.sum()),
        "ap": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", default="SP500_features_vw60_lw60")
    parser.add_argument("--val_start", default="2022-01-01")
    parser.add_argument("--val_end", default="2022-12-31")
    parser.add_argument("--test_start", default="2023-01-01")
    parser.add_argument("--test_end", default="2024-12-31")
    parser.add_argument("--threshold_percentile", type=float, default=99.0)
    parser.add_argument("--out_dir", default="research_paper/weekly/2026-W26/multitype_unsupervised")
    args = parser.parse_args()

    frame = add_scores(load_features(args.feature_dir))
    val = frame[(frame["date"] >= args.val_start) & (frame["date"] <= args.val_end)]
    test = frame[(frame["date"] >= args.test_start) & (frame["date"] <= args.test_end)]

    rows = []
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for name, (label_col, score_col) in LABELS.items():
        threshold = np.percentile(val[score_col].astype(float), args.threshold_percentile)
        row = {"task": name, **metrics_for(test, label_col, score_col, threshold)}
        rows.append(row)
        precision, recall, _ = precision_recall_curve(
            test[label_col].astype(int), test[score_col].astype(float)
        )
        ax.plot(recall, precision, linewidth=2, label=f"{name} AP={row['ap']:.3f}")

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "metrics.csv", index=False)
    test.to_csv(out_dir / "daily_scores.csv", index=False)

    ax.set_title("Unsupervised Multi-Type Financial Anomaly Scores")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "precision_recall_by_type.png", dpi=180)
    plt.close(fig)

    print(result.round(4).to_string(index=False))
    print(out_dir)


if __name__ == "__main__":
    main()
