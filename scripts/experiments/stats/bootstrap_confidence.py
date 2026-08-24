"""Ticker-clustered bootstrap confidence intervals for the frozen canonical
model score (journey/03) and the causal behavior-filtered detector
(journey/05). No retraining: resamples tickers (not rows) from already-frozen
test artifacts, since ticker-days from the same ticker are correlated and an
IID row bootstrap would understate variance.

Usage:
    python scripts/experiments/stats/bootstrap_confidence.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

JOURNEY_ROOT = Path(r"D:\financial-prior-research-paper\journey")
N_BOOTSTRAP = 2000
SEED = 42


def bootstrap_ranker(scores_path, n_boot=N_BOOTSTRAP, seed=SEED):
    df = pd.read_csv(scores_path)
    tickers = df["ticker"].unique()
    rng = np.random.default_rng(seed)

    point = {
        "auc_pr": average_precision_score(df["label"], df["score"]),
        "auc_roc": roc_auc_score(df["label"], df["score"]),
    }

    boot = {"auc_pr": [], "auc_roc": []}
    by_ticker = {t: g for t, g in df.groupby("ticker")}
    for _ in range(n_boot):
        sample_tickers = rng.choice(tickers, size=len(tickers), replace=True)
        parts = [by_ticker[t] for t in sample_tickers]
        sample = pd.concat(parts, ignore_index=True)
        if sample["label"].nunique() < 2:
            continue
        boot["auc_pr"].append(average_precision_score(sample["label"], sample["score"]))
        boot["auc_roc"].append(roc_auc_score(sample["label"], sample["score"]))

    return point, boot, len(tickers)


def bootstrap_detector(confusion_path, n_boot=N_BOOTSTRAP, seed=SEED):
    df = pd.read_csv(confusion_path)
    tickers = df["ticker"].values
    rng = np.random.default_rng(seed)

    def prf(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return precision, recall, f1

    tp_tot, fp_tot, fn_tot = df["tp"].sum(), df["fp"].sum(), df["fn"].sum()
    p, r, f1 = prf(tp_tot, fp_tot, fn_tot)
    point = {"precision": p, "recall": r, "f1": f1}

    boot = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_boot):
        idx = rng.choice(len(tickers), size=len(tickers), replace=True)
        sample = df.iloc[idx]
        tp, fp, fn = sample["tp"].sum(), sample["fp"].sum(), sample["fn"].sum()
        p, r, f1 = prf(tp, fp, fn)
        boot["precision"].append(p)
        boot["recall"].append(r)
        boot["f1"].append(f1)

    return point, boot, len(tickers)


def ci_row(name, point_val, boot_vals):
    boot_vals = np.asarray(boot_vals)
    lo, hi = np.percentile(boot_vals, [2.5, 97.5])
    return {
        "metric": name,
        "point_estimate": point_val,
        "ci_low_2.5pct": lo,
        "ci_high_97.5pct": hi,
        "bootstrap_std": boot_vals.std(ddof=1),
        "n_bootstrap": len(boot_vals),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(JOURNEY_ROOT / "07_statistical_confidence" / "data"))
    parser.add_argument("--n_boot", type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_path = JOURNEY_ROOT / "03_canonical_model" / "data" / "test_ticker_day_scores.csv"
    confusion_path = JOURNEY_ROOT / "05_causal_local_behavior_detector" / "data" / "per_ticker_confusion.csv"

    ranker_point, ranker_boot, n_tickers_ranker = bootstrap_ranker(scores_path, args.n_boot)
    detector_point, detector_boot, n_tickers_detector = bootstrap_detector(confusion_path, args.n_boot)

    rows = []
    for name, val in ranker_point.items():
        rows.append({**ci_row(f"03_canonical_ranker.{name}", val, ranker_boot[name]),
                     "n_tickers": n_tickers_ranker})
    for name, val in detector_point.items():
        rows.append({**ci_row(f"05_causal_detector.{name}", val, detector_boot[name]),
                     "n_tickers": n_tickers_detector})

    out_df = pd.DataFrame(rows)
    out_path = out_dir / "bootstrap_ci_metrics.csv"
    out_df.to_csv(out_path, index=False)

    print(out_df.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
