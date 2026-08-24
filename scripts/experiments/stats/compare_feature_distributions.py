"""Are log_return_1d, volume_z, and gap similarly distributed across the
train (2018-2021), validation (2022), and test (2023-2024) splits, or has
the market regime drifted? Reads the already-cached per-ticker feature
files directly (no model involved) and compares distributions with summary
stats plus a two-sample Kolmogorov-Smirnov test.

Usage:
    python scripts/experiments/stats/compare_feature_distributions.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

FEATURE_DIR = Path("SP500_features_vw60_lw60")
FEATURES = ["log_return_1d", "volume_z", "gap"]
SPLITS = {
    "train": ("2018-01-01", "2021-12-31"),
    "val": ("2022-01-01", "2022-12-31"),
    "test": ("2023-01-01", "2024-12-31"),
}


def load_all():
    frames = []
    for path in sorted(FEATURE_DIR.glob("*_features.csv")):
        df = pd.read_csv(path, usecols=["date", "ticker"] + FEATURES, parse_dates=["date"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main():
    df = load_all().replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES)

    split_data = {}
    for name, (start, end) in SPLITS.items():
        mask = (df["date"] >= start) & (df["date"] <= end)
        split_data[name] = df.loc[mask]

    print("=== Summary stats per split ===")
    rows = []
    for feat in FEATURES:
        for name, sdf in split_data.items():
            s = sdf[feat]
            rows.append({
                "feature": feat, "split": name, "n": len(s),
                "mean": s.mean(), "std": s.std(),
                "p1": s.quantile(0.01), "p50": s.quantile(0.5), "p99": s.quantile(0.99),
            })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    print("\n=== KS test: train vs test, train vs val (same feature = drift check) ===")
    ks_rows = []
    for feat in FEATURES:
        for a, b in [("train", "test"), ("train", "val"), ("val", "test")]:
            stat, pvalue = ks_2samp(split_data[a][feat], split_data[b][feat])
            ks_rows.append({"feature": feat, "compare": f"{a}_vs_{b}", "ks_stat": stat, "p_value": pvalue})
    ks_df = pd.DataFrame(ks_rows)
    print(ks_df.to_string(index=False))

    out_dir = Path(r"D:\financial-prior-research-paper\journey\07_statistical_confidence\data")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "feature_distribution_summary_by_split.csv", index=False)
    ks_df.to_csv(out_dir / "feature_distribution_ks_test.csv", index=False)
    print(f"\nWrote {out_dir / 'feature_distribution_summary_by_split.csv'}")
    print(f"Wrote {out_dir / 'feature_distribution_ks_test.csv'}")


if __name__ == "__main__":
    main()
