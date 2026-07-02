import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p / (p.sum() + 1e-12)
    return float(-(p * np.log(p + 1e-12)).sum())


def kl(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float((p * (np.log(p + 1e-12) - np.log(q + 1e-12))).sum())


def parse_name(path):
    stem = path.stem.removeprefix("attention_values_")
    parts = stem.split("_")
    ticker = parts[0]
    date = "_".join(parts[-3:])
    case_name = "_".join(parts[1:-3])
    group = "normal" if case_name.startswith("normal") else "abnormal"
    return ticker, case_name, date, group


def summarize_one(path):
    ticker, case_name, event_date, group = parse_name(path)
    z = np.load(path, allow_pickle=True)
    dates = [str(x) for x in z["dates"]]
    features = [str(x) for x in z["feature_names"]]
    series = np.squeeze(z["series"])
    prior = np.squeeze(z["prior"])
    if series.ndim == 3:
        series = series[0]
        prior = prior[0]

    event_pos = len(dates) - 1
    s = series[event_pos].astype(float).copy()
    p = prior[event_pos].astype(float).copy()
    s[event_pos] = 0.0
    p[event_pos] = 0.0
    s = s / (s.sum() + 1e-12)
    p = p / (p.sum() + 1e-12)
    lags = np.abs(np.arange(len(s)) - event_pos)
    top_idx = int(np.argmax(s))
    prior_top_idx = int(np.argmax(p))
    feature_error = np.asarray(z["feature_error"])[event_pos].astype(float)
    top_feats = np.argsort(feature_error)[::-1][:3]

    return {
        "ticker": ticker,
        "case": case_name,
        "group": group,
        "event_date": event_date,
        "window_start": dates[0],
        "window_end": dates[-1],
        "score": float(np.asarray(z["score"])[event_pos]),
        "discrepancy": float(np.asarray(z["discrepancy"])[event_pos]),
        "selected_sym_kl": kl(p, s) + kl(s, p),
        "series_entropy": entropy(s),
        "prior_entropy": entropy(p),
        "top_key_date": dates[top_idx],
        "top_key_lag": int(event_pos - top_idx),
        "top_key_weight": float(s[top_idx]),
        "prior_top_key_date": dates[prior_top_idx],
        "prior_top_key_lag": int(event_pos - prior_top_idx),
        "near5_mass": float(s[lags <= 5].sum()),
        "near10_mass": float(s[lags <= 10].sum()),
        "far20_mass": float(s[lags >= 20].sum()),
        "top_feature_1": features[top_feats[0]],
        "top_feature_1_error": float(feature_error[top_feats[0]]),
        "top_feature_2": features[top_feats[1]],
        "top_feature_2_error": float(feature_error[top_feats[1]]),
        "top_feature_3": features[top_feats[2]],
        "top_feature_3_error": float(feature_error[top_feats[2]]),
        "source": str(path),
    }


def plot_outputs(df, out_dir):
    out_dir = Path(out_dir)
    metrics = [
        ("discrepancy", "Association discrepancy"),
        ("selected_sym_kl", "Selected-row symmetric KL"),
        ("series_entropy", "Series attention entropy"),
        ("top_key_lag", "Top attention lag"),
        ("near10_mass", "Attention mass within 10 days"),
        ("far20_mass", "Attention mass at lag >= 20"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.8))
    for ax, (col, title) in zip(axes.ravel(), metrics):
        groups = [df.loc[df.group == "abnormal", col].dropna(), df.loc[df.group == "normal", col].dropna()]
        ax.boxplot(groups, tick_labels=["abnormal", "normal"], showmeans=True)
        ax.scatter(
            np.repeat([1, 2], [len(groups[0]), len(groups[1])]),
            pd.concat(groups).values,
            s=18,
            alpha=0.55,
            color="#1f77b4",
        )
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Attention hypothesis checks from exported artifacts", y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "attention_hypothesis_boxplots.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.6))
    bins = np.arange(0, max(61, int(df.top_key_lag.max()) + 5), 5)
    ax.hist(df.loc[df.group == "abnormal", "top_key_lag"], bins=bins, alpha=0.65, label="abnormal")
    ax.hist(df.loc[df.group == "normal", "top_key_lag"], bins=bins, alpha=0.65, label="normal")
    ax.set_xlabel("Top attention lag")
    ax.set_ylabel("Case count")
    ax.set_title("Where does selected-query attention point?")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "attention_top_lag_histogram.png", dpi=180)
    plt.close(fig)

    top_features = df.groupby(["group", "top_feature_1"]).size().reset_index(name="count")
    pivot = top_features.pivot(index="top_feature_1", columns="group", values="count").fillna(0)
    pivot = pivot.sort_values(by=list(pivot.columns), ascending=False)
    ax = pivot.plot(kind="bar", figsize=(9, 4.8))
    ax.set_title("Most dominant feature-error dimension")
    ax.set_xlabel("Top feature")
    ax.set_ylabel("Case count")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "attention_top_feature_counts.png", dpi=180)
    plt.close()


def write_readme(df, out_dir):
    out_dir = Path(out_dir)
    summary = df.groupby("group")[
        ["discrepancy", "selected_sym_kl", "series_entropy", "top_key_lag", "near10_mass", "far20_mass"]
    ].agg(["count", "mean", "median"])
    with open(out_dir / "README.md", "w", encoding="utf-8") as f:
        f.write("# Attention summary\n\n")
        f.write("This summarizes exported `attention_values_*.npz` files using the final query date in each 60-day window.\n\n")
        f.write("Question: do abnormal cases show a different association structure from normal reference cases?\n\n")
        f.write("Current scope: existing exported artifacts only. If the export folder contains only AKAM, this is a case-study proof of method, not a market-wide conclusion.\n\n")
        f.write(summary.to_string())
        f.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention_dir", default="D:/multi-prior-at-run/AT-TimeState/figures")
    parser.add_argument("--out_dir", default="research_paper/weekly/2026-W26/attention_summary")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [summarize_one(path) for path in sorted(Path(args.attention_dir).glob("attention_values_*.npz"))]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "attention_summary.csv", index=False)
    plot_outputs(df, out_dir)
    write_readme(df, out_dir)
    print(df.groupby("group")[["discrepancy", "selected_sym_kl", "series_entropy", "top_key_lag", "near10_mass", "far20_mass"]].agg(["count", "mean", "median"]))
    print(f"wrote {len(df)} rows to {out_dir}")


if __name__ == "__main__":
    main()
