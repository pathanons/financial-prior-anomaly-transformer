import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from solver import Solver


def split_csv(text):
    return [part.strip() for part in str(text).split(",") if part.strip()]


def aggregate_split(solver, loader, feature_names, batch_size):
    feature_idx = [solver.feature_cols.index(name) for name in feature_names]
    buckets = defaultdict(lambda: [0.0, 0, 0])
    dataset = loader.dataset
    solver.model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))
            xs = np.asarray(dataset.samples[start:end], dtype=np.float32)
            labels = np.asarray(dataset.labels[start:end], dtype=np.int64)
            out = solver._forward_losses(torch.from_numpy(xs))
            errors = out["feature_error"][:, :, feature_idx].detach().cpu().numpy()
            for row, meta in enumerate(dataset.metadata[start:end]):
                ticker = meta["ticker"]
                for t, date in enumerate(meta["dates"]):
                    label = int(labels[row, t])
                    for f, name in enumerate(feature_names):
                        key = (ticker, date, name)
                        bucket = buckets[key]
                        bucket[0] += float(errors[row, t, f])
                        bucket[1] += 1
                        bucket[2] = max(bucket[2], label)

    rows = []
    for (ticker, date, feature), (total, count, label) in buckets.items():
        rows.append({
            "ticker": ticker,
            "date": date,
            "feature": feature,
            "error": total / count,
            "label": label,
        })
    return pd.DataFrame(rows)


def summarize(scores):
    rows = []
    for (split, feature), group in scores.groupby(["split", "feature"]):
        y = group["label"].astype(int).values
        s = group["error"].astype(float).values
        row = {
            "split": split,
            "feature": feature,
            "n": len(group),
            "positives": int(y.sum()),
            "mean": float(np.mean(s)),
            "median": float(np.median(s)),
            "p95": float(np.percentile(s, 95)),
            "roc_auc": float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else np.nan,
            "pr_auc": float(average_precision_score(y, s)) if len(np.unique(y)) > 1 else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def plot_stats(summary, out_dir):
    import matplotlib.pyplot as plt

    for metric in ["mean", "median", "p95"]:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        pivot = summary.pivot(index="feature", columns="split", values=metric)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(f"Per-feature reconstruction error {metric}")
        ax.set_ylabel("raw squared error")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"reconstruction_error_{metric}.png"), dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    for ax, metric in zip(axes, ["roc_auc", "pr_auc"]):
        pivot = summary[summary["split"].eq("test")].pivot(index="feature", columns="split", values=metric)
        pivot.plot(kind="bar", ax=ax, legend=False)
        ax.set_title(f"Test {metric}")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reconstruction_error_test_roc_pr.png"), dpi=160)
    plt.close(fig)


def plot_distribution(scores, out_dir):
    import matplotlib.pyplot as plt

    test = scores[scores["split"].eq("test")]
    features = list(test["feature"].drop_duplicates())
    fig, axes = plt.subplots(len(features), 1, figsize=(9, 2.8 * len(features)), sharex=False)
    if len(features) == 1:
        axes = [axes]
    for ax, feature in zip(axes, features):
        group = test[test["feature"].eq(feature)]
        for label, color in [(0, "0.45"), (1, "#d62728")]:
            values = group[group["label"].eq(label)]["error"].values
            if len(values):
                ax.hist(np.log10(values + 1e-12), bins=80, alpha=0.55, color=color, label=f"label={label}")
        ax.set_title(f"Test reconstruction error distribution: {feature}")
        ax.set_xlabel("log10(error + 1e-12)")
        ax.set_ylabel("count")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reconstruction_error_test_distribution.png"), dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target_features", default="log_return_1d,volume_z,gap")
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = json.load(f)
    config["run_root"] = None
    config["model_save_path"] = os.path.dirname(args.checkpoint)
    config["mode"] = "test"

    solver = Solver(config)
    solver.model.load_state_dict(torch.load(args.checkpoint, map_location=solver.device, weights_only=True))

    os.makedirs(args.out_dir, exist_ok=True)
    feature_names = split_csv(args.target_features)
    frames = []
    for split, loader in [("train", solver.train_loader), ("val", solver.vali_loader), ("test", solver.test_loader)]:
        frame = aggregate_split(solver, loader, feature_names, args.batch_size)
        frame.insert(0, "split", split)
        frames.append(frame)
        print(f"{split}: {len(frame)} feature-date rows")

    scores = pd.concat(frames, ignore_index=True)
    summary = summarize(scores)
    scores.to_csv(os.path.join(args.out_dir, "per_feature_reconstruction_scores.csv"), index=False)
    summary.to_csv(os.path.join(args.out_dir, "per_feature_reconstruction_summary.csv"), index=False)
    plot_stats(summary, args.out_dir)
    plot_distribution(scores, args.out_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
