import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


BASE_ARGS = [
    "--dataset", "STOCK",
    "--data_path", "SP500",
    "--win_size", "60",
    "--batch_size", "32",
    "--num_epochs", "5",
    "--train_start", "2018-01-01",
    "--train_end", "2021-12-31",
    "--val_start", "2022-01-01",
    "--val_end", "2022-12-31",
    "--test_start", "2023-01-01",
    "--test_end", "2024-12-31",
    "--k", "3",
    "--mode", "test",
]


EXPERIMENTS = [
    ("AT-TimeState_mean_none", ["--prior_type", "time_state", "--score_type", "original"]),
    ("AT-TimeState_max_none", ["--prior_type", "time_state", "--score_type", "original", "--score_aggregation", "max"]),
    ("AT-TimeState_mean_pct", ["--prior_type", "time_state", "--score_type", "original", "--score_normalization", "ticker_percentile"]),
    ("AT-TimeState_max_pct", ["--prior_type", "time_state", "--score_type", "original", "--score_aggregation", "max", "--score_normalization", "ticker_percentile"]),
    ("AT-TimeState_feature_weighted_max_pct", ["--prior_type", "time_state", "--score_type", "feature_weighted", "--score_aggregation", "max", "--score_normalization", "ticker_percentile"]),
    ("AT-TimeState_ReturnNLL_max_pct", ["--prior_type", "time_state", "--use_return_nll", "true", "--nll_weight", "0.1", "--return_loss_weight", "0.1", "--score_type", "return_nll", "--score_aggregation", "max", "--score_normalization", "ticker_percentile"]),
    ("AT-TimeState_mean_none_bestf1", ["--prior_type", "time_state", "--score_type", "original", "--threshold_method", "best_f1"]),
    ("AT-TimeState_max_none_bestf1", ["--prior_type", "time_state", "--score_type", "original", "--score_aggregation", "max", "--threshold_method", "best_f1"]),
    ("AT-TimeState_mean_pct_bestf1", ["--prior_type", "time_state", "--score_type", "original", "--score_normalization", "ticker_percentile", "--threshold_method", "best_f1"]),
    ("AT-TimeState_max_pct_bestf1", ["--prior_type", "time_state", "--score_type", "original", "--score_aggregation", "max", "--score_normalization", "ticker_percentile", "--threshold_method", "best_f1"]),
    ("AT-TimeState_feature_weighted_max_pct_bestf1", ["--prior_type", "time_state", "--score_type", "feature_weighted", "--score_aggregation", "max", "--score_normalization", "ticker_percentile", "--threshold_method", "best_f1"]),
    ("AT-TimeState_ReturnNLL_max_pct_bestf1", ["--prior_type", "time_state", "--use_return_nll", "true", "--nll_weight", "0.1", "--return_loss_weight", "0.1", "--score_type", "return_nll", "--score_aggregation", "max", "--score_normalization", "ticker_percentile", "--threshold_method", "best_f1"]),
]


def ensure_checkpoint(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", default="D:/multi-prior-at-run/score_ablation")
    parser.add_argument("--source_run", default="D:/multi-prior-at-run/AT-TimeState")
    parser.add_argument("--return_nll_source_run", default="D:/multi-prior-at-run/AT-TimeState-ReturnNLL")
    parser.add_argument("--summary", default=None)
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    summary_path = Path(args.summary) if args.summary else run_root / "score_ablation_metrics.csv"
    rows = []

    for name, extra in EXPERIMENTS:
        source_run = Path(args.return_nll_source_run if "ReturnNLL" in name else args.source_run)
        source_ckpt = source_run / "checkpoints" / "STOCK_checkpoint.pth"
        run_dir = run_root / name
        metrics_path = run_dir / "metrics.json"
        ensure_checkpoint(source_ckpt, run_dir / "checkpoints" / "STOCK_checkpoint.pth")

        if not (args.skip_existing and metrics_path.exists()):
            cmd = [
                sys.executable,
                "-u",
                "main.py",
                *BASE_ARGS,
                "--run_root", str(run_root),
                "--experiment_name", name,
                *extra,
            ]
            print("RUN", name, flush=True)
            subprocess.run(cmd, check=True)

        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        rows.append({"experiment": name, **metrics})

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment", *[k for k in rows[0].keys() if k != "experiment"]]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", summary_path)


if __name__ == "__main__":
    main()
