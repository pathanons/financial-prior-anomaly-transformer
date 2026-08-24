"""Train and evaluate AT-Time variants across fold/seed combinations.

Examples:
    python scripts/experiments/run_model_grid.py
    python scripts/experiments/run_model_grid.py --model time_state --folds C
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
PYTHON = r"C:\Users\Acer\miniconda3\python.exe"
RUN_ROOT = Path(r"D:\multi-prior-at-run-walkforward-multiseed")
SUMMARY_OUT = Path(r"D:\financial-prior-research-paper\journey\08_walkforward_multiseed\data")

FOLDS = {
    "A": dict(train_start="2018-01-01", train_end="2019-12-31",
              val_start="2020-01-01", val_end="2020-12-31",
              test_start="2021-01-01", test_end="2021-12-31"),
    "B": dict(train_start="2018-01-01", train_end="2020-12-31",
              val_start="2021-01-01", val_end="2021-12-31",
              test_start="2022-01-01", test_end="2022-12-31"),
    "C": dict(train_start="2018-01-01", train_end="2021-12-31",
              val_start="2022-01-01", val_end="2022-12-31",
              test_start="2023-01-01", test_end="2024-12-31"),
}
MODELS = {
    "time": {
        "experiment": "AT-Time",
        "default_folds": list(FOLDS),
        "summary": "walkforward_multiseed_metrics.json",
    },
    "time_state": {
        "experiment": "AT-TimeState",
        "default_folds": ["C"],
        "summary": "foldC_at_timestate_metrics.json",
    },
    "state": {
        "experiment": "AT-State",
        "default_folds": ["C"],
        "summary": "foldC_at_state_metrics.json",
    },
}
COMMON_ARGS = [
    "--dataset", "STOCK", "--data_path", "SP500",
    "--batch_size", "32", "--num_epochs", "7",
    "--k", "3", "--score_type", "original",
    "--reconstruction_features", "log_return_1d,volume_z,gap",
    "--label_type", "contextual", "--e_layers", "4", "--n_heads", "16",
    "--run_root", str(RUN_ROOT),
]


def safe_tag(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def experiment_name(model, fold, seed, tag=None):
    suffix = f"_{safe_tag(tag)}" if tag else ""
    return f"{MODELS[model]['experiment']}_fold{fold}_seed{seed}{suffix}"


def run(mode, model, fold_id, seed, args):
    name = experiment_name(model, fold_id, seed, args.tag)
    fold = FOLDS[fold_id]
    command = [
        PYTHON, "main.py", "--mode", mode,
        "--experiment_name", name,
        "--seed", str(seed),
        "--win_size", str(args.win_size),
        "--prior_type", model,
        "--train_start", fold["train_start"], "--train_end", fold["train_end"],
        "--val_start", fold["val_start"], "--val_end", fold["val_end"],
        "--test_start", fold["test_start"], "--test_end", fold["test_end"],
        "--model_save_path", str(RUN_ROOT / name / "checkpoints"),
        *COMMON_ARGS,
    ]
    if args.features:
        command.extend(["--features", args.features])
    if args.z_state_features:
        command.extend(["--z_state_features", args.z_state_features])
    if args.reconstruction_features:
        command.extend(["--reconstruction_features", args.reconstruction_features])
    if args.score_type:
        command.extend(["--score_type", args.score_type])
    if args.calibrated_score_features:
        command.extend(["--calibrated_score_features", args.calibrated_score_features])
    if args.state_projection_dim:
        command.extend(["--state_projection_dim", str(args.state_projection_dim)])
    if args.token_embedding:
        command.extend(["--token_embedding", args.token_embedding])
    if args.no_positional_embedding:
        command.extend(["--use_positional_embedding", "false"])
    if args.position_sigma:
        command.extend(["--position_sigma", str(args.position_sigma)])
    started = time.time()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    print(
        f"[{name}] mode={mode} exit={result.returncode} elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    if result.returncode:
        print(result.stdout[-3000:], flush=True)
        print(result.stderr[-3000:], flush=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, default="time")
    parser.add_argument("--folds", nargs="+", choices=FOLDS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--tag", default=None)
    parser.add_argument("--win_size", type=int, default=60)
    parser.add_argument("--features", default=None)
    parser.add_argument("--z_state_features", default=None)
    parser.add_argument("--reconstruction_features", default=None)
    parser.add_argument("--score_type", default=None,
                        choices=["original", "return_recon", "return_nll", "feature_weighted",
                                 "calibrated_components"])
    parser.add_argument("--calibrated_score_features", default=None)
    parser.add_argument("--state_projection_dim", type=int, default=0)
    parser.add_argument("--token_embedding", default=None,
                        choices=["conv3_circular", "linear", "linear_norm", "linear_residual_mlp",
                                 "grouped"])
    parser.add_argument("--no_positional_embedding", action="store_true")
    parser.add_argument("--position_sigma", type=float, default=0.0)
    args = parser.parse_args()

    fold_ids = args.folds or MODELS[args.model]["default_folds"]
    total = len(fold_ids) * len(args.seeds)
    rows = []
    SUMMARY_OUT.mkdir(parents=True, exist_ok=True)

    for done, (fold_id, seed) in enumerate(
        ((fold, seed) for fold in fold_ids for seed in args.seeds), start=1
    ):
        print(f"=== [{done}/{total}] fold={fold_id} seed={seed} ===", flush=True)
        if not run("train", args.model, fold_id, seed, args):
            print("Training failed; test skipped", flush=True)
            continue
        if not run("test", args.model, fold_id, seed, args):
            print("Test failed; metrics skipped", flush=True)
            continue

        name = experiment_name(args.model, fold_id, seed, args.tag)
        metrics_path = RUN_ROOT / name / "metrics.json"
        if metrics_path.exists():
            rows.append({"fold": fold_id, "seed": seed, **json.loads(metrics_path.read_text())})
        else:
            print(f"Missing metrics: {metrics_path}", flush=True)

    summary_name = MODELS[args.model]["summary"]
    if args.tag:
        summary_name = f"{Path(summary_name).stem}_{safe_tag(args.tag)}.json"
    summary_path = SUMMARY_OUT / summary_name
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {summary_path} ({len(rows)}/{total} runs succeeded)", flush=True)


if __name__ == "__main__":
    main()
