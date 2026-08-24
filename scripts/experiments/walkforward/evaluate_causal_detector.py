"""Evaluate the causal detector across the walk-forward fold/seed grid.

``frozen`` reuses the canonical parameters. ``validation`` selects parameters
on each run's validation set before evaluating its test set.

Usage:
    python scripts/experiments/walkforward/evaluate_causal_detector.py
    python scripts/experiments/walkforward/evaluate_causal_detector.py --selection validation
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from experiments._causal_detector import (  # noqa: E402
    load_detector,
    load_timeline_scores,
    prepare_rows,
    select_on_validation,
)


JOURNEY_ROOT = Path(r"D:\financial-prior-research-paper")
RUN_ROOT = Path(r"D:\multi-prior-at-run-walkforward-multiseed")
REFERENCE = JOURNEY_ROOT / "weekly/2026-W26/at_score_q99_return_volume_gap_std_bounds/plot_ticker_summary.csv"
FEATURES = ROOT / "SP500_features_vw60_lw60"
OUT_DIR = JOURNEY_ROOT / "journey" / "08_walkforward_multiseed" / "data"

FOLDS = ["A", "B", "C"]
SEEDS = [0, 1, 2]
FROZEN_PARAMS = dict(lookback=15, local_z=8.0, prominence_z=3.0, max_width=2.0, tolerance=2)
FROZEN_FILTER = dict(cooldown=3, behavior_min=3.0, behavior_window=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", choices=("all", "frozen", "validation"), default="all")
    args = parser.parse_args()

    detector = load_detector(FEATURES)
    tickers = sorted(pd.read_csv(REFERENCE)["ticker"].astype(str).unique())
    features = detector.load_features(tickers)
    frozen_rows = []
    selected_rows = []

    for fold in FOLDS:
        for seed in SEEDS:
            run_dir = RUN_ROOT / f"AT-Time_fold{fold}_seed{seed}"
            train_end = json.loads((run_dir / "config.json").read_text())["train_end"]
            test_scores = load_timeline_scores(run_dir, "test", tickers)
            test = prepare_rows(detector, test_scores, features, train_end)

            if args.selection in ("all", "frozen"):
                local = detector.evaluate(test, **FROZEN_PARAMS)
                filtered = detector.evaluate(test, **FROZEN_PARAMS, **FROZEN_FILTER)
                frozen_rows.extend((
                    {"fold": fold, "seed": seed, "method": "local_only", **local},
                    {"fold": fold, "seed": seed, "method": "behavior_filtered", **filtered},
                ))
                print(
                    f"fold={fold} seed={seed} frozen: "
                    f"local.f1={local['f1']:.4f} filtered.f1={filtered['f1']:.4f}",
                    flush=True,
                )

            if args.selection in ("all", "validation"):
                val_scores = load_timeline_scores(run_dir, "val", tickers)
                validation = prepare_rows(detector, val_scores, features, train_end)
                params, local, filtered = select_on_validation(detector, validation, test)
                selected_rows.extend((
                    {"fold": fold, "seed": seed, "method": "local_only", **params, **local},
                    {"fold": fold, "seed": seed, "method": "behavior_filtered", **params, **filtered},
                ))
                print(
                    f"fold={fold} seed={seed} selected: "
                    f"local.f1={local['f1']:.4f} filtered.f1={filtered['f1']:.4f}",
                    flush=True,
                )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if frozen_rows:
        path = OUT_DIR / "causal_detector_by_fold_seed.csv"
        pd.DataFrame(frozen_rows).to_csv(path, index=False)
        print(f"Wrote {path}")
    if selected_rows:
        path = OUT_DIR / "causal_detector_reselected_per_run.csv"
        pd.DataFrame(selected_rows).to_csv(path, index=False)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
