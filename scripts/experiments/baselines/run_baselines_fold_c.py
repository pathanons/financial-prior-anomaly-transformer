"""Run the fixed baselines (Isolation Forest, One-Class SVM, Autoencoder, VAE,
LSTM-AE, Transformer-AE, 1D-CNN-AE) on fold C (the canonical split), with the
fairness fixes from PUBLICATION_PLAN.md item 4: matched reconstruction
features (log_return_1d,volume_z,gap, same as canonical AT-Time), matched
label (contextual_label, same as AT-Time/AT-TimeState), bumped capacity for
the sequence models, and gradient steps matched to AT-Time's own training
regime (batch_size=32, num_epochs=7, full ~101,887-window train set instead
of a 20,000-row subsample) instead of the old batch_size=512/epochs=15
combination that gave ~585 steps vs. AT-Time's ~22,288.

Runs on CPU by default so it doesn't contend with AT-TimeState training via
``run_model_grid.py --model time_state --folds C``.

Usage:
    python scripts/experiments/baselines/run_baselines_fold_c.py
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluate_baselines import baselines  # noqa: E402

OUT_DIR = Path(r"D:\financial-prior-research-paper\journey\08_walkforward_multiseed\data\baselines_fold_c_gradstep_fix")


def main():
    args = argparse.Namespace(
        feature_dir="SP500_features_vw60_lw60",
        features="log_return_1d,return_5d,return_20d,volume_z,abs_return,squared_return,"
                  "rolling_vol_5,rolling_vol_20,vol_ratio_5_20,gap,high_low_range",
        reconstruction_features="log_return_1d,volume_z,gap",
        label_col="contextual_label",
        epochs=7,
        batch_size=32,
        max_train_rows=60000,
        max_sequence_rows=110000,
        win_size=60,
        seed=0,
        cpu=True,
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df, scores = baselines(args, OUT_DIR)
    df.to_csv(OUT_DIR / "baseline_metrics_fixed.csv", index=False)
    scores.to_csv(OUT_DIR / "baseline_scores_fixed.csv", index=False)
    print(df.to_string(index=False))
    print(f"\nWrote {OUT_DIR / 'baseline_metrics_fixed.csv'}")


if __name__ == "__main__":
    main()
