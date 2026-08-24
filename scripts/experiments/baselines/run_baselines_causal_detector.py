"""Put every baseline on the SAME field as AT-Time's headline number: the
causal local-peak + behavior detector (journey/05 method), not raw PR-AUC
and not the naive Q99 threshold. Fits Isolation Forest, LSTM-AE,
Transformer-AE, and 1D-CNN-AE (same fixed models as
scripts/experiments/baselines/run_baselines_fold_c.py) on fold C, builds a
ticker-day score timeline for each (matching AT-Time's
val/test_timeline_scores.csv format), restricts to the same 25-ticker
reference set, and runs the exact same detector-parameter-selection
procedure used for AT-Time in
scripts/experiments/walkforward/evaluate_causal_detector.py:
grid-search on each model's own validation scores, then evaluate once on
test.

Usage:
    python scripts/experiments/baselines/run_baselines_causal_detector.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from evaluate_baselines import (  # noqa: E402
    DEFAULT_FEATURES, LSTMAutoencoder, TransformerAutoencoder, Conv1DAutoencoder,
    read_features, split_daily, sample_rows, reconstruction_error, fit_sequence_autoencoder,
)
from experiments._causal_detector import load_detector, prepare_rows, select_on_validation  # noqa: E402

JOURNEY_ROOT = Path(r"D:\financial-prior-research-paper")
REFERENCE = JOURNEY_ROOT / "weekly/2026-W26/at_score_q99_return_volume_gap_std_bounds/plot_ticker_summary.csv"
FEATURE_DIR = ROOT.parent / "SP500_features_vw60_lw60"
OUT_DIR = JOURNEY_ROOT / "journey" / "08_walkforward_multiseed" / "data" / "baseline_causal_detector"

FOLD_C_TRAIN_END = "2021-12-31"
WIN_SIZE = 60
LABEL_COL = "contextual_label"
SEED = 0
EPOCHS = 15
BATCH_SIZE = 64          # smaller batch -> more gradient steps, closes the step-count gap
MAX_SEQUENCE_ROWS = 60000  # more sampled windows -> more gradient steps too

det = load_detector(FEATURE_DIR)


def make_windows_with_meta(frame, scaler, features, label_col, win):
    xs, labels, tickers, dates = [], [], [], []
    for ticker, g in frame.sort_values(["ticker", "date"]).groupby("ticker"):
        vals = scaler.transform(g[features].values)
        lab = g[label_col].values.astype(int)
        date_vals = g["date"].values
        for end in range(win - 1, len(g)):
            xs.append(vals[end - win + 1: end + 1])
            labels.append(lab[end])
            tickers.append(ticker)
            dates.append(date_vals[end])
    return (np.asarray(xs, dtype=np.float32), np.asarray(labels, dtype=int),
            np.asarray(tickers), np.asarray(dates))


def timeline_df(tickers, dates, scores, labels):
    return pd.DataFrame({"ticker": tickers, "date": pd.to_datetime(dates), "score": scores, "label": labels})


def fit_isolation_forest(x_train, x_val, x_test, seed):
    iso = IsolationForest(n_estimators=120, contamination="auto", random_state=seed, n_jobs=-1)
    iso.fit(x_train)
    return -iso.score_samples(x_val), -iso.score_samples(x_test)


def main():
    features = DEFAULT_FEATURES
    frame = read_features(str(FEATURE_DIR), features, LABEL_COL)
    train, val, test = split_daily(frame)
    scaler = StandardScaler().fit(train[features].values)
    x_train = scaler.transform(train[features].values)
    x_val = scaler.transform(val[features].values)
    x_test = scaler.transform(test[features].values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # nvidia-smi confirmed AT-TimeState's concurrent training uses ~1GB/8.5GB VRAM
    # at ~35% util, so there's headroom for these much smaller baseline nets too.
    reference_tickers = sorted(pd.read_csv(REFERENCE)["ticker"].astype(str).unique())
    det_features = det.load_features(reference_tickers)

    results = {}

    # --- Isolation Forest: per-row, no windowing ---
    train_sample = sample_rows(x_train, 60000, SEED)
    val_score, test_score = fit_isolation_forest(train_sample, x_val, x_test, SEED)
    val_tl = timeline_df(val["ticker"].values, val["date"].values, val_score, val[LABEL_COL].values)
    test_tl = timeline_df(test["ticker"].values, test["date"].values, test_score, test[LABEL_COL].values)
    results["Isolation Forest"] = (val_tl, test_tl)

    # --- Sequence models: LSTM-AE, Transformer-AE, 1D-CNN-AE ---
    seq_train, _, _, _ = make_windows_with_meta(train, scaler, features, LABEL_COL, WIN_SIZE)
    seq_val, y_val, tick_val, date_val = make_windows_with_meta(val, scaler, features, LABEL_COL, WIN_SIZE)
    seq_test, y_test, tick_test, date_test = make_windows_with_meta(test, scaler, features, LABEL_COL, WIN_SIZE)
    seq_train = sample_rows(seq_train, MAX_SEQUENCE_ROWS, SEED)
    recon_features = ["log_return_1d", "volume_z", "gap"]
    recon_idx = [features.index(name) for name in recon_features]

    for name, model in [
        ("LSTM Autoencoder", LSTMAutoencoder(len(features))),
        ("Transformer Autoencoder", TransformerAutoencoder(len(features))),
        ("1D-CNN Autoencoder", Conv1DAutoencoder(len(features))),
    ]:
        val_score, test_score = fit_sequence_autoencoder(
            model, seq_train, seq_val, seq_test, EPOCHS, BATCH_SIZE, device,
            recon_idx=recon_idx,
        )
        val_tl = timeline_df(tick_val, date_val, val_score, y_val)
        test_tl = timeline_df(tick_test, date_test, test_score, y_test)
        results[name] = (val_tl, test_tl)
        print(f"[{name}] trained, val n={len(val_tl)} test n={len(test_tl)}", flush=True)

    # --- Causal detector on each model's timeline, restricted to the 25-ticker reference set ---
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for name, (val_tl, test_tl) in results.items():
        val_ref = val_tl[val_tl["ticker"].isin(reference_tickers)].dropna(subset=["score"])
        test_ref = test_tl[test_tl["ticker"].isin(reference_tickers)].dropna(subset=["score"])
        val_labeled = prepare_rows(det, val_ref, det_features, FOLD_C_TRAIN_END)
        test_labeled = prepare_rows(det, test_ref, det_features, FOLD_C_TRAIN_END)

        _, local_only, behavior_filtered = select_on_validation(det, val_labeled, test_labeled)
        summary_rows.append({"model": name, "method": "local_only", **local_only})
        summary_rows.append({"model": name, "method": "behavior_filtered", **behavior_filtered})
        print(f"[{name}] causal detector: local_only.f1={local_only['f1']:.4f} "
              f"behavior_filtered.f1={behavior_filtered['f1']:.4f}", flush=True)

    summary = pd.DataFrame(summary_rows)
    out_path = OUT_DIR / "baseline_causal_detector_metrics.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
