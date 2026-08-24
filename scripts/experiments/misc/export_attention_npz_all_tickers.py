import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from solver import Solver, kl_point


def ticker_names(feature_dir):
    return [
        path.name[: -len("_features.csv")]
        for path in sorted(Path(feature_dir).glob("*_features.csv"))
    ]


def build_config(args):
    return SimpleNamespace(
        lr=1e-4,
        num_epochs=1,
        k=args.k,
        win_size=args.win_size,
        input_c=11,
        output_c=11,
        batch_size=args.batch_size,
        pretrained_model=None,
        dataset="STOCK",
        mode="visualize",
        data_path=args.data_path,
        model_save_path=args.model_save_path,
        anormly_ratio=1.0,
        date_col="date",
        ticker_col="ticker",
        open_col="open",
        high_col="high",
        low_col="low",
        close_col="close",
        volume_col="volume",
        tickers=None,
        features="log_return_1d,return_5d,return_20d,volume_z,abs_return,squared_return,rolling_vol_5,rolling_vol_20,vol_ratio_5_20,gap,high_low_range",
        z_state_features="log_return_1d,abs_return,volume_z,rolling_vol_5,rolling_vol_20,vol_ratio_5_20",
        volume_window=60,
        label_window=60,
        feature_cache_dir=args.feature_dir,
        train_start=args.train_start,
        train_end=args.train_end,
        val_start=args.val_start,
        val_end=args.val_end,
        test_start=args.test_start,
        test_end=args.test_end,
        prior_type=args.prior_type,
        use_return_nll=False,
        nll_weight=0.0,
        return_loss_weight=0.0,
        score_type="original",
        run_root=None,
        experiment_name=None,
        output_dir=args.out_dir,
        auto_case_ticker=None,
        visualize_ticker=None,
        event_date=None,
        stride=1,
        plot_layer=0,
        plot_head="average",
        threshold_method="percentile",
        threshold_percentile=99.0,
        event_tolerance=1,
        top_k=None,
        score_aggregation="mean",
        label_type="absolute",
        feature_weights=None,
    )


def save_attention_npz(solver, ticker, event_date_text, case_name, skip_existing=False):
    out_dir = Path(solver.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"attention_values_{ticker}_{case_name}_{event_date_text.replace('-', '_')}.npz"
    if skip_existing and output.exists():
        return output

    dataset = solver.test_loader.dataset
    event_date = np.datetime64(event_date_text)
    candidates = [
        (i, meta)
        for i, meta in enumerate(dataset.metadata)
        if meta["ticker"] == ticker
        and np.datetime64(meta["dates"][0]) <= event_date <= np.datetime64(meta["dates"][-1])
    ]
    if not candidates:
        return None

    index, meta = min(
        candidates,
        key=lambda item: abs((np.datetime64(item[1]["end_date"]) - event_date).astype(int)),
    )
    x_np, _ = dataset[index]
    x = torch.from_numpy(x_np).unsqueeze(0).float().to(solver.device)
    with torch.no_grad():
        out = solver._forward_losses(x)
        score = solver.compute_anomaly_score(out).squeeze(0).cpu().numpy()

    layer = int(solver.plot_layer)
    series = out["series"][layer].detach().mean(dim=1).squeeze(0).cpu()
    prior = out["prior"][layer].detach().mean(dim=1).squeeze(0).cpu()
    discrepancy = (
        kl_point(out["prior"][layer], out["series"][layer])
        + kl_point(out["series"][layer], out["prior"][layer])
    ).mean(dim=1).squeeze(0).cpu().numpy()
    feature_error = out["feature_error"].squeeze(0).detach().cpu().numpy()

    np.savez_compressed(
        output,
        series=series.numpy(),
        prior=prior.numpy(),
        discrepancy=discrepancy,
        score=score,
        feature_error=feature_error,
        dates=np.array(meta["dates"]),
        feature_names=np.array(solver.feature_cols),
    )
    return output


def auto_case_dates_from_scores(solver, ticker, scores):
    frame = solver.test_loader.dataset.frame
    ticker_frame = frame[
        frame[solver.test_loader.dataset.ticker_col].astype(str) == str(ticker)
    ].copy()
    if ticker_frame.empty:
        raise ValueError(f"No test rows found for ticker {ticker}")

    ticker_frame = ticker_frame.sort_values(solver.test_loader.dataset.date_col).reset_index(drop=True)
    date_col = solver.test_loader.dataset.date_col
    label_col = solver.test_loader.dataset.label_col
    dates = ticker_frame[date_col].dt.strftime("%Y-%m-%d").tolist()

    max_return_idx = int(ticker_frame["log_return_1d"].astype(float).idxmax())
    normal_frame = ticker_frame[ticker_frame[label_col] == 0]
    if normal_frame.empty:
        normal_frame = ticker_frame
    normal_idx = int(normal_frame["log_return_1d"].abs().astype(float).idxmin())

    ticker_scores = {date: score for (score_ticker, date), score in scores.items() if score_ticker == ticker}
    if not ticker_scores:
        raise ValueError(f"No scores found for ticker {ticker}")
    max_score_date = max(ticker_scores, key=ticker_scores.get)
    date_to_idx = {date: i for i, date in enumerate(dates)}
    max_score_idx = date_to_idx[max_score_date]

    selected = []
    for name, idx in [
        ("max_log_return", max_return_idx),
        ("normal_low_abs_log_return", normal_idx),
        ("max_score", max_score_idx),
    ]:
        for day_idx in range(max(0, idx - 5), idx + 1):
            selected.append((name, dates[day_idx]))
    return selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="SP500")
    parser.add_argument("--feature_dir", default="SP500_features_vw60_lw60")
    parser.add_argument("--model_save_path", default="D:/multi-prior-at-run/AT-TimeState/checkpoints")
    parser.add_argument("--out_dir", default="D:/multi-prior-at-run/AT-TimeState/attention_npz_all_tickers")
    parser.add_argument("--manifest", default="research_paper/weekly/2026-W26/attention_summary/all_ticker_attention_manifest.csv")
    parser.add_argument("--prior_type", default="time_state")
    parser.add_argument("--win_size", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--train_start", default="2018-01-01")
    parser.add_argument("--train_end", default="2021-12-31")
    parser.add_argument("--val_start", default="2022-01-01")
    parser.add_argument("--val_end", default="2022-12-31")
    parser.add_argument("--test_start", default="2023-01-01")
    parser.add_argument("--test_end", default="2024-12-31")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--start_after", default=None)
    args = parser.parse_args()

    tickers = ticker_names(args.feature_dir)
    if args.start_after:
        start_at = tickers.index(args.start_after) + 1
        tickers = tickers[start_at:]
    if args.limit:
        tickers = tickers[: args.limit]

    solver = Solver(build_config(args).__dict__)
    solver.model.load_state_dict(
        torch.load(
            os.path.join(str(solver.model_save_path), str(solver.dataset) + "_checkpoint.pth"),
            map_location=solver.device,
            weights_only=True,
        )
    )
    solver.model.eval()
    scores, _ = solver.aggregate_window_scores(solver.test_loader)

    rows = []
    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    for idx, ticker in enumerate(tickers, 1):
        try:
            cases = auto_case_dates_from_scores(solver, ticker, scores)
            for case_name, event_date in cases:
                output = save_attention_npz(solver, ticker, event_date, case_name, args.skip_existing)
                if output is not None:
                    rows.append({"ticker": ticker, "case": case_name, "event_date": event_date, "path": str(output)})
            print(f"{idx}/{len(tickers)} {ticker}: {len(cases)} selected, total npz={len(rows)}", flush=True)
        except Exception as exc:
            rows.append({"ticker": ticker, "case": "ERROR", "event_date": "", "path": str(exc)})
            print(f"{idx}/{len(tickers)} {ticker}: ERROR {exc}", flush=True)
        pd.DataFrame(rows).to_csv(manifest, index=False)

    pd.DataFrame(rows).to_csv(manifest, index=False)
    print(f"wrote {len(rows)} rows to {manifest}")


if __name__ == "__main__":
    main()
