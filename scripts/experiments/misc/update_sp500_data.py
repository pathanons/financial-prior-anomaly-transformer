"""Extend the local SP500 OHLCV data up to a target date, then regenerate
the derived feature caches (SP500_features_vw60_lw60 and
SP500_features_vw60_lw60_z5, which are identical duplicates of each other)
using the actual pipeline code in data_factory/data_loader.py — not a
reimplementation of the feature math.

Fetches only new rows per ticker via yfinance (from the day after each
ticker's last cached date through --end), appends to the existing raw CSV,
then rebuilds that ticker's full feature history.

Usage:
    python scripts/experiments/misc/update_sp500_data.py --end 2026-07-31
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data_factory.data_loader import build_stock_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = ROOT / "SP500"
FEATURE_DIRS = [ROOT / "SP500_features_vw60_lw60", ROOT / "SP500_features_vw60_lw60_z5"]
RAW_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def fetch_new_rows(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df[RAW_COLUMNS]


def update_ticker(ticker, end_date):
    raw_path = RAW_DIR / f"{ticker}_ohlcv.csv"
    existing = pd.read_csv(raw_path)
    last_date = pd.to_datetime(existing["Date"]).max()
    start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if start >= end_date:
        return "up_to_date", 0

    new_rows = fetch_new_rows(ticker, start, end_date)
    if new_rows is None or new_rows.empty:
        return "no_new_rows", 0

    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"]).dt.strftime("%Y-%m-%d")
    combined = combined.drop_duplicates(subset="Date", keep="last").sort_values("Date")
    combined.to_csv(raw_path, index=False)

    full = combined.rename(columns={"Date": "date", "Open": "open", "High": "high",
                                     "Low": "low", "Close": "close", "Volume": "volume"})
    full["ticker"] = ticker
    features = build_stock_features(full)
    for feature_dir in FEATURE_DIRS:
        features.to_csv(feature_dir / f"{ticker}_features.csv", index=False)

    return "updated", len(new_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--end", required=True, help="YYYY-MM-DD, fetch up to (exclusive) this date")
    parser.add_argument("--sleep", type=float, default=0.3, help="seconds between tickers, be polite to the API")
    args = parser.parse_args()

    tickers = sorted(p.stem.replace("_ohlcv", "") for p in RAW_DIR.glob("*_ohlcv.csv"))
    print(f"{len(tickers)} tickers, fetching through {args.end}", flush=True)

    summary = {"updated": [], "up_to_date": [], "no_new_rows": [], "failed": []}
    for i, ticker in enumerate(tickers, 1):
        try:
            status, n_rows = update_ticker(ticker, args.end)
            summary[status].append(ticker)
            print(f"[{i}/{len(tickers)}] {ticker}: {status} (+{n_rows} rows)", flush=True)
        except Exception as exc:
            summary["failed"].append(ticker)
            print(f"[{i}/{len(tickers)}] {ticker}: FAILED - {exc}", flush=True)
        time.sleep(args.sleep)

    print("\n=== Summary ===", flush=True)
    for status, names in summary.items():
        print(f"{status}: {len(names)}", flush=True)
        if status == "failed" and names:
            print("  " + ", ".join(names), flush=True)


if __name__ == "__main__":
    main()
