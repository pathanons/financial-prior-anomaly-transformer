import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.collections import LineCollection, PolyCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from solver import Solver


DEFAULT_RUN = Path(r"D:\multi-prior-at-run-walkforward-multiseed\AT-State_foldC_seed0_linear_token")


def add_candles(ax, frame):
    x = mdates.date2num(frame["date"])
    up = frame["close"].to_numpy() >= frame["open"].to_numpy()
    colors = np.where(up, "#2ca02c", "#d62728")
    ax.add_collection(LineCollection(
        [[(xi, lo), (xi, hi)] for xi, lo, hi in zip(x, frame["low"], frame["high"])],
        colors=colors,
        linewidths=0.7,
        alpha=0.85,
    ))
    width = 0.55
    polys = []
    for xi, op, cl in zip(x, frame["open"], frame["close"]):
        lo, hi = sorted([op, cl])
        if hi == lo:
            hi = lo + 1e-6
        polys.append([(xi - width / 2, lo), (xi - width / 2, hi), (xi + width / 2, hi), (xi + width / 2, lo)])
    ax.add_collection(PolyCollection(polys, facecolors=colors, edgecolors=colors, alpha=0.75))
    ax.set_xlim(x.min() - 3, x.max() + 3)
    ax.set_ylim(float(frame["low"].min()) * 0.97, float(frame["high"].max()) * 1.03)


def build_solver(run_dir):
    config = json.loads((run_dir / "config.json").read_text())
    config.update({
        "run_root": None,
        "model_save_path": str(run_dir / "checkpoints"),
        "batch_size": 1,
        "use_attention_future_loss": False,
    })
    solver = Solver(config)
    solver.model.load_state_dict(torch.load(
        run_dir / "checkpoints" / "STOCK_checkpoint.pth",
        map_location=solver.device,
        weights_only=True,
    ))
    solver.model.eval()
    return solver


def collect_attention(solver, ticker, max_frames=None):
    maps, rows = [], []
    dataset = solver.test_loader.dataset
    indices = [(i, m) for i, m in enumerate(dataset.metadata) if str(m["ticker"]) == str(ticker)]
    if max_frames:
        indices = indices[:max_frames]
    layer = int(solver.plot_layer)
    with torch.no_grad():
        for n, (i, meta) in enumerate(indices, 1):
            x_np, _ = dataset[i]
            x = torch.from_numpy(x_np).unsqueeze(0).float().to(solver.device)
            out = solver._forward_losses(x)
            series = out["series"][layer].detach().mean(dim=1).squeeze(0).cpu().numpy()
            maps.append(series.astype(np.float32))
            rows.append({
                "end_date": str(meta["end_date"]),
                "window_start": str(meta["dates"][0]),
                "window_end": str(meta["dates"][-1]),
                "dates": list(meta["dates"]),
            })
            if n % 50 == 0:
                print(f"collected {n}/{len(indices)}", flush=True)
    return rows, maps


def render_video(ticker, ticker_frame, score_frame, rows, maps, out_path, fps, hold_frames):
    ticker_frame = ticker_frame.sort_values("date").reset_index(drop=True)
    score_frame = score_frame.sort_values("date").reset_index(drop=True)
    date_to_score = dict(zip(score_frame["date"].dt.strftime("%Y-%m-%d"), score_frame["score"].astype(float)))
    scores = [date_to_score.get(d.strftime("%Y-%m-%d"), np.nan) for d in ticker_frame["date"]]
    x_full = mdates.date2num(ticker_frame["date"])

    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 0.65, 0.75, 2.2])
    ax_price = fig.add_subplot(gs[0])
    ax_volume = fig.add_subplot(gs[1], sharex=ax_price)
    ax_score = fig.add_subplot(gs[2], sharex=ax_price)
    ax_attn = fig.add_subplot(gs[3])

    add_candles(ax_price, ticker_frame)
    ax_price.set_ylabel("price")
    ax_price.grid(alpha=0.2)

    vol_colors = np.where(ticker_frame["close"] >= ticker_frame["open"], "#8fd19e", "#ff9896")
    ax_volume.bar(ticker_frame["date"], ticker_frame["volume"], color=vol_colors, width=0.8, alpha=0.7)
    ax_volume.set_ylabel("volume")
    ax_volume.grid(alpha=0.2)

    ax_score.plot(ticker_frame["date"], scores, color="#6f4e7c", linewidth=1.0)
    ax_score.set_ylabel("score")
    ax_score.grid(alpha=0.2)

    locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
    ax_score.xaxis.set_major_locator(locator)
    ax_score.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    first = rows[0]
    im = ax_attn.imshow(maps[0], origin="lower", aspect="auto", vmin=0, vmax=1, cmap="viridis")
    cbar = fig.colorbar(im, ax=ax_attn, fraction=0.025, pad=0.01)
    cbar.set_label("series attention weight")
    ax_attn.set_xlabel("key day in 60-day window")
    ax_attn.set_ylabel("query day in 60-day window")

    price_v = ax_price.axvline(pd.to_datetime(first["end_date"]), color="red", linewidth=1.0)
    vol_v = ax_volume.axvline(pd.to_datetime(first["end_date"]), color="red", linewidth=1.0)
    score_v = ax_score.axvline(pd.to_datetime(first["end_date"]), color="red", linewidth=1.0)
    price_span = vol_span = score_span = None
    score_dot, = ax_score.plot([], [], "o", color="red", markersize=4)

    def set_window_ticks(win_dates):
        tick_pos = list(range(0, len(win_dates), max(1, len(win_dates) // 6)))
        if tick_pos[-1] != len(win_dates) - 1:
            tick_pos.append(len(win_dates) - 1)
        labels = [pd.to_datetime(win_dates[i]).strftime("%m-%d") for i in tick_pos]
        ax_attn.set_xticks(tick_pos, labels, rotation=35, ha="right")
        ax_attn.set_yticks(tick_pos, labels)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out_path, fps=fps, codec="libx264", quality=8, macro_block_size=16)
    try:
        for i, (row, attn) in enumerate(zip(rows, maps), 1):
            end = pd.to_datetime(row["end_date"])
            start = pd.to_datetime(row["window_start"])
            end_num = mdates.date2num(end)
            score = date_to_score.get(row["end_date"], np.nan)

            im.set_data(attn)
            set_window_ticks(row["dates"])
            ax_attn.set_title(f"{ticker} | attention window {row['window_start']} to {row['window_end']} | end {row['end_date']}")

            for line in [price_v, vol_v, score_v]:
                line.set_xdata([end, end])
            score_dot.set_data([end], [score])

            for span in [price_span, vol_span, score_span]:
                if span is not None:
                    span.remove()
            price_span = ax_price.axvspan(start, end, color="green", alpha=0.08)
            vol_span = ax_volume.axvspan(start, end, color="green", alpha=0.08)
            score_span = ax_score.axvspan(start, end, color="green", alpha=0.08)

            fig.suptitle(f"AT-State + LinearToken seed0 | {ticker} | frame {i}/{len(rows)} | {row['end_date']}", fontsize=13)
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            for _ in range(max(1, hold_frames)):
                writer.append_data(image)
            if i % 50 == 0:
                print(f"wrote frame {i}/{len(rows)}", flush=True)
    finally:
        writer.close()
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=str(DEFAULT_RUN))
    parser.add_argument("--ticker", default="AKAM")
    parser.add_argument("--out", default=r"tmp\attention_videos\AT-State-LinearToken-AKAM-full-timeline.mp4")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--hold_frames", type=int, default=2)
    parser.add_argument("--max_frames", type=int, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    solver = build_solver(run_dir)
    rows, maps = collect_attention(solver, args.ticker, args.max_frames)

    feature_path = Path("SP500_features_vw60_lw60") / f"{args.ticker}_features.csv"
    ticker_frame = pd.read_csv(feature_path, parse_dates=["date"])
    ticker_frame = ticker_frame[(ticker_frame["date"] >= "2023-01-01") & (ticker_frame["date"] <= "2024-12-31")]
    score_frame = pd.read_csv(run_dir / "test_timeline_scores.csv", parse_dates=["date"])
    score_frame = score_frame[score_frame["ticker"].astype(str).eq(str(args.ticker))]

    render_video(args.ticker, ticker_frame, score_frame, rows, maps, Path(args.out), args.fps, args.hold_frames)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
