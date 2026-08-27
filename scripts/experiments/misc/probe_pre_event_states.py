import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from solver import Solver, association_point, reconstruction_error


LABELS = {0: "normal", 1: "pre_event", 2: "anomaly"}


def split_classes(dataset, horizon):
    frame = dataset.frame.copy()
    date_col = dataset.date_col
    ticker_col = dataset.ticker_col
    label_col = dataset.label_col
    out = {}
    for ticker, group in frame.sort_values([ticker_col, date_col]).groupby(ticker_col):
        dates = group[date_col].dt.strftime("%Y-%m-%d").tolist()
        labels = group[label_col].astype(int).to_numpy()
        classes = np.zeros(len(group), dtype=int)
        classes[labels == 1] = 2
        for offset in range(1, int(horizon) + 1):
            future = np.zeros_like(labels)
            future[:-offset] = labels[offset:]
            classes[(future == 1) & (classes == 0)] = 1
        out.update({(str(ticker), date): int(cls) for date, cls in zip(dates, classes)})
    return out


def forward_probe_features(solver, input_data):
    x = input_data.float().to(solver.device)
    with torch.no_grad():
        embedded = solver.model.embedding(x)
        x_state = x
        if solver.model.state_projection is not None:
            x_state = x[:, :, solver.model.z_state_indices] if solver.model.z_state_indices else x
            x_state = solver.model.state_projection(x_state)
        hidden, series, prior, _ = solver.model.encoder(embedded, x_state=x_state)
        x_hat = solver.model.projection(hidden)
        rec_point, _, feature_error, _ = reconstruction_error(
            x, x_hat, solver.return_idx, solver.reconstruction_indices
        )
        discrepancy = association_point(series, prior)
        weight = torch.softmax(-discrepancy, dim=1)
        features = torch.cat(
            [
                hidden[:, -1],
                feature_error[:, -1],
                rec_point[:, -1:].detach(),
                discrepancy[:, -1:].detach(),
                weight[:, -1:].detach(),
            ],
            dim=1,
        )
    return features.cpu().numpy()


def extract_split(solver, loader, classes):
    xs, ys, rows = [], [], []
    sample_start = 0
    solver.model.eval()
    for input_data, _ in loader:
        batch_size = len(input_data)
        batch_meta = loader.dataset.metadata[sample_start : sample_start + batch_size]
        sample_start += batch_size
        feats = forward_probe_features(solver, input_data)
        for feat, meta in zip(feats, batch_meta):
            key = (str(meta["ticker"]), str(meta["end_date"]))
            if key not in classes:
                continue
            xs.append(feat)
            ys.append(classes[key])
            rows.append({"ticker": key[0], "date": key[1], "label": classes[key]})
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=int), pd.DataFrame(rows)


def metrics(name, y_true, probs, out_dir):
    pred = probs.argmax(axis=1)
    report = classification_report(
        y_true, pred, target_names=[LABELS[i] for i in range(3)], output_dict=True, zero_division=0
    )
    one_hot = np.eye(3)[y_true]
    result = {
        "split": name,
        "rows": int(len(y_true)),
        "pre_event_auc_pr": float(average_precision_score(one_hot[:, 1], probs[:, 1])),
        "anomaly_auc_pr": float(average_precision_score(one_hot[:, 2], probs[:, 2])),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1, 2]).tolist(),
        "classification_report": report,
    }
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result


def save_top(name, meta, probs, out_dir):
    out = meta.copy()
    out["normal_prob"] = probs[:, 0]
    out["pre_event_prob"] = probs[:, 1]
    out["anomaly_prob"] = probs[:, 2]
    out["label_name"] = out["label"].map(LABELS)
    out.sort_values("pre_event_prob", ascending=False).to_csv(out_dir / f"{name}_top_pre_event_scores.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", default=r"D:\multi-prior-at-run-walkforward-multiseed\AT-State_foldC_seed0_linear_token")
    parser.add_argument("--out_dir", default=r"tmp\pre_event_probe_at_state_linear_token")
    parser.add_argument("--future_horizon", type=int, default=3)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((run_dir / "config.json").read_text())
    config.update({"run_root": None, "model_save_path": str(run_dir / "checkpoints"), "batch_size": args.batch_size})
    config["use_attention_future_loss"] = False
    solver = Solver(config)
    solver.model.load_state_dict(
        torch.load(run_dir / "checkpoints" / "STOCK_checkpoint.pth", map_location=solver.device, weights_only=True)
    )

    train_classes = split_classes(solver.train_loader.dataset, args.future_horizon)
    val_classes = split_classes(solver.vali_loader.dataset, args.future_horizon)
    test_classes = split_classes(solver.test_loader.dataset, args.future_horizon)

    train_loader = DataLoader(solver.train_loader.dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    val_loader = DataLoader(solver.vali_loader.dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(solver.test_loader.dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    x_train, y_train, train_meta = extract_split(solver, train_loader, train_classes)
    x_val, y_val, val_meta = extract_split(solver, val_loader, val_classes)
    x_test, y_test, test_meta = extract_split(solver, test_loader, test_classes)

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=args.max_iter, class_weight="balanced", n_jobs=-1),
    )
    clf.fit(x_train, y_train)
    val_probs = clf.predict_proba(x_val)
    test_probs = clf.predict_proba(x_test)

    results = {
        "classes": LABELS,
        "future_horizon": int(args.future_horizon),
        "train_rows": int(len(y_train)),
        "train_class_counts": {LABELS[i]: int((y_train == i).sum()) for i in range(3)},
        "val": metrics("val", y_val, val_probs, out_dir),
        "test": metrics("test", y_test, test_probs, out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    save_top("val", val_meta, val_probs, out_dir)
    save_top("test", test_meta, test_probs, out_dir)
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
