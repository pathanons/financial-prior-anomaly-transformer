import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from torch.utils.data import DataLoader, TensorDataset


DEFAULT_FEATURES = [
    "log_return_1d",
    "return_5d",
    "return_20d",
    "volume_z",
    "abs_return",
    "squared_return",
    "rolling_vol_5",
    "rolling_vol_20",
    "vol_ratio_5_20",
    "gap",
    "high_low_range",
]


class Autoencoder(nn.Module):
    def __init__(self, dim, latent=8):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(dim, 32), nn.ReLU(), nn.Linear(32, latent), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, dim))

    def forward(self, x):
        return self.decoder(self.encoder(x))


class VariationalAutoencoder(nn.Module):
    def __init__(self, dim, latent=8):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(dim, 32), nn.ReLU())
        self.mu = nn.Linear(32, latent)
        self.logvar = nn.Linear(32, latent)
        self.decoder = nn.Sequential(nn.Linear(latent, 32), nn.ReLU(), nn.Linear(32, dim))

    def forward(self, x):
        h = self.base(x)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-8, 8)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return self.decoder(z), mu, logvar


class LSTMAutoencoder(nn.Module):
    def __init__(self, dim, hidden=128, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(dim, hidden, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(hidden, hidden, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden, dim)

    def forward(self, x):
        _, (h, _) = self.encoder(x)
        repeated = h[-1].unsqueeze(1).repeat(1, x.shape[1], 1)
        y, _ = self.decoder(repeated)
        return self.out(y)


class TransformerAutoencoder(nn.Module):
    def __init__(self, dim, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.inp = nn.Linear(dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=256, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.out = nn.Linear(d_model, dim)

    def forward(self, x):
        return self.out(self.encoder(self.inp(x)))


class Conv1DAutoencoder(nn.Module):
    def __init__(self, dim, channels=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(dim, channels, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(channels, dim, kernel_size=5, padding=2),
        )

    def forward(self, x):
        h = self.encoder(x.transpose(1, 2))
        out = self.decoder(h)
        return out.transpose(1, 2)


def read_features(feature_dir, features, label_col):
    frames = []
    for path in sorted(Path(feature_dir).glob("*_features.csv")):
        frame = pd.read_csv(path, parse_dates=["date"])
        if label_col not in frame.columns and label_col == "absolute_z5_label":
            frame[label_col] = (frame["z_return"].abs() >= 5.0).astype(int)
        if label_col not in frame.columns and label_col == "return_volume_3std_label":
            frame[label_col] = ((frame["z_return"].abs() >= 3.0) | (frame["volume_z"].abs() >= 3.0)).astype(int)
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=features + [label_col])
        frames.append(frame[["ticker", "date", label_col] + features])
    if not frames:
        raise FileNotFoundError(f"No *_features.csv files under {feature_dir}")
    return pd.concat(frames, ignore_index=True)


def split_daily(frame):
    return (
        frame[(frame.date >= "2018-01-01") & (frame.date <= "2021-12-31")],
        frame[(frame.date >= "2022-01-01") & (frame.date <= "2022-12-31")],
        frame[(frame.date >= "2023-01-01") & (frame.date <= "2024-12-31")],
    )


def sample_rows(x, max_rows, seed):
    if len(x) <= max_rows:
        return x
    rng = np.random.default_rng(seed)
    return x[rng.choice(len(x), size=max_rows, replace=False)]


def reconstruction_error(recon, x, recon_idx):
    err = (recon - x).pow(2)
    if recon_idx is not None:
        err = err[:, recon_idx] if err.ndim == 2 else err[:, :, recon_idx]
    return err


def metrics_from_scores(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "auc_pr": float(average_precision_score(y_true, y_score)),
        "threshold": float(threshold),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def write_roc_plot(score_df, out_path):
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    fig, ax = plt.subplots(figsize=(8, 6))
    for model, g in score_df.groupby("model", sort=False):
        if g["label"].nunique() < 2:
            continue
        fpr, tpr, _ = roc_curve(g["label"].values.astype(int), g["score"].values.astype(float))
        ax.plot(fpr, tpr, linewidth=1.8, label=f"{model} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="0.6", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC comparison")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_confusion_plot(row, out_path, title):
    import matplotlib.pyplot as plt

    mat = np.array([[row["tn"], row["fp"]], [row["fn"], row["tp"]]])
    fig, ax = plt.subplots(figsize=(4.8, 3.9))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred normal", "Pred anomaly"])
    ax.set_yticks([0, 1], ["Label normal", "Label positive"])
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.subplots_adjust(left=0.28)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_summary_plot(df, out_path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.6), gridspec_kw={"height_ratios": [1.25, 1]})
    metrics = ["auc_roc", "auc_pr", "f1", "precision", "recall"]
    x = np.arange(len(df))
    width = 0.15
    for i, metric in enumerate(metrics):
        axes[0].bar(x + (i - 2) * width, df[metric], width, label=metric)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["model"], rotation=25, ha="right")
    axes[0].set_ylim(0, 1)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(ncols=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.19))
    axes[0].set_title("Detection metrics comparison", pad=34)

    w = 0.35
    axes[1].bar(x - w / 2, df["tp"], w, label="TP (pred anomaly and label positive)", color="#2ca02c")
    axes[1].bar(x + w / 2, df["fp"], w, label="FP (pred anomaly but label normal)", color="#d62728")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["model"], rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False)
    axes[1].set_title("Predicted anomaly overlap with evaluation label")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def append_explainability_row(df, summary_path):
    if not summary_path or not Path(summary_path).exists():
        return df
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    row = {
        "model": "AT-TimeState + reasons",
        "auc_roc": np.nan,
        "auc_pr": np.nan,
        "threshold": summary["threshold_q99_from_validation"],
        "tn": np.nan,
        "fp": summary["n_unexplained"],
        "fn": np.nan,
        "tp": summary["n_explained"],
        "precision": summary["explanation_coverage"],
        "recall": np.nan,
        "f1": np.nan,
    }
    at_idx = df.index[df["model"].eq("AT-TimeState")]
    insert_at = int(at_idx[0]) + 1 if len(at_idx) else len(df)
    return pd.concat([df.iloc[:insert_at], pd.DataFrame([row]), df.iloc[insert_at:]], ignore_index=True)


def at_confusions(run_root, out_dir):
    rows = []
    score_rows = []
    for name in ["AT-Time", "AT-TimeState", "AT-TimeState-ReturnNLL"]:
        run = Path(run_root) / name
        with open(run / "metrics.json", encoding="utf-8") as f:
            threshold = json.load(f)["threshold"]
        scores = pd.read_csv(run / "test_timeline_scores.csv")
        row = {"model": name, **metrics_from_scores(scores.label.values, scores.score.values, threshold)}
        rows.append(row)
        score_rows.append(scores[["score", "label"]].assign(model=name))
        write_confusion_plot(row, Path(out_dir) / f"confusion_{name}.png", name)
    return pd.DataFrame(rows), pd.concat(score_rows, ignore_index=True)


def fit_dense_autoencoder(model, x_train, x_val, x_test, epochs, batch_size, device, recon_idx=None, vae=False):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.tensor(x_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            if vae:
                recon, mu, logvar = model(xb)
                err = reconstruction_error(recon, xb, recon_idx)
                loss = err.mean() + 1e-3 * (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean())
            else:
                err = reconstruction_error(model(xb), xb, recon_idx)
                loss = err.mean()
            loss.backward()
            opt.step()

    def score(x):
        with torch.no_grad():
            xb = torch.tensor(x, dtype=torch.float32, device=device)
            recon = model(xb)[0] if vae else model(xb)
            err = reconstruction_error(recon, xb, recon_idx)
            return err.mean(dim=1).cpu().numpy()

    return score(x_val), score(x_test)


def make_windows(frame, scaler, features, label_col, win=60):
    xs, labels = [], []
    for _, g in frame.sort_values(["ticker", "date"]).groupby("ticker"):
        vals = scaler.transform(g[features].values)
        lab = g[label_col].values.astype(int)
        for end in range(win - 1, len(g)):
            xs.append(vals[end - win + 1 : end + 1])
            labels.append(lab[end])
    return np.asarray(xs, dtype=np.float32), np.asarray(labels, dtype=int)


def fit_sequence_autoencoder(model, x_train, x_val, x_test, epochs, batch_size, device, recon_idx=None):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(torch.tensor(x_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        for (xb,) in loader:
            xb = xb.to(device)
            opt.zero_grad()
            err = reconstruction_error(model(xb), xb, recon_idx)
            loss = err.mean()
            loss.backward()
            opt.step()

    def score(x):
        out = []
        with torch.no_grad():
            for (xb,) in DataLoader(TensorDataset(torch.tensor(x, dtype=torch.float32)), batch_size=batch_size):
                xb = xb.to(device)
                err = reconstruction_error(model(xb), xb, recon_idx)
                out.append(err.mean(dim=(1, 2)).cpu().numpy())
        return np.concatenate(out)

    return score(x_val), score(x_test)


def baselines(args, out_dir):
    features = [x.strip() for x in args.features.split(",") if x.strip()]
    recon_features = [x.strip() for x in args.reconstruction_features.split(",") if x.strip()] if args.reconstruction_features else features
    missing = [name for name in recon_features if name not in features]
    if missing:
        raise ValueError(f"--reconstruction_features not in --features: {','.join(missing)}")
    recon_idx = None if recon_features == features else [features.index(name) for name in recon_features]
    frame = read_features(args.feature_dir, features, args.label_col)
    train, val, test = split_daily(frame)
    scaler = StandardScaler().fit(train[features].values)
    x_train = scaler.transform(train[features].values)
    x_val = scaler.transform(val[features].values)
    x_test = scaler.transform(test[features].values)
    y_val = val[args.label_col].values.astype(int)
    y_test = test[args.label_col].values.astype(int)
    rows = []
    score_rows = []

    train_sample = sample_rows(x_train, args.max_train_rows, args.seed)

    iso = IsolationForest(n_estimators=120, contamination="auto", random_state=args.seed, n_jobs=-1)
    iso.fit(train_sample)
    val_score = -iso.score_samples(x_val)
    test_score = -iso.score_samples(x_test)
    rows.append({"model": "Isolation Forest", **metrics_from_scores(y_test, test_score, np.percentile(val_score, 99))})
    score_rows.append(pd.DataFrame({"model": "Isolation Forest", "score": test_score, "label": y_test}))

    # ponytail: sample OCSVM training; full SP500 daily train is too slow for an RBF kernel.
    svm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.02)
    svm.fit(sample_rows(x_train, min(args.max_train_rows, 20000), args.seed))
    val_score = -svm.score_samples(x_val)
    test_score = -svm.score_samples(x_test)
    rows.append({"model": "One-Class SVM", **metrics_from_scores(y_test, test_score, np.percentile(val_score, 99))})
    score_rows.append(pd.DataFrame({"model": "One-Class SVM", "score": test_score, "label": y_test}))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    x_train_deep = sample_rows(x_train, args.max_train_rows, args.seed)
    for name, model, vae in [
        ("Autoencoder", Autoencoder(len(features)), False),
        ("Variational Autoencoder", VariationalAutoencoder(len(features)), True),
    ]:
        val_score, test_score = fit_dense_autoencoder(
            model, x_train_deep, x_val, x_test, args.epochs, args.batch_size, device,
            recon_idx=recon_idx, vae=vae
        )
        rows.append({"model": name, **metrics_from_scores(y_test, test_score, np.percentile(val_score, 99))})
        score_rows.append(pd.DataFrame({"model": name, "score": test_score, "label": y_test}))

    seq_train, _ = make_windows(train, scaler, features, args.label_col, args.win_size)
    seq_val, y_seq_val = make_windows(val, scaler, features, args.label_col, args.win_size)
    seq_test, y_seq_test = make_windows(test, scaler, features, args.label_col, args.win_size)
    seq_train = sample_rows(seq_train, args.max_sequence_rows, args.seed)
    for name, model in [
        ("LSTM Autoencoder", LSTMAutoencoder(len(features))),
        ("Transformer Autoencoder", TransformerAutoencoder(len(features))),
        ("1D-CNN Autoencoder", Conv1DAutoencoder(len(features))),
    ]:
        val_score, test_score = fit_sequence_autoencoder(
            model, seq_train, seq_val, seq_test, args.epochs, args.batch_size, device,
            recon_idx=recon_idx
        )
        rows.append({"model": name, **metrics_from_scores(y_seq_test, test_score, np.percentile(val_score, 99))})
        score_rows.append(pd.DataFrame({"model": name, "score": test_score, "label": y_seq_test}))

    df = pd.DataFrame(rows)
    for row in rows:
        write_confusion_plot(row, Path(out_dir) / f"confusion_{row['model'].replace(' ', '_')}.png", row["model"])
    return df, pd.concat(score_rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_dir", default="SP500_features_vw60_lw60")
    parser.add_argument("--features", default=",".join(DEFAULT_FEATURES))
    parser.add_argument("--reconstruction_features", default="")
    parser.add_argument("--label_col", default="absolute_label")
    parser.add_argument("--run_root", default="D:/multi-prior-at-run")
    parser.add_argument("--out_dir", default="research_paper/weekly/2026-W26/baseline_comparison")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--max_train_rows", type=int, default=60000)
    parser.add_argument("--max_sequence_rows", type=int, default=20000)
    parser.add_argument("--win_size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--explainability_summary", default="")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    at_df, at_scores = at_confusions(args.run_root, out_dir)
    at_df.to_csv(out_dir / "at_confusion_metrics.csv", index=False)
    base_df, base_scores = baselines(args, out_dir)
    base_df.to_csv(out_dir / "baseline_metrics.csv", index=False)
    all_df = pd.concat([at_df, base_df], ignore_index=True)
    all_df = append_explainability_row(all_df, args.explainability_summary)
    all_df.to_csv(out_dir / "all_model_metrics.csv", index=False)
    all_scores = pd.concat([at_scores, base_scores], ignore_index=True)
    all_scores.to_csv(out_dir / "all_model_scores.csv", index=False)
    write_summary_plot(all_df, out_dir / "baseline_comparison_summary.png")
    write_roc_plot(all_scores, out_dir / "roc_comparison.png")
    print(all_df.to_string(index=False))


if __name__ == "__main__":
    main()
