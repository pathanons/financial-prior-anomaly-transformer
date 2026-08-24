from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "research_paper" / "weekly" / "summary_12072026"
CANONICAL = Path(r"D:\multi-prior-at-run-recon3-contextual\AT-Time")


def feature_ablation():
    runs = [
        ("Full 11", Path(r"D:\multi-prior-at-run-contextual\AT-Time")),
        ("Return+Volume", Path(r"D:\multi-prior-at-run-recon2-contextual\AT-Time")),
        ("Gap only", Path(r"D:\multi-prior-at-run-recon-gap-contextual\AT-Time")),
        ("Return+Volume+Gap", CANONICAL),
    ]
    rows = []
    for name, run in runs:
        config = pd.read_json(run / "config.json", typ="series")
        metrics = pd.read_json(run / "metrics.json", typ="series")
        rows.append({
            "variant": name,
            "reconstruction_features": config.get("reconstruction_features", "all 11") or "all 11",
            "auc_pr": metrics.auc_pr,
            "auc_roc": metrics.auc_roc,
            "f1": metrics.f1,
            "event_f1": metrics.event_f1,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "feature_ablation_metrics.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for metric, label, marker in [("auc_pr", "PR-AUC", "o"), ("f1", "F1", "s"), ("event_f1", "Event-F1", "^")]:
        ax.plot(frame.variant, frame[metric], marker=marker, linewidth=2, label=label)
    ax.set_ylim(0, 0.55)
    ax.set_ylabel("Score")
    ax.set_title("Feature reconstruction ablation (same contextual protocol)")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "feature_ablation_line.png", dpi=180)
    plt.close(fig)
    return frame


def model_comparison():
    source = ROOT / "research_paper" / "weekly" / "2026-W26" / "baseline_comparison_contextual_recon3" / "all_model_metrics.csv"
    frame = pd.read_csv(source).sort_values("auc_pr")
    frame.to_csv(OUT / "model_comparison_metrics.csv", index=False)

    colors = ["#d62728" if name == "AT-Time" else "#4c78a8" for name in frame.model]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].barh(frame.model, frame.auc_pr, color=colors)
    axes[0].set_title("PR-AUC comparison")
    axes[0].set_xlabel("PR-AUC")
    axes[1].barh(frame.model, frame.f1, color=colors)
    axes[1].set_title("F1 comparison")
    axes[1].set_xlabel("F1")
    for ax in axes:
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=180)
    plt.close(fig)
    return frame


def old_vs_current_curves():
    runs = [
        ("Previous: Full 11", Path(r"D:\multi-prior-at-run-contextual\AT-Time"), "#7f7f7f"),
        ("Current: Return+Volume+Gap", CANONICAL, "#d62728"),
    ]
    curves = []
    rows = []
    labels_ref = None
    for name, run, color in runs:
        timeline = pd.read_csv(run / "test_timeline_scores.csv")
        metrics = pd.read_json(run / "metrics.json", typ="series")
        labels = timeline.label.to_numpy()
        if labels_ref is None:
            labels_ref = labels
        else:
            assert (labels_ref == labels).all()
        fpr, tpr, _ = roc_curve(labels, timeline.score)
        precision, recall, _ = precision_recall_curve(labels, timeline.score)
        curves.append((name, color, fpr, tpr, recall, precision, float(metrics.auc_roc), float(metrics.auc_pr)))
        rows.append({
            "version": name,
            "auc_roc": metrics.auc_roc,
            "auc_pr": metrics.auc_pr,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "event_f1": metrics.event_f1,
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "old_vs_current_metrics.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for name, color, fpr, tpr, recall, precision, roc_auc, pr_auc in curves:
        axes[0].plot(fpr, tpr, color=color, linewidth=2, label=f"{name} (AUC={roc_auc:.4f})")
        axes[1].plot(recall, precision, color=color, linewidth=2, label=f"{name} (AP={pr_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], "--", color="0.65", linewidth=1)
    axes[0].set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="ROC: previous vs current")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall: previous vs current")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "roc_pr_old_vs_current.png", dpi=180)
    plt.close(fig)

    for name, color, fpr, tpr, _, _, roc_auc, _ in curves:
        slug = "old_full11" if name.startswith("Previous") else "current_recon3"
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.plot(fpr, tpr, color=color, linewidth=2.2, label=f"ROC-AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="0.65", linewidth=1)
        ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=name)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(OUT / f"roc_curve_{slug}.png", dpi=180)
        plt.close(fig)
    return frame


def canonical_four_panels(ticker="AKAM"):
    scores = pd.read_csv(CANONICAL / "test_timeline_scores.csv", parse_dates=["date"])
    scores = scores[scores.ticker.eq(ticker)].copy()
    features = pd.read_csv(ROOT / "SP500_features_vw60_lw60" / f"{ticker}_features.csv", parse_dates=["date"])
    rows = features.merge(scores[["date", "score"]], on="date", how="inner")
    rows = rows[rows.date.between("2023-01-01", "2024-12-31")].sort_values("date")
    threshold = rows.score.quantile(0.99)
    return_std = rows.log_return_1d.std()
    extreme = rows.log_return_1d.abs() > 3 * return_std
    predicted = rows.score >= threshold

    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=True)
    axes[0].plot(rows.date, rows.log_return_1d, linewidth=1, label="log_return_1d")
    axes[0].axhline(3 * return_std, color="orange", linestyle="--", label="+/-3 std")
    axes[0].axhline(-3 * return_std, color="orange", linestyle="--")
    axes[0].scatter(rows.loc[extreme, "date"], rows.loc[extreme, "log_return_1d"], color="orange", s=25, label="return extreme")
    axes[1].plot(rows.date, rows.volume_z, color="#8c564b", linewidth=1, label="volume_z")
    axes[2].plot(rows.date, rows.gap, color="#17becf", linewidth=1, label="gap")
    axes[3].plot(rows.date, rows.score, color="#6f4e7c", linewidth=1, label="canonical anomaly score")
    axes[3].axhline(threshold, color="red", linestyle="--", label="per-ticker Q99 (visualization)")
    axes[3].scatter(rows.loc[predicted, "date"], rows.loc[predicted, "score"], color="red", s=28, label="Q99 prediction")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left")
    axes[0].set_ylabel("Log return")
    axes[1].set_ylabel("Volume Z")
    axes[2].set_ylabel("Gap")
    axes[3].set_ylabel("Score")
    axes[3].set_xlabel("Date")
    fig.suptitle(f"Canonical AT-Time Recon3: {ticker} (2023-2024)")
    fig.tight_layout()
    fig.savefig(OUT / f"canonical_{ticker}_4panels.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ablation = feature_ablation()
    comparison = model_comparison()
    old_vs_current_curves()
    canonical_four_panels()
    source = ROOT / "research_paper" / "weekly" / "2026-W26" / "at_score_q99_log_return_1d_std_bounds" / "AKAM_score_close.png"
    shutil.copy2(source, OUT / "canonical_AKAM_detailed.png")
    expected = 0.4548558785768767
    assert abs(ablation.loc[ablation.variant.eq("Return+Volume+Gap"), "auc_pr"].iat[0] - expected) < 1e-12
    assert abs(comparison.loc[comparison.model.eq("AT-Time"), "auc_pr"].iat[0] - expected) < 1e-12
    print(f"Created summary figures and tables in {OUT}")


if __name__ == "__main__":
    main()
