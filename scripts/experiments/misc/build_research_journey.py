"""Assemble the paper-facing research journey from reproducible source artifacts."""

import csv
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
JOURNEY = ROOT / "research_paper/journey"
RUN = Path(r"D:\multi-prior-at-run-recon3-contextual\AT-Time")
WEEKLY = ROOT / "research_paper/weekly"
CURRENT = ROOT / "research_paper/results/canonical_event_detector"
SCORE_DISTRIBUTION = ROOT / "research_paper/results/canonical_score_distribution"
SPLIT_AUDIT = ROOT / "research_paper/results/training_split_audit"

STAGES = {
    "00_protocol_and_data": {
        "question": "What data, split, labels, and leakage controls define the experiment?",
        "files": [
            (RUN / "config.json", "data/canonical_config.json", "canonical run configuration"),
            (ROOT / "requirements-research.txt", "requirements-research.txt", "verified package versions"),
            (SPLIT_AUDIT / "canonical_training_split_summary.csv", "data/canonical_training_split_summary.csv", "rows, windows, and contextual-label prevalence by split"),
            (SPLIT_AUDIT / "canonical_training_split_audit.png", "figures/canonical_training_split_audit.png", "chronological split and window audit"),
            (WEEKLY / "2026-W26/dataset_diagnostics/dataset_diagnostic_summary.json", "data/dataset_diagnostic_summary.json", "dataset summary"),
            (WEEKLY / "2026-W26/dataset_diagnostics/label_prevalence_by_split.csv", "data/label_prevalence_by_split.csv", "raw split-level label prevalence"),
        ],
    },
    "01_model_and_baseline_search": {
        "question": "Which model family produces the strongest anomaly ranking?",
        "files": [
            (WEEKLY / "summary_12072026/model_comparison_metrics.csv", "data/model_comparison_metrics.csv", "model-level summary"),
            (WEEKLY / "2026-W26/baseline_comparison_contextual_recon3/all_model_scores.csv", "data/all_model_ticker_day_scores.csv", "raw per-row scores for every compared model"),
            (WEEKLY / "summary_12072026/model_comparison.png", "figures/model_comparison.png", "model comparison"),
            (WEEKLY / "summary_12072026/roc_pr_old_vs_current.png", "figures/roc_pr_comparison.png", "ROC and PR comparison"),
        ],
    },
    "02_feature_and_score_ablation": {
        "question": "Which reconstructed features and score definition should be retained?",
        "files": [
            (WEEKLY / "summary_12072026/feature_ablation_metrics.csv", "data/feature_ablation_metrics.csv", "feature ablation summary"),
            (WEEKLY / "2026-W26/score_ablation/combined_model_metrics.csv", "data/score_ablation_metrics.csv", "score ablation summary"),
            (WEEKLY / "2026-W26/reconstruction_error_diagnostics_recon3_at_time/per_feature_reconstruction_scores.csv", "data/per_feature_reconstruction_errors.csv", "raw per-row per-feature errors"),
            (WEEKLY / "2026-W26/reconstruction_error_diagnostics_recon3_at_time/per_feature_reconstruction_summary.csv", "data/per_feature_reconstruction_summary.csv", "per-feature summary"),
            (WEEKLY / "summary_12072026/feature_ablation_line.png", "figures/feature_ablation.png", "feature ablation"),
            (WEEKLY / "2026-W26/reconstruction_error_diagnostics_recon3_at_time/reconstruction_error_test_roc_pr.png", "figures/reconstruction_error_roc_pr.png", "reconstruction error ROC/PR"),
        ],
    },
    "03_canonical_model": {
        "question": "What exact model produces the score used by every downstream experiment?",
        "files": [
            (RUN / "config.json", "data/config.json", "canonical configuration"),
            (RUN / "metrics.json", "data/metrics.json", "canonical aggregate metrics"),
            (RUN / "train_losses.csv", "data/train_losses.csv", "raw epoch losses"),
            (RUN / "val_timeline_scores.csv", "data/validation_ticker_day_scores.csv", "raw validation scores"),
            (RUN / "test_timeline_scores.csv", "data/test_ticker_day_scores.csv", "raw test scores"),
            (SCORE_DISTRIBUTION / "raw_score_distribution_summary.csv", "data/raw_score_distribution_summary.csv", "raw score distribution summary"),
            (SCORE_DISTRIBUTION / "raw_score_distribution_all_tickers.png", "figures/raw_score_distribution_all_tickers.png", "raw score distribution across all test ticker-days"),
            (WEEKLY / "2026-W26/CANONICAL_MODEL.md", "METHOD.md", "canonical method declaration"),
        ],
    },
    "04_global_threshold_and_shape": {
        "question": "Why does a global Q threshold miss useful temporal score structure?",
        "files": [
            (WEEKLY / "2026-W27/score_spike_shape_results/aligned_raw_score_snapshot.csv", "data/raw_score_snapshot.csv", "raw aligned ticker-day scores"),
            (WEEKLY / "2026-W27/score_spike_shape_results/selected_shape_score_snapshot.csv", "data/shape_score_snapshot.csv", "derived ticker-day shape scores"),
            (WEEKLY / "2026-W27/score_spike_shape_results/metrics.csv", "data/shape_grid_metrics.csv", "complete shape grid"),
            (WEEKLY / "2026-W27/score_spike_shape_results/threshold_sweep.csv", "data/threshold_sweep.csv", "threshold sensitivity"),
            (WEEKLY / "2026-W27/score_spike_shape_results/precision_recall.png", "figures/raw_vs_shape_pr.png", "raw versus shape PR"),
            (WEEKLY / "2026-W27/score_spike_shape_results/height_prominence_width_sensitivity.png", "figures/shape_sensitivity.png", "shape sensitivity"),
            (WEEKLY / "2026-W27/README.md", "METHOD.md", "retrospective shape method and limitation"),
        ],
    },
    "05_causal_local_behavior_detector": {
        "question": "Can causal local peaks plus observable behavior recover events with fewer false alerts?",
        "files": [
            (CURRENT / "validation_grid.csv", "data/validation_grid.csv", "complete validation search"),
            (CURRENT / "behavior_filter_grid.csv", "data/behavior_filter_grid.csv", "complete behavior-filter search"),
            (CURRENT / "selected_results.csv", "data/local_detector_metrics.csv", "selected validation and test result"),
            (CURRENT / "behavior_filter_results.csv", "data/behavior_detector_metrics.csv", "selected behavior result"),
            (CURRENT / "validation_ticker_day_predictions.csv", "data/validation_ticker_day_predictions.csv", "raw validation features, thresholds, and decisions"),
            (CURRENT / "test_ticker_day_predictions.csv", "data/test_ticker_day_predictions.csv", "raw test features, thresholds, and decisions"),
            (CURRENT / "panel_index.csv", "data/per_ticker_confusion.csv", "per-ticker TP/FP/FN"),
            (CURRENT / "precision_recall_grid.png", "figures/local_validation_grid.png", "local validation grid"),
            (CURRENT / "behavior_filter_grid.png", "figures/behavior_validation_grid.png", "behavior validation grid"),
            (CURRENT / "detector_comparison.png", "figures/detector_comparison.png", "test detector comparison"),
            (CURRENT / "README.md", "METHOD.md", "current method, result, and limitations"),
        ],
    },
    "06_ticker_case_studies": {
        "question": "What do successes, false alerts, and missed events look like on real ticker timelines?",
        "files": [
            (CURRENT / "selected_examples/LEN_success.png", "figures/LEN_success.png", "high-TP example"),
            (CURRENT / "selected_examples/AKAM_mixed.png", "figures/AKAM_mixed.png", "mixed familiar example"),
            (CURRENT / "selected_examples/BLDR_false_positive_case.png", "figures/BLDR_false_positive_case.png", "false-positive case"),
            (CURRENT / "selected_examples/AVGO_missed_event_case.png", "figures/AVGO_missed_event_case.png", "missed-event case"),
            (CURRENT / "panel_index.csv", "data/selection_index.csv", "objective example-selection table"),
            (CURRENT / "test_ticker_day_predictions.csv", "data/all_ticker_day_predictions.csv", "raw source for regenerating every panel"),
        ],
    },
}


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def main():
    missing = [str(source) for stage in STAGES.values() for source, _, _ in stage["files"] if not source.is_file()]
    if missing:
        raise FileNotFoundError("Missing journey inputs:\n" + "\n".join(missing))

    JOURNEY.mkdir(parents=True, exist_ok=True)
    manifest = []
    for stage_name, stage in STAGES.items():
        stage_dir = JOURNEY / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        inventory = []
        for source, relative, role in stage["files"]:
            target = stage_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            checksum = digest(target)
            inventory.append((relative, role))
            manifest.append({"stage": stage_name, "artifact": relative, "role": role,
                             "source": str(source), "bytes": target.stat().st_size, "sha256": checksum})
        lines = [f"# {stage_name}\n", stage["question"] + "\n", "## Artifacts\n"]
        lines += [f"- `{path}` — {role}" for path, role in inventory]
        (stage_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with (JOURNEY / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)

    raw = JOURNEY / "05_causal_local_behavior_detector/data/test_ticker_day_predictions.csv"
    with raw.open(newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    observed = {
        "test_alerts": sum(row["final_alert"] == "True" for row in predictions),
        "test_tp": sum(row["true_positive"] == "True" for row in predictions),
        "test_fp": sum(row["false_positive"] == "True" for row in predictions),
        "test_fn": sum(row["missed_event"] == "True" for row in predictions),
    }
    with (CURRENT / "behavior_filter_results.csv").open(newline="", encoding="utf-8") as stream:
        expected = next(csv.DictReader(stream))
    assert all(observed[key] == int(float(expected[key])) for key in observed), (observed, expected)
    (JOURNEY / "README.md").write_text(
        "# Research journey\n\n" +
        "Read the numbered folders in order. Every stage contains its question, method, figures, "
        "summary tables, and the lowest-level CSV currently available. `MANIFEST.csv` records "
        "the original source and SHA-256 of every copied artifact.\n\n"
        "See [`REPRODUCE.md`](REPRODUCE.md) for producer scripts and commands, and "
        "[`CODE_AUDIT.md`](CODE_AUDIT.md) for verified properties and known gaps.\n\n" +
        "## Order\n\n" + "\n".join(f"- [{name}]({name}/README.md) — {stage['question']}"
                                      for name, stage in STAGES.items()) +
        "\n\n## Rebuild\n\n```powershell\n"
        "C:\\Users\\Acer\\miniconda3\\python.exe -u research_paper\\weekly\\2026-W27\\adaptive_local_spike_experiment.py\n"
        "C:\\Users\\Acer\\miniconda3\\python.exe scripts\\build_research_journey.py\n```\n",
        encoding="utf-8")
    print(f"Built {len(STAGES)} stages and {len(manifest)} verified artifacts under {JOURNEY}")
    print(f"Semantic check passed: {observed}")


if __name__ == "__main__":
    main()
