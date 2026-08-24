"""Shared causal-detector loading, labeling, and validation selection."""

import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd


DETECTOR_SOURCE = Path(
    r"D:\financial-prior-research-paper\weekly\2026-W27\adaptive_local_spike_experiment.py"
)
EVENT_COLUMNS = {
    "return": "log_return_1d",
    "volume": "volume_z",
    "gap": "gap",
}


def load_detector(feature_dir, source=DETECTOR_SOURCE):
    spec = importlib.util.spec_from_file_location("adaptive_local_spike_experiment", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load causal detector from {source}")
    detector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(detector)
    detector.FEATURES = Path(feature_dir)
    return detector


def add_event_labels(scores, features, train_end, std_multiplier=3.0):
    """Add leakage-safe event labels using only each fold's training period."""
    rows = (
        scores.merge(features, on=["ticker", "date"], how="inner")
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )
    train = features[features["date"] <= train_end]
    extreme_columns = []
    severity_columns = []
    for name, column in EVENT_COLUMNS.items():
        std_column = f"{name}_std"
        extreme_column = f"{name}_extreme"
        rows = rows.join(train.groupby("ticker")[column].std().rename(std_column), on="ticker")
        rows[extreme_column] = rows[column].abs() > std_multiplier * rows[std_column]
        extreme_columns.append(extreme_column)
        severity_columns.append(rows[column].abs() / rows[std_column])
    rows["event_day"] = rows[extreme_columns].any(axis=1)
    rows["severity"] = np.maximum.reduce(severity_columns)
    rows["pos"] = rows.groupby("ticker").cumcount()
    return rows


def load_timeline_scores(run_dir, split, tickers):
    frame = pd.read_csv(Path(run_dir) / f"{split}_timeline_scores.csv", parse_dates=["date"])
    return (
        frame[frame["ticker"].astype(str).isin(tickers)]
        .dropna(subset=["score"])
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )


def prepare_rows(detector, scores, features, train_end):
    rows = add_event_labels(scores, features, train_end)
    return detector.add_behavior_features(detector.add_adaptive_features(rows))


def select_on_validation(detector, validation, test):
    """Select detector parameters on validation, then evaluate test once."""
    local_rows = []
    for values in itertools.product(
        detector.LOOKBACKS,
        detector.LOCAL_ZS,
        detector.PROMINENCE_ZS,
        detector.MAX_WIDTHS,
        detector.TOLERANCES,
    ):
        params = dict(zip(
            ("lookback", "local_z", "prominence_z", "max_width", "tolerance"), values
        ))
        local_rows.append({**params, **detector.evaluate(validation, **params)})

    best = pd.DataFrame(local_rows).sort_values("f1", ascending=False).iloc[0]
    base_params = {
        "lookback": int(best.lookback),
        "local_z": float(best.local_z),
        "prominence_z": float(best.prominence_z),
        "max_width": float(best.max_width),
        "tolerance": int(best.tolerance),
    }
    local_test = detector.evaluate(test, **base_params)

    behavior_rows = []
    for values in itertools.product(
        detector.COOLDOWNS,
        detector.BEHAVIOR_MINS,
        detector.BEHAVIOR_WINDOWS,
    ):
        params = dict(zip(("cooldown", "behavior_min", "behavior_window"), values))
        behavior_rows.append({**params, **detector.evaluate(validation, **base_params, **params)})

    best = pd.DataFrame(behavior_rows).sort_values("f1", ascending=False).iloc[0]
    behavior_params = {
        "cooldown": int(best.cooldown),
        "behavior_min": float(best.behavior_min),
        "behavior_window": int(best.behavior_window),
    }
    return (
        {**base_params, **behavior_params},
        local_test,
        detector.evaluate(test, **base_params, **behavior_params),
    )


def _self_check():
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2021-01-01"])
    features = pd.DataFrame({
        "ticker": ["A"] * 3,
        "date": dates,
        "log_return_1d": [-1.0, 1.0, 10.0],
        "volume_z": [-1.0, 1.0, 0.0],
        "gap": [-1.0, 1.0, 0.0],
    })
    scores = pd.DataFrame({"ticker": ["A"], "date": dates[-1:], "score": [1.0]})
    labeled = add_event_labels(scores, features, "2020-12-31")
    assert labeled["event_day"].tolist() == [True]
    assert labeled["pos"].tolist() == [0]


if __name__ == "__main__":
    _self_check()
