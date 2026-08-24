# Financial Prior Anomaly Transformer

An Anomaly Transformer adapted to detect anomalous trading days in S&P 500
stocks, using time (and optionally time+state) priors instead of the original
sensor-series setup.

**Looking for research results, findings, or how to reproduce them?**
Start at `D:\financial-prior-research-paper\journey\README.md`
(see [research_paper/README.md](research_paper/README.md) for the pointer).
That folder is the canonical, checksum-verified record of what was tried,
what was kept, and why. This repo is the code that produces it.

## Core pipeline

```
main.py -> solver.py -> data_factory/data_loader.py -> model/
```

- [main.py](main.py) — CLI entry point (`--mode train|test|visualize`). Parses
  all experiment config (data window, model shape, prior type, score
  definition, detector threshold, plotting).
- [solver.py](solver.py) — training loop, scoring, thresholding/detection, and
  event visualization. This is where prior types and score definitions
  (`--prior_type`, `--score_type`) are actually implemented.
- [data_factory/data_loader.py](data_factory/data_loader.py) — loads raw
  SP500 OHLCV, engineers features (returns, volume z-score, volatility,
  gap, ...), builds windows/labels, and caches engineered features to disk.
  `StockSegLoader` is the only dataset loader this project uses.
- [model/](model/) — the Anomaly Transformer itself
  (`AnomalyTransformer.py`, `attn.py`, `embed.py`).

Run it directly, or via [run_3models.bat](run_3models.bat), which trains/tests/
visualizes three prior configurations (AT-Time, AT-TimeState,
AT-TimeState-ReturnNLL) back to back.

## Supporting scripts (`scripts/`)

Everything under `scripts/` is exploratory/analysis tooling built on top of
the core pipeline's outputs — not part of the trained model itself:

- [scripts/evaluate_baselines.py](scripts/evaluate_baselines.py) — the one
  file other scripts import as a library (baseline model definitions,
  fitting, and metrics). Stays at the top level for that reason.
- `scripts/experiments/` — one-off tooling grouped by purpose: `baselines/`,
  `walkforward/`, `stats/`, `plots/`, and `misc/`.
- [scripts/experiments/run_model_grid.py](scripts/experiments/run_model_grid.py)
  — the single fold/seed runner for AT-Time and AT-TimeState.
- `scripts/experiments/_causal_detector.py` and `_plot.py` — shared detector
  logic and figure styling. Experiment scripts import these instead of
  copying implementations.

One-off scripts live in `scripts/experiments/`. Move a script back up to
`scripts/` only if something else starts importing it as a module.

## Data

`SP500/`, `SP500_features_vw60_lw60/`, `SP500_features_vw60_lw60_z5/` hold raw
and cached engineered features (git-ignored — regenerated from
`data_factory/data_loader.py`, not committed).

## Setup

```
pip install -r requirements-research.txt
```
