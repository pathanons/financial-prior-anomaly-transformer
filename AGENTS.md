# Agent operating memory for this repo

This file is a guardrail for future Codex sessions. Read it before proposing
experiments or editing code in this repository.

## Baseline and current thesis

- Best current model: `AT-State full + linear token embedding`.
- Treat this as the default baseline unless the user explicitly changes it.
- Under the current fair fold-C contextual-label comparison, this baseline
  beats the checked non-AT baselines: Isolation Forest, One-Class SVM,
  Autoencoder, VAE, LSTM-AE, Transformer-AE, and 1D-CNN-AE.
  - AT-State LinearToken mean PR-AUC is about 0.705.
  - Fair Isolation Forest PR-AUC is about 0.533.
  - LSTM/Transformer/1D-CNN sequence AE baselines are near-random in the
    current implementation, but note their known scoring issue before making a
    strong architectural claim.
- Current thesis: the useful improvement came from simplifying the input token
  projection, not from heavier preprocessing, grouping, norm layers, positional
  tricks, or post-hoc score shortcuts.
- `AT-State` behaves more like a detector of multi-dimensional financial shock
  than a detector of isolated single-feature spikes. High scores tend to occur
  when return, volume, and gap move together.

## Do not repeat failed ideas without explicit user approval

- Do not propose or implement `max(...)`, OR-style, or component-wise shortcut
  scoring to rescue sparse/single-feature anomalies.
  - Includes `max(return_score, volume_score, gap_score)`.
  - Includes calibrated/percentile component OR readouts.
  - Reason: these variants promote marginal single-feature noise, saturate
    scores near 1.0, and hurt PR-AUC/F1/event-F1.
- Do not switch score aggregation from mean to max. It was already tested and
  worsened metrics by inflating noisy windows.
- Do not retry these failed ablations as if they are new ideas:
  - `StateNorm`
  - `ContextState`
  - `MagnitudeInput`
  - `GroupedToken`
  - `LinearTokenNorm`
  - `Integrated positional encoding / IPE sigma=1`
  - `No positional embedding`
  - `CalibratedComponents`

## Required reasoning loop before any new experiment

1. State the current baseline.
2. State which prior attempts are related and why they failed.
3. If the new idea resembles a rejected pattern, stop and ask before doing it.
4. Explain the smallest possible change and what evidence would falsify it.
5. Only then implement or give a run command.

## Code hygiene rules from the user

- Do not create extra files, runners, templates, or helper scripts unless the
  existing pipeline cannot do the job.
- Reuse existing entry points: `main.py`, `solver.py`, existing plot scripts,
  and `scripts/experiments/run_model_grid.py`.
- If logic is duplicated and only display names differ, consolidate it.
- Prefer clear names over clever names.
- If the task is only to analyze/explain/check results, do not edit code.
- If the requested change is ambiguous, ask until the user and agent share the
  same simple implementation picture.
- Keep changes small enough to review. Avoid "million-line" solutions.

## Repo-specific technical facts

- Use `C:\Users\Acer\miniconda3\python.exe`.
- Current comparison split is fold C:
  - train `2018-01-01` to `2021-12-31`
  - val `2022-01-01` to `2022-12-31`
  - test `2023-01-01` to `2024-12-31`
- Main reconstruction features for current AT-State work:
  `log_return_1d,volume_z,gap`.
- Feature cache: `SP500_features_vw60_lw60`.
- Panel plots use `scripts/experiments/plots/plot_at_score_examples.py`.
- Multi-seed fold-C runner uses `scripts/experiments/run_model_grid.py`.

## Reporting rules

- Negative results are useful; record them instead of silently moving on.
- When summarizing results, compare against `AT-State full + linear token`, not
  against stale AT-Time/AT-TimeState numbers unless that comparison is asked.
- Do not sell an idea because it sounds plausible. Tie it to observed evidence.
