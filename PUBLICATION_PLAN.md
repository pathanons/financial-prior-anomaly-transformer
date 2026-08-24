# Publication Plan

## 0. Paper drafting — STARTED (2026-08-08)

- [x] Created the evidence-first working draft: [PAPER_DRAFT.md](PAPER_DRAFT.md).
- [x] Locked the paper framing: the contribution is the leakage-safe causal
      event detector, not a claim that the Transformer is the best raw ranker.
- [ ] Keep headline numbers provisional until the all-ticker detector and fair
      Isolation Forest comparison are complete.

Latest baseline correction: the existing
`scripts/experiments/baselines/run_baselines_causal_detector.py` now evaluates Isolation Forest with
the same validation-selected causal detector on the same 25-ticker subset.
AT-Time + detector F1 is 0.606 versus Isolation Forest + detector F1 0.541.
The all-111-ticker version remains open.

Running log of what's been reviewed, decided, and still open on the path to an
IEEE submission. Keep scope to what's listed here — if a new idea comes up
mid-work, add it below before chasing it, don't just go do it.

Research findings/results themselves stay in
`D:\financial-prior-research-paper\journey`. This file tracks the *decisions
and action items* around getting to a submittable paper. See
[RESEARCH_LOG.md](RESEARCH_LOG.md) for the chronological history of what's
been tried and why — check it before re-investigating something that feels
new; it may have already been resolved (or dropped) once before.

## 1. Repo hygiene — DONE (2026-07-29)

- [x] Added root `README.md`: core pipeline (`main.py -> solver.py ->
      data_factory/data_loader.py -> model/`) vs. supporting `scripts/`.
- [x] Removed dead code inherited from the original Anomaly-Transformer
      template, never used by this project: `PSMSegLoader`, `MSLSegLoader`,
      `SMAPSegLoader`, `SMDSegLoader` in `data_factory/data_loader.py`;
      `scripts/{MSL,PSM,SMAP,SMD,Start}.sh`; `pics/{result,structure}.png`.
- [x] Fixed stale `main.py` argparse defaults (`--dataset credit` ->
      `STOCK`, `--data_path` -> `SP500`).

## 2. Reviewer pass on current results — DONE, action items open

Reviewed `journey/00-06`, `METHOD.md` files, `metrics.json`,
`model_comparison_metrics.csv`, `CODE_AUDIT.md`.

Critical issues found, not yet fixed:

- [ ] **Isolation Forest beats AT-Time as a raw ranker** (PR-AUC 0.653 vs
      0.455, ROC-AUC 0.996 vs 0.984, `01_model_and_baseline_search`). Never
      discussed in any narrative doc — only sits in a CSV. Must be addressed
      head-on before writing the paper (see item 4 — likely the baseline
      comparison itself is unfair before concluding IF is genuinely better).
- [ ] **Evaluation population mismatch**: model comparison (01/03) runs on
      all 111 tickers; the causal detector (05/06), the actual headline
      result, runs on only 25 hand-picked tickers. Isolation Forest has never
      been run through the causal-detector pipeline at all. Need IF (and
      ideally the fixed baselines from item 4) evaluated through the exact
      same 05 pipeline, same ticker set, for a fair head-to-head.
- [ ] Expand causal-detector eval from 25 -> all 111 tickers (already
      tracked as blocker #1 in `CODE_AUDIT.md`).
- [ ] Stages 01/02/03 (model family, feature subset, score type) all appear
      to have been chosen by looking directly at test-set metrics across
      many configs, not selected on validation only. Stage 05 does this
      correctly (validation-select, evaluate test once) — 01-03 should be
      redone with the same discipline before any number there is called final.

## 3. Adaptive threshold / causal detector — this is the real contribution

- [x] Confirmed: this is already `journey/05_causal_local_behavior_detector`,
      the current paper-facing result. Not a side experiment to fold in —
      it's the core story.
- [ ] **Numbers mismatch**: `weekly/2026-W27/findings_14072026.md` reports
      F1 0.6282 (behavior-filtered); frozen `journey/05` reports F1 0.6057.
      Difference is a real leakage fix (per-split ±3σ stats -> train-only
      stats). **Only ever cite the 0.6057 journey number in the paper.**
- [ ] Peak shape measurement (`scipy.find_peaks` prominence/width) looks at
      both sides of a candidate peak — not fully causal, despite the
      "causal" name. Documented in the weekly log (§10.3) but missing from
      `journey/CODE_AUDIT.md`. Add it there; decide whether to fix it before
      claiming any "near-real-time" framing, or just disclose the limitation.
- [ ] Port the negative-results section (§9) and limitations (§10) from
      `findings_14072026.md` into the official journey (e.g. into
      `05_causal_local_behavior_detector/METHOD.md` or a LIMITATIONS.md) so
      they don't stay buried in a weekly log a future reader won't open.

## 4. Baseline fairness fix (LSTM-AE / Transformer-AE / VAE)

Diagnosed why these score near-random (PR-AUC ~0.01) in
`scripts/evaluate_baselines.py`:

- Toy capacity: `LSTMAutoencoder(hidden=24)`, `TransformerAutoencoder(d_model=32,
  nhead=4, 1 layer)` — vs. this project's own AT-Time (`d_model=512,
  n_heads=16, e_layers=4`) and vs. the original Anomaly Transformer paper
  (Xu et al., ICLR 2022: `d_model=512, 3 layers, 8 heads` — matches this
  repo's own `main.py` argparse defaults exactly). Baseline Transformer is
  ~16x smaller in `d_model` than either.
- Fixed `--epochs 3` for every deep baseline regardless of architecture.
- Default `--reconstruction_features ""` makes baselines reconstruct all 11
  raw features, while AT-Time's own ablation shows 3 features
  (`log_return_1d,volume_z,gap`) roughly doubles PR-AUC over reconstructing
  all 11 (0.455 vs 0.239) — and the exact CLI used to produce
  `01_model_and_baseline_search`'s numbers was never preserved, so we can't
  confirm which setting the baselines actually got.

Action items:

- [x] Rerun with `--reconstruction_features log_return_1d,volume_z,gap` AND
      `--label_col contextual_label` (done 2026-07-31,
      `scripts/experiments/baselines/run_baselines_fold_c.py`). **Isolation Forest PR-AUC dropped
      0.653 → 0.533** once forced onto `contextual_label` — strong evidence
      the original comparison used a different label for baselines than for
      AT-Time, a second apples-to-oranges bug beyond the reconstruction
      features one.
- [x] Bumped LSTM-AE (hidden 24→128, 2 layers), Transformer-AE (d_model
      32→128, 2 layers), added a new **1D-CNN-AE** baseline, epochs 3→15.
      **Did not fix it**: LSTM-AE/Transformer-AE/1D-CNN-AE are still at
      PR-AUC 0.016/0.015/0.032 — barely above the pre-fix numbers. Root
      cause: capacity wasn't the (only) problem — total gradient steps are
      still ~600 (`max_sequence_rows=20000 / batch_size=512 * epochs=15`)
      vs. AT-Time's own ~22,288 steps. **Still need to fix**: shrink
      `batch_size` and/or raise `max_sequence_rows` and/or add more epochs
      so the sequence baselines get a comparable number of weight updates,
      not just a comparable epoch count.
- [x] **Gradient-step fix tried, negative result (2026-08-10)**: reran with
      `batch_size=32, epochs=7, max_sequence_rows=110000` (no subsampling,
      full train set) — exactly matches AT-Time's own ~22,288 steps. PR-AUC
      barely changed: LSTM-AE 0.0165, Transformer-AE 0.0144, 1D-CNN-AE
      0.0138 (1D-CNN actually *worse* than before). Gradient-step count was
      never the bottleneck. See `RESEARCH_LOG.md` 2026-08-10 row.
- [ ] **Real bug found while ruling out gradient steps**: `evaluate_baselines.py`'s
      `fit_sequence_autoencoder.score()` does `err.mean(dim=(1, 2))` —
      averages reconstruction error across the *whole 60-day window* (all
      timesteps + features) into one scalar and uses it as the score for
      only the window's last day. A single anomalous day gets diluted ~60x.
      The dense `Autoencoder`/`VAE` baselines score single rows directly (no
      window averaging) and get PR-AUC 0.178/0.190 in the same run despite
      far less capacity — direct evidence this is the real cause, not
      capacity or training budget. AT-Time's own `solver.py` never does this
      (`rec_point` keeps per-timestep error; `aggregate_window_scores` means
      per-*day* across overlapping windows, not across days within one
      window). **Not yet fixed**: rewrite sequence-baseline scoring to use
      per-timestep (last-position, or overlapping-window-aggregated like
      AT-Time) error, then rerun before trusting any LSTM-AE/Transformer-AE/
      1D-CNN-AE number.
- [ ] Re-check whether Isolation Forest still wins after the scoring-formula
      fix above (gradient steps are ruled out, this is the next candidate
      fix) — if yes, lean into item 3 (detector mechanism) as the
      contribution rather than "best raw scorer." (Already true at PR-AUC
      0.533 vs. AT-Time fold-C fresh-seed range 0.284-0.419 — IF is still
      ahead even after both label and reconstruction-feature fixes.)
- [ ] (Optional, cheap sanity check before/after the rerun) Inspect
      reconstruction outputs directly for near-constant/collapsed predictions
      to confirm the undertraining hypothesis empirically, since no direct
      literature citation was found for that specific mechanism (see item 6).

## 5. Statistical rigor plan

Priority order, cheapest/most-confirming first:

- [x] **Ticker-clustered bootstrap CI** on the frozen canonical test scores —
      done 2026-07-29 via `scripts/experiments/stats/bootstrap_confidence.py`, output at
      `journey/07_statistical_confidence`. Canonical ranker PR-AUC 95% CI is
      [0.388, 0.530] — Isolation Forest's 0.653 point estimate sits **outside**
      that interval, so the gap isn't ticker-sampling noise. Sharpens item 2,
      doesn't soften it.
- [x] **Add `--seed` to `main.py`/`solver.py`** — done 2026-07-29.
      `main.py` seeds `random`, `numpy`, and `torch` (+ `cuda`) before
      building the `Solver`; new `--seed` arg defaults to 0. Caveat:
      `cudnn.benchmark = True` stays on (left as-is for training speed), so
      bit-exact reproducibility across reruns of the *same* seed isn't
      guaranteed — but weight init, dropout, and data-loader shuffling are
      now controlled, which is what multi-seed comparison actually needs.
- [x] **Multi-seed retrain** (3 seeds, same split) -> report mean ± std.
      Blocker #2 in `CODE_AUDIT.md`. **Done 2026-07-30** (combined with
      walk-forward below): AT-Time x 3 folds x seeds {0,1,2} = 9 real
      retrains via `scripts/experiments/run_model_grid.py`. First attempt had
      a bug (missing `--label_type contextual`, silently used the wrong
      label) — deleted and rerun correctly, see `RESEARCH_LOG.md`.
      **Result: fold C (current setup) PR-AUC = 0.339 ± 0.071 across seeds
      (0.314/0.284/0.419) — the frozen canonical's reported 0.4549 sits above
      all 3 fresh seeds.** Report mean±std in the paper, not the single
      0.4549 point estimate. Plots at
      `journey/08_walkforward_multiseed/figures/walkforward_multiseed_stability.png`.
- [x] **Walk-forward / rolling-origin folds** — done 2026-07-30, combined
      with the multi-seed run above (`journey/08_walkforward_multiseed`).
      PR-AUC (threshold-free) across folds: A 0.224±0.083, B 0.331±0.022,
      C 0.339±0.071.
      **Important correction**: the first pass of this analysis used
      solver.py's default Q99 event-F1, which understates every fold
      (that's the whole reason journey/05 replaced Q99 with a real
      detector). Re-ran with the actual frozen causal detector
      (`scripts/experiments/walkforward/evaluate_causal_detector.py --selection frozen`) — **F1 becomes A
      0.432±0.093, B 0.353±0.125, C 0.585±0.048.** The regime gap is real
      but much smaller than Q99 suggested, and fold B has the highest
      seed-to-seed variance (one seed was a clear outlier). Caveat: folds
      A/B also have less training data (2-3 years vs fold C's 4) — can't
      yet cleanly separate "harder regime" from "less training data."
      Comparison chart: `journey/08_walkforward_multiseed/figures/q99_vs_causal_detector_by_fold.png`.
      **Then properly redone**: re-selected detector parameters on each run's
      OWN validation instead of reusing one frozen set
      (`scripts/experiments/walkforward/evaluate_causal_detector.py --selection validation`). F1 became A
      0.353±0.031, B 0.379±0.082, C 0.593±0.061 — conclusion mostly holds
      (C still clearly best), but fold A's reused-params number (0.432) was
      an overstatement. Two new findings to act on: (1) every one of the 9
      runs selected `lookback=20`, the grid's max — widen the grid
      (10/15/20/30/40?) before trusting 20 as optimal; (2) fold A's
      validation year is 2020 (COVID crash), where nearly every day looks
      behaviorally extreme, so validation selected `behavior_min=0` (no
      behavior filter) for all 3 seeds — the detector's own hyperparameters
      are regime-sensitive, not just the raw scores.
- [x] **Checked whether a single validation year is representative** — it
      isn't. Done 2026-07-29 via `scripts/experiments/stats/compare_feature_distributions.py`
      (KS test on `log_return_1d`/`volume_z`/`gap` across splits, using the
      raw cached `SP500_features_vw60_lw60/*_features.csv`). Train vs. test
      are fairly close (KS stat 0.02-0.04); **validation (2022) is the
      outlier — more different from both train and test (KS stat 0.08-0.12)
      than train and test are from each other.** All config/threshold
      decisions so far were made against this one atypical year. Output at
      `journey/07_statistical_confidence/data/feature_distribution_*.csv`.
      This is the concrete evidence behind prioritizing walk-forward above.
- [ ] **Paired statistical test** (e.g. Wilcoxon signed-rank across
      per-ticker AUC-PR) instead of comparing pooled single numbers when
      claiming one model beats another.

## 6. Literature-backing notes

Credibility of the "baseline is undersized, not genuinely worse" claim from
item 4, broken into three tiers so it doesn't get overstated later:

**Verified directly (not inference)** — hyperparameters checked against the
source paper this repo forks from, Anomaly Transformer (Xu et al., ICLR
2022): `d_model=512, 3 encoder layers, 8 heads`. This repo's own
`main.py` argparse defaults match it exactly (`d_model=512, n_heads=8,
e_layers=3`); the actual canonical run goes even bigger
(`e_layers=4, n_heads=16`). The baseline `TransformerAutoencoder` in
`scripts/evaluate_baselines.py` uses `d_model=32, 1 layer, nhead=4` — about
16x smaller in `d_model` and a third of the layers, measured against the
*published literature config*, not just against this project's own model.

**Literature-adjacent but a different mechanism** — "Autoencoders for
Anomaly Detection are Unreliable" (arXiv:2501.13864) proves theoretically
that reconstruction-loss-based anomaly scoring is fundamentally shaky
(anomalies far from normal data can still reconstruct perfectly). Good
supporting citation for "naive AE reconstruction-error baselines are known
to be unreliable in general" — but it does not address undertraining or
model capacity specifically, so don't cite it as if it explains *why* the
LSTM-AE/Transformer-AE baselines here score near-random.

**Still just inference, no citation found** — the specific mechanism
"undertrained model -> converges to reconstructing a near-constant output ->
reconstruction error carries no anomaly signal." Searched for this and only
found the opposite-direction finding in the literature (training *too long*
lets the model reconstruct anomalies too, which also kills detection). This
claim should be labeled as domain reasoning from experience, not an
established fact, until verified.

**How to actually verify it** (stronger than hunting for more citations):
inspect the trained LSTM-AE/Transformer-AE reconstruction outputs directly
and check the variance of predicted values across timesteps — if it's
collapsed near-constant relative to the real input variance, that's direct
evidence from this project's own data. Not yet done (see item 4's optional
check).

**Verified, directly explains the IF-vs-AT direction reversal** — read the
full Anomaly Transformer paper (arXiv:2110.02642v5, local copy at
`C:\Users\Acer\Documents\cmu\research\2110.02642v5.pdf`) cover to cover.
Their Table 1 (p.7) shows Isolation Forest as one of the *worst* baselines
(F1 53.64-68.62% across 5 datasets vs. their model's 92-98%) — opposite of
our result. Two paper-cited reasons this doesn't transfer to our setup: (1)
their evaluation uses **point-adjustment** (p.6 — one detected point credits
the whole contiguous anomaly segment), which structurally favors temporal
models over row-independent ones like IF, and our per-day `contextual_label`
evaluation has no such adjustment; (2) their anomalies are **persistent
multi-timestep faults** (sensor/server failures lasting many consecutive
readings) vs. our **single-day point events** — a structural mismatch, not
an implementation flaw. Their anomaly rate (4.2-27.8%, Table 13) is also far
higher than ours (~0.3-0.4%). This is citable, defensible material for the
paper's discussion section explaining the IF result honestly rather than
treating it as just a weakness.

Sources:
- [Anomaly Transformer, ICLR 2022 (Tsinghua mirror)](https://ise.thss.tsinghua.edu.cn/~mlong/doc/anomaly-transformer-iclr22.pdf)
- [Anomaly Transformer, ICLR 2022 (author PDF)](https://wuhaixu2016.github.io/pdf/ICLR2022_Anomaly.pdf)
- [thuml/Anomaly-Transformer (original repo this project forked)](https://github.com/thuml/Anomaly-Transformer)
- [Autoencoders for Anomaly Detection are Unreliable (arXiv:2501.13864)](https://arxiv.org/pdf/2501.13864)
- Anomaly Transformer full text (arXiv:2110.02642v5) — local copy: `C:\Users\Acer\Documents\cmu\research\2110.02642v5.pdf`

## Open questions (not decided yet)

- How many walk-forward folds are actually worth the GPU time?
- Does the paper's contribution get framed as "detector mechanism" (item 3)
  or do we still try to win on raw ranking too?
