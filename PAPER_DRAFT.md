# From Reconstruction Scores to Causal Event Detection for Abnormal Trading Days

Working paper draft. Numbers marked **provisional** must not be used as final
claims until the all-ticker and fair-baseline checks are complete.

## Abstract (draft)

Daily financial anomalies are not reliably identified by a fixed threshold on
return alone. We adapt the Anomaly Transformer to daily S&P 500 OHLCV data and
evaluate two priors: a temporal prior (AT-Time) and a temporal-plus-state prior
(AT-TimeState). The model reconstructs a selected set of return, volume, and
overnight-gap channels and produces a point-wise association-weighted score.
Because a raw global quantile threshold ignores local score shape and market
context, we add a validation-selected causal local-peak detector with
observable behavior confirmation from return, volume, and gap. On the frozen
25-ticker evaluation subset, the detector reaches precision 0.668, recall
0.554, and F1 0.606, compared with the corresponding local-only detector's F1
0.578. The result is statistically supported by a ticker-clustered bootstrap
95% CI of [0.562, 0.653] for detector F1, but the detector must still be
expanded to all 111 tickers before publication. Walk-forward multi-seed
experiments show regime and seed sensitivity, so the final paper will report
mean and standard deviation rather than a single favorable run. The study
frames the contribution as a leakage-safe event-detection pipeline, not as a
claim that the Transformer is the strongest raw ranker for every financial
anomaly definition.

## 1. Introduction

### Problem

An abnormal trading day may combine a price jump, an overnight gap, unusual
volume, and a change in local volatility. A single global return threshold can
miss this context and can be poorly calibrated across tickers and regimes.

### Research questions

1. Can an Anomaly Transformer reconstruct a compact return-volume-gap signal
   and rank contextual abnormal trading days?
2. Does a causal local-peak and behavior-confirmation layer improve event
   detection over a global score quantile?
3. How stable are the results across tickers, chronological folds, and random
   seeds?

### Contributions

- A leakage-safe daily OHLCV pipeline with chronological windows and
  contextual event labels.
- A controlled AT-Time/AT-TimeState comparison using the same reconstruction
  protocol.
- A validation-selected causal detector that converts diffuse Transformer
  scores into event alerts using only past score history and observable
  return/volume/gap behavior.
- A negative-results and robustness analysis that separates raw ranking,
  thresholding, detector behavior, and ticker/seed variance.

## 2. Related Work

Cover: Anomaly Transformer; reconstruction-based anomaly detection; financial
price/volume co-movement; robust and heavy-tailed financial statistics; local
peak and changepoint detection. Do not claim that a universal power-law
exponent is used by the model.

## 3. Data and Protocol

- Universe: S&P 500 daily OHLCV, with the exact ticker inclusion recorded in
  `journey/00_protocol_and_data`.
- Primary chronological split: train 2018--2021, validation 2022, test
  2023--2024.
- Window length: 60 trading days.
- Features: log return, multi-horizon return, log-volume z-score, absolute and
  squared return, rolling volatility, volatility ratio, overnight gap, and
  high-low range.
- Primary reconstruction channels: `log_return_1d`, `volume_z`, `gap`.
- Labels: contextual labels combine an abnormal return with abnormal volume
  or volatility; they are evaluation labels, not supervised training targets.
- Leakage controls: shifted rolling volume baselines, train-only scaling,
  split-contained windows, and validation-only detector selection.

## 4. Method

### 4.1 AT-Time and AT-TimeState

The encoder receives a 60-day multivariate window. The canonical AT-Time model
uses a temporal Gaussian prior. AT-TimeState adds a pairwise state-distance
term based on standardized market-state channels.

### 4.2 Reconstruction score

The model reconstructs the selected channels at each timestep. The canonical
point score is:

```text
mean MSE(log_return_1d, volume_z, gap)
× softmax(-symmetric KL(series, prior))
```

This is a reconstruction score, not next-day forecasting.

### 4.3 Causal event detector

1. Apply `log1p` to the raw score.
2. Estimate a causal rolling median/MAD threshold from the previous 15 days.
3. Keep local peaks selected on validation.
4. Confirm candidates using causal robust return/volume/gap behavior.
5. Apply event matching and cooldown rules.

The current paper-facing implementation is documented in
`journey/05_causal_local_behavior_detector/METHOD.md`.

## 5. Results

### 5.1 Canonical raw ranking (111 tickers; provisional point estimate)

| Model | PR-AUC | ROC-AUC | F1 | Event-F1 |
|---|---:|---:|---:|---:|
| AT-Time | 0.455 | 0.984 | 0.443 | 0.494 |
| AT-TimeState | 0.431 | 0.979 | 0.441 | 0.506 |

The single canonical AT-Time run is not sufficient for a final robustness
claim; fold-C fresh seeds produced PR-AUC 0.339 +/- 0.071.

### 5.2 Feature ablation

Reconstructing `log_return_1d + volume_z + gap` produced PR-AUC 0.455,
compared with 0.239 when all 11 channels were reconstructed. This supports a
compact reconstruction target, but does not prove that averaging the three
errors is optimal.

### 5.3 Detector result (25-ticker subset; provisional)

| Detector | Precision | Recall | F1 |
|---|---:|---:|---:|
| Local peaks only | 0.532 | 0.633 | 0.578 |
| Local peaks + behavior | **0.668** | 0.554 | **0.606** |

Bootstrap 95% CI for detector F1: [0.562, 0.653].

### 5.4 Baselines and negative results

Isolation Forest obtains PR-AUC 0.653 in the original comparison and 0.533
under the corrected contextual-label protocol. Raw ranking therefore favors
Isolation Forest. However, when both methods are passed through the same
validation-selected causal detector on the same 25-ticker subset, the event
comparison reverses:

| Method | Detector precision | Detector recall | Detector F1 |
|---|---:|---:|---:|
| AT-Time + adaptive detector | **0.668** | 0.554 | **0.606** |
| Isolation Forest + adaptive detector | 0.492 | **0.600** | 0.541 |

This supports a narrower claim: the temporal score plus behavior-confirmation
layer is better at the evaluated event-level operating point, while Isolation
Forest remains the stronger raw ranker. The comparison still needs to be
expanded from 25 to all 111 tickers before it becomes a final headline claim.

The tested joint reconstruction loss and max window aggregation were dropped
because they reduced performance relative to the canonical mean-score pipeline.

## 6. Discussion

The main empirical result is a detector improvement, not a universal raw-score
win. Local temporal context and behavior confirmation reduce false alerts while
retaining event recall. AT-TimeState is a useful ablation for regime context,
but its advantage is not consistent under the causal detector.

## 7. Limitations

- The detector headline currently covers 25 rather than all 111 tickers.
- The contextual label is a proxy, not universal ground truth.
- Validation 2022 is distributionally atypical relative to train and test.
- Walk-forward folds confound test regime with available training history.
- The current peak extraction implementation needs an explicit disclosure of
  its two-sided peak-width calculation before any real-time claim.

## 8. Required checks before submission

- [ ] Run the selected causal detector on all 111 tickers.
- [ ] Run Isolation Forest through the identical detector and label protocol.
- [ ] Finish the fair sequence-baseline training-step comparison or remove
      those baselines from the main claim.
- [ ] Add paired per-ticker statistical comparisons.
- [ ] Freeze final figures, tables, configs, and a reproducibility command.

## Artifact map

- Protocol: `D:\financial-prior-research-paper\journey\00_protocol_and_data`
- Feature/score ablation: `...\02_feature_and_score_ablation`
- Canonical model: `...\03_canonical_model`
- Causal detector: `...\05_causal_local_behavior_detector`
- Statistical confidence: `...\07_statistical_confidence`
- Walk-forward and baselines: `...\08_walkforward_multiseed`
