# Evaluation & Benchmarking

Status: **plan**. The project optimizes **probability quality and calibration**, not
single-match accuracy. We never call a model better because it "picked more winners"
over a short run — only proper scoring rules and calibration diagnostics decide.

## Metrics (all written to `evaluation_metrics`)

- **Log loss** (primary): `−(1/N) Σ log p(actual outcome)`. Punishes confident misses.
- **Brier score** (multiclass): mean squared error between the probability vector and
  the one-hot outcome, summed over H/D/A.
- **Ranked Probability Score (RPS)**: the right metric for an **ordered** 1X2 outcome
  (home > draw > away as a result spectrum). Rewards probability mass placed *near* the
  truth, not just on it. Often omitted elsewhere — we include it.
- **Reliability / calibration table:** bin predictions, compare mean predicted vs
  observed frequency; render as a reliability diagram. Stored as `calibration_json`.
- **Sharpness:** average distance of predictions from the base rate — reported *alongside*
  calibration so we can see the calibration↔sharpness trade-off, never sharpness alone.

## Validation scheme

- **Time-respecting only.** No random K-fold — that leaks future info. Use rolling-origin
  / expanding-window backtests: train on matches before date `t`, score matches at `t`,
  advance. Calibration is fit on earlier folds and applied forward.
- **As-of reproducibility:** features and ratings for date `t` must be reconstructable
  from stored tables for that cutoff (fixed-as-of-date test in the suite).

## Benchmarks (stored in `benchmark_predictions`)

1. **Market de-vig** — convert bookmaker 1X2 odds to probabilities and remove overround.
   Implement at least two methods and label which is used:
   - *Proportional / basic normalization* (`1/odds`, then divide by the sum).
   - *Shin / power* method (accounts for favorite-longshot bias more honestly).
   Use **closing or latest** price as the benchmark line.
2. **Opta Analyst public** probabilities, when a public page is available.
3. **Naive baselines:** Elo-only, rank-only (FIFA/seed).

The model is reported *relative to* these — agreement and disagreement shown without
sensational language. Market is a benchmark, **not** ground truth.

## Acceptance gates (per model run)

A model run is "shippable to the dashboard" only if:
- 1X2 probability triplets sum to 1 (±tol) for every prediction.
- It does not regress vs the **calibrated Elo+Dixon-Coles** baseline on log loss / Brier / RPS
  over the backtest window.
- Calibration table shows no gross miscalibration (predicted ≈ observed within bins).
- An evaluation report is written to `models/reports/<model_run_id>/`.

## Reporting

Each evaluation emits: the metric row(s), a reliability diagram, a sharpness figure, and
a head-to-head table (model vs market vs Opta vs naive). Every figure is stamped with the
`model_run_id` and the evaluation `as_of_utc`.
