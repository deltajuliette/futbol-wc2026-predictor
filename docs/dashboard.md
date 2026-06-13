# Dashboard

Status: **plan**. Framework: **Streamlit** (decided). The dashboard is a *read-only
view over stored tables* — it never computes predictions or holds ad-hoc state. Every
chart states its timestamp and the `model_run_id` it was built from. Every metric has a
definition (link or tooltip) — no undefined numbers shipped.

## Hard data contract
The app reads only from `db/worldcup.sqlite`: `matches`, `predictions`,
`benchmark_predictions`, `odds_snapshots`, `evaluation_metrics`, `team_match_features`.
If a number isn't traceable to a row, it doesn't appear.

## Pages / panels

### 1. Upcoming matches
- List of scheduled fixtures (team, stage, kickoff in UTC + local).
- Per match: **model 1X2** (calibrated), shown as both percentages and visual bars.
- **Implied market** probabilities after overround removal (de-vig method labeled).
- **Opta public** benchmark when available.
- **Uncertainty bands** on each probability — made visually obvious, not buried.
- **Why (reasoning):** a headline + top-3 ranked drivers per fixture, generated
  deterministically from the stored model quantities (`predictions.reasoning_json`),
  not an LLM narrative. Drivers: `strength_gap` (Elo), `goals` (the exact log-linear
  attack/defense/venue split of the expected-goals edge), `shape` (top scoreline,
  draw risk, tempo), `edge` (vs Elo benchmark), `calibration` (raw→cal shift),
  `uncertainty` (thin-sample caveat). Thresholds defining each phrase live in
  `src/explain/reasons.py`. Reproducible: same prediction → identical text.

### 2. Match detail
- Expected goals (λ_home, λ_away).
- **Most-likely scorelines** (top-N from the goal matrix) as a small heatmap/table.
- Derived markets: BTTS, over/under 2.5.
- **Edge vs market**: model − de-vigged market, with sign and magnitude; upset flags.
- For knockouts: separate **progression** probability panel (advance ≠ win-in-90).
- Stamp: model run + timestamp + data freshness.

### 3. Model vs Market vs Opta
- Side-by-side comparison highlighting agreement/disagreement. Neutral language —
  "model assigns 8pp more to the draw than the market," not "model loves the draw."

### 4. Calibration & performance
- Reliability diagram, sharpness, and the proper-score table (log loss / Brier / RPS)
  for the model and each benchmark, from `evaluation_metrics`.
- Backtest history over the tournament as matches accrue.

## Presentation rules (from CLAUDE.md)
- Percentages **and** simple bars together.
- Uncertainty obvious by default.
- Differences vs market/Opta highlighted without sensationalism.
- Every chart: timestamp + model run used.
- Metric definitions always reachable.

## Non-goals
- No in-play / live-updating predictions.
- No writing back to the database from the UI.
- No model fitting in the app process.
