# Modeling

Status: **plan**. Build order is strictly simplest-first; a layer is added only if
it beats the calibrated baseline on proper scoring rules (see [evaluation.md](evaluation.md)).
Training and scoring code stay in separate modules. Every fit writes a
`model_runs` row with params, feature-set version, training window, and git sha.

## Prediction targets (in priority order)
1. Pre-match **1X2** probabilities (home / draw / away).
2. **Expected goals** per team (λ_home, λ_away).
3. **Scoreline distribution** (full goal matrix) → most-likely scorelines.
4. Derived: **BTTS**, **over/under**, upset flag, **edge vs market**.
5. Knockout only: **progression** probability (who advances after ET/pens).

Focus is pre-match, not in-play.

---

## Layer 1 — Elo baseline

A transparent team-strength rating, World Cup-aware.

- **Update:** `R' = R + K · (S − E)` where expected
  `E_home = 1 / (1 + 10^(−(R_home − R_away + H)/400))`.
- **Neutral venues:** home advantage `H = 0` for neutral matches (most WC games);
  apply `H` only when a team is the genuine host.
- **Margin of victory:** scale `K` by a goal-difference multiplier (damped so blowouts
  don't over-update), e.g. the standard `log(GD+1)` style adjustment.
- **Match importance:** higher `K` weight for competitive internationals vs friendlies.
- **Priors:** initialize from confederation/historical baselines only where justified;
  otherwise a common starting rating that converges over the historical window.
- **1X2 from Elo:** map rating diff to win/draw/loss via a draw-aware function
  (e.g. ordered logit or a calibrated mapping fit on historical results), since raw
  Elo gives win-expectancy, not a three-way split.

Elo also produces `elo_pre`/`elo_diff` features consumed by later layers.

## Layer 2 — Poisson goals with Dixon-Coles

The core scoreline engine.

- **Strengths:** each team has attack `α_i` and defense `β_i`; home advantage `γ`
  (zero at neutral). Expected goals:
  - `λ_home = exp(α_home + β_away + γ_home)`
  - `λ_away = exp(α_away + β_home)`
- **Scoreline matrix:** `P(home=x, away=y)` over a goal grid (0..N) from independent
  Poisson, then apply the **Dixon-Coles low-score correction** `τ(x,y)` to fix the
  dependence in 0-0 / 1-0 / 0-1 / 1-1. This matters because draw probability lives
  in exactly those cells; plain bivariate Poisson misprices draws.
- **Time decay:** weight historical matches by recency (exponential half-life) when
  fitting strengths, so current form is reflected without overreacting to tiny samples.
- **Outputs from one matrix:** 1X2 (sum the win/draw/loss regions), expected goals,
  top-N scorelines, BTTS, over/under — all internally consistent.

## Layer 3 — Covariates (only if they earn it)

Candidate additions, each validated against the calibrated baseline:
- Rest days / congestion, travel proxy.
- Recent **xG for/against** form (from the optional xG source).
- **Market residual** features (for residual analysis / blending — only when explicitly
  asked to blend; market stays a benchmark by default).
Keep the feature set small and robust before any high-dimensional experiments.
Strict anti-leakage: features use only information available before kickoff, stamped
with `as_of_utc`.

## Layer 4 — Knockout handling

Separate target from regulation 1X2.
- **Regulation result** uses the same goal model.
- **Progression** = P(win in 90) + P(draw in 90)·P(advance via ET+pens). Model ET as a
  shortened-time Poisson and penalties as near-coin-flip with a small favorite lean
  (estimated from historical shootouts). Stored distinctly so "win the match" and
  "advance" never get conflated.

## Layer 5 — Calibration

Raw model probabilities are calibrated on held-out, time-respecting folds:
- **Multiclass** calibration for 1X2 (temperature scaling or per-class isotonic / Platt),
  chosen by held-out log loss + reliability.
- Persist **both** `*_raw` and `*_cal` in `predictions`.
- Refit calibration as the tournament accrues finished matches (see runbooks).

## Baselines to always carry
- **Elo-only** and **rank-only** (FIFA/seed rank) models, plus **market de-vig**, as
  reference lines in every evaluation. A new layer must beat the calibrated Elo+DC
  baseline on log loss / Brier / RPS — not on "more winners called."

## Artifacts per run
Saved under `models/artifacts/<model_run_id>/`: fitted parameters, feature-set version,
training window, calibration object, and an evaluation report in `models/reports/`.
