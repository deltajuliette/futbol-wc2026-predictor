# Methodology Memo — World Cup 2026 Match Forecasting

**To:** Project stakeholders (technical and non-technical)
**From:** Modeling / engineering
**Date:** 2026-06-13
**Re:** How the forecasting system produces, calibrates, benchmarks, and explains its
pre-match probabilities

---

## 0. Executive summary

The system produces **pre-match probabilities** for every World Cup fixture: home
win / draw / away win (1X2), each team's expected goals, the most likely scorelines,
and derived markets (both-teams-to-score, over/under 2.5). It is built to be
**well-calibrated and reproducible**, not to maximize headline "winners picked."

The pipeline is a four-stage assembly line:

1. **Rate every team's strength** from ~11,850 historical internationals (Elo).
2. **Convert strengths into a goal model** and expand it into a full scoreline grid
   (Dixon-Coles / Poisson).
3. **Calibrate** the raw probabilities so the stated confidence is honest.
4. **Grade** everything with proper scoring rules, and **benchmark** it against the
   betting market and an Elo-only baseline.

A fifth, read-only layer attaches a **deterministic plain-English explanation** to
each forecast, derived entirely from the stored model quantities.

Every number shown anywhere is traceable to a stored database row or a saved model
artifact. Nothing is computed at display time.

---

## 1. Data foundation

**Plain English.** The World Cup is only ~64 matches — far too few to judge how good
each team is. So the *spine* of the system is the long history of international
football; the tournament fixtures sit on top of that.

**Specifics.**
- **Historical internationals (~11,852 matches):** public, CC0-licensed dataset
  (`martj42/international_results`), loaded via
  `scripts/etl/pull_open_results.py`. No API key required.
- **Tournament fixtures & results (68 World Cup 2026 fixtures):** football-data.org
  v4 API (`X-Auth-Token` header), via `scripts/etl/pull_fixtures.py`. Idempotent
  upserts; undecided knockout slots ("Winner Group A") are skipped until teams are
  known.
- **Storage:** SQLite via SQLAlchemy 2.0 (`db/worldcup.sqlite`), schema portable to
  Postgres. Ten tables (`matches`, `teams`, `odds_snapshots`, `predictions`,
  `benchmark_predictions`, `model_runs`, `evaluation_metrics`, …). Raw payloads are
  immutable; odds are **append-only** (we never overwrite a historical price); model
  outputs are **versioned** by `model_run_id`.
- **Provenance:** every ingested row carries `source`, `source_url`, `ingested_at`,
  `run_id`. Timestamps are UTC.
- **Anti-leakage:** features and training only ever use information available
  **before** kickoff (`src/features/build.py`).

---

## 2. Stage 1 — Team strength (Elo)

**Plain English.** Elo is the chess-style rating idea. Every team carries one number.
When two teams play, the result that was *expected* (from the rating gap) is compared
to what *actually happened*; the bigger the surprise, the bigger the rating move.

**Specifics** (`src/models/elo.py`).
- Update rule: `R' = R + K · m · (S − E)`, where `S ∈ {1, 0.5, 0}` is the result,
  `E = 1 / (1 + 10^(−(ΔR + H)/400))` is the expected score, **K = 24**, base rating
  **1500**, and home bonus **H = 60** Elo points — applied **only at non-neutral
  venues** (most World Cup games are neutral, so H is switched off).
- **Margin-of-victory damping:** `m = ln(|goal difference| + 1)`. A 5–0 moves a
  rating more than a 1–0, but with diminishing returns, so freak blowouts don't
  distort ratings.
- **Elo → 1X2 benchmark:** rather than a bare win-expectancy, we fit a multinomial
  logistic regression of outcome on `(rating difference, home flag)`, giving a
  calibrated three-way split. This is stored as the `elo_only` benchmark, **not** as
  the primary forecast.

---

## 3. Stage 2 — Goal model (Dixon-Coles) and the scoreline grid

This is the heart of the system, and it is what lets us output expected goals and
scorelines rather than only win/draw/loss.

### 3a. The goal model

**Plain English.** Goals arrive somewhat randomly, like raindrops — you can't predict
the exact count, but you can predict the *rate*. We give each team an **attack** dial
(how many they tend to score) and a **defense** dial (how many they tend to concede),
plus a shared baseline and a home bonus. A team's expected goals in a match =
baseline + its attack − the opponent's defense (+ home bonus if not neutral).

**Specifics** (`src/models/dixon_coles.py`). The model is **log-linear**:

```
log(λ_home) = intercept + attack_home − defense_away + home_adv · (venue is not neutral)
log(λ_away) = intercept + attack_away − defense_home
```

- **Fitting:** weighted maximum likelihood, optimized with SciPy `L-BFGS-B`
  (`maxiter = 250`). Identifiability handled by centering attack/defense at zero each
  evaluation, plus a small **ridge penalty (1e-3)** that also stabilizes teams with
  few matches.
- **Time decay:** older matches count less, via exponential decay with a
  **half-life of 1095 days (~3 years)** (a match that old counts half as much as a fresh
  one). This keeps the ratings current without throwing away history. The half-life was
  tuned out-of-fold — see §3c. An optional `tournament_weight` lever can up-weight
  World Cup-stage games on top of the decay; it is **off by default** because it does not
  help out-of-fold (also §3c).
- **Dixon-Coles low-score correction (`ρ`, bounded to ±0.18):** plain Poisson slightly
  misprices the very common 0–0 / 1–0 / 0–1 / 1–1 results; `ρ` nudges exactly those
  four cells.
- **A note on the optimizer:** the numeric-gradient search reports
  `converged = False` because it hits a bounded iteration budget on the flat,
  regularized objective. This is benign — the expected-goals and probability outputs
  are stable well before the strict gradient tolerance would be met, which we verified
  by comparing against longer runs. The iteration count is logged for transparency.

### 3b. From rates to a full scoreline distribution

**Plain English.** "Expecting 1.8 goals" doesn't mean exactly 1.8 will be scored. We
compute the probability of 0, 1, 2, … goals for each side, combine them into a grid of
*every* plausible scoreline, then read everything off that one grid.

**Specifics** (`src/models/scoreline.py`).
- Build the `(11 × 11)` matrix `P(home = x, away = y)` (`max_goals = 10`) from two
  independent Poisson margins, apply the Dixon-Coles `τ` correction to the low-score
  block, and renormalize.
- Read off: **1X2** (sum the home-win / draw / away-win regions), **expected goals**
  (row/column means), **most likely scorelines** (top cells), **BTTS**, and
  **over/under 2.5**. All outputs come from one coherent object, so they are mutually
  consistent.

### 3c. Do recent (in-tournament) results change the forecasts?

**Question.** Once World Cup group games are played, how much do they move the
predictions for the games still to come — and should recent games count for *more*?

**How recent games enter.** The training corpus is the public international-results feed
(`scripts/etl/pull_open_results.py`), which already carries the live World Cup games under
stage `FIFA World Cup`. So every finished tournament match feeds the next retrain
automatically. (The `world_cup_2026` table is only the fixture list to *predict*; its rows
stay `scheduled` and are never trained on, which is what keeps the same game from being
counted twice.)

**Ablation (`scripts/evaluation/recency_impact.py`).** We fit two otherwise-identical
models — one trained *excluding* the 54 played WC games, one *including* them — and
compared their calibrated forecasts for the 12 upcoming fixtures (as of 2026-06-26).
At the production half-life the games move forecasts **modestly**: mean total-variation
distance **≈0.015** (about 1.5 percentage points of probability mass shifted per match),
max ≈0.034, and **zero** fixtures changed their most-likely outcome. Biggest movers were
matches involving teams whose group form diverged from their long-run rating
(e.g. DR Congo, Croatia). This is expected: a handful of games per team is a small sample
against years of history.

**Tuning the recency lever (`scripts/evaluation/recency_sweep.py`).** We swept the
time-decay half-life and the optional tournament weight in a time-respecting rolling-origin
backtest, scoring held-out matches overall **and** on the tournament subset:

| half-life (days) | log loss (all) | log loss (tournament) |
|---|---|---|
| 365 | 0.9030 | 0.8018 |
| 540 (previous default) | 0.8984 | 0.7977 |
| 730 | 0.8968 | 0.7960 |
| **1095 (new default)** | **0.8963** | **0.7956** |
| 1460 | 0.8966 | 0.7961 |
| 2000 | 0.8972 | 0.7971 |

Up-weighting tournament games (`tournament_weight` 4× / 8×) made log loss **worse** at
every half-life, so it stays off. The best half-life is an interior optimum at **~1095
days**, consistent across log loss, Brier, and RPS on both the overall and tournament sets.

**Takeaway (counterintuitive but honest).** Recent results do shift the forecasts, but
trying to make them count *more* — a shorter half-life or a tournament up-weight — *hurts*
calibration out-of-fold. The best forecasts come from a slightly **longer** memory than
before (540 → 1095 days), i.e. recent games should count a little *less*, not more. The
improvement is small (~0.002 log loss) but consistent, so we adopted it.

---

## 4. Stage 3 — Calibration (honest confidence)

**Plain English.** A model can pick winners and still lie about its confidence.
Calibration asks: *when we say "70%," does it happen about 70% of the time?* If our
70%s only come true 60% of the time, we rein them in. This is the principle the whole
project is built around.

**Specifics** (`src/models/calibration.py`). A multiclass calibrator (multinomial
logistic regression on the log-probabilities) maps **raw → calibrated** 1X2
probabilities. We store **both** sets: `*_raw` (straight from the goal model) and
`*_cal` (honesty-adjusted). The calibrated probabilities are what we display and
trust. In scoring (Stage 4) we fit calibration **out-of-fold** so the evaluation is
not flattered by training on its own test data.

---

## 5. Stage 4 — Benchmarking and evaluation

### 5a. Benchmarks

**Plain English.** We sanity-check ourselves against the betting market and a simple
Elo-only baseline. The market is a *strong reference*, **not ground truth**.

**Specifics** (`src/evaluation/devig.py`). Bookmaker odds are padded so the three
outcomes sum to more than 100% (the "overround"). We remove that padding two ways —
**proportional** and **Shin** (which accounts for informed money) — to recover fair
implied probabilities. The dashboard's **edge** column is simply *our probability minus
the benchmark's*.

### 5b. Scoring rules

**Plain English.** We never grade ourselves on "how many winners we picked" — over 64
matches that's mostly luck. We use scores that reward *honest probabilities*.

**Specifics** (`src/evaluation/metrics.py`).
- **Log loss** — punishes confident wrong calls hardest (headline metric).
- **Brier score** — mean squared error between predicted probability and outcome.
- **Ranked Probability Score (RPS)** — football-aware: ordering matters, so predicting
  a win when it was a draw is penalized less than when it was a loss.
- **Reliability table** — the calibration check, plotted as a curve (dots on the
  diagonal = well calibrated).
- **Sharpness** — how decisive the forecasts are. The goal is to be as sharp as
  possible *while staying calibrated*.

### 5c. Backtest design

**Plain English.** We only let the model learn from matches that happened *before* the
one it's predicting — never peeking at the future.

**Specifics** (`scripts/evaluation/backtest.py`). Rolling-origin (time-respecting)
splits with out-of-fold calibration.

### 5d. Headline results (latest backtest)

| Model | Log loss | Interpretation |
|---|---:|---|
| **Calibrated Dixon-Coles** (`dc_cal`) | **0.878** | the production forecast |
| Calibrated DC + confederation (`dc_cal_conf`) | 0.878 | evaluated, **not shipped** (see §8) |
| Raw Dixon-Coles (`dc_raw`) | 0.882 | before calibration |
| Elo-only benchmark | 1.014 | simple baseline |
| Uniform (1/3 each) | 1.099 | blind guessing |

Lower is better (2,975 held-out matches, 3 rolling-origin folds, half-life 1095 days,
trained through the 2026-06-24 results). The calibrated goal model beats both baselines,
and calibration both improves the scores and reduces overconfidence (sharpness). Sample
real forecasts pass the smell test (e.g., Spain favored over Uruguay; France over Norway).

---

## 6. Stage 5 — Qualitative reasoning (the "why")

**Plain English.** Each forecast comes with a short, plain-English explanation of *why*
the model landed where it did — generated from the model's own numbers, not written by
a language model. So the text is reproducible and every sentence traces to a quantity.

**Specifics** (`src/explain/reasons.py`). The key enabler is that Dixon-Coles is
log-linear, so the home side's expected-goals edge decomposes **exactly**:

```
log(λ_home) − log(λ_away)
   = (attack_home − attack_away)     ← attacking edge
   + (defense_home − defense_away)   ← defensive edge
   + home_adv · (venue is not neutral) ← venue
```

Each forecast gets a headline plus up to six **ranked drivers**:
`strength_gap` (Elo), `goals` (the exact split above + where each team's ratings sit in
the field), `shape` (top scoreline, draw risk, tempo), `edge` (vs the Elo benchmark),
`calibration` (raw → calibrated shift), and `uncertainty` (a thin-sample caveat when a
team has few recent matches). The thresholds that turn numbers into words (e.g. what
counts as a "clear favorite") are named constants in one file — they *are* the
definitions. Stored in `predictions.reasoning_json`, versioned by `model_run_id`.

---

## 7. Reproducibility, testing, and provenance

- **One-command refresh** (`scripts/update.py`): pull → features → train → predict,
  ~45s, fully idempotent. `--skip-pull` and `--backtest` flags available.
- **Read-only dashboard** (Streamlit, `app/dashboard/app.py`): renders only stored
  tables; never recomputes. Every panel is stamped with its `model_run_id` and
  timestamps.
- **Tests:** 64 automated tests. Notably, the reasoning layer has a *rigor* test that
  asserts the goals decomposition equals the model's own `log(λ_home/λ_away)` to
  machine precision, plus a reproducibility test (same prediction → byte-identical
  text).
- **Versioning:** every model fit writes a `model_runs` row; predictions and
  benchmarks are versioned and never overwritten.

---

## 8. Assumptions, caveats, and what's next

**Assumptions / caveats (stated plainly).**
- **Training corpus vs fixture list (no double-counting).** Live World Cup results enter
  training through the international-results feed (stage `FIFA World Cup`); the
  `world_cup_2026` table is only the list of fixtures to *predict* and its rows stay
  `scheduled`. Training reads all `finished` matches, so flipping `world_cup_2026` rows to
  `finished` (e.g. via the football-data `pull_fixtures` path) would count each played WC
  game twice. The refresh therefore uses `pull_open_results`, and the recency study relies
  on this invariant. Already-played fixtures still marked `scheduled` are dropped from the
  dashboard by a wall-clock guard.
- Team strength is inferred from results only; **no lineup, injury, or xG inputs are
  currently in the model** — so the reasoning layer never cites them (that would be
  fiction). xG enrichment is scaffolded behind an adapter for later.
- **Confederation correction — evaluated, not shipped.** A cross-confederation
  relative-strength term (correcting the weak rating linkage between pools that rarely
  meet) is fully implemented, tested, and gated behind `--confederation`. On the earlier
  *synthetic* training data it did not beat the calibrated baseline (Δlog-loss ≈ +7e-5,
  a hair worse), and the large coefficients seen at weak regularization were overfitting
  the bound. On the **real** results dataset a backtest shows it marginally *helping*
  (Δlog-loss ≈ −4e-4) — a sign flip, but still tiny and from a single run. Per "avoid
  complexity unless it clearly beats calibrated baselines," production still runs with it
  **off** pending a proper re-evaluation (multiple seeds/folds) now that the data is real.
  Teams' confederations are populated (`scripts/etl/populate_confederations.py`) from a
  curated reference map; CONIFA/unaffiliated entities are intentionally left NULL.
- **Team-identity dedup.** The two ingest sources spelled several teams differently
  (e.g. "Czechia"/"Czech Republic"), and resolution keyed only on the name slug, so
  duplicates were created — leaving four World Cup sides (Cape Verde, DR Congo, Czech
  Republic, Bosnia) scored as rating-less averages. Team resolution is now alias-aware
  (`team_aliases`), and `scripts/etl/merge_duplicate_teams.py` merged the splits onto
  the canonical, history-bearing team. This materially corrected those forecasts
  (e.g. Spain v Cape Verde 98%→87%).
- **Real data, real fixtures.** Team strength is fit on ~11.8k real international
  results (2014→present, martj42 CC0 dataset via `scripts/etl/pull_open_results.py`),
  and the World Cup fixtures to predict are the actual schedule, derived from the cached
  football-data.org pull by `scripts/etl/build_wc_fixtures.py` into
  `data/reference/wc2026_fixtures.csv`. (Earlier the pipeline trained on a synthetic
  generator and fabricated fixtures by randomly pairing the strongest synthetic teams —
  producing nonsense like "Qatar vs Brazil" and Germany rated below league-average;
  `make_sample_data` is now retained only for deterministic offline tests.)
- **Thin-sample filter (`--min-matches`, default 25).** The public results dataset
  includes CONIFA/non-FIFA micro-nations (Tibet, Menorca, Chagos Islands, …) that play
  few games against each other and otherwise acquire wildly inflated attack/defense
  ratings. `fit_dixon_coles(min_matches=...)` drops any team below the threshold (and the
  matches involving it) before fitting; 25 removes ~83 such teams while keeping every WC
  side (the thinnest, New Zealand, has 81 matches). Note this does not fully solve
  cross-pool rating identifiability — FIFA minnows that beat weaker regional opponents
  but never face top sides can still rank surprisingly high in the raw attack−defense
  table; this does not distort head-to-heads between well-connected teams (the
  confederation correction in the bullet above targets the same weak-linkage problem).
- Neutral-venue handling is applied throughout (all World Cup fixtures are marked
  neutral; host-nation home advantage for USA/Canada/Mexico is **not** modeled), but
  **knockout-specific dynamics** (extra time, penalties) are not yet modeled as a
  separate target.
- The market benchmark depends on odds being ingested; where odds are absent, only the
  Elo benchmark is shown.
- The optimizer's `converged = False` flag is benign (see §3a) but is called out for
  honesty.

**Natural next steps.**
- Add covariates (rest days, recent xG form) once xG ingestion is wired.
- Model knockout progression (advance ≠ win-in-90) as a distinct output.
- Accumulate live results through the tournament to refresh calibration and the
  backtest as real matches finish.

---

*All figures in this memo are reproducible from `db/worldcup.sqlite` (model run #5) via
`python -m scripts.update`. File references point to the committed source.*
