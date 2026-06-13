# Runbooks

Status: **plan** (commands are the intended interface; scripts not built yet).
Each `scripts/` entrypoint will expose a CLI and an example invocation in its module
docstring, mirrored here.

## 0. Environment setup (one-time)
System Python is 3.9.6 — too old. Create a dedicated venv on Python 3.11/3.12.

```bash
# Option A (preferred): uv — install first if missing
#   curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .            # from pyproject.toml

# Option B: stdlib venv + pip (needs a 3.11/3.12 interpreter on PATH)
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .

cp .env.example .env           # then fill in FOOTBALL_DATA_API_KEY
git init                       # repo is not yet versioned
```

## 1. Bootstrap the database
```bash
python -m scripts.etl.init_db          # creates db/worldcup.sqlite (idempotent)
```

## 2. Load historical bootstrap data
Two ways to get real internationals (anchors Elo + Poisson strengths):

```bash
# No API key needed — public CC0 CSV (martj42/international_results):
python -m scripts.etl.pull_open_results --since 2010

# Or load any CSV you already have (date, competition, home_team, away_team, ...):
python -m scripts.etl.load_intl_results --path data/raw/intl_results/results.csv
```
Both land an immutable raw snapshot and stage into `matches`.

## 3. Daily/live ETL (during the tournament)
```bash
python -m scripts.etl.pull_fixtures   --competition world_cup_2026
python -m scripts.etl.pull_results    --competition world_cup_2026
python -m scripts.etl.pull_odds       --competition world_cup_2026   # append-only
```
Idempotent: re-running upserts fixtures/results and inserts (never overwrites) odds.
Respect `429`/`Retry-After`; raw payloads cached under `data/raw/<source>/<run_id>/`.

## 4. Features → train → predict
```bash
python -m scripts.features.build_features --as-of 2026-06-13T00:00:00Z
python -m scripts.modeling.train_elo
python -m scripts.modeling.train_dixon_coles
python -m scripts.modeling.predict --competition world_cup_2026   # writes predictions
```
Every fit writes a `model_runs` row; predictions are versioned by `model_run_id`.

## 5. Calibration refit (as matches finish)
```bash
python -m scripts.modeling.fit_calibration --as-of <date>
```
Refits the calibration layer on accrued finished matches; updates `*_cal` columns.

## 6. Evaluate
```bash
python -m scripts.evaluation.backtest --window expanding
python -m scripts.evaluation.report   --model-run <id>
```
Writes `evaluation_metrics` rows + reliability/sharpness figures to `models/reports/`.

## 7. Dashboard
```bash
streamlit run app/dashboard/app.py
```
Reads only from `db/worldcup.sqlite`.

## 8. Daily refresh (one command)
Chains pull → features → train → predict (~45s). Idempotent; safe to re-run.
```bash
python -m scripts.update            # pull latest WC results/fixtures, retrain, re-predict
python -m scripts.update --skip-pull   # retrain/predict on current data (no API call)
python -m scripts.update --backtest    # also refresh evaluation metrics

# Or via the logging wrapper (writes data/update.log):
./scripts/daily_update.sh
```
Not scheduled — run manually when you want fresh forecasts. To automate later, point
cron or a launchd LaunchAgent at `scripts/daily_update.sh`.

---

## Troubleshooting
- **Schema drift in a scraped source:** parser raises loudly (we never swallow errors).
  Inspect the cached raw snapshot under `data/raw/<source>/<run_id>/`, update the single
  selector/parser module, add a regression test with the captured payload.
- **Rate limited (429):** back off per `Retry-After`; reduce pacing in client config.
- **xG source down:** pipeline still runs — xG is optional enrichment behind `XGSource`.
- **Probabilities don't sum to 1:** insert validation will reject; check the goal-matrix
  normalization / calibration step.
- **football-data key missing/invalid:** set `FOOTBALL_DATA_API_KEY` in `.env`.
