# World Cup Forecast

Calibrated pre-match probability forecasts for the FIFA World Cup — home/draw/away,
expected goals, and scoreline distributions — benchmarked against market-implied odds
and public Opta Analyst probabilities, published in a Streamlit dashboard.

This is a **probabilistic forecasting** project: it optimizes for calibration and proper
scoring rules (log loss, Brier, RPS), reproducibility, and traceability — not headline
single-match accuracy. See [`CLAUDE.md`](CLAUDE.md) for principles and [`docs/`](docs/)
for the design.

## Status
**End-to-end pipeline working on real data** (Elo + calibrated Dixon-Coles →
evaluation → Streamlit). Team strength is fit on ~11.8k real international results
(2014→present, martj42 CC0 dataset, no API key needed) and predictions cover the real
World Cup 2026 schedule. Docs:
- [Data sources](docs/data_sources.md) · [Schema](docs/database_schema.md) ·
  [Modeling](docs/modeling.md) · [Evaluation](docs/evaluation.md) ·
  [Dashboard](docs/dashboard.md) · [Runbooks](docs/runbooks.md) · [Roadmap](docs/roadmap.md)

In a backtest, calibrated Dixon-Coles beats the Elo-only and uniform baselines on
log loss, Brier, and RPS.

**Recent updates.** Training now uses **real international results** (`pull_open_results`,
the martj42 CC0 dataset) rather than the synthetic generator, so ratings reflect real
football (e.g. Germany 92% to beat Curaçao, Spain 86% over Cape Verde). A `--min-matches`
filter (default 25) drops CONIFA/non-FIFA micro-nations that otherwise acquire inflated
ratings from thin samples. The World Cup fixtures to predict are the **real tournament
schedule** (`data/reference/wc2026_fixtures.csv`, derived from the cached football-data.org
pull by `scripts.etl.build_wc_fixtures`); an earlier synthetic slate randomly paired the
strongest synthetic teams and produced impossible matchups like "Qatar vs Brazil".
`make_sample_data` remains for deterministic offline tests but is no longer the training
default. The model is refreshed through the **2026-06-24** results; an out-of-fold study of
how in-tournament games affect forecasts moved the time-decay half-life from 540 to **1095
days** and confirmed that up-weighting tournament games does not help (see
`docs/methodology.md` §3c, `scripts/evaluation/recency_impact.py`, `recency_sweep.py`).

Team-identity resolution is now alias-aware (`team_aliases`), with
a merge step that collapses sources' duplicate spellings onto one canonical, history-
bearing team — fixing sides that were otherwise scored as league-average. A
cross-confederation relative-strength correction is implemented, tested, and gated
behind `--confederation`; it did **not** beat the calibrated baseline out-of-fold, so
production runs with it off (see [methodology §8](docs/methodology.md)).

```bash
# After install (see Quickstart), reproduce the whole pipeline:
python -m scripts.etl.init_db
python -m scripts.etl.pull_open_results --since 2014    # real international results (CC0, no API key); loads into matches
python -m scripts.etl.build_wc_fixtures                 # real WC fixtures from cached pull (checked-in CSV is the fallback)
python -m scripts.etl.load_intl_results --path data/reference/wc2026_fixtures.csv
python -m scripts.etl.merge_duplicate_teams            # collapse split team identities
python -m scripts.etl.populate_confederations          # fill teams.confederation
python -m scripts.features.build_features
python -m scripts.modeling.train_dixon_coles --half-life 1095   # --min-matches 25 (default) drops CONIFA minnows; --confederation for the gated experiment
python -m scripts.modeling.predict --competition world_cup_2026
python -m scripts.evaluation.backtest --folds 4
streamlit run app/dashboard/app.py
# Offline/deterministic alternative to pull_open_results (synthetic history, for tests):
#   python -m scripts.etl.make_sample_data --seed 7 && \
#   python -m scripts.etl.load_intl_results --path data/raw/intl_results/results.csv
```

## Quickstart

```bash
# 1. Environment (Python 3.12 via uv; system 3.9 is too old)
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dashboard,dev]"     # add ",xg" for soccerdata enrichment

# 2. Configure
cp .env.example .env                      # then set FOOTBALL_DATA_API_KEY

# 3. Bootstrap the database
python -m scripts.etl.init_db

# 4. Run tests
pytest -q
```

See [docs/runbooks.md](docs/runbooks.md) for the full pipeline (ETL → features →
train → predict → evaluate → dashboard).

## Architecture (one line)
Source adapters → staging ETL (SQLite) → leak-safe features → Elo + Poisson/Dixon-Coles
goal model → calibration → evaluation vs market/Opta/naive baselines → Streamlit view.
Scraping, features, modeling, and UI live in separate modules by design.
