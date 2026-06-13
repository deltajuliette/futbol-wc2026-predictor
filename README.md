# World Cup Forecast

Calibrated pre-match probability forecasts for the FIFA World Cup — home/draw/away,
expected goals, and scoreline distributions — benchmarked against market-implied odds
and public Opta Analyst probabilities, published in a Streamlit dashboard.

This is a **probabilistic forecasting** project: it optimizes for calibration and proper
scoring rules (log loss, Brier, RPS), reproducibility, and traceability — not headline
single-match accuracy. See [`CLAUDE.md`](CLAUDE.md) for principles and [`docs/`](docs/)
for the design.

## Status
**End-to-end pipeline working on synthetic data** (Elo + calibrated Dixon-Coles →
evaluation → Streamlit). Live football-data.org ingestion is implemented and tested
at the parser level, pending an API key. Docs:
- [Data sources](docs/data_sources.md) · [Schema](docs/database_schema.md) ·
  [Modeling](docs/modeling.md) · [Evaluation](docs/evaluation.md) ·
  [Dashboard](docs/dashboard.md) · [Runbooks](docs/runbooks.md) · [Roadmap](docs/roadmap.md)

In a backtest on the bundled synthetic data, calibrated Dixon-Coles beats the
Elo-only and uniform baselines on log loss, Brier, and RPS.

```bash
# After install (see Quickstart), reproduce the whole pipeline:
python -m scripts.etl.init_db
python -m scripts.etl.make_sample_data --seed 7
python -m scripts.etl.load_intl_results --path data/raw/intl_results/results.csv
python -m scripts.etl.load_intl_results --path data/raw/intl_results/upcoming_wc.csv
python -m scripts.features.build_features
python -m scripts.modeling.train_dixon_coles --half-life 540
python -m scripts.modeling.predict --competition world_cup_2026
python -m scripts.evaluation.backtest --folds 4
streamlit run app/dashboard/app.py
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
