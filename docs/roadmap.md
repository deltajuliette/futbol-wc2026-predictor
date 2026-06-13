# Roadmap & Task Checklist

Status: **M0–M7 built and tested on synthetic data; live football-data ingestion
pending an API key.** Phased, simplest-first, each phase shippable and tested before
the next. Maps to the "default first tasks" in `CLAUDE.md`.

Verified end-to-end (clean-room runbook reproduction): init_db → make_sample_data →
load → build_features → train → predict → backtest, 30/30 tests passing, Streamlit
dashboard boots. Backtest: calibrated Dixon-Coles beats Elo and uniform on log
loss / Brier / RPS.

## M0 — Foundation
- [ ] Create venv on Python 3.11/3.12; `pyproject.toml` with pandas/polars/sqlalchemy/
      pydantic/scikit-learn/soccerdata/streamlit.
- [ ] Repo skeleton per CLAUDE.md tree (`src/`, `scripts/`, `app/`, `data/`, `db/`,
      `models/`, `tests/`).
- [ ] `.env.example` (`FOOTBALL_DATA_API_KEY`), structured config via pydantic, logging.
- [ ] `git init`, `.gitignore` (`.env`, `db/*.sqlite`, `data/raw/*`, `.venv`).

## M1 — Database
- [ ] Bootstrap DDL for all 10 tables (see [database_schema.md](database_schema.md)).
- [ ] `init_db` entrypoint (idempotent). Schema + dedup tests.

## M2 — Source clients (behind adapters)
- [ ] `FixtureSource`/`ResultSource`/`OddsSource` for football-data.org v4 (paced, cached,
      429-aware).
- [ ] Historical international results loader.
- [ ] `XGSource` (soccerdata) + `BenchmarkSource` (Opta public) — optional, gated.
- [ ] Adapter interface tests + parser tests on captured raw snapshots.

## M3 — Staging ETL
- [ ] Land fixtures/results/odds idempotently with provenance cols; immutable raw.
- [ ] Dedup rules enforced (matches upsert; odds append-only).
- [ ] ETL runs on a small sample without errors (acceptance check).

## M4 — Features
- [ ] `team_match_features` builder: Elo pre, diffs, rest days, form, optional xG.
- [ ] Anti-leakage guards + fixed-as-of-date reproducibility test.

## M5 — Models
- [ ] Elo baseline (neutral-venue + MOV aware) → 1X2 mapping.
- [ ] Poisson + Dixon-Coles goal matrix → 1X2, xG, scorelines, BTTS, O/U.
- [ ] Knockout progression model (ET + penalties).
- [ ] Calibration layer (raw + calibrated persisted).
- [ ] Each fit writes `model_runs`; predictions versioned. Sum-to-1 test.

## M6 — Evaluation
- [ ] Log loss, Brier, RPS, reliability table, sharpness.
- [ ] Market de-vig (proportional + Shin) + naive baselines + Opta benchmark.
- [ ] Rolling-origin backtest; acceptance gates vs calibrated baseline.

## M7 — Dashboard (Streamlit)
- [ ] Upcoming matches, match detail (xG/scorelines/edge), model-vs-market-vs-Opta,
      calibration/performance panels.
- [ ] Reads only from stored tables; every chart stamped with run + timestamp.

## Cross-cutting (every phase)
- [ ] Tests alongside code (transformations, parsers, scoring, reproducibility).
- [ ] No scraping/feature/model/UI logic mixed in one file.
- [ ] Fail loudly on schema drift; never train on future info.

## Open inputs needed from user
- football-data.org free API key (for M2/M3 live pulls).
- Confirm which historical international results dataset to load (M2).
