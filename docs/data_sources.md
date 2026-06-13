# Data Sources

Status: **plan** (no ingestion code yet). This document specifies what we pull, how,
and the provenance/ToS rules every source must obey. It is the contract the
`src/clients/` and `src/scraping/` adapters will implement.

## Design rule: everything sits behind an adapter

Downstream code (ETL, features, models) must never import a source client directly.
Each source implements a thin adapter exposing a stable, typed interface so a fragile
scraper can be swapped for another source without touching feature or model code.

```
FixtureSource.get_fixtures(competition, season) -> list[FixtureRecord]
ResultSource.get_results(competition, season)   -> list[ResultRecord]
OddsSource.get_odds(match_id)                    -> list[OddsSnapshot]
XGSource.get_match_xg(match_id)                  -> MatchXG | None      # optional enrichment
BenchmarkSource.get_probs(match_id)              -> BenchmarkProb | None
```

Every record carries provenance: `source`, `source_url`, `ingested_at` (UTC),
`run_id`. Raw payloads are written immutably to `data/raw/<source>/<run_id>/`
before any parsing, so we can replay and debug schema drift.

## Primary sources

### 1. football-data.org (v4) — fixtures, results, odds  *(primary, API)*
- **Use for:** World Cup fixtures, final scores, and betting odds where exposed.
- **Auth:** free tier, header `X-Auth-Token`. Key lives in `.env` as
  `FOOTBALL_DATA_API_KEY` (never committed). The user supplies this key.
- **Endpoints of interest:** competition matches, single match.
- **Rate limits:** free tier is throttled (low requests/min). Client must pace
  requests, honor `429`/`Retry-After`, and cache responses for reproducibility.
- **Notes:** odds coverage on the free tier is partial; treat odds as
  best-effort and never overwrite a prior `odds_snapshots` row.

### 2. Historical international results — model bootstrap  *(primary, static)*
- **Why this exists:** the World Cup is ~64 matches at neutral venues between teams
  that rarely meet. Team strength **cannot** be fit from tournament data alone.
  We anchor Elo and Poisson attack/defense priors on decades of international
  results (e.g. the well-known "International football results 1872–present"
  dataset: date, home, away, score, tournament, venue, neutral flag).
- **Loader:** lands as immutable CSV/parquet under `data/raw/intl_results/`, then
  staged into `matches` with `competition='international'`.
- **Leakage rule:** only matches with kickoff strictly before a given as-of date
  may inform that date's ratings/features.

### 3. soccerdata → Sofascore / Understat — xG signals  *(secondary, scraping, optional)*
- **Use for:** xG for/against and shot-quality proxies as model covariates.
- **Fragility:** HTML/endpoint drift is expected. Gated behind `XGSource` so the
  pipeline runs fully without it; xG is enrichment, not a hard dependency.
- **Discipline:** keep raw snapshots, parse in one place, add source-specific
  parser tests, polite pacing + clear user-agent, respect ToS. Fail loudly on
  schema drift — never silently swallow parse errors.

### 4. Opta Analyst public prediction pages — benchmark  *(secondary, reference)*
- **Use for:** external benchmark probabilities only, stored in
  `benchmark_predictions` with `source='opta_public'`.
- **Hard rule:** public outputs only. Do **not** imply access to a licensed Opta
  event feed. If a page is unavailable, the benchmark is simply absent for that match.

## Environment & toolchain (decided)

- **Python:** system 3.9.6 is too old for the modern `soccerdata`/`polars` stack.
  We create a dedicated **virtualenv on Python 3.11/3.12**. (`uv` not installed;
  install it or fall back to `venv`+`pip` — see [runbooks.md](runbooks.md).)
- **Secrets:** all keys in `.env` (gitignored); `.env.example` documents the names.
  - `FOOTBALL_DATA_API_KEY` — user-supplied, free tier.
- **Caching:** HTTP responses cached on disk where legal/useful for reproducibility.

## Provenance columns (every ingested row)

| column        | meaning                                            |
|---------------|----------------------------------------------------|
| `source`      | logical source name (e.g. `football_data`)         |
| `source_url`  | exact URL/endpoint the row came from               |
| `ingested_at` | UTC timestamp of ingestion                         |
| `run_id`      | ETL run identifier (groups one pipeline execution) |
