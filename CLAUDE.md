# CLAUDE.md

## Project purpose
Build a football match prediction system for the ongoing FIFA World Cup that produces calibrated pre-match probabilities, compares them against market-implied odds and Opta-style benchmarks, and publishes the results in a clear dashboard.

The system should optimize for probability quality, calibration, reproducibility, and transparency rather than headline accuracy on single matches.

## What success looks like
- Reliable ETL for fixtures, results, odds, and event-level signals.
- A reproducible feature store in SQLite first, with an easy upgrade path to Postgres.
- Match-level probabilities for home win / draw / away win, plus likely scorelines.
- Confidence intervals and uncertainty bands, not just point estimates.
- Benchmark views against market prices and Opta Analyst style probabilities.
- A dashboard that explains where the model agrees or disagrees with market and Opta.

## Core principles
- Treat this as a probabilistic forecasting project, not a deterministic pick engine.
- Prefer simple, well-calibrated models before adding complexity.
- Keep the full pipeline reproducible from raw pull to dashboard output.
- Separate raw data, cleaned tables, features, model artifacts, and presentation outputs.
- Every number shown in the dashboard must be traceable to a stored table or model artifact.
- Use market odds as a benchmark, not as ground truth.
- Respect website terms of use and rate limits for all scraped or unofficial data sources.

## Data sources
Primary sources to support:
- FIFA schedule/results for tournament fixtures and final scores.
- football-data.org for fixtures and betting odds where available.
- WhoScored and Sofascore for match events and xG-related signals when legally and technically feasible.
- Opta Analyst public prediction pages for benchmark comparison when available.

Practical guidance:
- Use official APIs first.
- If event/xG collection from WhoScored or Sofascore is fragile, capture the dependency behind adapters so the pipeline can swap to another source without changing downstream code.
- If Opta event data is not licensed, do not imply access to proprietary feeds; use public Opta Analyst outputs only as benchmark inputs.
- Preserve source provenance on every ingested row.

## Recommended repository shape
```text
project/
  app/
    dashboard/                 # Dash or Streamlit app
  data/
    raw/
    staging/
    curated/
  db/
    worldcup.sqlite
  models/
    artifacts/
    reports/
  notebooks/
  scripts/
    etl/
    features/
    modeling/
    evaluation/
  src/
    config/
    clients/
    scraping/
    pipelines/
    features/
    models/
    evaluation/
    dashboards/
    utils/
  tests/
  .env.example
  pyproject.toml
  README.md
```

## Working approach
When asked to implement anything, follow this order:
1. Understand the exact prediction target.
2. Check existing schema, scripts, and artifacts before adding new files.
3. Prefer extending current adapters and pipelines over creating parallel ones.
4. Write or update tests for transformations and model logic.
5. Run the smallest meaningful validation step before moving on.
6. Summarize assumptions, caveats, and next actions clearly.

## Prediction targets
Default target hierarchy:
1. Pre-match 1X2 probabilities.
2. Expected goals for each team.
3. Scoreline distribution from Poisson or related goal model.
4. Secondary derived outputs: both teams to score, over/under lines, upset flags, edge vs market.

Unless explicitly requested, focus on pre-match forecasts, not in-play forecasting.

## Data engineering standards
- Build ETL as modular Python scripts or pipeline entrypoints, not notebook-only logic.
- Store timestamps in UTC and keep source-local kickoff time fields where relevant.
- Add `source`, `source_url`, `ingested_at`, and `run_id` fields where possible.
- Keep raw payloads immutable.
- Cleaned tables should have explicit primary keys and deduplication rules.
- Make scrapers idempotent where possible.
- Cache fetched responses when legal and useful for reproducibility.
- Prefer SQLite during prototyping; keep SQL portable enough to move to Postgres.

## Database guidance
Minimum tables to maintain:
- `matches`
- `teams`
- `odds_snapshots`
- `match_events`
- `team_match_features`
- `model_runs`
- `predictions`
- `benchmark_predictions`
- `evaluation_metrics`

Database rules:
- Use normalized keys for team names and tournament stages.
- Never overwrite historical odds snapshots.
- Version model outputs by `model_run_id`.
- Persist both raw probabilities and calibrated probabilities.

## Feature engineering guidance
Prioritize interpretable, tournament-relevant features:
- Elo or power rating differential.
- Recent form, but avoid overweighting tiny sample windows.
- Goal production and concession rates.
- xG for and xG against, plus shot quality proxies when available.
- Rest days and travel proxies if available.
- Team strength by confederation or historical baseline only if justified.
- Market-implied probability inputs only for benchmarking or residual analysis unless explicitly asked to blend them.

Feature rules:
- Prevent leakage from post-match data.
- Use only information available before kickoff.
- Keep a documented feature generation timestamp.
- Prefer a small, robust feature set before high-dimensional experiments.

## Modeling guidance
Start with simple baselines and only add complexity if evaluation justifies it.

Preferred modeling order:
1. Elo baseline for team strength.
2. Poisson goals model using attack/defense strength and home/neutral adjustments.
3. Add covariates such as rest, recent xG form, injuries if available, and market residual features.
4. Add calibration layer on top of raw probabilities.

Model rules:
- For World Cup matches, account for neutral venues when appropriate.
- Handle knockout matches separately if extra time or penalties matter to the target.
- Keep training and scoring code separate.
- Save parameters, feature definitions, training window, and evaluation output for every run.
- Avoid black-box complexity unless it clearly beats calibrated baselines.

## Benchmarking and calibration
Benchmark against:
- Closing or latest available market-implied probabilities.
- Public Opta Analyst match prediction outputs when available.
- Simple naive baselines such as Elo-only and rank-only models.

Evaluation priorities:
- Log loss.
- Brier score.
- Calibration curves / reliability tables.
- Ranked probability score if implemented.
- Sharpness vs calibration trade-off.

Important rule:
Do not describe a model as better because it picked more winners over a short run. Prefer proper scoring rules and calibration diagnostics.

## Dashboard expectations
Default dashboard should include:
- Upcoming matches list.
- Model 1X2 probabilities.
- Implied market probabilities after overround adjustment.
- Opta benchmark probabilities if available.
- Confidence intervals or uncertainty bands.
- Expected goals and most likely scorelines.
- Model edge vs market.
- Historical calibration and performance panels.

Presentation rules:
- Show both percentages and simple visual bars.
- Make uncertainty obvious.
- Highlight differences between model, market, and Opta without sensational language.
- Every chart must state timestamp and model run used.

## Implementation preferences
- Use Python throughout the pipeline.
- Prefer `pandas`, `polars`, `sqlalchemy`, `pydantic`, and `scikit-learn` where useful.
- Use `soccerdata` where it reduces scraping fragility and supports the needed source cleanly.
- Prefer typed functions and small modules.
- Keep configuration in `.env` and structured config files, not hardcoded constants.
- Use logging, not scattered print statements.

## Code quality rules
- Write clear, boring, maintainable code.
- Avoid premature abstraction.
- Keep functions focused and testable.
- Add docstrings for non-obvious logic.
- Do not silently swallow scraping or parsing errors.
- Fail loudly on schema drift.
- For any new script, include an example CLI invocation in the module docstring or README.

## Testing and validation
Minimum checks before calling work complete:
- ETL runs on a small sample without errors.
- Deduplication and schema tests pass.
- Feature generation is reproducible for a fixed as-of date.
- Model training runs end-to-end on a controlled dataset.
- Probability outputs sum to 1 for 1X2.
- Calibration and scoring metrics are updated.
- Dashboard loads from stored outputs, not ad hoc notebook state.

## When working with scraped sources
- Expect HTML structure drift.
- Put selectors and parsing logic in one place.
- Keep raw snapshots for debugging.
- Add source-specific tests for parsers.
- Use polite request pacing and clear user-agent settings where allowed.

## Communication style for this repo
When explaining work:
- Be concise and structured.
- State assumptions explicitly.
- Separate facts from estimates.
- Flag data quality issues early.
- For modeling changes, explain impact in plain English first, then technical detail.

## Default first tasks for a fresh repo
If the repository is empty, start by creating:
1. Project skeleton and config.
2. SQLite schema and migrations/bootstrap SQL.
3. Source clients for FIFA schedule/results and football-data.org.
4. A staging ETL that lands fixtures, results, and odds.
5. Baseline Elo + Poisson pipeline.
6. Evaluation script with log loss, Brier score, and calibration table.
7. Basic Streamlit dashboard for match-level probabilities and benchmark comparison.

## Anti-patterns
- Do not mix scraping, feature engineering, model fitting, and UI logic in one file.
- Do not train on rows that include future information.
- Do not overwrite prior model outputs.
- Do not claim Opta data access unless the repository truly has licensed access.
- Do not optimize to anecdotal match narratives.
- Do not ship dashboard metrics without definitions.

## Useful external references
- Claude Code reads `CLAUDE.md` as project guidance, and its docs recommend starting with this file for project conventions.
- Keep this file short enough to stay useful; move task-specific or source-specific detail into focused docs under `docs/` when the project grows.
- football-data.org v4 exposes competition and match endpoints suitable for fixture/result ingestion.
- The `soccerdata` Python package supports sources including Sofascore, Understat, and WhoScored, which can reduce custom scraping burden.
- Opta Analyst public prediction pages can serve as benchmark comparison inputs, but they are not a substitute for licensed Opta event feeds.

## Supporting docs
Planning docs live under `docs/` (created; implementation pending):
- [`docs/data_sources.md`](docs/data_sources.md) — sources, adapters, provenance, env/keys.
- [`docs/database_schema.md`](docs/database_schema.md) — the 10 tables, keys, dedup rules.
- [`docs/modeling.md`](docs/modeling.md) — Elo → Poisson/Dixon-Coles → knockout → calibration.
- [`docs/evaluation.md`](docs/evaluation.md) — log loss/Brier/RPS, calibration, de-vig, backtests.
- [`docs/dashboard.md`](docs/dashboard.md) — Streamlit read-only view, panels, data contract.
- [`docs/runbooks.md`](docs/runbooks.md) — env setup and CLI entrypoints per pipeline stage.
- [`docs/roadmap.md`](docs/roadmap.md) — phased task checklist (M0–M7).

## If unsure
Prefer the simplest implementation that improves reproducibility, calibration, and traceability.
