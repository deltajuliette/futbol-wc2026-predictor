# Database Schema

Status: **plan**. SQLite first (`db/worldcup.sqlite`), SQL kept portable for a
later Postgres move. Bootstrap via idempotent `CREATE TABLE IF NOT EXISTS` DDL in
`scripts/etl/` (or Alembic if we adopt SQLAlchemy migrations).

## Conventions
- **Time:** all timestamps UTC (`*_utc`). Keep source-local kickoff separately
  (`kickoff_local`, `kickoff_tz`).
- **Provenance:** `source`, `source_url`, `ingested_at`, `run_id` on ingested tables.
- **Keys:** normalized team and stage keys (see `teams`, stage enum below).
- **Immutability:** never overwrite historical odds; version model outputs by
  `model_run_id`; raw payloads live on disk, not mutated in place.
- **Probabilities:** persist BOTH raw and calibrated where a model emits them.

## Core tables

### `teams`
Canonical team registry; the join target for all name normalization.
| column          | type | notes                                  |
|-----------------|------|----------------------------------------|
| `team_id` (PK)  | int  | surrogate key                          |
| `team_key`      | text | normalized slug, unique                |
| `display_name`  | text | human label                            |
| `confederation` | text | UEFA/CONMEBOL/… (nullable)             |
| `fifa_code`     | text | 3-letter code (nullable)               |
| aliases handled via a `team_aliases(alias, team_id)` side table |

### `matches`
One row per fixture/result (historical + tournament).
| column            | type | notes                                         |
|-------------------|------|-----------------------------------------------|
| `match_id` (PK)   | int  | surrogate                                     |
| `competition`     | text | `world_cup_2026`, `international`, …           |
| `season`          | text |                                               |
| `stage`           | text | enum: group / r32 / r16 / qf / sf / final …   |
| `kickoff_utc`     | ts   |                                               |
| `kickoff_local`   | ts   | source-local                                  |
| `kickoff_tz`      | text |                                               |
| `home_team_id`    | int  | FK → teams                                    |
| `away_team_id`    | int  | FK → teams                                    |
| `neutral`         | bool | true for most WC matches                      |
| `status`          | text | scheduled / live / finished                   |
| `home_goals`      | int  | nullable until finished                       |
| `away_goals`      | int  | nullable                                      |
| `home_goals_et`   | int  | extra time (knockouts), nullable              |
| `away_goals_et`   | int  | nullable                                      |
| `pens_home`       | int  | nullable                                      |
| `pens_away`       | int  | nullable                                      |
| provenance cols   |      |                                               |
- **Natural/dedup key:** (`competition`, `kickoff_utc`, `home_team_id`, `away_team_id`).
  Upserts match on this; surrogate `match_id` is stable.

### `odds_snapshots`  *(append-only)*
| column               | type | notes                               |
|----------------------|------|-------------------------------------|
| `snapshot_id` (PK)   | int  |                                     |
| `match_id`           | int  | FK                                  |
| `captured_at_utc`    | ts   | when the price was observed         |
| `bookmaker`          | text |                                     |
| `market`             | text | `1x2`, `ou_2_5`, `btts`, …          |
| `home_odds`          | real | decimal                             |
| `draw_odds`          | real |                                     |
| `away_odds`          | real |                                     |
| `overround`          | real | computed                            |
| provenance cols      |      |                                     |
- Never updated, only inserted. Latest/closing price is a query, not an overwrite.

### `match_events`
Event/xG-level signals (optional, source-gated).
| column            | type | notes                                  |
|-------------------|------|----------------------------------------|
| `event_id` (PK)   | int  |                                        |
| `match_id`        | int  | FK                                     |
| `team_id`         | int  | FK                                     |
| `minute`          | int  |                                        |
| `event_type`      | text | shot/goal/…                            |
| `xg`              | real | nullable                               |
| `payload_json`    | text | raw parsed blob                        |
| provenance cols   |      |                                        |

## Feature & modeling tables

### `team_match_features`
One row per (match, team) with pre-kickoff features only.
| column                  | type | notes                                  |
|-------------------------|------|----------------------------------------|
| `match_id` + `team_id`  | PK   | composite                              |
| `as_of_utc`             | ts   | feature generation cutoff (anti-leak)  |
| `elo_pre`               | real | rating before this match               |
| `elo_diff`              | real | vs opponent                            |
| `rest_days`             | int  |                                        |
| `xg_for_form`           | real | rolling, nullable                      |
| `xg_against_form`       | real | rolling, nullable                      |
| `gf_rate` / `ga_rate`   | real | goal production/concession             |
| `is_home` / `neutral`   | bool |                                        |
| `feature_set_version`   | text |                                        |

### `model_runs`
| column               | type | notes                                       |
|----------------------|------|---------------------------------------------|
| `model_run_id` (PK)  | int  |                                             |
| `model_name`         | text | `elo`, `dixon_coles`, …                      |
| `created_at_utc`     | ts   |                                             |
| `training_window`    | text | start/end of training data                  |
| `params_json`        | text | hyperparameters / fitted params pointer     |
| `feature_set_version`| text |                                             |
| `code_git_sha`       | text | reproducibility                             |
| `artifact_path`      | text | `models/artifacts/...`                       |

### `predictions`
| column               | type | notes                                       |
|----------------------|------|---------------------------------------------|
| `prediction_id` (PK) | int  |                                             |
| `model_run_id`       | int  | FK → model_runs                             |
| `match_id`           | int  | FK                                          |
| `p_home_raw`         | real | sums to 1 with draw/away                     |
| `p_draw_raw`         | real |                                             |
| `p_away_raw`         | real |                                             |
| `p_home_cal`         | real | calibrated (nullable until calib fit)       |
| `p_draw_cal`         | real |                                             |
| `p_away_cal`         | real |                                             |
| `exp_goals_home`     | real |                                             |
| `exp_goals_away`     | real |                                             |
| `scoreline_json`     | text | top-N scorelines + probs                    |
| `p_btts` / `p_over25`| real | derived from scoreline matrix               |
| `ci_json`            | text | uncertainty bands                           |
| `predicted_at_utc`   | ts   |                                             |

### `benchmark_predictions`
| column            | type | notes                                          |
|-------------------|------|------------------------------------------------|
| `benchmark_id`(PK)| int  |                                                |
| `match_id`        | int  | FK                                             |
| `source`          | text | `market_devig`, `opta_public`, `elo_only`, …   |
| `method`          | text | de-vig method for market (`proportional`/`shin`)|
| `p_home`/`p_draw`/`p_away` | real |                                       |
| `captured_at_utc` | ts   |                                                |
| provenance cols   |      |                                                |

### `evaluation_metrics`
| column            | type | notes                                          |
|-------------------|------|------------------------------------------------|
| `eval_id` (PK)    | int  |                                                |
| `model_run_id`    | int  | FK (or benchmark id)                            |
| `as_of_utc`       | ts   | evaluation cutoff                              |
| `n_matches`       | int  |                                                |
| `log_loss`        | real |                                                |
| `brier`           | real | multiclass                                     |
| `rps`             | real | ranked probability score                       |
| `calibration_json`| text | reliability-table bins                          |
| `sharpness`       | real |                                                |
| `notes`           | text |                                                |

## Dedup & integrity rules (summary)
- `matches`: upsert on natural key; never duplicate a fixture.
- `odds_snapshots`: insert-only.
- `predictions`/`evaluation_metrics`: versioned by `model_run_id`, never overwritten.
- All probability triplets validated to sum to 1 (±tol) before insert.
