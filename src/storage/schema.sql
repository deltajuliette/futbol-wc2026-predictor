-- World Cup forecast schema (SQLite first; kept portable for Postgres).
-- Conventions: timestamps are TEXT ISO-8601 UTC (suffixed *_utc); booleans are
-- INTEGER 0/1; surrogate PKs are INTEGER. Provenance columns appear on ingested
-- tables. See docs/database_schema.md for the rationale.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- Reference: teams & aliases
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS teams (
    team_id       INTEGER PRIMARY KEY,
    team_key      TEXT NOT NULL UNIQUE,           -- normalized slug
    display_name  TEXT NOT NULL,
    confederation TEXT,                           -- UEFA / CONMEBOL / ...
    fifa_code     TEXT
);

CREATE TABLE IF NOT EXISTS team_aliases (
    alias    TEXT PRIMARY KEY,                    -- raw name seen in a source
    team_id  INTEGER NOT NULL REFERENCES teams(team_id)
);

-- ---------------------------------------------------------------------------
-- Matches (historical internationals + tournament fixtures/results)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS matches (
    match_id      INTEGER PRIMARY KEY,
    competition   TEXT NOT NULL,                  -- world_cup_2026 / international / ...
    season        TEXT,
    stage         TEXT,                           -- group/r32/r16/qf/sf/final/...
    kickoff_utc   TEXT NOT NULL,
    kickoff_local TEXT,
    kickoff_tz    TEXT,
    home_team_id  INTEGER NOT NULL REFERENCES teams(team_id),
    away_team_id  INTEGER NOT NULL REFERENCES teams(team_id),
    neutral       INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'scheduled', -- scheduled/live/finished
    home_goals    INTEGER,
    away_goals    INTEGER,
    home_goals_et INTEGER,
    away_goals_et INTEGER,
    pens_home     INTEGER,
    pens_away     INTEGER,
    -- provenance
    source        TEXT,
    source_url    TEXT,
    ingested_at   TEXT,
    run_id        TEXT,
    -- natural key for idempotent upserts
    UNIQUE (competition, kickoff_utc, home_team_id, away_team_id)
);
CREATE INDEX IF NOT EXISTS ix_matches_kickoff ON matches(kickoff_utc);
CREATE INDEX IF NOT EXISTS ix_matches_competition ON matches(competition);

-- ---------------------------------------------------------------------------
-- Odds snapshots (APPEND-ONLY; never overwrite historical prices)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS odds_snapshots (
    snapshot_id    INTEGER PRIMARY KEY,
    match_id       INTEGER NOT NULL REFERENCES matches(match_id),
    captured_at_utc TEXT NOT NULL,
    bookmaker      TEXT,
    market         TEXT NOT NULL DEFAULT '1x2',
    home_odds      REAL,
    draw_odds      REAL,
    away_odds      REAL,
    overround      REAL,
    source         TEXT,
    source_url     TEXT,
    ingested_at    TEXT,
    run_id         TEXT
);
CREATE INDEX IF NOT EXISTS ix_odds_match ON odds_snapshots(match_id, captured_at_utc);

-- ---------------------------------------------------------------------------
-- Match events / xG (optional, source-gated)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS match_events (
    event_id     INTEGER PRIMARY KEY,
    match_id     INTEGER NOT NULL REFERENCES matches(match_id),
    team_id      INTEGER REFERENCES teams(team_id),
    minute       INTEGER,
    event_type   TEXT,
    xg           REAL,
    payload_json TEXT,
    source       TEXT,
    source_url   TEXT,
    ingested_at  TEXT,
    run_id       TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_match ON match_events(match_id);

-- ---------------------------------------------------------------------------
-- Features: one row per (match, team), pre-kickoff only
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS team_match_features (
    match_id            INTEGER NOT NULL REFERENCES matches(match_id),
    team_id             INTEGER NOT NULL REFERENCES teams(team_id),
    as_of_utc           TEXT NOT NULL,            -- feature cutoff (anti-leak)
    elo_pre             REAL,
    elo_diff            REAL,
    rest_days           INTEGER,
    xg_for_form         REAL,
    xg_against_form     REAL,
    gf_rate             REAL,
    ga_rate             REAL,
    is_home             INTEGER,
    neutral             INTEGER,
    feature_set_version TEXT,
    PRIMARY KEY (match_id, team_id)
);

-- ---------------------------------------------------------------------------
-- Model runs & outputs (versioned; never overwritten)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_runs (
    model_run_id        INTEGER PRIMARY KEY,
    model_name          TEXT NOT NULL,
    created_at_utc      TEXT NOT NULL,
    training_window     TEXT,
    params_json         TEXT,
    feature_set_version TEXT,
    code_git_sha        TEXT,
    artifact_path       TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id   INTEGER PRIMARY KEY,
    model_run_id    INTEGER NOT NULL REFERENCES model_runs(model_run_id),
    match_id        INTEGER NOT NULL REFERENCES matches(match_id),
    p_home_raw      REAL,
    p_draw_raw      REAL,
    p_away_raw      REAL,
    p_home_cal      REAL,
    p_draw_cal      REAL,
    p_away_cal      REAL,
    exp_goals_home  REAL,
    exp_goals_away  REAL,
    scoreline_json  TEXT,
    p_btts          REAL,
    p_over25        REAL,
    ci_json         TEXT,
    reasoning_json  TEXT,                          -- deterministic per-prediction drivers
    predicted_at_utc TEXT,
    UNIQUE (model_run_id, match_id)
);
CREATE INDEX IF NOT EXISTS ix_pred_match ON predictions(match_id);

CREATE TABLE IF NOT EXISTS benchmark_predictions (
    benchmark_id    INTEGER PRIMARY KEY,
    match_id        INTEGER NOT NULL REFERENCES matches(match_id),
    source          TEXT NOT NULL,                -- market_devig / opta_public / elo_only / ...
    method          TEXT,                         -- proportional / shin / ...
    p_home          REAL,
    p_draw          REAL,
    p_away          REAL,
    captured_at_utc TEXT,
    source_url      TEXT,
    ingested_at     TEXT,
    run_id          TEXT
);
CREATE INDEX IF NOT EXISTS ix_bench_match ON benchmark_predictions(match_id, source);

CREATE TABLE IF NOT EXISTS evaluation_metrics (
    eval_id          INTEGER PRIMARY KEY,
    model_run_id     INTEGER REFERENCES model_runs(model_run_id),
    label            TEXT,                         -- model name or benchmark source
    as_of_utc        TEXT NOT NULL,
    n_matches        INTEGER,
    log_loss         REAL,
    brier            REAL,
    rps              REAL,
    calibration_json TEXT,
    sharpness        REAL,
    notes            TEXT
);
