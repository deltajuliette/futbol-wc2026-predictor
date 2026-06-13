"""Read-only queries backing the dashboard.

Kept separate from the UI so they are unit-testable and so the dashboard can only
ever read from stored tables (never recompute predictions). Every frame carries the
model run / timestamps needed to stamp charts.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


def latest_model_run(engine: Engine, model_name: str = "dixon_coles") -> dict | None:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT model_run_id, model_name, created_at_utc, training_window, "
                "params_json, artifact_path FROM model_runs WHERE model_name=:n "
                "ORDER BY model_run_id DESC LIMIT 1"
            ),
            {"n": model_name},
        ).fetchone()
    if not row:
        return None
    keys = ["model_run_id", "model_name", "created_at_utc", "training_window",
            "params_json", "artifact_path"]
    return dict(zip(keys, row))


def upcoming_predictions(engine: Engine, competition: str = "world_cup_2026") -> pd.DataFrame:
    """Scheduled fixtures with model (calibrated + raw) and benchmark probabilities.

    Uses only the latest model run's predictions and the most recent Elo benchmark
    snapshot per match, so re-running the pipeline never multiplies rows.
    """
    q = """
        SELECT m.match_id, m.kickoff_utc, m.stage, m.neutral,
               ht.display_name AS home, at.display_name AS away,
               p.model_run_id,
               p.p_home_cal, p.p_draw_cal, p.p_away_cal,
               p.p_home_raw, p.p_draw_raw, p.p_away_raw,
               p.exp_goals_home, p.exp_goals_away, p.scoreline_json,
               p.p_btts, p.p_over25, p.predicted_at_utc,
               b.p_home AS elo_home, b.p_draw AS elo_draw, b.p_away AS elo_away
        FROM matches m
        JOIN predictions p
               ON p.match_id = m.match_id
              AND p.model_run_id = (SELECT MAX(model_run_id) FROM predictions)
        JOIN teams ht ON ht.team_id = m.home_team_id
        JOIN teams at ON at.team_id = m.away_team_id
        LEFT JOIN (
            SELECT b.match_id, b.p_home, b.p_draw, b.p_away
            FROM benchmark_predictions b
            JOIN (SELECT match_id, MAX(benchmark_id) AS mx
                  FROM benchmark_predictions WHERE source = 'elo_only'
                  GROUP BY match_id) t
              ON t.match_id = b.match_id AND t.mx = b.benchmark_id
        ) b ON b.match_id = m.match_id
        WHERE m.competition = :c AND m.status = 'scheduled'
        ORDER BY m.kickoff_utc
    """
    with engine.connect() as conn:
        df = pd.read_sql_query(text(q), conn, params={"c": competition})
    if not df.empty:
        df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
        # Edge vs the Elo benchmark on the home line (market edge when odds exist).
        df["edge_home_vs_elo"] = df["p_home_cal"] - df["elo_home"]
    return df


def evaluation_summary(engine: Engine) -> pd.DataFrame:
    """Latest evaluation_metrics row per model label."""
    q = """
        SELECT e.label, e.as_of_utc, e.n_matches, e.log_loss, e.brier, e.rps,
               e.sharpness, e.notes
        FROM evaluation_metrics e
        JOIN (SELECT label, MAX(eval_id) AS mx FROM evaluation_metrics GROUP BY label) t
          ON t.label = e.label AND t.mx = e.eval_id
        ORDER BY e.log_loss
    """
    with engine.connect() as conn:
        return pd.read_sql_query(text(q), conn)


def reliability_bins(engine: Engine, label: str = "dc_cal") -> pd.DataFrame:
    """Reliability-table bins (mean predicted vs observed) for one model label."""
    import json
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT calibration_json FROM evaluation_metrics WHERE label=:l "
                "ORDER BY eval_id DESC LIMIT 1"
            ),
            {"l": label},
        ).fetchone()
    if not row or not row[0]:
        return pd.DataFrame(columns=["mean_pred", "frac_obs", "count"])
    return pd.DataFrame(json.loads(row[0]))
