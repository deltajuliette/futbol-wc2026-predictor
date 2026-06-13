"""Data-access helpers: idempotent upserts + typed reads.

Centralizes SQL so ETL/feature/model code never writes raw INSERTs. Team and match
writes are idempotent on their natural keys (see docs/database_schema.md).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from clients.types import FixtureRecord
from utils.naming import team_key


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def upsert_team(engine: Engine, display_name: str, confederation: str | None = None,
                fifa_code: str | None = None) -> int:
    """Insert a team if absent (keyed by normalized slug); return its team_id.

    Resolution order: exact ``team_key`` match, then ``team_aliases`` (so known name
    variants like "Czechia"/"Czech Republic" collapse onto one canonical team), then
    create. Aliases are stored as slugs (see :mod:`utils.naming`) for robust matching.
    """
    key = team_key(display_name)
    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT team_id FROM teams WHERE team_key = :k"), {"k": key}
        ).fetchone()
        if row:
            return int(row[0])
        alias = conn.execute(
            text("SELECT team_id FROM team_aliases WHERE alias = :k"), {"k": key}
        ).fetchone()
        if alias:
            return int(alias[0])
        res = conn.execute(
            text(
                "INSERT INTO teams (team_key, display_name, confederation, fifa_code) "
                "VALUES (:k, :n, :c, :f)"
            ),
            {"k": key, "n": display_name, "c": confederation, "f": fifa_code},
        )
        return int(res.lastrowid)


def upsert_match(engine: Engine, rec: FixtureRecord, run_id: str | None = None) -> int:
    """Idempotently upsert a fixture/result on its natural key; return match_id."""
    home_id = upsert_team(engine, rec.home_team)
    away_id = upsert_team(engine, rec.away_team)
    prov = rec.provenance
    params = {
        "competition": rec.competition,
        "season": rec.season,
        "stage": rec.stage,
        "kickoff_utc": rec.kickoff_utc.isoformat(),
        "kickoff_local": rec.kickoff_local.isoformat() if rec.kickoff_local else None,
        "kickoff_tz": rec.kickoff_tz,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "neutral": int(rec.neutral),
        "status": rec.status,
        "home_goals": rec.home_goals,
        "away_goals": rec.away_goals,
        "home_goals_et": rec.home_goals_et,
        "away_goals_et": rec.away_goals_et,
        "pens_home": rec.pens_home,
        "pens_away": rec.pens_away,
        "source": prov.source if prov else None,
        "source_url": prov.source_url if prov else None,
        "ingested_at": _now_iso(),
        "run_id": run_id or (prov.run_id if prov else None),
    }
    cols = ", ".join(params.keys())
    placeholders = ", ".join(f":{k}" for k in params)
    # Upsert on the natural key; refresh result/status fields on conflict.
    upsert = text(
        f"INSERT INTO matches ({cols}) VALUES ({placeholders}) "
        "ON CONFLICT (competition, kickoff_utc, home_team_id, away_team_id) DO UPDATE SET "
        "status=excluded.status, home_goals=excluded.home_goals, away_goals=excluded.away_goals, "
        "home_goals_et=excluded.home_goals_et, away_goals_et=excluded.away_goals_et, "
        "pens_home=excluded.pens_home, pens_away=excluded.pens_away, stage=excluded.stage, "
        "ingested_at=excluded.ingested_at, run_id=excluded.run_id"
    )
    with engine.begin() as conn:
        conn.execute(upsert, params)
        row = conn.execute(
            text(
                "SELECT match_id FROM matches WHERE competition=:competition "
                "AND kickoff_utc=:kickoff_utc AND home_team_id=:home_team_id "
                "AND away_team_id=:away_team_id"
            ),
            params,
        ).fetchone()
    return int(row[0])


def bulk_upsert_matches(engine: Engine, recs: list[FixtureRecord],
                        run_id: str | None = None) -> int:
    """Fast path for large loads: resolve teams once, upsert matches in one transaction.

    Equivalent to calling :func:`upsert_match` per record but with two round-trips
    instead of several per row. Returns the number of records processed.
    """
    if not recs:
        return 0
    # 1) Resolve every team in a single pass (team_key, then alias, then create).
    names = {n for r in recs for n in (r.home_team, r.away_team)}
    with engine.begin() as conn:
        existing = {row[0]: row[1] for row in
                    conn.execute(text("SELECT team_key, team_id FROM teams")).fetchall()}
        aliases = {row[0]: row[1] for row in
                   conn.execute(text("SELECT alias, team_id FROM team_aliases")).fetchall()}
        to_create = []
        seen = set()
        for name in names:
            key = team_key(name)
            if key not in existing and key not in aliases and key not in seen:
                seen.add(key)
                to_create.append({"k": key, "n": name})
        if to_create:
            conn.execute(
                text("INSERT OR IGNORE INTO teams (team_key, display_name) VALUES (:k, :n)"),
                to_create,
            )
        # Rebuild cache including the new rows (executemany gives no per-row ids).
        cache = {row[0]: row[1] for row in
                 conn.execute(text("SELECT team_key, team_id FROM teams")).fetchall()}
        # Aliases resolve to their canonical team_id (slugs absent from teams).
        for alias_key, tid in aliases.items():
            cache.setdefault(alias_key, tid)

    # 2) Upsert all matches in one transaction.
    now = _now_iso()
    params = []
    for r in recs:
        prov = r.provenance
        params.append({
            "competition": r.competition, "season": r.season, "stage": r.stage,
            "kickoff_utc": r.kickoff_utc.isoformat(),
            "kickoff_local": r.kickoff_local.isoformat() if r.kickoff_local else None,
            "kickoff_tz": r.kickoff_tz,
            "home_team_id": cache[team_key(r.home_team)],
            "away_team_id": cache[team_key(r.away_team)],
            "neutral": int(r.neutral), "status": r.status,
            "home_goals": r.home_goals, "away_goals": r.away_goals,
            "home_goals_et": r.home_goals_et, "away_goals_et": r.away_goals_et,
            "pens_home": r.pens_home, "pens_away": r.pens_away,
            "source": prov.source if prov else None,
            "source_url": prov.source_url if prov else None,
            "ingested_at": now, "run_id": run_id or (prov.run_id if prov else None),
        })
    cols = list(params[0].keys())
    placeholders = ", ".join(f":{k}" for k in cols)
    stmt = text(
        f"INSERT INTO matches ({', '.join(cols)}) VALUES ({placeholders}) "
        "ON CONFLICT (competition, kickoff_utc, home_team_id, away_team_id) DO UPDATE SET "
        "status=excluded.status, home_goals=excluded.home_goals, away_goals=excluded.away_goals, "
        "home_goals_et=excluded.home_goals_et, away_goals_et=excluded.away_goals_et, "
        "pens_home=excluded.pens_home, pens_away=excluded.pens_away, stage=excluded.stage, "
        "ingested_at=excluded.ingested_at, run_id=excluded.run_id"
    )
    with engine.begin() as conn:
        conn.execute(stmt, params)
    return len(params)


def load_matches_df(engine: Engine, competition: str | None = None,
                    finished_only: bool = False) -> pd.DataFrame:
    """Return matches joined to team display names as a DataFrame."""
    q = (
        "SELECT m.*, ht.display_name AS home_name, at.display_name AS away_name, "
        "ht.confederation AS home_conf, at.confederation AS away_conf "
        "FROM matches m "
        "JOIN teams ht ON ht.team_id = m.home_team_id "
        "JOIN teams at ON at.team_id = m.away_team_id"
    )
    clauses = []
    params: dict[str, object] = {}
    if competition:
        clauses.append("m.competition = :competition")
        params["competition"] = competition
    if finished_only:
        clauses.append("m.status = 'finished'")
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY m.kickoff_utc"
    with engine.connect() as conn:
        df = pd.read_sql_query(text(q), conn, params=params)
    if not df.empty:
        df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
    return df


def create_model_run(engine: Engine, model_name: str, training_window: str | None = None,
                     params_json: str | None = None, feature_set_version: str | None = None,
                     code_git_sha: str | None = None, artifact_path: str | None = None) -> int:
    """Insert a model_runs row and return its id."""
    with engine.begin() as conn:
        res = conn.execute(
            text(
                "INSERT INTO model_runs (model_name, created_at_utc, training_window, "
                "params_json, feature_set_version, code_git_sha, artifact_path) "
                "VALUES (:n, :t, :w, :p, :f, :g, :a)"
            ),
            {"n": model_name, "t": _now_iso(), "w": training_window, "p": params_json,
             "f": feature_set_version, "g": code_git_sha, "a": artifact_path},
        )
        return int(res.lastrowid)


def save_predictions(engine: Engine, model_run_id: int, rows: list[dict]) -> int:
    """Upsert predictions for a model run (unique on model_run_id, match_id)."""
    cols = [
        "match_id", "p_home_raw", "p_draw_raw", "p_away_raw",
        "p_home_cal", "p_draw_cal", "p_away_cal", "exp_goals_home", "exp_goals_away",
        "scoreline_json", "p_btts", "p_over25", "ci_json", "reasoning_json",
        "predicted_at_utc",
    ]
    placeholders = ", ".join(f":{c}" for c in cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c != "match_id")
    stmt = text(
        f"INSERT INTO predictions (model_run_id, {', '.join(cols)}) "
        f"VALUES (:model_run_id, {placeholders}) "
        f"ON CONFLICT (model_run_id, match_id) DO UPDATE SET {updates}"
    )
    payload = [{**{c: r.get(c) for c in cols}, "model_run_id": model_run_id} for r in rows]
    with engine.begin() as conn:
        conn.execute(stmt, payload)
    return len(payload)


def save_benchmarks(engine: Engine, rows: list[dict], run_id: str | None = None) -> int:
    """Insert benchmark_predictions rows (e.g. elo_only, market_devig, opta_public)."""
    cols = ["match_id", "source", "method", "p_home", "p_draw", "p_away",
            "captured_at_utc", "source_url"]
    placeholders = ", ".join(f":{c}" for c in cols)
    stmt = text(
        f"INSERT INTO benchmark_predictions ({', '.join(cols)}, ingested_at, run_id) "
        f"VALUES ({placeholders}, :ingested_at, :run_id)"
    )
    payload = [
        {**{c: r.get(c) for c in cols}, "ingested_at": _now_iso(), "run_id": run_id}
        for r in rows
    ]
    with engine.begin() as conn:
        conn.execute(stmt, payload)
    return len(payload)
