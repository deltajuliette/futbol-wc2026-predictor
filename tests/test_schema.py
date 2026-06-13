"""Schema bootstrap + integrity tests against a throwaway SQLite file."""

from __future__ import annotations

import pytest
from sqlalchemy import text

from storage.database import get_engine, init_db, list_tables

EXPECTED_TABLES = {
    "teams",
    "team_aliases",
    "matches",
    "odds_snapshots",
    "match_events",
    "team_match_features",
    "model_runs",
    "predictions",
    "benchmark_predictions",
    "evaluation_metrics",
}


@pytest.fixture()
def engine(tmp_path):
    url = f"sqlite:///{tmp_path / 'test.sqlite'}"
    return init_db(get_engine(url))


def test_all_tables_created(engine):
    assert EXPECTED_TABLES.issubset(set(list_tables(engine)))


def test_init_db_is_idempotent(engine):
    # Re-applying the schema must not raise.
    before = set(list_tables(engine))
    init_db(engine)
    assert set(list_tables(engine)) == before


def test_matches_natural_key_is_unique(engine):
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO teams (team_id, team_key, display_name) VALUES (1,'bra','Brazil')"))
        conn.execute(text("INSERT INTO teams (team_id, team_key, display_name) VALUES (2,'arg','Argentina')"))
        ins = (
            "INSERT INTO matches (competition, kickoff_utc, home_team_id, away_team_id) "
            "VALUES ('world_cup_2026','2026-06-20T18:00:00Z',1,2)"
        )
        conn.execute(text(ins))
    # Duplicate on the natural key must be rejected.
    with pytest.raises(Exception):  # noqa: B017 - IntegrityError surfaces as DBAPIError
        with engine.begin() as conn:
            conn.execute(text(ins))


def test_foreign_keys_enforced(engine):
    # away_team_id references a non-existent team -> rejected.
    with pytest.raises(Exception):  # noqa: B017
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO matches (competition, kickoff_utc, home_team_id, away_team_id) "
                    "VALUES ('international','2026-01-01T00:00:00Z',999,998)"
                )
            )
