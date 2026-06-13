"""Duplicate-team merge + alias-aware resolution.

Guards the fix for split team entities (e.g. "Cape Verde" vs "Cape Verde Islands"):
the merge must repoint every reference onto the canonical team, seed an alias, drop
the orphan, be idempotent, and thereafter resolve the variant name automatically.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import text

from clients.types import FixtureRecord, Provenance
from scripts.etl.merge_duplicate_teams import merge_teams
from storage.dao import upsert_match, upsert_team
from storage.database import get_engine, init_db
from utils.naming import team_key


@pytest.fixture()
def engine(tmp_path):
    return init_db(get_engine(f"sqlite:///{tmp_path / 'merge.sqlite'}"))


def _match(engine, home, away, status="finished", gh=1, ga=0):
    rec = FixtureRecord(
        competition="international", stage="friendly",
        kickoff_utc=datetime(2025, 3, 1, 18, tzinfo=UTC),
        home_team=home, away_team=away, neutral=False, status=status,
        home_goals=gh, away_goals=ga, provenance=Provenance(source="test"),
    )
    return upsert_match(engine, rec)


def test_merge_repoints_and_aliases(engine):
    # Canonical team with history; variant entered under a different slug.
    _match(engine, "Cape Verde", "Senegal")
    _match(engine, "Cape Verde Islands", "Spain", status="scheduled", gh=None, ga=None)
    cid = upsert_team(engine, "Cape Verde")
    vid = upsert_team(engine, "Cape Verde Islands")
    assert cid != vid  # distinct before merge

    n = merge_teams(engine, {"cape-verde-islands": "cape-verde"})
    assert n == 1

    with engine.connect() as conn:
        # Orphan removed.
        assert conn.execute(text("SELECT COUNT(*) FROM teams WHERE team_id=:v"),
                            {"v": vid}).scalar() == 0
        # Every match now points at the canonical id.
        assert conn.execute(
            text("SELECT COUNT(*) FROM matches WHERE :v IN (home_team_id, away_team_id)"),
            {"v": vid}).scalar() == 0
        assert conn.execute(
            text("SELECT COUNT(*) FROM matches WHERE :c IN (home_team_id, away_team_id)"),
            {"c": cid}).scalar() == 2
        # Alias seeded.
        assert conn.execute(text("SELECT team_id FROM team_aliases WHERE alias=:a"),
                            {"a": "cape-verde-islands"}).scalar() == cid


def test_merge_is_idempotent(engine):
    _match(engine, "Cape Verde", "Senegal")
    _match(engine, "Cape Verde Islands", "Spain", status="scheduled", gh=None, ga=None)
    merge_teams(engine, {"cape-verde-islands": "cape-verde"})
    # Second run finds nothing to merge but keeps the alias.
    assert merge_teams(engine, {"cape-verde-islands": "cape-verde"}) == 0
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM team_aliases WHERE alias=:a"),
                            {"a": "cape-verde-islands"}).scalar() == 1


def test_variant_name_resolves_via_alias_after_merge(engine):
    cid = upsert_team(engine, "Cape Verde")
    _match(engine, "Cape Verde Islands", "Spain", status="scheduled", gh=None, ga=None)
    merge_teams(engine, {"cape-verde-islands": "cape-verde"})
    # A fresh ingest of the variant name must now resolve to the canonical team.
    assert upsert_team(engine, "Cape Verde Islands") == cid
    assert team_key("Cape Verde Islands") == "cape-verde-islands"  # slug itself unchanged


def test_distinct_nations_not_merged(engine):
    """Sanity: only configured variants merge; lookalike nations are left alone."""
    congo = upsert_team(engine, "Congo")
    drc = upsert_team(engine, "DR Congo")
    merge_teams(engine, {"congo-dr": "dr-congo"})  # neither slug present → no-op
    with engine.connect() as conn:
        assert conn.execute(text("SELECT COUNT(*) FROM teams WHERE team_id IN (:a,:b)"),
                            {"a": congo, "b": drc}).scalar() == 2
