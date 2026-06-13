"""ETL: name normalization, idempotent upserts, and CSV loading."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from clients.types import FixtureRecord, Provenance
from storage.dao import load_matches_df, upsert_match, upsert_team
from storage.database import get_engine, init_db
from utils.naming import team_key


@pytest.fixture()
def engine(tmp_path):
    return init_db(get_engine(f"sqlite:///{tmp_path / 'etl.sqlite'}"))


def test_team_key_normalization():
    assert team_key("Côte d'Ivoire") == "cote-divoire"
    assert team_key("South Korea") == "south-korea"
    assert team_key("  Brazil  ") == "brazil"
    with pytest.raises(ValueError):
        team_key("")


def test_upsert_team_is_idempotent(engine):
    a = upsert_team(engine, "Brazil")
    b = upsert_team(engine, "brazil")  # same slug
    assert a == b


def _fixture(home, away, gh=None, ga=None):
    return FixtureRecord(
        competition="international",
        kickoff_utc=datetime(2026, 3, 1, 18, 0, tzinfo=UTC),
        home_team=home,
        away_team=away,
        neutral=False,
        status="finished" if gh is not None else "scheduled",
        home_goals=gh,
        away_goals=ga,
        provenance=Provenance(source="test"),
    )


def test_upsert_match_idempotent_and_updates_result(engine):
    # First as scheduled, then re-load with a score -> same row, updated.
    mid1 = upsert_match(engine, _fixture("Brazil", "Argentina"))
    mid2 = upsert_match(engine, _fixture("Brazil", "Argentina", gh=2, ga=1))
    assert mid1 == mid2
    df = load_matches_df(engine, finished_only=True)
    assert len(df) == 1
    assert int(df.iloc[0]["home_goals"]) == 2
    assert df.iloc[0]["status"] == "finished"


def test_load_matches_df_filters(engine):
    upsert_match(engine, _fixture("Brazil", "Argentina", 1, 0))
    df_all = load_matches_df(engine)
    assert {"home_name", "away_name"}.issubset(df_all.columns)
    assert pd.api.types.is_datetime64_any_dtype(df_all["kickoff_utc"])
