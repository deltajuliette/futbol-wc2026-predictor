"""Normalizer tests for the public international-results CSV (no network)."""

from __future__ import annotations

import pandas as pd
import pytest

from clients.open_results import normalize_results

RAW = pd.DataFrame({
    "date": ["2018-06-14", "2022-12-18", "1990-07-08", "2023-03-01"],
    "home_team": ["Russia", "Argentina", "West Germany", "Brazil"],
    "away_team": ["Saudi Arabia", "France", "Argentina", "Peru"],
    "home_score": [5, 3, 1, None],          # last row unplayed -> dropped
    "away_score": [0, 3, 0, None],
    "tournament": ["FIFA World Cup", "FIFA World Cup", "FIFA World Cup", "Friendly"],
    "city": ["Moscow", "Lusail", "Rome", "Rio"],
    "country": ["Russia", "Qatar", "Italy", "Brazil"],
    "neutral": ["FALSE", "TRUE", "TRUE", "False"],
})


def test_normalize_maps_and_filters():
    out = normalize_results(RAW, since_year=2000)
    # 2018 and 2022 remain (>=2000, finished); 1990 filtered by year; 2023 dropped (no score).
    assert len(out) == 2
    assert set(out.columns) == {
        "date", "competition", "stage", "home_team", "away_team",
        "home_goals", "away_goals", "neutral",
    }
    first = out.iloc[0]
    assert first["competition"] == "international"
    assert first["stage"] == "FIFA World Cup"
    assert first["home_goals"] == 5 and first["away_goals"] == 0
    assert first["neutral"] == 0
    assert out.iloc[1]["neutral"] == 1
    assert first["date"].endswith("Z")


def test_schema_drift_raises():
    with pytest.raises(ValueError):
        normalize_results(pd.DataFrame({"foo": [1]}))


def test_since_filter_excludes_old():
    out = normalize_results(RAW, since_year=2020)
    assert len(out) == 1
    assert out.iloc[0]["home_team"] == "Argentina"
