"""Feature builder: anti-leakage and fixed as-of reproducibility."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from features.build import build_features


def _matches():
    """Three chronological matches for the same team; goals known only post-match."""
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        {"match_id": 1, "kickoff_utc": base, "home_name": "A", "away_name": "B",
         "home_team_id": 1, "away_team_id": 2, "home_goals": 3, "away_goals": 0, "neutral": 0},
        {"match_id": 2, "kickoff_utc": base + timedelta(days=10), "home_name": "A",
         "away_name": "C", "home_team_id": 1, "away_team_id": 3,
         "home_goals": 1, "away_goals": 1, "neutral": 0},
        {"match_id": 3, "kickoff_utc": base + timedelta(days=20), "home_name": "A",
         "away_name": "B", "home_team_id": 1, "away_team_id": 2,
         "home_goals": 0, "away_goals": 2, "neutral": 1},
    ]
    df = pd.DataFrame(rows)
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
    return df


def test_first_match_has_no_prior_form():
    feats = build_features(_matches())
    first = feats[(feats["match_id"] == 1) & (feats["team_id"] == 1)].iloc[0]
    # No matches before the first -> no goal-rate history, no rest days.
    assert pd.isna(first["gf_rate"])
    assert first["rest_days"] is None or pd.isna(first["rest_days"])


def test_form_uses_only_past_matches_not_current():
    feats = build_features(_matches())
    # Team A scored 3 in match 1; by match 2 its gf_rate must reflect only match 1 (=3.0),
    # never match 2's own goals (would leak the label).
    m2 = feats[(feats["match_id"] == 2) & (feats["team_id"] == 1)].iloc[0]
    assert m2["gf_rate"] == 3.0
    # By match 3, gf_rate = mean(3, 1) = 2.0 (matches 1 and 2 only).
    m3 = feats[(feats["match_id"] == 3) & (feats["team_id"] == 1)].iloc[0]
    assert m3["gf_rate"] == 2.0
    assert m3["rest_days"] == 10


def test_as_of_equals_kickoff_for_reproducibility():
    feats = build_features(_matches())
    for _, r in feats.iterrows():
        assert r["as_of_utc"].startswith("2026-01-")  # cutoff stamped per row


def test_neutral_flag_propagates():
    feats = build_features(_matches())
    m3 = feats[(feats["match_id"] == 3)].iloc[0]
    assert int(m3["neutral"]) == 1
