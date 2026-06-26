"""Tests for the tournament-importance weight in fit_dixon_coles.

``tournament_weight`` multiplies the time-decay weight of matches whose ``stage`` matches
``tournament_pattern``, so a few tournament results can count for more than the same
number of friendlies. The default 1.0 must be an exact no-op.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from models.dixon_coles import fit_dixon_coles


def _frame() -> pd.DataFrame:
    """Alpha dominates Beta in friendlies but loses to it at the World Cup."""
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(20):
        rows.append({
            "kickoff_utc": base + timedelta(days=i),
            "home_name": "Alpha", "away_name": "Beta",
            "home_goals": 3, "away_goals": 0, "neutral": 1, "stage": "Friendly",
        })
    for i in range(5):
        rows.append({
            "kickoff_utc": base + timedelta(days=100 + i),
            "home_name": "Beta", "away_name": "Alpha",
            "home_goals": 3, "away_goals": 0, "neutral": 1, "stage": "FIFA World Cup",
        })
    return pd.DataFrame(rows)


def test_default_weight_is_noop():
    """tw=1.0 must give the same fit whether or not a stage column is present."""
    df = _frame()
    with_stage = fit_dixon_coles(df, half_life_days=1e6, tournament_weight=1.0)
    without_stage = fit_dixon_coles(df.drop(columns=["stage"]), half_life_days=1e6,
                                    tournament_weight=1.0)
    for k in with_stage.attack:
        assert with_stage.attack[k] == without_stage.attack[k]
        assert with_stage.defense[k] == without_stage.defense[k]


def test_weight_shifts_strength_toward_tournament_result():
    """Up-weighting the World Cup games should lift Beta's standing vs Alpha.

    Asserts on the predicted goal difference for an Alpha-vs-Beta fixture (an identified
    quantity) rather than on raw attack/defense, which are only pinned by the ridge with
    so few teams. Large half-life isolates the weighting effect from time decay.
    """
    df = _frame()
    low = fit_dixon_coles(df, half_life_days=1e6, tournament_weight=1.0)
    high = fit_dixon_coles(df, half_life_days=1e6, tournament_weight=10.0)

    def goal_diff(m):  # Alpha (home) minus Beta expected goals, neutral venue
        lam_h, lam_a = m.predict_lambdas("Alpha", "Beta", neutral=True)
        return lam_h - lam_a

    # Friendlies favor Alpha; weighting up the WC losses must swing the edge toward Beta.
    assert goal_diff(high) < goal_diff(low)


def test_pattern_must_match_to_have_effect():
    """A pattern that matches no stage leaves the fit unchanged."""
    df = _frame()
    base = fit_dixon_coles(df, half_life_days=1e6, tournament_weight=1.0)
    nomatch = fit_dixon_coles(df, half_life_days=1e6, tournament_weight=10.0,
                              tournament_pattern="Olympics")
    for k in base.attack:
        assert abs(base.attack[k] - nomatch.attack[k]) < 1e-9
