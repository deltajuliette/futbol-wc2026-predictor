"""Tests for the thin-sample (min_matches) filter in fit_dixon_coles.

The public international-results dataset contains CONIFA/non-FIFA micro-nations that
play few games and otherwise acquire wildly inflated ratings. ``min_matches`` drops any
team below the threshold (and the matches involving it) before fitting.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from models.dixon_coles import fit_dixon_coles


def _frame() -> pd.DataFrame:
    """Two well-sampled teams playing repeatedly, plus a one-off micro-nation."""
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(30):
        rows.append({
            "kickoff_utc": base + timedelta(days=i),
            "home_name": "Alpha", "away_name": "Beta",
            "home_goals": 2, "away_goals": 1, "neutral": 1,
        })
    # A single appearance for a thin team that thrashes Alpha once.
    rows.append({
        "kickoff_utc": base + timedelta(days=40),
        "home_name": "Microland", "away_name": "Alpha",
        "home_goals": 9, "away_goals": 0, "neutral": 1,
    })
    return pd.DataFrame(rows)


def test_filter_drops_thin_team():
    model = fit_dixon_coles(_frame(), half_life_days=540, min_matches=5)
    assert "microland" not in model.attack
    assert "alpha" in model.attack and "beta" in model.attack
    # The 9-0 outlier is excluded, so Alpha's attack is not dragged down by it.
    assert model.n_matches == 30


def test_default_keeps_all_teams():
    model = fit_dixon_coles(_frame(), half_life_days=540)  # min_matches=0
    assert "microland" in model.attack
    assert model.n_matches == 31


def test_filter_is_recursive_safe_and_nonempty():
    # Threshold above every team's count must raise rather than silently fit on nothing.
    with pytest.raises(ValueError):
        fit_dixon_coles(_frame(), half_life_days=540, min_matches=1000)
