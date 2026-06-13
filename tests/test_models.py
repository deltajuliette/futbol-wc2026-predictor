"""Modeling core: scoreline algebra, Dixon-Coles fit, Elo, feature leak-safety."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from models.dixon_coles import fit_dixon_coles
from models.elo import train_elo
from models.scoreline import derive_markets, probabilities, score_matrix


def test_score_matrix_normalized_and_1x2_sums_to_one():
    mat = score_matrix(1.6, 1.1, rho=-0.05, max_goals=10)
    assert mat.sum() == pytest.approx(1.0, abs=1e-9)
    mp = derive_markets(mat)
    assert mp.p_home + mp.p_draw + mp.p_away == pytest.approx(1.0, abs=1e-9)
    # Home stronger -> home favored.
    assert mp.p_home > mp.p_away


def test_expected_goals_recovered_from_matrix():
    mp = probabilities(2.0, 0.8, rho=0.0, max_goals=15)
    assert mp.exp_goals_home == pytest.approx(2.0, abs=0.02)
    assert mp.exp_goals_away == pytest.approx(0.8, abs=0.02)


def _synth_matches(seed=1, n_teams=10, n_days=400):
    """Small independent-Poisson dataset with a known home advantage."""
    rng = np.random.default_rng(seed)
    teams = [f"Team{i}" for i in range(n_teams)]
    atk = rng.normal(0, 0.3, n_teams)
    dfn = rng.normal(0, 0.3, n_teams)
    home_adv, mu = 0.25, 0.2
    rows = []
    start = datetime(2024, 1, 1, tzinfo=UTC)
    for d in range(n_days):
        order = rng.permutation(n_teams)
        for i in range(0, n_teams - 1, 2):
            h, a = int(order[i]), int(order[i + 1])
            lam_h = np.exp(mu + atk[h] - dfn[a] + home_adv)
            lam_a = np.exp(mu + atk[a] - dfn[h])
            rows.append({
                "match_id": len(rows),
                "kickoff_utc": start + timedelta(days=d),
                "home_name": teams[h], "away_name": teams[a],
                "home_team_id": h, "away_team_id": a,
                "home_goals": int(rng.poisson(lam_h)),
                "away_goals": int(rng.poisson(lam_a)),
                "neutral": 0,
            })
    df = pd.DataFrame(rows)
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
    return df


def test_dixon_coles_recovers_home_advantage_and_sums_to_one():
    df = _synth_matches()
    model = fit_dixon_coles(df, half_life_days=0)  # no decay for a stationary DGP
    # True home advantage was 0.25 on the log scale.
    assert model.home_adv == pytest.approx(0.25, abs=0.08)
    lam_h, lam_a = model.predict_lambdas("Team0", "Team1", neutral=False)
    assert lam_h > 0 and lam_a > 0
    p = probabilities(lam_h, lam_a, model.rho)
    assert sum(p.as_1x2()) == pytest.approx(1.0, abs=1e-9)


def test_dixon_coles_neutral_removes_home_term():
    df = _synth_matches()
    model = fit_dixon_coles(df, half_life_days=0)
    lam_h_home, _ = model.predict_lambdas("Team0", "Team1", neutral=False)
    lam_h_neut, _ = model.predict_lambdas("Team0", "Team1", neutral=True)
    assert lam_h_home > lam_h_neut  # home advantage raises home lambda


def test_elo_benchmark_sums_to_one():
    df = _synth_matches()
    model = train_elo(df)
    p = model.predict_1x2("Team0", "Team1", neutral=True)
    assert sum(p) == pytest.approx(1.0, abs=1e-9)
    assert all(0 <= x <= 1 for x in p)
