"""Cross-confederation correction: mechanics, fitting, and reasoning decomposition.

The correction must (1) be an exact no-op when disabled, (2) shift only the goal
*difference* and only on inter-confederation matches, (3) stay centered at zero, and
(4) keep the reasoning decomposition exact: goals + continent == log(lam_h/lam_a).
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from explain.reasons import explain
from models.dixon_coles import DCModel, fit_dixon_coles
from models.scoreline import probabilities


def _model(conf_adj=None) -> DCModel:
    return DCModel(
        attack={"alpha": 0.20, "beta": -0.20, "gamma": 0.10, "delta": -0.10},
        defense={"alpha": 0.15, "beta": -0.15, "gamma": 0.05, "delta": -0.05},
        intercept=0.10, home_adv=0.25, rho=-0.05, half_life_days=540.0,
        trained_through="2026-01-01T00:00:00+00:00", n_matches=4,
        conf_adj=conf_adj or {},
    )


def test_no_op_when_disabled():
    m = _model()  # empty conf_adj
    assert m.conf_edge("UEFA", "AFC") == 0.0
    base = m.predict_lambdas("Alpha", "Beta", neutral=True)
    with_conf = m.predict_lambdas("Alpha", "Beta", neutral=True,
                                  home_conf="UEFA", away_conf="AFC")
    assert base == with_conf


def test_antisymmetric_and_difference_only():
    m = _model({"UEFA": 0.30, "AFC": -0.30})
    lam_h, lam_a = m.predict_lambdas("Alpha", "Beta", neutral=True,
                                     home_conf="UEFA", away_conf="AFC")
    lam_h0, lam_a0 = m.predict_lambdas("Alpha", "Beta", neutral=True)
    edge = m.conf_edge("UEFA", "AFC")
    assert edge == pytest.approx(0.60)
    # Goal difference shifts by exactly conf_edge...
    assert (math.log(lam_h) - math.log(lam_a)) - (math.log(lam_h0) - math.log(lam_a0)) \
        == pytest.approx(edge, abs=1e-12)
    # ...while expected total goals (sum of logs) is unchanged.
    assert math.log(lam_h) + math.log(lam_a) \
        == pytest.approx(math.log(lam_h0) + math.log(lam_a0), abs=1e-12)


def test_intra_and_unknown_confederation_are_zero():
    m = _model({"UEFA": 0.30, "AFC": -0.30})
    assert m.conf_edge("UEFA", "UEFA") == 0.0      # same pool
    assert m.conf_edge("UEFA", None) == 0.0        # unknown side
    assert m.conf_edge(None, None) == 0.0


def test_reasoning_decomposition_includes_confederation():
    m = _model({"UEFA": 0.30, "AFC": -0.30})
    lam_h, lam_a = m.predict_lambdas("Alpha", "Beta", neutral=True,
                                     home_conf="UEFA", away_conf="AFC")
    mp = probabilities(lam_h, lam_a, rho=m.rho)
    bundle = explain("Alpha", "Beta", neutral=True, model=m, elo_diff=120.0,
                     p_cal=mp.as_1x2(), p_raw=mp.as_1x2(), elo_probs=(0.6, 0.25, 0.15),
                     mp=mp, recent_counts={"alpha": 40, "beta": 40},
                     home_conf="UEFA", away_conf="AFC")
    goals = next(d for d in bundle.drivers if d.kind == "goals")
    cont = next(d for d in bundle.drivers if d.kind == "continent")
    # Exact additive decomposition: within-pool + cross-pool == total log-ratio.
    assert goals.magnitude + cont.magnitude \
        == pytest.approx(math.log(lam_h) - math.log(lam_a), abs=1e-12)
    assert cont.direction == "home" and cont.magnitude > 0


def test_no_continent_driver_for_intra_confederation():
    m = _model({"UEFA": 0.30, "AFC": -0.30})
    lam_h, lam_a = m.predict_lambdas("Alpha", "Gamma", neutral=True,
                                     home_conf="UEFA", away_conf="UEFA")
    mp = probabilities(lam_h, lam_a, rho=m.rho)
    bundle = explain("Alpha", "Gamma", neutral=True, model=m, elo_diff=80.0,
                     p_cal=mp.as_1x2(), p_raw=mp.as_1x2(), elo_probs=None, mp=mp,
                     recent_counts={"alpha": 40, "gamma": 40},
                     home_conf="UEFA", away_conf="UEFA")
    assert not [d for d in bundle.drivers if d.kind == "continent"]


def _synthetic_matches() -> pd.DataFrame:
    """Two pools A/B. Intra-pool is balanced; inter-pool A dominates B — so the
    cross-pool term (not attack/defense) should carry A above B."""
    rows = []
    base = pd.Timestamp("2025-01-01T00:00:00Z")
    teams_a = ["A1", "A2", "A3"]
    teams_b = ["B1", "B2", "B3"]
    conf = {**{t: "CONF_A" for t in teams_a}, **{t: "CONF_B" for t in teams_b}}
    n = 0
    # Balanced intra-pool draws (no internal attack/defense spread).
    for _ in range(6):
        for h, a in [("A1", "A2"), ("A2", "A3"), ("A3", "A1"),
                     ("B1", "B2"), ("B2", "B3"), ("B3", "B1")]:
            rows.append((h, a, 1, 1)); n += 1
    # Inter-pool: A beats B 3-0 repeatedly.
    for _ in range(8):
        for h in teams_a:
            for a in teams_b:
                rows.append((h, a, 3, 0)); n += 1
    df = pd.DataFrame(rows, columns=["home_name", "away_name", "home_goals", "away_goals"])
    df["kickoff_utc"] = [base + pd.Timedelta(days=i) for i in range(len(df))]
    df["neutral"] = 1
    df["home_conf"] = df["home_name"].map(conf)
    df["away_conf"] = df["away_name"].map(conf)
    return df


def test_fit_recovers_directional_and_centered_offsets():
    df = _synthetic_matches()
    m = fit_dixon_coles(df, half_life_days=100000, use_confederation=True, maxiter=400)
    assert set(m.conf_adj) == {"CONF_A", "CONF_B"}
    # Centered at zero (identifiability) and A above B (the inter-pool dominance).
    assert sum(m.conf_adj.values()) == pytest.approx(0.0, abs=1e-9)
    assert m.conf_adj["CONF_A"] > m.conf_adj["CONF_B"]


def test_fit_without_confederation_leaves_conf_adj_empty():
    df = _synthetic_matches()
    m = fit_dixon_coles(df, half_life_days=100000, use_confederation=False)
    assert m.conf_adj == {}
