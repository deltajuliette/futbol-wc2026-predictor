"""Deterministic prediction-reasoning engine.

The rigor check: because Dixon-Coles is log-linear, the goals driver's magnitude
(attack + defense + venue) must equal the model's own log(lambda_home/lambda_away)
*exactly*. We also assert ranking order, the thin-sample caveat, and byte-for-byte
reproducibility (same inputs -> identical JSON).
"""

from __future__ import annotations

import math

import pytest

from explain.reasons import REASONING_VERSION, ReasoningBundle, explain
from models.dixon_coles import DCModel
from models.scoreline import probabilities
from utils.naming import team_key


def _model() -> DCModel:
    # Spain: strong attack, strong defense; Cape Verde: weak both. (defense: higher = better)
    return DCModel(
        attack={"spain": 0.55, "cape-verde": -0.40, "brazil": 0.45, "argentina": 0.40},
        defense={"spain": 0.45, "cape-verde": -0.35, "brazil": 0.40, "argentina": 0.38},
        intercept=0.10, home_adv=0.25, rho=-0.05, half_life_days=540.0,
        trained_through="2026-06-01T00:00:00+00:00", n_matches=4,
    )


def _bundle(model, home, away, neutral, recent=None):
    lam_h, lam_a = model.predict_lambdas(home, away, neutral)
    mp = probabilities(lam_h, lam_a, rho=model.rho)
    p_raw = mp.as_1x2()
    # A mild calibration nudge so the calibration driver has something to report.
    p_cal = (p_raw[0] - 0.04, p_raw[1] + 0.02, p_raw[2] + 0.02)
    return explain(
        home, away, neutral=neutral, model=model, elo_diff=210.0,
        p_cal=p_cal, p_raw=p_raw, elo_probs=(0.80, 0.12, 0.08), mp=mp,
        recent_counts=recent if recent is not None
        else {team_key(home): 60, team_key(away): 40},
    )


def test_goals_decomposition_is_exact():
    """attack_edge + defense_edge + venue == log(lambda_home) - log(lambda_away)."""
    m = _model()
    lam_h, lam_a = m.predict_lambdas("Spain", "Cape Verde", neutral=True)
    bundle = _bundle(m, "Spain", "Cape Verde", neutral=True)
    goals = next(d for d in bundle.drivers if d.kind == "goals")
    assert goals.magnitude == pytest.approx(math.log(lam_h) - math.log(lam_a), abs=1e-12)


def test_venue_term_drops_out_on_neutral():
    m = _model()
    h_neutral = _bundle(m, "Spain", "Cape Verde", neutral=True)
    h_home = _bundle(m, "Spain", "Cape Verde", neutral=False)
    g_neutral = next(d for d in h_neutral.drivers if d.kind == "goals").magnitude
    g_home = next(d for d in h_home.drivers if d.kind == "goals").magnitude
    # Home venue adds exactly home_adv to the log-ratio.
    assert g_home - g_neutral == pytest.approx(m.home_adv, abs=1e-12)


def test_drivers_ranked_by_salience():
    bundle = _bundle(_model(), "Spain", "Cape Verde", neutral=True)
    sals = [d.salience for d in bundle.drivers]
    assert sals == sorted(sals, reverse=True)
    assert {"strength_gap", "goals"} <= {d.kind for d in bundle.drivers}


def test_thin_sample_caveat_fires():
    bundle = _bundle(_model(), "Spain", "Cape Verde", neutral=True,
                     recent={"spain": 60, "cape-verde": 3})
    unc = [d for d in bundle.drivers if d.kind == "uncertainty"]
    assert unc and "Cape Verde" in unc[0].text


def test_no_thin_caveat_when_well_sampled():
    bundle = _bundle(_model(), "Spain", "Cape Verde", neutral=True,
                     recent={"spain": 60, "cape-verde": 40})
    assert not [d for d in bundle.drivers if d.kind == "uncertainty"]


def test_reproducible_and_round_trips():
    m = _model()
    a = _bundle(m, "Spain", "Cape Verde", neutral=True)
    b = _bundle(m, "Spain", "Cape Verde", neutral=True)
    assert a.to_json() == b.to_json()                 # deterministic
    restored = ReasoningBundle.from_json(a.to_json())
    assert restored.headline == a.headline
    assert [d.kind for d in restored.drivers] == [d.kind for d in a.drivers]
    assert restored.version == REASONING_VERSION


def test_edge_driver_skipped_without_benchmark():
    m = _model()
    lam_h, lam_a = m.predict_lambdas("Brazil", "Argentina", neutral=True)
    mp = probabilities(lam_h, lam_a, rho=m.rho)
    bundle = explain(
        "Brazil", "Argentina", neutral=True, model=m, elo_diff=20.0,
        p_cal=mp.as_1x2(), p_raw=mp.as_1x2(), elo_probs=None, mp=mp,
        recent_counts={"brazil": 50, "argentina": 50},
    )
    assert not [d for d in bundle.drivers if d.kind == "edge"]
