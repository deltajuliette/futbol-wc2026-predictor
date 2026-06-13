"""Scoring rules, calibration, and de-vig methods."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.devig import overround, proportional, shin
from evaluation.metrics import (
    brier_score,
    log_loss,
    reliability_table,
    rps,
    score_all,
)


def test_perfect_forecast_scores_zero():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    outcomes = np.array([0, 1, 2])
    assert log_loss(probs, outcomes) == pytest.approx(0.0, abs=1e-9)
    assert brier_score(probs, outcomes) == pytest.approx(0.0, abs=1e-12)
    assert rps(probs, outcomes) == pytest.approx(0.0, abs=1e-12)


def test_uniform_forecast_reference_values():
    probs = np.tile([1 / 3, 1 / 3, 1 / 3], (3, 1))
    outcomes = np.array([0, 1, 2])
    assert log_loss(probs, outcomes) == pytest.approx(np.log(3), abs=1e-9)
    # Brier for uniform = sum of (1/3-onehot)^2 = (2/3)^2 + 2*(1/3)^2 = 2/3.
    assert brier_score(probs, outcomes) == pytest.approx(2 / 3, abs=1e-9)


def test_rps_rewards_near_misses_over_far_misses():
    outcome = np.array([0])  # home actually won
    near = np.array([[0.4, 0.5, 0.1]])  # mass on draw (adjacent)
    far = np.array([[0.4, 0.1, 0.5]])   # mass on away (far)
    assert rps(near, outcome) < rps(far, outcome)


def test_invalid_probs_rejected():
    with pytest.raises(ValueError):
        log_loss(np.array([[0.5, 0.4, 0.4]]), np.array([0]))


def test_reliability_table_well_calibrated():
    rng = np.random.default_rng(0)
    # Draw outcomes from the forecasts themselves -> should be calibrated.
    base = rng.dirichlet([3, 2, 3], size=4000)
    outs = np.array([rng.choice(3, p=row) for row in base])
    bins = reliability_table(base, outs, n_bins=10)
    for b in bins:
        if b.count > 50:
            assert abs(b.mean_pred - b.frac_obs) < 0.06


def test_devig_methods_sum_to_one_and_reduce_overround():
    ho, do, ao = 2.10, 3.40, 3.60
    assert overround(ho, do, ao) > 1.0
    for method in (proportional, shin):
        p = method(ho, do, ao)
        assert sum(p) == pytest.approx(1.0, abs=1e-9)
        assert all(x > 0 for x in p)
    # Favorite keeps the highest probability.
    assert proportional(ho, do, ao)[0] == max(proportional(ho, do, ao))


def test_score_all_shape():
    probs = np.tile([0.5, 0.3, 0.2], (10, 1))
    outcomes = np.zeros(10, dtype=int)
    s = score_all(probs, outcomes)
    assert s.n == 10 and s.log_loss > 0
