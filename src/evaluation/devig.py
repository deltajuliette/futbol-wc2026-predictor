"""Convert bookmaker 1X2 decimal odds to de-vigged probabilities.

Two methods (label which one you used): basic proportional normalization, and Shin's
method which accounts for favorite-longshot bias more honestly. See docs/evaluation.md.
Market is a benchmark, never ground truth.
"""

from __future__ import annotations

import numpy as np


def overround(home_odds: float, draw_odds: float, away_odds: float) -> float:
    """Book sum (1/odds). >1 implies the bookmaker margin (vig)."""
    return 1.0 / home_odds + 1.0 / draw_odds + 1.0 / away_odds


def proportional(home_odds: float, draw_odds: float, away_odds: float) -> tuple[float, float, float]:
    """Normalize raw implied probabilities to sum to 1."""
    raw = np.array([1.0 / home_odds, 1.0 / draw_odds, 1.0 / away_odds])
    p = raw / raw.sum()
    return float(p[0]), float(p[1]), float(p[2])


def shin(home_odds: float, draw_odds: float, away_odds: float,
         max_iter: int = 100, tol: float = 1e-12) -> tuple[float, float, float]:
    """Shin (1992) de-vig: solve for insider-trading proportion z, return fair probs.

    p_i = (sqrt(z^2 + 4(1-z) * q_i^2 / sum_q) - z) / (2(1-z)), with q_i = 1/odds_i.
    """
    q = np.array([1.0 / home_odds, 1.0 / draw_odds, 1.0 / away_odds])
    booksum = q.sum()
    z = 0.0
    for _ in range(max_iter):
        denom = 2.0 * (1.0 - z)
        p = (np.sqrt(z * z + 4.0 * (1.0 - z) * q * q / booksum) - z) / denom
        p_sum = p.sum()
        # Newton-free fixed point: nudge z so probabilities sum to 1.
        z_new = z + (p_sum - 1.0)
        z_new = min(max(z_new, 0.0), 0.2)
        if abs(z_new - z) < tol:
            z = z_new
            break
        z = z_new
    denom = 2.0 * (1.0 - z)
    p = (np.sqrt(z * z + 4.0 * (1.0 - z) * q * q / booksum) - z) / denom
    p = p / p.sum()
    return float(p[0]), float(p[1]), float(p[2])
