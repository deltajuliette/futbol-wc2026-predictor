"""Scoreline distribution utilities (Dixon-Coles corrected) and derived markets.

Given expected goals (lambda_home, lambda_away) and the DC low-score dependence
parameter ``rho``, build the full P(home=x, away=y) grid and derive 1X2, BTTS,
over/under, and the most likely scorelines — all from one coherent object.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import poisson


def dc_tau(x: int, y: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Dixon-Coles low-score correction factor for the (x, y) cell."""
    if x == 0 and y == 0:
        return 1.0 - lam_h * lam_a * rho
    if x == 0 and y == 1:
        return 1.0 + lam_h * rho
    if x == 1 and y == 0:
        return 1.0 + lam_a * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(lam_h: float, lam_a: float, rho: float = 0.0, max_goals: int = 10) -> np.ndarray:
    """Return a (max_goals+1, max_goals+1) matrix P[x, y] = P(home=x, away=y)."""
    gh = poisson.pmf(np.arange(max_goals + 1), lam_h)
    ga = poisson.pmf(np.arange(max_goals + 1), lam_a)
    mat = np.outer(gh, ga)
    # Apply DC correction to the 2x2 low-score block.
    for x in (0, 1):
        for y in (0, 1):
            mat[x, y] *= dc_tau(x, y, lam_h, lam_a, rho)
    total = mat.sum()
    if total <= 0:
        raise ValueError("degenerate score matrix")
    return mat / total


@dataclass
class MatchProbabilities:
    p_home: float
    p_draw: float
    p_away: float
    exp_goals_home: float
    exp_goals_away: float
    p_btts: float
    p_over25: float
    top_scorelines: list[tuple[str, float]]

    def as_1x2(self) -> tuple[float, float, float]:
        return self.p_home, self.p_draw, self.p_away


def derive_markets(mat: np.ndarray, top_n: int = 5) -> MatchProbabilities:
    """Collapse a score matrix into 1X2 + derived markets."""
    n = mat.shape[0]
    idx = np.arange(n)
    p_home = float(np.tril(mat, -1).sum())   # x > y
    p_away = float(np.triu(mat, 1).sum())    # x < y
    p_draw = float(np.trace(mat))
    exp_h = float((mat.sum(axis=1) * idx).sum())
    exp_a = float((mat.sum(axis=0) * idx).sum())
    p_btts = float(mat[1:, 1:].sum())
    over = 0.0
    for x in range(n):
        for y in range(n):
            if x + y >= 3:
                over += mat[x, y]
    flat = [((x, y), float(mat[x, y])) for x in range(n) for y in range(n)]
    flat.sort(key=lambda t: t[1], reverse=True)
    top = [(f"{x}-{y}", p) for (x, y), p in flat[:top_n]]
    return MatchProbabilities(
        p_home=p_home, p_draw=p_draw, p_away=p_away,
        exp_goals_home=exp_h, exp_goals_away=exp_a,
        p_btts=p_btts, p_over25=float(over), top_scorelines=top,
    )


def probabilities(lam_h: float, lam_a: float, rho: float = 0.0,
                  max_goals: int = 10, top_n: int = 5) -> MatchProbabilities:
    """Convenience: lambdas -> derived market probabilities."""
    return derive_markets(score_matrix(lam_h, lam_a, rho, max_goals), top_n=top_n)
