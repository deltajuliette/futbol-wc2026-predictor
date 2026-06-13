"""Proper scoring rules and calibration diagnostics for 1X2 forecasts.

Conventions: ``probs`` is an (N, 3) array of [P(home), P(draw), P(away)] rows that
sum to 1; ``outcomes`` is an (N,) integer array with 0=home, 1=draw, 2=away.

We optimize these — never single-match accuracy. See docs/evaluation.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-15


def _validate(probs: np.ndarray, outcomes: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("probs must be (N, 3)")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each prob row must sum to 1")
    return np.clip(p, EPS, 1.0)


def log_loss(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Mean negative log-likelihood of the realized outcomes."""
    p = _validate(probs, outcomes)
    idx = np.asarray(outcomes, dtype=int)
    return float(-np.mean(np.log(p[np.arange(len(idx)), idx])))


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error vs the one-hot outcome."""
    p = _validate(probs, outcomes)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(outcomes)), np.asarray(outcomes, dtype=int)] = 1.0
    return float(np.mean(np.sum((p - onehot) ** 2, axis=1)))


def rps(probs: np.ndarray, outcomes: np.ndarray) -> float:
    """Ranked Probability Score for ordered outcomes (home > draw > away).

    Rewards probability mass placed *near* the truth, the right metric for an
    ordinal 1X2. Lower is better; range [0, 1].
    """
    p = _validate(probs, outcomes)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(outcomes)), np.asarray(outcomes, dtype=int)] = 1.0
    cum_p = np.cumsum(p, axis=1)
    cum_o = np.cumsum(onehot, axis=1)
    # Sum of squared cumulative differences over the first K-1 thresholds / (K-1).
    return float(np.mean(np.sum((cum_p[:, :-1] - cum_o[:, :-1]) ** 2, axis=1)) / (p.shape[1] - 1))


def sharpness(probs: np.ndarray, base_rate: np.ndarray | None = None) -> float:
    """Mean distance of forecasts from the base rate (higher = sharper)."""
    p = _validate(probs, np.zeros(len(probs)))
    if base_rate is None:
        base_rate = p.mean(axis=0)
    return float(np.mean(np.linalg.norm(p - base_rate, axis=1)))


@dataclass
class ReliabilityBin:
    lo: float
    hi: float
    mean_pred: float
    frac_obs: float
    count: int


def reliability_table(probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10) -> list[ReliabilityBin]:
    """Flattened one-vs-rest reliability across all three classes.

    Each (match, class) pair is a binary event with predicted prob = p and observed
    = 1 if that class occurred. Bins compare mean predicted vs observed frequency.
    """
    p = _validate(probs, outcomes)
    onehot = np.zeros_like(p)
    onehot[np.arange(len(outcomes)), np.asarray(outcomes, dtype=int)] = 1.0
    pred = p.ravel()
    obs = onehot.ravel()
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[ReliabilityBin] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (pred >= lo) & (pred < hi) if i < n_bins - 1 else (pred >= lo) & (pred <= hi)
        if mask.sum() == 0:
            continue
        bins.append(ReliabilityBin(
            lo=float(lo), hi=float(hi),
            mean_pred=float(pred[mask].mean()),
            frac_obs=float(obs[mask].mean()),
            count=int(mask.sum()),
        ))
    return bins


@dataclass
class ScoreSet:
    n: int
    log_loss: float
    brier: float
    rps: float
    sharpness: float


def score_all(probs: np.ndarray, outcomes: np.ndarray) -> ScoreSet:
    return ScoreSet(
        n=len(outcomes),
        log_loss=log_loss(probs, outcomes),
        brier=brier_score(probs, outcomes),
        rps=rps(probs, outcomes),
        sharpness=sharpness(probs),
    )
