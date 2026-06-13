"""Probability calibration layer for 1X2 forecasts.

Multinomial logistic recalibration on log-probabilities (a multiclass Platt scaling).
Fit on held-out / earlier data, then applied forward; both raw and calibrated
probabilities are persisted. See docs/modeling.md (Layer 5).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-15


class ProbabilityCalibrator:
    """Maps raw 1X2 probabilities to calibrated ones via multinomial logistic."""

    def __init__(self, C: float = 1.0):
        self._clf = LogisticRegression(max_iter=1000, C=C)
        self._classes: list[int] = []
        self.fitted = False

    @staticmethod
    def _features(probs: np.ndarray) -> np.ndarray:
        return np.log(np.clip(np.asarray(probs, dtype=float), EPS, 1.0))

    def fit(self, probs: np.ndarray, outcomes: np.ndarray) -> "ProbabilityCalibrator":
        x = self._features(probs)
        y = np.asarray(outcomes, dtype=int)
        # Need all three classes present to calibrate a 3-way split.
        if len(np.unique(y)) < 3:
            self.fitted = False
            return self
        self._clf.fit(x, y)
        self._classes = self._clf.classes_.tolist()
        self.fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities; identity if not fitted."""
        p = np.asarray(probs, dtype=float)
        if not self.fitted:
            return p
        proba = self._clf.predict_proba(self._features(p))
        # Reorder columns to (home, draw, away) = (0, 1, 2).
        out = np.zeros((len(p), 3))
        for i, c in enumerate(self._classes):
            out[:, c] = proba[:, i]
        return out
