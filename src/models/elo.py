"""Elo ratings (World Cup-aware) and a calibrated Elo-only 1X2 benchmark.

The Elo engine produces a strength signal (``elo_pre`` per match, used as a feature
and as a naive benchmark). Home advantage is applied only at non-neutral venues; a
margin-of-victory multiplier damps blowout updates. The Elo-only *probability*
benchmark fits a multinomial logistic of outcome on (rating difference, home flag),
which yields a calibrated three-way split rather than a bare win-expectancy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from utils.logging import get_logger
from utils.naming import team_key

log = get_logger(__name__)

# Outcome encoding shared with the benchmark.
HOME, DRAW, AWAY = 0, 1, 2


def _expected_home(r_home: float, r_away: float, home_adv: float) -> float:
    return 1.0 / (1.0 + 10 ** (-(r_home - r_away + home_adv) / 400.0))


def run_elo(
    matches: pd.DataFrame,
    k: float = 24.0,
    home_adv: float = 60.0,
    base_rating: float = 1500.0,
    mov: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Replay finished matches chronologically, returning pre-match ratings.

    Returns (per-match frame with elo_home_pre/elo_away_pre/elo_diff, final ratings).
    ``matches`` must be sorted-able by kickoff_utc and have home_name/away_name/
    home_goals/away_goals/neutral.
    """
    df = matches.sort_values("kickoff_utc").reset_index(drop=True)
    ratings: dict[str, float] = {}
    out = []
    for row in df.itertuples(index=False):
        hk, ak = team_key(row.home_name), team_key(row.away_name)
        rh = ratings.get(hk, base_rating)
        ra = ratings.get(ak, base_rating)
        hadv = 0.0 if int(getattr(row, "neutral", 0) or 0) else home_adv
        out.append((getattr(row, "match_id", None), rh, ra, rh - ra + hadv))

        # Update only on finished matches.
        if pd.notna(row.home_goals) and pd.notna(row.away_goals):
            gh, ga = int(row.home_goals), int(row.away_goals)
            s_home = 1.0 if gh > ga else (0.5 if gh == ga else 0.0)
            e_home = _expected_home(rh, ra, hadv)
            mult = np.log(abs(gh - ga) + 1) if mov else 1.0
            delta = k * mult * (s_home - e_home)
            ratings[hk] = rh + delta
            ratings[ak] = ra - delta

    pre = pd.DataFrame(out, columns=["match_id", "elo_home_pre", "elo_away_pre", "elo_diff"])
    return pre, ratings


@dataclass
class EloModel:
    """Final ratings + a fitted logistic mapping to 1X2 probabilities."""

    ratings: dict[str, float]
    k: float
    home_adv: float
    base_rating: float
    _clf: LogisticRegression
    _classes: list[int]

    def predict_1x2(self, home: str, away: str, neutral: bool = False) -> tuple[float, float, float]:
        rh = self.ratings.get(team_key(home), self.base_rating)
        ra = self.ratings.get(team_key(away), self.base_rating)
        home_flag = 0 if neutral else 1
        x = np.array([[rh - ra, home_flag]])
        proba = self._clf.predict_proba(x)[0]
        # Reorder to (home, draw, away) regardless of class ordering.
        p = {c: proba[i] for i, c in enumerate(self._classes)}
        return float(p[HOME]), float(p[DRAW]), float(p[AWAY])


def train_elo(
    matches: pd.DataFrame,
    k: float = 24.0,
    home_adv: float = 60.0,
    base_rating: float = 1500.0,
) -> EloModel:
    """Run Elo over finished matches, then fit the 1X2 logistic on pre-match state."""
    fin = matches.dropna(subset=["home_goals", "away_goals"]).sort_values("kickoff_utc")
    pre, ratings = run_elo(fin, k=k, home_adv=home_adv, base_rating=base_rating)
    pre = pre.reset_index(drop=True)
    fin = fin.reset_index(drop=True)

    gh = fin["home_goals"].astype(int).to_numpy()
    ga = fin["away_goals"].astype(int).to_numpy()
    y = np.where(gh > ga, HOME, np.where(gh == ga, DRAW, AWAY))
    home_flag = 1 - fin["neutral"].fillna(0).astype(int).to_numpy()
    # Pure rating diff (without home term) as feature 1; home flag as feature 2.
    rating_diff = (pre["elo_home_pre"] - pre["elo_away_pre"]).to_numpy()
    X = np.column_stack([rating_diff, home_flag])

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, y)
    log.info("elo_trained", n_matches=len(fin), classes=clf.classes_.tolist())
    return EloModel(
        ratings=ratings, k=k, home_adv=home_adv, base_rating=base_rating,
        _clf=clf, _classes=clf.classes_.tolist(),
    )
