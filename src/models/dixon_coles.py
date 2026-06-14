"""Dixon-Coles bivariate-Poisson goals model with time decay.

Fits per-team attack/defense strengths, a baseline scoring rate, a home advantage
(applied only at non-neutral venues), and the low-score dependence parameter ``rho``,
by weighted maximum likelihood with exponential time decay. Identifiability is handled
by centering attack/defense at zero each evaluation, plus a small ridge penalty that
also stabilizes teams with few matches.

Training and scoring are separate: :func:`fit_dixon_coles` returns a serializable
:class:`DCModel`; :meth:`DCModel.predict_lambdas` scores a fixture.

Example::

    model = fit_dixon_coles(matches_df, half_life_days=365)
    lam_h, lam_a = model.predict_lambdas("brazil", "argentina", neutral=True)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

from utils.logging import get_logger
from utils.naming import team_key

log = get_logger(__name__)


@dataclass
class DCModel:
    """Fitted Dixon-Coles parameters (JSON-serializable)."""

    attack: dict[str, float]
    defense: dict[str, float]
    intercept: float
    home_adv: float
    rho: float
    half_life_days: float
    trained_through: str          # ISO date of latest match used
    n_matches: int
    max_goals: int = 10
    # Per-confederation relative-strength offsets (centered at zero). Empty => the
    # confederation correction is off and predict_lambdas is an exact no-op for it.
    conf_adj: dict[str, float] = field(default_factory=dict)

    def conf_edge(self, home_conf: str | None, away_conf: str | None) -> float:
        """Home-minus-away confederation strength on inter-confederation matches.

        Zero when the correction is disabled, a confederation is unknown, or both
        sides share a confederation (so it never perturbs intra-confederation games).
        """
        if not self.conf_adj or not home_conf or not away_conf or home_conf == away_conf:
            return 0.0
        return self.conf_adj.get(home_conf, 0.0) - self.conf_adj.get(away_conf, 0.0)

    def predict_lambdas(self, home: str, away: str, neutral: bool = False,
                        home_conf: str | None = None,
                        away_conf: str | None = None) -> tuple[float, float]:
        """Expected goals (lambda_home, lambda_away) for a fixture by team key.

        The confederation term is applied antisymmetrically (+d/2 to home, -d/2 to
        away), so it shifts the goal *difference* by exactly ``conf_edge`` while
        leaving expected total goals unchanged.
        """
        h, a = team_key(home), team_key(away)
        ah, dh = self.attack.get(h, 0.0), self.defense.get(h, 0.0)
        aa, da = self.attack.get(a, 0.0), self.defense.get(a, 0.0)
        hadv = 0.0 if neutral else self.home_adv
        cdelta = 0.5 * self.conf_edge(home_conf, away_conf)
        lam_h = float(np.exp(self.intercept + ah - da + hadv + cdelta))
        lam_a = float(np.exp(self.intercept + aa - dh - cdelta))
        return lam_h, lam_a

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DCModel:
        return cls(**d)


def _decay_weights(dates: pd.Series, as_of: datetime, half_life_days: float) -> np.ndarray:
    age_days = (as_of - dates).dt.total_seconds().to_numpy() / 86400.0
    if half_life_days <= 0:
        return np.ones_like(age_days)
    xi = np.log(2) / half_life_days
    return np.exp(-xi * np.clip(age_days, 0, None))


def fit_dixon_coles(
    matches: pd.DataFrame,
    half_life_days: float = 365.0,
    as_of: datetime | None = None,
    max_goals: int = 10,
    ridge: float = 1e-3,
    rho_bound: float = 0.18,
    maxiter: int = 250,
    use_confederation: bool = False,
    conf_ridge: float = 5.0,
    conf_bound: float = 0.5,
    min_matches: int = 0,
) -> DCModel:
    """Fit the model on finished matches.

    ``matches`` needs columns: kickoff_utc (tz-aware), home_name, away_name,
    home_goals, away_goals, neutral. Only rows with both goals set are used.

    When ``use_confederation`` is set and the frame carries ``home_conf``/``away_conf``
    columns, a per-confederation relative-strength offset is fit *jointly* and applied
    only to inter-confederation matches — a correction for the weak rating linkage
    between confederations that rarely play each other. It is centered at zero, ridge-
    shrunk (``conf_ridge``), and bounded (``conf_bound``). Teams with an unknown
    confederation contribute no confederation term.

    ``min_matches`` drops thin-sample teams: any team appearing in fewer than
    ``min_matches`` finished games is excluded, along with the matches involving it.
    This removes CONIFA/non-FIFA micro-nations (which play few games against each other
    and otherwise acquire wildly inflated ratings) from the public-results dataset. The
    default 0 keeps all teams.
    """
    df = matches.dropna(subset=["home_goals", "away_goals"]).copy()
    if df.empty:
        raise ValueError("no finished matches to fit on")

    if min_matches > 0:
        hk = df["home_name"].map(team_key)
        ak = df["away_name"].map(team_key)
        appearances = pd.concat([hk, ak]).value_counts()
        keep_keys = set(appearances[appearances >= min_matches].index)
        mask = hk.isin(keep_keys) & ak.isin(keep_keys)
        dropped_teams = len(appearances) - len(keep_keys)
        if dropped_teams:
            log.info("dixon_coles_min_matches_filter", min_matches=min_matches,
                     dropped_teams=dropped_teams, dropped_matches=int((~mask).sum()),
                     kept_matches=int(mask.sum()))
        df = df[mask].copy()
        if df.empty:
            raise ValueError(f"no matches left after min_matches={min_matches} filter")

    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    as_of = as_of or df["kickoff_utc"].max().to_pydatetime()

    teams = sorted(set(df["home_name"]).union(df["away_name"]), key=team_key)
    keys = [team_key(t) for t in teams]
    kidx = {k: i for i, k in enumerate(keys)}
    n = len(teams)

    hi = df["home_name"].map(lambda t: kidx[team_key(t)]).to_numpy()
    ai = df["away_name"].map(lambda t: kidx[team_key(t)]).to_numpy()
    gh = df["home_goals"].to_numpy()
    ga = df["away_goals"].to_numpy()
    neutral = df["neutral"].fillna(0).astype(int).to_numpy()
    w = _decay_weights(df["kickoff_utc"], as_of, half_life_days)

    # Optional confederation indices (-1 = unknown). Only inter-confederation matches
    # where BOTH sides are known carry the term.
    fit_conf = use_confederation and {"home_conf", "away_conf"} <= set(df.columns)
    if fit_conf:
        confs = sorted({c for c in pd.concat([df["home_conf"], df["away_conf"]]).dropna()
                        .unique()})
        ucidx = {c: i for i, c in enumerate(confs)}
        ch = df["home_conf"].map(lambda c: ucidx.get(c, -1)).to_numpy()
        ca = df["away_conf"].map(lambda c: ucidx.get(c, -1)).to_numpy()
        inter = (ch >= 0) & (ca >= 0) & (ch != ca)
        nc = len(confs)
    else:
        confs, nc = [], 0

    # Precompute constant log-factorials.
    const = -(gammaln(gh + 1) + gammaln(ga + 1))

    # Param layout: [attack(n), defense(n), intercept, home_adv, rho, conf(nc)].
    def unpack(p):
        atk = p[:n] - p[:n].mean()
        dfn = p[n:2 * n] - p[n:2 * n].mean()
        conf = (p[2 * n + 3:2 * n + 3 + nc] - p[2 * n + 3:2 * n + 3 + nc].mean()
                if nc else np.zeros(0))
        return atk, dfn, p[2 * n], p[2 * n + 1], p[2 * n + 2], conf

    def neg_ll(p):
        atk, dfn, c, gamma, rho, conf = unpack(p)
        loglam_h = c + atk[hi] - dfn[ai] + gamma * (1 - neutral)
        loglam_a = c + atk[ai] - dfn[hi]
        if nc:
            sh = conf[np.clip(ch, 0, nc - 1)]
            sa = conf[np.clip(ca, 0, nc - 1)]
            delta = np.where(inter, 0.5 * (sh - sa), 0.0)
            loglam_h = loglam_h + delta
            loglam_a = loglam_a - delta
        lam_h = np.exp(loglam_h)
        lam_a = np.exp(loglam_a)
        ll_pois = gh * loglam_h - lam_h + ga * loglam_a - lam_a + const
        # Dixon-Coles tau (vectorized over the low-score cells).
        tau = np.ones_like(lam_h)
        m00 = (gh == 0) & (ga == 0)
        m01 = (gh == 0) & (ga == 1)
        m10 = (gh == 1) & (ga == 0)
        m11 = (gh == 1) & (ga == 1)
        tau[m00] = 1 - lam_h[m00] * lam_a[m00] * rho
        tau[m01] = 1 + lam_h[m01] * rho
        tau[m10] = 1 + lam_a[m10] * rho
        tau[m11] = 1 - rho
        # Guard against non-positive tau (invalid rho region).
        tau = np.clip(tau, 1e-9, None)
        ll = w * (ll_pois + np.log(tau))
        penalty = ridge * (np.sum(p[:n] ** 2) + np.sum(p[n:2 * n] ** 2))
        if nc:
            penalty += conf_ridge * np.sum(p[2 * n + 3:2 * n + 3 + nc] ** 2)
        return -(ll.sum()) + penalty

    x0 = np.zeros(2 * n + 3 + nc)
    x0[2 * n] = np.log(max(gh.mean(), 0.1))   # intercept ~ log mean goals
    x0[2 * n + 1] = 0.25                       # home advantage prior
    x0[2 * n + 2] = 0.0                        # rho
    bounds = ([(None, None)] * (2 * n) + [(None, None), (-1.0, 1.0), (-rho_bound, rho_bound)]
              + [(-conf_bound, conf_bound)] * nc)

    # Numeric gradient burns ~2*n_params function evals per iteration, so the default
    # maxfun (15000) stops L-BFGS after a dozen iterations with many teams. Give it a
    # bounded-but-generous budget; output (lambdas/probabilities) is stable well before
    # the strict gtol criterion is met on the regularized ridge.
    res = minimize(neg_ll, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": maxiter,
                            "maxfun": 50 * maxiter * (2 * n + 3 + nc),
                            "ftol": 1e-10, "gtol": 1e-6})
    atk, dfn, c, gamma, rho, conf = unpack(res.x)
    conf_adj = {cf: float(v) for cf, v in zip(confs, conf)}
    log.info("dixon_coles_fit", n_teams=n, n_matches=len(df),
             converged=bool(res.success), iterations=int(res.nit),
             home_adv=round(float(gamma), 3), rho=round(float(rho), 3),
             confederation=bool(nc), inter_conf_matches=int(inter.sum()) if nc else 0,
             neg_ll=round(float(res.fun), 1))

    return DCModel(
        attack={k: float(v) for k, v in zip(keys, atk)},
        defense={k: float(v) for k, v in zip(keys, dfn)},
        intercept=float(c),
        home_adv=float(gamma),
        rho=float(rho),
        half_life_days=half_life_days,
        trained_through=as_of.isoformat(),
        n_matches=int(len(df)),
        max_goals=max_goals,
        conf_adj=conf_adj,
    )
