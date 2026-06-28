"""Ablation: how much do recently-played tournament games move the forecasts?

Fits two Dixon-Coles models on identical settings and compares their forecasts for the
still-upcoming fixtures:

* **Model A (pre-tournament)** — trained on all finished matches *excluding* the recent
  tournament games (the knowledge state before those games were played).
* **Model B (current)** — trained on all finished matches *including* them (production).

For every upcoming fixture it reports the change in calibrated 1X2 probabilities and
expected goals, plus summary statistics (mean/median/max total-variation distance and the
number of fixtures whose most-likely outcome flips). This directly answers "do the recent
results change what we predict, and for whom?" without touching the production
``predictions`` table — output goes to a CSV under ``models/reports/``.

Each model carries its own out-of-sample-style calibrator fit on its own training slice,
mirroring :mod:`scripts.modeling.predict`, so the comparison reflects the full published
pipeline rather than only the raw goal model.

"Recent" defaults to finished matches whose stage names a World Cup and whose kickoff is
on/after ``--since`` (the current tournament). Adjust with ``--stage-like``/``--since``.

Example::

    python -m scripts.evaluation.recency_impact
    python -m scripts.evaluation.recency_impact --since 2026-06-01 --half-life 540
    python -m scripts.evaluation.recency_impact --competition world_cup_2026 --top 15
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from config.settings import PROJECT_ROOT
from models.calibration import ProbabilityCalibrator
from models.dixon_coles import fit_dixon_coles
from models.elo import AWAY, DRAW, HOME
from models.scoreline import probabilities
from storage.dao import load_matches_df
from storage.database import get_engine, init_db
from utils.logging import get_logger

log = get_logger(__name__)

_OUTCOME = {HOME: "home", DRAW: "draw", AWAY: "away"}


def _outcomes(df: pd.DataFrame) -> np.ndarray:
    gh = df["home_goals"].astype(int).to_numpy()
    ga = df["away_goals"].astype(int).to_numpy()
    return np.where(gh > ga, HOME, np.where(gh == ga, DRAW, AWAY))


def _recent_mask(finished: pd.DataFrame, stage_like: str, since: pd.Timestamp) -> pd.Series:
    """Finished games that count as 'recent': stage matches ``stage_like`` and on/after
    ``since``. Stage is matched case-insensitively on a substring."""
    stage = finished["stage"].fillna("").str.contains(stage_like, case=False)
    return stage & (finished["kickoff_utc"] >= since)


def _calibrated_probs(model, calib: ProbabilityCalibrator, df: pd.DataFrame) -> np.ndarray:
    raw = np.array([
        probabilities(*model.predict_lambdas(
            r.home_name, r.away_name, bool(r.neutral),
            home_conf=getattr(r, "home_conf", None),
            away_conf=getattr(r, "away_conf", None)), rho=model.rho).as_1x2()
        for r in df.itertuples(index=False)
    ])
    return calib.transform(raw) if len(raw) else raw


def _fit_with_calibrator(train: pd.DataFrame, half_life: float, min_matches: int):
    """Fit a DC model plus an in-sample calibrator on the same training slice."""
    model = fit_dixon_coles(train, half_life_days=half_life, min_matches=min_matches)
    raw = np.array([
        probabilities(*model.predict_lambdas(
            r.home_name, r.away_name, bool(r.neutral),
            home_conf=getattr(r, "home_conf", None),
            away_conf=getattr(r, "away_conf", None)), rho=model.rho).as_1x2()
        for r in train.itertuples(index=False)
    ])
    calib = ProbabilityCalibrator().fit(raw, _outcomes(train))
    return model, calib


def _exp_goals(model, df: pd.DataFrame) -> np.ndarray:
    out = []
    for r in df.itertuples(index=False):
        lam_h, lam_a = model.predict_lambdas(
            r.home_name, r.away_name, bool(r.neutral),
            home_conf=getattr(r, "home_conf", None),
            away_conf=getattr(r, "away_conf", None))
        out.append((lam_h, lam_a))
    return np.array(out) if out else np.zeros((0, 2))


def recency_impact(competition: str = "world_cup_2026", stage_like: str = "World Cup",
                   since: str = "2026-06-01", half_life: float = 540.0,
                   min_matches: int = 25, top: int = 10,
                   as_of: datetime | None = None) -> pd.DataFrame:
    engine = init_db(get_engine())
    finished = load_matches_df(engine, finished_only=True).reset_index(drop=True)
    if finished.empty:
        raise SystemExit("no finished matches — run the ETL first")

    since_ts = pd.Timestamp(since, tz="UTC")
    recent = _recent_mask(finished, stage_like, since_ts)
    n_recent = int(recent.sum())
    if n_recent == 0:
        raise SystemExit(f"no 'recent' games match stage~'{stage_like}' since {since}")

    # Upcoming fixtures to score: scheduled, still in the future.
    now = as_of or datetime.now(UTC)
    fixtures = load_matches_df(engine, competition=competition)
    fixtures = fixtures[(fixtures["status"] == "scheduled")
                        & (fixtures["kickoff_utc"] > pd.Timestamp(now))].reset_index(drop=True)
    if fixtures.empty:
        raise SystemExit(f"no upcoming scheduled fixtures for {competition}")

    # Model B includes the recent games; Model A excludes them. Same settings otherwise.
    model_b, calib_b = _fit_with_calibrator(finished, half_life, min_matches)
    model_a, calib_a = _fit_with_calibrator(finished[~recent], half_life, min_matches)

    pa = _calibrated_probs(model_a, calib_a, fixtures)
    pb = _calibrated_probs(model_b, calib_b, fixtures)
    ega = _exp_goals(model_a, fixtures)
    egb = _exp_goals(model_b, fixtures)

    tv = 0.5 * np.abs(pb - pa).sum(axis=1)   # total-variation distance per fixture
    flip = pa.argmax(axis=1) != pb.argmax(axis=1)

    rows = []
    for i, r in enumerate(fixtures.itertuples(index=False)):
        rows.append({
            "date": pd.Timestamp(r.kickoff_utc).date().isoformat(),
            "stage": r.stage, "home": r.home_name, "away": r.away_name,
            "pA_home": pa[i, 0], "pA_draw": pa[i, 1], "pA_away": pa[i, 2],
            "pB_home": pb[i, 0], "pB_draw": pb[i, 1], "pB_away": pb[i, 2],
            "d_home": pb[i, 0] - pa[i, 0], "d_draw": pb[i, 1] - pa[i, 1],
            "d_away": pb[i, 2] - pa[i, 2],
            "egA_home": ega[i, 0], "egA_away": ega[i, 1],
            "egB_home": egb[i, 0], "egB_away": egb[i, 1],
            "tv_distance": tv[i],
            "argmax_flip": bool(flip[i]),
            "favorite_A": _OUTCOME[int(pa[i].argmax())],
            "favorite_B": _OUTCOME[int(pb[i].argmax())],
        })
    out = pd.DataFrame(rows).sort_values("tv_distance", ascending=False).reset_index(drop=True)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    rep_dir = PROJECT_ROOT / "models" / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    csv_path = rep_dir / f"recency_impact_{stamp}.csv"
    out.to_csv(csv_path, index=False)

    _print_summary(out, n_recent, len(fixtures), half_life, top, csv_path)
    log.info("recency_impact_done", recent_games=n_recent, fixtures=len(fixtures),
             mean_tv=round(float(out["tv_distance"].mean()), 4),
             flips=int(out["argmax_flip"].sum()), csv=str(csv_path.relative_to(PROJECT_ROOT)))
    return out


def _print_summary(out: pd.DataFrame, n_recent: int, n_fix: int, half_life: float,
                   top: int, csv_path) -> None:
    tv = out["tv_distance"].to_numpy()
    print(f"\nRecency impact — {n_recent} recent games, {n_fix} upcoming fixtures, "
          f"half_life={half_life:g}d")
    print(f"  total-variation distance  mean={tv.mean():.4f}  median={np.median(tv):.4f}  "
          f"max={tv.max():.4f}")
    for c in ("d_home", "d_draw", "d_away"):
        v = out[c].to_numpy()
        print(f"  {c:<7} mean abs={np.abs(v).mean():.4f}  max abs={np.abs(v).max():.4f}")
    print(f"  most-likely-outcome flips: {int(out['argmax_flip'].sum())}/{n_fix}")
    cols = ["date", "home", "away", "d_home", "d_draw", "d_away", "tv_distance",
            "favorite_A", "favorite_B"]
    show = out.head(top)[cols].copy()
    for c in ("d_home", "d_draw", "d_away", "tv_distance"):
        show[c] = show[c].map(lambda x: f"{x:+.3f}")
    print(f"\nTop {min(top, len(out))} movers (B−A on calibrated probabilities):")
    print(show.to_string(index=False))
    print(f"\nFull table: {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--competition", default="world_cup_2026")
    ap.add_argument("--stage-like", default="World Cup",
                    help="substring identifying the recent tournament's stage label")
    ap.add_argument("--since", default="2026-06-01", help="recent games on/after this date")
    ap.add_argument("--half-life", type=float, default=540.0)
    ap.add_argument("--min-matches", type=int, default=25)
    ap.add_argument("--top", type=int, default=10, help="rows to print in the mover table")
    args = ap.parse_args()
    recency_impact(args.competition, args.stage_like, args.since, args.half_life,
                   args.min_matches, args.top)


if __name__ == "__main__":
    main()
