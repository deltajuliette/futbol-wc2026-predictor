"""Out-of-sample evaluation on the games already played at this World Cup.

The live tournament games sit in the training corpus (stage ``FIFA World Cup``), so scoring
the production model on them would be in-sample and flattering. Instead we fit a
*pre-tournament* model — trained on everything *except* those games, with calibration fit
on the same pre-tournament slice — and score its forecasts against the realized results.
This is the honest "how good are the World Cup predictions?" test.

Reports proper scoring rules (log loss, Brier, RPS) for the calibrated and raw Dixon-Coles
model against an Elo-only benchmark and a uniform baseline, plus a one-vs-rest reliability
table and the favorite hit-rate (descriptive only — we optimize scores, not accuracy).

Example::

    python -m scripts.evaluation.wc_holdout
    python -m scripts.evaluation.wc_holdout --stage-like "World Cup" --since 2026-06-01
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from evaluation.metrics import reliability_table, score_all
from models.calibration import ProbabilityCalibrator
from models.dixon_coles import fit_dixon_coles
from models.elo import AWAY, DRAW, HOME, train_elo
from models.scoreline import probabilities
from storage.dao import load_matches_df
from storage.database import get_engine, init_db
from utils.logging import get_logger

log = get_logger(__name__)


def _outcomes(df) -> np.ndarray:
    gh = df["home_goals"].astype(int).to_numpy()
    ga = df["away_goals"].astype(int).to_numpy()
    return np.where(gh > ga, HOME, np.where(gh == ga, DRAW, AWAY))


def _dc_probs(model, df) -> np.ndarray:
    out = []
    for r in df.itertuples(index=False):
        lam_h, lam_a = model.predict_lambdas(
            r.home_name, r.away_name, bool(r.neutral),
            home_conf=getattr(r, "home_conf", None), away_conf=getattr(r, "away_conf", None))
        out.append(probabilities(lam_h, lam_a, rho=model.rho).as_1x2())
    return np.array(out)


def evaluate(stage_like: str = "World Cup", since: str = "2026-06-01",
             half_life: float = 1095.0, min_matches: int = 25) -> dict:
    engine = init_db(get_engine())
    df = load_matches_df(engine, finished_only=True).reset_index(drop=True)
    since_ts = pd.Timestamp(since, tz="UTC")
    is_wc = (df["stage"].fillna("").str.contains(stage_like, case=False, regex=True)
             & (df["kickoff_utc"] >= since_ts))
    test = df[is_wc].reset_index(drop=True)
    train = df[~is_wc].reset_index(drop=True)
    if test.empty:
        raise SystemExit(f"no played games match stage~'{stage_like}' since {since}")

    dc = fit_dixon_coles(train, half_life_days=half_life, min_matches=min_matches)
    elo = train_elo(train)
    calib = ProbabilityCalibrator().fit(_dc_probs(dc, train), _outcomes(train))

    outcomes = _outcomes(test)
    probs = {
        "dc_cal": calib.transform(_dc_probs(dc, test)),
        "dc_raw": _dc_probs(dc, test),
        "elo": np.array([elo.predict_1x2(r.home_name, r.away_name, bool(r.neutral))
                         for r in test.itertuples(index=False)]),
        "uniform": np.tile([1 / 3, 1 / 3, 1 / 3], (len(test), 1)),
    }

    print(f"\nWorld Cup out-of-sample evaluation — {len(test)} played games "
          f"(pre-tournament model, half_life={half_life:g}d)")
    print(f"{'model':<9} {'logloss':>8} {'brier':>7} {'rps':>7} {'sharp':>7} {'fav_acc':>8}")
    print("-" * 50)
    results = {}
    for name, p in probs.items():
        s = score_all(p, outcomes)
        fav_acc = float((p.argmax(axis=1) == outcomes).mean())
        results[name] = {"log_loss": s.log_loss, "brier": s.brier, "rps": s.rps,
                         "sharpness": s.sharpness, "fav_acc": fav_acc}
        print(f"{name:<9} {s.log_loss:>8.4f} {s.brier:>7.4f} {s.rps:>7.4f} "
              f"{s.sharpness:>7.4f} {fav_acc:>7.1%}")

    print("\nReliability (calibrated DC, one-vs-rest):")
    print(f"  {'pred range':<14} {'mean_pred':>9} {'obs_freq':>9} {'n':>5}")
    for b in reliability_table(probs["dc_cal"], outcomes, n_bins=5):
        print(f"  [{b.lo:.2f},{b.hi:.2f})    {b.mean_pred:>9.3f} {b.frac_obs:>9.3f} "
              f"{b.count:>5d}")

    base = {k: results["uniform"][k] for k in ("log_loss",)}
    log.info("wc_holdout_done", n=len(test),
             dc_cal_logloss=round(results["dc_cal"]["log_loss"], 4),
             elo_logloss=round(results["elo"]["log_loss"], 4),
             beats_uniform=bool(results["dc_cal"]["log_loss"] < base["log_loss"]),
             dc_cal_fav_acc=round(results["dc_cal"]["fav_acc"], 3))
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage-like", default="World Cup")
    ap.add_argument("--since", default="2026-06-01")
    ap.add_argument("--half-life", type=float, default=1095.0)
    ap.add_argument("--min-matches", type=int, default=25)
    args = ap.parse_args()
    evaluate(args.stage_like, args.since, args.half_life, args.min_matches)


if __name__ == "__main__":
    main()
