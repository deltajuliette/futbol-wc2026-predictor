"""Rolling-origin backtest: score Dixon-Coles (raw + calibrated) vs Elo and a
uniform baseline on held-out matches, writing rows to ``evaluation_metrics``.

Time-respecting only: each fold trains on matches strictly before the fold window and
scores the matches inside it. Calibration is fit on the training slice and applied
forward, so there is no test leakage. See docs/evaluation.md.

Example::

    python -m scripts.evaluation.backtest --folds 4 --test-frac 0.3 --half-life 540
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime

import numpy as np

from evaluation.metrics import reliability_table, score_all
from models.calibration import ProbabilityCalibrator
from models.dixon_coles import fit_dixon_coles
from models.elo import AWAY, DRAW, HOME, train_elo
from models.scoreline import probabilities
from sqlalchemy import text
from storage.dao import load_matches_df
from storage.database import get_engine, init_db
from utils.logging import get_logger

log = get_logger(__name__)


def _outcomes(df) -> np.ndarray:
    gh = df["home_goals"].astype(int).to_numpy()
    ga = df["away_goals"].astype(int).to_numpy()
    return np.where(gh > ga, HOME, np.where(gh == ga, DRAW, AWAY))


def _dc_probs(model, df) -> np.ndarray:
    has_conf = {"home_conf", "away_conf"} <= set(df.columns)
    out = []
    for r in df.itertuples(index=False):
        hc = getattr(r, "home_conf", None) if has_conf else None
        ac = getattr(r, "away_conf", None) if has_conf else None
        lam_h, lam_a = model.predict_lambdas(r.home_name, r.away_name, bool(r.neutral),
                                             home_conf=hc, away_conf=ac)
        out.append(probabilities(lam_h, lam_a, rho=model.rho).as_1x2())
    return np.array(out)


def backtest(folds: int = 4, test_frac: float = 0.3, half_life: float = 540.0,
             min_matches: int = 25) -> None:
    engine = init_db(get_engine())
    df = load_matches_df(engine, finished_only=True).sort_values("kickoff_utc").reset_index(drop=True)
    if len(df) < 200:
        raise SystemExit("not enough finished matches to backtest")

    n = len(df)
    test_start = int(n * (1 - test_frac))
    fold_edges = np.linspace(test_start, n, folds + 1).astype(int)

    acc: dict[str, list[np.ndarray]] = {
        k: [] for k in ("dc_raw", "dc_cal", "dc_cal_conf", "elo", "uniform")}
    acc_out: list[np.ndarray] = []

    for i in range(folds):
        lo, hi = fold_edges[i], fold_edges[i + 1]
        if hi <= lo:
            continue
        train = df.iloc[:lo]
        test = df.iloc[lo:hi]

        dc = fit_dixon_coles(train, half_life_days=half_life, min_matches=min_matches)
        elo = train_elo(train)
        # OOF calibration: fit on train DC probs, apply to test.
        calib = ProbabilityCalibrator().fit(_dc_probs(dc, train), _outcomes(train))

        dc_raw = _dc_probs(dc, test)
        acc["dc_raw"].append(dc_raw)
        acc["dc_cal"].append(calib.transform(dc_raw))

        # Confederation variant: same pipeline, with the cross-pool correction on.
        dc_c = fit_dixon_coles(train, half_life_days=half_life, use_confederation=True,
                               min_matches=min_matches)
        calib_c = ProbabilityCalibrator().fit(_dc_probs(dc_c, train), _outcomes(train))
        acc["dc_cal_conf"].append(calib_c.transform(_dc_probs(dc_c, test)))

        acc["elo"].append(np.array([
            elo.predict_1x2(r.home_name, r.away_name, bool(r.neutral))
            for r in test.itertuples(index=False)
        ]))
        acc["uniform"].append(np.tile([1 / 3, 1 / 3, 1 / 3], (len(test), 1)))
        acc_out.append(_outcomes(test))

    outcomes = np.concatenate(acc_out)
    as_of = df["kickoff_utc"].max().isoformat()
    run_stamp = datetime.now(UTC).isoformat()

    print(f"\nBacktest over {len(outcomes)} held-out matches ({folds} folds)")
    print(f"{'model':<10} {'logloss':>9} {'brier':>8} {'rps':>8} {'sharp':>8}")
    print("-" * 47)
    results = {}
    for name, parts in acc.items():
        probs = np.concatenate(parts)
        s = score_all(probs, outcomes)
        results[name] = s
        print(f"{name:<10} {s.log_loss:>9.4f} {s.brier:>8.4f} {s.rps:>8.4f} {s.sharpness:>8.4f}")
        rel = [asdict(b) for b in reliability_table(probs, outcomes, n_bins=10)]
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO evaluation_metrics (label, as_of_utc, n_matches, log_loss, "
                    "brier, rps, calibration_json, sharpness, notes) VALUES "
                    "(:l,:a,:n,:ll,:b,:r,:c,:s,:notes)"
                ),
                {"l": name, "a": as_of, "n": s.n, "ll": s.log_loss, "b": s.brier,
                 "r": s.rps, "c": json.dumps(rel), "s": s.sharpness,
                 "notes": f"rolling-origin folds={folds} half_life={half_life} run={run_stamp}"},
            )

    best = min(("dc_cal", "dc_cal_conf", "dc_raw"), key=lambda k: results[k].log_loss)
    verdict = "beats" if results[best].log_loss < results["elo"].log_loss else "does NOT beat"
    conf_delta = results["dc_cal_conf"].log_loss - results["dc_cal"].log_loss
    log.info("backtest_done", best_model=best, verdict=f"{best} {verdict} elo on log loss",
             confederation_logloss_delta=round(conf_delta, 5),
             confederation_helps=bool(conf_delta < 0), n=len(outcomes))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--half-life", type=float, default=540.0)
    ap.add_argument("--min-matches", type=int, default=25,
                    help="drop teams with fewer than N finished matches (0 = keep all)")
    args = ap.parse_args()
    backtest(args.folds, args.test_frac, args.half_life, min_matches=args.min_matches)


if __name__ == "__main__":
    main()
