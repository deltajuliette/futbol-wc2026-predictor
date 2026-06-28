"""Sweep the recency levers (time-decay half-life x tournament weight) out-of-fold.

Rolling-origin, time-respecting backtest (same design as :mod:`scripts.evaluation.backtest`):
each fold trains strictly before its test window and the calibrator is fit on the training
slice only. For every (half_life, tournament_weight) combination it reports proper scores
twice:

* **all** held-out matches (overall calibration), and
* **tourn** held-out matches only (stage matches ``--stage-like``) — the subset the
  recency levers are meant to help, and the one relevant to "do recent games matter?".

Lower log loss is better. The winner is chosen on tournament log loss (ties broken by
overall log loss), since that is the recency-relevant objective; results are printed so the
trade-off against overall calibration is visible. Nothing is written to the database.

Example::

    python -m scripts.evaluation.recency_sweep
    python -m scripts.evaluation.recency_sweep --half-lives 365,540,730 --weights 1,4,8
    python -m scripts.evaluation.recency_sweep --folds 4 --stage-like "World Cup"
"""

from __future__ import annotations

import argparse

import numpy as np

from evaluation.metrics import score_all
from models.calibration import ProbabilityCalibrator
from models.dixon_coles import fit_dixon_coles
from models.elo import AWAY, DRAW, HOME
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


def sweep(half_lives: list[float], weights: list[float], folds: int = 3,
          test_frac: float = 0.3, min_matches: int = 25,
          stage_like: str = "World Cup") -> list[dict]:
    engine = init_db(get_engine())
    df = (load_matches_df(engine, finished_only=True)
          .sort_values("kickoff_utc").reset_index(drop=True))
    if len(df) < 200:
        raise SystemExit("not enough finished matches to sweep")

    n = len(df)
    fold_edges = np.linspace(int(n * (1 - test_frac)), n, folds + 1).astype(int)
    is_tourn_all = df["stage"].fillna("").str.contains(stage_like, case=False, regex=True)

    results: list[dict] = []
    for hl in half_lives:
        for tw in weights:
            cal_parts, out_parts, tourn_parts = [], [], []
            for i in range(folds):
                lo, hi = fold_edges[i], fold_edges[i + 1]
                if hi <= lo:
                    continue
                train, test = df.iloc[:lo], df.iloc[lo:hi]
                dc = fit_dixon_coles(train, half_life_days=hl, min_matches=min_matches,
                                     tournament_weight=tw, tournament_pattern=stage_like)
                calib = ProbabilityCalibrator().fit(_dc_probs(dc, train), _outcomes(train))
                cal_parts.append(calib.transform(_dc_probs(dc, test)))
                out_parts.append(_outcomes(test))
                tourn_parts.append(is_tourn_all.iloc[lo:hi].to_numpy())

            probs = np.concatenate(cal_parts)
            outs = np.concatenate(out_parts)
            tmask = np.concatenate(tourn_parts)
            s_all = score_all(probs, outs)
            row = {"half_life": hl, "tournament_weight": tw,
                   "n_all": s_all.n, "ll_all": s_all.log_loss,
                   "brier_all": s_all.brier, "rps_all": s_all.rps,
                   "n_tourn": int(tmask.sum())}
            if tmask.sum() >= 20:
                s_t = score_all(probs[tmask], outs[tmask])
                row.update({"ll_tourn": s_t.log_loss, "brier_tourn": s_t.brier,
                            "rps_tourn": s_t.rps})
            else:
                row.update({"ll_tourn": float("nan"), "brier_tourn": float("nan"),
                            "rps_tourn": float("nan")})
            results.append(row)
            log.info("sweep_combo", half_life=hl, tournament_weight=tw,
                     ll_all=round(s_all.log_loss, 4),
                     ll_tourn=round(row["ll_tourn"], 4), n_tourn=row["n_tourn"])

    _print(results)
    return results


def _print(results: list[dict]) -> None:
    base = next((r for r in results if r["tournament_weight"] == 1.0
                 and r["half_life"] == 540.0), results[0])
    print(f"\nRecency sweep — {results[0]['n_all']} held-out matches "
          f"({results[0]['n_tourn']} tournament). Lower log loss is better.")
    print(f"{'half_life':>9} {'tourn_w':>8} {'ll_all':>8} {'ll_tourn':>9} "
          f"{'brier_t':>8} {'rps_t':>7}")
    print("-" * 54)
    for r in sorted(results, key=lambda x: (x["half_life"], x["tournament_weight"])):
        print(f"{r['half_life']:>9g} {r['tournament_weight']:>8g} {r['ll_all']:>8.4f} "
              f"{r['ll_tourn']:>9.4f} {r['brier_tourn']:>8.4f} {r['rps_tourn']:>7.4f}")

    ranked = [r for r in results if not np.isnan(r["ll_tourn"])]
    if ranked:
        best = min(ranked, key=lambda r: (r["ll_tourn"], r["ll_all"]))
        d_t = base["ll_tourn"] - best["ll_tourn"]
        d_a = best["ll_all"] - base["ll_all"]
        print(f"\nbaseline (hl=540, tw=1): ll_tourn={base['ll_tourn']:.4f} "
              f"ll_all={base['ll_all']:.4f}")
        print(f"best on tournament ll : hl={best['half_life']:g} tw={best['tournament_weight']:g}"
              f"  ll_tourn={best['ll_tourn']:.4f} (-{d_t:.4f})  "
              f"ll_all={best['ll_all']:.4f} ({'+' if d_a >= 0 else ''}{d_a:.4f})")


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--half-lives", default="365,540,730", help="comma-separated days")
    ap.add_argument("--weights", default="1,4,8", help="comma-separated tournament weights")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--min-matches", type=int, default=25)
    ap.add_argument("--stage-like", default="World Cup")
    args = ap.parse_args()
    sweep(_floats(args.half_lives), _floats(args.weights), args.folds, args.test_frac,
          args.min_matches, args.stage_like)


if __name__ == "__main__":
    main()
