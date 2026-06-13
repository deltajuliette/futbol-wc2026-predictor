"""Score scheduled fixtures with the latest Dixon-Coles run + benchmarks.

Writes calibrated + raw predictions (Dixon-Coles) and benchmark probabilities
(Elo-only, and market de-vig where odds exist). Calibration is fit on the finished
matches' in-sample DC probabilities (rigorous out-of-fold calibration lives in the
backtest).

Example::

    python -m scripts.modeling.predict --competition world_cup_2026
"""

from __future__ import annotations

import argparse
import json
import uuid
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import text

from config.settings import PROJECT_ROOT
from models.calibration import ProbabilityCalibrator
from models.dixon_coles import DCModel
from models.elo import HOME, AWAY, DRAW, train_elo
from models.scoreline import probabilities
from storage.dao import load_matches_df, save_benchmarks, save_predictions
from storage.database import get_engine, init_db
from utils.logging import get_logger

log = get_logger(__name__)


def _latest_dc_run(engine):
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT model_run_id, artifact_path FROM model_runs "
                "WHERE model_name='dixon_coles' ORDER BY model_run_id DESC LIMIT 1"
            )
        ).fetchone()
    if not row:
        raise SystemExit("no dixon_coles model run — run train_dixon_coles first")
    return int(row[0]), row[1]


def _outcomes(df) -> np.ndarray:
    gh = df["home_goals"].astype(int).to_numpy()
    ga = df["away_goals"].astype(int).to_numpy()
    return np.where(gh > ga, HOME, np.where(gh == ga, DRAW, AWAY))


def predict(competition: str) -> None:
    engine = init_db(get_engine())
    model_run_id, art_path = _latest_dc_run(engine)
    model = DCModel.from_dict(json.loads((PROJECT_ROOT / art_path).read_text()))

    finished = load_matches_df(engine, finished_only=True)
    fixtures = load_matches_df(engine, competition=competition)
    fixtures = fixtures[fixtures["status"] == "scheduled"].reset_index(drop=True)
    if fixtures.empty:
        log.info("no_scheduled_fixtures", competition=competition)
        return

    # Fit calibration on finished matches' DC raw probabilities.
    raw_fin = np.array([
        probabilities(*model.predict_lambdas(r.home_name, r.away_name, bool(r.neutral)),
                      rho=model.rho).as_1x2()
        for r in finished.itertuples(index=False)
    ])
    calibrator = ProbabilityCalibrator().fit(raw_fin, _outcomes(finished))

    # Score fixtures.
    pred_rows, elo_rows = [], []
    elo_model = train_elo(finished)
    now = datetime.now(UTC).isoformat()
    raw_fix = []
    for r in fixtures.itertuples(index=False):
        lam_h, lam_a = model.predict_lambdas(r.home_name, r.away_name, bool(r.neutral))
        mp = probabilities(lam_h, lam_a, rho=model.rho)
        raw_fix.append(mp.as_1x2())
        pred_rows.append({
            "match_id": r.match_id,
            "p_home_raw": mp.p_home, "p_draw_raw": mp.p_draw, "p_away_raw": mp.p_away,
            "exp_goals_home": mp.exp_goals_home, "exp_goals_away": mp.exp_goals_away,
            "scoreline_json": json.dumps(mp.top_scorelines),
            "p_btts": mp.p_btts, "p_over25": mp.p_over25,
            "predicted_at_utc": now,
        })
        eh, ed, ea = elo_model.predict_1x2(r.home_name, r.away_name, bool(r.neutral))
        elo_rows.append({
            "match_id": r.match_id, "source": "elo_only", "method": "logistic",
            "p_home": eh, "p_draw": ed, "p_away": ea, "captured_at_utc": now,
        })

    cal = calibrator.transform(np.array(raw_fix))
    for row, c in zip(pred_rows, cal):
        row["p_home_cal"], row["p_draw_cal"], row["p_away_cal"] = (
            float(c[0]), float(c[1]), float(c[2]))

    run = uuid.uuid4().hex[:12]
    n_pred = save_predictions(engine, model_run_id, pred_rows)
    n_bench = save_benchmarks(engine, elo_rows, run_id=run)
    log.info("predictions_written", model_run_id=model_run_id, predictions=n_pred,
             benchmarks=n_bench, calibrated=calibrator.fitted)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--competition", default="world_cup_2026")
    args = ap.parse_args()
    predict(args.competition)


if __name__ == "__main__":
    main()
