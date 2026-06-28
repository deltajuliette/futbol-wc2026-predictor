"""One-command daily refresh: pull latest results/fixtures -> features -> train ->
predict. Safe to run repeatedly (every step is idempotent / versioned).

During the tournament this keeps forecasts current: as matches finish, football-data
flips them to ``finished`` with scores, they feed the next training run, and the
remaining scheduled fixtures get fresh predictions.

Example::

    python -m scripts.update                      # pull WC, retrain, re-predict
    python -m scripts.update --skip-pull          # retrain/predict on current data only
    python -m scripts.update --backtest           # also refresh evaluation metrics
"""

from __future__ import annotations

import argparse
import time

from config.settings import settings
from features.build import build_features, write_features
from storage.dao import load_matches_df
from storage.database import get_engine, init_db
from utils.logging import get_logger

log = get_logger("update")


def run(wc_code: str = "WC", season: str | None = "2026",
        store_as: str = "world_cup_2026", half_life: float = 1095.0,
        skip_pull: bool = False, do_backtest: bool = False) -> None:
    t0 = time.monotonic()
    engine = init_db(get_engine())

    # 1) Pull latest fixtures/results (skipped if asked or no API key).
    if skip_pull:
        log.info("step_pull_skipped", reason="--skip-pull")
    elif not settings.football_data_api_key:
        log.warning("step_pull_skipped", reason="FOOTBALL_DATA_API_KEY not set; "
                    "using existing data")
    else:
        from scripts.etl.pull_fixtures import pull as pull_fixtures
        n = pull_fixtures(wc_code, season, store_as)
        log.info("step_pull_done", fixtures=n)

    # 2) Rebuild leak-safe features.
    feats = build_features(load_matches_df(engine))
    write_features(engine, feats)
    log.info("step_features_done", rows=len(feats))

    # 3) Retrain Dixon-Coles (new model run; prior runs are preserved).
    from scripts.modeling.train_dixon_coles import train as train_dc
    run_id = train_dc(half_life)
    log.info("step_train_done", model_run_id=run_id)

    # 4) Re-predict the scheduled fixtures.
    from scripts.modeling.predict import predict as predict_fn
    predict_fn(store_as)
    log.info("step_predict_done", competition=store_as)

    # 5) Optional: refresh evaluation metrics.
    if do_backtest:
        from scripts.evaluation.backtest import backtest
        backtest(folds=3, test_frac=0.25, half_life=half_life)

    log.info("update_complete", seconds=round(time.monotonic() - t0, 1),
             model_run_id=run_id)
    print(f"\n✅ Update complete in {time.monotonic() - t0:.0f}s — model run #{run_id}. "
          f"View: streamlit run app/dashboard/app.py")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wc-code", default="WC", help="football-data competition code")
    ap.add_argument("--season", default="2026")
    ap.add_argument("--store-as", default="world_cup_2026")
    ap.add_argument("--half-life", type=float, default=1095.0)
    ap.add_argument("--skip-pull", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    args = ap.parse_args()
    run(args.wc_code, args.season, args.store_as, args.half_life,
        args.skip_pull, args.backtest)


if __name__ == "__main__":
    main()
