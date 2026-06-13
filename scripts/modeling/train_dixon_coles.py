"""Fit the Dixon-Coles model on finished matches and register a model run.

Saves a JSON artifact under models/artifacts/<run>/ and inserts a model_runs row.

Example::

    python -m scripts.modeling.train_dixon_coles --half-life 365
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime

from config.settings import PROJECT_ROOT
from models.dixon_coles import fit_dixon_coles
from storage.dao import create_model_run, load_matches_df
from storage.database import get_engine, init_db
from utils.logging import get_logger

log = get_logger(__name__)


def train(half_life_days: float = 365.0, use_confederation: bool = False) -> int:
    engine = init_db(get_engine())
    matches = load_matches_df(engine, finished_only=True)
    if matches.empty:
        raise SystemExit("no finished matches — run the ETL first")
    model = fit_dixon_coles(matches, half_life_days=half_life_days,
                            use_confederation=use_confederation)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    art_dir = PROJECT_ROOT / "models" / "artifacts" / f"dixon_coles_{stamp}"
    art_dir.mkdir(parents=True, exist_ok=True)
    art_path = art_dir / "model.json"
    art_path.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")

    window = f"{matches['kickoff_utc'].min().date()}..{matches['kickoff_utc'].max().date()}"
    run_id = create_model_run(
        engine,
        model_name="dixon_coles",
        training_window=window,
        params_json=json.dumps({"half_life_days": half_life_days,
                                "home_adv": model.home_adv, "rho": model.rho,
                                "use_confederation": use_confederation,
                                "conf_adj": model.conf_adj}),
        feature_set_version="dc-strengths",
        artifact_path=str(art_path.relative_to(PROJECT_ROOT)),
    )
    log.info("model_run_created", model_run_id=run_id, n_matches=model.n_matches,
             artifact=str(art_path.relative_to(PROJECT_ROOT)))
    return run_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--half-life", type=float, default=365.0, help="time-decay half life (days)")
    ap.add_argument("--confederation", action="store_true",
                    help="fit the cross-confederation relative-strength correction")
    args = ap.parse_args()
    train(args.half_life, use_confederation=args.confederation)


if __name__ == "__main__":
    main()
