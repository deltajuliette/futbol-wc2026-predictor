"""Build leak-safe team-match features into ``team_match_features``.

Example::

    python -m scripts.features.build_features
"""

from __future__ import annotations

import argparse

from features.build import build_features, write_features
from storage.dao import load_matches_df
from storage.database import get_engine, init_db
from utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--form-window", type=int, default=5)
    args = ap.parse_args()
    engine = init_db(get_engine())
    matches = load_matches_df(engine)
    if matches.empty:
        raise SystemExit("no matches — run the ETL first")
    feats = build_features(matches, form_window=args.form_window)
    n = write_features(engine, feats)
    log.info("features_done", rows=n)


if __name__ == "__main__":
    main()
