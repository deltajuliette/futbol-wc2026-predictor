"""Load a results/fixtures CSV into the ``matches`` table (idempotent).

Accepts the schema produced by ``make_sample_data`` (and the historical
international-results dataset): columns ``date, competition, home_team, away_team``
plus optional ``home_goals, away_goals, neutral, stage``. Rows with goals are marked
``finished``; rows without are ``scheduled``.

Example::

    python -m scripts.etl.load_intl_results --path data/raw/intl_results/results.csv
    python -m scripts.etl.load_intl_results --path data/raw/intl_results/upcoming_wc.csv
"""

from __future__ import annotations

import argparse
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from clients.types import FixtureRecord, Provenance
from config.settings import PROJECT_ROOT
from storage.database import get_engine, init_db
from storage.dao import bulk_upsert_matches
from utils.logging import get_logger

log = get_logger(__name__)


def _row_to_record(row: pd.Series, source_url: str, run_id: str) -> FixtureRecord:
    has_score = pd.notna(row.get("home_goals")) and pd.notna(row.get("away_goals"))
    return FixtureRecord(
        competition=str(row["competition"]),
        stage=str(row["stage"]) if pd.notna(row.get("stage")) else None,
        kickoff_utc=pd.to_datetime(row["date"], utc=True).to_pydatetime(),
        home_team=str(row["home_team"]),
        away_team=str(row["away_team"]),
        neutral=bool(int(row.get("neutral", 0) or 0)),
        status="finished" if has_score else "scheduled",
        home_goals=int(row["home_goals"]) if has_score else None,
        away_goals=int(row["away_goals"]) if has_score else None,
        provenance=Provenance(
            source="sample_intl" if "intl_results" in source_url else "csv",
            source_url=source_url,
            ingested_at=datetime.now(UTC),
            run_id=run_id,
        ),
    )


def load_csv(path: Path) -> int:
    engine = init_db(get_engine())
    df = pd.read_csv(path)
    required = {"date", "competition", "home_team", "away_team"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV {path} missing required columns: {sorted(missing)}")
    run_id = uuid.uuid4().hex[:12]
    recs = [_row_to_record(row, str(path), run_id) for _, row in df.iterrows()]
    n = bulk_upsert_matches(engine, recs, run_id=run_id)
    log.info("loaded_matches", path=str(path), rows=n, run_id=run_id)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--path", required=True, help="CSV path (absolute or repo-relative)")
    args = ap.parse_args()
    path = Path(args.path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    load_csv(path)


if __name__ == "__main__":
    main()
