"""Populate ``teams.confederation`` from the curated reference map.

The historical results source (martj42) carries no confederation, so this fills it
from ``data/reference/confederations.csv`` (team_key -> UEFA/CONMEBOL/CONCACAF/CAF/
AFC/OFC). Teams absent from the map — CONIFA sides, defunct entities, unaffiliated
territories — are intentionally left NULL and are excluded from any confederation
adjustment downstream. Idempotent: matches on the canonical ``team_key``.

Example::

    python -m scripts.etl.populate_confederations
    python -m scripts.etl.populate_confederations --csv data/reference/confederations.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

from config.settings import PROJECT_ROOT
from storage.database import get_engine, init_db
from utils.logging import get_logger

log = get_logger(__name__)

VALID = {"UEFA", "CONMEBOL", "CONCACAF", "CAF", "AFC", "OFC"}
DEFAULT_CSV = PROJECT_ROOT / "data" / "reference" / "confederations.csv"


def load_map(csv_path: Path) -> dict[str, str]:
    """Read team_key -> confederation, validating the confederation labels."""
    mapping: dict[str, str] = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            key, conf = row["team_key"].strip(), row["confederation"].strip()
            if conf not in VALID:
                raise ValueError(f"unknown confederation {conf!r} for {key!r}")
            mapping[key] = conf
    if not mapping:
        raise ValueError(f"no rows in {csv_path}")
    return mapping


def populate(engine: Engine, mapping: dict[str, str]) -> dict[str, int]:
    """Set confederation on matching teams. Returns coverage counters."""
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE teams SET confederation = :c WHERE team_key = :k"),
            [{"k": k, "c": c} for k, c in mapping.items()],
        )
        # Coverage among teams that actually appear in matches (the fit population).
        pool = conn.execute(text(
            "SELECT COUNT(DISTINCT t.team_id) FROM teams t "
            "JOIN matches m ON t.team_id IN (m.home_team_id, m.away_team_id)"
        )).scalar() or 0
        mapped = conn.execute(text(
            "SELECT COUNT(DISTINCT t.team_id) FROM teams t "
            "JOIN matches m ON t.team_id IN (m.home_team_id, m.away_team_id) "
            "WHERE t.confederation IS NOT NULL"
        )).scalar() or 0
    stats = {"in_map": len(mapping), "pool": int(pool), "mapped": int(mapped),
             "unmapped": int(pool) - int(mapped)}
    log.info("confederations_populated", **stats)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = ap.parse_args()
    engine = init_db(get_engine())
    populate(engine, load_map(args.csv))


if __name__ == "__main__":
    main()
