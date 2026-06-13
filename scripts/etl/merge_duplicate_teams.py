"""Merge duplicate team entities onto a single canonical team.

Some teams entered the database under two slugs because the historical results source
(martj42) and the fixtures source (football-data.org) spell them differently, e.g.
"Cape Verde" vs "Cape Verde Islands", "Czech Republic" vs "Czechia". Team resolution
keys on the normalized slug, so the variants became *separate* teams — and the World
Cup fixtures pointed at the rating-less variant, scoring those sides as league-average.

This repoints every reference (matches, team_match_features) from each duplicate slug
onto its canonical team, seeds a ``team_aliases`` row so future ingests resolve the
variant correctly (see :func:`storage.dao.upsert_team`), and deletes the now-orphaned
team row. Idempotent: re-running after a merge is a no-op.

Only well-established same-nation spelling variants are merged. Distinct nations that
merely look similar (e.g. "Congo" the Republic of the Congo vs "DR Congo") are NOT
merged.

Example::

    python -m scripts.etl.merge_duplicate_teams            # apply
    python -m scripts.etl.merge_duplicate_teams --dry-run  # report only
"""

from __future__ import annotations

import argparse

from sqlalchemy import text
from sqlalchemy.engine import Engine

from storage.database import get_engine, init_db
from utils.logging import get_logger

log = get_logger(__name__)

# variant_slug -> canonical_slug. The canonical side holds the long history (and the
# fitted ratings keyed by its slug); the variant is the rating-less duplicate.
MERGES: dict[str, str] = {
    "cape-verde-islands": "cape-verde",
    "congo-dr": "dr-congo",
    "czechia": "czech-republic",
    "bosnia-herzegovina": "bosnia-and-herzegovina",
}


def _team_id(conn, slug: str) -> int | None:
    row = conn.execute(
        text("SELECT team_id FROM teams WHERE team_key = :k"), {"k": slug}
    ).fetchone()
    return int(row[0]) if row else None


def merge_teams(engine: Engine, merges: dict[str, str], dry_run: bool = False) -> int:
    """Repoint duplicates onto canonical teams. Returns the number merged."""
    merged = 0
    for variant, canonical in merges.items():
        with engine.begin() as conn:
            vid = _team_id(conn, variant)
            cid = _team_id(conn, canonical)
            if cid is None:
                log.warning("merge_skip_no_canonical", variant=variant, canonical=canonical)
                continue
            if vid is None:
                # Already merged (variant gone) — ensure the alias still exists.
                if not dry_run:
                    conn.execute(
                        text("INSERT OR IGNORE INTO team_aliases (alias, team_id) "
                             "VALUES (:a, :t)"),
                        {"a": variant, "t": cid},
                    )
                continue
            n_home = conn.execute(
                text("SELECT COUNT(*) FROM matches WHERE home_team_id = :v"), {"v": vid}
            ).scalar()
            n_away = conn.execute(
                text("SELECT COUNT(*) FROM matches WHERE away_team_id = :v"), {"v": vid}
            ).scalar()
            n_feat = conn.execute(
                text("SELECT COUNT(*) FROM team_match_features WHERE team_id = :v"),
                {"v": vid},
            ).scalar()
            log.info("merge_plan", variant=variant, variant_id=vid, canonical=canonical,
                     canonical_id=cid, matches_home=n_home, matches_away=n_away,
                     features=n_feat, dry_run=dry_run)
            if dry_run:
                merged += 1
                continue
            # Repoint references, then seed alias, then drop the orphan team row.
            conn.execute(text("UPDATE matches SET home_team_id = :c WHERE home_team_id = :v"),
                         {"c": cid, "v": vid})
            conn.execute(text("UPDATE matches SET away_team_id = :c WHERE away_team_id = :v"),
                         {"c": cid, "v": vid})
            conn.execute(
                text("UPDATE team_match_features SET team_id = :c WHERE team_id = :v"),
                {"c": cid, "v": vid},
            )
            conn.execute(
                text("INSERT OR IGNORE INTO team_aliases (alias, team_id) VALUES (:a, :t)"),
                {"a": variant, "t": cid},
            )
            conn.execute(text("DELETE FROM teams WHERE team_id = :v"), {"v": vid})
            merged += 1
    log.info("merge_done", merged=merged, dry_run=dry_run)
    return merged


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="report planned merges only")
    args = ap.parse_args()
    engine = init_db(get_engine())
    merge_teams(engine, MERGES, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
