"""Derive the curated World Cup 2026 fixtures CSV from the cached football-data.org pull.

This replaces the old synthetic fixtures slate (random pairings among the strongest
synthetic teams, which produced impossible matchups like "Qatar vs Brazil"). The output
is the *real* tournament schedule, sourced from the immutable raw pull under
``data/raw/football_data/<hash>/WC_matches.json`` and written to a curated, checked-in
file that ``make_sample_data`` no longer overwrites.

Only matches with both teams known are emitted, so knockout slots that are still "TBD"
(``Winner Group A`` etc.) are skipped until the draw resolves. Finished matches carry
their real scores (status ``finished``); the rest are ``scheduled``. All World Cup
matches are marked ``neutral=1`` to match the existing modeling convention — host-nation
home advantage (USA/Canada/Mexico) is a known caveat, not modeled here.

Outputs (curated, checked in):
    data/reference/wc2026_fixtures.csv

Example::

    python -m scripts.etl.build_wc_fixtures
    python -m scripts.etl.build_wc_fixtures --out data/reference/wc2026_fixtures.csv
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd

from config.settings import PROJECT_ROOT
from utils.logging import get_logger

log = get_logger(__name__)

RAW_GLOB = "data/raw/football_data/*/WC_matches.json"
DEFAULT_OUT = PROJECT_ROOT / "data" / "reference" / "wc2026_fixtures.csv"

# football-data.org stage codes -> our normalized stage labels.
STAGE_MAP = {
    "GROUP_STAGE": "group",
    "LAST_32": "round_of_32",
    "LAST_16": "round_of_16",
    "QUARTER_FINALS": "quarter_final",
    "SEMI_FINALS": "semi_final",
    "THIRD_PLACE": "third_place",
    "FINAL": "final",
}


def _latest_pull(project_root: Path) -> Path:
    """Pick the cached WC pull whose matches were most recently updated."""
    paths = [Path(p) for p in glob.glob(str(project_root / RAW_GLOB))]
    if not paths:
        raise FileNotFoundError(f"no cached WC pull found under {RAW_GLOB}")

    def freshness(p: Path) -> str:
        matches = json.loads(p.read_text()).get("matches", [])
        return max((m.get("lastUpdated", "") for m in matches), default="")

    return max(paths, key=freshness)


def build_fixtures(src: Path) -> pd.DataFrame:
    """Map a football-data WC payload to the curated fixtures schema.

    Drops matches missing either team (unresolved knockout slots). Emits real scores
    for finished matches so they are not re-predicted as if unplayed.
    """
    payload = json.loads(src.read_text())
    rows: list[dict] = []
    for m in payload.get("matches", []):
        home = (m.get("homeTeam") or {}).get("name")
        away = (m.get("awayTeam") or {}).get("name")
        if not home or not away:
            continue  # unresolved knockout slot ("Winner Group A" etc.)
        stage = STAGE_MAP.get(m.get("stage", ""), (m.get("stage") or "").lower())
        finished = m.get("status") == "FINISHED"
        ft = (m.get("score") or {}).get("fullTime") or {}
        rows.append({
            "date": m["utcDate"],
            "competition": "world_cup_2026",
            "stage": stage,
            "home_team": home,
            "away_team": away,
            "neutral": 1,
            "home_goals": ft.get("home") if finished else None,
            "away_goals": ft.get("away") if finished else None,
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", help="WC_matches.json path; default: freshest cached pull")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output CSV path")
    args = ap.parse_args()

    src = Path(args.src) if args.src else _latest_pull(PROJECT_ROOT)
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    out = Path(args.out)
    if not out.is_absolute():
        out = PROJECT_ROOT / out

    df = build_fixtures(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    n_finished = int(df["home_goals"].notna().sum())
    log.info(
        "wc_fixtures_built",
        src=str(src.relative_to(PROJECT_ROOT)),
        out=str(out.relative_to(PROJECT_ROOT)),
        fixtures=len(df),
        finished=n_finished,
        scheduled=len(df) - n_finished,
    )


if __name__ == "__main__":
    main()
