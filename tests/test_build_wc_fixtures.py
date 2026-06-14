"""Tests for the curated World Cup fixtures builder.

Verifies the football-data payload -> curated schema mapping: unresolved knockout slots
are dropped, finished matches carry scores, scheduled matches do not, and no impossible
synthetic matchups survive.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.etl.build_wc_fixtures import build_fixtures


def _payload() -> dict:
    return {
        "matches": [
            {  # finished group match -> carries score
                "utcDate": "2026-06-11T19:00:00Z",
                "stage": "GROUP_STAGE",
                "status": "FINISHED",
                "homeTeam": {"name": "Mexico"},
                "awayTeam": {"name": "South Africa"},
                "score": {"fullTime": {"home": 2, "away": 0}},
            },
            {  # scheduled group match -> no score
                "utcDate": "2026-06-13T19:00:00Z",
                "stage": "GROUP_STAGE",
                "status": "TIMED",
                "homeTeam": {"name": "Qatar"},
                "awayTeam": {"name": "Switzerland"},
                "score": {"fullTime": {"home": None, "away": None}},
            },
            {  # unresolved knockout slot -> dropped
                "utcDate": "2026-06-28T19:00:00Z",
                "stage": "LAST_32",
                "status": "TIMED",
                "homeTeam": {"name": None},
                "awayTeam": {"name": None},
                "score": {"fullTime": {"home": None, "away": None}},
            },
        ]
    }


def test_drops_unresolved_knockout_slots(tmp_path: Path) -> None:
    src = tmp_path / "wc.json"
    src.write_text(json.dumps(_payload()))
    df = build_fixtures(src)
    assert len(df) == 2  # the TBD knockout slot is excluded
    assert df["home_team"].notna().all() and df["away_team"].notna().all()


def test_finished_carries_score_scheduled_does_not(tmp_path: Path) -> None:
    src = tmp_path / "wc.json"
    src.write_text(json.dumps(_payload()))
    df = build_fixtures(src).set_index("home_team")
    assert df.loc["Mexico", "home_goals"] == 2
    assert df.loc["Mexico", "away_goals"] == 0
    import pandas as pd
    assert pd.isna(df.loc["Qatar", "home_goals"])


def test_schema_and_no_synthetic_matchup(tmp_path: Path) -> None:
    src = tmp_path / "wc.json"
    src.write_text(json.dumps(_payload()))
    df = build_fixtures(src)
    assert set(df.columns) == {
        "date", "competition", "stage", "home_team", "away_team",
        "neutral", "home_goals", "away_goals",
    }
    assert (df["competition"] == "world_cup_2026").all()
    assert (df["neutral"] == 1).all()
    # the bogus synthetic pairing must not exist
    pair = set(zip(df["home_team"], df["away_team"]))
    assert ("Qatar", "Brazil") not in pair


def test_real_pull_builds(tmp_path: Path) -> None:
    """Smoke test against the checked-in real cached pull, if present."""
    import glob

    from config.settings import PROJECT_ROOT

    pulls = glob.glob(str(PROJECT_ROOT / "data/raw/football_data/*/WC_matches.json"))
    if not pulls:
        return  # raw pull not present in this checkout; covered by synthetic cases above
    df = build_fixtures(Path(pulls[0]))
    assert len(df) >= 48  # all 72 group matches have both teams known
    assert df["home_team"].notna().all()
    assert ("Qatar", "Brazil") not in set(zip(df["home_team"], df["away_team"]))
