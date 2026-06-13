"""Parser tests for football-data.org payloads (the fragile part)."""

from __future__ import annotations

import pytest

from clients.football_data import parse_matches

SAMPLE = {
    "competition": {"code": "WC"},
    "matches": [
        {
            "utcDate": "2026-06-11T16:00:00Z",
            "status": "FINISHED",
            "stage": "GROUP_STAGE",
            "season": {"startDate": "2026-06-11"},
            "homeTeam": {"name": "Brazil"},
            "awayTeam": {"name": "Croatia"},
            "score": {"fullTime": {"home": 2, "away": 1}},
        },
        {
            "utcDate": "2026-06-12T19:00:00Z",
            "status": "SCHEDULED",
            "stage": "GROUP_STAGE",
            "season": {"startDate": "2026-06-11"},
            "homeTeam": {"name": "Argentina"},
            "awayTeam": {"name": "Mexico"},
            "score": {"fullTime": {"home": None, "away": None}},
        },
    ],
}


def test_parse_finished_and_scheduled():
    recs = parse_matches(SAMPLE, source_url="http://x", run_id="r1")
    assert len(recs) == 2
    finished, scheduled = recs
    assert finished.status == "finished"
    assert finished.home_team == "Brazil" and finished.home_goals == 2
    assert finished.neutral is True
    assert finished.season == "2026"
    assert scheduled.status == "scheduled"
    assert scheduled.home_goals is None
    assert recs[0].provenance.source == "football_data"


def test_schema_drift_raises_loudly():
    with pytest.raises(ValueError):
        parse_matches({"unexpected": []}, source_url="x", run_id="r")


def test_undecided_knockout_slots_skipped():
    # Real football-data knockout fixtures use null team names until decided.
    payload = {
        "competition": {"code": "WC"},
        "matches": [
            {"utcDate": "2026-07-05T19:00:00Z", "status": "SCHEDULED", "stage": "QUARTER_FINALS",
             "season": {"startDate": "2026-06-11"},
             "homeTeam": {"name": None}, "awayTeam": {"name": None},
             "score": {"fullTime": {"home": None, "away": None}}},
            SAMPLE["matches"][0],
        ],
    }
    recs = parse_matches(payload, source_url="x", run_id="r")
    assert len(recs) == 1  # the TBD match is dropped
    assert recs[0].home_team == "Brazil"


def test_kickoff_parsed_as_utc():
    recs = parse_matches(SAMPLE, source_url="x", run_id="r")
    assert recs[0].kickoff_utc.tzinfo is not None
    assert recs[0].kickoff_utc.isoformat().startswith("2026-06-11T16:00:00")
