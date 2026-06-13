"""Typed source records exchanged between adapters and the ETL layer.

These are the stable contract: downstream code depends on these shapes, never on a
specific source's raw payload. Every record can carry provenance.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Provenance(BaseModel):
    source: str
    source_url: str | None = None
    ingested_at: datetime | None = None
    run_id: str | None = None


class FixtureRecord(BaseModel):
    """A scheduled or finished match from any fixture/result source."""

    competition: str
    season: str | None = None
    stage: str | None = None
    kickoff_utc: datetime
    kickoff_local: datetime | None = None
    kickoff_tz: str | None = None
    home_team: str
    away_team: str
    neutral: bool = False
    status: str = "scheduled"
    home_goals: int | None = None
    away_goals: int | None = None
    home_goals_et: int | None = None
    away_goals_et: int | None = None
    pens_home: int | None = None
    pens_away: int | None = None
    provenance: Provenance | None = None


class OddsSnapshotRecord(BaseModel):
    """A single observed 1X2 (or other market) price."""

    competition: str
    kickoff_utc: datetime
    home_team: str
    away_team: str
    captured_at_utc: datetime
    bookmaker: str | None = None
    market: str = "1x2"
    home_odds: float | None = None
    draw_odds: float | None = None
    away_odds: float | None = None
    provenance: Provenance | None = None


class BenchmarkProbRecord(BaseModel):
    """External benchmark probabilities (e.g. public Opta)."""

    competition: str
    kickoff_utc: datetime
    home_team: str
    away_team: str
    source: str
    p_home: float = Field(ge=0, le=1)
    p_draw: float = Field(ge=0, le=1)
    p_away: float = Field(ge=0, le=1)
    captured_at_utc: datetime | None = None
    provenance: Provenance | None = None
