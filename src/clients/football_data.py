"""football-data.org v4 client (primary fixtures/results source).

Paced + retried (honors HTTP 429 / Retry-After), caches raw JSON for reproducibility,
and parses into :class:`FixtureRecord`. The parser is a pure function with its own
tests, because the API payload shape is the fragile part. Requires
``FOOTBALL_DATA_API_KEY``; calling networked methods without it raises.

Example::

    client = FootballDataClient()
    fixtures = client.get_fixtures("WC", season="2026")
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from clients.types import FixtureRecord, Provenance
from config.settings import PROJECT_ROOT, settings
from utils.logging import get_logger

log = get_logger(__name__)

BASE_URL = "https://api.football-data.org/v4"
_STATUS_MAP = {
    "FINISHED": "finished",
    "IN_PLAY": "live",
    "PAUSED": "live",
}


class RateLimited(Exception):
    """Raised on HTTP 429 so tenacity can back off."""


def parse_matches(payload: dict, source_url: str, run_id: str) -> list[FixtureRecord]:
    """Pure parser: football-data v4 ``/matches`` payload -> FixtureRecords.

    Fails loudly on missing top-level ``matches`` (schema drift), per project rules.
    """
    if "matches" not in payload:
        raise ValueError("football-data payload missing 'matches' (schema drift?)")
    out: list[FixtureRecord] = []
    skipped_tbd = 0
    ingested = datetime.now(UTC)
    for m in payload["matches"]:
        # Knockout slots whose participants aren't decided yet have null team names
        # (e.g. "Winner Group A"). Skip — nothing to forecast for a TBD matchup.
        home_name = (m.get("homeTeam") or {}).get("name")
        away_name = (m.get("awayTeam") or {}).get("name")
        if not home_name or not away_name:
            skipped_tbd += 1
            continue
        ft = (m.get("score") or {}).get("fullTime") or {}
        status = _STATUS_MAP.get(m.get("status", ""), "scheduled")
        out.append(FixtureRecord(
            competition=(m.get("competition") or {}).get("code")
            or payload.get("competition", {}).get("code", "WC"),
            season=str((m.get("season") or {}).get("startDate", ""))[:4] or None,
            stage=m.get("stage"),
            kickoff_utc=datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00")),
            home_team=home_name,
            away_team=away_name,
            neutral=True,  # World Cup matches are at neutral venues
            status=status,
            home_goals=ft.get("home"),
            away_goals=ft.get("away"),
            provenance=Provenance(source="football_data", source_url=source_url,
                                  ingested_at=ingested, run_id=run_id),
        ))
    if skipped_tbd:
        log.info("parse_matches_skipped_tbd", skipped=skipped_tbd, kept=len(out))
    return out


class FootballDataClient:
    def __init__(self, api_key: str | None = None, min_interval: float | None = None):
        self.api_key = api_key or settings.football_data_api_key
        self.min_interval = (min_interval if min_interval is not None
                             else settings.http_min_interval_seconds)
        self._last_call = 0.0
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": settings.http_user_agent})
        if self.api_key:
            self._session.headers["X-Auth-Token"] = self.api_key

    def _pace(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call = time.monotonic()

    @retry(retry=retry_if_exception_type(RateLimited),
           wait=wait_exponential(multiplier=2, min=2, max=60),
           stop=stop_after_attempt(5), reraise=True)
    def _get(self, path: str, params: dict | None = None) -> dict:
        if not self.api_key:
            raise RuntimeError("FOOTBALL_DATA_API_KEY not set — cannot call the API")
        self._pace()
        resp = self._session.get(f"{BASE_URL}{path}", params=params, timeout=30)
        if resp.status_code == 429:
            wait = float(resp.headers.get("Retry-After", "10"))
            log.warning("rate_limited", path=path, retry_after=wait)
            time.sleep(wait)
            raise RateLimited(path)
        resp.raise_for_status()
        return resp.json()

    def _cache_raw(self, payload: dict, run_id: str, name: str) -> Path:
        d = PROJECT_ROOT / "data" / "raw" / "football_data" / run_id
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def get_fixtures(self, competition: str = "WC",
                     season: str | None = None) -> list[FixtureRecord]:
        run_id = uuid.uuid4().hex[:12]
        params = {"season": season} if season else None
        payload = self._get(f"/competitions/{competition}/matches", params=params)
        cache_path = self._cache_raw(payload, run_id, f"{competition}_matches")
        url = f"{BASE_URL}/competitions/{competition}/matches"
        records = parse_matches(payload, source_url=url, run_id=run_id)
        log.info("football_data_fixtures", competition=competition, n=len(records),
                 cache=str(cache_path.relative_to(PROJECT_ROOT)))
        return records
