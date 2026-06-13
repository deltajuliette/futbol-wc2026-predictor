"""No-key adapter for public international-results data (CC0).

Source: martj42/international_results — every men's international since 1872 as a raw
CSV on GitHub (public domain, no API key). Used to bootstrap Elo/Dixon-Coles strengths
when no licensed feed is available. The normalizer is a pure, tested function; the
fetch is a thin network wrapper.

CSV columns: date, home_team, away_team, home_score, away_score, tournament, city,
country, neutral.
"""

from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

from config.settings import settings
from utils.logging import get_logger

log = get_logger(__name__)

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)


def normalize_results(df: pd.DataFrame, since_year: int | None = None) -> pd.DataFrame:
    """Map the public CSV to the loader schema; keep only finished matches.

    Output columns: date, competition, stage, home_team, away_team, home_goals,
    away_goals, neutral. ``competition`` is set to ``international``; the original
    ``tournament`` is preserved in ``stage`` for provenance.
    """
    required = {"date", "home_team", "away_team", "home_score", "away_score",
                "tournament", "neutral"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"open-results CSV missing columns: {sorted(missing)} (schema drift?)")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True)
    out = out.dropna(subset=["date", "home_score", "away_score"])
    if since_year is not None:
        out = out[out["date"].dt.year >= since_year]
    result = pd.DataFrame({
        "date": out["date"].dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "competition": "international",
        "stage": out["tournament"].astype(str),
        "home_team": out["home_team"].astype(str),
        "away_team": out["away_team"].astype(str),
        "home_goals": out["home_score"].astype(int),
        "away_goals": out["away_score"].astype(int),
        "neutral": out["neutral"].map(
            lambda v: 1 if str(v).strip().lower() in {"true", "1"} else 0
        ),
    })
    return result.reset_index(drop=True)


def fetch_results(url: str = RESULTS_URL, timeout: int = 60) -> pd.DataFrame:
    """Download the raw results CSV into a DataFrame."""
    resp = requests.get(url, headers={"User-Agent": settings.http_user_agent}, timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    log.info("open_results_fetched", url=url, rows=len(df))
    return df
