"""Leak-safe feature builder → ``team_match_features``.

For every match, emit one row per team using ONLY information available before
kickoff: pre-match Elo, rolling goal rates over prior matches, rest days, and venue
flags. Each row records ``as_of_utc`` = kickoff (the cutoff) so reproducibility for a
fixed as-of date is auditable, and no post-match data can leak in.
"""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

from models.elo import run_elo
from utils.logging import get_logger
from utils.naming import team_key

log = get_logger(__name__)

FEATURE_SET_VERSION = "v1"


def build_features(matches: pd.DataFrame, form_window: int = 5) -> pd.DataFrame:
    """Return a tidy (match_id, team_id, ...) feature frame.

    ``matches`` must include match_id, home_team_id, away_team_id, home_name,
    away_name, home_goals, away_goals, neutral, kickoff_utc.
    """
    df = matches.sort_values("kickoff_utc").reset_index(drop=True)
    pre, _ = run_elo(df)
    elo = pre.set_index("match_id")

    # Long form: one row per (match, team) with that team's goals for/against.
    last_played: dict[str, pd.Timestamp] = {}
    gf_hist: dict[str, list[int]] = {}
    ga_hist: dict[str, list[int]] = {}
    rows = []
    for row in df.itertuples(index=False):
        mid = row.match_id
        ko = row.kickoff_utc
        for side, team_id, name, opp_goals_attr, goals_attr, is_home in (
            ("home", row.home_team_id, row.home_name, "away_goals", "home_goals", 1),
            ("away", row.away_team_id, row.away_name, "home_goals", "away_goals", 0),
        ):
            k = team_key(name)
            gf_list, ga_list = gf_hist.get(k, []), ga_hist.get(k, [])
            rest = (ko - last_played[k]).days if k in last_played else None
            elo_home = float(elo.loc[mid, "elo_home_pre"])
            elo_away = float(elo.loc[mid, "elo_away_pre"])
            elo_pre = elo_home if is_home else elo_away
            elo_diff = (elo_home - elo_away) if is_home else (elo_away - elo_home)
            rows.append({
                "match_id": mid,
                "team_id": team_id,
                "as_of_utc": ko.isoformat(),
                "elo_pre": elo_pre,
                "elo_diff": elo_diff,
                "rest_days": rest,
                "xg_for_form": None,
                "xg_against_form": None,
                "gf_rate": (sum(gf_list[-form_window:]) / len(gf_list[-form_window:]))
                if gf_list else None,
                "ga_rate": (sum(ga_list[-form_window:]) / len(ga_list[-form_window:]))
                if ga_list else None,
                "is_home": is_home,
                "neutral": int(getattr(row, "neutral", 0) or 0),
                "feature_set_version": FEATURE_SET_VERSION,
            })
        # AFTER emitting features, update history with this match's outcome.
        if pd.notna(row.home_goals) and pd.notna(row.away_goals):
            hk, ak = team_key(row.home_name), team_key(row.away_name)
            gh, ga = int(row.home_goals), int(row.away_goals)
            gf_hist.setdefault(hk, []).append(gh)
            ga_hist.setdefault(hk, []).append(ga)
            gf_hist.setdefault(ak, []).append(ga)
            ga_hist.setdefault(ak, []).append(gh)
            last_played[hk] = ko
            last_played[ak] = ko

    return pd.DataFrame(rows)


def write_features(engine: Engine, feats: pd.DataFrame) -> int:
    """Upsert features into ``team_match_features`` (PK match_id, team_id)."""
    cols = [
        "match_id", "team_id", "as_of_utc", "elo_pre", "elo_diff", "rest_days",
        "xg_for_form", "xg_against_form", "gf_rate", "ga_rate", "is_home",
        "neutral", "feature_set_version",
    ]
    placeholders = ", ".join(f":{c}" for c in cols)
    updates = ", ".join(f"{c}=excluded.{c}" for c in cols if c not in ("match_id", "team_id"))
    stmt = text(
        f"INSERT INTO team_match_features ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT (match_id, team_id) DO UPDATE SET {updates}"
    )
    records = feats[cols].to_dict("records")
    with engine.begin() as conn:
        conn.execute(stmt, records)
    log.info("features_written", rows=len(records), version=FEATURE_SET_VERSION)
    return len(records)
