"""Generate a deterministic synthetic international-results dataset for offline runs.

The data-generating process is an independent-Poisson goal model with per-team latent
attack/defense strengths (slow drift over time) and a home advantage applied only at
non-neutral venues. Because we control the DGP, downstream calibration/backtests are
meaningful even without live data. Replace with real ingestion once an API key exists.

Outputs (immutable raw):
    data/raw/intl_results/results.csv       finished historical matches
    data/raw/intl_results/upcoming_wc.csv   scheduled neutral-venue fixtures to predict

Example::

    python -m scripts.etl.make_sample_data --teams 48 --years 12 --seed 7
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import PROJECT_ROOT
from utils.logging import get_logger

log = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "data" / "raw" / "intl_results"

# A pool of national-team names (display form). Trimmed to requested count.
TEAM_POOL = [
    "Brazil", "Argentina", "France", "England", "Spain", "Germany", "Portugal",
    "Netherlands", "Belgium", "Italy", "Croatia", "Uruguay", "Colombia", "Mexico",
    "United States", "Japan", "South Korea", "Senegal", "Morocco", "Nigeria",
    "Ghana", "Cameroon", "Ivory Coast", "Switzerland", "Denmark", "Sweden",
    "Poland", "Serbia", "Austria", "Wales", "Ecuador", "Peru", "Chile", "Canada",
    "Australia", "Iran", "Saudi Arabia", "Qatar", "Tunisia", "Egypt", "Algeria",
    "Costa Rica", "Paraguay", "Turkey", "Ukraine", "Norway", "Greece", "Scotland",
]


def _simulate(n_teams: int, years: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    teams = TEAM_POOL[:n_teams]
    n = len(teams)

    # Latent strengths (log scale). Attack positive = scores more; defense positive = concedes less.
    attack = rng.normal(0.0, 0.35, n)
    defense = rng.normal(0.0, 0.35, n)
    mu = 0.15          # baseline log scoring rate
    home_adv = 0.25    # log home advantage (non-neutral only)
    drift = 0.02       # per-matchday random-walk sd

    start = datetime(2026, 6, 1, tzinfo=UTC) - timedelta(days=int(years * 365.25))
    rows: list[dict] = []
    matchdays = years * 10  # ~10 international windows per year
    days_between = int((years * 365.25) / matchdays)

    for md in range(matchdays):
        date = start + timedelta(days=md * days_between)
        # Slow drift in strengths.
        attack += rng.normal(0, drift, n)
        defense += rng.normal(0, drift, n)
        # Random pairing of ~n/2 matches this window.
        order = rng.permutation(n)
        for i in range(0, n - 1, 2):
            h, a = int(order[i]), int(order[i + 1])
            neutral = rng.random() < 0.25
            lam_h = np.exp(mu + attack[h] - defense[a] + (0.0 if neutral else home_adv))
            lam_a = np.exp(mu + attack[a] - defense[h])
            gh, ga = int(rng.poisson(lam_h)), int(rng.poisson(lam_a))
            rows.append({
                "date": date.isoformat(),
                "competition": "international",
                "home_team": teams[h],
                "away_team": teams[a],
                "home_goals": gh,
                "away_goals": ga,
                "neutral": int(neutral),
            })

    results = pd.DataFrame(rows)

    # Upcoming World Cup: a neutral-venue round-robin sample among the 24 strongest
    # teams by current latent strength, dated just after the history ends.
    strength = attack - defense
    top = [teams[i] for i in np.argsort(strength)[::-1][:24]]
    kickoff0 = datetime(2026, 6, 11, 18, 0, tzinfo=UTC)
    fixtures = []
    # Pair them into a manageable slate (12 matches) without repeats.
    pairs = list(combinations(range(len(top)), 2))
    rng.shuffle(pairs)
    used: set[int] = set()
    slot = 0
    for i, j in pairs:
        if i in used or j in used:
            continue
        used.update({i, j})
        fixtures.append({
            "date": (kickoff0 + timedelta(days=slot // 4, hours=3 * (slot % 4))).isoformat(),
            "competition": "world_cup_2026",
            "stage": "group",
            "home_team": top[i],
            "away_team": top[j],
            "neutral": 1,
        })
        slot += 1
        if len(used) >= len(top):
            break

    return results, pd.DataFrame(fixtures)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--teams", type=int, default=48)
    ap.add_argument("--years", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results, fixtures = _simulate(min(args.teams, len(TEAM_POOL)), args.years, args.seed)
    results.to_csv(OUT_DIR / "results.csv", index=False)
    fixtures.to_csv(OUT_DIR / "upcoming_wc.csv", index=False)
    log.info(
        "sample_data_written",
        results=len(results),
        fixtures=len(fixtures),
        out=str(OUT_DIR.relative_to(PROJECT_ROOT)),
    )


if __name__ == "__main__":
    main()
