"""Pull fixtures/results from football-data.org into ``matches`` (idempotent).

Requires ``FOOTBALL_DATA_API_KEY`` in ``.env`` (free tier works — see the error
message / README if you don't have one). For a no-key path that lands real historical
internationals, use ``scripts.etl.pull_open_results`` instead.

Example::

    python -m scripts.etl.pull_fixtures --competition WC --season 2026
"""

from __future__ import annotations

import argparse
import uuid

from clients.football_data import FootballDataClient
from config.settings import settings
from storage.dao import upsert_match
from storage.database import get_engine, init_db
from utils.logging import get_logger

log = get_logger(__name__)

_NO_KEY_HELP = (
    "FOOTBALL_DATA_API_KEY is not set.\n"
    "  Option A (free, ~2 min): register at https://www.football-data.org/client/register, "
    "confirm your email, copy the token into .env as FOOTBALL_DATA_API_KEY. "
    "The free tier includes the World Cup (code WC).\n"
    "  Option B (no key): run `python -m scripts.etl.pull_open_results` to load real "
    "historical internationals from a public CSV, and add upcoming fixtures via a small "
    "CSV through `scripts.etl.load_intl_results`."
)


def pull(competition: str, season: str | None, target_comp: str) -> int:
    if not settings.football_data_api_key:
        raise SystemExit(_NO_KEY_HELP)
    engine = init_db(get_engine())
    client = FootballDataClient()
    fixtures = client.get_fixtures(competition=competition, season=season)
    run_id = uuid.uuid4().hex[:12]
    n = 0
    for rec in fixtures:
        # Normalize the stored competition label (e.g. world_cup_2026) for downstream joins.
        rec.competition = target_comp
        upsert_match(engine, rec, run_id=run_id)
        n += 1
    log.info("fixtures_pulled", source_competition=competition, stored_as=target_comp,
             rows=n, run_id=run_id)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--competition", default="WC", help="football-data competition code")
    ap.add_argument("--season", default=None, help="season start year, e.g. 2026")
    ap.add_argument("--store-as", default="world_cup_2026",
                    help="competition label to store under (downstream join key)")
    args = ap.parse_args()
    pull(args.competition, args.season, args.store_as)


if __name__ == "__main__":
    main()
