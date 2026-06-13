"""No-key ETL: download real historical internationals and load into ``matches``.

Fetches the public martj42/international_results CSV (CC0), normalizes it, writes an
immutable raw snapshot, and loads it through the shared idempotent loader.

Example::

    python -m scripts.etl.pull_open_results --since 2010
    python -m scripts.etl.pull_open_results --since 2024 --limit 500   # quick sample
"""

from __future__ import annotations

import argparse
import uuid

from clients.open_results import RESULTS_URL, fetch_results, normalize_results
from config.settings import PROJECT_ROOT
from scripts.etl.load_intl_results import load_csv
from utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", type=int, default=2010, help="keep matches from this year on")
    ap.add_argument("--limit", type=int, default=None, help="cap rows (sampling/testing)")
    ap.add_argument("--url", default=RESULTS_URL)
    args = ap.parse_args()

    raw = fetch_results(args.url)
    norm = normalize_results(raw, since_year=args.since)
    if args.limit:
        norm = norm.tail(args.limit)

    run_id = uuid.uuid4().hex[:12]
    out_dir = PROJECT_ROOT / "data" / "raw" / "open_results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "results.csv"
    norm.to_csv(raw_path, index=False)
    log.info("open_results_normalized", rows=len(norm), since=args.since,
             raw=str(raw_path.relative_to(PROJECT_ROOT)))

    load_csv(raw_path)


if __name__ == "__main__":
    main()
