"""Bootstrap the SQLite database (idempotent).

Example::

    python -m scripts.etl.init_db
"""

from __future__ import annotations

from storage.database import init_db, list_tables
from utils.logging import get_logger

log = get_logger(__name__)


def main() -> None:
    engine = init_db()
    log.info("schema_ready", tables=list_tables(engine))


if __name__ == "__main__":
    main()
