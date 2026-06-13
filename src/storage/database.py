"""Database engine + schema bootstrap.

SQLite-first via SQLAlchemy, with ``PRAGMA foreign_keys=ON`` enforced on every
connection. SQL is kept portable so a later move to Postgres only swaps the URL.

Example::

    from storage.database import get_engine, init_db
    init_db()                       # idempotent: creates db/worldcup.sqlite
    eng = get_engine()
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine

from config.settings import settings
from utils.logging import get_logger

log = get_logger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def _enable_sqlite_fk(engine: Engine) -> None:
    """Ensure foreign keys are enforced (off by default in SQLite)."""

    @event.listens_for(engine, "connect")
    def _set_pragma(dbapi_conn, _record):  # noqa: ANN001
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()


def get_engine(database_url: str | None = None) -> Engine:
    """Create a SQLAlchemy engine for the configured (or given) database URL."""
    url = database_url or settings.database_url
    if url.startswith("sqlite") and "memory" not in url:
        settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, future=True)
    if url.startswith("sqlite"):
        _enable_sqlite_fk(engine)
    return engine


def init_db(engine: Engine | None = None) -> Engine:
    """Apply the schema DDL idempotently. Returns the engine used."""
    engine = engine or get_engine()
    ddl = SCHEMA_PATH.read_text(encoding="utf-8")
    if engine.dialect.name == "sqlite":
        # executescript handles comments + multiple statements (incl. ';' in comments).
        raw = engine.raw_connection()
        try:
            raw.executescript(ddl)
            raw.commit()
        finally:
            raw.close()
    else:
        # Portable fallback: strip line comments, then split on ';'.
        cleaned = "\n".join(
            line for line in ddl.splitlines() if not line.lstrip().startswith("--")
        )
        with engine.begin() as conn:
            for stmt in (s.strip() for s in cleaned.split(";")):
                if stmt:
                    conn.execute(text(stmt))
    _apply_migrations(engine)
    log.info("db_initialized", url=settings.database_url, tables=len(list_tables(engine)))
    return engine


def _apply_migrations(engine: Engine) -> None:
    """Additive, idempotent schema upgrades for already-created databases.

    ``CREATE TABLE IF NOT EXISTS`` never alters existing tables, so columns added
    after a DB was first built must be patched in here. Each step is a no-op when
    the column already exists.
    """
    _add_column_if_missing(engine, "predictions", "reasoning_json", "TEXT")


def _add_column_if_missing(engine: Engine, table: str, column: str, decl: str) -> None:
    with engine.begin() as conn:
        cols = {r[1] for r in conn.execute(text(f"PRAGMA table_info({table})")).fetchall()}
        if column not in cols:
            conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {column} {decl}"))
            log.info("schema_migrated", table=table, added_column=column)


def list_tables(engine: Engine | None = None) -> list[str]:
    """Return user table names (SQLite)."""
    engine = engine or get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        ).fetchall()
    return [r[0] for r in rows]
