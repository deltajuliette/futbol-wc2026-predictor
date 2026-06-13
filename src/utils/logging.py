"""Structured logging setup. Use this instead of ``print`` everywhere.

Example::

    from utils.logging import get_logger
    log = get_logger(__name__)
    log.info("ingested", source="football_data", rows=42, run_id=run_id)
"""

from __future__ import annotations

import logging

import structlog

from config.settings import settings

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    """Idempotently configure structlog + stdlib logging for the process."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = getattr(logging, (level or settings.log_level).upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", level=log_level)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a configured structured logger."""
    configure_logging()
    return structlog.get_logger(name)
