"""Typed application configuration loaded from environment / ``.env``.

All runtime configuration funnels through :data:`settings` so nothing is hardcoded
in pipeline code. Secrets (API keys) live only in ``.env`` (gitignored).

Example::

    from config.settings import settings
    print(settings.database_url)
    print(settings.football_data_api_key is not None)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root = three levels up from this file (src/config/settings.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Process-wide configuration. Field names map to UPPER_CASE env vars."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Data sources ---
    football_data_api_key: str | None = Field(default=None)

    # --- Storage ---
    database_url: str = Field(default="sqlite:///db/worldcup.sqlite")

    # --- Logging ---
    log_level: str = Field(default="INFO")

    # --- Polite scraping ---
    http_user_agent: str = Field(
        default="worldcup-forecast/0.1 (research)"
    )
    http_min_interval_seconds: float = Field(default=6.0)

    @property
    def sqlite_path(self) -> Path:
        """Absolute path to the SQLite file, resolved against the repo root."""
        if not self.database_url.startswith("sqlite"):
            raise ValueError("sqlite_path is only valid for a sqlite DATABASE_URL")
        # sqlite:///relative/path  or  sqlite:////absolute/path
        raw = self.database_url.split("sqlite:///", 1)[-1]
        p = Path(raw)
        return p if p.is_absolute() else (PROJECT_ROOT / p)


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


settings = get_settings()
