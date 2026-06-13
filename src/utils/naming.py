"""Team-name normalization → stable ``team_key`` slugs.

Keeps all name canonicalization in one place so joins across sources are reliable.

Example::

    from utils.naming import team_key
    team_key("Côte d'Ivoire")  # -> "cote-divoire"
"""

from __future__ import annotations

import re
import unicodedata


def team_key(name: str) -> str:
    """Return a normalized, ASCII, hyphenated slug for a team name."""
    if not name or not name.strip():
        raise ValueError("team name must be non-empty")
    # Strip accents → ASCII.
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_name = decomposed.encode("ascii", "ignore").decode("ascii")
    ascii_name = ascii_name.lower().strip()
    # Drop apostrophes entirely, replace any other non-alnum run with a hyphen.
    ascii_name = ascii_name.replace("'", "")
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name).strip("-")
    if not slug:
        raise ValueError(f"name {name!r} normalized to empty slug")
    return slug
