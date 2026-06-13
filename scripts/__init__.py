"""Script entrypoints package.

Ensures ``src/`` is importable for ``python -m scripts.*`` invocations regardless of
how (or whether) the project is installed, so the documented CLI commands always work.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
