#!/usr/bin/env bash
# Daily World Cup forecast refresh. Called by cron/launchd (or run by hand).
# Logs to data/update.log. Self-contained: uses absolute paths so it works
# under cron's minimal environment.
set -euo pipefail
PROJECT="/Users/djc/Desktop/code/futbol"
cd "$PROJECT"
mkdir -p data
echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') : starting update =====" >> data/update.log
"$PROJECT/.venv/bin/python" -m scripts.update >> data/update.log 2>&1
echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') : done =====" >> data/update.log
