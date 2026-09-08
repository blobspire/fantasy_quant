#!/usr/bin/env bash
# Install the daily capture. Idempotent: re-running replaces the existing entry.
#
# 6am local, which is after ESPN's overnight waiver processing (~3-4am ET) and
# before you would look at anything.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LINE="0 6 * * * $REPO/scripts/daily.sh # fantasy_quant"

current="$(crontab -l 2>/dev/null | grep -v '# fantasy_quant' || true)"
printf '%s\n%s\n' "$current" "$LINE" | grep -v '^$' | crontab -

echo "installed:"
crontab -l | grep 'fantasy_quant'
echo
echo "log: $REPO/data/cron.log"
echo "test it now with: $REPO/scripts/daily.sh && tail -20 $REPO/data/cron.log"
