#!/usr/bin/env bash
# Daily capture + health check. Install with `scripts/install-cron.sh`.
#
# The snapshot is the time-sensitive half: percent rostered, ADP drift and injury
# designations are overwritten in place by ESPN, so a day not captured is gone
# permanently. `fq doctor` runs after it and exits non-zero when something needs
# a human -- notably an expired espn_s2, which otherwise fails silently.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 1
LOG="$REPO/data/cron.log"
mkdir -p "$(dirname "$LOG")"

UV="$(command -v uv || echo /opt/homebrew/bin/uv)"

{
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  "$UV" run fq snapshot || echo "SNAPSHOT FAILED"
  "$UV" run fq doctor || echo "DOCTOR REPORTED PROBLEMS"
} >> "$LOG" 2>&1

# Keep the log from growing without bound.
tail -n 5000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
