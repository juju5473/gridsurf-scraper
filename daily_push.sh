#!/usr/bin/env bash
# daily_push.sh — regenerate STATS.md and push to GitHub
# Run via cron: 0 9 * * * /path/to/daily_push.sh

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
DB="$REPO_DIR/data/provider_snapshots.db"
STATS_FILE="$REPO_DIR/STATS.md"
LOG="$REPO_DIR/data/daily_push.log"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

cd "$REPO_DIR"

# ── Generate STATS.md from live DB ────────────────────────────────────────

log "Generating STATS.md..."
PYTHON=$(command -v /opt/homebrew/bin/python3 || command -v python3)
"$PYTHON" "$REPO_DIR/generate_stats.py"

# ── Commit and push if anything changed ───────────────────────────────────

git pull --quiet origin main 2>/dev/null || true

git add STATS.md

if git diff --cached --quiet; then
  log "No changes to STATS.md — nothing to push."
  exit 0
fi

TODAY=$(date '+%Y-%m-%d')
git commit -m "stats: daily snapshot update $TODAY

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

git push origin main
log "Pushed STATS.md update for $TODAY."
