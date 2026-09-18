#!/usr/bin/env bash
# Watchdog: keep claudius alive. Run by launchd at load/login + every ~5 min.
# Idempotent — run.sh's own pgrep guard means this never double-starts a daemon.
# Touch ./WATCHDOG_OFF in the repo to pause auto-restart during maintenance.
cd "$(dirname "$0")" || exit 0
[ -f WATCHDOG_OFF ] && exit 0
if pgrep -f "jumpbot.daemon run" >/dev/null 2>&1; then
  exit 0   # alive — nothing to do
fi
echo "$(date '+%Y-%m-%dT%H:%M:%S') watchdog: claudius not running -> starting" >> logs/watchdog.log
./run.sh >> logs/watchdog.log 2>&1
