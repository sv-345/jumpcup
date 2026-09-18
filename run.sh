#!/usr/bin/env bash
# Launch the floor bot detached via nohup + caffeinate so a closed lid / dropped
# SSH / logout won't kill it and the Mac won't idle-sleep. (tmux isn't installed
# on this host; nohup+caffeinate meets the same survive-disconnect requirement.)
# Idempotent: the daemon recovers existing predictions and only PATCHes genuine
# moves, so a restart never double-submits.
set -euo pipefail
cd "$(dirname "$0")"

if pgrep -f "jumpbot.daemon run" >/dev/null 2>&1; then
  echo "already running (pid $(pgrep -f 'jumpbot.daemon run' | tr '\n' ' ')). ./stop.sh first."
  exit 1
fi

rm -f ./STOP   # clear any stale kill switch
mkdir -p logs
nohup caffeinate -i .venv/bin/python -m jumpbot.daemon run >> logs/daemon.out 2>&1 &
echo $! > daemon.pid

sleep 4
if pgrep -f "jumpbot.daemon run" >/dev/null 2>&1; then
  echo "started via nohup+caffeinate (wrapper pid $(cat daemon.pid))."
  echo "  logs:   tail -f logs/jumpbot.log"
  echo "  status: .venv/bin/python -m jumpbot.daemon status"
  echo "  stop:   ./stop.sh   (or: touch ./STOP)"
else
  echo "FAILED to start — last lines of logs/daemon.out:"; tail -n 20 logs/daemon.out; exit 1
fi
