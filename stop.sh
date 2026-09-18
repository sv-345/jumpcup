#!/usr/bin/env bash
# Stop the floor bot gracefully (kill switch), then hard-kill as a backstop.
set -euo pipefail
cd "$(dirname "$0")"
touch ./STOP                       # graceful: the loop exits at its next check
echo "STOP file written; waiting for graceful exit..."
sleep 9
# Backstop: kill the caffeinate wrapper + python if still alive.
pkill -f "jumpbot.daemon run" 2>/dev/null && echo "hard-killed remaining process(es)." || echo "already exited."
rm -f daemon.pid
echo "stopped. (the STOP file is auto-removed by ./run.sh on next start)"
