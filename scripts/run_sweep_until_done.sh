#!/usr/bin/env bash
# Supervisor for the 40-run hyperparameter sweep.
#
# The sweep itself is resumable: every finished run writes
# logs/<algo>/<run_id>.json and a relaunch skips completed work. This wrapper
# simply keeps relaunching until all 40 results exist, so a crashed worker,
# an OOM kill or a reboot costs at most the runs that were in flight.
set -uo pipefail
cd "$(dirname "$0")/.."

UV="${UV_BIN:-$HOME/.local/bin/uv}"
EXPECTED=40
MAX_ATTEMPTS=40
LOG=logs/_sweep/sweep_run.log
mkdir -p logs/_sweep

count_done() { ls logs/dqn/*.json logs/reinforce/*.json logs/ppo/*.json logs/a2c/*.json 2>/dev/null | wc -l; }

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  have=$(count_done)
  if [ "$have" -ge "$EXPECTED" ]; then
    echo "[supervisor] all $EXPECTED runs present; done." | tee -a "$LOG"
    exit 0
  fi
  echo "[supervisor] attempt $attempt/$MAX_ATTEMPTS — $have/$EXPECTED complete, launching sweep $(date '+%F %T')" | tee -a "$LOG"
  "$UV" run python -m training.sweep --algo all --workers "${SWEEP_WORKERS:-10}" >> "$LOG" 2>&1
  rc=$?
  echo "[supervisor] sweep exited rc=$rc with $(count_done)/$EXPECTED complete" | tee -a "$LOG"
  # If it exited without progressing at all, back off before retrying.
  [ "$(count_done)" -le "$have" ] && sleep 20
done

echo "[supervisor] giving up after $MAX_ATTEMPTS attempts ($(count_done)/$EXPECTED)" | tee -a "$LOG"
exit 1
