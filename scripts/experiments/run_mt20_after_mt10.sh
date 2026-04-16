#!/usr/bin/env bash
# Wait until the in-flight Multi(10) job exits, then start Multi(20).
#
# Use when MT10 was started alone, e.g.:
#   bash scripts/experiments/run_cross_task_scaling.sh multi10
# and you want MT20 to run immediately after without babysitting.
#
#   nohup bash scripts/experiments/run_mt20_after_mt10.sh \
#     >> outputs/artifacts/kernelbench/mt20_after_mt10.log 2>&1 &
#
# Next time, prefer one session (no waiter needed):
#   bash scripts/experiments/run_cross_task_scaling.sh
#   # default MODE=both → multi10 then multi20 sequentially.
set -euo pipefail
cd "$(dirname "$0")/../.."

LOG="${MT20_FOLLOW_LOG:-outputs/artifacts/kernelbench/mt20_after_mt10.log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

echo "=== run_mt20_after_mt10: $(date -Is) ==="
echo "Waiting for: experiments.kernelbench.main ... --run-name cross_task_mt10"

while pgrep -f "experiments\.kernelbench\.main.*--run-name cross_task_mt10" >/dev/null 2>&1; do
  sleep 60
done

echo "MT10 process finished ($(date -Is))."

PYTHON="${PYTHON:-python3}"
"$PYTHON" <<'PY' || true
import cloudpickle
from pathlib import Path

p = Path("outputs/artifacts/kernelbench/cross_task_mt10/gepa_state.bin")
if not p.is_file():
    print("WARN: no gepa_state.bin for MT10 — did the run fail early?")
else:
    s = cloudpickle.load(p.open("rb"))
    te = s["total_num_evals"] if isinstance(s, dict) else s.total_num_evals
    print(f"MT10 checkpoint total_num_evals={te} (300 when budget exhausted)")
PY

echo "--- Starting Multi(20) ---"
bash scripts/experiments/run_cross_task_scaling.sh multi20

echo "=== run_mt20_after_mt10 done: $(date -Is) ==="
