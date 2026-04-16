#!/usr/bin/env bash
# Multi-task scaling phase (10- and 20-problem joint runs) + comparison plots.
# Run ONLY after the single-task baseline is complete (not "cross-task" — those are separate ST runs).
#
# Phase 1:  bash scripts/experiments/run_single_task_parallel_fill.sh [--force]
# Phase 2:  bash scripts/experiments/run_mt_scaling_after_st.sh
#
# Waits for in-flight single-task jobs, verifies each of the 20 problems has enough metric log lines
# (st_baseline_<stem>/ or legacy cross_task_st_<stem>/), then runs strictly in order:
#   Multi(10) → Multi(20) → plot_cross_task_scaling.py

set -euo pipefail
cd "$(dirname "$0")/../.."

LOG="${LOG:-outputs/artifacts/kernelbench/mt_scaling_after_st.log}"
mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1

PYTHON="${PYTHON:-python3}"
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-30}"
JSON="scripts/experiments/cross_task_problem_set.json"
echo "=== mt_scaling_after_st: $(date -Is) ==="
echo "Order: single-task (already done or in flight) → Multi(10) → Multi(20) → plots"
echo "Waiting for single-task parallel fill / per-problem kernelbench jobs to finish..."

while pgrep -f "run_single_task_parallel_fill\.sh" >/dev/null 2>&1; do
  sleep 30
done
# Wait for either st_baseline_* or legacy cross_task_st_* runs
while pgrep -f "experiments\.kernelbench\.main.*--run-name st_baseline_" >/dev/null 2>&1 \
   || pgrep -f "experiments\.kernelbench\.main.*--run-name cross_task_st_" >/dev/null 2>&1; do
  sleep 30
done

echo "Checking single-task baseline (expect ${MAX_METRIC_CALLS} metric lines per problem)..."
if ! "$PYTHON" <<PY
import json, sys
from pathlib import Path

need = int("${MAX_METRIC_CALLS}")
data = json.loads(Path("${JSON}").read_text())
twenty = data["twenty"]
art = Path("outputs/artifacts/kernelbench")
bad = []
for p in twenty:
    stem = p.removesuffix(".py")
    log = None
    for sub in (f"st_baseline_{stem}", f"cross_task_st_{stem}"):
        cand = art / sub / "metric_logs.jsonl"
        if cand.is_file():
            log = cand
            break
    if log is None:
        bad.append(f"{p}: no metric_logs.jsonl under st_baseline_{stem} or cross_task_st_{stem}")
        continue
    n = sum(1 for _ in log.open("rb"))
    if n < need:
        bad.append(f"{p}: {n}/{need} lines")
if bad:
    print("Single-task baseline INCOMPLETE — run run_single_task_parallel_fill.sh first:", file=sys.stderr)
    for b in bad[:15]:
        print(" ", b, file=sys.stderr)
    if len(bad) > 15:
        print(f"  ... and {len(bad) - 15} more", file=sys.stderr)
    sys.exit(1)
print(f"Single-task baseline OK: {len(twenty)} problems, >={need} metric lines each")
PY
then
  echo "Aborting: fix ST runs, then re-run this script."
  exit 1
fi

echo "Single-task baseline verified at $(date -Is)"

rm -rf outputs/artifacts/kernelbench/cross_task_mt10 outputs/artifacts/kernelbench/cross_task_mt20

export MAX_REFINEMENTS="${MAX_REFINEMENTS:-1}"
export PYTHONUNBUFFERED=1
echo "--- Phase 2a: Multi(10) ---"
bash scripts/experiments/run_cross_task_scaling.sh multi10
echo "--- Phase 2b: Multi(20) ---"
bash scripts/experiments/run_cross_task_scaling.sh multi20

echo "Generating comparison plots (ST vs MT10 vs MT20)..."
source .venv/bin/activate
export PATH="${HOME}/.local/bin:${PATH}"
python scripts/analysis/plot_cross_task_scaling.py

echo "=== mt_scaling_after_st done: $(date -Is) ==="
