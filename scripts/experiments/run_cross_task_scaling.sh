#!/usr/bin/env bash
# Cross-task scaling: Multi(10) and Multi(20), cartesian frontier, 8 GPUs.
#
# Problem sets come from scripts/experiments/cross_task_problem_set.json
# (20 drawn from 31 LEVEL_PROBLEMS, then 10 drawn from those 20; see pick_cross_task_problems.py).
#
# Before multi-task runs, fill single-task baselines:
#   bash scripts/experiments/run_single_task_parallel_fill.sh
#
# Per-problem budget ~30 metric calls: Multi(10)=300, Multi(20)=600.
#
# Order when using "both": Multi(10) first, then Multi(20) — never the reverse.
#
# Usage:
#   bash scripts/experiments/run_cross_task_scaling.sh           # both: multi10 → multi20
#   bash scripts/experiments/run_cross_task_scaling.sh multi10
#   bash scripts/experiments/run_cross_task_scaling.sh multi20
#
# If multi10 is already running and you want multi20 to start when it exits (no cron):
#   nohup bash scripts/experiments/run_mt20_after_mt10.sh >> outputs/artifacts/kernelbench/mt20_after_mt10.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/../.."

JSON="scripts/experiments/cross_task_problem_set.json"
if [[ ! -f "$JSON" ]]; then
  echo "Missing $JSON"
  exit 1
fi

PYTHON="${PYTHON:-python3}"
TWENTY="$("$PYTHON" -c "import json; print(','.join(json.load(open('$JSON'))['twenty']))")"
TEN="$("$PYTHON" -c "import json; print(','.join(json.load(open('$JSON'))['ten']))")"

MAX_REFINEMENTS="${MAX_REFINEMENTS:-1}"
COMMON_ARGS="--parallel --gpus 0,1,2,3,4,5,6,7 --frontier-type cartesian --max-refinements $MAX_REFINEMENTS"

MODE="${1:-both}"

if [[ "$MODE" == "multi10" || "$MODE" == "both" ]]; then
  mkdir -p outputs/artifacts/kernelbench/cross_task_mt10
  echo "=========================================="
  echo "  Multi(10): 10 problems, 300 metric calls"
  echo "=========================================="
  PYTHONUNBUFFERED=1 "$PYTHON" -m experiments.kernelbench.main \
    --problems "$TEN" \
    --max-metric-calls 300 \
    --run-name cross_task_mt10 \
    $COMMON_ARGS \
    2>&1 | tee outputs/artifacts/kernelbench/cross_task_mt10/run.ans
fi

if [[ "$MODE" == "multi20" || "$MODE" == "both" ]]; then
  mkdir -p outputs/artifacts/kernelbench/cross_task_mt20
  echo "=========================================="
  echo "  Multi(20): 20 problems, 600 metric calls"
  echo "=========================================="
  PYTHONUNBUFFERED=1 "$PYTHON" -m experiments.kernelbench.main \
    --problems "$TWENTY" \
    --max-metric-calls 600 \
    --run-name cross_task_mt20 \
    $COMMON_ARGS \
    2>&1 | tee outputs/artifacts/kernelbench/cross_task_mt20/run.ans
fi

echo "Done."
