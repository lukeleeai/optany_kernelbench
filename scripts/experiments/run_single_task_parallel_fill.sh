#!/usr/bin/env bash
# Single-task baseline (one GEPA run per problem) for the scaling comparison.
# Not multi-task / not "cross-task" — independent runs under outputs/.../st_baseline_<stem>/.
#
# Problem list: cross_task_problem_set.json's "twenty" (shared with the MT phase).
# Skips runs that already have gepa_state.bin unless --force.
#
# Launches up to MAX_PARALLEL jobs (default 8), each bound to one GPU via
# CUDA_VISIBLE_DEVICES so eight one-problem jobs can run concurrently.
#
# Usage (from repo root, after activating venv):
#   export OPENAI_API_KEY=...
#   export DSPY_CACHEDIR="$PWD/.dspy_cache"
#   bash scripts/experiments/run_single_task_parallel_fill.sh
#   bash scripts/experiments/run_single_task_parallel_fill.sh --force

set -euo pipefail
cd "$(dirname "$0")/../.."

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
  FORCE=1
fi

JSON="scripts/experiments/cross_task_problem_set.json"
if [[ ! -f "$JSON" ]]; then
  echo "Missing $JSON — run: python3 scripts/experiments/pick_cross_task_problems.py --seed <seed>"
  exit 1
fi

MAX_PARALLEL="${MAX_PARALLEL:-8}"
MAX_METRIC_CALLS="${MAX_METRIC_CALLS:-30}"
MAX_REFINEMENTS="${MAX_REFINEMENTS:-1}"
PYTHON="${PYTHON:-python3}"

readarray -t PROBLEMS < <("$PYTHON" -c "import json; d=json.load(open('$JSON')); print('\n'.join(d['twenty']))")

echo "Single-task fill: ${#PROBLEMS[@]} problems, max_metric_calls=$MAX_METRIC_CALLS, max_refinements=$MAX_REFINEMENTS, parallel=$MAX_PARALLEL"

prune_finished_pids() {
  local -a alive=()
  local pid
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "$pid" 2>/dev/null; then
      alive+=("$pid")
    fi
  done
  PIDS=("${alive[@]}")
}

launch_one() {
  local prob="$1"
  local gpu="$2"
  local slug="${prob%.py}"
  local run_name="st_baseline_${slug}"
  local log_dir="outputs/artifacts/kernelbench/${run_name}"

  mkdir -p "$log_dir"
  if [[ "$FORCE" -eq 0 && -f "$log_dir/gepa_state.bin" ]]; then
    echo "[skip] $prob -> $run_name (gepa_state.bin exists)"
    return 0
  fi

  echo "[launch] GPU $gpu  $prob -> $run_name"
  # --gpus N means "use indices 0..N-1" (see main.py). Use 1 for a single visible device.
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 \
    nohup "$PYTHON" -m experiments.kernelbench.main \
      --problems "$prob" \
      --max-metric-calls "$MAX_METRIC_CALLS" \
      --max-refinements "$MAX_REFINEMENTS" \
      --frontier-type instance \
      --gpus 1 \
      --run-name "$run_name" \
      >"$log_dir/run.ans" 2>&1 &
  PIDS+=($!)
}

PIDS=()
gpu=0
launched=0
for prob in "${PROBLEMS[@]}"; do
  prune_finished_pids
  while (( ${#PIDS[@]} >= MAX_PARALLEL )); do
    sleep 3
    prune_finished_pids
  done
  before=${#PIDS[@]}
  launch_one "$prob" "$gpu"
  after=${#PIDS[@]}
  if (( after > before )); then
    launched=$((launched + 1))
  fi
  gpu=$(( (gpu + 1) % MAX_PARALLEL ))
done

echo "All $launched jobs launched. Waiting for completion..."
for pid in "${PIDS[@]}"; do
  wait "$pid" || true
done

echo "Done. Launched $launched single-task runs (see outputs/artifacts/kernelbench/st_baseline_*/)."
