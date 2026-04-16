#!/usr/bin/env bash
# Stronger checks after smoke_test: unit-style setup + GPU compile/run + optional scoring e2e.
# Usage:
#   bash scripts/verify_all.sh # smoke + test_setup + run_test
#   bash scripts/verify_all.sh --scoring   # also test_scoring_e2e (extra compile time)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f "$ROOT/.venv/bin/activate" ]]; then
  echo "No .venv — run: bash scripts/bootstrap.sh"
  exit 1
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

RUN_SCORING=0
if [[ "${1:-}" == "--scoring" ]]; then
  RUN_SCORING=1
fi

echo "=== [1/3] smoke_test ==="
python "$ROOT/scripts/smoke_test.py"

echo ""
echo "=== [2/3] experiments.kernelbench.tests.test_setup ==="
python -m experiments.kernelbench.tests.test_setup

echo ""
echo "=== [3/3] experiments.kernelbench.tests.run_test (GPU compile + subprocess) ==="
python -m experiments.kernelbench.tests.run_test

if [[ "$RUN_SCORING" == 1 ]]; then
  echo ""
  echo "=== [extra] experiments.kernelbench.tests.test_scoring_e2e ==="
  python -m experiments.kernelbench.tests.test_scoring_e2e
fi

echo ""
echo "=== verify_all.sh finished OK ==="
