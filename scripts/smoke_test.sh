#!/usr/bin/env bash
# Run scripts/smoke_test.py from repo root with .venv activated.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -f "$ROOT/.venv/bin/activate" ]]; then
  echo "No .venv — run: bash scripts/bootstrap.sh"
  exit 1
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

exec python "$ROOT/scripts/smoke_test.py" "$@"
