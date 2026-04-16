#!/usr/bin/env bash
# Run the job for NODE_RANK from config/node_manifest.json (CUDA + command).
# Usage: NODE_RANK=0 bash scripts/run_node_manifest.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RANK="${NODE_RANK:?Set NODE_RANK (e.g. 0 or 1)}"
MANIFEST="${NODE_MANIFEST:-$ROOT/config/node_manifest.json}"

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing manifest: $MANIFEST"
  exit 1
fi
if [[ ! -f "$ROOT/.venv/bin/activate" ]]; then
  echo "No .venv — run: bash scripts/bootstrap.sh"
  exit 1
fi

export PYTHONUNBUFFERED=1
export MANIFEST

eval "$(python3 <<'PY'
import json, os, shlex
m = json.load(open(os.environ["MANIFEST"]))
rank = int(os.environ["NODE_RANK"])
node = next(x for x in m["nodes"] if int(x["rank"]) == rank)
gpus = node["cuda_visible_devices"]
cmd = node["command"]
print(f'export CUDA_VISIBLE_DEVICES={shlex.quote(gpus)}')
print(f'CMD={shlex.quote(cmd)}')
PY
)"

echo "[run_node_manifest] NODE_RANK=$RANK CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[run_node_manifest] $CMD"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
cd "$ROOT"
exec bash -c "$CMD"
