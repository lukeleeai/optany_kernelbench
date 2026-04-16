#!/usr/bin/env bash
# Run the job for this machine's NODE_RANK using config/node_manifest.json.
# Usage on node A: NODE_RANK=0 bash scripts/run_node_manifest.sh
# Usage on node B: NODE_RANK=1 bash scripts/run_node_manifest.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

RANK="${NODE_RANK:?Set NODE_RANK to 0 or 1 (or edit script for more ranks)}"
MANIFEST="${NODE_MANIFEST:-$ROOT/config/node_manifest.json}"

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
bash -c "$CMD"
