#!/usr/bin/env bash
# Create venv and install deps. Run from repo root after sync_from_gepa_luke.sh.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv first: https://docs.astral.sh/uv/"
  exit 1
fi

uv venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

TORCH_IDX="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
uv pip install torch --index-url "$TORCH_IDX"
uv pip install ninja dspy-ai litellm fasteners llama-index-core llama-index-embeddings-huggingface llama-index-llms-openai cloudpickle

if [[ -d external/gepa-optimize-anything ]]; then
  uv pip install -e external/gepa-optimize-anything
else
  echo "WARN: external/gepa-optimize-anything missing — run scripts/sync_from_gepa_luke.sh"
  exit 1
fi

# KernelBench is vendored under experiments/kernelbench/KernelBench after sync.
uv pip install -e experiments/kernelbench/KernelBench --no-deps || {
  echo "If editable install fails, try: uv pip install git+https://github.com/ScalingIntelligence/KernelBench.git --no-deps"
  exit 1
}

echo "bootstrap.sh done. Activate with: source .venv/bin/activate"
