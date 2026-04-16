#!/usr/bin/env bash
# One-shot env: uv venv + pinned deps + editable gepa + KernelBench.
# Run from repo root. Tested stack: Python 3.12, torch 2.6.0+cu124.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
TORCH_IDX="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

# Recreate venv only when missing or when BOOTSTRAP_FRESH_VENV=1
if [[ "${BOOTSTRAP_FRESH_VENV:-0}" == 1 ]] && [[ -d .venv ]]; then
  rm -rf .venv
fi
if [[ ! -d .venv ]]; then
  uv venv --python "$PYTHON_VERSION" .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

uv pip install --upgrade pip setuptools wheel >/dev/null

uv pip install "torch==2.6.0" --index-url "$TORCH_IDX"

uv pip install \
  "dspy==2.6.27" \
  "litellm==1.80.0" \
  "cloudpickle==3.1.2" \
  "fasteners==0.20" \
  "ninja==1.13.0" \
  "tqdm==4.67.1" \
  "pydantic>=2.12,<3" \
  "llama-index-core==0.14.13" \
  "llama-index-embeddings-huggingface==0.6.1" \
  "llama-index-llms-openai==0.6.13"

if [[ ! -d external/gepa-optimize-anything ]]; then
  echo "ERROR: missing external/gepa-optimize-anything/"
  echo "  Clone the full repo, or: SOURCE_REPO=/path/to/gepa_luke bash scripts/sync_from_gepa_luke.sh"
  exit 1
fi
uv pip install -e "external/gepa-optimize-anything"

if [[ ! -d experiments/kernelbench/KernelBench ]]; then
  echo "ERROR: missing experiments/kernelbench/KernelBench/"
  exit 1
fi
if ! uv pip install -e experiments/kernelbench/KernelBench --no-deps; then
  echo "ERROR: KernelBench editable install failed."
  echo "  Try: PYTHON_VERSION=3.12 BOOTSTRAP_FRESH_VENV=1 bash scripts/bootstrap.sh"
  exit 1
fi

# KernelBench's package __init__ pulls in utils (dotenv, openai, litellm, …); --no-deps skips those.
uv pip install \
  "python-dotenv>=1.0" \
  "openai>=1.0" \
  "requests>=2.28" \
  "numpy>=1.24"

echo ""
echo "Verifying imports..."
python -c "
import torch, dspy, litellm, fasteners, ninja, cloudpickle, gepa, tqdm
import kernelbench  # noqa: F401
print(f'  torch {torch.__version__}  CUDA {torch.version.cuda}')
print(f'  dspy {dspy.__version__}')
print(f'  kernelbench import OK')
print('All imports OK.')
" || {
  echo "ERROR: import check failed."
  exit 1
}

echo ""
echo "Done. From repo root: source .venv/bin/activate"
echo "Run: python -m experiments.kernelbench.main --help"
