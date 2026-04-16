#!/usr/bin/env python3
"""Minimal sanity check: imports, CUDA, dataset paths, optional one-token LLM ping.

Run from repo root with venv active:
  source .venv/bin/activate
  python scripts/smoke_test.py

Or: bash scripts/smoke_test.sh
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)


def main() -> int:
    p = argparse.ArgumentParser(description="Verify optany_kernelbench setup without a full GEPA run.")
    p.add_argument(
        "--ping-openai",
        action="store_true",
        help="Send one tiny chat completion (costs API $; needs OPENAI_API_KEY).",
    )
    args = p.parse_args()

    print("== smoke_test: optany_kernelbench ==\n")

    print("[1] Core imports")
    import cloudpickle  # noqa: F401
    import fasteners  # noqa: F401
    import ninja  # noqa: F401
    import torch
    import dspy
    import litellm
    import gepa
    import kernelbench  # noqa: F401
    import tqdm  # noqa: F401

    from experiments.kernelbench.eval import KERNELBENCH_ROOT, get_free_gpus
    from gepa.optimize_anything import GEPAConfig  # noqa: F401

    print(f"    torch {torch.__version__}  CUDA {torch.version.cuda}")
    print(f"    dspy {dspy.__version__}")
    print("    gepa, kernelbench, experiments.kernelbench.eval OK")

    print("\n[2] CUDA devices")
    print(f"    torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"    device_count = {torch.cuda.device_count()}")
        print(f"    device0 = {torch.cuda.get_device_name(0)!r}")

    print("\n[3] get_free_gpus() (nvidia-smi)")
    try:
        free = get_free_gpus()
        print(f"    idle indices: {free}")
    except Exception as e:
        print(f"    WARN: {e}")

    print("\n[4] KernelBench problem files")
    level1 = KERNELBENCH_ROOT / "KernelBench" / "level1"
    if level1.is_dir():
        n = len([x for x in level1.glob("*.py") if x.is_file()])
        print(f"    {level1}: {n} .py files")
        if n < 10:
            print("    WARN: expected many level1 problems; is the dataset tree complete?")
    else:
        print(f"    FAIL: missing {level1}")
        return 1

    print("\n[5] Baseline timing JSON (for speedup scoring)")
    timing = KERNELBENCH_ROOT / "results" / "timing"
    found = list(timing.rglob("baseline_time_torch.json")) if timing.is_dir() else []
    if found:
        print(f"    found {len(found)} baseline_time_torch.json (e.g. {found[0].relative_to(KERNELBENCH_ROOT)})")
    else:
        print(f"    WARN: no baseline_time_torch.json under {timing} — generate or copy for this hardware")

    if args.ping_openai:
        print("\n[6] OpenAI ping (--ping-openai)")
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            print("    FAIL: OPENAI_API_KEY not set")
            return 1
        model = os.environ.get("SMOKE_LLM", "gpt-4o-mini")
        try:
            r = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "Reply with exactly: OK"}],
                max_tokens=4,
            )
            text = (r.choices[0].message.content or "").strip()
            print(f"    model={model!r} response={text!r}")
        except Exception as e:
            print(f"    FAIL: {e}")
            return 1
    else:
        print("\n[6] OpenAI ping skipped (use --ping-openai to test API)")

    print("\n== all checks passed ==")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
