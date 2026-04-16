#!/usr/bin/env python3
"""Measure PyTorch reference baselines on this machine and compare to shipped JSON.

Uses the same subprocess path as LLM kernel eval (`execute_baseline`).

Examples:
  uv run python scripts/experiments/compare_kernelbench_baselines.py --levels level1 --representative
  uv run python scripts/experiments/compare_kernelbench_baselines.py \\
    --items level1/1_Square_matrix_multiplication_.py,level3/1_MLP.py
  KERNELBENCH_BASELINE_HW=H100_PCIe_LambdaLabs uv run python scripts/experiments/compare_kernelbench_baselines.py --max-per-level 5
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.kernelbench.eval import (  # noqa: E402
    BASELINE_PATH,
    KERNELBENCH_ROOT,
    LEVEL_PROBLEMS,
    execute_baseline,
)


def _visible_gpu_line() -> str:
    try:
        r = subprocess.run(
            ["nvidia-smi", "-i", "0", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode == 0 and (r.stdout or "").strip():
            return (r.stdout.strip().splitlines()[0] or "").strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return "unknown"


def _parse_items(raw: str | None) -> list[tuple[str, str]]:
    if not raw:
        return []
    out: list[tuple[str, str]] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "/" not in part:
            raise SystemExit(f"--items entry must be level/file.py, got: {part!r}")
        level, fname = part.split("/", 1)
        level = level.strip()
        fname = fname.strip()
        if level not in ("level1", "level2", "level3"):
            raise SystemExit(f"Invalid level in --items: {level!r}")
        out.append((level, fname))
    return out


def _iter_problems(
    levels: list[str],
    *,
    representative: bool,
    max_per_level: int,
    seed: int,
    items: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    if items:
        seen: set[tuple[str, str]] = set()
        ordered: list[tuple[str, str]] = []
        for level, pid in items:
            if level not in levels:
                continue
            key = (level, pid)
            if key not in seen:
                seen.add(key)
                ordered.append(key)
        if not ordered:
            raise SystemExit("No --items matched --levels.")
        return ordered

    rng = random.Random(seed)
    out: list[tuple[str, str]] = []
    for level in levels:
        d = KERNELBENCH_ROOT / "KernelBench" / level
        if not d.is_dir():
            continue
        if representative:
            names = [n for n in LEVEL_PROBLEMS.get(level, []) if (d / n).is_file()]
        else:
            names = sorted(p.name for p in d.glob("*.py"))
        if not names:
            continue
        if max_per_level <= 0 or len(names) <= max_per_level:
            chosen = names
        else:
            chosen = rng.sample(names, max_per_level)
            chosen.sort()
        for pid in chosen:
            out.append((level, pid))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--levels",
        default="level1",
        help="Comma-separated: level1,level2,level3 (default: level1)",
    )
    p.add_argument(
        "--published-json",
        type=Path,
        default=None,
        help=f"baseline_time_torch.json to compare against (default: {BASELINE_PATH})",
    )
    p.add_argument(
        "--representative",
        action="store_true",
        help="Use LEVEL_PROBLEMS subset from eval (same as load_dataset), not a random sample.",
    )
    p.add_argument(
        "--max-per-level",
        type=int,
        default=8,
        help="When not using --items or --representative: sample this many .py files per level (default: 8). Use 0 for all.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for --max-per-level sampling (default: 0).",
    )
    p.add_argument(
        "--items",
        default=None,
        help="Comma list of level/file.py (e.g. level1/1_Square_matrix_multiplication_.py). Overrides sampling.",
    )
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index (default: 0).")
    p.add_argument("--timeout", type=int, default=360, help="Per-problem timeout seconds.")
    p.add_argument("--output", type=Path, default=None, help="Write JSON summary to this path.")
    args = p.parse_args()

    levels = [x.strip() for x in args.levels.split(",") if x.strip()]
    for lv in levels:
        if lv not in ("level1", "level2", "level3"):
            raise SystemExit(f"Invalid level: {lv!r}")

    pub_path = args.published_json or BASELINE_PATH
    if not pub_path.is_file():
        raise SystemExit(f"Published baseline JSON not found: {pub_path}")

    with open(pub_path) as f:
        published_root = json.load(f)

    items = _parse_items(args.items)
    pairs = _iter_problems(
        levels,
        representative=args.representative,
        max_per_level=args.max_per_level,
        seed=args.seed,
        items=items,
    )
    if not pairs:
        raise SystemExit("No problems selected (check --levels and KernelBench checkout).")

    gpu_name = _visible_gpu_line()
    print(f"Visible GPU (index {args.gpu}): {gpu_name}", flush=True)
    print(f"Published JSON: {pub_path.resolve()}", flush=True)
    print(f"Problems: {len(pairs)}", flush=True)
    print(flush=True)

    rows: list[dict] = []
    hdr = f"{'level':<8} {'problem':<52} {'pub_ms':>10} {'meas_ms':>10} {'ratio':>8} {'d_pct':>8} notes"
    print(hdr)
    print("-" * len(hdr), flush=True)

    for level, problem_id in pairs:
        path = KERNELBENCH_ROOT / "KernelBench" / level / problem_id
        if not path.is_file():
            print(f"{level:<8} {problem_id:<52} {'—':>10} {'—':>10} {'—':>8} {'—':>8}  missing file", flush=True)
            rows.append(
                {
                    "level": level,
                    "problem_id": problem_id,
                    "published_ms": None,
                    "measured_ms": None,
                    "error": "missing_problem_file",
                }
            )
            continue

        pub_block = published_root.get(level, {}).get(problem_id, {})
        pub_mean = pub_block.get("mean")
        if pub_mean is None:
            pub_mean_f: float | None = None
        else:
            try:
                pub_mean_f = float(pub_mean)
            except (TypeError, ValueError):
                pub_mean_f = None

        ref_arch = path.read_text()
        res = execute_baseline(ref_arch, timeout=args.timeout, device=args.gpu)
        meas = res.get("PerformanceStatsMean")
        meas_f: float | None = float(meas) if meas is not None and meas > 0 else None

        notes: list[str] = []
        if not res.get("CompilationSucceeded"):
            notes.append("compile_fail")
        if not res.get("CorrectnessSucceeded"):
            notes.append("correctness_fail")
        if meas_f is None:
            notes.append("no_runtime")

        ratio_s = d_pct_s = "—"
        ratio: float | None = None
        d_pct: float | None = None
        if pub_mean_f and pub_mean_f > 0 and meas_f is not None:
            ratio = meas_f / pub_mean_f
            ratio_s = f"{ratio:.3f}"
            d_pct = 100.0 * (meas_f - pub_mean_f) / pub_mean_f
            d_pct_s = f"{d_pct:+.1f}"

        pub_s = f"{pub_mean_f:.4g}" if pub_mean_f is not None else "—"
        meas_s = f"{meas_f:.4g}" if meas_f is not None else "—"
        note_str = ", ".join(notes) if notes else ""
        print(
            f"{level:<8} {problem_id:<52} {pub_s:>10} {meas_s:>10} {ratio_s:>8} {d_pct_s:>8}  {note_str}",
            flush=True,
        )
        prof = res.get("ProfilingInfo") or {}
        rows.append(
            {
                "level": level,
                "problem_id": problem_id,
                "published_ms": pub_mean_f,
                "published_meta": {k: v for k, v in pub_block.items() if k != "mean"},
                "measured_ms": meas_f,
                "measured_side": {
                    "CompilationSucceeded": res.get("CompilationSucceeded"),
                    "CorrectnessSucceeded": res.get("CorrectnessSucceeded"),
                    "runtime_std_ms": prof.get("runtime_std_ms"),
                },
                "ratio_measured_over_published": ratio,
                "delta_percent_vs_published": d_pct,
                "notes": notes,
            }
        )

    summary = {
        "visible_gpu_name": gpu_name,
        "cuda_device_index": args.gpu,
        "published_json": str(pub_path.resolve()),
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nWrote {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
