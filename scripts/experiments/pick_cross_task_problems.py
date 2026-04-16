#!/usr/bin/env python3
"""Draw 20 problems from LEVEL_PROBLEMS (31 active), then 10 from those 20.

Writes scripts/experiments/cross_task_problem_set.json for reproducible launches.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Repo root = parent of scripts/
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.kernelbench.eval import LEVEL_PROBLEMS  # noqa: E402


def pool_31_ordered() -> list[str]:
    """Stable order: level1, then level2, then level3 (matches eval.load_dataset)."""
    out: list[str] = []
    for level in ("level1", "level2", "level3"):
        out.extend(LEVEL_PROBLEMS.get(level, []))
    if len(out) != 31:
        raise RuntimeError(f"Expected 31 active LEVEL_PROBLEMS, got {len(out)}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="RNG seed (record in JSON)")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "scripts" / "experiments" / "cross_task_problem_set.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--export-shell",
        action="store_true",
        help="Print export TWENTY='...' TEN='...' for bash",
    )
    args = parser.parse_args()

    pool = pool_31_ordered()
    rng = __import__("random").Random(args.seed)
    twenty = sorted(rng.sample(pool, 20))
    ten = sorted(rng.sample(twenty, 10))

    payload = {
        "schema": "cross_task_problem_set_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "pool_size": len(pool),
        "twenty": twenty,
        "ten": ten,
        "ten_subset_of_twenty": set(ten) <= set(twenty),
    }
    if not payload["ten_subset_of_twenty"]:
        raise RuntimeError("internal: ten must be subset of twenty")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.out} (seed={args.seed}, |twenty|={len(twenty)}, |ten|={len(ten)})")

    if args.export_shell:
        t20 = ",".join(twenty)
        t10 = ",".join(ten)
        print(f"export CROSS_TASK_TWENTY='{t20}'")
        print(f"export CROSS_TASK_TEN='{t10}'")


if __name__ == "__main__":
    main()
