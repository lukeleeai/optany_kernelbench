#!/usr/bin/env bash
# Copy a minimal working tree from the full gepa_luke repo into this bundle.
# Run ONLY on a machine that actually has that monorepo. H100 nodes usually do not
# have /data/lukedhlee — sync on your dev box, commit/push, then clone on the cluster.
set -euo pipefail
DEST="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE="${SOURCE_REPO:?Set SOURCE_REPO to the gepa_luke root (no default — not portable across clusters)}"

if [[ ! -d "$SOURCE/experiments/kernelbench" ]]; then
  echo "Cannot find gepa_luke at: $SOURCE"
  echo "Set SOURCE_REPO to the monorepo root on THIS host, e.g.:"
  echo "  export SOURCE_REPO=/path/to/gepa_luke"
  echo "If you are on a cluster with no gepa_luke tree, skip this script: clone optany_kernelbench"
  echo "after experiments/ + external/ are already in git (or rsync a tarball from a machine that ran sync)."
  exit 1
fi

mkdir -p "$DEST/external" "$DEST/experiments" "$DEST/scripts/experiments"

rsync -a --delete \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.pytest_cache/' \
  --exclude '.git/' \
  "$SOURCE/experiments/kernelbench/" \
  "$DEST/experiments/kernelbench/"

rsync -a --delete \
  --exclude '__pycache__/' \
  --exclude '.git/' \
  "$SOURCE/external/gepa-optimize-anything/" \
  "$DEST/external/gepa-optimize-anything/"

rsync -a "$SOURCE/run_with_GPUs.py" "$DEST/run_with_GPUs.py"

for f in \
  run_cross_task_scaling.sh \
  run_single_task_parallel_fill.sh \
  run_mt20_after_mt10.sh \
  run_mt_scaling_after_st.sh \
  run_cross_task_driver.sh \
  pick_cross_task_problems.py \
  cross_task_problem_set.json \
  ; do
  if [[ -f "$SOURCE/scripts/experiments/$f" ]]; then
    rsync -a "$SOURCE/scripts/experiments/$f" "$DEST/scripts/experiments/$f"
  fi
done

echo "Synced from $SOURCE -> $DEST"
echo "Next: bash scripts/bootstrap.sh"
