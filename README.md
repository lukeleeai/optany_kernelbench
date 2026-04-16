# optany_kernelbench

Portable slice of the KernelBench + GEPA (“optimize anything”) experiment stack for **extra clusters** (e.g. 2× nodes with 8× H100 each).

## Design choices

1. **Folder + git, not a manual tarball** — Keep this directory as a **small repo**: run `scripts/sync_from_gepa_luke.sh` after changes upstream, then commit. Prefer a **private** GitHub repo until you are sure there are no secrets, internal paths, or unpublished artifacts.
2. **Multi-task (MT10/MT20) stays on one node** — One `experiments.kernelbench.main` process expects **shared GPU locks under one `outputs/.../run_dir`**. Do **not** split a single MT10 run across two nodes without new distributed coordination.
3. **Use two nodes for parallelism** by running **different** jobs: e.g. ST shard A vs ST shard B, two seeds, MT10 on node0 and a long ST sweep on node 1, or two disjoint problem lists.

## Paths: dev box vs compute cluster

- **`sync_from_gepa_luke.sh` only runs where `gepa_luke` (or a copy of it) actually lives** — e.g. your current cluster under `/data/lukedhlee/gepa_luke`.
- **Other clusters (e.g. 2× H100 nodes) often have no `/data/lukedhlee/` at all.** There you **do not** run sync; you **`git clone`** this repo after it already contains `experiments/` and `external/` (committed or released), then **`bootstrap.sh`**.

## Quick start

### A — Update the bundle from `gepa_luke` (one machine that has the monorepo)

```bash
cd /path/to/optany_kernelbench
export SOURCE_REPO=/path/to/gepa_luke    # e.g. /data/lukedhlee/gepa_luke on that host only
bash scripts/sync_from_gepa_luke.sh
bash scripts/bootstrap.sh
git add experiments external run_with_GPUs.py scripts/experiments
git commit -m "Sync from gepa_luke" && git push   # optional: ship to GitHub for other clusters
```

### B — H100 / remote nodes (no `gepa_luke`, no `/data/lukedhlee`)

```bash
cd $HOME   # or your project root on that cluster
git clone https://github.com/lukeleeai/optany_kernelbench.git
cd optany_kernelbench
bash scripts/bootstrap.sh
cp config/env.example .env && vim .env
set -a && source .env && set +a
```

Do **not** assume `/data/lukedhlee/optany_kernelbench` exists on those nodes; use wherever you clone.

If the repo is **private**, HTTPS clone needs credentials (e.g. GitHub **personal access token** as the password, or `gh auth login`).

Smoke test:

```bash
python -m experiments.kernelbench.main --help
```

For **CUDA / PyTorch** wheels on H100, set the index your cluster uses, e.g. `export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126` before `bootstrap.sh`.

## Two-node pattern (8× H100 per node)

On **each** node, clone or rsync the **same** repo revision, run `bootstrap.sh`, and set:

- `CUDA_VISIBLE_DEVICES` is applied by the manifest (default all eight)
- `NODE_RANK=0` or `NODE_RANK=1`

Then:

```bash
bash scripts/run_node_manifest.sh
```

Default `config/node_manifest.json` runs **two disjoint ST-style shards** (10 problems each) so both nodes stay busy without sharing one MT job.

**Do not** schedule **MT10 on node0** and **MT20 on node1** **at the same time** if you need the same **sequential** protocol as flaminio (MT20 after MT10, shared trajectory). For that, run `run_cross_task_scaling.sh` **both** on a **single** node, or run MT10 then MT20 back-to-back on one node.

Edit `config/node_manifest.json` for other splits. See `config/node_manifest.example.json`.

For **single-task** parallelism, split the problem list (e.g. 10 + 10 of the “twenty”) into two files and point each node at one file via `--problems "$(cat …)"`.

## Baselines on H100

`experiments/kernelbench/eval.py` picks `KernelBench/results/timing/<hw>/baseline_time_torch.json`. On new hardware you may need to generate or copy an **H100** baseline JSON under `experiments/kernelbench/KernelBench/results/timing/` (see KernelBench `utils/generate_baseline.py` or your flaminio workflow).

## Layout after sync

- `experiments/kernelbench/` — main entrypoint and eval stack (includes nested `KernelBench/` tree)
- `external/gepa-optimize-anything/` — `gepa` package (editable install)
- `run_with_GPUs.py` — GPU lock helper (repo root; must stay importable)
- `scripts/experiments/` — scaling helpers copied from the source repo
