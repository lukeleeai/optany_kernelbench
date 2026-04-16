# optany_kernelbench

Self-contained **KernelBench + GEPA** runner for clusters that do **not** have the `gepa_luke` monorepo (e.g. 2× nodes, 8× H100 each). Clone with **HTTPS**, run **`scripts/bootstrap.sh`**, then experiments.

## What ships in git

| Path | Purpose |
|------|---------|
| `experiments/kernelbench/` | `main.py`, eval, RAG, nested `KernelBench/` dataset |
| `external/gepa-optimize-anything/` | `gepa` package (`pip install -e`) |
| `run_with_GPUs.py` | GPU locks (imported from `eval.py`; keep at repo root) |
| `scripts/` | `bootstrap.sh`, `sync_from_gepa_luke.sh`, `run_node_manifest.sh`, copied scaling helpers |

## Rules of thumb

1. **MT10 / MT20** = **one Python process, one node** (shared `outputs/…` + GPU locks). Do not split one MT run across two nodes.
2. **Two nodes** = two **different** jobs (e.g. ST shard A vs B, or unrelated sweeps).
3. **`sync_from_gepa_luke.sh`** runs only on a machine that **has** the full `gepa_luke` tree; then commit + push. Remote nodes **only** `git clone` + `bootstrap.sh`.

## Remote cluster (typical)

```bash
cd /data/home/lakshyaaagrawal/repos   # or your layout
git clone https://github.com/lukeleeai/optany_kernelbench.git OptAnyRebuttal
cd OptAnyRebuttal
bash scripts/bootstrap.sh
cp config/env.example .env && $EDITOR .env   # OPENAI_API_KEY
set -a && source .env && set +a
python -m experiments.kernelbench.main --help
bash scripts/smoke_test.sh
# Optional API check (small cost): bash scripts/smoke_test.sh --ping-openai
```

**Stronger checks (still not a full GEPA run):**

```bash
bash scripts/verify_all.sh              # smoke + test_setup + run_test (JIT CUDA on GPU)
bash scripts/verify_all.sh --scoring    # + test_scoring_e2e (more compile time)
```

- **`test_setup`** — imports, `compute_score`, `extract_code`, dataset load from `KernelBench/`.
- **`run_test`** — real **`load_inline`** compile, correct kernel, subprocess crash isolation, compile-failure path.
- **`test_scoring_e2e`** — several kernel variants; checks the score ordering (optional; slowest).

Private repo: use a GitHub **PAT** as the HTTPS password or `gh auth login`.

**H100 / newer CUDA:** before `bootstrap.sh`, e.g. `export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126`.

**Baselines:** `eval.py` looks under `experiments/kernelbench/KernelBench/results/timing/<hw>/`. On new hardware, generate or copy `baseline_time_torch.json` for that GPU name.

## Two nodes (8 GPUs each)

Same repo revision on both nodes; each runs `bootstrap.sh` and `.env`.

```bash
NODE_RANK=0 bash scripts/run_node_manifest.sh   # node A
NODE_RANK=1 bash scripts/run_node_manifest.sh   # node B
```

Edit `config/node_manifest.json`. Default = two **disjoint ST-style**10-problem shards. For **MT10 → MT20** on one trajectory, run `scripts/experiments/run_cross_task_scaling.sh` (default `both`) on **one** node only.

## Refresh bundle from `gepa_luke` (dev host only)

```bash
cd /path/to/optany_kernelbench
export SOURCE_REPO=/path/to/gepa_luke
bash scripts/sync_from_gepa_luke.sh
bash scripts/bootstrap.sh
git add experiments external run_with_GPUs.py scripts/experiments
git commit -m "Sync from gepa_luke" && git push
```

`rsync` is **source host → this repo directory**; push to GitHub; remotes **never** need `SOURCE_REPO`.

## Layout

- **Entry:** `python -m experiments.kernelbench.main` from **repo root** (so `experiments.*` and `run_with_GPUs` resolve).
- **Scaling helpers:** `scripts/experiments/*.sh`, `cross_task_problem_set.json`.
