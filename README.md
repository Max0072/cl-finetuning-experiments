# cl-experiments

Curvature-initialised **Bayesian Learning Rule** for continual **fine-tuning**
without forgetting, on Permuted-MNIST.

## Quickstart (clone → run)

Uses [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.
Nothing else is needed — MNIST downloads itself on first use, and anchors/results
are regenerated on demand (both dirs are gitignored).

```bash
# 1. install (creates the venv, installs deps + dev tools)
uv sync --extra dev

# 2. smoke-check the install (fast, CPU-only, no downloads beyond MNIST)
uv run pytest -q                                     # 25 tests, ~2s
uv run python -m experiments.setup.train_anchor --n 1   # trains+caches one anchor

# 3. run any experiment (each is standalone, resumable, and writes to results/)
uv run python -m experiments.benchmark.final_grid --n-list 1 3 --seeds 0
```

The first experiment that needs data triggers a one-time MNIST download into
`datasets/`. Every script takes `--help`; heavier ones are **resumable** (each
result is saved as it lands, re-running skips finished work). See the full
reproduce-in-order list [below](#experiments-experiments).

## Layout

```
src/cl_experiments/   # library code (installable package)
experiments/          # entry-point scripts, grouped: setup/ benchmark/ figures/ justification/
tests/                # pytest suite
docs/SETTING.md       # the fixed experimental setting (read this first)
datasets/             # local data (gitignored)
results/              # run outputs, checkpoints, logs (gitignored)
```

## Dev

```bash
uv run ruff check .
uv run ruff format .
uv run mypy src
uv run pytest
```

## Experimental setting

The fixed setting (model, data, continual-learning protocol, metrics) is specified
in [`docs/SETTING.md`](docs/SETTING.md). In brief: continual **fine-tuning** of an
MLP on Permuted-MNIST — train an anchor (jointly on the first `n` tasks), then
fine-tune on a 10-task stream, **retention-first** and **data-free**.

**Method**: curvature-initialised **Bayesian Learning Rule** (`cl_experiments.methods.blr`)
— the mean update is the natural gradient `Δμ ∝ σ²·∇`, σ tracks the inverse
curvature, and the posterior is initialised from the diagonal Fisher (Laplace). The
`blr_const` (constant-σ) variant is the ablation isolating the value of curvature
init. With data available, `BLR+replay` strictly improves plain replay.

## Reproducibility

- **Seeding**: `cl_experiments.repro.set_seed` seeds python/numpy/torch and is called
  inside `harness.run_experiment` and `harness.get_anchor`, so every run is
  reproducible from its seed. Anchors are cached per `(n, seed)`. Multi-seed runs
  pass `--seeds 0 1 2` (grid / hybrid / stream-length aggregate mean±std over them).
  The test suite seeds before every test (`conftest.py`). (Note: MPS +
  `torch.func.vmap` are not bit-deterministic; exact reproduction is on CPU/CUDA.)
- **Config**: all tuned hyper-parameters and setting constants live in
  `cl_experiments.config` (`SETTING`, `TUNED_METHODS`) — the single source of truth.
- **Manifests**: each experiment writes `*.manifest.json` (git commit, timestamp,
  device, versions, config, seed, elapsed) next to its results, and a `run.log`.
- **Environment**: pinned in `uv.lock`.

## Experiments (`experiments/`)

Grouped by role (see `experiments/README.md` for the one-line map):

| folder | what it is |
|---|---|
| `setup/` | prepare: train + certify the anchor, tune each method's hyper-parameters |
| `benchmark/` | **the results** — headline method comparisons (grid, frontier, hybrid, stream length) |
| `figures/` | render the paper figures from the benchmark outputs |
| `justification/` | **why the choices are sound** — Fisher validation + all ablations/sensitivity |

Run in order to reproduce `results/`:

> **Determinism**: all committed numbers were produced on **CPU** — MPS +
> `torch.func.vmap` are not bit-reproducible. Every benchmark/justification script
> accepts `--device cpu`; use it to match the committed results.

```bash
# setup
uv run python -m experiments.setup.train_anchor --n 1 3 5 10          # train + cache + certify anchors
uv run python -m experiments.setup.tune_methods --n 3 --n-stream 10   # per-method tuning (Optuna)

# benchmark (the results) -- CPU for reproducibility
uv run python -m experiments.benchmark.final_grid --device cpu                    # method x n x seed grid
uv run python -m experiments.benchmark.frontier --n 3 --device cpu               # retention/plasticity frontier
uv run python -m experiments.benchmark.hybrid_seeds --seeds 0 1 2 --device cpu    # BLR+replay vs replay (β sweep)
uv run python -m experiments.benchmark.stream_length --N 10 20 30 --seeds 0 --device cpu  # scaling; BLR shown as β sweep

# figures
uv run python -m experiments.figures.plot_grid

# justification (the choices are not cherry-picked)
uv run python -m experiments.justification.validate_fisher --n 3      # MC vs exact Fisher agreement
uv run python -m experiments.justification.anchor_convergence         # why moderate (not deep) convergence
uv run python -m experiments.justification.sensitivity                # sigma_prior / rho sweeps
uv run python -m experiments.justification.design_ablations           # n_mc / kappa / freeze_sigma
```
The step-size `β` choice is justified by the whole `benchmark/frontier.py` sweep (not
a single cherry-picked point), and the design knobs by `justification/design_ablations.py`.