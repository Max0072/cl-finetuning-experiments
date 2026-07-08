# fisher_density — per-weight importance across the stream

**Script:** `experiments/justification/fisher_density.py`
**Output:** `results/justification/fisher_density.png`
**Reproduce:** `uv run python -m experiments.justification.fisher_density --n 3 --N 12 --device cpu`

Advance the anchor through a Permuted-MNIST stream (BLR, β=16). After each task, look at
the diagonal Fisher and the Laplace σ it implies (`σ = 1/√(N·F + 1/prior²)`). A weight is
called **important** when its σ drops below `0.9·prior` (a tight admissible zone).

## What the figure shows

**Weights marked important, per-task vs cumulative** — two curves vs number of tasks seen:
- *per-task* — the fraction important under **one task's own** Fisher (a control);
- *cumulative* — the fraction important under the **running Fisher sum** over all tasks so far.

## What it gives (the takeaways)

- **Per-task importance stays sparse and flat (~10%).** Each individual task constrains only
  a small slice of the network — the densification below is *not* because tasks get denser.
- **Cumulative importance saturates (≈2% → ≈87%).** The *union* of the tasks' important sets
  fills the network: as the stream grows, nearly every weight ends up constrained. This is
  capacity filling up.

(Whether the tasks' importance structures *align* across the stream — the pairwise Fisher
correlation and its dynamics — is a separate, cleaner probe in `zone_convergence` (which
measures every task at the SAME point θ). Don't read cross-task overlap off this figure.)

## Why it matters for the method

The anchor's Fisher (task 0) sees only its own ~few-% of weights and is **blind to the sets
later tasks will occupy**. So a one-shot curvature *init* cannot protect what it never saw —
which is why the value of curvature shows up in **online per-task re-consolidation** (keep
accumulating the Fisher), not in the initialisation. And because the *small* net saturates so
fast, this is partly a **capacity artifact**: a larger model would stay unsaturated much
longer (see `capacity_scaling`). This figure is the mechanism behind the 2×2 ablation in
`docs/FINDINGS.md` §4.

*Note:* "importance becomes uniform at capacity" means uniformly **important** (σ small, zones
tight) — not flat/low curvature. The direct cause of `blr ≈ blr_const` is separate: the σ-EMA
overwrites the init within one task (see `curvature_horizon`).
