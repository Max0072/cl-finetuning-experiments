# zone_convergence — importance-structure alignment, and how width changes it

**Script:** `experiments/justification/zone_convergence.py`
**Output:** `results/justification/zone_convergence.png` (+ reusable `zone_convergence.json`)
**Reproduce:** `uv run python -m experiments.justification.zone_convergence --n 3 --N 8 --widths 256 1024 --device cpu`
(re-plot only, no recompute: add `--replot`)

## Setting

An anchor (MLP width×width, trained jointly on the first `n=3` Permuted-MNIST tasks) is
fine-tuned with BLR (β=16) over a stream of `N` further permuted tasks. **At each checkpoint
θ_t** (after learning *t* stream tasks) we recompute **every** task's diagonal Fisher **at that
single point θ_t** and take the Pearson correlation of all task pairs. Tracking this over
training removes the confound of a naive "trajectory-diagonal" matrix, where each task would be
measured at a different point.

We run two hidden widths (256, 1024) with the same protocol.

**What the correlation is (and is not).** The diagonal Fisher is the *local* curvature of the
loss w.r.t. each weight. Its correlation across tasks tells you whether the tasks agree on
**which weights are sensitive/important** — i.e. their **importance structure**. Crucially, the
`true` Fisher samples the label from the *model*, so it depends on the inputs and weights, not
the task labels. Therefore:

- it measures **importance-structure alignment**, *not* directly "a single θ solves all tasks"
  (that is retention / joint accuracy);
- it **cannot** test compatible-vs-incompatible task streams (conflicting labels on the same
  inputs give an identical Fisher); it **can** test the effect of capacity (width).

## What each panel shows

- **Top — mean pairwise Fisher correlation vs training progress**, one line per width. Rising &
  plateauing = the trajectory settles into a region where the tasks' important weights
  increasingly coincide.
- **Bottom — the full pairwise correlation matrix at the final θ**, one per width (task 0 = the
  joint anchor, 1..N = stream tasks). Bright = tasks share importance structure; dark = disjoint.

## The question this adjudicates

Does a **wider** model align more or less? Two well-motivated, opposite predictions:

- **spreading** — a wide net has free capacity, so different tasks can occupy **different**
  weights → correlation stays **low**;
- **larger zones** — a wide net keeps **wider σ** (bigger admissible zones, measured in
  `capacity_scaling`), so their intersection is larger and a shared point is easier to reach →
  correlation is **higher**.

Both mechanisms are real; the width sweep decides which dominates on this testbed.

## Result & conclusions (n=3, N=8, seed 0)

The mean pairwise correlation **rises and levels off** for both widths (256: 0.48→0.76,
1024: 0.62→0.74), so a shared importance structure IS found. The two overall means are nearly
equal — which **hides** the real story. Decomposing the final matrix (anchor = task 0):

| width | stream-vs-stream | anchor-vs-stream |
|---|---|---|
| 256  | 0.78 | **0.68** |
| 1024 | **0.81** | **0.47** |

- **The "spreading → low correlation" prediction is refuted.** The wider net's *stream* tasks
  align **more**, not less (0.81 vs 0.78), and it starts from a higher baseline — consistent
  with **wider σ / larger admissible zones making a shared point easier to reach**.
- **But spreading appears exactly where it should:** the wide net keeps the **anchor distinct**
  (0.47 vs 0.68). It has the free capacity to *not* fold the anchor into the shared stream
  structure; the small net, saturating, is **forced to merge everything, the anchor included**.

**Takeaway:** width buys two things at once — a strong **shared** solution for the new stream
**and** a **preserved, separate** structure for the old anchor. That is a concrete, curvature-
level reason a larger model protects old knowledge better (it shares weights *among the
stream*, leaving the anchor its own), and it ties directly to `capacity_scaling`.

*Open:* at N=8 the small net is still climbing (saturation forcing it up) while the wide net has
plateaued; a longer stream may separate them further. Single seed — a mechanism illustration.

## What it does NOT license

- It is **not** a claim that "a shared low-loss solution exists" — that is retention/joint
  accuracy, measured elsewhere (`benchmark/frontier.py`, `final_grid.py`).
- The plateau sits **below 1** by construction: the first layer is a pixel permutation and is
  irreducibly task-specific; alignment happens in the deeper representation.
- Single seed, one stream; treat as a mechanism illustration, not a headline number.
