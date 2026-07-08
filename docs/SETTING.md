# Experimental Setting

This document fixes the **experimental setting** for the study: the model, the
data, the continual-learning scenario, the evaluation, and the reproducibility
controls. It gives the *method* only in outline (§1) — the sigma estimation and the
BLR update plug into steps 2–3 and are detailed in code (`cl_experiments.methods`,
`cl_experiments.methods.blr`, `cl_experiments.config`). The point is that any method we
compare runs inside the *same* fixed setting, so differences are attributable to
the method alone.

Status: settled for the MNIST study; remaining open items at the end.

---

## 1. Research goal & framing

**Continual fine-tuning:** adapt an already-trained model to new data **without
forgetting** what it knows — **data-free** (no stored/pretraining data). The
mechanism estimates per-weight importance (curvature) of the knowledge to preserve
and uses it in a Bayesian update so important weights move little and unimportant
ones stay plastic.

**Update rule = canonical Bayesian Learning Rule (BLR)** (Khan & Rue 2021; VON/Vadam
family), implemented in `cl_experiments.methods.blr`. The mean step is the natural gradient
— scaled by the posterior variance `Δμ ∝ σ²·∇` — and the variance tracks the inverse
curvature (precision EMA toward the Laplace target). The mean step-size `β` is the
plasticity dial. Our specific contribution is the **curvature-informed initialisation**
of the posterior (Laplace/Fisher σ), which is orthogonal to and plugs into the BLR
update. It follows directly from the **admissible-zones** view: each weight's posterior
width σ is the half-width of the zone within which the anchor's loss stays low, so
importance-scaled updates are the natural consequence, not an add-on. (Uncertainty-
scaled plasticity is a long-standing idea in Bayesian continual learning; BLR is
simply its canonical variational-inference form.)

**Framing (important):** the point is *fine-tuning*, not standard sequential-CL
benchmarking. So:
- **Primary metric is retention-first** — keep the anchor (the model's knowledge)
  while adapting; not equal-weight average accuracy over many tasks.
- **Competitors are fine-tuning-retention methods** (naive, EWC, online-EWC,
  replay), **not** sequential-CL SoTA (VCL/DER++/A-GEM). Replay is the
  data-storing *foil* (its edge over data-free methods quantifies the price of
  being data-free); the real SoTA arena is LLM scale.
- Methods are compared as **retention↔plasticity frontiers** (each has a knob:
  BLR step β, EWC λ), not single points.

## 2. Pipeline

```
(1) trained model        pretrain on the anchor task (the knowledge to keep)
      │
(2) estimate sigma        <-- METHOD (specified separately)
      │
(3) fine-tune             <-- METHOD (BLR / baselines)
      │
(4) metrics               accuracy on all seen tasks -> ACC, BWT, retention
```

Steps (1) and (4) and the data are fixed here. Steps (2)–(3) are the method
slot (all evaluated in this identical setting) — see §6 for the full list.

## 3. Model

**MLP: 784 → 256 → 256 → 10, ReLU activations, no bias tricks, no BatchNorm.**

Rationale:
- **Comparability** — small MLPs are the canonical architecture for
  Permuted-MNIST continual learning, so our numbers sit next to a known reference.
- **Cost** — fast enough to run ≥5 seeds, Bayesian-optimisation sweeps, and full
  ablations without a GPU budget.
- **Clean isolation** — no convolutional / normalisation confounds; the only
  thing being protected is a matrix of weights, which is exactly what the
  importance mechanism acts on.
- **Known capacity behaviour** — with 256-256 we already observe capacity
  saturation around ~20 tasks, which is itself an informative regime.

Second tier (robustness, planned): a small CNN on a harder benchmark, to show
the effect is not MLP-specific.

### Anchor training (step 1)

The "trained model" we start from is the **anchor**: a model that holds the
knowledge we want to preserve. It can be trained on the first `n` tasks
(`anchor_tasks = n`), not just one — see §4.

**Convergence: a sweet spot, NOT the deepest minimum.** Two opposing pressures:

- *Toward convergence:* the importance/protection scheme is second-order,
  `ΔL ≈ g·Δθ + ½ F·Δθ²`; it uses only the quadratic term, so it assumes the
  first-order `g ≈ 0` (a near-minimum). Off-minimum a low-Fisher/high-gradient
  weight would be freed but moving it hurts linearly.
- *Away from over-convergence:* on near-separable data, driving the loss to ~0
  SATURATES the softmax (p→1, log p→0, gradients collapse), which collapses the
  true Fisher — the model becomes locally insensitive to its weights, the loss
  basin flattens, and the importance signal vanishes (empirically: deep-annealed
  anchor at grad~1e-9 → Laplace sigmas all at the prior, ~0% weights protected;
  MC-vs-exact Fisher corr drops to 0.54). This is *not* an accuracy loss (Fisher
  is still exact) — the finite-width admissible zone is what we actually care
  about, and the local curvature at an over-confident point is a poor proxy for
  it.

**Resolution — moderate convergence:** stop at moderate train loss (`stop_loss`,
default 0.03), Adam constant lr, no deep annealing. Anchor learns all tasks
(acc ~0.97) but stays un-saturated, so the Fisher is informative (n=3: ~7 epochs,
grad~0.07, sigma spread [0.01, 0.05], MC-vs-exact Fisher corr 0.92). This also
matches the realistic LLM regime (Adam, limited budget, never over-fit).

Self-consistency note: sigma is estimated *at whichever minimum we land in*, so
the choice of optimiser is not critical — different optimisers reach minima of
different sharpness, but we always measure the curvature at the actual anchor
point. Hence we pick the simplest reliable optimiser.

**Protocol (fixed):**
- Multi-task **joint** training over tasks `0..n-1`: data mixed and shuffled,
  **balanced** across tasks, single shared 10-way head.
- Optimiser: **Adam, lr = 1e-3, batch size = 128, weight decay = 0.**
  - weight decay is 0 on purpose: it would add `lambda` to the Hessian diagonal
    (an implicit importance floor) and pull `mu` toward 0, contaminating the
    curvature we measure. The importance floor stays an explicit, controlled
    knob of the method, not a training side-effect.
- Train to **moderate convergence**: constant lr, stop when mean train loss ≤
  `stop_loss` (default 0.03), max_epochs cap. Do NOT anneal to a deep minimum —
  that saturates the softmax and collapses the Fisher (see "Convergence" above).
  n=3 stops at ~7 epochs, grad_norm ~0.07, acc ~0.974.
- **Anchors are deterministic → trained once and cached** to
  `results/anchor/anchor_n{n}_seed{seed}.pt`; all downstream experiments load via
  `cl_experiments.harness.get_anchor`, so anchor training is a one-time cost.
- Inputs standardised with the canonical MNIST mean/std (0.1307 / 0.3081).
- Weight init: PyTorch default; fixed seed for init and data order.

**Anchor certification (logged before step 2):**
- GATE: test accuracy on **each** of the `n` anchor tasks ≥ 0.95 — all high and
  balanced (no under-learned task).
- DIAGNOSTICS (logged, not gated): gradient norm and final train loss. We do NOT
  require a tiny gradient — a moderate one is expected and desired (see
  "Convergence" above); over-driving it collapses the Fisher.
- Implementation: `cl_experiments/anchor/train.py::train_anchor` returns an
  `AnchorReport` with `.certified()` (accuracy-based).

The anchor is a **plain** model (no Bayesian weights); the sigmas / BLR update
enter only at the fine-tuning step.

## 4. Data & continual-learning scenario

**Permuted-MNIST, domain-incremental, single shared 10-way head.**

- **Task 0** = plain MNIST (identity permutation).
- **Tasks 1..T-1** = MNIST with a fixed, seed-determined random permutation of
  the 784 input pixels each. Labels and the classifier head are shared across
  all tasks.
- Standard train/test split (60k / 10k); test sets are per-task (same
  permutation applied to the test images).

**Anchor set (`anchor_tasks = n`).** The knowledge to preserve is tasks
`0..n-1`, trained jointly (§3). The model then continues onto tasks `n, n+1, …`
via the method. Two regimes we compare:
- `n = 1` — single-task anchor (plain MNIST); the simplest baseline.
- `n > 1` — **multi-task anchor**: the model already knows several tasks, and
  the curvature captures the geometry of the *intersection* of their admissible
  zones. This is closer to the real use case (a pretrained model knows many
  things) and lets us dial how "full" the network already is before fine-tuning.
Retention is then measured over **all** anchor tasks `0..n-1`.

We sweep `n ∈ {1, 3, 5, 10}`. **Capacity check (resolved):** all four anchors
certify cleanly at the moderate-convergence protocol (§3) — min test acc ~0.974
across n, 7–9 epochs, grad_norm ~0.06–0.23 (moderate, by design — NOT driven to
~1e-9, which collapses the Fisher). The 256-256 MLP has ample capacity for the
*joint* fit; capacity only bites under the *sequential* constraint (the stream),
not the multi-task anchor. So all `n` are kept. (Accuracy certification stays a
hard gate for larger nets/benchmarks where it may bind.)

Rationale:
- **Canonical & comparable** — the standard CL benchmark, matching prior
  Permuted-MNIST evaluations.
- **Single shared head** — forgetting is purely at the *representation* level;
  it avoids the class-incremental "output-head bias" confound (which we
  confirmed dominates and is orthogonal to our mechanism in a Split-MNIST test).
- **Honest testbed for the mechanism** — the method protects weight importance
  (i.e. the learned representation), which is exactly what a permutation shift
  stresses.
- **Scales to streams** — extends naturally to a boundary-free sequence of N
  tasks with no task-boundary information required at fine-tune time.

Known limitation (accepted): a pixel permutation is a full-rank, semantically
unnatural input scramble. We accept this as the price of using the standard
benchmark, and plan Rotated-MNIST and a CIFAR split as robustness checks.

### Two configurations of the same scenario
- **2-task (A → B):** pretrain on task 0, fine-tune on one permuted task.
  Cheap; used for tradeoff curves and Bayesian optimisation.
- **N-task stream:** after the `n` anchor tasks, fine-tune on a **fixed
  continuation stream** of `N = 10` further permuted tasks, **1 epoch each**
  (standard Permuted-MNIST streaming). The stream is the same slice of the
  permutation bank for every anchor size `n` (`stream_perms` = bank indices
  `[10 : 10+N]`), so runs are comparable across `n`. This is the primary
  continual-learning evaluation. Implemented in
  `cl_experiments.harness.stream_loaders`.

## 5. Metrics

Because the anchor is trained **jointly** on tasks `0..n-1` (all known at t=0)
and then the stream is learned one task at a time, the accuracy matrix `R[i][j]`
has rows i = 0 (post-anchor), 1..N (post each stream task) and columns j over all
`T = n + N` tasks. Each task has a `learned_at` event (anchor tasks → 0, stream
task k → k+1). Implemented and tested in `cl_experiments.metrics`
(`evaluate_tasks` builds a row, `compute_metrics` returns the numbers):

- **ACC** — final average accuracy over all tasks (mean of the last row).
- **BWT** — mean over tasks not learned last of `R[-1][j] − R[learned_at][j]`;
  0 = no forgetting, negative = forgetting.
- **learning_acc** — mean accuracy on each task right when learned (plasticity).
- **forgetting** — mean of (peak accuracy seen − final accuracy) per task.
- **anchor_final / anchor_forgetting** — final accuracy on the anchor tasks and
  their drop vs the post-anchor row: **the model's-knowledge retention**, our
  headline quantity.
- **stream_final** — final accuracy on the stream tasks.
- **Retention↔plasticity frontier**: `(stream_final, anchor_final)` as a pair;
  each method's knob (BLR step β, EWC λ) traces a frontier — the primary
  comparison is which method's frontier dominates (up-right), not a single scalar.
  Report mean±std over seeds and across anchor sizes `n`.

## 6. Methods & baselines (run in this same setting)

Data-free fine-tuning-retention methods (the honest comparison set):
- `naive` — plain fine-tuning, no protection (forgetting floor).
- `EWC` (`ewc`) — one-shot anchor-EWC: diagonal-Fisher penalty toward the anchor.
- `online-EWC` (`ewc_online`) — running Fisher, re-consolidated after every task;
  the stronger, standard EWC baseline (protects stream knowledge too, not just the
  anchor).
- `uniform-sigma BLR` (`blr_const`) — BLR with a constant sigma-init (ablation:
  isolates the value of curvature-informed sigma).
- `Fisher-sigma BLR` (`blr`, **ours**) — curvature-initialised BLR; dial = `β`.
- `online BLR` (`blr_online`, **ours**) — BLR with per-task zone re-consolidation
  (the BLR analog of online-EWC): after each task recompute the Fisher, accumulate
  precision, re-derive sigma, move the prior mean to the current weights. Preliminary
  (single-seed CPU): raises the retention↔plasticity frontier over canonical BLR and
  beats online-EWC.

Data-storing (foil / upper bound):
- `replay` — rehearsal buffer of anchor samples (its edge = the price of data-free).
- `BLR + replay` (`blr_replay`, hybrid) — BLR update on replay-augmented batches;
  **strictly dominates** plain replay at matched plasticity (higher ACC & retention,
  ~55% less forgetting, 3 seeds) — a drop-in improvement to replay when data IS available.
- `online BLR + replay` (`blr_online_replay`, hybrid) — online BLR (per-task
  re-consolidation) on replay-augmented batches: the online mechanism plus real
  rehearsal data. The strongest configuration when data is available.

(Not used, and why: sequential-CL SoTA — VCL / DER++ / A-GEM — targets a different
problem, equal-weight multi-task accuracy, not fine-tuning retention; see §1.)

## 7. Reproducibility controls (implemented)

- **Seeding**: `cl_experiments.repro.set_seed` (python/numpy/torch) is called inside
  `harness.run_experiment` and `pipeline.get_anchor`, so every run is reproducible
  from its seed. Caveat: MPS + `torch.func.vmap` are not bit-deterministic — exact
  reproduction is guaranteed on CPU/CUDA.
- **Config (single source of truth)**: `cl_experiments.config` (`SETTING`,
  `TUNED_METHODS`, bases) — no per-script magic constants. We chose a **lightweight
  dataclass config, not Hydra** (deliberate: avoids a framework rewrite).
- **Manifests**: each experiment writes `*.manifest.json` (git commit + dirty flag,
  UTC timestamp, device, torch/python versions, config, seed, elapsed) and a
  `run.log` next to its results.
- **Seeds**: headline numbers reported as **mean ± std over 3 seeds** (0,1,2).
- Results written under `results/<experiment>/`. Env pinned in `uv.lock`.

## 8. Decisions

Resolved:
- [x] **Model = MLP 784-256-256-10**.
- [x] **Benchmark = Permuted-MNIST**, domain-incremental, single head.
- [x] **Framing = data-free continual fine-tuning, retention-first**; competitors
      are fine-tuning-retention methods, not sequential-CL SoTA (§1).
- [x] Anchor = **Adam** (lr 1e-3, batch 128, wd 0), joint multi-task over `0..n-1`,
      trained to **MODERATE convergence** (`stop_loss = 0.03`, NOT a deep minimum —
      that collapses the Fisher). Certification is **accuracy-based** (≥0.95);
      grad_norm/loss are logged diagnostics, not gated.
- [x] Multi-task anchor sweep **`n ∈ {1,3,5,10}`**; retention over all anchor tasks.
- [x] Continuation stream **N = 10** fixed tasks, **1 epoch/task**, same for all `n`.
- [x] Baselines incl. **online-EWC** and **online-BLR** (per-task re-consolidation),
      plus the data-storing hybrids **BLR+replay** / **BLR-online+replay** (§6).
- [x] Reproducibility = **lightweight** (repro.py + config.py + manifests + logging),
      **not Hydra**; **3 seeds** for headline numbers.
- [x] **Update rule = canonical BLR** (`cl_experiments.methods.blr`). All canonical results
      (grid, frontier, hybrid) run under BLR.

Still open:
- [ ] Second-tier scaling target (small CNN + which dataset).
- [ ] **LLM scale** (self-sampled true Fisher) — the real SoTA arena and the whole point.
