# Findings — every claim, the experiment that backs it, and the honest verdict

This is the claims ledger: each statement we make is tied to the script that produces its
evidence, with a verdict — **confirmed**, **refined** (true but not how we first framed it),
or **inconclusive** (kept honestly). All numbers are deterministic **CPU**, anchor `n=3`,
seed 0 unless noted. Reproduce any row by running its script with `--device cpu`.

The one-line story:

> Data-free continual fine-tuning is a retention↔plasticity **frontier** traced by the BLR
> step β. On this MNIST testbed the value of curvature is **not** in the static σ-init
> (which coincides with a flat init once the tiny net saturates) but in **online per-task
> re-consolidation** — continually re-measuring importance as the stream grows.

---

## 1. Setting & headline comparisons

| Claim | Experiment | Verdict / evidence |
|---|---|---|
| Methods must be compared as retention↔plasticity **frontiers** (β is the dial), not single points | `benchmark/frontier.py` | β sweep 1→40 traces each curve; up-right dominance is the comparison |
| **online-BLR** raises the data-free frontier and beats online-EWC | `benchmark/frontier.py`, `benchmark/final_grid.py` | at matched plasticity 0.70: online-BLR retention **0.951** vs EWC-online point below the curve |
| **BLR+replay** strictly dominates plain replay when data is available | `benchmark/hybrid_seeds.py` | at matched plasticity: forgetting **0.082 vs 0.182 (−55%)**, higher retention, 3 seeds, tiny std |
| One method spans **super-memory ↔ recency** via β (+ online = sliding window) | `benchmark/memory_window.py` | small β → retention flat vs recency; large β / online → decays with recency |

## 2. Anchor & Fisher (why the estimator is sound)

| Claim | Experiment | Verdict / evidence |
|---|---|---|
| Anchor needs **moderate**, not deep, convergence — deep collapses the Fisher | `justification/anchor_convergence.py` | %-protected and MC↔exact corr both fall as epochs grow (0.92 @ ep1-3 → 0.48 @ ep30) |
| The scalable **MC Fisher ≈ exact** in the moderate regime | `justification/validate_fisher.py` | Pearson **0.923** (MC vs expected) on the certified anchor |

## 3. Robustness (the choices are not cherry-picked)

| Claim | Experiment | Verdict / evidence |
|---|---|---|
| `sigma_prior`, `rho`, `n_samples`, `fisher_batches` are robust / smooth dials | `justification/sensitivity.py` | ρ flat over 20× range; σ_prior & N are smooth frontier dials, not knife-edges |
| Design knobs (`n_mc`, `kappa`) shift the operating point, don't beat the frontier | `justification/design_ablations.py` | n_mc=4 buys ACC at 4× compute; canonical is the cheapest point on the frontier |

## 4. Where does curvature actually live? (the 2×2 ablation)

The core investigation. `frontier.py` sweeps β for the full 2×2: {curvature-σ, flat-σ} ×
{static init, online re-consolidation}.

| Claim | Experiment | Verdict |
|---|---|---|
| Static **curvature-σ init ≈ flat-σ init** on the full stream | `frontier.py` (blr vs blr_const), `curvature_horizon.py` | **Refined.** Curves coincide; at matched plasticity 0.70 both give retention 0.902, forget 0.103 |
| …because both σ lock to the **current (stream) geometry within one task**, not the anchor's | `curvature_horizon.py` | corr(σ_flat, σ_curv) → **0.99 after 1 task**; corr with the anchor-init σ stays **~0.1** |
| …and because tasks **fill capacity**: importance densifies until nearly every weight is protected, so σ tends to uniform and curvature-init coincides with flat-init (it is *not* "washed out") | `justification/fisher_density.py` | per-task ~10% important; cumulative **2%→87%** important; cross-task corr ~0.25 (disjoint sets). At ~full occupancy the protected set is almost everything → no structure to exploit |
| **Online** curvature re-consolidation IS decisive | `frontier.py` (blr_online vs blr_const_online) | at matched plasticity 0.70: retention **0.951 vs 0.886**, forget **0.034 vs 0.091** |

**Conclusion (refined contribution):** the contribution is **online curvature re-consolidation**
— accumulating the Fisher and protecting the *growing* occupied set — not a one-shot
curvature init. Curvature as static init is dominated by the prior-mean anchoring on this
saturating net.

## 5. Scale: the honest bridge to the LLM regime (open frontier, tested only in outline)

| Claim | Experiment | Verdict |
|---|---|---|
| Fast σ-saturation is a **small-capacity artifact**: bigger models need far more data to compress σ | `justification/capacity_scaling.py` | after 10 tasks, % important: width 256 → **84%**, 512 → **40%**, 1024 → **10%**; importance stays structured longer for wider nets |
| So at large scale σ stays loose over a long **data** horizon → curvature protection stays relevant longer | *extrapolation* | **Precondition confirmed** on MNIST; the *sufficiency* ("curvature matters long at scale") needs a real large-model run — out of scope here |
| A cheap steps-per-task proxy shows curvature-init persistence | `justification/curvature_persistence.py` | **Inconclusive (kept honestly).** Fewer steps ⇒ little movement ⇒ little forgetting for the init to save; the gap stays ~0.002 with no trend. The data-vs-capacity test (§5 row 1) is the right one. |
| Width lets a model share a solution across the stream **and** keep the anchor distinct | `justification/zone_convergence.py` | Fisher-importance alignment rises & levels off for both widths. Decomposed at the final θ: stream-vs-stream 256=0.78 / 1024=**0.81** (wider aligns the stream *more*, refuting "spreading→low"); anchor-vs-stream 256=0.68 / 1024=**0.47** (wider keeps the anchor separate — the small net saturates and is forced to merge it). Curvature-level reason a bigger model protects old knowledge. Single seed; probes importance-structure alignment, **not** "a shared low-loss θ exists" (that is retention). |

---

## Reproduce

```bash
# headline comparisons
uv run python -m experiments.benchmark.frontier --n 3 --device cpu          # §1, §4 (2x2)
uv run python -m experiments.benchmark.final_grid --device cpu              # §1
uv run python -m experiments.benchmark.hybrid_seeds --seeds 0 1 2 --device cpu   # §1
uv run python -m experiments.benchmark.memory_window --device cpu           # §1
# justification / mechanism
uv run python -m experiments.justification.validate_fisher --device cpu     # §2
uv run python -m experiments.justification.anchor_convergence --device cpu  # §2
uv run python -m experiments.justification.sensitivity --device cpu         # §3
uv run python -m experiments.justification.design_ablations --device cpu    # §3
uv run python -m experiments.justification.curvature_horizon --device cpu   # §4
uv run python -m experiments.justification.fisher_density --device cpu      # §4
uv run python -m experiments.justification.capacity_scaling --device cpu    # §5
uv run python -m experiments.justification.curvature_persistence --device cpu    # §5 (inconclusive)
```

The fixed protocol behind all of it is [`docs/SETTING.md`](SETTING.md).
