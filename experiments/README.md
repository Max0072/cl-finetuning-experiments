# experiments/

Entry-point scripts, grouped by role. Every script is runnable with
`uv run python -m experiments.<folder>.<name>` and writes to `results/`.

```
setup/          prepare the study
  train_anchor.py    train + cache + certify the anchor(s) (n = 1,3,5,10)
  tune_methods.py    per-method Optuna search -> tuned configs (cl_experiments.config)

benchmark/      THE RESULTS — headline method comparisons
  final_grid.py      method x anchor-size n x seed grid          -> results/benchmark/grid.json
  frontier.py        retention <-> plasticity frontier + the 2x2 curvature ablation
  hybrid_seeds.py    BLR+replay vs plain replay, beta-swept (3 seeds)
  stream_length.py   how methods scale with stream length N (BLR as a beta sweep)
  memory_window.py   one method, every regime: retention vs recency (super-memory <-> recency)

figures/        render the paper figures from benchmark outputs
  plot_grid.py       ACC / forgetting vs n
                     (frontier / stream_length / hybrid_seeds / memory_window save their own PNG)

justification/  WHY THE CHOICES ARE SOUND (defends against "cherry-picked")
  validate_fisher.py       scalable MC Fisher ~ exact Fisher (on the moderate anchor)
  anchor_convergence.py    why MODERATE (not deep) anchor convergence
  sensitivity.py           sigma_prior / rho / n_samples / fisher_batches sweeps
  design_ablations.py      one knob off canonical BLR (n_mc / kappa)
  curvature_horizon.py     curvature-sigma vs flat-sigma over the horizon (init washes to the
                           stream geometry within 1 task)
  fisher_density.py        importance densifies to ~uniform as the stream fills capacity
  capacity_scaling.py      wider models need more data to compress sigma (scale bridge)
  curvature_persistence.py steps-per-task proxy (INCONCLUSIVE, kept honestly)
  zone_convergence.py      importance-structure alignment vs training + model width
```

Read order to validate the study: **setup → benchmark → figures** gives the claims;
**justification** shows every parameter/design choice is robust and locates where the
curvature actually helps. The claims ledger (claim ↔ experiment ↔ verdict) is
`docs/FINDINGS.md`; the fixed protocol is `docs/SETTING.md`.
