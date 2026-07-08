# experiments/

Entry-point scripts, grouped by role. Every script is runnable with
`uv run python -m experiments.<folder>.<name>` and writes to `results/`.

```
setup/          prepare the study
  train_anchor.py    train + cache + certify the anchor(s) (n = 1,3,5,10)
  tune_methods.py    per-method Optuna search -> tuned configs (cl_experiments.config)

benchmark/      THE RESULTS — headline method comparisons
  final_grid.py      method x anchor-size n x seed grid          -> results/benchmark/grid.json
  frontier.py        retention <-> plasticity frontier per method
  hybrid_seeds.py    BLR+replay vs plain replay (3 seeds)
  stream_length.py   how methods scale with stream length N

figures/        render the paper figures from benchmark outputs
  plot_grid.py       ACC / forgetting vs n
                     (frontier / stream_length / hybrid_seeds each save their own PNG)

justification/  WHY THE CHOICES ARE SOUND (defends against "cherry-picked")
  validate_fisher.py       scalable MC Fisher ~ exact Fisher (on the moderate anchor)
  anchor_convergence.py    why MODERATE (not deep) anchor convergence
  sensitivity.py           sigma_prior / rho sweeps around the tuned point
  design_ablations.py      one knob off canonical BLR (n_mc / kappa / freeze_sigma)
```

Read order to validate the study: **setup → benchmark → figures** gives the claims;
**justification** shows every parameter/design choice is robust, not fitted to the
result. The fixed protocol behind all of it is `docs/SETTING.md`.
