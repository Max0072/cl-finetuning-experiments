"""Single source of truth for the fixed experimental setting and tuned configs.

Numbers that used to be duplicated across experiment scripts live here so there is
exactly one place to change them. See docs/SETTING.md for the rationale.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Setting:
    # model
    hidden: tuple[int, ...] = (256, 256)
    # data / stream
    batch_size: int = 128
    anchor_bank: int = 10          # size of the shared permutation bank
    n_stream: int = 10             # continuation-stream length
    epochs_per_task: int = 1
    # anchor training (moderate convergence -- NOT deep, see docs/SETTING.md)
    anchor_lr: float = 1e-3
    anchor_stop_loss: float = 0.03
    anchor_max_epochs: int = 60
    # sweep defaults
    anchor_ns: tuple[int, ...] = (1, 3, 5, 10)
    seeds: tuple[int, ...] = (0, 1, 2)


SETTING = Setting()

# The method is curvature-initialised **BLR** (canonical Bayesian Learning Rule).
# Tuned per-method on n=3 over the FULL 10-task stream, DETERMINISTIC CPU, Optuna/TPE,
# 30 trials, ACC objective (results/tune/tuned_n3_stream10_seed0.json + optuna_*.db).
# Each entry: (build_learner method name, kwargs). Single source of truth for the
# comparison scripts (final_grid, frontier, hybrid_seeds). Sigma always consolidates
# (no freeze_sigma). Ablations/foils:
#   blr_const  -- constant sigma-init (isolates the value of curvature init).
#   ewc_online / blr_online -- per-task re-consolidated variants.
#   *_replay   -- data-storing foils (rehearsal buffer of anchor samples).
TUNED_METHODS: dict[str, tuple[str, dict]] = {
    "naive": ("naive", {"lr": 3.18e-4}),
    "ewc": ("ewc", {"ewc_lambda": 104.4, "lr": 3.01e-4, "fisher_batches": 30}),
    "ewc_online": ("ewc_online", {"ewc_lambda": 103.1, "lr": 5.36e-4, "fisher_batches": 30}),
    "blr_const": ("blr", {"sigma_mode": "const", "sigma_const": 0.0638,
                          "sigma_prior": 0.0523, "n_samples": 3.57e4,
                          "beta": 19.21, "rho": 0.357}),
    "blr_laplace": ("blr", {"sigma_mode": "laplace", "fisher_mode": "true",
                            "fisher_batches": 30, "sigma_prior": 0.0481,
                            "n_samples": 8.06e3, "beta": 18.35, "rho": 0.0205}),
    "blr_online": ("blr_online", {"sigma_mode": "laplace", "fisher_mode": "true",
                                  "fisher_batches": 30, "sigma_prior": 0.1359,
                                  "n_samples": 5.24e3, "beta": 6.48, "rho": 0.0434}),
    # data-storing foils / upper bound
    "replay": ("replay", {"buffer_size": 6598, "replay_bs": 100, "lr": 3.02e-4}),
    "blr_replay": ("blr_replay", {"sigma_mode": "laplace", "fisher_mode": "true",
                                  "fisher_batches": 30, "sigma_prior": 0.0742,
                                  "n_samples": 3.54e4, "beta": 28.85, "rho": 0.326,
                                  "buffer_size": 706, "replay_bs": 168}),
    "blr_online_replay": ("blr_online_replay", {"sigma_mode": "laplace", "fisher_mode": "true",
                                                "fisher_batches": 30, "sigma_prior": 0.1512,
                                                "n_samples": 8.68e3, "beta": 31.10, "rho": 0.165,
                                                "buffer_size": 2083, "replay_bs": 77}),
}

# Shared BLR / replay bases for frontier & hybrid sweeps (curvature sigma-init +
# replay buffer). beta is swept by the caller; other knobs are the tuned operating point.
BLR_SIG_BASE: dict = {"sigma_mode": "laplace", "fisher_mode": "true", "fisher_batches": 30,
                      "sigma_prior": 0.05, "n_samples": 1.6e4, "rho": 0.06}
# Constant-sigma ablation base: IDENTICAL BLR dynamics to BLR_SIG_BASE (same sigma_prior,
# n_samples, rho) -- only the sigma INITIALISATION differs (flat sigma_const instead of the
# curvature-derived Laplace sigma). Sweeping beta over this vs BLR_SIG_BASE isolates the
# value of curvature-informed sigma init (the contribution). sigma_const = sigma_prior is
# the most-plastic flat init, i.e. generous to the ablation.
CONST_SIG_BASE: dict = {"sigma_mode": "const", "sigma_const": 0.05,
                        "sigma_prior": 0.05, "n_samples": 1.6e4, "rho": 0.06}
REPLAY_BASE: dict = {"buffer_size": 6598, "replay_bs": 100}
