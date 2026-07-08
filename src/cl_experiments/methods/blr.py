"""Gaussian mean-field Bayesian update for continual fine-tuning: the canonical
**Bayesian Learning Rule** (BLR; Khan & Rue 2021; VON/Vadam family).

The mean step is the natural gradient (scaled by the posterior variance), and the
variance tracks the inverse curvature via a precision EMA toward the Laplace target
(``N*H + 1/sigma_prior^2``), where ``H`` is the diagonal Hessian obtained from
Bonnet's identity ``E[grad*eps]/sigma`` (PSD-clamped). The mean step-size ``lr`` IS
the BLR step-size beta.

The update uses reparameterised gradients over ``omega = mu + eps * sigma``:
    dC/dmu = E_eps[dL/domega],   dC/dsigma = E_eps[dL/domega * eps].
``sigma^2`` is the per-weight adaptive step (low-uncertainty weights move little);
the prior pulls ``mu`` toward ``mu_prior``. Curvature-informed sigma INITIALISATION
(Laplace / Fisher) is our contribution and plugs into this update.

This optimiser leaves the model a plain ``nn.Module``: the posterior means live in
``model.parameters()`` (so checkpoints stay standard) and the sigmas live in
optimiser state. Each step samples the weights in place, so use it as:

    opt.zero_accum()
    for _ in range(n_mc):
        opt.sample()                    # writes omega into the model
        loss = criterion(model(x), y)
        loss.backward()
        opt.accumulate()                # folds grads into the MC estimators
    opt.step()                          # applies the BLR update, restores mu
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

MuPrior = Literal["init", "zero"]


class BLR:
    def __init__(
        self,
        model: nn.Module,
        *,
        sigma_init: dict[str, torch.Tensor] | float,
        sigma_prior: float = 0.01,
        n_samples: float = 1e5,
        n_mc: int = 1,
        mu_prior: MuPrior = "init",
        sigma_min: float = 1e-6,
        lr: float = 1.0,
        noise_scale: float = 1.0,
        rho: float = 0.1,
    ) -> None:
        self.model = model
        self.sigma_prior = float(sigma_prior)
        self.N = float(n_samples)
        self.n_mc = int(n_mc)
        self.sigma_min = float(sigma_min)
        self.rho = float(rho)  # precision EMA rate
        # Sampling noise scale kappa: sample omega = mu + eps*(kappa*sigma) but keep the
        # step scale sigma^2. kappa<1 decouples forward noise from the step -> more
        # plasticity at less corruption (heuristic; sigma-update stays as-is).
        self.noise_scale = float(noise_scale)
        self.lr = float(lr)  # BLR mean step-size beta
        # Optional per-weight precision FLOOR for the sigma-EMA target. Default None ->
        # the flat scalar prior 1/sigma_prior^2 (canonical, unchanged). online-BLR sets
        # this to the accumulated precision (N_data*running_Fisher + prior) so the
        # within-task EMA relaxes toward the RIGHT Laplace target (accumulated importance
        # + current curvature), instead of drifting above it. Keyed by param name.
        self.precision_floor: dict[str, torch.Tensor] | None = None

        # Only manage trainable parameters -- so BLR can wrap a LoRA model and
        # touch just the adapter, leaving the frozen backbone alone.
        self._params = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.mu = {n: p.detach().clone() for n, p in self._params.items()}
        self.mu_prior = {
            n: (m.clone() if mu_prior == "init" else torch.zeros_like(m))
            for n, m in self.mu.items()
        }
        if isinstance(sigma_init, float):
            self.sigma = {n: torch.full_like(m, sigma_init) for n, m in self.mu.items()}
        else:
            self.sigma = {n: sigma_init[n].detach().clone() for n in self.mu}

        self._eps: dict[str, torch.Tensor] = {}
        self._g_mu: dict[str, torch.Tensor] = {}
        self._g_sigma: dict[str, torch.Tensor] = {}
        self.zero_accum()

    # -- gradient accumulation over Monte-Carlo weight samples ---------------

    def zero_accum(self) -> None:
        self._g_mu = {n: torch.zeros_like(m) for n, m in self.mu.items()}
        self._g_sigma = {n: torch.zeros_like(m) for n, m in self.mu.items()}

    @torch.no_grad()
    def sample(self) -> None:
        """Draw omega = mu + eps * sigma and write it into the live model."""
        for n, p in self._params.items():
            eps = torch.randn_like(p)
            self._eps[n] = eps
            p.copy_(self.mu[n] + eps * (self.noise_scale * self.sigma[n]))

    @torch.no_grad()
    def accumulate(self) -> None:
        """Fold the current sample's grads into the MC estimators of dC/d{mu,sigma}."""
        for n, p in self._params.items():
            if p.grad is None:
                continue
            g = p.grad.detach()
            self._g_mu[n] += g
            self._g_sigma[n] += g * self._eps[n]

    @torch.no_grad()
    def step(self) -> None:
        inv = 1.0 / self.n_mc
        denom = self.N * self.sigma_prior**2
        s_prior = 1.0 / self.sigma_prior**2
        for n in self.mu:
            mu, sigma = self.mu[n], self.sigma[n]
            var = sigma**2
            dC_dmu = self._g_mu[n] * inv
            dC_dsigma = self._g_sigma[n] * inv

            # Variance update: a precision EMA toward the Laplace target
            # (N*H + floor), H the diagonal Hessian via Bonnet's identity
            # E[grad*eps]/sigma (PSD-clamped). The floor is the flat prior s_prior
            # (canonical) or, for online-BLR, the accumulated precision so the EMA
            # relaxes toward accumulated-importance + current-curvature. Sigma always
            # consolidates -- there is no "freeze" mode.
            floor = s_prior if self.precision_floor is None else self.precision_floor[n]
            hess = (dC_dsigma / sigma).clamp_min(0.0)
            s_new = (1.0 - self.rho) * (1.0 / var) + self.rho * (self.N * hess + floor)
            sigma = (1.0 / s_new).sqrt()
            self.sigma[n] = sigma.clamp_min(self.sigma_min)
            var = sigma**2
            # Mean step: the natural gradient lr*sigma^2*grad plus a gentle prior
            # pull (scaled by 1/(N sigma_prior^2)). lr here IS the step-size beta.
            self.mu[n] = mu - self.lr * var * dC_dmu + (var / denom) * (self.mu_prior[n] - mu)

        # Restore the model to its posterior mean (for eval / next sample base).
        for n, p in self._params.items():
            p.copy_(self.mu[n])
        self.zero_accum()

    def zero_grad(self) -> None:
        self.model.zero_grad(set_to_none=True)

    # -- convenience ---------------------------------------------------------

    def to(self, device: torch.device | str) -> BLR:
        for d in (self.mu, self.sigma, self.mu_prior):
            for n in d:
                d[n] = d[n].to(device)
        return self
