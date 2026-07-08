"""Map a diagonal curvature estimate to initial posterior standard deviations.

The admissible zone is a diagonal ellipsoid; its half-widths are the sigmas.
Sharp directions (large curvature ``C``) get small sigma -> protected during
BLR fine-tuning; flat directions get large sigma -> free to move.

Near a minimum the raw Fisher is tiny in absolute terms, so a fixed
``sigma^2 = tau / (C + lam)`` saturates and throws away all the relative
structure. Instead we calibrate against a reference curvature quantile so only
the *relative* importance matters (scale-invariant to the Fisher magnitude):

    sigma_i = sigma_ref * sqrt( (C_ref + lam) / (C_i + lam) )

  * ``C_ref`` = the ``ref_quantile`` quantile of the curvature -> weights at that
    quantile get ``sigma_ref``; sharper weights get less, flatter weights get more.
  * ``sigma_ref`` sets the overall plasticity scale (natural choice: sigma_prior).
  * ``lam`` is a floor so near-zero-curvature weights don't blow up.
  * result is clamped to ``[sigma_min, sigma_max]``.
"""

from __future__ import annotations

import torch


def laplace_sigma(
    fisher: dict[str, torch.Tensor],
    n_data: int,
    sigma_prior: float,
    *,
    sigma_min: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Canonical diagonal-Laplace posterior std from the mean Fisher.

        sigma_i^2 = 1 / ( N * F_i  +  1 / sigma_prior^2 )
                        \\_______/     \\____________/
                         data term       prior term

    * ``fisher`` is the MEAN per-sample diagonal Fisher (as returned by
      ``diagonal_fisher``); ``n_data`` scales it back to the total evidence
      ``N * mean(F)`` = the precision contributed by all N anchor examples.
    * ``sigma_prior`` is the prior std: the resting freedom of a weight the data
      does not constrain (``F_i -> 0`` gives ``sigma_i -> sigma_prior``), and it
      keeps sigma finite for flat directions. It is the same prior the BLR update
      relaxes toward. This is the one genuine free hyperparameter (set by marginal
      likelihood, or tuned on the CL metric).

    ``sigma_prior`` is the natural ceiling (sigma_i <= sigma_prior always), so no
    separate sigma_max is needed.
    """
    prior_prec = 1.0 / sigma_prior**2
    return {
        name: (1.0 / (n_data * f + prior_prec)).clamp_min(sigma_min**2).sqrt()
        for name, f in fisher.items()
    }


def sigma_from_curvature(
    curvature: dict[str, torch.Tensor],
    *,
    sigma_ref: float = 0.01,
    ref_quantile: float = 0.5,
    lam: float = 1e-8,
    sigma_min: float = 1e-5,
    sigma_max: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return per-parameter initial sigma from a diagonal curvature dict."""
    all_c = torch.cat([c.flatten() for c in curvature.values()])
    c_ref = torch.quantile(all_c, ref_quantile).item()

    sigmas: dict[str, torch.Tensor] = {}
    for name, c in curvature.items():
        sigma = sigma_ref * ((c_ref + lam) / (c + lam)).sqrt()
        sigmas[name] = sigma.clamp(sigma_min, sigma_max)
    return sigmas
