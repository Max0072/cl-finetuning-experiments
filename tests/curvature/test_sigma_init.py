"""Sigma initialisation: the Laplace closed form and inverse-to-curvature calibration."""

from __future__ import annotations

import math

import torch

from cl_experiments.curvature import laplace_sigma, sigma_from_curvature


def test_laplace_sigma_form():
    sp = 0.1
    F = {"a": torch.tensor([0.0, 1.0, 100.0])}
    s = laplace_sigma(F, n_data=1000, sigma_prior=sp)["a"]
    assert abs(s[0].item() - sp) < 1e-4          # F=0 -> sigma = sigma_prior (ceiling)
    assert s[0] > s[1] > s[2]                     # larger F -> smaller sigma
    exact = 1.0 / math.sqrt(1000 * 1.0 + 1.0 / sp**2)
    assert abs(s[1].item() - exact) < 1e-6        # matches 1/sqrt(N F + 1/sp^2)


def test_sigma_inverse_to_curvature():
    curv = {"a": torch.tensor([1e-6, 1.0, 100.0])}
    sigma = sigma_from_curvature(curv, sigma_ref=0.01, ref_quantile=0.5)["a"]
    # larger curvature -> smaller sigma
    assert sigma[0] > sigma[1] > sigma[2]
