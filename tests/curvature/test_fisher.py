"""Diagonal Fisher estimator: shapes/sign, determinism, and MC ~ expected agreement."""

from __future__ import annotations

import torch

from cl_experiments.curvature import diagonal_fisher


def test_fisher_shapes_and_sign(separable_loader, toy_model):
    torch.manual_seed(0)
    model = toy_model()
    fisher = diagonal_fisher(model, separable_loader(), mode="true")
    for n, p in model.named_parameters():
        assert fisher[n].shape == p.shape
        assert torch.all(fisher[n] >= 0)


def test_expected_fisher_deterministic_and_psd(separable_loader, toy_model):
    torch.manual_seed(0)
    model = toy_model()
    loader = separable_loader(n=256, shuffle=False)
    f1 = diagonal_fisher(model, loader, mode="expected")
    f2 = diagonal_fisher(model, loader, mode="expected")
    for n in f1:
        assert torch.allclose(f1[n], f2[n])       # no sampling -> deterministic
        assert (f1[n] >= 0).all()                 # Fisher is PSD


def test_mc_fisher_seeded_reproducible(separable_loader, toy_model):
    torch.manual_seed(0)
    model = toy_model()
    loader = separable_loader(n=256, shuffle=False)
    f1 = diagonal_fisher(model, loader, mode="true", generator=torch.Generator().manual_seed(7))
    f2 = diagonal_fisher(model, loader, mode="true", generator=torch.Generator().manual_seed(7))
    for n in f1:
        assert torch.allclose(f1[n], f2[n])       # same seed -> same estimate


def test_mc_and_expected_positively_correlated(separable_loader, toy_model):
    """MC and expected Fisher measure the same thing (directional check).

    The quantitative agreement (corr ~0.9+) needs scale/confidence and is
    validated on real MNIST in experiments/justification/validate_fisher.py.
    """
    torch.manual_seed(0)
    model = toy_model()
    loader = separable_loader(n=512, shuffle=False)
    exp = diagonal_fisher(model, loader, mode="expected")
    mc = diagonal_fisher(model, loader, mode="true", generator=torch.Generator().manual_seed(0))
    a = torch.cat([exp[n].flatten() for n in exp])
    b = torch.cat([mc[n].flatten() for n in mc])
    corr = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    assert corr > 0.3
