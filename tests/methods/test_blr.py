"""BLR optimiser mechanics: it fits the toy task and protects low-sigma weights."""

from __future__ import annotations

import torch
from torch import nn

from cl_experiments.methods import BLR, train_blr, train_plain


def test_blr_reduces_loss(separable_loader, toy_model):
    """The canonical Bayesian Learning Rule fits the toy task."""
    torch.manual_seed(0)
    model = toy_model()
    loader = separable_loader()
    opt = BLR(model, sigma_init=0.05, sigma_prior=0.1, lr=1.0)
    criterion = nn.CrossEntropyLoss()

    def full_loss() -> float:
        model.eval()
        with torch.no_grad():
            return sum(criterion(model(x), y).item() for x, y in loader) / len(loader)

    before = full_loss()
    train_blr(model, opt, loader, epochs=15)
    after = full_loss()
    assert after < before, f"BLR did not reduce loss: {before:.3f} -> {after:.3f}"


def test_blr_protects_important_weights(separable_loader, toy_model):
    """Weights with tiny sigma should barely move; large-sigma weights move more."""
    torch.manual_seed(0)
    model = toy_model()
    loader = separable_loader()
    train_plain(model, loader, epochs=5)

    # Hand-craft sigma: tiny on layer-0 weight (important), large on layer-2 (free).
    sigma = {n: torch.full_like(p, 0.1) for n, p in model.named_parameters()}
    sigma["net.0.weight"] = torch.full_like(model.net[0].weight, 1e-5)
    mu_before = {n: p.detach().clone() for n, p in model.named_parameters()}

    opt = BLR(model, sigma_init=sigma, sigma_prior=0.1, n_samples=512.0)
    train_blr(model, opt, loader, epochs=5)

    frozen_shift = (opt.mu["net.0.weight"] - mu_before["net.0.weight"]).abs().mean()
    free_shift = (opt.mu["net.2.weight"] - mu_before["net.2.weight"]).abs().mean()
    assert frozen_shift < free_shift
