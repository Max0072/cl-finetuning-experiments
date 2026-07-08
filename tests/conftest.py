"""Shared fixtures: tiny synthetic datasets and models (no MNIST download).

Every test builds its data from these factories so there is one definition of the
toy tasks. ``separable_loader`` is a learnable k-way classification task (used by
anchor / BLR / Fisher tests); ``random_loader`` is unlearnable noise for shape /
plumbing tests; ``toy_model`` is a small MLP sized to match.
"""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from cl_experiments.models import MLP
from cl_experiments.repro import set_seed


@pytest.fixture(autouse=True)
def _deterministic_seed():
    """Seed python/numpy/torch before EVERY test so results never depend on
    collection order or a previous test's RNG draws. Tests that need a specific
    seed still set it themselves; this is the floor. (CPU/CUDA are bit-exact;
    MPS + torch.func.vmap are not -- the suite runs on CPU.)"""
    set_seed(0)


@pytest.fixture
def separable_loader():
    """Factory for a linearly-separable k-way toy task: ``y = argmax(x @ W)``.

    Inputs/weights are drawn from a seeded generator, so the data is reproducible
    independently of the global RNG (only the shuffle order follows it).
    """

    def make(n=512, d=20, k=4, seed=0, batch_size=64, shuffle=True):
        g = torch.Generator().manual_seed(seed)
        w = torch.randn(d, k, generator=g)
        x = torch.randn(n, d, generator=g)
        y = (x @ w).argmax(dim=1)
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle)

    return make


@pytest.fixture
def random_loader():
    """Factory for shape / plumbing tests: random inputs and labels (not learnable)."""

    def make(n=64, d=784, k=10, seed=0, batch_size=32):
        g = torch.Generator().manual_seed(seed)
        x = torch.randn(n, d, generator=g)
        y = torch.randint(0, k, (n,), generator=g)
        return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    return make


@pytest.fixture
def toy_model():
    """Factory for a small MLP sized to the toy tasks (default 20 -> 32 -> 4)."""

    def make(in_dim=20, hidden=(32,), out_dim=4):
        return MLP(in_dim=in_dim, hidden=hidden, out_dim=out_dim)

    return make
