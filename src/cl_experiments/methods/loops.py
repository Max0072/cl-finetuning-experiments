"""Training loops: plain SGD (anchor) and BLR continual fine-tuning (stream)."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.utils.data import DataLoader

from cl_experiments.methods.blr import BLR


def train_plain(
    model: nn.Module,
    loader: DataLoader,
    *,
    epochs: int = 3,
    lr: float = 1e-3,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Standard cross-entropy training to establish task-A weights."""
    model.train().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
    return model


def train_blr(
    model: nn.Module,
    opt: BLR,
    loader: DataLoader,
    *,
    epochs: int = 1,
    device: torch.device | str = "cpu",
    on_batch: Callable[[int], None] | None = None,
) -> None:
    """Continual fine-tuning with the BLR update.

    ``criterion`` uses mean reduction, matching C = E_q[mean-NLL]; the memory
    window N (in ``opt``) carries the data-scale factor for the prior terms.
    """
    model.train().to(device)
    opt.to(device)
    criterion = nn.CrossEntropyLoss()
    step = 0
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_accum()
            for _ in range(opt.n_mc):
                opt.zero_grad()
                opt.sample()
                loss = criterion(model(x), y)
                loss.backward()
                opt.accumulate()
            opt.step()
            if on_batch is not None:
                on_batch(step)
            step += 1
