"""A plain multi-layer perceptron for MNIST-scale experiments.

Deliberately a vanilla ``nn.Module`` (no Bayesian layers): the BLR optimiser
wraps its parameters as the posterior means ``mu`` and keeps ``sigma`` as
optimiser state, so the model itself stays a standard, checkpoint-compatible net.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 784,
        hidden: Sequence[int] = (256, 256),
        out_dim: int = 10,
    ) -> None:
        super().__init__()
        dims = [in_dim, *hidden, out_dim]
        layers: list[nn.Module] = []
        for i, (a, b) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            layers.append(nn.Linear(a, b))
            if i < len(dims) - 2:  # no activation after the final layer
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))
