"""Data & the continual-learning scenario (Permuted-MNIST).

The task stream and its loaders. See ``permuted_mnist`` for the protocol.
"""

from cl_experiments.data.permuted_mnist import (
    DATA_ROOT,
    anchor_loaders,
    make_permutations,
    permuted_loaders,
    split_loaders,
)

__all__ = [
    "DATA_ROOT",
    "anchor_loaders",
    "make_permutations",
    "permuted_loaders",
    "split_loaders",
]
