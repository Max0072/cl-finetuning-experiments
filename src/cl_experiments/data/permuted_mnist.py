"""MNIST data plumbing for continual-learning experiments.

The Permuted-MNIST protocol: each "task" applies a fixed random permutation of
the 784 input pixels. Task 0 is the identity permutation (plain MNIST) and is the
data we do not want to forget; later tasks are new distributions to adapt to.
The label space (10 digits) and the network head are shared across tasks, so
forgetting is measured cleanly on a single head.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from cl_experiments.repro import REPO_ROOT

DATA_ROOT = REPO_ROOT / "datasets"


def _load_mnist(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ds = datasets.MNIST(
        root=str(DATA_ROOT),
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )
    x = ds.data.float().div_(255.0).view(-1, 784)
    # Standardise with the canonical MNIST mean/std.
    x = (x - 0.1307) / 0.3081
    y = ds.targets.clone()
    return x, y


def make_permutations(n_tasks: int, seed: int = 0) -> list[torch.Tensor]:
    """Return one pixel permutation per task; task 0 is the identity."""
    g = torch.Generator().manual_seed(seed)
    perms = [torch.arange(784)]
    for _ in range(1, n_tasks):
        perms.append(torch.randperm(784, generator=g))
    return perms


def split_loaders(
    classes: list[int],
    batch_size: int = 128,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Train/test loaders restricted to ``classes``, labels kept in 0..9.

    For class-incremental Split-MNIST: the base learns one class subset, the
    adapter another, sharing a single 10-way head.
    """
    xtr, ytr = _load_mnist(train=True)
    xte, yte = _load_mnist(train=False)
    cls = torch.tensor(classes)
    m_tr = torch.isin(ytr, cls)
    m_te = torch.isin(yte, cls)
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(
        TensorDataset(xtr[m_tr], ytr[m_tr]), batch_size=batch_size, shuffle=True, generator=g
    )
    test = DataLoader(TensorDataset(xte[m_te], yte[m_te]), batch_size=512, shuffle=False)
    return train, test


def anchor_loaders(
    perms: list[torch.Tensor],
    batch_size: int = 128,
    seed: int = 0,
) -> tuple[DataLoader, list[DataLoader]]:
    """Joint multi-task anchor training set over all ``perms`` + per-task test loaders.

    The training set is the union of the (identically-labelled) permuted copies,
    shuffled together -> balanced joint multi-task training with a shared head.
    Returns ``(train_loader, [test_loader_per_task])``.
    """
    xtr, ytr = _load_mnist(train=True)
    xte, yte = _load_mnist(train=False)
    x_joint = torch.cat([xtr[:, p] for p in perms])
    y_joint = ytr.repeat(len(perms))
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(
        TensorDataset(x_joint, y_joint), batch_size=batch_size, shuffle=True, generator=g
    )
    tests = [
        DataLoader(TensorDataset(xte[:, p], yte), batch_size=512, shuffle=False) for p in perms
    ]
    return train, tests


def permuted_loaders(
    perm: torch.Tensor,
    batch_size: int = 128,
    seed: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Train/test loaders for one permuted-MNIST task."""
    xtr, ytr = _load_mnist(train=True)
    xte, yte = _load_mnist(train=False)
    xtr, xte = xtr[:, perm], xte[:, perm]
    g = torch.Generator().manual_seed(seed)
    train = DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True, generator=g)
    test = DataLoader(TensorDataset(xte, yte), batch_size=512, shuffle=False)
    return train, test
