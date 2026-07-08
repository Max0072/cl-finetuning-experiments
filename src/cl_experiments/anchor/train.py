"""Anchor training: produce the trained model whose knowledge we later preserve.

Per docs/SETTING.md, the anchor is trained JOINTLY on the first ``n`` permuted
tasks (single shared head), to CONVERGENCE (loss plateau, not a fixed epoch
count), with a plain optimiser (Adam, weight decay 0). The method's whole
importance/protection story assumes the anchor sits at a near-minimum -- so we
train until the training loss stops improving and then CERTIFY the anchor:
per-task accuracy (must be high & balanced) and the gradient norm on the
preserved data (must be small, i.e. we really are at a minimum).

``train_anchor`` returns the trained model and a certification report; nothing
here is Bayesian -- sigmas/BLR enter only downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader

from cl_experiments.metrics.eval import accuracy


@dataclass
class AnchorReport:
    epochs_run: int
    final_train_loss: float
    grad_norm: float
    per_task_test_acc: list[float]
    loss_history: list[float] = field(default_factory=list)

    @property
    def min_task_acc(self) -> float:
        return min(self.per_task_test_acc) if self.per_task_test_acc else 0.0

    def certified(self, acc_threshold: float = 0.95) -> bool:
        """A usable anchor: every task learned well.

        We deliberately do NOT gate on a tiny gradient norm: driving the loss to
        ~0 (grad -> 0) overconfidently saturates the softmax and COLLAPSES the
        Fisher, which is what the importance signal comes from. So we converge to
        moderate loss (informative Fisher) and only require the tasks to be
        learned. ``grad_norm`` is reported as a diagnostic. See
        docs/SETTING.md and the anchor-convergence note.
        """
        return self.min_task_acc >= acc_threshold


@torch.enable_grad()
def _grad_norm(model: nn.Module, loader: DataLoader, device) -> float:
    """||(1/N) sum_i grad(loss_i)|| over the full preserved set.

    Small => the anchor is at a minimum of the preserved loss (the assumption the
    second-order importance picture relies on).
    """
    model.eval()
    model.zero_grad(set_to_none=True)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        criterion(model(x), y).backward()
        n += y.size(0)
    sq = sum((p.grad**2).sum().item() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    return sq**0.5 / max(n, 1)


def train_anchor(
    model: nn.Module,
    train_loader: DataLoader,
    test_loaders: list[DataLoader],
    *,
    lr: float = 1e-3,
    stop_loss: float = 0.03,
    max_epochs: int = 60,
    device: torch.device | str = "cpu",
    verbose: bool = True,
) -> tuple[nn.Module, AnchorReport]:
    """Joint multi-task training to MODERATE convergence, then certify.

    Adam at constant lr; stop when the mean train loss falls to ``stop_loss`` (or
    ``max_epochs``). We deliberately stop at moderate loss rather than driving it
    to ~0: over-training saturates the softmax and collapses the Fisher (the
    importance signal), so a moderately-converged anchor -- learned all tasks but
    not overconfident -- keeps the curvature informative. ``stop_loss`` is the
    single knob controlling how far from saturation we stop.
    """
    model.train().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    history: list[float] = []
    epochs_run = 0
    for epoch in range(max_epochs):
        model.train()
        running, nb = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item()
            nb += 1
        epoch_loss = running / max(nb, 1)
        history.append(epoch_loss)
        epochs_run = epoch + 1
        if verbose:
            print(f"  epoch {epoch + 1:3d}  train_loss={epoch_loss:.5f}")
        if epoch_loss <= stop_loss:
            break

    per_task = [accuracy(model, t, device) for t in test_loaders]
    gnorm = _grad_norm(model, train_loader, device)
    report = AnchorReport(
        epochs_run=epochs_run,
        final_train_loss=history[-1],
        grad_norm=gnorm,
        per_task_test_acc=per_task,
        loss_history=history,
    )
    if verbose:
        print(f"  [anchor] epochs={epochs_run} loss={report.final_train_loss:.5f} "
              f"grad_norm={gnorm:.2e} min_task_acc={report.min_task_acc:.4f} "
              f"certified={report.certified()}")
    return model, report
