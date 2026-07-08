"""Anchor training: learns the task and certifies (accuracy-based), and stops at
the moderate loss target before the epoch cap (docs/SETTING.md)."""

from __future__ import annotations

import torch

from cl_experiments.anchor import train_anchor


def test_anchor_learns_and_certifies(separable_loader, toy_model):
    torch.manual_seed(0)
    loader = separable_loader(n=400)
    model = toy_model(hidden=(64, 64))
    _, report = train_anchor(
        model, loader, [loader], stop_loss=0.05, max_epochs=200, device="cpu", verbose=False
    )
    # separable-ish toy: should learn the task well (certification is acc-based)
    assert report.min_task_acc > 0.9
    assert report.certified()
    assert report.epochs_run <= 200
    assert len(report.loss_history) == report.epochs_run


def test_stops_at_target_loss(separable_loader, toy_model):
    """Training stops once the moderate loss target is reached, before the cap."""
    torch.manual_seed(0)
    loader = separable_loader(n=200)
    model = toy_model(hidden=(64,))
    _, report = train_anchor(
        model, loader, [loader], stop_loss=0.1, max_epochs=300, device="cpu", verbose=False
    )
    assert report.epochs_run < 300
    assert report.final_train_loss <= 0.1 + 1e-3
