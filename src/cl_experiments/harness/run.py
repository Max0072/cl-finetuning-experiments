"""End-to-end harness: anchor -> stream fine-tuning -> metrics.

Ties the fixed setting together. `run_experiment` loads (or trains) the cached
anchor for `(n, seed)`, builds the chosen method, streams through the
continuation tasks while recording the accuracy matrix, and returns the CL
metrics.
"""

from __future__ import annotations

import copy

import torch
from torch import nn

from cl_experiments.harness.pipeline import get_anchor, stream_loaders
from cl_experiments.methods import build_learner
from cl_experiments.metrics import CLMetrics, compute_metrics, evaluate_tasks
from cl_experiments.repro import set_seed


def run_stream(
    model: nn.Module,
    learner,
    stream_train_loaders: list,
    all_test_loaders: list,
    device,
) -> list[list[float]]:
    """Fine-tune across the stream, recording accuracy on ALL tasks after each step.

    Row 0 = post-anchor; rows 1..N = after each stream task.
    """
    R = [evaluate_tasks(model, all_test_loaders, device)]
    for train_loader in stream_train_loaders:
        learner.finetune(model, train_loader, device)
        R.append(evaluate_tasks(model, all_test_loaders, device))
    return R


def run_experiment(
    method: str,
    n_anchor: int,
    *,
    seed: int = 0,
    n_stream: int = 10,
    device: torch.device | str = "cpu",
    method_kwargs: dict | None = None,
) -> tuple[CLMetrics, list[list[float]]]:
    """Run one (method, n_anchor, seed) experiment; return (metrics, accuracy matrix)."""
    set_seed(seed)  # reproducible data order, BLR MC sampling, replay sampling
    anchor_model, _, (anchor_train, anchor_tests) = get_anchor(
        n_anchor, seed, device=device, verbose=False
    )
    stream = stream_loaders(n_stream=n_stream, seed=seed)
    stream_trains = [tr for tr, _ in stream]
    all_tests = anchor_tests + [te for _, te in stream]

    model = copy.deepcopy(anchor_model)  # don't mutate the cached anchor
    learner = build_learner(method, model, anchor_train, device, **(method_kwargs or {}))
    R = run_stream(model, learner, stream_trains, all_tests, device)
    return compute_metrics(R, n_anchor=n_anchor), R
