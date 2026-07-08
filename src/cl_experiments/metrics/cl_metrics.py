"""Continual-learning metrics for the fixed setting (step 4 of the pipeline).

Evaluation timeline: the anchor is trained JOINTLY on tasks ``0..n_anchor-1`` (all
"known" at t=0), then we fine-tune on a stream of tasks one at a time. After the
anchor and after each stream task we evaluate on **every** task, producing an
accuracy matrix

    R[i][j] = test accuracy on task j after learning event i

with rows i = 0 (post-anchor), 1..N (post each stream task) and columns
j = 0..T-1 over all T = n_anchor + N tasks, ordered [anchor tasks, stream tasks].

Each task's "learned_at" event: anchor tasks -> 0; stream task k -> k+1.

Metrics (generalising Lopez-Paz & Ranzato to the joint-anchor layout):
  * ACC              final average accuracy over all tasks (mean of last row)
  * BWT              mean over tasks not learned last of R[-1][j] - R[learned][j]
  * learning_acc     mean accuracy on each task right when it was learned (plasticity)
  * forgetting       mean over tasks of (max accuracy seen) - (final accuracy)
  * anchor_final     final accuracy averaged over the anchor tasks (the model's
                     preserved knowledge)
  * anchor_forgetting  mean drop on the anchor tasks vs the post-anchor row
  * stream_final     final accuracy averaged over the stream tasks
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from cl_experiments.metrics.eval import accuracy


@torch.no_grad()
def evaluate_tasks(
    model: nn.Module, test_loaders: list[DataLoader], device: torch.device | str = "cpu"
) -> list[float]:
    """One row of R: accuracy on every task's test set."""
    return [accuracy(model, t, device) for t in test_loaders]


@dataclass
class CLMetrics:
    acc: float
    bwt: float
    learning_acc: float
    forgetting: float
    anchor_final: float
    anchor_forgetting: float
    stream_final: float


def _learned_at(n_anchor: int, n_tasks: int) -> list[int]:
    # anchor tasks learned at event 0; stream task k (column n_anchor+k) at event k+1
    return [0 if j < n_anchor else (j - n_anchor + 1) for j in range(n_tasks)]


def compute_metrics(R: list[list[float]] | torch.Tensor, n_anchor: int) -> CLMetrics:
    """Compute CL metrics from an accuracy matrix ``R`` (rows = events, cols = tasks).

    Requires ``R`` to have ``N + 1`` rows (post-anchor + one per stream task) and
    ``T = n_anchor + N`` columns.
    """
    R = torch.as_tensor(R, dtype=torch.double)
    n_events, n_tasks = R.shape
    last = n_events - 1  # index of the final event
    la = _learned_at(n_anchor, n_tasks)

    acc = R[last].mean().item()

    # BWT / learning_acc / forgetting over tasks (exclude tasks learned at the last event)
    bwt, learn, forget = [], [], []
    for j in range(n_tasks):
        learn.append(R[la[j], j].item())
        if la[j] < last:
            bwt.append((R[last, j] - R[la[j], j]).item())
            peak = R[la[j] : last + 1, j].max().item()
            forget.append(peak - R[last, j].item())
    bwt_v = sum(bwt) / len(bwt) if bwt else 0.0
    forget_v = sum(forget) / len(forget) if forget else 0.0
    learn_v = sum(learn) / len(learn)

    anchor_final = R[last, :n_anchor].mean().item()
    anchor_forgetting = (R[last, :n_anchor] - R[0, :n_anchor]).mean().item()
    stream_final = R[last, n_anchor:].mean().item() if n_tasks > n_anchor else float("nan")

    return CLMetrics(
        acc=acc,
        bwt=bwt_v,
        learning_acc=learn_v,
        forgetting=forget_v,
        anchor_final=anchor_final,
        anchor_forgetting=anchor_forgetting,
        stream_final=stream_final,
    )
