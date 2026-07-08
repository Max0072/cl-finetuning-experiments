"""One-pass diagonal curvature estimation.

We approximate the local geometry of the "admissible zone" of a trained model
with the diagonal of the Fisher information. This is the cheap, non-iterative
alternative to fitting variational sigmas: a single sweep over the data
accumulates the mean squared gradient of the per-sample log-likelihood.

Three modes (all return the mean per-sample diagonal Fisher):
  * ``empirical``: use the dataset labels. Cheap (one pass), biased away from a
                   minimum -- an ablation.
  * ``true``:      MC estimate -- sample one label per example from the model's
                   predictive distribution. O(1) in the number of classes, so it
                   is the scalable choice for large output spaces.
                   Seeded via ``generator`` for reproducibility.
  * ``expected``:  deterministic true Fisher -- take the expectation over labels
                   exactly, ``sum_c p(c|x) (d log p(c|x)/dtheta)^2``. Exact, no
                   sampling noise, but costs one pass PER class (O(C)); use for
                   small C and as the reference that validates the MC estimate.

Per-sample gradients are computed with ``torch.func.vmap(grad(...))`` so a whole
batch is differentiated in one vectorised pass rather than a Python loop.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader

FisherMode = Literal["empirical", "true", "expected"]


def diagonal_fisher(
    model: nn.Module,
    loader: DataLoader,
    *,
    mode: FisherMode = "true",
    max_batches: int | None = None,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Return the mean per-parameter diagonal Fisher, keyed by parameter name.

    Large ``F_ii`` => sharp direction => important weight => small sigma.
    """
    model.eval().to(device)
    # Differentiate only w.r.t. trainable params; frozen ones (e.g. a LoRA
    # backbone) enter the forward as constants.
    trainable = {n: p.detach() for n, p in model.named_parameters() if p.requires_grad}
    frozen = {n: p.detach() for n, p in model.named_parameters() if not p.requires_grad}
    buffers = {n: b.detach() for n, b in model.named_buffers()}

    def sample_nll(params: dict[str, torch.Tensor], x: torch.Tensor, target: torch.Tensor):
        logits = functional_call(model, ({**params, **frozen}, buffers), (x.unsqueeze(0),))
        return torch.nn.functional.cross_entropy(logits, target.unsqueeze(0))

    per_sample_grad = vmap(grad(sample_nll), in_dims=(None, 0, 0))

    fisher = {n: torch.zeros_like(p) for n, p in trainable.items()}
    n_seen = 0

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device)

        if mode == "expected":
            # Exact expectation over labels: sum_c p(c|x) * grad(log p(c|x))^2.
            with torch.no_grad():
                probs = torch.softmax(model(x), dim=1)  # (B, C)
            n_classes = probs.shape[1]
            for c in range(n_classes):
                target_c = torch.full((x.size(0),), c, device=device, dtype=torch.long)
                grads = per_sample_grad(trainable, x, target_c)
                w = probs[:, c]  # (B,)
                for n in fisher:
                    g2 = grads[n] ** 2
                    fisher[n] += (w.view(-1, *([1] * (g2.dim() - 1))) * g2).sum(dim=0)
        else:
            if mode == "true":
                with torch.no_grad():
                    probs = torch.softmax(model(x), dim=1)
                    # Robustness: if the model has diverged (e.g. a too-large BLR step in a
                    # bad tuning trial), logits can be nan/inf -> probs invalid and
                    # multinomial crashes. Sanitise to a valid distribution so a diverged
                    # run still returns a (bad) score instead of killing the whole sweep.
                    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                    bad = probs.sum(dim=1) <= 0
                    if bad.any():
                        probs[bad] = 1.0 / probs.shape[1]  # uniform fallback for dead rows
                    if generator is not None:  # reproducible: sample on CPU
                        target = torch.multinomial(
                            probs.cpu(), 1, generator=generator
                        ).squeeze(1).to(device)
                    else:
                        target = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:  # empirical
                target = y.to(device)
            grads = per_sample_grad(trainable, x, target)
            for n in fisher:
                fisher[n] += (grads[n] ** 2).sum(dim=0)

        n_seen += x.size(0)

    return {n: f / max(n_seen, 1) for n, f in fisher.items()}
