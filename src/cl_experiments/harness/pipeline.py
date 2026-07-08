"""Fixed-setting pipeline helpers.

`get_anchor` is the single entry point every downstream experiment uses to obtain
a trained, certified anchor for `(n, seed)`. Anchors are deterministic, so they
are trained ONCE and cached to `results/anchor/anchor_n{n}_seed{seed}.pt`; later
calls load the checkpoint instantly. This makes anchor training a one-time cost,
not something re-run per method comparison.

The permutation bank is shared (`make_permutations(MAX_N, seed)`), so the first
`n` tasks are identical across the `n`-sweep — anchors are nested and comparable.
"""

from __future__ import annotations

from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from cl_experiments.anchor import AnchorReport, train_anchor
from cl_experiments.data import anchor_loaders, make_permutations, permuted_loaders
from cl_experiments.models import MLP
from cl_experiments.repro import REPO_ROOT, set_seed

# One shared permutation bank of size ANCHOR_BANK + (max) stream length. The first
# `n` permutations are the anchor tasks; a FIXED slice past ANCHOR_BANK is the
# continuation stream, identical for every anchor size `n` -> comparable across
# the n-sweep. make_permutations is sequential+deterministic, so widening the bank
# never changes earlier permutations (cached anchors stay valid).
ANCHOR_BANK = 10  # max anchor tasks; stream tasks start at index ANCHOR_BANK
CACHE_DIR = REPO_ROOT / "results" / "anchor"


def anchor_perms(n: int, seed: int) -> list[torch.Tensor]:
    """The `n` permutations defining the anchor tasks for `(n, seed)`."""
    return make_permutations(ANCHOR_BANK, seed=seed)[:n]


def stream_perms(n_stream: int, seed: int) -> list[torch.Tensor]:
    """The `n_stream` continuation-task permutations (fixed slice past the anchor)."""
    return make_permutations(ANCHOR_BANK + n_stream, seed=seed)[ANCHOR_BANK:]


def stream_loaders(
    n_stream: int = 10, seed: int = 0, batch_size: int = 128
) -> list[tuple[DataLoader, DataLoader]]:
    """Per-task ``(train, test)`` loaders for the continuation stream.

    These are the tasks we fine-tune on after the anchor; the same set for any
    anchor size `n`.
    """
    return [permuted_loaders(p, batch_size, seed) for p in stream_perms(n_stream, seed)]


def get_anchor(
    n: int,
    seed: int = 0,
    *,
    device: torch.device | str = "cpu",
    force: bool = False,
    verbose: bool = True,
) -> tuple[MLP, AnchorReport, tuple[DataLoader, list[DataLoader]]]:
    """Return `(model, report, (train_loader, test_loaders))` for the anchor.

    Loads from cache if present (unless ``force``), else trains and caches.
    """
    perms = anchor_perms(n, seed)
    train_loader, test_loaders = anchor_loaders(perms, batch_size=128, seed=seed)
    ckpt = CACHE_DIR / f"anchor_n{n}_seed{seed}.pt"

    if ckpt.exists() and not force:
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model = MLP().to(device)
        model.load_state_dict(state["state_dict"])
        report = AnchorReport(**state["report"])
        if verbose:
            print(f"[anchor n={n} seed={seed}] loaded from cache; "
                  f"min_acc={report.min_task_acc:.4f} grad_norm={report.grad_norm:.2e} "
                  f"certified={report.certified()}")
        return model, report, (train_loader, test_loaders)

    set_seed(seed)
    model = MLP().to(device)
    model, report = train_anchor(model, train_loader, test_loaders, device=device, verbose=verbose)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "report": asdict(report)}, ckpt)
    if verbose:
        print(f"[anchor n={n} seed={seed}] trained & cached -> {ckpt.name}")
    return model, report, (train_loader, test_loaders)
