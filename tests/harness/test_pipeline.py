"""Pipeline data-plumbing tests: anchor/stream permutation banks + path anchoring."""

from __future__ import annotations

import torch

from cl_experiments.data.permuted_mnist import DATA_ROOT
from cl_experiments.harness import ANCHOR_BANK, anchor_perms, stream_perms
from cl_experiments.harness.pipeline import CACHE_DIR
from cl_experiments.repro import REPO_ROOT


def test_paths_anchored_to_repo_root():
    """Regression guard: DATA_ROOT / CACHE_DIR must resolve under the REAL repo root
    (the one holding pyproject.toml), not some nested dir. Moving a module to a
    different folder depth silently broke `parents[N]`-based paths before -- this
    catches that class of bug."""
    assert (REPO_ROOT / "pyproject.toml").exists()
    assert DATA_ROOT == REPO_ROOT / "datasets"
    assert CACHE_DIR == REPO_ROOT / "results" / "anchor"


def test_anchor_perms_nested_and_deterministic():
    # First permutation is identity; anchors are nested and reproducible.
    a3 = anchor_perms(3, seed=0)
    a5 = anchor_perms(5, seed=0)
    assert len(a3) == 3 and len(a5) == 5
    assert torch.equal(a3[0], torch.arange(784))  # task 0 = identity
    for i in range(3):
        assert torch.equal(a3[i], a5[i])  # nested: same first tasks


def test_stream_disjoint_and_fixed_across_n():
    # Stream is the fixed slice past the anchor bank -> same regardless of anchor n.
    s = stream_perms(4, seed=0)
    assert len(s) == 4
    # not the identity, and not among the anchor tasks
    anchors = anchor_perms(ANCHOR_BANK, seed=0)
    for sp in s:
        assert not torch.equal(sp, torch.arange(784))
        assert not any(torch.equal(sp, ap) for ap in anchors)


def test_stream_reproducible():
    a = stream_perms(3, seed=1)
    b = stream_perms(3, seed=1)
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)
