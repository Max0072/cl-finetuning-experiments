"""Reproducibility-utility tests."""

from __future__ import annotations

import json

import torch

from cl_experiments.repro import Timer, run_manifest, set_seed


def test_set_seed_reproducible():
    set_seed(0)
    a = torch.randn(8)
    set_seed(0)
    b = torch.randn(8)
    assert torch.equal(a, b)


def test_different_seed_differs():
    set_seed(0)
    a = torch.randn(8)
    set_seed(1)
    b = torch.randn(8)
    assert not torch.equal(a, b)


def test_manifest_written(tmp_path):
    p = tmp_path / "m.json"
    m = run_manifest(p, config={"x": 1, "y": "z"}, seed=3, elapsed_s=1.5)
    assert p.exists()
    on_disk = json.loads(p.read_text())
    assert on_disk["seed"] == 3
    assert on_disk["config"] == {"x": 1, "y": "z"}
    assert "git_commit" in on_disk and "torch_version" in on_disk and "timestamp_utc" in on_disk
    assert m["elapsed_s"] == 1.5


def test_timer():
    with Timer() as t:
        _ = sum(range(1000))
    assert t.elapsed >= 0.0
