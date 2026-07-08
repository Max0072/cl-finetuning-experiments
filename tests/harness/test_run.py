"""Harness driver: run_stream builds a well-formed accuracy matrix (tiny synthetic data)."""

from __future__ import annotations

from cl_experiments.harness import run_stream
from cl_experiments.methods import NaiveLearner
from cl_experiments.models import MLP


def test_run_stream_matrix_shape(random_loader):
    model = MLP()
    n_stream, n_tasks = 3, 5  # 5 test loaders total (e.g. 2 anchor + 3 stream)
    all_tests = [random_loader(seed=i) for i in range(n_tasks)]
    stream_trains = [random_loader(seed=100 + i) for i in range(n_stream)]
    R = run_stream(model, NaiveLearner(lr=1e-3, epochs=1), stream_trains, all_tests, "cpu")
    assert len(R) == n_stream + 1          # post-anchor + one row per stream task
    assert all(len(row) == n_tasks for row in R)
    assert all(0.0 <= a <= 1.0 for row in R for a in row)
