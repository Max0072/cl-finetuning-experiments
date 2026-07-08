"""Metric computations verified against a hand-worked accuracy matrix."""

from __future__ import annotations

import math

from cl_experiments.metrics import compute_metrics

# n_anchor=2, N=2 stream tasks -> T=4 tasks, 3 events.
# columns: [anchor0, anchor1, stream0, stream1]; learned_at = [0,0,1,2]
R = [
    [0.90, 0.90, 0.10, 0.10],  # after anchor
    [0.80, 0.85, 0.95, 0.10],  # after stream task 0 (col 2)
    [0.70, 0.80, 0.90, 0.90],  # after stream task 1 (col 3)
]


def _close(a, b, tol=1e-6):
    return abs(a - b) < tol


def test_metrics_match_hand_computation():
    m = compute_metrics(R, n_anchor=2)
    assert _close(m.acc, (0.70 + 0.80 + 0.90 + 0.90) / 4)          # 0.825
    assert _close(m.anchor_final, (0.70 + 0.80) / 2)                # 0.75
    assert _close(m.anchor_forgetting, ((0.70 - 0.90) + (0.80 - 0.90)) / 2)  # -0.15
    assert _close(m.stream_final, (0.90 + 0.90) / 2)                # 0.90
    # BWT over j in {0,1,2} (j=3 learned last, excluded)
    assert _close(m.bwt, ((0.70 - 0.90) + (0.80 - 0.90) + (0.90 - 0.95)) / 3)
    # learning_acc = mean of R[learned_at][j]
    assert _close(m.learning_acc, (0.90 + 0.90 + 0.95 + 0.90) / 4)
    # forgetting = mean over j<last of (peak_since_learned - final)
    assert _close(m.forgetting, (0.20 + 0.10 + 0.05) / 3)


def test_no_stream_is_wellformed():
    # anchor only (one event, N=0): metrics should not crash; BWT/forgetting = 0
    m = compute_metrics([[0.9, 0.95]], n_anchor=2)
    assert _close(m.acc, 0.925)
    assert _close(m.bwt, 0.0)
    assert _close(m.anchor_forgetting, 0.0)
    assert math.isnan(m.stream_final)
