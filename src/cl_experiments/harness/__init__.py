"""Orchestration -- tie the fixed setting together end-to-end.

``pipeline`` provides cached anchors and the continuation-stream loaders every
experiment shares; ``run`` streams a method through the tasks and returns metrics.
"""

from cl_experiments.harness.pipeline import (
    ANCHOR_BANK,
    CACHE_DIR,
    anchor_perms,
    get_anchor,
    stream_loaders,
    stream_perms,
)
from cl_experiments.harness.run import run_experiment, run_stream

__all__ = [
    "ANCHOR_BANK",
    "CACHE_DIR",
    "anchor_perms",
    "get_anchor",
    "stream_loaders",
    "stream_perms",
    "run_experiment",
    "run_stream",
]
