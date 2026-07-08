"""Pipeline step 4 -- evaluation.

``eval`` is the bare accuracy helper; ``cl_metrics`` builds the accuracy matrix and
the continual-learning metrics (ACC / BWT / retention / plasticity).
"""

from cl_experiments.metrics.cl_metrics import (
    CLMetrics,
    compute_metrics,
    evaluate_tasks,
)
from cl_experiments.metrics.eval import accuracy

__all__ = [
    "CLMetrics",
    "compute_metrics",
    "evaluate_tasks",
    "accuracy",
]
