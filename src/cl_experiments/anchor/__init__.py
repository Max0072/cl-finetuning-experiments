"""Pipeline step 1 -- the anchor: train the model whose knowledge we preserve.

Joint multi-task training to MODERATE convergence, then accuracy certification.
See ``train`` and docs/SETTING.md.
"""

from cl_experiments.anchor.train import AnchorReport, train_anchor

__all__ = ["AnchorReport", "train_anchor"]
