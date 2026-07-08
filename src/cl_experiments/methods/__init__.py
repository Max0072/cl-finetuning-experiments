"""Pipeline step 3 -- the fine-tuning "method slot".

``blr`` is the BLR optimiser (our update rule); ``learners`` wraps every method
(naive / ewc / ewc_online / replay / blr / blr_online / blr_replay) behind a common
``finetune`` call and ``build_learner``; ``loops`` holds the raw training loops.
"""

from cl_experiments.methods.blr import BLR, MuPrior
from cl_experiments.methods.learners import (
    BLRLearner,
    BLROnlineLearner,
    BLRReplayLearner,
    EWCLearner,
    NaiveLearner,
    OnlineEWCLearner,
    ReplayLearner,
    build_learner,
)
from cl_experiments.methods.loops import train_blr, train_plain

__all__ = [
    "BLR",
    "MuPrior",
    "NaiveLearner",
    "EWCLearner",
    "OnlineEWCLearner",
    "ReplayLearner",
    "BLRLearner",
    "BLROnlineLearner",
    "BLRReplayLearner",
    "build_learner",
    "train_plain",
    "train_blr",
]
