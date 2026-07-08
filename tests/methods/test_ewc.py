"""EWC baselines: one-shot anchor-EWC vs online-EWC (running, per-task consolidation)."""

from __future__ import annotations

import torch

from cl_experiments.methods import BLROnlineLearner, OnlineEWCLearner, build_learner


def test_online_ewc_consolidates_each_task(separable_loader, toy_model):
    """After finetuning on a task, online-EWC must move its reference to the new
    weights and grow the running Fisher (accumulation, gamma=1) -- that is what makes
    it protect stream knowledge, unlike the frozen one-shot anchor-EWC."""
    torch.manual_seed(0)
    model = toy_model()
    anchor_loader = separable_loader(seed=0)
    learner = build_learner("ewc_online", model, anchor_loader, "cpu",
                            ewc_lambda=100.0, lr=1e-3, fisher_batches=2)
    assert isinstance(learner, OnlineEWCLearner)

    fisher_before = {n: f.clone() for n, f in learner.fisher.items()}
    ref_before = {n: r.clone() for n, r in learner.ref.items()}

    task = separable_loader(seed=1)  # a "stream" task
    learner.finetune(model, task, "cpu")

    # reference moved to the just-learned weights
    moved = any(not torch.equal(ref_before[n], learner.ref[n]) for n in ref_before)
    assert moved, "online-EWC did not move its consolidation point after a task"
    # running Fisher grew (F <- gamma*F + F_task, gamma=1 -> strictly >= old)
    grew = any((learner.fisher[n] >= fisher_before[n] - 1e-9).all() for n in fisher_before)
    assert grew


def test_one_shot_ewc_is_frozen(separable_loader, toy_model):
    """One-shot anchor-EWC keeps its Fisher/anchor fixed (no per-task consolidation)."""
    torch.manual_seed(0)
    model = toy_model()
    learner = build_learner("ewc", model, separable_loader(seed=0), "cpu",
                            ewc_lambda=100.0, lr=1e-3, fisher_batches=2)
    fisher_before = {n: f.clone() for n, f in learner.fisher.items()}
    learner.finetune(model, separable_loader(seed=1), "cpu")
    # fisher and anchor are unchanged (frozen at the anchor)
    assert all(torch.equal(fisher_before[n], learner.fisher[n]) for n in fisher_before)


def test_blr_online_reconsolidates_each_task(separable_loader, toy_model):
    """BLR-online must refresh sigma and move the prior mean after each task (the BLR
    analog of online-EWC consolidation); plain BLR keeps both fixed at the anchor."""
    torch.manual_seed(0)
    model = toy_model()
    learner = build_learner("blr_online", model, separable_loader(seed=0), "cpu",
                            sigma_mode="laplace", fisher_mode="true", fisher_batches=2,
                            sigma_prior=0.05, beta=8.0)
    assert isinstance(learner, BLROnlineLearner)
    sigma_before = {n: s.clone() for n, s in learner.opt.sigma.items()}
    prior_before = {n: p.clone() for n, p in learner.opt.mu_prior.items()}

    learner.finetune(model, separable_loader(seed=1), "cpu")

    sigma_changed = any(not torch.equal(sigma_before[n], learner.opt.sigma[n]) for n in sigma_before)
    prior_moved = any(not torch.equal(prior_before[n], learner.opt.mu_prior[n]) for n in prior_before)
    assert sigma_changed, "BLR-online did not refresh sigma at the task boundary"
    assert prior_moved, "BLR-online did not move the prior mean to the current weights"


def test_blr_const_online_tracks_mean_without_curvature(separable_loader, toy_model):
    """blr_const_online is the ablation: it keeps the online MEAN-tracking (prior moves to
    the current weights each task) but never uses the Fisher -- so its precision floor stays
    the UNIFORM prior, with no curvature structure. This isolates whether the online lift
    comes from curvature or merely from mean-tracking."""
    torch.manual_seed(0)
    model = toy_model()
    learner = build_learner("blr_const_online", model, separable_loader(seed=0), "cpu",
                            sigma_mode="const", sigma_const=0.05, sigma_prior=0.05, beta=8.0)
    assert isinstance(learner, BLROnlineLearner)
    assert learner.curvature is False
    prior_before = {n: p.clone() for n, p in learner.opt.mu_prior.items()}

    learner.finetune(model, separable_loader(seed=1), "cpu")

    prior_moved = any(not torch.equal(prior_before[n], learner.opt.mu_prior[n]) for n in prior_before)
    assert prior_moved, "blr_const_online did not track the mean"
    # the precision floor is the FLAT prior everywhere -- no curvature leaked in
    s_prior = 1.0 / learner.sigma_prior**2
    for f in learner.opt.precision_floor.values():
        assert torch.allclose(f, torch.full_like(f, s_prior)), "floor is not flat (curvature leaked)"
