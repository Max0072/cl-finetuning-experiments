"""Continual-learning methods (the pipeline's step-3 "method slot").

Each method is a *learner* with a single ``finetune(model, train_loader, device)``
call that adapts the model to one stream task. The harness (``run_stream``) calls
it once per stream task, evaluating between calls to build the accuracy matrix.

Methods share the fixed setting; they differ only in how they protect the
anchor's knowledge:

  * ``NaiveLearner``  — plain Adam fine-tuning, no protection (forgetting floor).
  * ``EWCLearner``    — one-shot anchor-EWC: diagonal-Fisher penalty toward the anchor.
  * ``OnlineEWCLearner`` — online-EWC: running Fisher, re-consolidated every task.
  * ``ReplayLearner`` — experience replay from a buffer of anchor-task samples.
  * ``BLRLearner``    — the Bayesian Learning Rule update with curvature-initialised
                        sigma (our method; ``const`` sigma is an ablation).

Anchor-derived state (EWC Fisher, replay buffer, BLR wrapping) is built by
``build_learner`` from the anchor model + its training data.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from cl_experiments.curvature import diagonal_fisher, laplace_sigma
from cl_experiments.methods.blr import BLR


class NaiveLearner:
    def __init__(self, lr: float = 1e-3, epochs: int = 1) -> None:
        self.lr, self.epochs = lr, epochs

    def finetune(self, model: nn.Module, loader: DataLoader, device) -> None:
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                crit(model(x), y).backward()
                opt.step()


class EWCLearner:
    def __init__(self, fisher, anchor, ewc_lambda: float = 1e4, lr: float = 1e-3, epochs: int = 1):
        self.fisher, self.anchor = fisher, anchor
        self.ewc_lambda, self.lr, self.epochs = ewc_lambda, lr, epochs

    def finetune(self, model: nn.Module, loader: DataLoader, device) -> None:
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(x), y)
                penalty = sum(
                    (self.fisher[n] * (p - self.anchor[n]) ** 2).sum()
                    for n, p in model.named_parameters()
                )
                (loss + 0.5 * self.ewc_lambda * penalty).backward()
                opt.step()


class OnlineEWCLearner:
    """Online-EWC (Schwarz et al. 2018): a *running* diagonal-Fisher penalty that is
    consolidated after EVERY task, not just at the anchor.

    Contrast with ``EWCLearner`` (our one-shot anchor-EWC), which freezes the penalty
    at the anchor and so only protects the anchor's knowledge. Here, after each stream
    task we (1) recompute the diagonal Fisher on that task's data at the current
    weights, (2) fold it into a running Fisher ``F <- gamma*F + F_task``, and (3) move
    the reference point to the current weights. So knowledge acquired DURING the stream
    is protected too -- the stronger, standard EWC baseline. Still data-free: only the
    Fisher diagonal + reference weights are kept, never the data itself.
    """

    def __init__(self, fisher, ref, ewc_lambda: float = 1e4, lr: float = 1e-3,
                 epochs: int = 1, gamma: float = 1.0, fisher_batches: int = 30):
        self.fisher, self.ref = fisher, ref
        self.ewc_lambda, self.lr, self.epochs = ewc_lambda, lr, epochs
        self.gamma, self.fisher_batches = gamma, fisher_batches

    def finetune(self, model: nn.Module, loader: DataLoader, device) -> None:
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                opt.zero_grad(set_to_none=True)
                loss = crit(model(x), y)
                penalty = sum(
                    (self.fisher[n] * (p - self.ref[n]) ** 2).sum()
                    for n, p in model.named_parameters()
                )
                (loss + 0.5 * self.ewc_lambda * penalty).backward()
                opt.step()
        # Consolidate: fold this task's Fisher into the running estimate and move the
        # reference to the just-learned weights (the online-EWC approximation).
        f_task = diagonal_fisher(model, loader, mode="true",
                                 max_batches=self.fisher_batches, device=device)
        self.fisher = {n: self.gamma * self.fisher[n] + f_task[n] for n in self.fisher}
        self.ref = {n: p.detach().clone() for n, p in model.named_parameters()}


class ReplayLearner:
    def __init__(self, buf_x, buf_y, lr: float = 1e-3, epochs: int = 1, replay_bs: int = 128):
        self.buf_x, self.buf_y = buf_x, buf_y
        self.lr, self.epochs, self.replay_bs = lr, epochs, replay_bs

    def finetune(self, model: nn.Module, loader: DataLoader, device) -> None:
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        crit = nn.CrossEntropyLoss()
        bx, by = self.buf_x.to(device), self.buf_y.to(device)
        for _ in range(self.epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                idx = torch.randint(0, bx.size(0), (min(self.replay_bs, bx.size(0)),), device=device)
                xc = torch.cat([x, bx[idx]])
                yc = torch.cat([y, by[idx]])
                opt.zero_grad(set_to_none=True)
                crit(model(xc), yc).backward()
                opt.step()


class BLRLearner:
    def __init__(self, opt: BLR, epochs: int = 1) -> None:
        self.opt, self.epochs = opt, epochs

    def finetune(self, model: nn.Module, loader: DataLoader, device) -> None:
        model.train()
        self.opt.to(device)
        crit = nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                self.opt.zero_accum()
                for _ in range(self.opt.n_mc):
                    self.opt.zero_grad()
                    self.opt.sample()
                    crit(model(x), y).backward()
                    self.opt.accumulate()
                self.opt.step()


class BLROnlineLearner:
    """BLR with online per-task re-consolidation -- the BLR analog of online-EWC.

    Plain ``BLRLearner`` fixes the admissible zone at the anchor: sigma comes from the
    anchor Fisher and the prior mean stays at the anchor weights. Here we REFRESH the
    zone after every stream task: recompute the diagonal Fisher on the just-learned
    task, fold it into a running Fisher (``F <- gamma*F + F_task``), re-derive the
    Laplace sigma from it, and move the prior mean to the current weights. So the
    protected point tracks the stream (not fixed at the anchor), while the accumulated
    precision keeps the old importance. Within a task sigma consolidates continuously
    (rho-EMA) toward the accumulated-precision floor; at each task boundary it is
    re-derived from the running Fisher -- mirroring online-EWC's per-task consolidation.

    If a replay buffer (``buf_x``/``buf_y``) is given, each new-task batch is augmented
    with a rehearsal minibatch (hybrid ``blr_online_replay``): the online mechanism plus
    real old-task data. NOT data-free in that mode.
    """

    def __init__(self, opt: BLR, fisher: dict, n_data: int, *, sigma_prior: float = 0.05,
                 gamma: float = 1.0, fisher_mode: str = "true", fisher_batches: int = 30,
                 fisher_seed: int = 0, epochs: int = 1,
                 buf_x=None, buf_y=None, replay_bs: int = 128) -> None:
        self.opt, self.fisher_run, self.n_data = opt, fisher, n_data
        self.sigma_prior, self.gamma = sigma_prior, gamma
        self.fisher_mode, self.fisher_batches = fisher_mode, fisher_batches
        self.fisher_seed, self.epochs = fisher_seed, epochs
        self.buf_x, self.buf_y, self.replay_bs = buf_x, buf_y, replay_bs

    def _set_floor(self, device) -> None:
        """precision_floor = accumulated precision (N_data*running_Fisher + prior). The
        within-task sigma-EMA then relaxes toward this floor plus the current task's
        curvature -- i.e. the RIGHT Laplace target, so sigma tracks it instead of
        drifting above (see the sigma-convergence diagnostic)."""
        s_prior = 1.0 / self.sigma_prior**2
        self.opt.precision_floor = {n: (self.n_data * f + s_prior).to(device)
                                    for n, f in self.fisher_run.items()}

    def finetune(self, model: nn.Module, loader: DataLoader, device) -> None:
        model.train()
        self.opt.to(device)
        if self.opt.precision_floor is None:
            self._set_floor(device)  # first task: seed the floor from the anchor Fisher
        crit = nn.CrossEntropyLoss()
        bx = by = None
        if self.buf_x is not None:
            bx, by = self.buf_x.to(device), self.buf_y.to(device)
        for _ in range(self.epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                if bx is not None:  # augment with a rehearsal minibatch
                    idx = torch.randint(0, bx.size(0), (min(self.replay_bs, bx.size(0)),),
                                        device=device)
                    x, y = torch.cat([x, bx[idx]]), torch.cat([y, by[idx]])
                self.opt.zero_accum()
                for _ in range(self.opt.n_mc):
                    self.opt.zero_grad()
                    self.opt.sample()
                    crit(model(x), y).backward()
                    self.opt.accumulate()
                self.opt.step()
        # Re-consolidate the admissible zone on the just-learned task.
        gen = (torch.Generator().manual_seed(self.fisher_seed)
               if self.fisher_mode == "true" else None)
        f_task = diagonal_fisher(model, loader, mode=self.fisher_mode,
                                 max_batches=self.fisher_batches, device=device, generator=gen)
        self.fisher_run = {n: self.gamma * self.fisher_run[n] + f_task[n]
                           for n in self.fisher_run}
        sigma_new = laplace_sigma(self.fisher_run, n_data=self.n_data, sigma_prior=self.sigma_prior)
        for n in self.opt.sigma:
            self.opt.sigma[n] = sigma_new[n].to(device)
            self.opt.mu_prior[n] = self.opt.mu[n].detach().clone()  # prior tracks latest
        self._set_floor(device)  # raise the floor with the newly accumulated importance


class BLRReplayLearner:
    """BLR update computed on replay-augmented batches (hybrid, NOT data-free).

    Mixes a rehearsal minibatch from the anchor buffer into each new-task batch, so
    the BLR gradient carries the exact old-task loss (fixing the local-quadratic
    approximation) while sigma^2 still preconditions the step. Upper-bound / "if you
    do have data" variant."""

    def __init__(self, opt: BLR, buf_x, buf_y, epochs: int = 1, replay_bs: int = 128):
        self.opt, self.buf_x, self.buf_y = opt, buf_x, buf_y
        self.epochs, self.replay_bs = epochs, replay_bs

    def finetune(self, model: nn.Module, loader: DataLoader, device) -> None:
        model.train()
        self.opt.to(device)
        bx, by = self.buf_x.to(device), self.buf_y.to(device)
        crit = nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                idx = torch.randint(0, bx.size(0), (min(self.replay_bs, bx.size(0)),), device=device)
                xc = torch.cat([x, bx[idx]])
                yc = torch.cat([y, by[idx]])
                self.opt.zero_accum()
                for _ in range(self.opt.n_mc):
                    self.opt.zero_grad()
                    self.opt.sample()
                    crit(model(xc), yc).backward()
                    self.opt.accumulate()
                self.opt.step()


def _sample_buffer(loader: DataLoader, size: int):
    xs, ys, got = [], [], 0
    for x, y in loader:
        xs.append(x)
        ys.append(y)
        got += x.size(0)
        if got >= size:
            break
    return torch.cat(xs)[:size], torch.cat(ys)[:size]


def build_learner(
    method: str,
    model: nn.Module,
    anchor_train: DataLoader,
    device,
    *,
    epochs: int = 1,
    lr: float = 1e-3,
    # ewc
    ewc_lambda: float = 1e4,
    fisher_batches: int = 100,
    ewc_gamma: float = 1.0,        # online-EWC running-Fisher decay (1.0 = pure accumulation)
    # replay
    buffer_size: int = 2000,
    replay_bs: int = 128,
    # blr
    sigma_mode: str = "laplace",   # "laplace" (curvature) or "const"
    sigma_prior: float = 0.05,     # prior std; also the BLR relaxation target
    fisher_mode: str = "expected",  # curvature estimator for sigma init
    sigma_const: float = 0.05,     # used only when sigma_mode == "const"
    n_samples: float = 6e4,        # BLR memory window N (NOT the Laplace n_data)
    n_mc: int = 1,
    fisher_seed: int = 0,
    beta: float = 1.0,             # BLR mean step-size (overall mu-step multiplier)
    noise_scale: float = 1.0,      # kappa: sampling-noise scale (decoupled from step)
    rho: float = 0.1,              # BLR precision EMA rate
):
    """Construct a learner, building any anchor-derived state it needs."""
    if method == "naive":
        return NaiveLearner(lr=lr, epochs=epochs)
    if method == "ewc":
        fisher = diagonal_fisher(model, anchor_train, mode="true",
                                 max_batches=fisher_batches, device=device)
        anchor = {n: p.detach().clone() for n, p in model.named_parameters()}
        return EWCLearner(fisher, anchor, ewc_lambda=ewc_lambda, lr=lr, epochs=epochs)
    if method == "ewc_online":  # running-Fisher EWC, consolidated after every task
        fisher = diagonal_fisher(model, anchor_train, mode="true",
                                 max_batches=fisher_batches, device=device)
        ref = {n: p.detach().clone() for n, p in model.named_parameters()}
        return OnlineEWCLearner(fisher, ref, ewc_lambda=ewc_lambda, lr=lr, epochs=epochs,
                                gamma=ewc_gamma, fisher_batches=fisher_batches)
    if method == "replay":
        bx, by = _sample_buffer(anchor_train, buffer_size)
        return ReplayLearner(bx, by, lr=lr, epochs=epochs, replay_bs=replay_bs)
    if method == "blr":
        if sigma_mode == "laplace":
            gen = torch.Generator().manual_seed(fisher_seed)
            fisher = diagonal_fisher(model, anchor_train, mode=fisher_mode,
                                     max_batches=fisher_batches, device=device, generator=gen)
            n_data = len(anchor_train.dataset)  # total anchor examples -> N * mean(F)
            sigma_init = laplace_sigma(fisher, n_data=n_data, sigma_prior=sigma_prior)
        else:  # const
            sigma_init = sigma_const
        opt = BLR(model, sigma_init=sigma_init, sigma_prior=sigma_prior,
                  n_samples=n_samples, n_mc=n_mc, mu_prior="init",
                  lr=beta, noise_scale=noise_scale, rho=rho)
        return BLRLearner(opt, epochs=epochs)
    if method in ("blr_online", "blr_online_replay"):
        # BLR with online per-task re-consolidation (~ online-EWC). blr_online_replay
        # additionally augments each batch with a rehearsal minibatch (hybrid, not data-free).
        gen = torch.Generator().manual_seed(fisher_seed)
        fisher = diagonal_fisher(model, anchor_train, mode=fisher_mode,
                                 max_batches=fisher_batches, device=device, generator=gen)
        n_data = len(anchor_train.dataset)
        sigma_init = laplace_sigma(fisher, n_data=n_data, sigma_prior=sigma_prior)
        # sigma consolidates continuously (rho-EMA) toward the accumulated-precision floor,
        # so within a task it already approaches the next per-task re-consolidation target.
        opt = BLR(model, sigma_init=sigma_init, sigma_prior=sigma_prior, n_samples=n_samples,
                  n_mc=n_mc, mu_prior="init", lr=beta,
                  noise_scale=noise_scale, rho=rho)
        bx = by = None
        if method == "blr_online_replay":
            bx, by = _sample_buffer(anchor_train, buffer_size)
        return BLROnlineLearner(opt, fisher, n_data, sigma_prior=sigma_prior, gamma=ewc_gamma,
                                fisher_mode=fisher_mode, fisher_batches=fisher_batches,
                                fisher_seed=fisher_seed, epochs=epochs,
                                buf_x=bx, buf_y=by, replay_bs=replay_bs)
    if method == "blr_replay":  # hybrid: BLR update on replay-augmented batches
        if sigma_mode == "laplace":
            gen = torch.Generator().manual_seed(fisher_seed)
            fisher = diagonal_fisher(model, anchor_train, mode=fisher_mode,
                                     max_batches=fisher_batches, device=device, generator=gen)
            sigma_init = laplace_sigma(fisher, n_data=len(anchor_train.dataset),
                                       sigma_prior=sigma_prior)
        else:
            sigma_init = sigma_const
        opt = BLR(model, sigma_init=sigma_init, sigma_prior=sigma_prior, n_samples=n_samples,
                  n_mc=n_mc, mu_prior="init", lr=beta,
                  noise_scale=noise_scale, rho=rho)
        bx, by = _sample_buffer(anchor_train, buffer_size)
        return BLRReplayLearner(opt, bx, by, epochs=epochs, replay_bs=replay_bs)
    raise ValueError(f"unknown method: {method}")
