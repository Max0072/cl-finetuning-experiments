"""Ablation: why MODERATE anchor convergence, not the deepest minimum.

Over-training the anchor (loss->0) saturates the softmax and collapses the true
Fisher, so the curvature-informed sigma loses all structure (everything sits at
the prior). We sweep training depth (epochs of plain training on the n=3 joint
anchor) and report: train loss, gradient norm, min per-task test acc, and Fisher
informativeness -- max data-term N*F and the fraction of weights the Laplace
posterior meaningfully protects. This justifies stopping at moderate loss.

We compute BOTH the exact deterministic (expected) Fisher AND the scalable MC
(true, S=1) Fisher that the method actually uses, at every convergence depth, and
report their agreement (Pearson/Spearman). This confirms (a) the collapse story
holds for the production MC estimator too, not only the reference, and (b) MC
stays faithful to exact across the WHOLE convergence range -- including the
saturated regime where MC almost always draws the argmax label -- so the choice
of MC (for LLM-scale output spaces) is not a source of pathology here.

Run:  uv run python -m experiments.justification.anchor_convergence
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cl_experiments.anchor.train import _grad_norm
from cl_experiments.config import SETTING
from cl_experiments.curvature import diagonal_fisher, laplace_sigma
from cl_experiments.data import anchor_loaders
from cl_experiments.harness import anchor_perms
from cl_experiments.methods import train_plain
from cl_experiments.metrics import accuracy
from cl_experiments.models import MLP
from cl_experiments.repro import get_logger, pick_device, run_manifest, set_seed, setup_logging

OUT = Path(__file__).resolve().parents[2] / "results" / "justification" / "anchor_convergence.json"
log = get_logger()
EPOCHS = [1, 3, 10, 30, 60]
SIGMA_PRIOR = 0.05


def _nf_pct(fisher, n_data):
    """max data-term N*F and %-of-weights the Laplace posterior protects."""
    nf = torch.cat([(n_data * f).flatten().cpu() for f in fisher.values()])
    sig = laplace_sigma(fisher, n_data=n_data, sigma_prior=SIGMA_PRIOR)
    flat = torch.cat([s.flatten().cpu() for s in sig.values()])
    pct = (flat < 0.9 * SIGMA_PRIOR).float().mean().item() * 100
    return nf.max().item(), pct


def _corr(a, b):
    pear = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    ra, rb = a.argsort().argsort().float(), b.argsort().argsort().float()
    spear = torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
    return pear, spear


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    setup_logging(OUT.parent / "anchor_convergence.log")
    device = pick_device()
    perms = anchor_perms(args.n, args.seed)
    train_loader, test_loaders = anchor_loaders(perms, batch_size=SETTING.batch_size, seed=args.seed)
    n_data = len(train_loader.dataset)
    crit = torch.nn.CrossEntropyLoss()

    out = {}
    log.info("anchor convergence sweep (n=%d, sigma_prior=%.2f, n_data=%d)",
             args.n, SIGMA_PRIOR, n_data)
    log.info("%6s %10s %10s %8s | %9s %8s | %9s %8s | %7s %7s",
             "epochs", "train_loss", "grad_norm", "min_acc",
             "exp N*F", "exp%prot", "mc N*F", "mc%prot", "pear", "spear")
    for ep in EPOCHS:
        set_seed(args.seed)
        model = MLP().to(device)
        train_plain(model, train_loader, epochs=ep, lr=SETTING.anchor_lr, device=device)
        model.eval()
        with torch.no_grad():
            tot = nb = 0
            for x, y in train_loader:
                tot += crit(model(x.to(device)), y.to(device)).item()
                nb += 1
                if nb >= 100:
                    break
        loss = tot / nb
        gnorm = _grad_norm(model, train_loader, device)
        min_acc = min(accuracy(model, t, device) for t in test_loaders)
        exp = diagonal_fisher(model, train_loader, mode="expected", max_batches=50, device=device)
        gen = torch.Generator().manual_seed(args.seed)
        mc = diagonal_fisher(model, train_loader, mode="true", max_batches=50,
                             device=device, generator=gen)
        exp_nf, exp_pct = _nf_pct(exp, n_data)
        mc_nf, mc_pct = _nf_pct(mc, n_data)
        fe = torch.cat([v.flatten().cpu() for v in exp.values()])
        fm = torch.cat([v.flatten().cpu() for v in mc.values()])
        pear, spear = _corr(fe, fm)
        out[str(ep)] = {"train_loss": loss, "grad_norm": gnorm, "min_acc": min_acc,
                        "max_NF": exp_nf, "pct_protected": exp_pct,
                        "mc_max_NF": mc_nf, "mc_pct_protected": mc_pct,
                        "mc_vs_exp_pearson": pear, "mc_vs_exp_spearman": spear}
        OUT.write_text(json.dumps(out, indent=2))
        log.info("%6d %10.4f %10.2e %8.4f | %9.1f %7.1f%% | %9.1f %7.1f%% | %7.4f %7.4f",
                 ep, loss, gnorm, min_acc, exp_nf, exp_pct, mc_nf, mc_pct, pear, spear)
    run_manifest(OUT.parent / "anchor_convergence.manifest.json",
                 config={"n": args.n, "seed": args.seed, "epochs": EPOCHS, "sigma_prior": SIGMA_PRIOR})


if __name__ == "__main__":
    main()
