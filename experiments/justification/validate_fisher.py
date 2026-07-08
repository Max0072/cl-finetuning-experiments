"""Validate the scalable MC Fisher against the exact deterministic (expected) one.

Justifies using MC (which scales to large output spaces) by showing it
matches the exact deterministic Fisher on the certified MNIST anchor. Reports the
Pearson/Spearman correlation of the per-weight Fisher, and of the resulting
Laplace sigmas.

Run:  uv run python -m experiments.justification.validate_fisher --n 3
"""

from __future__ import annotations

import argparse

import torch

from cl_experiments.curvature import diagonal_fisher, laplace_sigma
from cl_experiments.harness import get_anchor
from cl_experiments.repro import pick_device


def _flat(d):
    return torch.cat([v.flatten().cpu() for v in d.values()])


def _corr(a, b):
    pear = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    spear = torch.corrcoef(torch.stack([ra, rb]))[0, 1].item()
    return pear, spear


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                    help="force a device; 'cpu' is bit-reproducible (MPS+vmap is not)")
    args = ap.parse_args()
    device = pick_device() if args.device == "auto" else torch.device(args.device)

    model, _, (anchor_train, _) = get_anchor(args.n, args.seed, device=device, verbose=True)
    n_data = len(anchor_train.dataset)
    print(f"anchor n={args.n}  n_data={n_data}  fisher batches={args.batches}")

    exp = diagonal_fisher(model, anchor_train, mode="expected", max_batches=args.batches,
                          device=device)
    gen = torch.Generator().manual_seed(args.seed)
    mc = diagonal_fisher(model, anchor_train, mode="true", max_batches=args.batches,
                         device=device, generator=gen)
    emp = diagonal_fisher(model, anchor_train, mode="empirical", max_batches=args.batches,
                          device=device)

    fe, fm, fp = _flat(exp), _flat(mc), _flat(emp)
    p_mc, s_mc = _corr(fe, fm)
    p_emp, s_emp = _corr(fe, fp)
    print("\nFisher agreement vs expected (deterministic):")
    print(f"  MC (true, S=1)  pearson={p_mc:.4f}  spearman={s_mc:.4f}")
    print(f"  empirical       pearson={p_emp:.4f}  spearman={s_emp:.4f}")

    sp = 0.05
    se, sm = _flat(laplace_sigma(exp, n_data, sp)), _flat(laplace_sigma(mc, n_data, sp))
    p_s, s_s = _corr(se, sm)
    print(f"\nLaplace sigma (sigma_prior={sp}) MC vs expected: pearson={p_s:.4f} spearman={s_s:.4f}")
    print(f"  sigma range  expected [{se.min():.2e},{se.max():.2e}]  "
          f"MC [{sm.min():.2e},{sm.max():.2e}]")


if __name__ == "__main__":
    main()
