"""Does importance DENSIFY as the stream fills the network? (why curvature lives in online)

The horizon test showed static curvature-init washes out (both σ converge to the current
stream geometry within one task), yet curvature is decisive in the ONLINE mechanism. Reason
suspected here: canonical BLR's σ chases only the CURRENT task, while online ACCUMULATES the
Fisher -- and, per the question raised, we estimate the anchor Fisher on the first n tasks
only; over MANY tasks importance may spread until "everything is important" (capacity fills).

We advance the model through the stream (BLR, matched β) and after each task measure:
  * per-task %important   -- fraction of weights THIS task alone marks important (σ<0.9·prior)
  * cumulative %important -- same fraction for the RUNNING Fisher sum over all tasks so far

Prediction if the capacity story holds: per-task stays ~flat (~few %), but cumulative RISES
toward saturation -- each permuted task uses a different sparse set, their union fills the net,
so protecting the ACCUMULATED set (online curvature) is what matters; a static anchor Fisher is
blind to the sets later tasks will occupy.

(Whether the tasks' importance structures ALIGN across the stream -- the pairwise Fisher
correlation and its dynamics -- is a separate, cleaner probe in `zone_convergence.py`, which
measures every task at the SAME point θ instead of along the trajectory.)

Run:  uv run python -m experiments.justification.fisher_density --n 3 --N 12 --device cpu
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cl_experiments.config import BLR_SIG_BASE
from cl_experiments.curvature import diagonal_fisher
from cl_experiments.harness.pipeline import get_anchor, stream_loaders
from cl_experiments.methods import build_learner
from cl_experiments.repro import (
    Timer,
    get_logger,
    pick_device,
    run_manifest,
    set_seed,
    setup_logging,
)

OUT = Path(__file__).resolve().parents[2] / "results" / "justification" / "fisher_density.json"
log = get_logger()
BETA = 16.0
PRIOR = 0.05
FISHER_BATCHES = 30


def _flat(d: dict) -> torch.Tensor:
    return torch.cat([d[k].detach().flatten().cpu() for k in sorted(d)])


def _frac_important(importance: torch.Tensor) -> float:
    """importance = per-weight data-term precision (n_data * mean Fisher). A weight is
    'important' when its Laplace σ = 1/sqrt(importance + 1/prior^2) drops below 0.9·prior."""
    sig = 1.0 / torch.sqrt(importance + 1.0 / PRIOR**2)
    return (sig < 0.9 * PRIOR).float().mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--N", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                    help="force a device; 'cpu' is bit-reproducible (MPS+vmap is not)")
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(OUT.parent / "fisher_density.log")
    device = pick_device() if args.device == "auto" else torch.device(args.device)

    set_seed(args.seed)
    anchor_model, _, (anchor_train, _) = get_anchor(args.n, args.seed, device=device, verbose=False)
    stream = stream_loaders(n_stream=args.N, seed=args.seed)
    model = copy.deepcopy(anchor_model)
    gen = torch.Generator().manual_seed(args.seed)

    with Timer() as t:
        # task 0 = the anchor itself
        f_anchor = diagonal_fisher(model, anchor_train, mode="true", max_batches=FISHER_BATCHES,
                                   device=device, generator=gen)
        imp_cum = len(anchor_train.dataset) * _flat(f_anchor)
        rows = [{"task": 0, "per_task_pct": _frac_important(imp_cum) * 100,
                 "cumulative_pct": _frac_important(imp_cum) * 100}]
        learner = build_learner("blr", model, anchor_train, device, **{**BLR_SIG_BASE, "beta": BETA})

        for k, (tr, _te) in enumerate(stream, start=1):
            learner.finetune(model, tr, device)
            f_t = diagonal_fisher(model, tr, mode="true", max_batches=FISHER_BATCHES,
                                  device=device, generator=gen)
            ft = _flat(f_t)
            n_t = len(tr.dataset)
            imp_cum = imp_cum + n_t * ft
            rows.append({"task": k, "per_task_pct": _frac_important(n_t * ft) * 100,
                         "cumulative_pct": _frac_important(imp_cum) * 100})

    log.info("β=%.0f n=%d seed=%d  (importance density over the stream)", BETA, args.n, args.seed)
    log.info("%5s %13s %14s", "task", "per_task_%", "cumulative_%")
    for r in rows:
        log.info("%5d %13.2f %14.2f", r["task"], r["per_task_pct"], r["cumulative_pct"])

    out = {"beta": BETA, "n": args.n, "seed": args.seed, "N": args.N, "prior": PRIOR, "rows": rows}
    OUT.write_text(json.dumps(out, indent=2))
    run_manifest(OUT.parent / "fisher_density.manifest.json",
                 config={"n": args.n, "N": args.N, "seed": args.seed, "beta": BETA, "prior": PRIOR},
                 elapsed_s=t.elapsed, device=device)
    _plot(out)


def _plot(o):
    rows = o["rows"]
    tasks = [r["task"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(7.5, 5.2))
    ax1.plot(tasks, [r["per_task_pct"] for r in rows], "-o", color="#7f7f7f", lw=2.0, ms=4,
             label="per-task (this task alone)")
    ax1.plot(tasks, [r["cumulative_pct"] for r in rows], "-o", color="#d62728", lw=2.6, ms=5,
             label="cumulative (all tasks so far)")
    ax1.set(xlabel="tasks seen", ylabel="% weights important (σ < 0.9·prior)",
            title=f"Importance fills capacity across a Permuted-MNIST stream "
                  f"(n={o['n']}, seed {o['seed']})")
    ax1.legend()
    ax1.grid(alpha=0.3)
    fig.tight_layout()
    out_png = OUT.parent / "fisher_density.png"
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
