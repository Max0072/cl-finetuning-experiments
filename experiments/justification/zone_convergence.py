"""Do stream tasks align to a shared importance structure — and does model WIDTH change it?

You cannot measure the "true" overlap of two tasks: the diagonal Fisher is local, so any
correlation is a property of (task_i, task_j, θ). The honest probe is the LEARNING DYNAMICS —
recompute EVERY task's Fisher at the SAME current point θ_t and track the mean pairwise
correlation as training proceeds. Rising & plateauing = the trajectory settles into a region
where the tasks agree on which weights matter (their importance structure aligns).

Caveat kept explicit (see docs/figures/zone_convergence.md): the `true` Fisher samples labels
from the model, so it depends on the INPUTS and weights, not the task labels. This measures
**importance-structure alignment**, NOT directly "a low-loss θ exists for all tasks" (that is
retention / joint accuracy). So it cannot do compatible-vs-incompatible; it CAN test capacity.

**Two competing predictions for model width (this is the test):**
  * spreading — a wider net has free capacity, so tasks use DIFFERENT weights → corr stays LOW;
  * larger-zones — a wider net keeps wider σ (bigger admissible zones, per capacity_scaling),
    so their intersection is larger and a shared point is easier → corr HIGHER.
The width sweep adjudicates.

Heavy: per width, (N+1) checkpoints × (N+1) tasks Fisher estimates + an inline anchor. Saved to
a reusable json; re-plot with `--replot`.

Run:  uv run python -m experiments.justification.zone_convergence --n 3 --N 8 --widths 256 1024 --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cl_experiments.config import BLR_SIG_BASE, SETTING
from cl_experiments.curvature import diagonal_fisher
from cl_experiments.data import anchor_loaders
from cl_experiments.harness import anchor_perms
from cl_experiments.harness.pipeline import stream_loaders
from cl_experiments.methods import build_learner, train_plain
from cl_experiments.models import MLP
from cl_experiments.repro import (
    Timer,
    get_logger,
    pick_device,
    run_manifest,
    set_seed,
    setup_logging,
)

OUT = Path(__file__).resolve().parents[2] / "results" / "justification" / "zone_convergence.json"
log = get_logger()
BETA = 16.0
FISHER_BATCHES = 30
ANCHOR_EPOCHS = 5          # fixed, comparable training across widths
WIDTHS = [256, 1024]


def _flat(d):
    return torch.cat([d[k].detach().flatten().cpu() for k in sorted(d)])


def _corr(a, b):
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def _matrix_at(model, task_loaders, device, gen):
    """Every task's diagonal Fisher at the model's CURRENT weights -> (T×T) corr matrix."""
    flats = [_flat(diagonal_fisher(model, ld, mode="true", max_batches=FISHER_BATCHES,
                                   device=device, generator=gen)) for ld in task_loaders]
    m = len(flats)
    return [[_corr(flats[i], flats[j]) for j in range(m)] for i in range(m)]


def _mean_offdiag(mat):
    m = len(mat)
    off = [mat[i][j] for i in range(m) for j in range(m) if i != j]
    return sum(off) / len(off)


def _run_width(width, n, seed, ns, device):
    set_seed(seed)
    perms = anchor_perms(n, seed)
    anchor_train, _ = anchor_loaders(perms, batch_size=SETTING.batch_size, seed=seed)
    model = MLP(hidden=(width, width)).to(device)
    train_plain(model, anchor_train, epochs=ANCHOR_EPOCHS, lr=SETTING.anchor_lr, device=device)
    task_loaders = [anchor_train] + [tr for tr, _ in stream_loaders(n_stream=ns, seed=seed)]
    learner = build_learner("blr", model, anchor_train, device, **{**BLR_SIG_BASE, "beta": BETA})
    gen = torch.Generator().manual_seed(seed)

    matrices = [_matrix_at(model, task_loaders, device, gen)]  # checkpoint 0 = at the anchor
    log.info("width=%-4d  after %2d tasks | mean pairwise Fisher corr = %.3f",
             width, 0, _mean_offdiag(matrices[0]))
    for k, (tr, _te) in enumerate(stream_loaders(n_stream=ns, seed=seed), start=1):
        learner.finetune(model, tr, device)
        matrices.append(_matrix_at(model, task_loaders, device, gen))
        log.info("width=%-4d  after %2d tasks | mean pairwise Fisher corr = %.3f",
                 width, k, _mean_offdiag(matrices[-1]))
    return {"width": width, "n_params": sum(p.numel() for p in model.parameters()),
            "checkpoints": list(range(ns + 1)), "matrices": matrices,
            "mean_offdiag": [_mean_offdiag(m) for m in matrices]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--N", type=int, default=8, help="stream length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--widths", type=int, nargs="+", default=WIDTHS)
    ap.add_argument("--replot", action="store_true", help="re-plot from the saved json, no recompute")
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                    help="force a device; 'cpu' is bit-reproducible (MPS+vmap is not)")
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(OUT.parent / "zone_convergence.log")
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    out = json.loads(OUT.read_text()) if OUT.exists() else {"widths": {}}

    if not args.replot:
        with Timer() as t:
            for w in args.widths:
                if str(w) in out["widths"]:
                    continue
                out["widths"][str(w)] = _run_width(w, args.n, args.seed, args.N, device)
                out.update({"n": args.n, "N": args.N, "seed": args.seed})
                OUT.write_text(json.dumps(out, indent=2))
        run_manifest(OUT.parent / "zone_convergence.manifest.json",
                     config={"n": args.n, "N": args.N, "seed": args.seed, "widths": args.widths,
                             "beta": BETA, "anchor_epochs": ANCHOR_EPOCHS},
                     elapsed_s=t.elapsed, device=device)
    _plot(out)


def _plot(o):
    widths = sorted(o["widths"], key=int)
    cols = {w: c for w, c in zip(widths, ["#08519c", "#e6550d", "#31a354", "#756bb1"], strict=False)}
    fig, axd = plt.subplot_mosaic([["line", "line"], [f"m{widths[0]}", f"m{widths[-1]}"]],
                                  figsize=(11, 9), height_ratios=[1, 1.3])
    ax = axd["line"]
    for w in widths:
        r = o["widths"][w]
        ax.plot(r["checkpoints"], r["mean_offdiag"], "-o", color=cols[w], lw=2.6, ms=6,
                label=f"width {w}  ({r['n_params'] / 1e6:.1f}M)")
    ax.set(xlabel="stream tasks learned (training progress)",
           ylabel="mean pairwise Fisher correlation",
           title="Mean pairwise Fisher correlation (all tasks at the SAME θ) vs training")
    ax.set_ylim(0, 1)
    ax.legend(title="hidden width")
    ax.grid(alpha=0.3)
    for w in (widths[0], widths[-1]):
        a = axd[f"m{w}"]
        mat = o["widths"][w]["matrices"][-1]  # final θ
        im = a.imshow(mat, cmap="magma", vmin=0, vmax=1)
        a.set(title=f"width {w}: pairwise corr at the final θ", xlabel="task", ylabel="task")
        a.set_xticks(range(len(mat)))
        a.set_yticks(range(len(mat)))
        fig.colorbar(im, ax=a, fraction=0.046, pad=0.04)
    fig.suptitle(f"Does importance structure align across tasks, and does width change it? "
                 f"(n={o.get('n')}, {o.get('N')} tasks, seed {o.get('seed')})", fontsize=12)
    fig.tight_layout()
    out_png = OUT.parent / "zone_convergence.png"
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
