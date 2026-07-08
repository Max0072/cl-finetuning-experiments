"""Does curvature-init matter LONGER when consolidation is slower? (the LLM extrapolation)

The horizon test showed curvature-σ init washes out within one task -- but that was with a
FULL epoch (~470 steps) per task, so the σ-EMA fully overwrites the init: fraction surviving
~ (1-rho)^steps -> ~0. A short, few-step fine-tune (the realistic LLM regime) consolidates
far less per task, so the init should survive across MANY tasks. This tests exactly that
knob: steps-per-task. If the curvature-vs-flat retention GAP persists over more tasks as
steps shrink, then "slow geometry convergence -> curvature matters for a long time" is a
measured MNIST fact, not a bare extrapolation to scale.

For each steps-per-task budget K we run curvature-σ and flat-σ init at matched β over the
stream and record the anchor-retention gap (curv - flat) after each task.

Run:  uv run python -m experiments.justification.curvature_persistence --n 3 --N 8 --device cpu
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib
import torch
from torch import nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cl_experiments.config import BLR_SIG_BASE, CONST_SIG_BASE
from cl_experiments.harness.pipeline import get_anchor, stream_loaders
from cl_experiments.methods import build_learner
from cl_experiments.metrics import accuracy
from cl_experiments.repro import (
    Timer,
    get_logger,
    pick_device,
    run_manifest,
    set_seed,
    setup_logging,
)

OUT = Path(__file__).resolve().parents[2] / "results" / "justification" / "curvature_persistence.json"
log = get_logger()
BETA = 16.0
# steps (batches) per task: few-step (LLM-like) -> full epoch. rho=0.06 -> surviving init
# fraction (1-rho)^K ~ 0.73 / 0.21 / 0.002 / ~0.
STEPS = [5, 25, 100, 10**9]
STEP_LABELS = {5: "5 steps", 25: "25 steps", 100: "100 steps", 10**9: "full epoch"}


def _anchor_ret(model, tests, device):
    return sum(accuracy(model, t, device) for t in tests) / len(tests)


def _run(base, n, seed, ns, max_batches, device):
    """Advance BLR over the stream capping each task at max_batches; return anchor retention
    after the anchor and after each task."""
    set_seed(seed)  # identical RNG across variants -> only the σ init differs
    anchor_model, _, (anchor_train, anchor_tests) = get_anchor(n, seed, device=device, verbose=False)
    stream_trains = [tr for tr, _ in stream_loaders(n_stream=ns, seed=seed)]
    model = copy.deepcopy(anchor_model)
    opt = build_learner("blr", model, anchor_train, device, **{**base, "beta": BETA}).opt
    opt.to(device)
    crit = nn.CrossEntropyLoss()
    rets = [_anchor_ret(model, anchor_tests, device)]
    for tr in stream_trains:
        model.train()
        for i, (x, y) in enumerate(tr):
            if i >= max_batches:
                break
            x, y = x.to(device), y.to(device)
            opt.zero_accum()
            for _ in range(opt.n_mc):
                opt.zero_grad()
                opt.sample()
                crit(model(x), y).backward()
                opt.accumulate()
            opt.step()
        rets.append(_anchor_ret(model, anchor_tests, device))
    return rets


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--N", type=int, default=8, help="stream length")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                    help="force a device; 'cpu' is bit-reproducible (MPS+vmap is not)")
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(OUT.parent / "curvature_persistence.log")
    device = pick_device() if args.device == "auto" else torch.device(args.device)

    series = {}
    with Timer() as t:
        for k in STEPS:
            ret_c = _run(BLR_SIG_BASE, args.n, args.seed, args.N, k, device)
            ret_f = _run(CONST_SIG_BASE, args.n, args.seed, args.N, k, device)
            gap = [c - f for c, f in zip(ret_c, ret_f, strict=True)]
            series[STEP_LABELS[k]] = {"steps": min(k, 469), "ret_curv": ret_c,
                                      "ret_flat": ret_f, "gap": gap}
            log.info("%-11s | mean gap over stream = %+.4f | final gap = %+.4f",
                     STEP_LABELS[k], sum(gap[1:]) / max(1, len(gap) - 1), gap[-1])

    out = {"beta": BETA, "n": args.n, "seed": args.seed, "N": args.N,
           "tasks": list(range(args.N + 1)), "series": series}
    OUT.write_text(json.dumps(out, indent=2))
    run_manifest(OUT.parent / "curvature_persistence.manifest.json",
                 config={"n": args.n, "N": args.N, "seed": args.seed, "beta": BETA, "steps": STEPS},
                 elapsed_s=t.elapsed, device=device)
    _plot(out)


def _plot(o):
    tasks = o["tasks"]
    order = ["5 steps", "25 steps", "100 steps", "full epoch"]
    cols = {"5 steps": "#08519c", "25 steps": "#3182bd", "100 steps": "#9ecae1",
            "full epoch": "#bdbdbd"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for name in order:
        s = o["series"].get(name)
        if s:
            ax1.plot(tasks, s["gap"], "-o", color=cols[name], lw=2.4, ms=4, label=name)
    ax1.axhline(0, color="#333", lw=0.8)
    ax1.set(xlabel="stream task", ylabel="retention gap  (curvature-σ − flat-σ)",
            title="(a) Gap stays tiny (~0.002) at EVERY step budget — no persistence")
    ax1.legend(title="steps per task")
    ax1.grid(alpha=0.3)
    # summary: mean gap over the stream vs steps-per-task
    xs = [o["series"][n]["steps"] for n in order if n in o["series"]]
    ys = [sum(o["series"][n]["gap"][1:]) / max(1, len(o["series"][n]["gap"]) - 1)
          for n in order if n in o["series"]]
    ax2.plot(xs, ys, "-o", color="#d62728", lw=2.4, ms=6)
    ax2.set(xscale="log", xlabel="steps per task (log)", ylabel="mean retention gap over stream",
            title="(b) No trend with steps/task — a confounded proxy")
    ax2.grid(alpha=0.3, which="both")
    fig.suptitle(f"INCONCLUSIVE: steps-per-task confounds 'init survives' with 'little movement "
                 f"→ little forgetting', so there is nothing for the σ-init to save. The "
                 f"data-vs-capacity test (capacity_scaling) is the right one "
                 f"(n={o['n']}, seed {o['seed']})", fontsize=10)
    fig.tight_layout()
    out_png = OUT.parent / "curvature_persistence.png"
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
