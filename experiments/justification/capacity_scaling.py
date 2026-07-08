"""Bigger models stay in the loose-σ regime longer: importance density vs model WIDTH.

The admissible-zones view: σ is a zone half-width, and more DATA tightens the intersection of
zones -> narrower σ -> more weights become important (fisher_density showed this densifies to
~87% in 12 MNIST tasks). The scale argument: a wider model spreads the representation over
more parameters, so each weight sees less evidence per datum -> it needs MORE data to compress
its σ to the same level. So the fast σ-saturation we see on the tiny 256-256 net is an
artifact of its small capacity; a big model keeps loose σ / sparse importance / free capacity
over a far longer data horizon -- which is exactly when curvature protection stays relevant.

Test: sweep the hidden width, and for each, advance an anchor through the stream measuring the
CUMULATIVE %-important curve vs tasks (= vs data). Prediction: wider -> the curve rises SLOWER
(needs more data to reach the same density). That is the MNIST evidence for "big models need
more data to compress their σ", i.e. the honest bridge to the LLM regime.

Run:  uv run python -m experiments.justification.capacity_scaling --n 3 --N 10 --device cpu
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
from cl_experiments.metrics import accuracy
from cl_experiments.models import MLP
from cl_experiments.repro import (
    Timer,
    get_logger,
    pick_device,
    run_manifest,
    set_seed,
    setup_logging,
)

OUT = Path(__file__).resolve().parents[2] / "results" / "justification" / "capacity_scaling.json"
log = get_logger()
BETA = 16.0
PRIOR = 0.05
FISHER_BATCHES = 30
ANCHOR_EPOCHS = 5          # fixed, comparable moderate training across widths
WIDTHS = [256, 512, 1024]


def _flat(d):
    return torch.cat([d[k].detach().flatten().cpu() for k in sorted(d)])


def _frac_important(importance):
    sig = 1.0 / torch.sqrt(importance + 1.0 / PRIOR**2)
    return (sig < 0.9 * PRIOR).float().mean().item() * 100


def _run_width(width, n, seed, ns, device):
    set_seed(seed)
    perms = anchor_perms(n, seed)
    anchor_train, anchor_tests = anchor_loaders(perms, batch_size=SETTING.batch_size, seed=seed)
    model = MLP(hidden=(width, width)).to(device)
    train_plain(model, anchor_train, epochs=ANCHOR_EPOCHS, lr=SETTING.anchor_lr, device=device)
    min_acc = min(accuracy(model, t, device) for t in anchor_tests)

    gen = torch.Generator().manual_seed(seed)
    f_anchor = diagonal_fisher(model, anchor_train, mode="true", max_batches=FISHER_BATCHES,
                               device=device, generator=gen)
    imp_cum = len(anchor_train.dataset) * _flat(f_anchor)
    pct = [_frac_important(imp_cum)]

    learner = build_learner("blr", model, anchor_train, device, **{**BLR_SIG_BASE, "beta": BETA})
    for tr, _te in stream_loaders(n_stream=ns, seed=seed):
        learner.finetune(model, tr, device)
        f_t = diagonal_fisher(model, tr, mode="true", max_batches=FISHER_BATCHES,
                              device=device, generator=gen)
        imp_cum = imp_cum + len(tr.dataset) * _flat(f_t)
        pct.append(_frac_important(imp_cum))
    n_params = sum(p.numel() for p in model.parameters())
    return {"width": width, "n_params": n_params, "anchor_min_acc": min_acc, "cum_pct": pct}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--widths", type=int, nargs="+", default=WIDTHS)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                    help="force a device; 'cpu' is bit-reproducible (MPS+vmap is not)")
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(OUT.parent / "capacity_scaling.log")
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    out = json.loads(OUT.read_text()) if OUT.exists() else {"rows": {}}

    with Timer() as t:
        for w in args.widths:
            key = str(w)
            if key in out["rows"]:
                continue
            r = _run_width(w, args.n, args.seed, args.N, device)
            out["rows"][key] = r
            out.update({"n": args.n, "N": args.N, "seed": args.seed})
            OUT.write_text(json.dumps(out, indent=2))
            log.info("width=%-4d params=%.2fM anchor_min_acc=%.3f | cum%%: %s",
                     w, r["n_params"] / 1e6, r["anchor_min_acc"],
                     " ".join(f"{x:.0f}" for x in r["cum_pct"]))
    run_manifest(OUT.parent / "capacity_scaling.manifest.json",
                 config={"n": args.n, "N": args.N, "seed": args.seed, "widths": args.widths,
                         "beta": BETA, "anchor_epochs": ANCHOR_EPOCHS}, elapsed_s=t.elapsed,
                 device=device)
    _plot(out)


def _plot(o):
    rows = o["rows"]
    cols = ["#6baed6", "#2171b5", "#08306b", "#c6dbef"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    widths = sorted(rows, key=int)
    for i, w in enumerate(widths):
        r = rows[w]
        tasks = list(range(len(r["cum_pct"])))
        ax1.plot(tasks, r["cum_pct"], "-o", color=cols[i % len(cols)], lw=2.4, ms=4,
                 label=f"width {w}  ({r['n_params'] / 1e6:.1f}M)")
    ax1.set(xlabel="tasks seen (= data)", ylabel="cumulative % weights important",
            title="(a) Wider models compress σ SLOWER (importance densifies later)")
    ax1.legend(title="hidden width")
    ax1.grid(alpha=0.3)
    # summary: cumulative %-important at the final task vs params
    xs = [rows[w]["n_params"] / 1e6 for w in widths]
    ys = [rows[w]["cum_pct"][-1] for w in widths]
    ax2.plot(xs, ys, "-o", color="#d62728", lw=2.4, ms=6)
    ax2.set(xlabel="parameters (M)", ylabel=f"% important after {o.get('N', '?')} tasks",
            title="(b) Same data fills a bigger model LESS")
    ax2.grid(alpha=0.3)
    fig.suptitle(f"Bigger model → needs more data to compress its σ to the same density "
                 f"(n={o.get('n')}, seed {o.get('seed')})", fontsize=11)
    fig.tight_layout()
    out_png = OUT.parent / "capacity_scaling.png"
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
