"""Retention <-> plasticity frontier of data-free fine-tuning methods (retention-first).

Our real setting is continual FINE-TUNING: adapt a pretrained (anchor) model to a
stream of new tasks without forgetting its knowledge, data-free. Our BLR methods have
a knob (mu-step size beta) that trades retention (accuracy kept on the anchor tasks)
against plasticity (accuracy on the new stream tasks), so we trace each as a CURVE:
  * BLR (ours)        -- beta sweep
  * BLR-online (ours) -- beta sweep
Every other benchmark method (naive, EWC, online-EWC, blr_const, and the data-storing
foils replay / BLR+replay / BLR-online+replay) is a single tuned point. Up-right wins.

Resumable: each point saved to results/benchmark/frontier_n{n}_seed{seed}.json.

Run:  uv run python -m experiments.benchmark.frontier --n 3 --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cl_experiments.config import BLR_SIG_BASE, TUNED_METHODS
from cl_experiments.harness import run_experiment
from cl_experiments.repro import pick_device

RESULTS = Path(__file__).resolve().parents[2] / "results" / "benchmark"


def _load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _save(path, d):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-stream", type=int, default=10)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                    help="force a device; 'cpu' is bit-reproducible and matches TUNED_METHODS")
    args = ap.parse_args()
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    n, seed, ns = args.n, args.seed, args.n_stream
    front = RESULTS / f"frontier_n{n}_seed{seed}.json"
    pts = _load(front)

    def rp_from_run(method, kw):
        m, _ = run_experiment(method, n, seed=seed, n_stream=ns, device=device, method_kwargs=kw)
        return m.anchor_final, m.stream_final

    # --- our tunable frontiers: sweep the BLR step-size beta for BLR and BLR-online ---
    # (each traces a retention<->plasticity curve; other methods are single tuned points)
    for group, method in [("blr", "blr"), ("blr_online", "blr_online")]:
        for lr in [1, 4, 8, 16, 24, 40]:
            key = f"{group}_lr{lr}"
            if key in pts:
                continue
            ret, pl = rp_from_run(method, {**BLR_SIG_BASE, "beta": lr})
            pts[key] = {"group": group, "dial": lr, "retention": ret, "plasticity": pl}
            _save(front, pts)
            print(f"{key:16s} ret={ret:.3f} pl={pl:.3f}")

    # --- other benchmark methods as single tuned points (from TUNED_METHODS) ---
    for key in ["naive", "ewc", "ewc_online", "blr_const", "replay", "blr_replay",
                "blr_online_replay"]:
        if key in pts:
            continue
        method, kw = TUNED_METHODS[key]
        ret, pl = rp_from_run(method, dict(kw))
        pts[key] = {"group": key, "dial": None, "retention": ret, "plasticity": pl}
        _save(front, pts)
        print(f"{key:16s} ret={ret:.3f} pl={pl:.3f}")

    _plot(pts, n, seed)


def _plot(pts, n, seed):
    # our two tunable frontiers as curves; every other benchmark method as a point.
    curves = {"blr": ("#2ca02c", "BLR (ours)"),
              "blr_online": ("#17becf", "BLR-online (ours)")}
    points = {  # (color, marker, label); data-storing foils are hollow
        "naive": ("#7f7f7f", "s", "naive", False),
        "ewc": ("#eb6834", "D", "EWC", False),
        "ewc_online": ("#f0a878", "D", "EWC-online", False),
        "blr_const": ("#98df8a", "^", "BLR-const (abl.)", False),
        "replay": ("#2a78d6", "*", "replay (foil)", True),
        "blr_replay": ("#6ba3e0", "*", "BLR+replay (foil)", True),
        "blr_online_replay": ("#c44e97", "*", "BLR-online+replay (foil)", True),
    }
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for group, (col, lab) in curves.items():
        # connect in beta order (the sweep parameter): plasticity saturates at high
        # beta while retention keeps dropping, so sorting by plasticity would zig-zag.
        g = sorted([v for v in pts.values() if v["group"] == group], key=lambda z: z["dial"])
        if g:
            ax.plot([v["plasticity"] for v in g], [v["retention"] for v in g],
                    "-o", color=col, label=f"{lab} — β sweep", ms=5, lw=2.4)
    for group, (col, mk, lab, hollow) in points.items():
        v = pts.get(group)
        if v:
            ax.scatter(v["plasticity"], v["retention"], s=130, marker=mk, zorder=5,
                       label=lab, edgecolors=col, linewidths=2,
                       facecolors="none" if hollow else col)
    ax.set(xlabel="plasticity (acc on new stream tasks)",
           ylabel="retention (acc on anchor tasks)",
           title=f"Retention–plasticity frontier (n={n}, seed {seed}); up-right = better\n"
                 f"curves = our β-tunable methods · hollow = data-storing foils")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / f"frontier_n{n}_seed{seed}.png", dpi=120)
    print(f"saved {RESULTS}/frontier_n{n}_seed{seed}.png")


if __name__ == "__main__":
    main()
