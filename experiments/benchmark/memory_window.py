"""One method, every memory regime: retention vs recency across the beta dial.

The point (the framing this figure closes with evidence): the SAME BLR update spans a
continuum from "super-memory" to "recency-memory", selected by the step-size beta -- not
two separate algorithms but two operating points.

We read it straight off the accuracy matrix R: at the END of the stream, plot each task's
final accuracy against its RECENCY (how many tasks ago it was learned; 0 = most recent
stream task, larger = older, anchor tasks are the oldest). Then:
  * small beta  -> the curve is ~flat and high: the model still remembers far back
                   (super-memory / fixed protection).
  * large beta  -> the curve decays with recency: only the recent tasks survive
                   (a recency window -- graceful, not catastrophic).
  * canonical BLR fixes the protected zone at the anchor (a fixed memory); blr_online
                   re-consolidates each task, so its window SLIDES with the stream.

This is a trade-off view, not a free lunch: beta chooses where on the memory<->plasticity
continuum you sit; the value is that one mechanism reaches all of them.

Resumable (results/benchmark/memory_window.json).

Run:  uv run python -m experiments.benchmark.memory_window --n 3 --N 20 --seed 0 --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cl_experiments.config import BLR_SIG_BASE
from cl_experiments.harness import run_experiment
from cl_experiments.repro import Timer, get_logger, pick_device, run_manifest, setup_logging

OUT = Path(__file__).resolve().parents[2] / "results" / "benchmark" / "memory_window.json"
log = get_logger()

# canonical BLR at three betas (the super-memory -> recency dial) + blr_online at two
# (the sliding window). Same sigma base; only beta / the online re-consolidation differ.
CONFIGS = {
    "blr β=4 (retentive)": ("blr", {**BLR_SIG_BASE, "beta": 4.0}),
    "blr β=12": ("blr", {**BLR_SIG_BASE, "beta": 12.0}),
    "blr β=24 (plastic)": ("blr", {**BLR_SIG_BASE, "beta": 24.0}),
    "blr_online β=8": ("blr_online", {**BLR_SIG_BASE, "beta": 8.0}),
    "blr_online β=24": ("blr_online", {**BLR_SIG_BASE, "beta": 24.0}),
}


def _profile(R, n_anchor):
    """From the accuracy matrix, return (recency, final_acc) per task at the last event.

    recency = events since learned; anchor tasks (learned at event 0) are the oldest. The
    anchor tasks are collapsed to their MEAN at the max recency (they share one age)."""
    last = len(R) - 1  # == N (stream length); row 0 is post-anchor
    stream = []
    for j in range(n_anchor, len(R[0])):
        learned = j - n_anchor + 1
        stream.append((last - learned, R[last][j]))  # (recency, final acc)
    anchor_mean = sum(R[last][:n_anchor]) / n_anchor
    return {"anchor_recency": last, "anchor_acc": anchor_mean,
            "stream": sorted(stream)}  # ascending recency (0 = newest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--N", type=int, default=20, help="stream length (recency axis resolution)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                    help="force a device; 'cpu' is bit-reproducible and matches TUNED_METHODS")
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(OUT.parent / "memory_window.log")
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    out = json.loads(OUT.read_text()) if OUT.exists() else {}

    with Timer() as t:
        for label, (method, kw) in CONFIGS.items():
            key = f"{label}_N{args.N}_seed{args.seed}"
            if key in out:
                continue
            m, R = run_experiment(method, args.n, seed=args.seed, n_stream=args.N,
                                  device=device, method_kwargs=dict(kw))
            prof = _profile(R, args.n)
            out[key] = {"label": label, "N": args.N, "seed": args.seed,
                        "acc": m.acc, "anchor_final": m.anchor_final,
                        "stream_final": m.stream_final, **prof}
            OUT.write_text(json.dumps(out, indent=2))
            log.info("%-22s ACC=%.3f ret=%.3f pl=%.3f | anchor(age %d)=%.3f",
                     label, m.acc, m.anchor_final, m.stream_final,
                     prof["anchor_recency"], prof["anchor_acc"])
    run_manifest(OUT.parent / "memory_window.manifest.json",
                 config={"n": args.n, "N": args.N, "seed": args.seed, "configs": list(CONFIGS)},
                 elapsed_s=t.elapsed, device=device)
    _plot(out, args.n, args.N, args.seed)


# canonical BLR: greens dark(retentive)->light(plastic); blr_online: cyans.
STYLE = {
    "blr β=4 (retentive)": ("#1a7a1a", "-"),
    "blr β=12": ("#41ab5d", "-"),
    "blr β=24 (plastic)": ("#a1d99b", "-"),
    "blr_online β=8": ("#0e7c8c", "--"),
    "blr_online β=24": ("#5fd0de", "--"),
}


def _plot(out, n, N, seed):
    """Final accuracy vs recency: flat = super-memory, decaying = recency window."""
    rows = {v["label"]: v for v in out.values() if v.get("N") == N and v.get("seed") == seed}
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for label, (col, ls) in STYLE.items():
        v = rows.get(label)
        if not v:
            continue
        xs = [r for r, _ in v["stream"]] + [v["anchor_recency"]]
        ys = [a for _, a in v["stream"]] + [v["anchor_acc"]]
        ax.plot(xs, ys, marker="o", ls=ls, color=col, lw=2.2, ms=4, label=label)
    ax.axvspan(N - 0.4, N + 0.4, color="#eee", zorder=0)
    ax.text(N, ax.get_ylim()[0], " anchor\n (oldest)", fontsize=7, color="#777", va="bottom", ha="center")
    ax.set(xlabel="recency = tasks since learned  (0 = most recent → older →)",
           ylabel="final accuracy on that task",
           title=f"One method, every memory regime (n={n}, N={N}, seed {seed})\n"
                 f"flat = super-memory · decaying with recency = recency window · "
                 f"β dials between them")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = OUT.parent / "memory_window.png"
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
