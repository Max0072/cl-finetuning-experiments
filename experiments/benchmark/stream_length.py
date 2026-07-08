"""How do methods scale with stream length N (more forgetting pressure)?

Fixed anchor n=3; sweep the continuation-stream length N. Longer streams stress
retention harder.

Our BLR methods have a knob -- the mu-step size beta, the retention<->plasticity
dial (small beta = retains everything, large beta = more plastic). Judging a
knob-having method at a single operating point is misleading, so we show BLR and
BLR-online as a **beta sweep** (a retention end and a plastic end each) and the
baselines / data-storing foils as single tuned points. The point to read off: at
small beta the BLR methods hold retention roughly flat as N grows (no catastrophic
forgetting), while a large-beta point trades that away for plasticity.

Averaged over seeds (mean+-std). Resumable (results/benchmark/stream_length.json).

Run:  uv run python -m experiments.benchmark.stream_length --N 10 20 30 --seeds 0 --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cl_experiments.config import BLR_SIG_BASE, SETTING, TUNED_METHODS
from cl_experiments.harness import run_experiment
from cl_experiments.repro import Timer, get_logger, pick_device, run_manifest, setup_logging

OUT = Path(__file__).resolve().parents[2] / "results" / "benchmark" / "stream_length.json"
log = get_logger()

# Our two headline BLR variants as a beta sweep (retention end + plastic end each),
# the strongest data-free baseline (online-EWC), and the data-storing foils as points.
CONFIGS = {
    "blr β=4": ("blr", {**BLR_SIG_BASE, "beta": 4.0}),               # retention end
    "blr β=16": ("blr", {**BLR_SIG_BASE, "beta": 16.0}),            # balanced/plastic
    "blr_online β=8": ("blr_online", {**BLR_SIG_BASE, "beta": 8.0}),   # retention end
    "blr_online β=24": ("blr_online", {**BLR_SIG_BASE, "beta": 24.0}),  # plastic end
    "ewc_online": TUNED_METHODS["ewc_online"],
    "replay": TUNED_METHODS["replay"],
    "blr_online_replay": TUNED_METHODS["blr_online_replay"],
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SETTING.seeds))
    ap.add_argument("--N", type=int, nargs="+", default=[10, 20, 30])
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                    help="force a device; 'cpu' is bit-reproducible and matches TUNED_METHODS")
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(OUT.parent / "stream_length.log")
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    out = json.loads(OUT.read_text()) if OUT.exists() else {}

    with Timer() as t:
        for ns in args.N:
            for seed in args.seeds:
                for label, (method, kw) in CONFIGS.items():
                    key = f"{label}_N{ns}_seed{seed}"
                    if key in out:
                        continue
                    m, _ = run_experiment(method, args.n, seed=seed, n_stream=ns,
                                          device=device, method_kwargs=dict(kw))
                    out[key] = {"label": label, "N": ns, "seed": seed, "acc": m.acc,
                                "anchor_final": m.anchor_final, "stream_final": m.stream_final,
                                "forgetting": m.forgetting}
                    OUT.write_text(json.dumps(out, indent=2))
                    log.info("%-18s N=%2d seed=%d | ACC=%.3f ret=%.3f pl=%.3f forget=%.3f",
                             label, ns, seed, m.acc, m.anchor_final, m.stream_final, m.forgetting)

    # aggregate mean +/- std over seeds, per (method, N)
    log.info("%-18s %3s  %13s %13s %13s %13s",
             "method", "N", "ACC", "retention", "plastic", "forget")
    for ns in args.N:
        for label in CONFIGS:
            rows = [v for v in out.values()
                    if "seed" in v and v["label"] == label and v["N"] == ns]
            if not rows:
                continue

            def ms(k, _rows=rows):
                x = torch.tensor([r[k] for r in _rows])
                return x.mean().item(), x.std(0, False).item()

            a, r, p, f = ms("acc"), ms("anchor_final"), ms("stream_final"), ms("forgetting")
            log.info("%-18s %3d  %.3f+-%.3f  %.3f+-%.3f  %.3f+-%.3f  %.3f+-%.3f",
                     label, ns, a[0], a[1], r[0], r[1], p[0], p[1], f[0], f[1])
    run_manifest(OUT.parent / "stream_length.manifest.json",
                 config={"n": args.n, "seeds": args.seeds, "N": args.N}, elapsed_s=t.elapsed)
    _plot(out, args.N, args.n)


# Our beta-swept methods: two greens for BLR, two cyans for BLR-online (same hue
# family, dark = retention end, light = plastic end); baselines/foils get their own
# hues. Data-storing foils are dashed.
OURS = ("blr β=4", "blr β=16", "blr_online β=8", "blr_online β=24")
COLORS = {"blr β=4": "#1a7a1a", "blr β=16": "#74c476",
          "blr_online β=8": "#0e7c8c", "blr_online β=24": "#5fd0de",
          "ewc_online": "#eb6834", "replay": "#2a78d6", "blr_online_replay": "#c44e97"}
STORING = ("replay", "blr_online_replay")  # data-storing foils -> dashed


def _plot(out, Ns, n):
    """ACC / retention / forgetting vs stream length N, mean±std over seeds."""
    def agg(label, ns, k):
        xs = [v[k] for v in out.values()
              if "seed" in v and v["label"] == label and v["N"] == ns]
        t = torch.tensor(xs)
        return (t.mean().item(), t.std(0, False).item()) if xs else (float("nan"), 0.0)

    panels = [("acc", "ACC (higher=better)"), ("anchor_final", "Retention (higher=better)"),
              ("forgetting", "Forgetting (lower=better)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (key, ttl) in zip(axes, panels, strict=True):
        for label in CONFIGS:
            means = [agg(label, ns, key)[0] for ns in Ns]
            stds = [agg(label, ns, key)[1] for ns in Ns]
            lw = 2.8 if label in OURS else 1.5  # our beta-swept methods are the stars
            ls = "--" if label in STORING else "-"
            ax.errorbar(Ns, means, yerr=stds, marker="o", ms=5, lw=lw, ls=ls,
                        color=COLORS.get(label, "#888"), label=label, capsize=3)
        ax.set_title(ttl)
        ax.set_xlabel("stream length N (tasks)")
        ax.set_xticks(Ns)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Scaling with stream length (anchor n={n}); BLR shown as β sweep, "
                 f"mean±std over seeds", fontsize=12)
    fig.tight_layout()
    out_png = OUT.parent / "stream_length.png"
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
