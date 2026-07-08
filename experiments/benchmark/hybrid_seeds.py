"""Does BLR+replay dominate plain replay -- across the whole beta range AND seeds?

This is the data-storing analogue of ``frontier.py``: when rehearsal data IS
available, is BLR-on-replay-batches strictly better than plain replay? ``final_grid``
already compares them, but only at ONE tuned operating point each; ``frontier`` sweeps
beta but only for the data-free methods (replay-family are single points there). This
script fills the gap: it sweeps the BLR step beta for ``blr_replay`` (tracing its
retention<->plasticity curve) and pits it against the plain-replay point, repeated over
seeds so the "dominates" claim is not a single-beta / single-seed artefact.

Read-off: at matched plasticity the blr_replay curve sits ABOVE plain replay
(higher retention), robustly across seeds.

Resumable (results/benchmark/hybrid_seeds.json), tuned configs from cl_experiments.config.

Run:  uv run python -m experiments.benchmark.hybrid_seeds --seeds 0 1 2 --device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cl_experiments.config import BLR_SIG_BASE, REPLAY_BASE, SETTING, TUNED_METHODS
from cl_experiments.harness import run_experiment
from cl_experiments.repro import Timer, get_logger, pick_device, run_manifest, setup_logging

OUT = Path(__file__).resolve().parents[2] / "results" / "benchmark" / "hybrid_seeds.json"
log = get_logger()

# Plain replay as a single tuned point; blr_replay as a beta sweep (its dial).
CONFIGS = {
    "replay": TUNED_METHODS["replay"],
    "blr_replay_lr12": ("blr_replay", {**BLR_SIG_BASE, **REPLAY_BASE, "beta": 12.0}),
    "blr_replay_lr16": ("blr_replay", {**BLR_SIG_BASE, **REPLAY_BASE, "beta": 16.0}),
    "blr_replay_lr20": ("blr_replay", {**BLR_SIG_BASE, **REPLAY_BASE, "beta": 20.0}),
    "blr_replay_lr24": ("blr_replay", {**BLR_SIG_BASE, **REPLAY_BASE, "beta": 24.0}),
}
BETA = {"blr_replay_lr12": 12, "blr_replay_lr16": 16, "blr_replay_lr20": 20,
        "blr_replay_lr24": 24}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SETTING.seeds))
    ap.add_argument("--n-stream", type=int, default=SETTING.n_stream)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                    help="force a device; 'cpu' is bit-reproducible and matches TUNED_METHODS")
    args = ap.parse_args()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(OUT.parent / "hybrid_seeds.log")
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    out = json.loads(OUT.read_text()) if OUT.exists() else {}

    with Timer() as t:
        for seed in args.seeds:
            for label, (method, kw) in CONFIGS.items():
                key = f"{label}_seed{seed}"
                if key in out:
                    continue
                m, _ = run_experiment(method, args.n, seed=seed, n_stream=args.n_stream,
                                      device=device, method_kwargs=dict(kw))
                out[key] = {"label": label, "seed": seed, "acc": m.acc,
                            "anchor_final": m.anchor_final, "stream_final": m.stream_final,
                            "forgetting": m.forgetting}
                OUT.write_text(json.dumps(out, indent=2))
                log.info("%-24s ACC=%.3f ret=%.3f pl=%.3f forget=%.3f",
                         key, m.acc, m.anchor_final, m.stream_final, m.forgetting)

    log.info("%-16s %13s %13s %13s %13s", "config", "ACC", "retention", "plastic", "forget")
    for label in CONFIGS:
        rows = [v for v in out.values() if v["label"] == label]
        if not rows:
            continue

        def ms(k, _rows=rows):
            x = torch.tensor([r[k] for r in _rows])
            return x.mean().item(), x.std(0, False).item()

        a, r, p, f = ms("acc"), ms("anchor_final"), ms("stream_final"), ms("forgetting")
        log.info("%-16s %.3f+-%.3f  %.3f+-%.3f  %.3f+-%.3f  %.3f+-%.3f",
                 label, a[0], a[1], r[0], r[1], p[0], p[1], f[0], f[1])

    run_manifest(OUT.parent / "hybrid_seeds.manifest.json",
                 config={"n": args.n, "seeds": args.seeds, "configs": CONFIGS}, elapsed_s=t.elapsed)
    _plot(out, args.n, args.seeds)


def _plot(out, n, seeds):
    """Retention<->plasticity: blr_replay beta-curve vs the plain-replay point,
    mean±std over seeds. Up-right = better."""
    def agg(label, k):
        xs = [v[k] for v in out.values() if v["label"] == label]
        x = torch.tensor(xs)
        return (x.mean().item(), x.std(0, False).item()) if xs else (float("nan"), 0.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    # blr_replay beta sweep as a curve (sorted by beta)
    labels = sorted(BETA, key=BETA.get)
    pl = [agg(lb, "stream_final") for lb in labels]
    ret = [agg(lb, "anchor_final") for lb in labels]
    ax.errorbar([p[0] for p in pl], [r[0] for r in ret],
                xerr=[p[1] for p in pl], yerr=[r[1] for r in ret],
                fmt="-o", color="#6ba3e0", lw=2.4, ms=6, capsize=3,
                label="BLR+replay — β sweep")
    for lb, p, r in zip(labels, pl, ret, strict=True):
        ax.annotate(f"β={BETA[lb]}", (p[0], r[0]), fontsize=7,
                    textcoords="offset points", xytext=(4, -10))
    # plain replay as a single point
    rp_pl, rp_ret = agg("replay", "stream_final"), agg("replay", "anchor_final")
    ax.errorbar(rp_pl[0], rp_ret[0], xerr=rp_pl[1], yerr=rp_ret[1],
                fmt="*", color="#2a78d6", ms=16, capsize=3, label="plain replay")
    ax.set(xlabel="plasticity (acc on new stream tasks)",
           ylabel="retention (acc on anchor tasks)",
           title=f"BLR+replay vs plain replay (n={n}, {len(seeds)} seeds); up-right = better")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = OUT.parent / "hybrid_seeds.png"
    fig.savefig(out_png, dpi=120)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
