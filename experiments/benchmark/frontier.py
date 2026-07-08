"""Retention <-> plasticity frontier of data-free fine-tuning methods (retention-first).

Our real setting is continual FINE-TUNING: adapt a pretrained (anchor) model to a
stream of new tasks without forgetting its knowledge, data-free. Our BLR methods have
a knob (mu-step size beta) that trades retention (accuracy kept on the anchor tasks)
against plasticity (accuracy on the new stream tasks), so we trace each as a CURVE:
  * BLR (ours)        -- beta sweep
  * BLR-online (ours) -- beta sweep
  * BLR-const (abl.)  -- beta sweep on the SAME BLR dynamics with a flat sigma init, so the
                         curvature-sigma frontier reads directly against the flat one
                         (isolates the contribution; not a single point).
Every other benchmark method (naive, EWC, online-EWC, and the data-storing foils
replay / BLR+replay / BLR-online+replay) is a single tuned point. Up-right wins.

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

from cl_experiments.config import BLR_SIG_BASE, CONST_SIG_BASE, TUNED_METHODS
from cl_experiments.harness import run_experiment
from cl_experiments.repro import pick_device

RESULTS = Path(__file__).resolve().parents[2] / "results" / "benchmark"
MATCHED_PL = 0.70  # plasticity at which the fair-scalar (matched operating point) is read


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

    def metrics_from_run(method, kw):
        # keep all four metrics per operating point: retention/plasticity are the frontier
        # axes; acc/forgetting let us plot every metric vs beta and do the matched-point
        # readout from the SAME runs (beta trades different advantages into different metrics).
        m, _ = run_experiment(method, n, seed=seed, n_stream=ns, device=device, method_kwargs=kw)
        return {"retention": m.anchor_final, "plasticity": m.stream_final,
                "acc": m.acc, "forgetting": m.forgetting}

    # --- tunable frontiers: sweep the BLR step-size beta as a curve ---
    # BLR and BLR-online are our methods; blr_const is the constant-sigma ABLATION on the
    # SAME BLR dynamics (only the sigma init differs) -- swept too, so the reader sees the
    # curvature-sigma frontier vs the flat-sigma frontier, not a single ablation point.
    # 4 swept curves = the 2x2 ablation: {curvature-σ, flat-σ} x {static init, online re-consol}.
    # It isolates WHERE curvature lives: blr vs blr_const (static) and blr_online vs
    # blr_const_online (online, curvature re-used per task).
    for group, method, base in [("blr", "blr", BLR_SIG_BASE),
                                ("blr_online", "blr_online", BLR_SIG_BASE),
                                ("blr_const", "blr", CONST_SIG_BASE),
                                ("blr_const_online", "blr_const_online", CONST_SIG_BASE)]:
        for lr in [1, 4, 8, 16, 24, 40]:
            key = f"{group}_lr{lr}"
            if key in pts and "acc" in pts[key]:  # recompute stale (old-schema) entries
                continue
            r = metrics_from_run(method, {**base, "beta": lr})
            pts[key] = {"group": group, "dial": lr, **r}
            _save(front, pts)
            print(f"{key:16s} ret={r['retention']:.3f} pl={r['plasticity']:.3f} "
                  f"acc={r['acc']:.3f} forget={r['forgetting']:.3f}")

    # --- other benchmark methods as single tuned points (from TUNED_METHODS) ---
    for key in ["naive", "ewc", "ewc_online", "replay", "blr_replay",
                "blr_online_replay"]:
        if key in pts and "acc" in pts[key]:  # recompute stale (old-schema) entries
            continue
        method, kw = TUNED_METHODS[key]
        r = metrics_from_run(method, dict(kw))
        pts[key] = {"group": key, "dial": None, **r}
        _save(front, pts)
        print(f"{key:16s} ret={r['retention']:.3f} pl={r['plasticity']:.3f}")

    _plot(pts, n, seed)
    _plot_metrics_vs_beta(pts, n, seed)
    _matched_readout(pts, n, seed)


def _plot(pts, n, seed):
    # tunable frontiers as curves (our two methods + the constant-sigma ablation, dashed);
    # every other benchmark method as a single point.
    curves = {  # (color, label, linestyle, linewidth)
        "blr": ("#2ca02c", "BLR (ours)", "-", 2.6),
        "blr_online": ("#17becf", "BLR-online (ours)", "-", 2.6),
        "blr_const": ("#bcbd22", "BLR-const (abl.)", "--", 1.8),
        "blr_const_online": ("#7fd6e0", "BLR-const-online (abl.)", "--", 1.8),
    }
    points = {  # (color, marker, label); data-storing foils are hollow
        "naive": ("#7f7f7f", "s", "naive", False),
        "ewc": ("#eb6834", "D", "EWC", False),
        "ewc_online": ("#f0a878", "D", "EWC-online", False),
        "replay": ("#2a78d6", "*", "replay (foil)", True),
        "blr_replay": ("#6ba3e0", "*", "BLR+replay (foil)", True),
        "blr_online_replay": ("#c44e97", "*", "BLR-online+replay (foil)", True),
    }
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for group, (col, lab, ls, lw) in curves.items():
        # connect in beta order (the sweep parameter): plasticity saturates at high
        # beta while retention keeps dropping, so sorting by plasticity would zig-zag.
        g = sorted([v for v in pts.values() if v["group"] == group], key=lambda z: z["dial"])
        if g:
            ax.plot([v["plasticity"] for v in g], [v["retention"] for v in g],
                    marker="o", ls=ls, color=col, label=f"{lab} — β sweep", ms=5, lw=lw)
    for group, (col, mk, lab, hollow) in points.items():
        v = pts.get(group)
        if v:
            ax.scatter(v["plasticity"], v["retention"], s=130, marker=mk, zorder=5,
                       label=lab, edgecolors=col, linewidths=2,
                       facecolors="none" if hollow else col)
    # matched operating point: the vertical slice along which the fair scalar is read
    ax.axvline(MATCHED_PL, color="#999", ls=":", lw=1.2,
               label=f"matched slice (pl={MATCHED_PL})")
    ax.set(xlabel="plasticity (acc on new stream tasks)",
           ylabel="retention (acc on anchor tasks)",
           title=f"Retention–plasticity frontier (n={n}, seed {seed}); up-right = better\n"
                 f"curves = our β-tunable methods · hollow = data-storing foils")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / f"frontier_n{n}_seed{seed}.png", dpi=120)
    print(f"saved {RESULTS}/frontier_n{n}_seed{seed}.png")


# --- (2) each metric vs beta, and (3) the matched-operating-point readout ---------------
# Both read the SAME swept runs. The point: beta trades different advantages into different
# metrics, so a single scalar is misleading -- we show every metric's beta-dependence AND a
# fair comparison at a fixed operating point.

SWEPT = {"blr": ("#2ca02c", "BLR (ours)", "-"),
         "blr_online": ("#17becf", "BLR-online (ours)", "-"),
         "blr_const": ("#bcbd22", "BLR-const (abl.)", "--"),
         "blr_const_online": ("#7fd6e0", "BLR-const-online (abl.)", ":")}


def _plot_metrics_vs_beta(pts, n, seed):
    """Each metric as a function of beta for the three swept curves. Makes explicit that
    beta buys different things in different metrics (retention/forgetting worsen, plasticity
    rises, ACC peaks) -- and that curvature-sigma dominates flat-sigma per-metric at every
    beta. EWC-online is drawn as a horizontal reference (strongest data-free baseline)."""
    panels = [("retention", "Retention (higher=better)"),
              ("plasticity", "Plasticity (higher=better)"),
              ("acc", "ACC (higher=better)"),
              ("forgetting", "Forgetting (lower=better)")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4))
    for ax, (key, ttl) in zip(axes, panels, strict=True):
        for grp, (col, lab, ls) in SWEPT.items():
            g = sorted([v for v in pts.values() if v["group"] == grp], key=lambda z: z["dial"])
            if g:
                ax.plot([v["dial"] for v in g], [v[key] for v in g],
                        marker="o", ls=ls, color=col, label=lab, lw=2.2, ms=4)
        ref = pts.get("ewc_online")
        if ref and key in ref:
            ax.axhline(ref[key], color="#eb6834", ls=":", lw=1.3, label="EWC-online (ref)")
        ax.set(title=ttl, xlabel="β (BLR step size)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Each metric vs β (n={n}, seed {seed}); β trades retention↔plasticity. "
                 f"Curvature-σ ≈ flat-σ (the init washes out under consolidation); "
                 f"the lift is from online re-consolidation", fontsize=11)
    fig.tight_layout()
    out = RESULTS / f"frontier_metrics_n{n}_seed{seed}.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


def _interp(g, x, ykey, xkey="plasticity"):
    """Linear-interpolate ykey at xkey==x along curve g (sorted ascending by xkey).
    Returns (value, clamped) where clamped=True means x is outside the curve's range."""
    xs = [v[xkey] for v in g]
    ys = [v[ykey] for v in g]
    if x <= xs[0]:
        return ys[0], True
    if x >= xs[-1]:
        return ys[-1], True
    for i in range(1, len(xs)):
        if xs[i] >= x:
            t = (x - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1]), False
    return ys[-1], True


def _matched_readout(pts, n, seed, target_pl=MATCHED_PL):
    """Fair scalar: at a FIXED plasticity, interpolate each swept curve's retention /
    forgetting / ACC. Removes the beta-choice confound -- 'holding plasticity equal,
    curvature-σ retains X more and forgets Y less'. Printed + saved to a small json."""
    rows = {}
    print(f"\nMatched operating point (plasticity={target_pl:.2f}); n={n} seed={seed}:")
    print(f"  {'method':20s} {'retention':>10s} {'forgetting':>11s} {'ACC':>7s}")
    for grp, (_c, lab, _ls) in SWEPT.items():
        g = sorted([v for v in pts.values() if v["group"] == grp], key=lambda z: z["plasticity"])
        if len(g) < 2:
            continue
        ret, c1 = _interp(g, target_pl, "retention")
        fgt, c2 = _interp(g, target_pl, "forgetting")
        acc, c3 = _interp(g, target_pl, "acc")
        clamped = c1 or c2 or c3
        rows[grp] = {"retention": ret, "forgetting": fgt, "acc": acc, "clamped": clamped}
        print(f"  {lab:20s} {ret:10.3f} {fgt:11.3f} {acc:7.3f}"
              + ("   (extrapolated: curve doesn't span this plasticity)" if clamped else ""))
    _save(RESULTS / f"frontier_matched_n{n}_seed{seed}.json",
          {"target_plasticity": target_pl, "n": n, "seed": seed, "rows": rows})
    print(f"saved {RESULTS}/frontier_matched_n{n}_seed{seed}.json")


if __name__ == "__main__":
    main()
