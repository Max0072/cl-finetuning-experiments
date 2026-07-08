"""Plot the final grid: ACC and forgetting vs anchor size n, mean +/- std over seeds.

Run:  uv run python -m experiments.figures.plot_grid
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

RESULTS = Path(__file__).resolve().parents[2] / "results" / "benchmark"
NS = [1, 3, 5, 10]
# data-free arena (+ the two online variants), data-storing foils drawn dashed.
METHODS = ["naive", "ewc", "ewc_online", "blr_const", "blr_laplace", "blr_online",
           "replay", "blr_replay", "blr_online_replay"]
COLORS = {"naive": "#7f7f7f", "ewc": "#9467bd", "ewc_online": "#c5b0e5",
          "blr_const": "#ff7f0e", "blr_laplace": "#2ca02c", "blr_online": "#17becf",
          "replay": "#1f77b4", "blr_replay": "#8c564b", "blr_online_replay": "#e377c2"}
LABELS = {"naive": "naive", "ewc": "EWC", "ewc_online": "EWC-online",
          "blr_const": "BLR-const (abl.)", "blr_laplace": "BLR (ours)",
          "blr_online": "BLR-online (ours)", "replay": "replay (foil)",
          "blr_replay": "BLR+replay", "blr_online_replay": "BLR-online+replay"}
STORING = {"replay", "blr_replay", "blr_online_replay"}  # data-storing foils -> dashed


def _agg(grid, n, method, key):
    xs = [v[key] for v in grid.values() if v["n"] == n and v["method"] == method]
    t = torch.tensor(xs)
    return t.mean().item(), (t.std(0, False).item() if len(xs) > 1 else 0.0)


def main() -> None:
    grid = json.loads((RESULTS / "grid.json").read_text())
    # four metrics vs anchor size n -- the full retention/plasticity dynamics.
    panels = [("acc", "ACC (higher=better)", "accuracy"),
              ("anchor_final", "Retention (anchor acc, higher=better)", "retention"),
              ("stream_final", "Plasticity (stream acc, higher=better)", "plasticity"),
              ("forgetting", "Forgetting (lower=better)", "forgetting")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (key, ttl, ylab) in zip(axes.flat, panels, strict=True):
        for m in METHODS:
            means = [_agg(grid, n, m, key)[0] for n in NS]
            stds = [_agg(grid, n, m, key)[1] for n in NS]
            lw = 2.8 if m in ("blr_laplace", "blr_online") else 1.3  # our methods are the stars
            ls = "--" if m in STORING else "-"  # data-storing foils dashed
            ax.errorbar(NS, means, yerr=stds, marker="o", ms=5, lw=lw, ls=ls,
                        color=COLORS[m], label=LABELS[m], capsize=3)
        ax.set_title(ttl)
        ax.set_xlabel("anchor size n (tasks known)")
        ax.set_ylabel(ylab)
        ax.set_xticks(NS)
        ax.grid(alpha=0.3)
    axes.flat[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Final grid: 9 methods x anchor size, 10-task Permuted-MNIST stream, "
                 "mean±std over 3 seeds", fontsize=13)
    fig.tight_layout()
    out = RESULTS / "grid.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
