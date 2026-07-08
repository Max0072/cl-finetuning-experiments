"""Sensitivity of the tuned BLR hyper-parameters. Sweep each knob around its value
at a fixed operating point (n=3, full stream, beta=8), reporting retention /
plasticity / forgetting / ACC. Covers every tuned BLR knob so none is a
cherry-picked optimum:
  * sigma_prior   -- a smooth plasticity dial (a movement ceiling), an operating point.
  * rho           -- robust across a 20x range.
  * n_samples     -- the BLR memory window N (sigma is consolidated here, not frozen):
                     a SMOOTH, monotonic retention<->plasticity dial (bigger N -> tighter
                     precision target -> less movement), so the tuned value is an
                     operating point on the frontier, not a knife-edge optimum.
  * fisher_batches -- how many batches estimate the sigma-init Fisher; retention/ACC are
                     FLAT from 10 to 100 batches, so the cheap default is fully justified.

Run:  uv run python -m experiments.justification.sensitivity
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cl_experiments.config import BLR_SIG_BASE, SETTING
from cl_experiments.harness import run_experiment
from cl_experiments.repro import get_logger, pick_device, run_manifest, setup_logging

OUT = Path(__file__).resolve().parents[2] / "results" / "justification" / "sensitivity.json"
log = get_logger()
LR = 8.0
SIGMA_PRIORS = [0.02, 0.035, 0.05, 0.07, 0.10, 0.15]
RHOS = [0.02, 0.04, 0.06, 0.1, 0.2, 0.4]
N_SAMPLES = [4e3, 8e3, 1.6e4, 3.2e4, 6.4e4]   # BLR memory window N (base = 1.6e4)
FISHER_BATCHES = [10, 20, 30, 50, 100]         # batches for the sigma-init Fisher (base = 30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-stream", type=int, default=SETTING.n_stream)
    args = ap.parse_args()
    setup_logging(OUT.parent / "sensitivity.log")
    device = pick_device()
    out = json.loads(OUT.read_text()) if OUT.exists() else {}

    def run(key, overrides):
        if key in out:
            return
        kw = {**BLR_SIG_BASE, "beta": LR, **overrides}
        m, _ = run_experiment("blr", args.n, seed=args.seed, n_stream=args.n_stream,
                              device=device, method_kwargs=kw)
        out[key] = {"acc": m.acc, "anchor_final": m.anchor_final,
                    "stream_final": m.stream_final, "forgetting": m.forgetting}
        OUT.write_text(json.dumps(out, indent=2))
        log.info("%-16s ACC=%.3f ret=%.3f pl=%.3f forget=%.3f",
                 key, m.acc, m.anchor_final, m.stream_final, m.forgetting)

    log.info("=== sigma_prior sweep (rho=%.2f, lr=%.0f) ===", BLR_SIG_BASE["rho"], LR)
    for sp in SIGMA_PRIORS:
        run(f"sp{sp}", {"sigma_prior": sp})
    log.info("=== rho sweep (sigma_prior=%.2f, lr=%.0f) ===", BLR_SIG_BASE["sigma_prior"], LR)
    for rho in RHOS:
        run(f"rho{rho}", {"rho": rho})
    log.info("=== n_samples sweep (memory window N, base=%.0e) ===", BLR_SIG_BASE["n_samples"])
    for ns in N_SAMPLES:
        run(f"nsamples{ns:.0e}", {"n_samples": ns})
    log.info("=== fisher_batches sweep (sigma-init Fisher, base=%d) ===",
             BLR_SIG_BASE["fisher_batches"])
    for fb in FISHER_BATCHES:
        run(f"fisherbatches{fb}", {"fisher_batches": fb})

    run_manifest(OUT.parent / "sensitivity.manifest.json",
                 config={"n": args.n, "lr": LR, "sigma_priors": SIGMA_PRIORS, "rhos": RHOS,
                         "n_samples": N_SAMPLES, "fisher_batches": FISHER_BATCHES})


if __name__ == "__main__":
    main()
