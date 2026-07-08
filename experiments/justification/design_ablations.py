"""Design ablations: one knob off the canonical BLR at a time (n=3, full stream,
beta=8), to justify each choice. Canonical = n_mc=1, noise_scale=1.

At a fixed step-size these knobs mostly shift WHERE on the retention/plasticity
frontier the run sits (e.g. n_mc=4 buys a bit more plasticity/ACC at 4x compute),
rather than beating the frontier. The canonical choices are the simplest/cheapest
that sit on it. The step-size beta itself is swept in full by frontier.py, so it is
not a single point here.

Run:  uv run python -m experiments.justification.design_ablations
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cl_experiments.config import BLR_SIG_BASE, SETTING
from cl_experiments.harness import run_experiment
from cl_experiments.repro import get_logger, pick_device, run_manifest, setup_logging

OUT = Path(__file__).resolve().parents[2] / "results" / "justification" / "design_ablations.json"
log = get_logger()
LR = 8.0

VARIANTS = {
    "canonical (BLR)": {},
    "n_mc=4": {"n_mc": 4},
    "noise_scale=0.5": {"noise_scale": 0.5},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-stream", type=int, default=SETTING.n_stream)
    args = ap.parse_args()
    setup_logging(OUT.parent / "design_ablations.log")
    device = pick_device()
    out = json.loads(OUT.read_text()) if OUT.exists() else {}

    log.info("design ablations (n=%d, lr=%.0f); one knob off canonical BLR", args.n, LR)
    log.info("%-20s %7s %9s %8s %8s", "variant", "ACC", "retention", "plastic", "forget")
    for name, ov in VARIANTS.items():
        if name not in out:
            kw = {**BLR_SIG_BASE, "beta": LR, **ov}
            m, _ = run_experiment("blr", args.n, seed=args.seed, n_stream=args.n_stream,
                                  device=device, method_kwargs=kw)
            out[name] = {"acc": m.acc, "anchor_final": m.anchor_final,
                         "stream_final": m.stream_final, "forgetting": m.forgetting}
            OUT.write_text(json.dumps(out, indent=2))
        v = out[name]
        log.info("%-20s %7.3f %9.3f %8.3f %8.3f",
                 name, v["acc"], v["anchor_final"], v["stream_final"], v["forgetting"])
    run_manifest(OUT.parent / "design_ablations.manifest.json",
                 config={"n": args.n, "lr": LR, "variants": list(VARIANTS)})


if __name__ == "__main__":
    main()
