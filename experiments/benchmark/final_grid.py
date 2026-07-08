"""Final comparison grid: methods x anchor size n x seed, on the full 10-task stream.

Uses the tuned configs from cl_experiments.config (single source of truth) and the
reproducibility utilities (seed, logging, run manifest). Resumable: each
(n, seed, method) result is saved immediately and skipped on re-run.

Run:  uv run python -m experiments.benchmark.final_grid --n-list 1 3 5 10 --seeds 0 1 2
      (call repeatedly; it resumes)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cl_experiments.config import SETTING, TUNED_METHODS
from cl_experiments.harness import run_experiment
from cl_experiments.repro import Timer, get_logger, pick_device, run_manifest, setup_logging

RESULTS = Path(__file__).resolve().parents[2] / "results" / "benchmark"
GRID = RESULTS / "grid.json"
METRICS = ["acc", "anchor_final", "stream_final", "forgetting"]
log = get_logger()


def _load() -> dict:
    return json.loads(GRID.read_text()) if GRID.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=list(SETTING.anchor_ns))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SETTING.seeds))
    ap.add_argument("--n-stream", type=int, default=SETTING.n_stream)
    ap.add_argument("--methods", nargs="+", default=list(TUNED_METHODS))
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                    help="force a device; 'cpu' is bit-reproducible (MPS+vmap is not) and "
                         "matches how TUNED_METHODS was tuned")
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    setup_logging(RESULTS / "final_grid.log")
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    log.info("final_grid device=%s n=%s seeds=%s methods=%s",
             device, args.n_list, args.seeds, args.methods)
    grid = _load()

    with Timer() as t:
        for n in args.n_list:
            for seed in args.seeds:
                for label in args.methods:
                    key = f"n{n}_seed{seed}_{label}"
                    if key in grid:
                        continue
                    method, kw = TUNED_METHODS[label]
                    m, _ = run_experiment(method, n, seed=seed, n_stream=args.n_stream,
                                          device=device, method_kwargs=dict(kw))
                    grid[key] = {"n": n, "seed": seed, "method": label,
                                 **{k: getattr(m, k) for k in
                                    ["acc", "bwt", "learning_acc", "forgetting",
                                     "anchor_final", "anchor_forgetting", "stream_final"]}}
                    GRID.write_text(json.dumps(grid, indent=2))
                    log.info("%-28s ACC=%.4f ret=%.4f forget=%.4f",
                             key, m.acc, m.anchor_final, m.forgetting)

    # aggregate mean +/- std over seeds, per (n, method)
    log.info("%3s %-13s %15s %15s %15s", "n", "method", "ACC", "retention", "forget")
    for n in args.n_list:
        for label in args.methods:
            rows = [v for v in grid.values() if v["n"] == n and v["method"] == label]
            if not rows:
                continue

            def ms(k, _rows=rows):
                x = torch.tensor([r[k] for r in _rows])
                return x.mean().item(), x.std(0, False).item()

            a, r, f = ms("acc"), ms("anchor_final"), ms("forgetting")
            log.info("%3d %-13s %.3f+-%.3f    %.3f+-%.3f    %.3f+-%.3f",
                     n, label, a[0], a[1], r[0], r[1], f[0], f[1])

    run_manifest(RESULTS / "final_grid.manifest.json",
                 config={"n_list": args.n_list, "seeds": args.seeds,
                         "n_stream": args.n_stream, "tuned": TUNED_METHODS},
                 elapsed_s=t.elapsed, device=device)


if __name__ == "__main__":
    main()
