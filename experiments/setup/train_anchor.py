"""Train (or load from cache) and certify anchors for the sweep n in {1,3,5,10}.

Uses `cl_experiments.pipeline.get_anchor`, so anchors are trained once and cached
to results/anchor/*.pt; re-running is instant. Certification (per-task acc +
gradient norm) is printed and gated.

Run:  uv run python -m experiments.setup.train_anchor --n 1 3 5 10
      uv run python -m experiments.setup.train_anchor --n 10 --force   # retrain
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cl_experiments.harness import CACHE_DIR, get_anchor
from cl_experiments.repro import pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="retrain even if cached")
    args = ap.parse_args()
    device = pick_device()
    print(f"device={device}  seed={args.seed}  n={args.n}")

    summary = {}
    for n in args.n:
        print(f"\n=== anchor n={n} ===")
        _, report, _ = get_anchor(n, args.seed, device=device, force=args.force)
        summary[n] = {
            "certified": report.certified(),
            "min_task_acc": report.min_task_acc,
            "mean_task_acc": sum(report.per_task_test_acc) / len(report.per_task_test_acc),
            "grad_norm": report.grad_norm,
            "epochs_run": report.epochs_run,
            "final_train_loss": report.final_train_loss,
        }

    print(f"\n{'n':>3s}  {'cert':>5s}  {'min_acc':>8s}  {'mean_acc':>8s}  {'grad_norm':>10s}  {'epochs':>6s}")
    for n, s in summary.items():
        print(f"{n:3d}  {str(s['certified']):>5s}  {s['min_task_acc']:8.4f}  "
              f"{s['mean_task_acc']:8.4f}  {s['grad_norm']:10.2e}  {s['epochs_run']:6d}")
    Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"summary_seed{args.seed}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary -> {CACHE_DIR}/summary_seed{args.seed}.json")


if __name__ == "__main__":
    main()
