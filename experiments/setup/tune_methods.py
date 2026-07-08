"""Per-method hyperparameter tuning on the fixed setting, then a fair comparison.

Each method gets its OWN Optuna (TPE) search maximising ACC (final average
accuracy over all tasks) on the n-anchor -> 10-task stream, so no method is
handicapped by untuned knobs. Uses the cached moderate anchor. BLR-laplace tunes
with the MC ("true") Fisher for speed (validated ~ the exact expected Fisher).

Tuning is on a single seed; the winners should later be re-checked over seeds.

Run:  uv run python -m experiments.setup.tune_methods --n 3 --trials 20
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import optuna
import torch

from cl_experiments.harness import run_experiment
from cl_experiments.repro import pick_device

RESULTS = Path(__file__).resolve().parents[2] / "results" / "tune"


def _suggest(method: str, t: optuna.Trial) -> dict:
    if method == "naive":
        return {"lr": t.suggest_float("lr", 3e-4, 3e-3, log=True)}
    if method in ("ewc", "ewc_online"):  # same search space; online recomputes Fisher per task
        return {
            "ewc_lambda": t.suggest_float("ewc_lambda", 1e2, 1e6, log=True),
            "lr": t.suggest_float("lr", 3e-4, 3e-3, log=True),
        }
    if method == "replay":
        return {
            "buffer_size": t.suggest_int("buffer_size", 200, 10000, log=True),
            "replay_bs": t.suggest_int("replay_bs", 32, 256),
            "lr": t.suggest_float("lr", 3e-4, 3e-3, log=True),
        }
    if method == "blr_const":
        return {
            "sigma_mode": "const",
            "sigma_const": t.suggest_float("sigma_const", 0.01, 0.3, log=True),
            "sigma_prior": t.suggest_float("sigma_prior", 0.01, 0.3, log=True),
            "n_samples": t.suggest_float("n_samples", 1e3, 1e6, log=True),
            "beta": t.suggest_float("beta", 1.0, 30.0, log=True),
            "rho": t.suggest_float("rho", 0.02, 0.5, log=True),
        }
    if method == "blr_laplace":
        return {
            "sigma_mode": "laplace",
            "fisher_mode": "true",  # MC, fast; ~ expected
            "sigma_prior": t.suggest_float("sigma_prior", 0.01, 0.5, log=True),
            "n_samples": t.suggest_float("n_samples", 1e3, 1e6, log=True),
            "beta": t.suggest_float("beta", 1.0, 30.0, log=True),
            "rho": t.suggest_float("rho", 0.02, 0.5, log=True),
        }
    if method in ("blr_online", "blr_online_replay"):
        # online BLR: per-task re-consolidation with continuous sigma-EMA toward the
        # accumulated-precision floor. beta ranges wider (online tolerates higher beta).
        # blr_online_replay also tunes the rehearsal buffer.
        kw = {
            "sigma_mode": "laplace",
            "fisher_mode": "true",
            "sigma_prior": t.suggest_float("sigma_prior", 0.01, 0.5, log=True),
            "n_samples": t.suggest_float("n_samples", 1e3, 1e6, log=True),
            "beta": t.suggest_float("beta", 1.0, 50.0, log=True),
            "rho": t.suggest_float("rho", 0.02, 0.5, log=True),
        }
        if method == "blr_online_replay":
            kw["buffer_size"] = t.suggest_int("buffer_size", 200, 10000, log=True)
            kw["replay_bs"] = t.suggest_int("replay_bs", 32, 256)
        return kw
    if method == "blr_replay":
        return {
            "sigma_mode": "laplace",
            "fisher_mode": "true",
            "sigma_prior": t.suggest_float("sigma_prior", 0.01, 0.5, log=True),
            "n_samples": t.suggest_float("n_samples", 1e3, 1e6, log=True),
            "beta": t.suggest_float("beta", 1.0, 30.0, log=True),
            "rho": t.suggest_float("rho", 0.02, 0.5, log=True),
            "buffer_size": t.suggest_int("buffer_size", 200, 10000, log=True),
            "replay_bs": t.suggest_int("replay_bs", 32, 256),
        }
    raise ValueError(method)


# method label -> actual build_learner method name
METHOD_NAME = {
    "naive": "naive", "ewc": "ewc", "ewc_online": "ewc_online", "replay": "replay",
    "blr_const": "blr", "blr_laplace": "blr", "blr_online": "blr_online",
    "blr_replay": "blr_replay", "blr_online_replay": "blr_online_replay",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-stream", type=int, default=10)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--fisher-batches", type=int, default=100)
    ap.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto",
                    help="force a device; 'cpu' is bit-reproducible (MPS+vmap is not)")
    ap.add_argument("--methods", nargs="+",
                    default=["naive", "replay", "ewc", "ewc_online", "blr_const",
                             "blr_laplace", "blr_online", "blr_replay", "blr_online_replay"])
    ap.add_argument("--force", action="store_true", help="re-tune methods already in the JSON")
    args = ap.parse_args()
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"device={device}  n={args.n}  stream={args.n_stream}  trials={args.trials}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"tuned_n{args.n}_stream{args.n_stream}_seed{args.seed}.json"
    prog_path = RESULTS / f"progress_n{args.n}_stream{args.n_stream}_seed{args.seed}.txt"
    # Optuna studies are PERSISTED to SQLite so short runs (this sandbox kills bg jobs
    # after ~1-2 min) accumulate trials into the SAME study across restarts, instead of
    # a mid-method kill losing all progress. A method is "done" when its study holds
    # >= args.trials COMPLETE trials. The JSON is the human-readable best-per-method view.
    storage = f"sqlite:///{RESULTS / f'optuna_n{args.n}_stream{args.n_stream}_seed{args.seed}.db'}"
    summary = json.loads(out_path.read_text()) if out_path.exists() else {}
    complete = optuna.trial.TrialState.COMPLETE
    with open(prog_path, "a") as f:
        f.write(f"--- run: methods={args.methods} trials={args.trials} "
                f"n_stream={args.n_stream} (have={list(summary)}) ---\n")

    def _record(label, study):
        bt = study.best_trial
        n_done = len([t for t in study.trials if t.state == complete])
        summary[label] = {"ACC": bt.value, "params": bt.params, "metrics": bt.user_attrs,
                          "trials_done": n_done}
        out_path.write_text(json.dumps(summary, indent=2))

    for label in args.methods:
        method = METHOD_NAME[label]
        study = optuna.create_study(direction="maximize", study_name=label, storage=storage,
                                    load_if_exists=True,
                                    sampler=optuna.samplers.TPESampler(seed=args.seed))
        n_done = len([t for t in study.trials if t.state == complete])
        # also honour a pre-persistence result already in the JSON (the first 5 methods
        # were tuned before SQLite existed, so their trials aren't in the study DB).
        json_done = summary.get(label, {}).get("trials_done", 0)
        if (n_done >= args.trials or json_done >= args.trials) and not args.force:
            src = f"{n_done} trials" if n_done >= args.trials else f"{json_done} (json)"
            print(f"[{label:13s}] already tuned ({src}), skipping")
            if n_done >= args.trials:
                _record(label, study)
            continue
        remaining = args.trials - n_done

        def log_cb(study, trial, _l=label):
            v = trial.value if trial.value is not None else float("nan")
            try:
                bt = study.best_trial  # raises if no trial has completed yet (all failed)
            except ValueError:
                with open(prog_path, "a") as f:
                    f.write(f"{_l} trial {trial.number:2d} ACC={v:.4f} best=none (all failed)\n")
                return
            with open(prog_path, "a") as f:
                f.write(f"{_l} trial {trial.number:2d} ACC={v:.4f} best={bt.value:.4f}\n")
            _record(_l, study)  # save best-so-far after EVERY trial

        def objective(t: optuna.Trial, _m=method, _l=label) -> float:
            kw = _suggest(_l, t)
            # every Fisher-using method takes fisher_batches; only naive/replay/blr_const don't.
            if _l not in ("naive", "replay", "blr_const"):
                kw["fisher_batches"] = args.fisher_batches
            m, _ = run_experiment(_m, args.n, seed=args.seed, n_stream=args.n_stream,
                                  device=device, method_kwargs=kw)
            for k, v in asdict(m).items():
                t.set_user_attr(k, v)
            return m.acc

        # catch: a single diverged trial (e.g. exploding BLR step) must not kill the sweep;
        # it is marked failed and tuning continues.
        study.optimize(objective, n_trials=remaining, show_progress_bar=False,
                       callbacks=[log_cb], catch=(Exception,))
        b = study.best_trial
        print(f"[{label:13s}] best ACC={b.value:.4f}  "
              f"retention={b.user_attrs['anchor_final']:.4f}  "
              f"forget={b.user_attrs['forgetting']:.4f}  params={b.params}")

    print(f"\n{'method':14s}  {'ACC':>7s}  {'retention':>9s}  {'stream':>7s}  {'forget':>7s}")
    for label, s in summary.items():
        me = s["metrics"]
        print(f"{label:14s}  {s['ACC']:7.4f}  {me['anchor_final']:9.4f}  "
              f"{me['stream_final']:7.4f}  {me['forgetting']:7.4f}")
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
