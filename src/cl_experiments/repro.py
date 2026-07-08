"""Reproducibility and logging utilities.

Every experiment should: (1) call ``set_seed`` up front, (2) use ``get_logger`` /
``setup_logging`` instead of print, and (3) write a ``run_manifest`` next to its
results so any output is traceable to a config + seed + git commit + environment.
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]  # single source of truth for the repo root


def set_seed(seed: int, deterministic: bool = True) -> int:
    """Seed python / numpy / torch (+CUDA). ``deterministic`` requests
    deterministic algorithms (warn-only; note MPS + vmap are not fully
    deterministic, so bit-exactness is only guaranteed on CPU/CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        with contextlib.suppress(Exception):  # best effort; some ops lack det. impls
            torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        )
        return bool(out.strip())
    except Exception:  # noqa: BLE001
        return False


def device_str() -> str:
    if torch.cuda.is_available():
        return f"cuda:{torch.cuda.get_device_name(0)}"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(logfile: str | Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure root logging to stdout (+ a file if given). Returns the package logger."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if logfile is not None:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(logfile))
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    return logging.getLogger("cl_experiments")


def get_logger(name: str = "cl_experiments") -> logging.Logger:
    return logging.getLogger(name)


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    return obj


def run_manifest(
    path: str | Path,
    *,
    config: Any,
    seed: int | None = None,
    extra: dict | None = None,
    elapsed_s: float | None = None,
    device: torch.device | str | None = None,
) -> dict:
    """Write a traceability manifest (git, time, env, config, seed) as JSON.

    ``device`` records the device the run ACTUALLY used; pass it explicitly (e.g. the
    ``--device`` the script resolved). Falls back to the auto-detected device only when
    not given -- do not rely on the fallback for reproducibility-critical runs.
    """
    manifest = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "device": str(device) if device is not None else device_str(),
        "torch_version": torch.__version__,
        "python": sys.version.split()[0],
        "seed": seed,
        "elapsed_s": elapsed_s,
        "config": _to_plain(config),
        "extra": extra or {},
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


class Timer:
    """Context manager measuring wall-clock seconds (`.elapsed`)."""

    def __enter__(self) -> Timer:
        self._t0 = time.time()
        self.elapsed = 0.0
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed = time.time() - self._t0
