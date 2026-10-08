"""Driving LM Studio's `lms` CLI, so a run controls how each model is loaded.

Load parameters decide the numbers. The instance this box had loaded when the
benchmark was written (16384 context, ``--parallel 4``) decoded at 22 tok/s
where the same model at ``--parallel 1`` does ~67 — so measuring "whatever is
loaded" measures the last thing someone happened to do, not the model.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional


class LoadFailed(RuntimeError):
    pass


def lms_bin() -> str:
    return (
        os.getenv("LMS_BIN")
        or shutil.which("lms")
        or str(Path.home() / ".lmstudio" / "bin" / "lms")
    )


def _run(args: list[str], timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [lms_bin(), *args], capture_output=True, text=True, timeout=timeout
    )


def unload_all(timeout: float = 120) -> None:
    _run(["unload", "--all"], timeout)


def loaded(timeout: float = 30) -> list[dict[str, Any]]:
    proc = _run(["ps", "--json"], timeout)
    if proc.returncode != 0:
        return []
    try:
        instances: list[dict[str, Any]] = json.loads(proc.stdout or "[]")
    except ValueError:
        return []
    return instances


def load(
    model: str, *, context_length: int, parallel: int = 1, timeout: float = 900
) -> float:
    """Load ``model`` with the given window and slot count; return seconds taken."""
    args = [
        "load",
        model,
        "-c",
        str(context_length),
        "--parallel",
        str(parallel),
        "-y",
    ]
    started = time.perf_counter()
    try:
        proc = _run(args, timeout)
    except subprocess.TimeoutExpired:
        raise LoadFailed(f"load timed out after {timeout:.0f}s") from None
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        raise LoadFailed(detail[-1] if detail else f"exit {proc.returncode}")
    return time.perf_counter() - started


def instance_of(model: str) -> Optional[dict[str, Any]]:
    for inst in loaded():
        if model in (inst.get("identifier"), inst.get("modelKey")):
            return inst
    return None


def verify_instance(
    model: str, *, context_length: int, parallel: int
) -> dict[str, Any]:
    """Confirm the loaded instance is the one asked for.

    A window or slot count carried over from an earlier session would produce
    numbers that quietly aren't comparable with the rest of the run, so a
    mismatch fails the model rather than being benchmarked.
    """
    inst = instance_of(model)
    if inst is None:
        raise LoadFailed(f"{model} is not loaded")
    actual_ctx = inst.get("contextLength")
    actual_parallel = inst.get("parallel")
    if actual_ctx != context_length or actual_parallel != parallel:
        raise LoadFailed(
            f"loaded as context={actual_ctx} parallel={actual_parallel}, "
            f"wanted context={context_length} parallel={parallel}"
        )
    return inst
