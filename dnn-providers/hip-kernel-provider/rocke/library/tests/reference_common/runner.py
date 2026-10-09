# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Execute selected frozen/current GPU workers with isolated import roots."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from .numeric import write_json
from .session import _ACTIVE, _NO_TORCH


def run_worker(
    request: dict,
    *,
    runner: Path,
    platform: Path,
    library: Path | None,
    work: Path,
    module: str,
) -> tuple[list[np.ndarray], dict]:
    work.mkdir(parents=True, exist_ok=False)
    request = dict(request, platform_root=str(platform.resolve()))
    paths = [platform.resolve()]
    if library is not None:
        paths.append(library.resolve())
        request["library_root"] = str(library.resolve())
    # The runner lives in tests/, which also contains a dispatch package.
    # Production packages must resolve before those test-only names.
    paths.append(runner.resolve())
    write_json(work / "request.json", request)
    env = dict(os.environ)
    # Never let an inherited PYTHONPATH or user site select the other rocKE.
    env.update(
        PYTHONPATH=os.pathsep.join(map(str, paths)),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    session = _ACTIVE.get()
    if session is not None:
        session.execute(request["mode"], work / "request.json", env, module=module)
    else:
        completed = subprocess.run(
            [
                sys.executable,
                "-s",
                "-c",
                _NO_TORCH
                + "\nimport runpy; module = sys.argv.pop(1); "
                + "runpy.run_module(module, run_name='__main__')",
                module,
                str(work / "request.json"),
                str(work),
            ],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if completed.returncode:
            raise RuntimeError(
                f"GPU reference {request['mode']} worker failed:\n"
                f"{completed.stdout[-4000:]}{completed.stderr[-12000:]}"
            )
    report = json.loads((work / "report.json").read_text())
    if report.get("torch_imported"):
        raise RuntimeError("GPU reference worker imported Torch")
    if report["launches"] != request["repetitions"]:
        raise RuntimeError("required GPU reference launches did not execute")
    with np.load(work / "outputs.npz", allow_pickle=False) as archive:
        outputs = [archive[f"out_{i}"] for i in range(request["repetitions"])]
    return outputs, report
