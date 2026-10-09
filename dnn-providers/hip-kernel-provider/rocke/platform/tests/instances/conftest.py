# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Real gfx1250 SCALE/SCALE16 launches; no torch dependency.

Ordinary CPU/other-GPU runs skip. Set ROCKE_REQUIRE_GFX1250=1 on the validation
node to fail instead of silently accepting an all-skipped run. Select the GPU
with HIP_VISIBLE_DEVICES, and use ROCKE_LLVM_FLAVOR=llvm23 with matching COMGR.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


_PYROOT = Path(__file__).resolve().parents[2] / "python"
_MODULE = "rocke.examples.gfx1250.gemm.block_scaled_gemm_verify"


@pytest.fixture(scope="module")
def gpu_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(_PYROOT), env.get("PYTHONPATH", "")])
    # Keep HIP initialization out of pytest collection and bound a hung probe.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "from rocke.runtime.hip_module import get_device_arch; "
            "print(get_device_arch(0))",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    arch = probe.stdout.strip().splitlines()[-1]
    if arch != "gfx1250":
        reason = f"requires visible HIP device 0 = gfx1250; detected {arch}"
        if env.get("ROCKE_REQUIRE_GFX1250") == "1":
            pytest.fail(reason)
        pytest.skip(reason)
    for dependency in ("numpy", "ml_dtypes"):
        assert importlib.util.find_spec(
            dependency
        ), f"install {dependency} for numerics"
    return env


def _run_numeric(gpu_env, dtype, matrix_path, route, m, n, k, case, count):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            _MODULE,
            "--dtype",
            dtype,
            "--matrix-path",
            matrix_path,
            "--compile-route",
            route,
            "--m",
            str(m),
            "--n",
            str(n),
            "--k",
            str(k),
            "--case",
            case,
        ],
        env=gpu_env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert f"PASS: verified {count} cases" in output, output
    assert output.count("bad=0") == count, output
    print(output, end="")


@pytest.fixture
def numeric_case(gpu_env):
    """Run one bounded scaled-GEMM case with the validated GPU environment."""
    from functools import partial

    return partial(_run_numeric, gpu_env)
