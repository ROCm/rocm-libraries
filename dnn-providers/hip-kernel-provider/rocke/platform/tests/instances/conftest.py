# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared, explicitly requested device fixture for gfx1250 numeric suites."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

_PYROOT = Path(__file__).resolve().parents[2] / "python"


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
