# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Mixed A/B matrix-format numerical cases with E8M0 scales."""

import subprocess
import sys
from itertools import product

import pytest

_MODULE = "rocke.examples.gfx1250.gemm.mixed_scaled_gemm"

# Every ordered mixed pair with E8M0 through both compiler routes.
_FORMATS = ("fp8e4m3", "bf8e5m2", "fp6", "bf6", "fp4")
_MIXED = [(a, b) for a, b in product(_FORMATS, repeat=2) if a != b]


@pytest.mark.parametrize("matrix_path", ["wmma_scale", "wmma_scale16"])
@pytest.mark.parametrize(
    "a,b,route",
    [(*case, route) for case in _MIXED for route in ("comgr", "hip")],
)
def test_mixed_scaled_wmma_numeric(gpu_env, matrix_path, a, b, route):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            _MODULE,
            "--dtype-a",
            a,
            "--dtype-b",
            b,
            "--matrix-path",
            matrix_path,
            "--compile-route",
            route,
            "--m",
            "32",
            "--n",
            "48",
            "--k",
            "256",
            "--case",
            "all",
        ],
        env=gpu_env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    count = 20 if matrix_path == "wmma_scale16" else 12
    assert f"PASS: verified {count} cases" in output, output
    assert output.count("bad=0") == count, output
    print(output, end="")
