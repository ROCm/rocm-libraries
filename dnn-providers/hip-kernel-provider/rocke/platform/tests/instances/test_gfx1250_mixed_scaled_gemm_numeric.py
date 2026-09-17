# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Mixed A/B matrix-format numerical cases with E8M0 scales."""

import subprocess
import sys
from itertools import product

import pytest

_MODULE = "rocke.examples.gfx1250.gemm.block_scaled_gemm_verify"

# Mixed matrix pairs with E8M0 through COMGR and representative FP6 HIP cases.
_FORMATS = ("fp8e4m3", "bf8e5m2", "fp6", "bf6", "fp4")
_MIXED = [(a, b) for a, b in product(_FORMATS, repeat=2) if a != b]


@pytest.mark.parametrize("matrix_path", ["wmma_scale", "wmma_scale16"])
@pytest.mark.parametrize(
    "a,b,route",
    [
        (*case, route)
        for case in _MIXED
        for route in ("comgr", "hip")
        if route == "comgr" or case in (("fp6", "bf6"), ("bf6", "fp6"))
    ],
)
def test_mixed_scaled_wmma_numeric(gpu_env, matrix_path, a, b, route):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            _MODULE,
            "--dtype",
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
            "mixed",
        ],
        env=gpu_env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "PASS: verified 1 cases" in output and output.count("bad=0") == 1, output
    print(output, end="")
