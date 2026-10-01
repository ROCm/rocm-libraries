# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 independent-scale numeric fixtures; distinct from the planned probe."""

import subprocess
import sys

import pytest

_MODULE = "rocke.examples.gfx1250.gemm.scale_formats_gemm"

# Additional scale encodings through COMGR, with representative HIP cases.
_FORMATS = ("fp8e4m3", "bf8e5m2", "fp6", "bf6", "fp4")
_MIXED = (
    [(a, "fp4", "e8m0", scale) for a in _FORMATS[:-1] for scale in ("e5m3", "e4m3")]
    + [("fp4", b, scale, "e8m0") for b in _FORMATS[:-1] for scale in ("e5m3", "e4m3")]
    + [("fp4", "fp4", scale, scale) for scale in ("e5m3", "e4m3")]
)


@pytest.mark.parametrize("matrix_path", ["wmma_scale", "wmma_scale16"])
@pytest.mark.parametrize(
    "a,b,sa,sb,route",
    [(*case, route) for case in _MIXED for route in ("comgr", "hip")],
)
def test_scale_format_wmma_numeric(gpu_env, matrix_path, a, b, sa, sb, route):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            _MODULE,
            "--dtype-a",
            a,
            "--dtype-b",
            b,
            "--scale-dtype-a",
            sa,
            "--scale-dtype-b",
            sb,
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
