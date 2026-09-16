# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Real gfx1250 SCALE/SCALE16 launches; no torch dependency.

Ordinary CPU/other-GPU runs skip. Set ROCKE_REQUIRE_GFX1250=1 on the validation
node to fail instead of silently accepting an all-skipped run. Select the GPU
with HIP_VISIBLE_DEVICES, and use ROCKE_LLVM_FLAVOR=llvm23 with matching COMGR.
"""

from __future__ import annotations

import subprocess
import sys
from itertools import product

import pytest


_MODULE = "rocke.examples.gfx1250.gemm.block_scaled_gemm_verify"


@pytest.mark.parametrize(
    "dtype,matrix_path,route,m,n,k,case,count",
    [
        (dtype, *case)
        for dtype in ("fp8e4m3", "bf8e5m2", "fp4", "fp6", "bf6")
        for case in [
            ("wmma_scale", "comgr", 16, 16, 128, "all", 8),
            ("wmma_scale16", "comgr", 16, 16, 128, "all", 12),
            ("wmma_scale", "comgr", 32, 48, 256, "mixed", 1),
            ("wmma_scale16", "comgr", 32, 48, 256, "mixed", 1),
            ("wmma_scale", "hip", 32, 48, 256, "mixed", 1),
            ("wmma_scale16", "hip", 32, 48, 256, "mixed", 1),
            ("wmma", "comgr", 16, 16, 128, "mixed", 1),
        ]
        if dtype == "fp8e4m3" or case[0] != "wmma"
    ],
)
def test_scaled_wmma_numeric(gpu_env, dtype, matrix_path, route, m, n, k, case, count):
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


@pytest.mark.parametrize("dtype", ["fp6", "bf6"])
@pytest.mark.parametrize("matrix_path", ["wmma_scale", "wmma_scale16"])
@pytest.mark.parametrize("route", ["comgr", "hip"])
def test_all_fp6_codes_numeric(gpu_env, dtype, matrix_path, route):
    # All 64 x 64 code products at both sides of the lane-half/K-group boundaries.
    script = """
from rocke.instances.gfx1250.block_scaled_gemm import BlockScaledGemmSpec
from rocke.examples.gfx1250.gemm.block_scaled_gemm_verify import run_cases
import sys
dtype, path, route = sys.argv[1:]
spec = BlockScaledGemmSpec(name='fp6_codes', M=64, N=64, K=128,
    dtype_a=dtype, dtype_b=dtype, matrix_path=path, scale_dtype='e8m0',
    block_k=16 if path == 'wmma_scale16' else 32)
assert run_cases(spec, tuple(f'codes-{k}' for k in (0,31,32,63,64,95,96,127)), compile_route=route) == 8
"""
    result = subprocess.run(
        [sys.executable, "-c", script, dtype, matrix_path, route],
        env=gpu_env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count("bad=0") == 8, output
    print(output, end="")
