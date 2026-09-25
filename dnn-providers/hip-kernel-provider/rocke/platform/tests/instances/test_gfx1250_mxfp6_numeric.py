# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 MXFP6 numerical cases."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "dtype,matrix_path,route,m,n,k,case,count",
    [
        (dtype, *case)
        for dtype in ("fp6", "bf6")
        for case in [
            ("wmma_scale", "comgr", 16, 16, 128, "all", 8),
            ("wmma_scale16", "comgr", 16, 16, 128, "all", 12),
            ("wmma_scale", "comgr", 32, 48, 256, "mixed", 1),
            ("wmma_scale16", "comgr", 32, 48, 256, "mixed", 1),
            ("wmma_scale", "hip", 32, 48, 256, "mixed", 1),
            ("wmma_scale16", "hip", 32, 48, 256, "mixed", 1),
        ]
    ],
)
def test_numeric(numeric_case, dtype, matrix_path, route, m, n, k, case, count):
    numeric_case(dtype, matrix_path, route, m, n, k, case, count)


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
