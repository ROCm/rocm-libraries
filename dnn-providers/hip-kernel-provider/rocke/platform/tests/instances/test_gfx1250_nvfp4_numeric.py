# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""NVFP4 tensor-scaled numerical fixtures."""

import subprocess
import sys
import pytest

@pytest.mark.parametrize("route", ["comgr", "hip"])
@pytest.mark.parametrize("dtype_c", ["bf16", "fp16"])
@pytest.mark.parametrize("m,n,k", [(16, 16, 128), (32, 48, 256)])
def test_nvfp4_numeric(gpu_env, route, dtype_c, m, n, k):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rocke.examples.gfx1250.gemm.nvfp4_gemm_verify",
            "--m",
            str(m),
            "--n",
            str(n),
            "--k",
            str(k),
            "--dtype-c",
            dtype_c,
            "--compile-route",
            route,
            "--case",
            "all",
        ],
        env=gpu_env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    count = 16 + k // 16
    assert result.returncode == 0, output
    assert f"PASS: verified {count} cases" in output, output
    assert output.count("bad=0") == count, output
    print(output, end="")
