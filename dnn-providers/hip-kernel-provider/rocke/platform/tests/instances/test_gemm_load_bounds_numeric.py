# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""gfx1250 GEMM tail numerics; require the target with ROCKE_REQUIRE_GFX1250=1."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture(scope="module", autouse=True)
def require_gfx1250():
    from rocke.runtime.hip_module import get_device_arch

    arch = get_device_arch(0)
    if arch != "gfx1250":
        if os.environ.get("ROCKE_REQUIRE_GFX1250") == "1":
            pytest.fail(f"required gfx1250, found {arch!r}")
        pytest.skip(f"requires gfx1250, found {arch!r}")


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize("k", [33, 40, 65])
@pytest.mark.parametrize(
    "load_flags",
    [
        ["--direct-to-lds"],
        ["--direct-to-lds", "--dtl-prefetch"],
        ["--tdm", "--tdm-depth", "1"],
        ["--tdm", "--tdm-depth", "2"],
        ["--tdm", "--tdm-depth", "3"],
        ["--tdm", "--tdm-depth", "4"],
    ],
)
def test_gemm_load_bounds_numeric(dtype, k, load_flags, tmp_path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "python")
    env["ROCKE_BACKEND"] = "python"
    env["ROCKE_LLVM_FLAVOR"] = "llvm23"
    # Nine output tiles and two CTAs exercise LDS reuse between persistent tiles.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "rocke.examples.common.universal_gemm_verify",
            "--arch",
            "gfx1250",
            "--dtype",
            dtype,
            "--m",
            "129",
            "--n",
            "130",
            "--k",
            str(k),
            "--persistent-ctas",
            "2",
            "--epilogue",
            "cshuffle",
            "--lds-k-pad",
            "8",
            "--output-dir",
            str(tmp_path),
            *load_flags,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (result.stdout + result.stderr)[-3000:]
    assert "PASS" in result.stdout
