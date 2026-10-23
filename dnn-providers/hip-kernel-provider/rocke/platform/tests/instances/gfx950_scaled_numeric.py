# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Bounded gfx950 compiler and hardware numeric validation helpers.

Missing compiler tools skip numeric validation.
"""

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

FORMATS = ("fp8", "bf8", "fp6", "bf6", "fp4")


def require_gpu():
    if not Path("/dev/kfd").exists():
        if os.environ.get("ROCKE_REQUIRE_GFX950") == "1":
            pytest.fail("gfx950 numeric validation requires /dev/kfd")
        pytest.skip("requires an AMD GPU")
    from rocke.runtime.hip_module import get_device_arch

    arch = get_device_arch(0)
    if arch != "gfx950":
        if os.environ.get("ROCKE_REQUIRE_GFX950") == "1":
            pytest.fail(f"requires gfx950; found {arch}")
        pytest.skip(f"requires gfx950; found {arch}")


def _require_tool(name: str) -> str:
    tool = shutil.which(name)
    if tool is None:
        pytest.skip(f"{name} is required for gfx950 numeric validation")
    return tool


def check_isa(artifact, spec):
    objdump = _require_tool("llvm-objdump")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kernel.hsaco"
        path.write_bytes(artifact.hsaco)
        if artifact.hsaco.startswith(b"__CLANG_OFFLOAD_BUNDLE__"):
            bundler = shutil.which("clang-offload-bundler")
            assert bundler, "clang-offload-bundler is required for HIP ISA evidence"
            listing = subprocess.run(
                [bundler, "--list", "--type=o", f"--input={path}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            targets = [x for x in listing.stdout.splitlines() if "gfx950" in x]
            assert len(targets) == 1, listing.stdout
            device = Path(tmp) / "device.hsaco"
            subprocess.run(
                [
                    bundler,
                    "--unbundle",
                    "--type=o",
                    f"--targets={targets[0]}",
                    f"--input={path}",
                    f"--output={device}",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=60,
            )
            path = device
        proc = subprocess.run(
            [objdump, "-d", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
    instructions = [
        line
        for line in proc.stdout.splitlines()
        if "v_mfma_scale_f32_32x32x64_f8f6f4" in line
    ]
    assert len(instructions) == spec.K // 64, proc.stdout
    from rocke.core.arch.mfma_scale import MATRIX_FORMATS

    fa, fb = MATRIX_FORMATS[spec.dtype_a], MATRIX_FORMATS[spec.dtype_b]
    for line in instructions:
        # Omitted selectors encode format zero. Matrix VGPR tuples are 8/6/4 words;
        # the final two operands must be independent VGPR scale sources.
        assert (
            int(re.search(r"cbsz:(\d+)", line).group(1)) == fa
            if "cbsz:" in line
            else fa == 0
        )
        assert (
            int(re.search(r"blgp:(\d+)", line).group(1)) == fb
            if "blgp:" in line
            else fb == 0
        )
        tuples = re.findall(r"(?:a|v)\[(\d+):(\d+)\]", line)
        assert [int(hi) - int(lo) + 1 for lo, hi in tuples[:3]] == [
            16,
            (8, 8, 6, 6, 4)[fa],
            (8, 8, 6, 6, 4)[fb],
        ], line
        assert re.search(r", v\d+, v\d+(?:\s|$)", line), line
    return proc.stdout


def run_numeric(monkeypatch, a, b, output, route, M, N, K, cases=None):
    require_gpu()
    # Check dependencies before compiling or importing the numeric harness.
    _require_tool("llvm-objdump")
    if route == "hip":
        _require_tool("hipcc")
    from rocke.examples.gfx950.gemm import block_scaled_gemm_verify as verifier
    from rocke.instances.gfx950.block_scaled_gemm import BlockScaledGemmSpec

    spec = BlockScaledGemmSpec("numeric_scaled", M, N, K, a, b, output)
    cases = cases or (
        ("neutral", "a-only", "b-only", "mixed", "positive")
        + tuple(f"group-{i}" for i in range(K // 32))
    )
    key = "compile_kernel" if route == "comgr" else "compile_kernel_via_hipcc"
    original = getattr(verifier, key)

    def compile_checked(kernel, **kwargs):
        art = original(kernel, **kwargs)
        check_isa(art, spec)
        return art

    monkeypatch.setattr(verifier, key, compile_checked)
    assert verifier.run_cases(spec, cases, compile_route=route) == len(cases)
