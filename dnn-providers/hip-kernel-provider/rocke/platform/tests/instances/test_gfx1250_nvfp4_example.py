# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""NVFP4 entry point retains the existing verifier's spec and case contract."""

from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from rocke.tests.subprocess_launcher import rocke_subprocess_env

from rocke.examples.gfx1250.gemm import (
    block_scaled_gemm_verify,
    nvfp4_gemm,
    nvfp4_gemm_verify,
)
from rocke.instances.gfx1250.nvfp4_gemm import NvFp4GemmSpec


def test_generic_verifier_rejects_tensor_spec_before_device_or_compile(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("tensor-enabled spec reached device discovery or compilation")

    monkeypatch.setattr(block_scaled_gemm_verify, "get_device_arch", unexpected)
    monkeypatch.setattr(block_scaled_gemm_verify, "compile_kernel", unexpected)
    monkeypatch.setattr(
        block_scaled_gemm_verify, "compile_kernel_via_hipcc", unexpected
    )
    spec = NvFp4GemmSpec("tensor", 16, 16, 128).block_spec()
    for route in ("comgr", "hip"):
        with pytest.raises(ValueError, match="nvfp4_gemm_verify.run_cases"):
            block_scaled_gemm_verify.run_cases(spec, ("mixed",), compile_route=route)


def test_nvfp4_example_delegates_cases(monkeypatch):
    calls = []

    def run(spec, cases, *, compile_route):
        calls.append((spec, cases, compile_route))
        return len(cases)

    monkeypatch.setattr(nvfp4_gemm_verify, "run_cases", run)
    assert (
        nvfp4_gemm.main(
            [
                "--m",
                "16",
                "--n",
                "16",
                "--k",
                "128",
                "--dtype-c",
                "fp16",
                "--compile-route",
                "hip",
                "--case",
                "tensor-rounding",
            ]
        )
        == 0
    )
    spec, cases, route = calls[0]
    assert (spec.M, spec.N, spec.K, spec.dtype_c) == (16, 16, 128, "fp16")
    assert cases == ("tensor-rounding",) and route == "hip"
    assert spec.block_spec().tensor_scale


def test_nvfp4_package_exports():
    from rocke.instances import gfx1250
    from rocke.instances.gfx1250 import nvfp4_gemm as instance

    for name in (
        "NvFp4GemmSpec",
        "build_nvfp4_gemm",
        "nvfp4_gemm_grid",
        "nvfp4_gemm_signature",
    ):
        assert name in gfx1250.__all__
        assert getattr(gfx1250, name) is getattr(instance, name)
    assert "is_valid_nvfp4_gemm_spec" in gfx1250.__all__
    assert gfx1250.is_valid_nvfp4_gemm_spec is instance.is_valid_spec


def test_numeric_verifier_subprocess_uses_selected_package(monkeypatch, tmp_path):
    numeric_test = runpy.run_path(
        str(Path(__file__).with_name("test_gfx1250_nvfp4_numeric.py"))
    )["test_nvfp4_numeric"]
    decoy = tmp_path / "rocke"
    decoy.mkdir()
    (decoy / "__init__.py").write_text("raise RuntimeError('imported cwd decoy')\n")
    monkeypatch.chdir(tmp_path)
    env = rocke_subprocess_env()
    real_run = subprocess.run
    module = "rocke.examples.gfx1250.gemm.nvfp4_gemm_verify"
    # Disable site hooks in the negative control so a pre-import cannot mask cwd shadowing.
    raw = real_run(
        [sys.executable, "-S", "-m", module, "--help"],
        env=env,
        capture_output=True,
        text=True,
    )
    assert raw.returncode != 0 and "imported cwd decoy" in raw.stderr
    calls = []

    def probe_launch(command, *, env, capture_output, text, timeout):
        assert capture_output and text and timeout == 300
        module_index = command.index("-m") + 1
        assert command[module_index] == module
        assert command[module_index + 1 :] == [
            "--m",
            "16",
            "--n",
            "16",
            "--k",
            "128",
            "--dtype-c",
            "bf16",
            "--compile-route",
            "comgr",
            "--case",
            "all",
        ]
        # Run the actual launch prefix, but --help requires neither GPU nor ml_dtypes.
        help_result = real_run(
            command[: module_index + 1] + ["--help"],
            env=env,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
        )
        assert help_result.returncode == 0, help_result.stdout + help_result.stderr
        assert "Verify packed NVFP4 GEMM" in help_result.stdout
        calls.append(env)
        # Only the numerical result is stubbed: this host test checks launch identity.
        return subprocess.CompletedProcess(
            command, 0, "PASS: verified 24 cases\n" + "bad=0\n" * 24, ""
        )

    monkeypatch.setattr(subprocess, "run", probe_launch)
    numeric_test(env, "comgr", "bf16", 16, 16, 128)
    assert calls == [env]
