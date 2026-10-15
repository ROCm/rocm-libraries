# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""NVFP4 entry point retains the existing verifier's spec and case contract."""

import pytest

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
