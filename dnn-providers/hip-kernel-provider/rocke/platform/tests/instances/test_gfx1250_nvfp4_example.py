# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""NVFP4 entry point retains the existing verifier's spec and case contract."""

from rocke.examples.gfx1250.gemm import nvfp4_gemm, nvfp4_gemm_verify


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
