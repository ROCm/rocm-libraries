# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Both homogeneous eight-bit formats are usable from the focused example."""

import argparse

import numpy as np
import pytest

from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.dtypes import normalize_dtype
from rocke.examples.gfx1250.gemm import _scaled_gemm_example, mxfp8_gemm
from rocke.examples.gfx1250.gemm.block_scaled_gemm_verify import (
    make_case_inputs,
    reference_result,
)
from rocke.instances.gfx1250.block_scaled_gemm import build_block_scaled_gemm


@pytest.mark.parametrize(
    "dtype,selector,ml_name", [("fp8", 0, "float8_e4m3fn"), ("bf8", 1, "float8_e5m2")]
)
@pytest.mark.parametrize("path", ["wmma_scale", "wmma_scale16"])
def test_mxfp8_formats_and_hip_selectors(dtype, selector, ml_name, path):
    ml = pytest.importorskip("ml_dtypes")
    spec = mxfp8_gemm.make_spec(
        argparse.Namespace(m=32, n=48, k=256, dtype=dtype, matrix_path=path)
    )
    assert spec.dtype_a == spec.dtype_b == normalize_dtype(dtype)
    kernel = build_block_scaled_gemm(spec)
    hip = lower_kernel_to_hip(kernel, arch="gfx1250")
    assert f"__builtin_amdgcn_{path}_f32_16x16x128_f8f6f4({selector}," in hip
    inputs = make_case_inputs(spec, "mixed")
    assert inputs[0].dtype == inputs[1].dtype == np.dtype(getattr(ml, ml_name))
    ref = reference_result(*inputs, spec.block_k, native=True)
    assert ref.shape == (32, 48) and np.isfinite(ref).all()


@pytest.mark.parametrize(
    "argv,dtypes",
    [
        (["--dtype", "fp8e4m3"], ("fp8e4m3",)),
        (["--dtype", " FP8 "], ("fp8e4m3",)),
        (["--dtype", "bf8e5m2"], ("bf8e5m2",)),
        (["--dtype", " BF8 "], ("bf8e5m2",)),
        ([], ("fp8e4m3", "bf8e5m2")),
        (["--dtype", "both"], ("fp8e4m3", "bf8e5m2")),
        (["--dtype", "fp8"], ("fp8e4m3",)),
        (["--dtype", "bf8"], ("bf8e5m2",)),
    ],
)
def test_mxfp8_cli_selects_formats(monkeypatch, argv, dtypes):
    calls = []

    def run(spec, cases, **kwargs):
        calls.append((spec, cases, kwargs))
        return len(cases)

    monkeypatch.setattr(_scaled_gemm_example, "run_cases", run)
    assert (
        mxfp8_gemm.main(
            argv
            + [
                "--matrix-path",
                "wmma_scale16",
                "--compile-route",
                "hip",
                "--case",
                "all",
            ]
        )
        == 0
    )
    assert tuple(spec.dtype_a for spec, _, _ in calls) == dtypes
    for spec, cases, kwargs in calls:
        assert spec.dtype_a == spec.dtype_b
        assert spec.block_k == 16 and len(cases) == 20
        assert kwargs == {"compile_route": "hip"}
