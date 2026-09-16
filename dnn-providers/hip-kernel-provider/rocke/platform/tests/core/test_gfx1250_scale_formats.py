# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Independent scale-format selection, decoding, and LLVM emission contracts."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.examples.gfx1250.gemm.block_scaled_gemm_verify import (
    decode_scale,
    make_case_inputs,
    reference_result,
)
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec,
    build_block_scaled_gemm,
    is_valid_spec,
)


def spec_for(a="fp6", b="bf6", mode="wmma_scale", **kwargs):
    return BlockScaledGemmSpec(
        name="fp6_test",
        M=32,
        N=48,
        K=256,
        dtype_a=a,
        dtype_b=b,
        scale_dtype="e8m0",
        matrix_path=mode,
        block_k=16 if mode == "wmma_scale16" else 32,
        **kwargs,
    )


@pytest.mark.parametrize(
    "case,expected_sha",
    json.loads(
        Path(__file__).with_name("gfx1250_scale_formats_llvm23.json").read_text()
    ).items(),
)
def test_scale_formats_llvm23_golden(case, expected_sha):
    mode, a, b, sa, sb = case.split("/")
    spec = spec_for(a, b, mode, scale_dtype_a=sa, scale_dtype_b=sb)
    llvm = lower_kernel_to_llvm(
        build_block_scaled_gemm(spec), arch="gfx1250", llvm_flavor="llvm23"
    )
    assert hashlib.sha256(llvm.encode()).hexdigest() == expected_sha


@pytest.mark.parametrize(
    "dtype,bias,max_code,max_value", [("e4m3", 7, 126, 448), ("e5m3", 15, 254, 114688)]
)
def test_scale_decoding(dtype, bias, max_code, max_value):
    codes = np.array([0, 1, 8, bias * 8, bias * 8 + 4, max_code], dtype=np.uint8)
    np.testing.assert_array_equal(
        decode_scale(codes, dtype),
        [0, 2.0 ** (-bias - 2), 2.0 ** (1 - bias), 1, 1.5, max_value],
    )
    with pytest.raises(ValueError):
        decode_scale(np.array([max_code + 1], dtype=np.uint8), dtype)


@pytest.mark.parametrize("mode", ["wmma_scale", "wmma_scale16"])
@pytest.mark.parametrize(
    "a,b,sa,sb",
    [
        ("fp4", "fp4", "e4m3", "e4m3"),
        ("fp4", "fp4", "e5m3", "e5m3"),
        ("fp6", "fp4", "e8m0", "e5m3"),
        ("fp4", "bf6", "e4m3", "e8m0"),
    ],
)
def test_independent_scale_selectors_and_names(mode, a, b, sa, sb):
    spec = spec_for(a, b, mode, scale_dtype_a=sa, scale_dtype_b=sb)
    assert is_valid_spec(spec)[0]
    assert spec.kernel_name() != spec_for(a, b, mode).kernel_name()
    llvm = lower_kernel_to_llvm(
        build_block_scaled_gemm(spec), arch="gfx1250", llvm_flavor="llvm23"
    )
    call = next(
        l for l in llvm.splitlines() if "call <8 x float> @llvm.amdgcn.wmma.scale" in l
    )
    ty = "i64" if mode == "wmma_scale16" else "i32"
    for dtype in (sa, sb):
        assert f"i32 0, i32 {('e8m0', 'e5m3', 'e4m3').index(dtype)}, {ty} %" in call
    inputs = make_case_inputs(spec, "mixed")
    ref = reference_result(
        *inputs,
        spec.block_k,
        native=True,
        dtype_a=a,
        dtype_b=b,
        scale_dtype_a=sa,
        scale_dtype_b=sb,
    )
    assert np.isfinite(ref).all() and np.any(ref)


def test_reject_mismatched_fp4_scale_formats():
    assert not is_valid_spec(
        spec_for("fp4", "fp4", scale_dtype_a="e4m3", scale_dtype_b="e5m3")
    )[0]


def test_scale_defaults_and_fp6_aliases():
    original = spec_for()
    alias = replace(original, dtype_a="fp6e2m3", dtype_b="fp6e3m2")
    assert alias.kernel_name() == original.kernel_name()
    assert is_valid_spec(alias)[0]
    explicit = replace(original, scale_dtype_a="e8m0", scale_dtype_b="i8")
    assert explicit.kernel_name() == original.kernel_name()
    assert lower_kernel_to_llvm(
        build_block_scaled_gemm(explicit), arch="gfx1250", llvm_flavor="llvm23"
    ) == lower_kernel_to_llvm(
        build_block_scaled_gemm(original), arch="gfx1250", llvm_flavor="llvm23"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"scale_dtype_a": "e5m3"},
        {"scale_dtype_b": "e4m3"},
        {"scale_dtype_b": "e5m2"},
    ],
)
def test_reject_invalid_fp6_scale_formats(changes):
    spec = replace(spec_for(), **changes)
    assert not is_valid_spec(spec)[0]
    with pytest.raises(ValueError):
        build_block_scaled_gemm(spec)
