# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""NVFP4 packed-input contract, tensor-scale epilogue, and reference checks."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rocke.core.ir import F32, I32, I64, IRBuilder
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.examples.gfx1250.gemm.nvfp4_gemm_verify import (
    decode_e4m3_scales,
    make_case_inputs,
    reference_result,
)
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec,
    build_block_scaled_gemm,
    is_valid_spec as is_valid_block_spec,
)
from rocke.instances.gfx1250.nvfp4_gemm import (
    NvFp4GemmSpec,
    build_nvfp4_gemm,
    is_valid_spec,
    nvfp4_gemm_grid,
    nvfp4_gemm_signature,
)


@pytest.mark.parametrize(
    "case,sha",
    json.loads(
        Path(__file__).with_name("gfx1250_nvfp4_llvm23.json").read_text()
    ).items(),
)
def test_nvfp4_golden(case, sha):
    dtype, k = case.split("/")
    kernel = build_nvfp4_gemm(NvFp4GemmSpec("golden", 32, 48, int(k), dtype))
    llvm = lower_kernel_to_llvm(kernel, arch="gfx1250", llvm_flavor="llvm23")
    assert hashlib.sha256(llvm.encode()).hexdigest() == sha


def test_nvfp4_output_type_has_distinct_name():
    spec = NvFp4GemmSpec("nv", 16, 16, 128)
    assert spec.kernel_name() != replace(spec, dtype_c="fp16").kernel_name()


@pytest.mark.parametrize("dtype", ["bf16", "fp16"])
@pytest.mark.parametrize("k", [128, 256])
def test_nvfp4_kernel_contract(dtype, k):
    spec = NvFp4GemmSpec("nv", 32, 48, k, dtype)
    assert is_valid_spec(spec)[0]
    assert nvfp4_gemm_grid(spec) == (3, 2, 1)
    assert nvfp4_gemm_signature(spec)[-2:] == [
        {"name": "A_tensor_scale", "type": "f32"},
        {"name": "B_tensor_scale", "type": "f32"},
    ]
    kernel = build_nvfp4_gemm(spec)
    llvm = lower_kernel_to_llvm(kernel, arch="gfx1250", llvm_flavor="llvm23")
    calls = [
        l
        for l in llvm.splitlines()
        if "call <8 x float> @llvm.amdgcn.wmma.scale16" in l
    ]
    assert len(calls) == k // 128
    assert all(
        l.count("i32 4, <16 x i32>") == 2 and l.count("i32 0, i32 2, i64") == 2
        for l in calls
    )
    assert "fmul float %A_tensor_scale, %B_tensor_scale" in llvm
    assert llvm.count("fmul float") == 9
    assert "mul nsw float" not in llvm
    # Each final conversion consumes the tensor-scaled scalar, not the raw accumulator.
    scaled_results = {
        l.strip().split(" = ")[0]
        for l in llvm.splitlines()
        if "fmul float" in l and "tensor_scale" not in l
    }
    assert len(scaled_results) == 8
    if dtype == "fp16":
        barriers = [l for l in llvm.splitlines() if 'asm "", "=v,0"' in l]
        assert len(barriers) == 8
        assert all(any(v + ")" in l for l in barriers) for v in scaled_results)
        scaled_results = {l.strip().split(" = ")[0] for l in barriers}
    assert all(
        any(
            v + " " in l or v + "," in l or v + ")" in l
            for l in llvm.splitlines()
            if "fptrunc" in l or "cvt" in l
        )
        for v in scaled_results
    )
    hip = lower_kernel_to_hip(kernel, arch="gfx1250")
    assert "A_tensor_scale * B_tensor_scale" in hip
    assert "0, 2," in hip
    for flavor in ("llvm20", "llvm22"):
        with pytest.raises(NotImplementedError):
            lower_kernel_to_llvm(kernel, arch="gfx1250", llvm_flavor=flavor)


@pytest.mark.parametrize(
    "changes", [{"M": 17}, {"N": 0}, {"K": 192}, {"dtype_c": "fp32"}]
)
def test_invalid_nvfp4_spec(changes):
    spec = replace(NvFp4GemmSpec("nv", 16, 16, 128), **changes)
    assert not is_valid_spec(spec)[0]
    with pytest.raises(ValueError):
        build_nvfp4_gemm(spec)


def test_tensor_scale_requires_native_path():
    assert not is_valid_block_spec(
        BlockScaledGemmSpec("legacy", 16, 16, 128, tensor_scale=True)
    )[0]


@pytest.mark.parametrize(
    "sa,sb", [("e4m3", "e8m0"), ("e5m3", "e4m3"), ("e5m2", "e5m2")]
)
def test_invalid_atom_scale_combination(sa, sb):
    b = IRBuilder("invalid")
    a = b.zero_vec(I32, 16)
    c = b.zero_vec(F32, 8)
    scale = b.const_i64(0)
    with pytest.raises(ValueError):
        b.mma(
            "wmma_scale16_f32_16x16x128_fp4_fp4",
            a,
            a,
            c,
            scale,
            scale,
            scale_dtype_a=sa,
            scale_dtype_b=sb,
        )


def test_e4m3_scale_decoding_against_numeric_dtype():
    ml = pytest.importorskip("ml_dtypes")
    codes = np.arange(127, dtype=np.uint8)
    np.testing.assert_array_equal(
        decode_e4m3_scales(codes), codes.view(ml.float8_e4m3fn).astype(np.float64)
    )
    for code in (127, 128, 255):
        with pytest.raises(ValueError):
            decode_e4m3_scales(np.array([code], dtype=np.uint8))


def test_tensor_scale_direction_and_conversion_order():
    # One nonzero FP4 product: 1 * 1. E4M3 scales are 1.25 * 1.75.
    a = np.zeros((16, 64), dtype=np.uint8)
    a[:, 0] = 2
    sa = np.full((16, 8), 58, dtype=np.uint8)
    sb = np.full((8, 16), 62, dtype=np.uint8)
    result = reference_result(a, a, sa, sb, (0.5, 2.0), dtype_c="fp16")
    np.testing.assert_array_equal(result, np.full((16, 16), 2.1875))
    result = reference_result(a, a, sa, sb, (0.5, 0.5), dtype_c="fp16")
    np.testing.assert_array_equal(result, np.full((16, 16), 0.546875))


def test_tensor_rounding_fixture_requires_fp32_intermediate():
    # This real GEMM cell lies on an FP16 tie after FP32 multiplication.
    # Fusing the multiply and FP16 conversion instead returns -375.25.
    spec = NvFp4GemmSpec("rounding", 32, 48, 256, "fp16")
    inputs, factors = make_case_inputs(spec, "tensor-rounding")
    result = reference_result(*inputs, factors, dtype_c="fp16")
    assert result[0, 13] == -375.5
    factor = np.float32(factors[0]) * np.float32(factors[1])
    assert np.float16(np.float64(-412.5) * np.float64(factor)) == -375.25


@pytest.mark.parametrize(
    "case",
    [
        "neutral",
        "a-only",
        "b-only",
        "mixed",
        "tensor",
        "tensor-rounding",
        "zero-tensor",
        "zero-block",
        "group-15",
        "codes-127",
    ],
)
def test_bounded_reference_fixtures(case):
    spec = NvFp4GemmSpec("nv", 32, 48, 256)
    inputs, factors = make_case_inputs(spec, case)
    result = reference_result(*inputs, factors)
    assert result.shape == (32, 48) and np.isfinite(result).all()
    if case in ("zero-tensor", "zero-block"):
        assert not np.any(result)
    else:
        assert np.any(result)


def test_default_fp4_api_keeps_kernel_identity():
    spec = BlockScaledGemmSpec(
        "fp4",
        16,
        16,
        128,
        dtype_a="fp4",
        dtype_b="fp4",
        matrix_path="wmma_scale16",
        block_k=16,
        scale_dtype="e8m0",
    )
    explicit = replace(spec, scale_dtype_a="i8", scale_dtype_b="e8m0")
    assert spec.kernel_name() == explicit.kernel_name()
    assert lower_kernel_to_llvm(
        build_block_scaled_gemm(spec), arch="gfx1250", llvm_flavor="llvm23"
    ) == lower_kernel_to_llvm(
        build_block_scaled_gemm(explicit), arch="gfx1250", llvm_flavor="llvm23"
    )
