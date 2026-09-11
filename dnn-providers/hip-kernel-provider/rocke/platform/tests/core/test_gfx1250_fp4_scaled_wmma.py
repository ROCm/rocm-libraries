# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Packed E2M1 contract and native FP4 WMMA regression tests."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rocke.core.arch import ArchTarget
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.examples.gfx1250.gemm.block_scaled_gemm_verify import (
    check_result,
    decode_fp4,
    make_case_inputs,
    pack_fp4_codes,
    reference_result,
)
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec,
    block_scaled_gemm_signature,
    build_block_scaled_gemm,
    is_valid_spec,
)


def fp4_spec(path: str = "wmma_scale", k: int = 128) -> BlockScaledGemmSpec:
    return BlockScaledGemmSpec(
        name="fp4_contract",
        M=32,
        N=48,
        K=k,
        dtype_a="fp4",
        dtype_b="fp4",
        matrix_path=path,
        scale_dtype="e8m0",
        block_k=16 if path == "wmma_scale16" else 32,
    )


def test_all_e2m1_codes_and_nibble_order():
    # Golden bytes and values are independent of the formula-based decoder.
    packed = np.array(
        [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]], dtype=np.uint8
    )
    expected = np.array(
        [
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ]
        ]
    )
    actual = decode_fp4(packed)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    np.testing.assert_array_equal(
        pack_fp4_codes(np.arange(16, dtype=np.uint8)[None]), packed
    )


def test_all_packed_bytes_preserve_both_codes():
    packed = np.arange(256, dtype=np.uint8)[None]
    codes = np.empty((1, 512), dtype=np.uint8)
    codes[:, 0::2] = packed % 16
    codes[:, 1::2] = packed // 16
    np.testing.assert_array_equal(pack_fp4_codes(codes), packed)
    golden = np.array(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ]
    )
    np.testing.assert_array_equal(decode_fp4(packed), golden[codes])


@pytest.mark.parametrize(
    "codes",
    [
        np.array([[16, 0]], dtype=np.uint8),
        np.array([[0]], dtype=np.uint8),
        np.array([[0, 1]], dtype=np.int32),
    ],
)
def test_pack_rejects_invalid_codes(codes):
    with pytest.raises(ValueError, match="FP4 codes"):
        pack_fp4_codes(codes)


def test_fp4_reference_hand_computed():
    pytest.importorskip("ml_dtypes")
    # A=[1,2,3,4], B=[1,-1,2,-2]; per-group scales A=[1,2], B=[2,1].
    a = np.array([[0x42, 0x65]], dtype=np.uint8)
    b = np.array([[0xA2, 0xC4]], dtype=np.uint8)
    sa = np.array([[127, 128]], dtype=np.uint8)
    sb = np.array([[128], [127]], dtype=np.uint8)
    np.testing.assert_array_equal(
        reference_result(a, b, sa, sb, 2, native=True), [[-6.0]]
    )


@pytest.mark.parametrize("path", ["wmma_scale", "wmma_scale16"])
@pytest.mark.parametrize("k", [128, 256])
def test_fp4_fixtures_detect_layout_and_scale_errors(path, k):
    pytest.importorskip("ml_dtypes")
    spec = fp4_spec(path, k)
    a, b, sa, sb = make_case_inputs(spec, "mixed")
    assert a.shape == (spec.M, k // 2) and b.shape == (spec.N, k // 2)
    assert set(np.unique(sa)) == set(range(125, 129))
    expected = reference_result(a, b, sa, sb, spec.block_k, native=True)
    # Wrong nibble order and reusing/permuting scale groups must be observable.
    wrong_a = (a << 4) | (a >> 4)
    for inputs in ((wrong_a, b, sa, sb), (a, b, sa[:, ::-1], sb), (a, b, sa, sb[::-1])):
        with pytest.raises(AssertionError, match="bad="):
            check_result(
                reference_result(*inputs, spec.block_k, native=True),
                expected,
                exact=True,
            )
    for group in range(k // spec.block_k):
        ga, gb, gsa, gsb = make_case_inputs(spec, f"group-{group}")
        outside = np.arange(k) // spec.block_k != group
        assert not np.any(decode_fp4(ga)[:, outside])
        assert not np.any(decode_fp4(gb)[:, outside])
        isolated = reference_result(ga, gb, gsa, gsb, spec.block_k, native=True)
        assert np.isfinite(isolated).all() and np.any(isolated)


@pytest.mark.parametrize(
    "path,scale_type", [("wmma_scale", "i32"), ("wmma_scale16", "i64")]
)
def test_fp4_catalog_signature_and_lowering(path, scale_type):
    spec = fp4_spec(path)
    assert is_valid_spec(spec)[0]
    op = ArchTarget.from_gfx("gfx1250").mma.by_op_id(f"{path}_f32_16x16x128_fp4_fp4")
    assert (op.a_frag_len, op.b_frag_len, op.c_frag_len) == (16, 16, 8)
    assert all(
        p["type"] == "ptr<i8, global>" for p in block_scaled_gemm_signature(spec)[:4]
    )
    kernel = build_block_scaled_gemm(spec)
    llvm = lower_kernel_to_llvm(kernel, arch="gfx1250", llvm_flavor="llvm23")
    call = next(
        l for l in llvm.splitlines() if "call <8 x float> @llvm.amdgcn.wmma.scale" in l
    )
    assert call.count("i32 4, <16 x i32>") == 2
    assert call.count(f", {scale_type} %") == 2
    assert llvm.count("load <16 x i8>") == 4  # Two packed chunks per operand.
    hip = lower_kernel_to_hip(kernel, arch="gfx1250")
    assert f"__builtin_amdgcn_{path}_f32_16x16x128_f8f6f4(4," in hip
    for flavor in ("llvm20", "llvm22"):
        with pytest.raises(NotImplementedError, match="requires llvm23"):
            lower_kernel_to_llvm(kernel, arch="gfx1250", llvm_flavor=flavor)
    with pytest.raises(NotImplementedError):
        lower_kernel_to_hip(kernel, arch="gfx950")


@pytest.mark.parametrize(
    "changes",
    [
        {"dtype_b": "fp8"},
        {"matrix_path": "wmma"},
        {"K": 192},
        {"scale_dtype": "fp32"},
        {"block_k": 64},
    ],
)
def test_reject_unsupported_fp4_contract(changes):
    spec = replace(fp4_spec(), **changes)
    assert not is_valid_spec(spec)[0]
    with pytest.raises(ValueError, match="invalid block_scaled_gemm"):
        build_block_scaled_gemm(spec)
