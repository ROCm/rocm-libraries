# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Packed E2M1 contract and native FP4 WMMA regression tests."""

from __future__ import annotations

import operator
from dataclasses import replace

import numpy as np
import pytest

from rocke.core.arch import ArchTarget
from rocke.core.backend import resolve_backend
from rocke.core.ir import I8
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
    op = ArchTarget.from_gfx("gfx1250").mma.by_op_id(
        f"wmma_gfx1250_f32_16x16x128_fp4_fp4_scale_e8m0_e8m0_k{spec.block_k}"
    )
    assert (op.a_frag_len, op.b_frag_len, op.c_frag_len) == (16, 16, 8)
    assert all(
        p["type"] == "ptr<i8, global>" for p in block_scaled_gemm_signature(spec)[:4]
    )
    kernel = build_block_scaled_gemm(spec)
    llvm = lower_kernel_to_llvm(kernel, arch="gfx1250", llvm_flavor="llvm23")
    call = next(
        line
        for line in llvm.splitlines()
        if "call <8 x float> @llvm.amdgcn.wmma.scale" in line
    )
    assert call.count("i32 4, <16 x i32>") == 2
    assert call.count(f", {scale_type} %") == 2
    assert llvm.count("load <4 x i32>") == 4  # Two packed chunks per operand.
    hip = lower_kernel_to_hip(kernel, arch="gfx1250")
    assert f"__builtin_amdgcn_{path}_f32_16x16x128_f8f6f4(4," in hip
    for flavor in ("llvm20", "llvm22"):
        error_type = RuntimeError if resolve_backend() == "cpp" else NotImplementedError
        with pytest.raises(error_type, match="requires llvm23"):
            lower_kernel_to_llvm(kernel, arch="gfx1250", llvm_flavor=flavor)
    with pytest.raises(NotImplementedError):
        lower_kernel_to_hip(kernel, arch="gfx950")


@pytest.mark.parametrize(
    "changes",
    [
        {"dtype_b": "i8"},
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


@pytest.mark.parametrize("path", ["wmma_scale", "wmma_scale16"])
@pytest.mark.parametrize(
    "a_dtype,b_dtype",
    [("fp4e2m1", "fp4e2m1"), ("fp4", "fp4e2m1"), ("fp4e2m1", "fp4")],
)
def test_fp4e2m1_alias_preserves_packed_contract(path, a_dtype, b_dtype):
    original = fp4_spec(path, 256)
    alias = replace(original, dtype_a=a_dtype, dtype_b=b_dtype)
    assert is_valid_spec(alias)[0]
    assert block_scaled_gemm_signature(alias) == block_scaled_gemm_signature(original)
    assert ArchTarget.from_gfx("gfx1250").mma.has_shape(
        family="wmma_scaled",
        scales=("e8m0", "e8m0", original.block_k),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        c_dtype="fp32",
        m=16,
        n=16,
        k=128,
    )
    for lower, options in (
        (lower_kernel_to_llvm, {"llvm_flavor": "llvm23"}),
        (lower_kernel_to_hip, {}),
    ):
        assert lower(
            build_block_scaled_gemm(alias), arch="gfx1250", **options
        ) == lower(build_block_scaled_gemm(original), arch="gfx1250", **options)
    for actual, expected in zip(
        make_case_inputs(alias, "mixed"), make_case_inputs(original, "mixed")
    ):
        np.testing.assert_array_equal(actual, expected)


def _scale_byte(role, offset):
    # Address-sensitive bytes with distinct A/B patterns and high bits set.
    return (offset * 37 + (11 if role == "A_scale" else 73)) % 256


def _evaluate_scale_word(value, *, lane, tile, loads):
    """Evaluate a scale operand's SSA dependencies and record absolute byte reads."""
    binary = {
        "arith.add": operator.add,
        "arith.mul": operator.mul,
        "arith.mod": operator.mod,
        "arith.div": operator.floordiv,
    }

    values = {}

    def evaluate(value):
        # An SSA vector load can feed several extracts but only reads once.
        if value.name not in values:
            values[value.name] = evaluate_uncached(value)
        return values[value.name]

    def evaluate_uncached(value):
        if value.name in ("%A_scale", "%B_scale"):
            return value.name[1:], 0
        op = value.op
        if op.name == "arith.constant":
            return op.attrs["value"]
        if op.name == "arith.constant_vec":
            return (int(op.attrs["fill"]),) * op.attrs["vec"]
        if op.name == "gpu.thread_id":
            return lane
        if op.name == "gpu.block_id":
            return tile[0 if op.attrs["axis"] == "x" else 1]
        args = [evaluate(v) for v in op.operands]
        if op.name == "arith.sext":
            return args[0]  # Evaluated addresses are already signed Python ints.
        if op.name == "tile.global_ptr_add":
            role, base = args[0]
            return role, base + args[1]
        if op.name in ("memref.global_load_typed", "memref.global_load_vN"):
            role, base = args[0]
            # E8M0 storage elements are single bytes, including vector lanes.
            scalar = op.name == "memref.global_load_typed"
            elem = value.type if scalar else value.type.elem
            assert elem == I8
            start = base + args[1]
            offsets = range(start, start + (1 if scalar else value.type.count))
            loads.extend((role, offset) for offset in offsets)
            encoded = tuple(_scale_byte(role, offset) for offset in offsets)
            return encoded[0] if scalar else encoded
        if op.name == "vector.extract":
            return args[0][op.attrs["index"]]
        if op.name == "vector.insert":
            index = op.attrs["index"]
            return args[0][:index] + (args[1],) + args[0][index + 1 :]
        if op.name == "arith.bitcast":
            return int.from_bytes(bytes(args[0]), "little")
        assert op.name in binary, op.name
        return binary[op.name](*args)

    return evaluate(value)


@pytest.mark.parametrize("path,count", [("wmma_scale", 4), ("wmma_scale16", 8)])
@pytest.mark.parametrize("dtype", ["fp8", "bf8", "fp4"])
def test_scale_operands_pack_consecutive_k_groups_low_byte_first(path, count, dtype):
    spec = replace(fp4_spec(path, 256), dtype_a=dtype, dtype_b=dtype)
    kernel = build_block_scaled_gemm(spec)
    calls = [op for op in kernel.body.ops if op.name == "tile.mma"]
    assert len(calls) == 2
    groups = spec.K // spec.block_k
    for x, y in ((0, 0), (2, 1)):
        for lane in range(32):
            row, col = 16 * y + lane % 16, 16 * x + lane % 16
            for step, call in enumerate(calls):
                k_groups = range(step * count, (step + 1) * count)
                for role, value, offsets in (
                    ("A_scale", call.operands[3], [row * groups + g for g in k_groups]),
                    ("B_scale", call.operands[4], [g * spec.N + col for g in k_groups]),
                ):
                    expected = int.from_bytes(
                        bytes(_scale_byte(role, offset) for offset in offsets), "little"
                    )
                    loads = []
                    actual = _evaluate_scale_word(
                        value, lane=lane, tile=(x, y), loads=loads
                    )
                    assert value.type.name == f"i{count * 8}"
                    assert loads == [(role, offset) for offset in offsets]
                    assert actual == expected
