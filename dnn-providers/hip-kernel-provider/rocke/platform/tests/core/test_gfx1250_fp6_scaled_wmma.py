# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Homogeneous FP6 packing and native WMMA with E8M0 scales."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rocke.core.arch import ArchTarget
from rocke.core.backend import resolve_backend
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import lower_kernel_to_llvm
from rocke.examples.gfx1250.gemm.block_scaled_gemm_verify import (
    decode_fp6,
    pack_fp6_codes,
)
from rocke.instances.gfx1250.block_scaled_gemm import (
    BlockScaledGemmSpec,
    block_scaled_gemm_signature,
    build_block_scaled_gemm,
    is_valid_spec,
)

FORMATS = ("fp8", "bf8", "fp6", "bf6", "fp4")


@pytest.mark.parametrize(
    "case,expected_sha",
    json.loads(
        Path(__file__).with_name("gfx1250_scaled_wmma_llvm23.json").read_text()
    ).items(),
)
def test_scaled_wmma_llvm23_golden(case, expected_sha):
    mode, a, b, sa, sb = case.split("/")
    assert sa == sb == "e8m0"
    spec = spec_for(a, b, mode)
    llvm = lower_kernel_to_llvm(
        build_block_scaled_gemm(spec), arch="gfx1250", llvm_flavor="llvm23"
    )
    assert hashlib.sha256(llvm.encode()).hexdigest() == expected_sha


@pytest.mark.parametrize("alias,canonical", [("fp6", "fp6e2m3"), ("bf6", "fp6e3m2")])
def test_fp6_catalog_aliases(alias, canonical):
    from rocke.core.arch.target import normalize_dtype

    assert normalize_dtype(alias) == canonical
    spec = spec_for(alias, alias)
    assert is_valid_spec(spec)[0]


def spec_for(a="fp6", b="fp6", mode="wmma_scale", **kwargs):
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
    "dtype,ml_name", [("fp6", "float6_e2m3fn"), ("bf6", "float6_e3m2fn")]
)
def test_all_fp6_codes_against_independent_dtype(dtype, ml_name):
    ml = pytest.importorskip("ml_dtypes")
    codes = np.arange(64, dtype=np.uint8).reshape(1, -1)
    actual = decode_fp6(pack_fp6_codes(codes), dtype)
    expected = codes.view(getattr(ml, ml_name)).astype(np.float64)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))
    assert actual[0, 31] == (7.5 if dtype == "fp6" else 28.0)


def test_fp6_cross_byte_packing():
    codes = np.array([[1, 2, 3, 4, 63, 0, 32, 1]], dtype=np.uint8)
    np.testing.assert_array_equal(
        pack_fp6_codes(codes), [[0x81, 0x30, 0x10, 0x3F, 0, 6]]
    )


@pytest.mark.parametrize(
    "codes",
    [
        np.array([[64, 0, 0, 0]], dtype=np.uint8),
        np.zeros((1, 3), dtype=np.uint8),
        np.zeros((1, 4), dtype=np.int32),
    ],
)
def test_invalid_fp6_codes(codes):
    with pytest.raises(ValueError, match="FP6 codes"):
        pack_fp6_codes(codes)


@pytest.mark.parametrize(
    "a,b,mode",
    [(d, d, m) for d in ("fp6", "bf6") for m in ("wmma_scale", "wmma_scale16")],
)
def test_all_native_matrix_pairs(a, b, mode):
    spec = spec_for(a, b, mode)
    assert is_valid_spec(spec)[0]
    atom = ArchTarget.from_gfx("gfx1250").mma.op_for_shape(
        family="wmma_scaled",
        a_dtype=a,
        b_dtype=b,
        c_dtype="fp32",
        m=16,
        n=16,
        k=128,
        scales=("e8m0", "e8m0", spec.block_k),
    )
    assert (atom.a_frag_len, atom.b_frag_len, atom.c_frag_len) == (16, 16, 8)
    kernel = build_block_scaled_gemm(spec)
    llvm = lower_kernel_to_llvm(kernel, arch="gfx1250", llvm_flavor="llvm23")
    call = next(
        l for l in llvm.splitlines() if "call <8 x float> @llvm.amdgcn.wmma.scale" in l
    )
    assert f"i32 {FORMATS.index(a)}, <16 x i32>" in call
    assert f"i32 {FORMATS.index(b)}, <16 x i32>" in call
    hip = lower_kernel_to_hip(kernel, arch="gfx1250")
    assert f"__builtin_amdgcn_{mode}_f32_16x16x128_f8f6f4({FORMATS.index(a)}," in hip
    for flavor in ("llvm20", "llvm22"):
        error_type = RuntimeError if resolve_backend() == "cpp" else NotImplementedError
        with pytest.raises(error_type, match="requires llvm23"):
            lower_kernel_to_llvm(kernel, arch="gfx1250", llvm_flavor=flavor)
    if a in ("fp6", "bf6"):
        assert block_scaled_gemm_signature(spec)[0]["type"] == "ptr<i8, global>"


@pytest.mark.parametrize(
    "changes",
    [
        {"matrix_path": "wmma"},
        {"K": 192},
        {"scale_dtype": "fp32"},
    ],
)
def test_reject_invalid_fp6_contract(changes):
    spec = replace(spec_for(), **changes)
    assert not is_valid_spec(spec)[0]
    with pytest.raises(ValueError):
        build_block_scaled_gemm(spec)


def test_fp6_aliases():
    original = spec_for()
    alias = replace(original, dtype_a="fp6e2m3", dtype_b="fp6e2m3")
    assert alias.kernel_name() == original.kernel_name()
    assert is_valid_spec(alias)[0]


@pytest.mark.parametrize("a,b", [("fp6", "bf6"), ("fp8", "fp4"), ("fp4", "fp4")])
def test_other_matrix_contracts_are_not_admitted(a, b):
    assert not is_valid_spec(spec_for(a, b))[0]


@pytest.mark.parametrize("mode", ["wmma_scale", "wmma_scale16"])
@pytest.mark.parametrize("a", FORMATS)
@pytest.mark.parametrize("b", FORMATS)
def test_homogeneous_catalog_boundary(a, b, mode):
    spec = spec_for(a, b, mode)
    accepted = a == b and a in ("fp8", "bf8", "fp6", "bf6")
    assert is_valid_spec(spec)[0] == accepted
    atom = ArchTarget.from_gfx("gfx1250").mma.op_for_shape(
        family="wmma_scaled",
        a_dtype=a,
        b_dtype=b,
        c_dtype="fp32",
        m=16,
        n=16,
        k=128,
        scales=("e8m0", "e8m0", spec.block_k),
    )
    assert (atom is not None) == accepted


@pytest.mark.parametrize("dtype", ["fp6", "bf6"])
@pytest.mark.parametrize("block_k", [16, 32])
def test_fp6_atom_storage_contract(dtype, block_k):
    from rocke.core.arch.wmma_scale import gfx1250_scaled_wmma

    atom = ArchTarget.from_gfx("gfx1250").mma.op_for_shape(
        family="wmma_scaled",
        a_dtype=dtype,
        b_dtype=dtype,
        c_dtype="fp32",
        m=16,
        n=16,
        k=128,
        scales=("e8m0", "e8m0", block_k),
    )
    contract = gfx1250_scaled_wmma(atom.op_id)
    for operand in ("a", "b"):
        layout = contract.matrix_layout(operand)
        assert layout.fragment.packing.group(32) == (16, 3)
        assert layout.chunk_elements == 32 and layout.chunk_bytes == 24
        assert layout.fragment.live_carriers == 12
        assert layout.fragment.padding_bits == 128
        assert contract.scale_packing(operand).association.block_k == block_k
        # Literal bit-stream oracle, independent of the shared packer.
        patterns = list(range(64))
        expected = sum(value << (6 * i) for i, value in enumerate(patterns))
        assert layout.fragment.pack(patterns) == tuple(
            (expected >> (32 * i)) & 0xFFFFFFFF for i in range(16)
        )
