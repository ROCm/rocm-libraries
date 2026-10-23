# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The CDNA MFMA instructions each target gained beyond f16 / bf16 / fp8 / bf8.

gfx908 gained i8 and the bf16 MFMAs without the ``_1k`` suffix; gfx90a the f32,
i8 and f64 MFMAs and the same bf16 MFMAs (smaller K than its ``_1k`` ones, so
selection keeps the ``_1k`` ones); gfx942 and gfx950 i8 and the mixed fp8 x bf8
/ bf8 x fp8 MFMAs, and gfx950 f64 and its wider i8 forms (gfx942 already had
f64; its row is pinned with the others). Each case below pins, for one target
and one instruction:

- the catalog row (dtypes, shape, fragment lengths per lane);
- the lowered call in both engines at every LLVM flavor: the intrinsic and the
  register type of every argument;
- that the module compiles and links for the target, and that LLVM selects the
  named instruction (compile-verified, not run; the gfx90a cases also run on
  hardware in ``tests/numeric/test_gfx90a_mfma_ops.py``);
- that its lane maps place every element of A, B and the accumulator exactly
  once, on the closed form the public matrix instruction calculator gives.

The C++ engine's catalog rows and lane maps are pinned to the same data by
``tests/core/cdna_mfma_ops.cpp``.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
from rocke.core.arch import ArchTarget
from rocke.core.ir import (
    I32,
    IRBuilder,
    PtrType,
    VectorType,
    dtype_to_ir_type,
    type_bits,
)
from rocke.core.lower_llvm import LLVM_FLAVORS
from rocke.tests.engines import lower_with_both_engines
from rocke.tests.ir_validity import assert_target_accepts, target_mnemonics
from rocke.tests.lane_maps import lane_placements


class _Case(NamedTuple):
    """One instruction on one target: its catalog contract and how it lowers."""

    a: str
    b: str
    c: str
    shape: tuple[int, int, int]
    frag: tuple[int, int, int]  # A, B and accumulator elements per lane
    intrinsic: str
    operand: str | dict[str, str]  # A and B register type, or one per flavor
    accumulator: str
    mnemonic: str


_FP8_X8 = {"llvm20": "<2 x i32>", "llvm22": "i64", "llvm23": "i64"}

_CDNA12_I8 = {
    "mfma_i32_16x16x16_i8": _Case(
        "i8", "i8", "i32", (16, 16, 16), (4, 4, 4),
        "llvm.amdgcn.mfma.i32.16x16x16i8", "i32", "<4 x i32>",
        "v_mfma_i32_16x16x16i8",
    ),
    "mfma_i32_32x32x8_i8": _Case(
        "i8", "i8", "i32", (32, 32, 8), (4, 4, 16),
        "llvm.amdgcn.mfma.i32.32x32x8i8", "i32", "<16 x i32>",
        "v_mfma_i32_32x32x8i8",
    ),
}  # fmt: skip

# bf16 without the _1k suffix: two bf16 per lane as <2 x i16>.
_CDNA12_BF16 = {
    "mfma_f32_16x16x8_bf16": _Case(
        "bf16", "bf16", "fp32", (16, 16, 8), (2, 2, 4),
        "llvm.amdgcn.mfma.f32.16x16x8bf16", "<2 x i16>", "<4 x float>",
        "v_mfma_f32_16x16x8bf16",
    ),
    "mfma_f32_32x32x4_bf16": _Case(
        "bf16", "bf16", "fp32", (32, 32, 4), (2, 2, 16),
        "llvm.amdgcn.mfma.f32.32x32x4bf16", "<2 x i16>", "<16 x float>",
        "v_mfma_f32_32x32x4bf16",
    ),
}  # fmt: skip


# The instructions gfx942 and gfx950 share.
_CDNA3 = {
    "mfma_i32_16x16x32_i8": _Case(
        "i8", "i8", "i32", (16, 16, 32), (8, 8, 4),
        "llvm.amdgcn.mfma.i32.16x16x32.i8", "i64", "<4 x i32>",
        "v_mfma_i32_16x16x32_i8",
    ),
    "mfma_i32_32x32x16_i8": _Case(
        "i8", "i8", "i32", (32, 32, 16), (8, 8, 16),
        "llvm.amdgcn.mfma.i32.32x32x16.i8", "i64", "<16 x i32>",
        "v_mfma_i32_32x32x16_i8",
    ),
    "mfma_f64_16x16x4_f64": _Case(
        "fp64", "fp64", "fp64", (16, 16, 4), (1, 1, 4),
        "llvm.amdgcn.mfma.f64.16x16x4f64", "double", "<4 x double>",
        "v_mfma_f64_16x16x4_f64",
    ),
    "mfma_f32_16x16x32_fp8_bf8": _Case(
        "fp8e4m3", "bf8e5m2", "fp32", (16, 16, 32), (8, 8, 4),
        "llvm.amdgcn.mfma.f32.16x16x32.fp8.bf8", _FP8_X8, "<4 x float>",
        "v_mfma_f32_16x16x32_fp8_bf8",
    ),
    "mfma_f32_16x16x32_bf8_fp8": _Case(
        "bf8e5m2", "fp8e4m3", "fp32", (16, 16, 32), (8, 8, 4),
        "llvm.amdgcn.mfma.f32.16x16x32.bf8.fp8", _FP8_X8, "<4 x float>",
        "v_mfma_f32_16x16x32_bf8_fp8",
    ),
    "mfma_f32_32x32x16_fp8_bf8": _Case(
        "fp8e4m3", "bf8e5m2", "fp32", (32, 32, 16), (8, 8, 16),
        "llvm.amdgcn.mfma.f32.32x32x16.fp8.bf8", _FP8_X8, "<16 x float>",
        "v_mfma_f32_32x32x16_fp8_bf8",
    ),
    "mfma_f32_32x32x16_bf8_fp8": _Case(
        "bf8e5m2", "fp8e4m3", "fp32", (32, 32, 16), (8, 8, 16),
        "llvm.amdgcn.mfma.f32.32x32x16.bf8.fp8", _FP8_X8, "<16 x float>",
        "v_mfma_f32_32x32x16_bf8_fp8",
    ),
}  # fmt: skip


# target -> op_id -> case: every instruction this target gained.
_CASES: dict[str, dict[str, _Case]] = {
    "gfx908": {
        **_CDNA12_I8,
        **_CDNA12_BF16,
    },
    "gfx90a": {
        "mfma_f32_16x16x4_f32": _Case(
            "fp32", "fp32", "fp32", (16, 16, 4), (1, 1, 4),
            "llvm.amdgcn.mfma.f32.16x16x4f32", "float", "<4 x float>",
            "v_mfma_f32_16x16x4f32",
        ),
        "mfma_f32_32x32x2_f32": _Case(
            "fp32", "fp32", "fp32", (32, 32, 2), (1, 1, 16),
            "llvm.amdgcn.mfma.f32.32x32x2f32", "float", "<16 x float>",
            "v_mfma_f32_32x32x2f32",
        ),
        **_CDNA12_I8,
        "mfma_f64_16x16x4_f64": _Case(
            "fp64", "fp64", "fp64", (16, 16, 4), (1, 1, 4),
            "llvm.amdgcn.mfma.f64.16x16x4f64", "double", "<4 x double>",
            "v_mfma_f64_16x16x4f64",
        ),
        **_CDNA12_BF16,
    },
    "gfx942": _CDNA3,
    "gfx950": {
        **_CDNA3,
        "mfma_i32_16x16x64_i8": _Case(
            "i8", "i8", "i32", (16, 16, 64), (16, 16, 4),
            "llvm.amdgcn.mfma.i32.16x16x64.i8", "<4 x i32>", "<4 x i32>",
            "v_mfma_i32_16x16x64_i8",
        ),
        "mfma_i32_32x32x32_i8": _Case(
            "i8", "i8", "i32", (32, 32, 32), (16, 16, 16),
            "llvm.amdgcn.mfma.i32.32x32x32.i8", "<4 x i32>", "<16 x i32>",
            "v_mfma_i32_32x32x32_i8",
        ),
    },
}  # fmt: skip

_PAIRS = [(gfx, op_id) for gfx, cases in _CASES.items() for op_id in cases]


def _operand(b: IRBuilder, name: str, dtype: str, count: int):
    """Operand ``name``: each lane's ``count`` values of ``dtype``, loaded as
    32-bit words from a kernel parameter and reinterpreted."""
    logical = VectorType(dtype_to_ir_type(dtype), count)
    words = type_bits(logical) // 32
    ptr = b.param(name, PtrType(I32, "global"))
    index = b.mul(b.thread_id_x(), b.const_i32(words))
    loaded = (
        b.global_load(ptr, index, I32)
        if words == 1
        else b.global_load_vN(ptr, index, I32, words)
    )
    return b.bitcast(loaded, logical)


def _kernel(gfx: str, op_id: str):
    """One MMA between operands loaded from memory, every accumulator element
    stored, so that optimization keeps the instruction."""
    op = ArchTarget.from_gfx(gfx).mma.by_op_id(op_id)
    b = IRBuilder(f"cdna_{op_id}")
    b.kernel.attrs["max_workgroup_size"] = op.wave_size
    a = _operand(b, "A", op.a_dtype, op.a_frag_len)
    bv = _operand(b, "B", op.b_dtype, op.b_frag_len)
    c_elem = dtype_to_ir_type(op.c_dtype)
    out = b.param("D", PtrType(c_elem, "global"))
    d = b.mma(op, a, bv, b.zero_vec(c_elem, op.c_frag_len))
    base = b.mul(b.thread_id_x(), b.const_i32(op.c_frag_len))
    for slot in range(op.c_frag_len):
        b.global_store(out, b.add(base, b.const_i32(slot)), b.vec_extract(d, slot))
    b.ret()
    return b.kernel


def _call_argument_types(text: str, intrinsic: str) -> list[str]:
    """The LLVM type of each argument of the one call to ``intrinsic``."""
    (line,) = [
        line
        for line in text.splitlines()
        if f"@{intrinsic}(" in line and " call " in line
    ]
    arguments = line.split(f"@{intrinsic}(", 1)[1].rsplit(")", 1)[0]
    return [argument.rsplit(" ", 1)[0] for argument in arguments.split(", ")]


@pytest.mark.parametrize("gfx, op_id", _PAIRS)
def test_the_target_lists_the_instruction(gfx, op_id):
    case = _CASES[gfx][op_id]
    op = ArchTarget.from_gfx(gfx).mma.by_op_id(op_id)
    assert op is not None
    assert (op.family, op.a_dtype, op.b_dtype, op.c_dtype) == (
        "mma",
        case.a,
        case.b,
        case.c,
    )
    assert (op.m, op.n, op.k) == case.shape
    assert (op.a_frag_len, op.b_frag_len, op.c_frag_len, op.wave_size) == (
        *case.frag,
        64,
    )


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
@pytest.mark.parametrize("gfx, op_id", _PAIRS)
def test_both_engines_call_the_intrinsic_with_its_register_types(gfx, op_id, flavor):
    case = _CASES[gfx][op_id]
    text = lower_with_both_engines(_kernel(gfx, op_id), gfx, flavor)
    operand = case.operand[flavor] if isinstance(case.operand, dict) else case.operand
    assert _call_argument_types(text, case.intrinsic) == [
        operand,
        operand,
        case.accumulator,
        "i32",
        "i32",
        "i32",
    ]


@pytest.mark.parametrize("gfx, op_id", _PAIRS)
def test_the_target_compiles_links_and_selects_the_instruction(gfx, op_id):
    """Compile-verified at the host's LLVM flavor, not run."""
    text = lower_with_both_engines(_kernel(gfx, op_id), gfx)
    assert_target_accepts(text, gfx)
    mnemonics = target_mnemonics(text, gfx)
    assert [m for m in mnemonics if m.startswith("v_mfma")] == [
        _CASES[gfx][op_id].mnemonic
    ]


@pytest.mark.parametrize("gfx, op_id", _PAIRS)
def test_the_lane_maps_place_every_element_once(gfx, op_id):
    """64 lanes x the fragment length cover A (M x K), B (K x N) and the
    accumulator (M x N) exactly once, on the dense MFMA closed form: lane l
    holds row l % M of A and the K run starting at (l // M) * (K per lane)."""
    op = ArchTarget.from_gfx(gfx).mma.by_op_id(op_id)
    m, n, k = op.m, op.n, op.k
    a = lane_placements(op.a_layout(), 64)
    b = lane_placements(op.b_layout(), 64)
    c = lane_placements(op.c_layout(), 64)
    assert sorted(a) == [(row, kk) for row in range(m) for kk in range(k)]
    assert sorted(b) == [(kk, col) for kk in range(k) for col in range(n)]
    assert sorted(c) == [(row, col) for row in range(m) for col in range(n)]
    per_lane = op.a_frag_len
    assert a == [
        (lane % m, (lane // m) * per_lane + slot)
        for lane in range(64)
        for slot in range(per_lane)
    ]
    assert b == [(kk, col) for col, kk in a]


def test_the_f64_accumulator_holds_rows_four_apart():
    """Slot i of lane l holds (4 * i + l // 16, l % 16), not the f32 layout."""
    op = ArchTarget.from_gfx("gfx90a").mma.by_op_id("mfma_f64_16x16x4_f64")
    assert lane_placements(op.c_layout(), 64) == [
        (4 * slot + lane // 16, lane % 16) for lane in range(64) for slot in range(4)
    ]


def test_gfx90a_bf16_selection_keeps_the_1k_mfmas():
    """The bf16 MFMAs without the ``_1k`` suffix have the smaller K at their
    shape on gfx90a, so the largest-K selection is unchanged."""
    catalog = ArchTarget.from_gfx("gfx90a").mma
    for m, op_id in ((16, "mfma_f32_16x16x16_bf16"), (32, "mfma_f32_32x32x8_bf16")):
        selected = catalog.select_largest_k(
            a_dtype="bf16", b_dtype="bf16", c_dtype="fp32", m=m, n=m
        )
        assert selected.op_id == op_id
