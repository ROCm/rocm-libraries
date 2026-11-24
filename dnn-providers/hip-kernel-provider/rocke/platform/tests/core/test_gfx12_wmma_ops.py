# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx12 WMMA instructions beyond f16 / bf16 -> f32.

The gfx12-generic row lists the four fp8 / bf8 -> f32 WMMAs, i8 -> i32, i4 ->
i32 at K 16 and K 32 (each integer one in every operand signedness), and the
two 16-bit-output WMMAs (f16 -> f16, bf16 -> bf16); gfx1200 and gfx1201 take
them through their ``facts_of`` pointers. Each case below pins, for one
instruction:

- the catalog row on gfx12-generic, gfx1200 and gfx1201 (dtypes, shape,
  fragment lengths per lane);
- the lowered call in both engines at every LLVM flavor: the intrinsic, the
  register type of every argument and the immediates, including the
  signedness flag the row's operand dtypes give;
- that the module compiles and links for gfx1200, gfx1201 and gfx12-generic,
  and that LLVM selects the named instruction (compile-verified, not run);
- that its lane maps place every element of A, B and the result exactly once:
  lane l holds row l % 16 of A (column l % 16 of B) and a contiguous K run.

The C++ engine's catalog rows and lane maps are pinned to the same data by
``tests/core/gfx12_wmma_ops.cpp``.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
from rocke.core.arch import ArchTarget
from rocke.core.arch.target import integer_mma_op_ids
from rocke.core.ir import (
    I32,
    IRBuilder,
    PtrType,
    VectorType,
    dtype_to_ir_type,
    is_packed_only,
    type_bits,
)
from rocke.core.lower_llvm import LLVM_FLAVORS
from rocke.tests.engines import lower_with_both_engines
from rocke.tests.ir_validity import assert_target_accepts, target_mnemonics
from rocke.tests.lane_maps import lane_placements


class _Case(NamedTuple):
    """One instruction: its catalog contract and how it lowers."""

    a: str
    b: str
    c: str
    intrinsic: str
    arguments: tuple[str, ...]  # every call argument's type, immediates as written
    mnemonic: str
    k: int = 16
    k_per_lane: int = 8  # values of A (and of B) per lane: one lane's K run


_FP8 = ("<2 x i32>", "<2 x i32>", "<8 x float>")
_SIXTEEN_BIT_OUT = "<8 x {t}>", "<8 x {t}>", "<8 x {t}>", "i1 0"
# The signedness flag of an operand of each integer dtype.
_FLAG = {"i8": "i1 1", "i4": "i1 1", "u8": "i1 0", "u4": "i1 0"}


def _integer_cases(op_id, bits, intrinsic, words, mnemonic, k=16, k_per_lane=8):
    """The case of each signedness combination of one integer WMMA."""
    dtypes = ((f"i{bits}", f"i{bits}"), (f"u{bits}", f"i{bits}"))
    dtypes += ((f"i{bits}", f"u{bits}"), (f"u{bits}", f"u{bits}"))
    return {
        row: _Case(
            a,
            b,
            "i32",
            intrinsic,
            (_FLAG[a], words, _FLAG[b], words, "<8 x i32>", "i1 0"),
            mnemonic,
            k,
            k_per_lane,
        )
        for row, (a, b) in zip(integer_mma_op_ids(op_id, bits), dtypes)
    }


_CASES = {
    "wmma_gfx12_f32_16x16x16_fp8_fp8": _Case(
        "fp8e4m3", "fp8e4m3", "fp32",
        "llvm.amdgcn.wmma.f32.16x16x16.fp8.fp8.v8f32.v2i32", _FP8,
        "v_wmma_f32_16x16x16_fp8_fp8",
    ),
    "wmma_gfx12_f32_16x16x16_fp8_bf8": _Case(
        "fp8e4m3", "bf8e5m2", "fp32",
        "llvm.amdgcn.wmma.f32.16x16x16.fp8.bf8.v8f32.v2i32", _FP8,
        "v_wmma_f32_16x16x16_fp8_bf8",
    ),
    "wmma_gfx12_f32_16x16x16_bf8_fp8": _Case(
        "bf8e5m2", "fp8e4m3", "fp32",
        "llvm.amdgcn.wmma.f32.16x16x16.bf8.fp8.v8f32.v2i32", _FP8,
        "v_wmma_f32_16x16x16_bf8_fp8",
    ),
    "wmma_gfx12_f32_16x16x16_bf8_bf8": _Case(
        "bf8e5m2", "bf8e5m2", "fp32",
        "llvm.amdgcn.wmma.f32.16x16x16.bf8.bf8.v8f32.v2i32", _FP8,
        "v_wmma_f32_16x16x16_bf8_bf8",
    ),
    **_integer_cases(
        "wmma_gfx12_i32_16x16x16_iu8", 8,
        "llvm.amdgcn.wmma.i32.16x16x16.iu8.v8i32.v2i32", "<2 x i32>",
        "v_wmma_i32_16x16x16_iu8",
    ),
    **_integer_cases(
        "wmma_gfx12_i32_16x16x16_iu4", 4,
        "llvm.amdgcn.wmma.i32.16x16x16.iu4.v8i32.i32", "i32",
        "v_wmma_i32_16x16x16_iu4",
    ),
    **_integer_cases(
        "wmma_gfx12_i32_16x16x32_iu4", 4,
        "llvm.amdgcn.wmma.i32.16x16x32.iu4.v8i32.v2i32", "<2 x i32>",
        "v_wmma_i32_16x16x32_iu4", k=32, k_per_lane=16,
    ),
    "wmma_gfx12_f16_16x16x16_f16": _Case(
        "fp16", "fp16", "fp16",
        "llvm.amdgcn.wmma.f16.16x16x16.f16.v8f16.v8f16",
        tuple(t.format(t="half") for t in _SIXTEEN_BIT_OUT),
        "v_wmma_f16_16x16x16_f16",
    ),
    "wmma_gfx12_bf16_16x16x16_bf16": _Case(
        "bf16", "bf16", "bf16",
        "llvm.amdgcn.wmma.bf16.16x16x16.bf16.v8i16.v8i16",
        tuple(t.format(t="i16") for t in _SIXTEEN_BIT_OUT),
        "v_wmma_bf16_16x16x16_bf16",
    ),
}  # fmt: skip

# The rows that hold the instructions: the generic row and the processors that
# point at it.
_CATALOG_TARGETS = ("gfx12-generic", "gfx1200", "gfx1201")
# The targets the lowered module must compile for: every member of the generic
# target and the generic target itself.
_COMPILE_TARGETS = ("gfx1200", "gfx1201", "gfx12-generic")


def _operand(b: IRBuilder, name: str, dtype: str, count: int):
    """Operand ``name``: each lane's ``count`` values of ``dtype``, loaded as
    32-bit words from a kernel parameter. Values of a type that has a vector
    form are reinterpreted as it; packed-only i4 / u4 values stay in their
    words (one word is a plain i32)."""
    element = dtype_to_ir_type(dtype)
    bits = 4 if is_packed_only(element) else type_bits(element)
    words = bits * count // 32
    ptr = b.param(name, PtrType(I32, "global"))
    index = b.mul(b.thread_id_x(), b.const_i32(words))
    if words == 1:
        return b.global_load(ptr, index, I32)
    loaded = b.global_load_vN(ptr, index, I32, words)
    if is_packed_only(element):
        return loaded
    return b.bitcast(loaded, VectorType(element, count))


def _kernel(gfx: str, op_id: str):
    """One MMA between operands loaded from memory, every result element
    stored, so that optimization keeps the instruction."""
    op = ArchTarget.from_gfx(gfx).mma.by_op_id(op_id)
    b = IRBuilder(f"gfx12_{op_id}")
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


def _call_arguments(text: str, intrinsic: str) -> list[str]:
    """Each argument of the one call to ``intrinsic``: its type, or the whole
    text for an ``i1`` immediate."""
    (line,) = [
        line
        for line in text.splitlines()
        if f"@{intrinsic}(" in line and " call " in line
    ]
    arguments = line.split(f"@{intrinsic}(", 1)[1].rsplit(")", 1)[0].split(", ")
    return [a if a.startswith("i1 ") else a.rsplit(" ", 1)[0] for a in arguments]


@pytest.mark.parametrize("gfx", _CATALOG_TARGETS)
@pytest.mark.parametrize("op_id", sorted(_CASES))
def test_the_target_lists_the_instruction(op_id, gfx):
    case = _CASES[op_id]
    op = ArchTarget.from_gfx(gfx).mma.by_op_id(op_id)
    assert op is not None
    assert (op.family, op.a_dtype, op.b_dtype, op.c_dtype) == (
        "wmma",
        case.a,
        case.b,
        case.c,
    )
    assert (op.m, op.n, op.k) == (16, 16, case.k)
    assert (op.a_frag_len, op.b_frag_len, op.c_frag_len, op.wave_size) == (
        case.k_per_lane,
        case.k_per_lane,
        8,
        32,
    )


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
@pytest.mark.parametrize("gfx", _CATALOG_TARGETS)
@pytest.mark.parametrize("op_id", sorted(_CASES))
def test_both_engines_call_the_intrinsic_with_its_arguments(op_id, gfx, flavor):
    case = _CASES[op_id]
    text = lower_with_both_engines(_kernel(gfx, op_id), gfx, flavor)
    assert _call_arguments(text, case.intrinsic) == list(case.arguments)


def test_the_bf16_result_is_reinterpreted_from_the_i16_register():
    """The bf16 -> bf16 WMMA returns <8 x i16>; the IR result is the same bits
    as <8 x bfloat>, one bitcast, so a kernel stores bf16 values."""
    op_id = "wmma_gfx12_bf16_16x16x16_bf16"
    text = lower_with_both_engines(_kernel("gfx12-generic", op_id), "gfx12-generic")
    (call,) = [
        line
        for line in text.splitlines()
        if "@llvm.amdgcn.wmma.bf16" in line and " call " in line
    ]
    returned = call.split("=", 1)[0].strip()
    assert f"bitcast <8 x i16> {returned} to <8 x bfloat>" in text


@pytest.mark.parametrize("target", _COMPILE_TARGETS)
@pytest.mark.parametrize("op_id", sorted(_CASES))
def test_the_target_compiles_links_and_selects_the_instruction(op_id, target):
    """Lowered for gfx12-generic at the host's LLVM flavor and compiled for each
    member of the generic target; compile-verified, not run."""
    text = lower_with_both_engines(_kernel("gfx12-generic", op_id), "gfx12-generic")
    assert_target_accepts(text, target)
    mnemonics = target_mnemonics(text, target)
    assert [m for m in mnemonics if m.startswith("v_wmma")] == [_CASES[op_id].mnemonic]


@pytest.mark.parametrize("op_id", sorted(_CASES))
def test_the_lane_maps_place_every_element_once(op_id):
    """32 lanes cover A (16 x K), B (K x 16) and the result (16 x 16) exactly
    once: lane l holds row l % 16 of A (column l % 16 of B) and the K run of
    k_per_lane values starting at (l // 16) * k_per_lane; result slot i of lane
    l is (row (l // 16) * 8 + i, column l % 16)."""
    case = _CASES[op_id]
    kpl = case.k_per_lane
    op = ArchTarget.from_gfx("gfx12-generic").mma.by_op_id(op_id)
    a = lane_placements(op.a_layout(), 32)
    b = lane_placements(op.b_layout(), 32)
    c = lane_placements(op.c_layout(), 32)
    assert a == [
        (lane % 16, (lane // 16) * kpl + slot)
        for lane in range(32)
        for slot in range(kpl)
    ]
    assert b == [(k, col) for col, k in a]
    assert c == [
        ((lane // 16) * 8 + slot, lane % 16) for lane in range(32) for slot in range(8)
    ]
    assert sorted(a) == sorted((i, k) for i in range(16) for k in range(case.k))
    assert sorted(c) == sorted((i, j) for i in range(16) for j in range(16))
