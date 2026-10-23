# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx1250 WMMA instructions with a 16-bit or integer result, and f32 16x16x4.

The gfx1250 row lists f16 -> f16 and bf16 -> bf16 at K 32, the four fp8 / bf8
-> f16 pairs at K 64 and at K 128, i8 -> i32 at K 64 in every operand
signedness, and f32 -> f32 at K 4, which was listed before but had no lowering.
Each case below pins, for one instruction:

- the catalog row (dtypes, shape, fragment lengths in logical values per lane);
- the lowered call in both engines at every LLVM flavor: the intrinsic, the
  register type of every argument and the immediates, including the signedness
  flags the row's operand dtypes give;
- that the module compiles and links for gfx1250 and that LLVM selects the
  named instruction (compile-verified, not run; skips where the compiler does
  not accept gfx1250, and for the i8 instruction where the compiler is older
  than LLVM 23).

The new instructions have no lane maps yet: no map is checked against an
independent source. The f32 16x16x4 A and B maps are CK's (lane l holds row
l % 16 of A, column l % 16 of B, and the K values 2 * (l // 16) + slot); they
are not validated on hardware. The C++ engine's rows and maps are pinned to
the same data by ``tests/core/gfx1250_wmma_ops.cpp``.
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
    type_bits,
)
from rocke.core.lower_llvm import (
    LLVM_FLAVOR_LLVM23,
    LLVM_FLAVORS,
    _resolve_llvm_flavor,
)
from rocke.tests.engines import lower_with_both_engines
from rocke.tests.ir_validity import assert_target_accepts, target_mnemonics
from rocke.tests.lane_maps import lane_placements


class _Case(NamedTuple):
    """One instruction: its catalog contract and how it lowers."""

    a: str
    b: str
    c: str
    k: int
    k_per_lane: int  # values of A (and of B) per lane
    intrinsic: str
    arguments: tuple[str, ...]  # every call argument's type, immediates as written
    mnemonic: str


# (neg A, A, neg B, B, C modifier, C, reuse A, reuse B), every modifier 0.
def _modified(operand: str, accumulator: str) -> tuple[str, ...]:
    return ("i1 0", operand, "i1 0", operand, "i16 0", accumulator, "i1 0", "i1 0")


# (A, B, C modifier, C, reuse A, reuse B), every modifier 0.
def _fp8(words: int) -> tuple[str, ...]:
    operand = f"<{words} x i32>"
    return (operand, operand, "i16 0", "<8 x half>", "i1 0", "i1 0")


# The signedness flag of an operand of each integer dtype.
_FLAG = {"i8": "i1 1", "u8": "i1 0"}


def _integer_cases() -> dict[str, _Case]:
    """The case of each signedness combination of the i8 WMMA: (signed A, A,
    signed B, B, C, clamp, reuse A, reuse B)."""
    dtypes = (("i8", "i8"), ("u8", "i8"), ("i8", "u8"), ("u8", "u8"))
    op_ids = integer_mma_op_ids("wmma_gfx1250_i32_16x16x64_iu8", 8)
    return {
        op_id: _Case(
            a,
            b,
            "i32",
            64,
            32,
            "llvm.amdgcn.wmma.i32.16x16x64.iu8.v8i32.v8i32",
            (_FLAG[a], "<8 x i32>", _FLAG[b], "<8 x i32>", "<8 x i32>") + ("i1 0",) * 3,
            "v_wmma_i32_16x16x64_iu8",
        )
        for op_id, (a, b) in zip(op_ids, dtypes)
    }


def _fp8_cases() -> dict[str, _Case]:
    """The fp8 / bf8 -> f16 cases: 32 values per lane in eight words at K 64,
    64 in sixteen words at K 128."""
    dtype = {"fp8": "fp8e4m3", "bf8": "bf8e5m2"}
    return {
        f"wmma_gfx1250_f16_16x16x{k}_{a}_{b}": _Case(
            dtype[a],
            dtype[b],
            "fp16",
            k,
            k // 2,
            f"llvm.amdgcn.wmma.f16.16x16x{k}.{a}.{b}.v8f16.v{k // 8}i32",
            _fp8(k // 8),
            f"v_wmma_f16_16x16x{k}_{a}_{b}",
        )
        for k in (64, 128)
        for a in ("fp8", "bf8")
        for b in ("fp8", "bf8")
    }


_F32_OP_ID = "wmma_gfx1250_f32_16x16x4_f32"
_CASES = {
    _F32_OP_ID: _Case(
        "fp32", "fp32", "fp32", 4, 2,
        "llvm.amdgcn.wmma.f32.16x16x4.f32.v8f32.v2f32",
        _modified("<2 x float>", "<8 x float>"),
        "v_wmma_f32_16x16x4_f32",
    ),
    "wmma_gfx1250_f16_16x16x32_f16": _Case(
        "fp16", "fp16", "fp16", 32, 16,
        "llvm.amdgcn.wmma.f16.16x16x32.f16.v8f16.v16f16",
        _modified("<16 x half>", "<8 x half>"),
        "v_wmma_f16_16x16x32_f16",
    ),
    "wmma_gfx1250_bf16_16x16x32_bf16": _Case(
        "bf16", "bf16", "bf16", 32, 16,
        "llvm.amdgcn.wmma.bf16.16x16x32.bf16.v8bf16.v16bf16",
        _modified("<16 x bfloat>", "<8 x bfloat>"),
        "v_wmma_bf16_16x16x32_bf16",
    ),
    **_fp8_cases(),
    **_integer_cases(),
}  # fmt: skip

# Every instruction this file covers except f32 16x16x4, which keeps its maps.
_UNMAPPED = sorted(set(_CASES) - {_F32_OP_ID})


def _op(op_id: str):
    return ArchTarget.from_gfx("gfx1250").mma.by_op_id(op_id)


def _operand(b: IRBuilder, name: str, dtype: str, count: int):
    """Operand ``name``: each lane's ``count`` values of ``dtype``, loaded as
    32-bit words from a kernel parameter (at most eight words per load) and
    reinterpreted as the values."""
    element = dtype_to_ir_type(dtype)
    words = type_bits(element) * count // 32
    ptr = b.param(name, PtrType(I32, "global"))
    base = b.mul(b.thread_id_x(), b.const_i32(words))
    chunk = min(words, 8)
    parts = [
        b.global_load_vN(ptr, b.add(base, b.const_i32(first)), I32, chunk)
        for first in range(0, words, chunk)
    ]
    loaded = parts[0] if len(parts) == 1 else b.vec_concat(*parts)
    return b.bitcast(loaded, VectorType(element, count))


def _kernel(op_id: str):
    """One MMA between operands loaded from memory, every result element
    stored, so that optimization keeps the instruction."""
    op = _op(op_id)
    b = IRBuilder(f"gfx1250_{op_id}")
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
    text for an ``i1`` / ``i16`` immediate."""
    (line,) = [
        line
        for line in text.splitlines()
        if f"@{intrinsic}(" in line and " call " in line
    ]
    arguments = line.split(f"@{intrinsic}(", 1)[1].rsplit(")", 1)[0].split(", ")
    return [
        a if a.startswith(("i1 ", "i16 ")) else a.rsplit(" ", 1)[0] for a in arguments
    ]


@pytest.mark.parametrize("op_id", sorted(_CASES))
def test_gfx1250_lists_the_instruction(op_id):
    case = _CASES[op_id]
    op = _op(op_id)
    assert op is not None
    assert (op.family, op.a_dtype, op.b_dtype, op.c_dtype, op.dst_dtype) == (
        "wmma",
        case.a,
        case.b,
        case.c,
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
@pytest.mark.parametrize("op_id", sorted(_CASES))
def test_both_engines_call_the_intrinsic_with_its_arguments(op_id, flavor):
    case = _CASES[op_id]
    text = lower_with_both_engines(_kernel(op_id), "gfx1250", flavor)
    assert _call_arguments(text, case.intrinsic) == list(case.arguments)


@pytest.mark.parametrize("op_id", sorted(_CASES))
def test_gfx1250_compiles_links_and_selects_the_instruction(op_id):
    """Lowered at the host's LLVM flavor and compiled for gfx1250; skips, naming
    the reason, where the compiler does not accept gfx1250. Compile-verified,
    not run.

    The i8 call is the LLVM 23 form: LLVM 22 declares the instruction without
    the clamp argument and rejects the call, which its arch-domain column
    records, so the i8 cases need an LLVM 23 compiler."""
    flavor = _resolve_llvm_flavor()
    if _CASES[op_id].c == "i32" and LLVM_FLAVORS.index(flavor) < LLVM_FLAVORS.index(
        LLVM_FLAVOR_LLVM23
    ):
        pytest.skip(f"the i8 WMMA call is the LLVM 23 form; the host is {flavor}")
    text = lower_with_both_engines(_kernel(op_id), "gfx1250")
    assert_target_accepts(text, "gfx1250")
    mnemonics = target_mnemonics(text, "gfx1250")
    assert [m for m in mnemonics if m.startswith("v_wmma")] == [_CASES[op_id].mnemonic]


@pytest.mark.parametrize("op_id", _UNMAPPED)
def test_the_new_instructions_have_no_lane_maps_yet(op_id):
    op = _op(op_id)
    for layout in (op.a_layout, op.b_layout, op.c_layout):
        with pytest.raises(NotImplementedError, match="no verified"):
            layout()


def test_the_f32_16x16x4_maps_are_ck_s_and_place_every_element_once():
    """Over 32 lanes x 2 slots, A lane l slot s holds (row l % 16,
    K 2 * (l // 16) + s) and B holds (K 2 * (l // 16) + s, column l % 16), CK's
    parameters; each covers its 16 x 4 (4 x 16) tile once. The result map is
    the gfx12 accumulator's, unchanged."""
    op = _op(_F32_OP_ID)
    a = lane_placements(op.a_layout(), 32)
    b = lane_placements(op.b_layout(), 32)
    c = lane_placements(op.c_layout(), 32)
    assert a == [
        (lane % 16, 2 * (lane // 16) + s) for lane in range(32) for s in (0, 1)
    ]
    assert b == [
        (2 * (lane // 16) + s, lane % 16) for lane in range(32) for s in (0, 1)
    ]
    assert sorted(a) == sorted((i, k) for i in range(16) for k in range(4))
    assert sorted(b) == sorted((k, j) for k in range(4) for j in range(16))
    assert c == [
        ((lane // 16) * 8 + slot, lane % 16) for lane in range(32) for slot in range(8)
    ]
