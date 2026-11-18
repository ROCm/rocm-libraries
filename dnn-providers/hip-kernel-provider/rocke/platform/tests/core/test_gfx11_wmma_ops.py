# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx11 WMMA instructions with a 16-bit result: f16 -> f16 and
bf16 -> bf16.

The gfx11-generic row lists both; every gfx11 processor takes them through its
``facts_of`` pointer. The instruction keeps its eight results per lane in the
even 16-bit halves of a sixteen-slot register (op_sel 0), so the catalog row
and the IR count the eight logical results and the lowering spreads the
accumulator into the even halves and gathers the result back. Each case below
pins, for one instruction:

- the catalog row on gfx11-generic, gfx1100 and gfx1151 (dtypes, shape,
  fragment lengths per lane);
- the lowered call in both engines at every LLVM flavor: one spread of the
  accumulator, the call with its argument types and op_sel 0, one gather of
  the result;
- that a sixteen-value accumulator is refused with the even-halves error;
- that the module compiles and links for gfx1100, gfx1151 and gfx11-generic
  and that LLVM selects the named instruction (compile-verified, not run);
- that the result's lane map is the f32-output instruction's, slot for slot.

The C++ engine's catalog rows and lane maps are pinned to the same data by
``tests/core/gfx11_wmma_ops.cpp``.
"""

from __future__ import annotations

from typing import NamedTuple

import pytest
from rocke.core.arch import ArchTarget
from rocke.core.ir import IRBuilder, PtrType, dtype_to_ir_type
from rocke.core.lower_llvm import LLVM_FLAVORS
from rocke.tests.engines import assert_both_engines_reject, lower_with_both_engines
from rocke.tests.ir_validity import assert_target_accepts, target_mnemonics
from rocke.tests.lane_maps import lane_placements


class _Case(NamedTuple):
    """One instruction: its catalog contract and how it lowers."""

    dtype: str  # the dtype of A, B, the accumulator and the result
    register: str  # the sixteen-slot register type of every operand
    logical: str  # the eight logical accumulator values in LLVM
    ir_accumulator: str  # an accumulator of the register's width, in IR
    intrinsic: str
    mnemonic: str
    f32_output_op_id: str  # the same instruction with an f32 result


_CASES = {
    "wmma_f16_16x16x16_f16": _Case(
        "fp16", "<16 x half>", "<8 x half>", "vec<f16x16>",
        "llvm.amdgcn.wmma.f16.16x16x16.f16.v16f16.v16f16",
        "v_wmma_f16_16x16x16_f16", "wmma_f32_16x16x16_f16",
    ),
    "wmma_bf16_16x16x16_bf16": _Case(
        "bf16", "<16 x i16>", "<8 x i16>", "vec<bf16x16>",
        "llvm.amdgcn.wmma.bf16.16x16x16.bf16.v16i16.v16i16",
        "v_wmma_bf16_16x16x16_bf16", "wmma_f32_16x16x16_bf16",
    ),
}  # fmt: skip

# The rows that hold the instructions: the generic row and two processors that
# point at it.
_CATALOG_TARGETS = ("gfx11-generic", "gfx1100", "gfx1151")
# The targets the lowered module must compile for.
_COMPILE_TARGETS = ("gfx1100", "gfx1151", "gfx11-generic")

_SPREAD_MASK = ", ".join(f"i32 {i}, i32 8" for i in range(8))
_GATHER_MASK = ", ".join(f"i32 {2 * i}" for i in range(8))


def _kernel(gfx: str, op_id: str, c_len: int | None = None):
    """One MMA between operands loaded from memory, every result element
    stored, so that optimization keeps the instruction. ``c_len`` overrides
    the length of the accumulator passed in."""
    op = ArchTarget.from_gfx(gfx).mma.by_op_id(op_id)
    elem = dtype_to_ir_type(op.c_dtype)
    b = IRBuilder(f"gfx11_{op_id}")
    b.kernel.attrs["max_workgroup_size"] = op.wave_size
    operands = []
    for name, count in (("A", op.a_frag_len), ("B", op.b_frag_len)):
        ptr = b.param(name, PtrType(elem, "global"))
        index = b.mul(b.thread_id_x(), b.const_i32(count))
        operands.append(b.global_load_vN(ptr, index, elem, count))
    out = b.param("D", PtrType(elem, "global"))
    d = b.mma(op, *operands, b.zero_vec(elem, c_len or op.c_frag_len))
    base = b.mul(b.thread_id_x(), b.const_i32(op.c_frag_len))
    for slot in range(op.c_frag_len):
        b.global_store(out, b.add(base, b.const_i32(slot)), b.vec_extract(d, slot))
    b.ret()
    return b.kernel


def _lines_with(text: str, needle: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if needle in line]


@pytest.mark.parametrize("gfx", _CATALOG_TARGETS)
@pytest.mark.parametrize("op_id", sorted(_CASES))
def test_the_target_lists_the_instruction(op_id, gfx):
    dtype = _CASES[op_id].dtype
    op = ArchTarget.from_gfx(gfx).mma.by_op_id(op_id)
    assert op is not None
    assert (op.family, op.a_dtype, op.b_dtype, op.c_dtype) == (
        "wmma",
        dtype,
        dtype,
        dtype,
    )
    assert (op.m, op.n, op.k) == (16, 16, 16)
    assert (op.a_frag_len, op.b_frag_len, op.c_frag_len, op.wave_size) == (
        16,
        16,
        8,
        32,
    )


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
@pytest.mark.parametrize("op_id", sorted(_CASES))
def test_the_accumulator_is_spread_into_the_even_halves_and_the_result_gathered(
    op_id, flavor
):
    """Exactly one spread before the call and one gather after it: the call
    takes A, B and the spread accumulator in the sixteen-slot register with
    op_sel 0, and only the even halves of what it returns are read back."""
    case = _CASES[op_id]
    text = lower_with_both_engines(
        _kernel("gfx11-generic", op_id), "gfx11-generic", flavor
    )
    (spread,) = _lines_with(text, f"= shufflevector {case.logical} ")
    (call,) = _lines_with(text, f"call {case.register} @{case.intrinsic}(")
    (gather,) = _lines_with(text, f"= shufflevector {case.register} ")
    spread_name, spread_value = spread.split(" = ", 1)
    assert spread_value.endswith(
        f"{case.logical} zeroinitializer, <16 x i32> <{_SPREAD_MASK}>"
    )
    returned, call_value = call.split(" = ", 1)
    arguments = call_value.split(f"@{case.intrinsic}(", 1)[1].rsplit(")", 1)[0]
    assert [a.rsplit(" ", 1)[0] for a in arguments.split(", ")] == [
        *[case.register] * 3,
        "i1",
    ]
    assert arguments.endswith(f"{case.register} {spread_name}, i1 0")
    gathered, gather_value = gather.split(" = ", 1)
    assert gather_value == (
        f"shufflevector {case.register} {returned}, {case.register} poison, "
        f"<8 x i32> <{_GATHER_MASK}>"
    )
    if case.dtype == "bf16":
        # The bf16 form returns i16 halves: one bitcast gives the bf16 result.
        (result,) = _lines_with(text, f"bitcast <8 x i16> {gathered} to")
        assert result.endswith(f"bitcast <8 x i16> {gathered} to <8 x bfloat>")


@pytest.mark.parametrize("op_id", sorted(_CASES))
def test_an_accumulator_of_the_register_width_is_refused(op_id):
    """The accumulator is the eight logical results; passing sixteen values
    (the register's width) is refused in both engines."""
    case = _CASES[op_id]
    assert_both_engines_reject(
        _kernel("gfx11-generic", op_id, c_len=16),
        "gfx11-generic",
        ValueError,
        f"MMA operand c of '{op_id}' is {case.ir_accumulator}; the instruction "
        f"keeps 8 results in the even halves of {case.register} -- pass the 8 "
        "logical accumulator values",
    )


@pytest.mark.parametrize("target", _COMPILE_TARGETS)
@pytest.mark.parametrize("op_id", sorted(_CASES))
def test_the_target_compiles_links_and_selects_the_instruction(op_id, target):
    """Lowered for gfx11-generic at the host's LLVM flavor and compiled for each
    target; compile-verified, not run."""
    text = lower_with_both_engines(_kernel("gfx11-generic", op_id), "gfx11-generic")
    assert_target_accepts(text, target)
    mnemonics = target_mnemonics(text, target)
    assert [m for m in mnemonics if m.startswith("v_wmma")] == [_CASES[op_id].mnemonic]


@pytest.mark.parametrize("op_id", sorted(_CASES))
def test_the_result_takes_the_f32_output_instructions_lane_map(op_id):
    """Logical result slot i of lane l is where the f32-output instruction puts
    its slot i: (row 2 * i + l // 16, column l % 16). A and B take that
    instruction's maps too."""
    catalog = ArchTarget.from_gfx("gfx11-generic").mma
    op = catalog.by_op_id(op_id)
    f32_output = catalog.by_op_id(_CASES[op_id].f32_output_op_id)
    c = lane_placements(op.c_layout(), 32)
    assert c == lane_placements(f32_output.c_layout(), 32)
    assert c == [
        (2 * slot + lane // 16, lane % 16) for lane in range(32) for slot in range(8)
    ]
    assert sorted(c) == sorted((i, j) for i in range(16) for j in range(16))
    assert lane_placements(op.a_layout(), 32) == lane_placements(
        f32_output.a_layout(), 32
    )
    assert lane_placements(op.b_layout(), 32) == lane_placements(
        f32_output.b_layout(), 32
    )
