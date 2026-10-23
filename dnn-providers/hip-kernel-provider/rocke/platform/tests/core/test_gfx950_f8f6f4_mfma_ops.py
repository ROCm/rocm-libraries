# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx950 f8f6f4 16x16x128 and 32x32x64 MFMA instructions.

The gfx950 row lists f8f6f4 -> f32 at 16x16x128 and at 32x32x64 for each of the
25 (A, B) format pairs: 50 rows. fp8 x fp8 at 16x16x128 is
``mfma_f32_16x16x128_fp8``, which its hand-written handler lowers; every other
row lowers from its ``MATRIX_CALLS`` entry. This file pins:

- the catalog rows (dtypes, shape, fragment lengths in logical values per lane);
- that selection now picks the f8f6f4 op for an 8-bit x 8-bit pair, the largest
  K at its shape;
- the lane maps: each operand's map is its own format's f8f6f4 operand layout,
  checked against the public CDNA4 ISA operand tables transcribed below, and
  the result map is the shape's f32 accumulator;
- the lowered call in both engines at every LLVM flavor: the nine-argument
  intrinsic, the format immediates, every scale argument 0, and the register
  type of every argument, with an 8-bit operand bitcast from its 32 logical
  values and a 6-bit or 4-bit one zero-padded from its packed words;
- that the module compiles and links for gfx950 and that LLVM selects the
  unscaled f8f6f4 instruction with the operands' formats in ``cbsz`` and
  ``blgp`` (compile-verified, not run; skips where the compiler does not accept
  gfx950).

The C++ engine's rows, maps and calls are pinned to the same data by
``tests/core/gfx950_f8f6f4_mfma_ops.cpp``.
"""

from __future__ import annotations

import re

import pytest
from rocke.core.arch import ArchTarget
from rocke.core.arch.mfma_scale import (
    MATRIX_FORMATS,
    eight_bit_matrix_format_pairs,
    narrow_matrix_format_pairs,
)
from rocke.core.arch.target import gfx950_eight_bit_f8f6f4_op_ids, narrow_f8f6f4_op_ids
from rocke.core.dtypes import dtype_info
from rocke.core.ir import I32, IRBuilder, PtrType, VectorType, dtype_to_ir_type
from rocke.core.lower_llvm import LLVM_FLAVORS
from rocke.tests.engines import lower_with_both_engines
from rocke.tests.ir_validity import target_instructions
from rocke.tests.lane_maps import lane_placements

# (M, N, K) -> (intrinsic, f32 results per lane).
_SHAPES = {
    (16, 16, 128): ("llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4", 4),
    (32, 32, 64): ("llvm.amdgcn.mfma.scale.f32.32x32x64.f8f6f4.v8i32.v8i32", 16),
}
_OPS = {
    op_id: (shape, a, b)
    for shape in _SHAPES
    for op_ids in (
        narrow_f8f6f4_op_ids("mfma_f32_{}x{}x{}_f8f6f4".format(*shape)),
        gfx950_eight_bit_f8f6f4_op_ids(shape),
    )
    for op_id, (a, b) in op_ids.items()
}


def _bits(dtype: str) -> int:
    return dtype_info(dtype).encoded_bits


def _op(op_id: str):
    return ArchTarget.from_gfx("gfx950").mma.by_op_id(op_id)


# --- the catalog rows -----------------------------------------------------------


def _op_id(shape, a: str, b: str) -> str:
    """The op_id of the f8f6f4 row reading ``a`` x ``b`` at ``shape``."""
    if (shape, a, b) == ((16, 16, 128), "fp8e4m3", "fp8e4m3"):
        return "mfma_f32_16x16x128_fp8"
    return "mfma_f32_{}x{}x{}_f8f6f4_{}_{}".format(*shape, a, b)


def test_gfx950_lists_one_f8f6f4_op_per_format_pair_and_shape():
    pairs = narrow_matrix_format_pairs() + eight_bit_matrix_format_pairs()
    assert len(pairs) == len(set(pairs)) == len(MATRIX_FORMATS) ** 2
    assert sorted(_OPS) == sorted(
        _op_id(shape, a, b) for shape in _SHAPES for a, b in pairs
    )
    listed = [
        op.op_id
        for op in ArchTarget.from_gfx("gfx950").mma.ops
        if op.family == "mma"
        and (op.m, op.n, op.k) in _SHAPES
        and {op.a_dtype, op.b_dtype} <= set(MATRIX_FORMATS)
    ]
    assert sorted(listed) == sorted(_OPS)


@pytest.mark.parametrize("shape", list(_SHAPES))
@pytest.mark.parametrize("a, b", eight_bit_matrix_format_pairs())
def test_an_8_bit_pair_selects_its_f8f6f4_op(shape, a, b):
    """The f8f6f4 op is the largest K for an 8-bit x 8-bit pair at its shape
    (K 128 at 16 x 16, K 64 at 32 x 32), so selection moves to it."""
    m, n, k = shape
    selected = ArchTarget.from_gfx("gfx950").mma.select_largest_k(
        a_dtype=a, b_dtype=b, c_dtype="fp32", m=m, n=n
    )
    assert selected.op_id == _op_id(shape, a, b)
    assert selected.k == k


@pytest.mark.parametrize("op_id", list(_OPS))
def test_an_f8f6f4_row_reads_its_pair_at_its_shape(op_id):
    (m, n, k), a, b = _OPS[op_id]
    op = _op(op_id)
    assert (op.family, op.a_dtype, op.b_dtype, op.c_dtype, op.dst_dtype) == (
        "mma",
        a,
        b,
        "fp32",
        "fp32",
    )
    assert (op.m, op.n, op.k) == (m, n, k)
    assert (op.a_frag_len, op.b_frag_len, op.c_frag_len, op.wave_size) == (
        32,
        32,
        _SHAPES[(m, n, k)][1],
        64,
    )


# --- the lane maps against the public ISA tables --------------------------------

# The public CDNA4 ISA operand tables for the dense 8-bit-and-smaller MFMAs
# (section 7.1.5.1), transcribed: for each shape and operand width, the lane
# group a lane belongs to and, per group, the registers' runs of values as
# (first value of the run, its first K, its length). A lane holds row
# lane % (lanes per group) of A, the same column of B.
_ISA_LANES_PER_GROUP = {(16, 16, 128): 16, (32, 32, 64): 32}
_ISA_RUNS = {
    # 16x16x128: four groups of 16 lanes.
    ((16, 16, 128), 8): lambda g: [(0, 16 * g, 16), (16, 64 + 16 * g, 16)],
    ((16, 16, 128), 6): lambda g: [(0, 32 * g, 16), (16, 32 * g + 16, 16)],
    ((16, 16, 128), 4): lambda g: [(0, 32 * g, 32)],
    # 32x32x64: two groups of 32 lanes.
    ((32, 32, 64), 8): lambda g: [(0, 16 * g, 16), (16, 32 + 16 * g, 16)],
    ((32, 32, 64), 6): lambda g: [(0, 32 * g, 16), (16, 32 * g + 16, 16)],
    ((32, 32, 64), 4): lambda g: [(0, 32 * g, 32)],
}


def _isa_placements(shape, dtype: str) -> list[tuple[int, int]]:
    """(row or column, K) of every (lane, value) of one operand, lane-major,
    from the transcribed ISA table."""
    per_group = _ISA_LANES_PER_GROUP[shape]
    placed = {}
    for lane in range(64):
        for first, k0, length in _ISA_RUNS[(shape, _bits(dtype))](lane // per_group):
            for e in range(length):
                placed[(lane, first + e)] = (lane % per_group, k0 + e)
    return [placed[(lane, value)] for lane in range(64) for value in range(32)]


@pytest.mark.parametrize("op_id", list(_OPS))
def test_an_f8f6f4_operand_map_is_the_isa_table_of_its_format(op_id):
    """Over 64 lanes x 32 values, A lane l value s sits at the (row, K) the ISA
    table of A's format gives and B at the (K, column) of B's format; each
    covers its tile once. The result map is the shape's f32 accumulator."""
    shape, a, b = _OPS[op_id]
    m, n, k = shape
    op = _op(op_id)
    placed_a = lane_placements(op.a_layout(), 64)
    placed_b = lane_placements(op.b_layout(), 64)
    assert placed_a == _isa_placements(shape, a)
    assert placed_b == [kc[::-1] for kc in _isa_placements(shape, b)]
    assert sorted(placed_a) == [(i, kk) for i in range(m) for kk in range(k)]
    assert sorted(placed_b) == [(kk, j) for kk in range(k) for j in range(n)]
    accumulator = "mfma_f32_16x16x16_f16" if m == 16 else "mfma_f32_32x32x8_f16"
    assert lane_placements(op.c_layout(), 64) == lane_placements(
        _op(accumulator).c_layout(), 64
    )


# --- lowering -------------------------------------------------------------------


def _operand(b: IRBuilder, name: str, dtype: str):
    """32 values of ``dtype`` per lane, loaded from a kernel parameter as 32-bit
    words, two at a time: an 8-bit format reinterpreted as its logical values
    (``vec<fp8e4m3x32>``), a 6-bit or 4-bit format kept as its packed words
    (``vec<i32x6>`` for 32 fp6 values)."""
    words = 32 * _bits(dtype) // 32
    ptr = b.param(name, PtrType(I32, "global"))
    base = b.mul(b.thread_id_x(), b.const_i32(words))
    loaded = b.global_load_vN(ptr, base, I32, 2)
    for first in range(2, words, 2):
        loaded = b.vec_concat(
            loaded, b.global_load_vN(ptr, b.add(base, b.const_i32(first)), I32, 2)
        )
    if _bits(dtype) < 8:
        return loaded
    return b.bitcast(loaded, VectorType(dtype_to_ir_type(dtype), 32))


def _kernel(op_id: str):
    """One MMA between operands loaded from memory, every result element
    stored, so that optimization keeps the instruction."""
    op = _op(op_id)
    b = IRBuilder(f"gfx950_{op_id}")
    b.kernel.attrs["max_workgroup_size"] = op.wave_size
    a = _operand(b, "A", op.a_dtype)
    bv = _operand(b, "B", op.b_dtype)
    c_elem = dtype_to_ir_type(op.c_dtype)
    out = b.param("D", PtrType(c_elem, "global"))
    d = b.mma(op, a, bv, b.zero_vec(c_elem, op.c_frag_len))
    base = b.mul(b.thread_id_x(), b.const_i32(op.c_frag_len))
    for slot in range(op.c_frag_len):
        b.global_store(out, b.add(base, b.const_i32(slot)), b.vec_extract(d, slot))
    b.ret()
    return b.kernel


def _call(text: str, intrinsic: str) -> tuple[list[str], list[str]]:
    """The one call to ``intrinsic``: each argument's type (the whole text for
    an immediate) and each argument's value text."""
    (line,) = [
        line
        for line in text.splitlines()
        if f"@{intrinsic}(" in line and " call " in line
    ]
    arguments = line.split(f"@{intrinsic}(", 1)[1].rsplit(")", 1)[0].split(", ")
    types = [a if a.startswith("i32 ") else a.rsplit(" ", 1)[0] for a in arguments]
    return types, [a.rsplit(" ", 1)[1] for a in arguments]


def _definition(text: str, name: str) -> str:
    """The right-hand side of the line that defines ``name``."""
    (line,) = [
        line for line in text.splitlines() if line.strip().startswith(f"{name} = ")
    ]
    return line.split(" = ", 1)[1]


def _assert_operand_reaches_the_register(text: str, value: str, dtype: str) -> None:
    """An 8-bit operand arrives by one bitcast from its 32 logical values; a
    6-bit or 4-bit one by one zero-padding shufflevector from its packed words."""
    produced = _definition(text, value)
    if _bits(dtype) == 8:
        assert re.fullmatch(r"bitcast <32 x i8> %\S+ to <8 x i32>", produced), produced
        return
    words = _bits(dtype)
    mask = ", ".join(
        [f"i32 {i}" for i in range(words)] + [f"i32 {words}"] * (8 - words)
    )
    assert re.fullmatch(
        rf"shufflevector <{words} x i32> %\S+, <{words} x i32> zeroinitializer, "
        rf"<8 x i32> <{mask}>",
        produced,
    ), produced


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
@pytest.mark.parametrize("op_id", list(_OPS))
def test_both_engines_call_the_f8f6f4_mfma_with_its_formats(op_id, flavor):
    shape, a, b = _OPS[op_id]
    intrinsic, results = _SHAPES[shape]
    text = lower_with_both_engines(_kernel(op_id), "gfx950", flavor)
    types, values = _call(text, intrinsic)
    assert types == [
        "<8 x i32>",
        "<8 x i32>",
        f"<{results} x float>",
        f"i32 {MATRIX_FORMATS[a]}",
        f"i32 {MATRIX_FORMATS[b]}",
        "i32 0",
        "i32 0",
        "i32 0",
        "i32 0",
    ]
    _assert_operand_reaches_the_register(text, values[0], a)
    _assert_operand_reaches_the_register(text, values[1], b)


# The pairs compiled for gfx950 at both shapes: both operands narrow, an 8-bit
# operand with a narrow one, the bf8 / bf6 format codes, and two 8-bit x 8-bit
# pairs (fp8 x fp8 lowered by its hand-written handler at 16x16x128).
_COMPILED_PAIRS = [
    ("fp4e2m1", "fp4e2m1"),
    ("fp8e4m3", "fp4e2m1"),
    ("fp6e2m3", "fp6e2m3"),
    ("bf8e5m2", "fp6e3m2"),
    ("fp8e4m3", "fp8e4m3"),
    ("bf8e5m2", "fp8e4m3"),
]


@pytest.mark.parametrize("shape", list(_SHAPES))
@pytest.mark.parametrize("a, b", _COMPILED_PAIRS)
def test_gfx950_compiles_links_and_selects_the_f8f6f4_instruction(shape, a, b):
    """Lowered at the host's LLVM flavor and compiled for gfx950; skips, naming
    the reason, where the compiler does not accept gfx950. Compile-verified, not
    run. Every scale argument is 0, so LLVM selects the unscaled instruction;
    cbsz carries A's format code and blgp B's."""
    m, n, k = shape
    text = lower_with_both_engines(_kernel(_op_id(shape, a, b)), "gfx950")
    (mfma,) = [i for i in target_instructions(text, "gfx950") if "v_mfma" in i]
    assert re.search(rf"\bv_mfma_f32_{m}x{n}x{k}_f8f6f4\b", mfma), mfma
    for modifier, dtype in (("cbsz", a), ("blgp", b)):
        code = re.search(rf"\b{modifier}:(\d+)", mfma)
        # The disassembler leaves out a format code that is the default, 0.
        assert (int(code.group(1)) if code else 0) == MATRIX_FORMATS[dtype], mfma
