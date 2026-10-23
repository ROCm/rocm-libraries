# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx1250 f8f6f4 16x16x128 and f4 32x16x128 WMMA instructions.

The gfx1250 row lists f8f6f4 -> f32 at 16x16x128 for each of the 21 (A, B)
format pairs with a 6-bit or 4-bit operand, and fp4 x fp4 -> f32 at 32x16x128.
This file pins:

- the f8f6f4 operand layout (``matrix_operand_layout``): the K runs it gives,
  that the gfx1250 scaled layout and the gfx950 scaled MFMA layout are calls to
  it, and the formats it accepts;
- ``f8f6f4_operand_form``: an 8-bit format passes with at most one bitcast, a
  6-bit or 4-bit format is zero-padded from its packed words;
- the catalog rows (dtypes, shape, fragment lengths in logical values per lane)
  and the lane maps: each f8f6f4 operand's map is its own format's layout over
  two groups of 16 lanes, the result map is the gfx12 accumulator's, and f4
  32x16x128 has no maps;
- the lowered call in both engines at every LLVM flavor: the intrinsic, the
  format immediates and the register type of every argument, with an 8-bit
  operand bitcast from its logical values and a 6-bit or 4-bit one zero-padded
  from its packed words;
- that the module compiles and links for gfx1250 and that LLVM selects the
  instruction with the operands' formats (compile-verified, not run; skips
  where the compiler does not accept gfx1250).

The C++ engine's rows, maps and layouts are pinned to the same data by
``tests/core/gfx1250_f8f6f4_wmma_ops.cpp``.
"""

from __future__ import annotations

import re
from functools import reduce

import pytest
from rocke.core.arch import ArchTarget
from rocke.core.arch.mfma_scale import (
    MATRIX_FORMATS,
    matrix_layout,
    matrix_operand_layout,
    narrow_matrix_format_pairs,
)
from rocke.core.arch.target import narrow_f8f6f4_op_ids
from rocke.core.arch.wmma_scale import scaled_matrix_layout
from rocke.core.dtypes import dtype_info
from rocke.core.ir import (
    I32,
    IRBuilder,
    PtrType,
    VectorType,
    dtype_to_ir_type,
)
from rocke.core.isa.matrix_operand_forms import AsIs, ZeroPadded, f8f6f4_operand_form
from rocke.core.lower_llvm import LLVM_FLAVORS
from rocke.tests.engines import lower_with_both_engines
from rocke.tests.ir_validity import target_instructions
from rocke.tests.lane_maps import lane_placements

_F8F6F4 = "llvm.amdgcn.wmma.f32.16x16x128.f8f6f4.v8f32.v16i32.v16i32"
_F4 = "llvm.amdgcn.wmma.f32.32x16x128.f4.v16f32.v16i32.v8i32"
_F4_OP_ID = "wmma_gfx1250_f32_32x16x128_f4"
_OPS = narrow_f8f6f4_op_ids("wmma_gfx1250_f32_16x16x128_f8f6f4")
# The format name the disassembler prints for each matrix format.
_ISA_FORMAT = {
    "fp8e4m3": "MATRIX_FMT_FP8",
    "bf8e5m2": "MATRIX_FMT_BF8",
    "fp6e2m3": "MATRIX_FMT_FP6",
    "fp6e3m2": "MATRIX_FMT_BF6",
    "fp4e2m1": "MATRIX_FMT_FP4",
}


def _bits(dtype: str) -> int:
    return dtype_info(dtype).encoded_bits


# --- the operand layout and form ----------------------------------------------


def test_the_narrow_pairs_are_every_pair_with_a_6_or_4_bit_operand():
    pairs = narrow_matrix_format_pairs()
    assert len(pairs) == 21 and len(set(pairs)) == 21
    assert pairs == [
        (a, b)
        for a in MATRIX_FORMATS
        for b in MATRIX_FORMATS
        if min(_bits(a), _bits(b)) < 8
    ]
    assert list(_OPS) == [
        f"wmma_gfx1250_f32_16x16x128_f8f6f4_{a}_{b}" for a, b in pairs
    ]


def test_an_operand_layout_takes_k_runs_of_16_or_32_in_turn():
    """gfx1250 fp8, 16 words over 2 x 16 lanes: lanes 0..15 hold K 0..15,
    32..47, 64..79, 96..111 and lanes 16..31 the runs between; a 6-bit or 4-bit
    format takes runs of 32 (lanes 0..15 hold K 0..31 and 64..95)."""
    fp8 = matrix_operand_layout("fp8e4m3", 16, 2, 16)
    assert fp8.fragment.count == 64 and fp8.chunk_elements == 16
    assert [fp8.coord(3, s)[1] for s in range(64)] == [
        k for run in (0, 32, 64, 96) for k in range(run, run + 16)
    ]
    assert [fp8.coord(19, s) for s in (0, 16)] == [(3, 16), (3, 48)]
    for dtype, words in (("fp6e2m3", 12), ("fp6e3m2", 12), ("fp4e2m1", 8)):
        narrow = matrix_operand_layout(dtype, 16, 2, 16)
        assert narrow.fragment.count == 64 and narrow.chunk_elements == 32
        assert narrow.fragment.live_carriers == words
        assert [narrow.coord(0, s)[1] for s in range(64)] == [
            *range(32),
            *range(64, 96),
        ]


@pytest.mark.parametrize("dtype", ["fp8", "bf8", "fp6", "bf6", "fp4"])
def test_the_scaled_layouts_are_operand_layouts(dtype):
    """The gfx1250 scaled WMMA operand (16 words, 2 x 16 lanes) and the gfx950
    scaled MFMA operand (8 words, 2 x 32 lanes) are f8f6f4 operand layouts."""
    canonical = dtype_info(dtype).name
    assert scaled_matrix_layout(dtype, 16) == matrix_operand_layout(dtype, 16, 2, 16)
    assert matrix_layout(canonical) == matrix_operand_layout(canonical, 8, 2, 32)


def test_an_operand_layout_names_the_matrix_formats():
    with pytest.raises(ValueError) as error:
        matrix_operand_layout("fp16", 16, 2, 16)
    assert str(error.value) == (
        "'fp16' is not an f8f6f4 matrix format; expected one of fp8e4m3, "
        "bf8e5m2, fp6e2m3, fp6e3m2, fp4e2m1"
    )


@pytest.mark.parametrize("dtype", list(MATRIX_FORMATS) + ["fp4", "bf6"])
def test_an_8_bit_operand_passes_as_is_and_a_narrower_one_is_zero_padded(dtype):
    register = VectorType(I32, 16)
    form = f8f6f4_operand_form(dtype, register)
    expected = AsIs if _bits(dtype) == 8 else ZeroPadded
    assert type(form) is expected and form.register_type == register


def test_an_operand_form_names_the_matrix_formats():
    with pytest.raises(ValueError, match="'i8' is not an f8f6f4 matrix format"):
        f8f6f4_operand_form("i8", VectorType(I32, 16))


# --- the catalog rows and lane maps -------------------------------------------


def _op(op_id: str):
    return ArchTarget.from_gfx("gfx1250").mma.by_op_id(op_id)


def test_gfx1250_lists_the_pairs_and_no_other_f8f6f4_op():
    listed = [
        op.op_id
        for op in ArchTarget.from_gfx("gfx1250").mma.ops
        if "_f8f6f4_" in op.op_id and op.family == "wmma"
    ]
    assert listed == list(_OPS)


@pytest.mark.parametrize("op_id", list(_OPS))
def test_an_f8f6f4_row_reads_its_pair_at_16x16x128(op_id):
    a, b = _OPS[op_id]
    op = _op(op_id)
    assert (op.family, op.a_dtype, op.b_dtype, op.c_dtype, op.dst_dtype) == (
        "wmma",
        a,
        b,
        "fp32",
        "fp32",
    )
    assert (op.m, op.n, op.k) == (16, 16, 128)
    assert (op.a_frag_len, op.b_frag_len, op.c_frag_len, op.wave_size) == (
        64,
        64,
        8,
        32,
    )


@pytest.mark.parametrize("op_id", list(_OPS))
def test_an_f8f6f4_operand_map_is_its_formats_layout(op_id):
    """Over 32 lanes x 64 slots, A lane l slot s sits where the A format's
    layout puts it, (row, K), and B where the B format's puts it, (K, column);
    each covers its 16 x 128 (128 x 16) tile once. The result map is the gfx12
    accumulator's."""
    a, b = _OPS[op_id]
    op = _op(op_id)
    a_layout = matrix_operand_layout(a, 16, 2, 16)
    b_layout = matrix_operand_layout(b, 16, 2, 16)
    placed_a = lane_placements(op.a_layout(), 32)
    placed_b = lane_placements(op.b_layout(), 32)
    assert placed_a == [
        a_layout.coord(lane, s) for lane in range(32) for s in range(64)
    ]
    assert placed_b == [
        b_layout.coord(lane, s)[::-1] for lane in range(32) for s in range(64)
    ]
    assert sorted(placed_a) == [(i, k) for i in range(16) for k in range(128)]
    assert sorted(placed_b) == [(k, j) for k in range(128) for j in range(16)]
    assert lane_placements(op.c_layout(), 32) == [
        ((lane // 16) * 8 + slot, lane % 16) for lane in range(32) for slot in range(8)
    ]


def test_the_f4_32x16x128_row_has_fragment_lengths_and_no_lane_maps():
    op = _op(_F4_OP_ID)
    assert (op.family, op.a_dtype, op.b_dtype, op.c_dtype) == (
        "wmma",
        "fp4e2m1",
        "fp4e2m1",
        "fp32",
    )
    assert (op.m, op.n, op.k) == (32, 16, 128)
    assert (op.a_frag_len, op.b_frag_len, op.c_frag_len, op.wave_size) == (
        128,
        64,
        16,
        32,
    )
    for layout in (op.a_layout, op.b_layout, op.c_layout):
        with pytest.raises(NotImplementedError, match="no verified"):
            layout()


# --- lowering -------------------------------------------------------------------


def _operand(b: IRBuilder, name: str, dtype: str, count: int):
    """``count`` values of ``dtype`` per lane, loaded from a kernel parameter as
    32-bit words, four at a time: an 8-bit format reinterpreted as its logical
    values (``vec<fp8e4m3x64>``), a 6-bit or 4-bit format kept as its packed
    words (``vec<i32x12>`` for 64 fp6 values)."""
    words = count * _bits(dtype) // 32
    ptr = b.param(name, PtrType(I32, "global"))
    base = b.mul(b.thread_id_x(), b.const_i32(words))
    parts = [
        b.global_load_vN(ptr, b.add(base, b.const_i32(first)), I32, 4)
        for first in range(0, words, 4)
    ]
    loaded = reduce(b.vec_concat, parts)
    if _bits(dtype) < 8:
        return loaded
    return b.bitcast(loaded, VectorType(dtype_to_ir_type(dtype), count))


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


def _call(text: str, intrinsic: str) -> tuple[list[str], list[str]]:
    """The one call to ``intrinsic``: each argument's type (the whole text for
    an immediate) and each argument's value name."""
    (line,) = [
        line
        for line in text.splitlines()
        if f"@{intrinsic}(" in line and " call " in line
    ]
    arguments = line.split(f"@{intrinsic}(", 1)[1].rsplit(")", 1)[0].split(", ")
    types = [
        a if a.startswith(("i1 ", "i16 ", "i32 ")) else a.rsplit(" ", 1)[0]
        for a in arguments
    ]
    return types, [a.rsplit(" ", 1)[1] for a in arguments]


def _definition(text: str, name: str) -> str:
    """The right-hand side of the line that defines ``name``."""
    (line,) = [
        line for line in text.splitlines() if line.strip().startswith(f"{name} = ")
    ]
    return line.split(" = ", 1)[1]


def _padding(words: int, width: int) -> str:
    """The shufflevector mask that widens ``words`` words to ``width`` with zeros."""
    return ", ".join(
        [f"i32 {i}" for i in range(words)] + [f"i32 {words}"] * (width - words)
    )


def _assert_operand_reaches_the_register(text: str, value: str, dtype: str) -> None:
    """An 8-bit operand arrives by one bitcast from its 64 logical values; a
    6-bit or 4-bit one by one zero-padding shufflevector from its packed words."""
    produced = _definition(text, value)
    if _bits(dtype) == 8:
        assert re.fullmatch(r"bitcast <64 x i8> %\S+ to <16 x i32>", produced)
        return
    words = 64 * _bits(dtype) // 32
    assert re.fullmatch(
        rf"shufflevector <{words} x i32> %\S+, <{words} x i32> zeroinitializer, "
        rf"<16 x i32> <{_padding(words, 16)}>",
        produced,
    ), produced


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
@pytest.mark.parametrize("op_id", list(_OPS))
def test_both_engines_call_the_f8f6f4_intrinsic_with_its_formats(op_id, flavor):
    a, b = _OPS[op_id]
    text = lower_with_both_engines(_kernel(op_id), "gfx1250", flavor)
    types, values = _call(text, _F8F6F4)
    assert types == [
        f"i32 {MATRIX_FORMATS[a]}",
        "<16 x i32>",
        f"i32 {MATRIX_FORMATS[b]}",
        "<16 x i32>",
        "i16 0",
        "<8 x float>",
    ]
    _assert_operand_reaches_the_register(text, values[1], a)
    _assert_operand_reaches_the_register(text, values[3], b)


def test_an_fp8_by_fp4_call_bitcasts_a_and_zero_pads_b():
    """fp8 x fp4: A arrives as vec<fp8e4m3x64> and passes with one bitcast; B
    arrives as the packed vec<i32x8> and gains eight zero words."""
    text = lower_with_both_engines(
        _kernel("wmma_gfx1250_f32_16x16x128_f8f6f4_fp8e4m3_fp4e2m1"), "gfx1250"
    )
    _, values = _call(text, _F8F6F4)
    assert re.fullmatch(
        r"bitcast <64 x i8> %\S+ to <16 x i32>", _definition(text, values[1])
    )
    assert re.fullmatch(
        r"shufflevector <8 x i32> %\S+, <8 x i32> zeroinitializer, <16 x i32> "
        r"<i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, "
        r"i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8, i32 8>",
        _definition(text, values[3]),
    )


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_both_engines_call_the_f4_32x16x128_intrinsic(flavor):
    """A's 128 fp4 values fill its sixteen words and B's 64 its eight, so both
    pass unchanged."""
    text = lower_with_both_engines(_kernel(_F4_OP_ID), "gfx1250", flavor)
    types, _ = _call(text, _F4)
    assert types == ["<16 x i32>", "<8 x i32>", "i16 0", "<16 x float>"]
    assert "zeroinitializer, <16 x i32>" not in text


@pytest.mark.parametrize(
    "op_id",
    [
        "wmma_gfx1250_f32_16x16x128_f8f6f4_fp8e4m3_fp4e2m1",
        "wmma_gfx1250_f32_16x16x128_f8f6f4_fp6e2m3_fp6e2m3",
        "wmma_gfx1250_f32_16x16x128_f8f6f4_bf8e5m2_fp6e3m2",
        "wmma_gfx1250_f32_16x16x128_f8f6f4_fp4e2m1_fp4e2m1",
    ],
)
def test_gfx1250_compiles_links_and_selects_the_f8f6f4_instruction(op_id):
    """Lowered at the host's LLVM flavor and compiled for gfx1250; skips, naming
    the reason, where the compiler does not accept gfx1250. Compile-verified,
    not run. The instruction carries each operand's format."""
    a, b = _OPS[op_id]
    text = lower_with_both_engines(_kernel(op_id), "gfx1250")
    (wmma,) = [i for i in target_instructions(text, "gfx1250") if "v_wmma" in i]
    assert "v_wmma_f32_16x16x128_f8f6f4" in wmma
    for operand, dtype in (("a", a), ("b", b)):
        fmt = re.search(rf"matrix_{operand}_fmt:(\w+)", wmma)
        # The disassembler leaves out a format that is the default, fp8.
        assert (fmt.group(1) if fmt else "MATRIX_FMT_FP8") == _ISA_FORMAT[dtype], wmma


def test_gfx1250_compiles_links_and_selects_the_f4_32x16x128_instruction():
    text = lower_with_both_engines(_kernel(_F4_OP_ID), "gfx1250")
    (wmma,) = [i for i in target_instructions(text, "gfx1250") if "v_wmma" in i]
    assert "v_wmma_f32_32x16x128_f4" in wmma
