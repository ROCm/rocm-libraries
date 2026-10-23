# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The operand forms the gfx1250 scaled WMMA ops accept for A and B.

Each scaled op takes A and B in a ``<16 x i32>`` register. A kernel may hold an
operand in three shapes, and each operand reaches the register through the
f8f6f4 form of its own matrix format (``f8f6f4_operand_form``):

- the sixteen register words themselves, which pass unchanged, so the call is
  the one the scaled ops always emitted;
- the 64 logical values of an 8-bit format (``vec<fp8e4m3x64>``), which pass
  with one ``bitcast``;
- the packed words of 64 values of a 6-bit or 4-bit format (``vec<i32x12>``,
  ``vec<i32x8>``), which are zero-padded to sixteen words.

An operand of any other shape is refused with the form's error. Every case runs
over all 86 scaled ops of the gfx1250 catalog, in both engines, at LLVM 23 (the
only flavor that lowers these ops). The last tests compile and link the stack
forms for gfx1250 and check the instruction LLVM selects (compile-verified, not
run; they skip where the compiler does not accept gfx1250).
"""

from __future__ import annotations

import re
from functools import reduce

import pytest
from rocke.core.arch import ArchTarget
from rocke.core.arch.wmma_scale import gfx1250_scaled_wmma
from rocke.core.dtypes import dtype_info
from rocke.core.ir import (
    F32,
    I32,
    I64,
    IRBuilder,
    PtrType,
    VectorType,
    dtype_to_ir_type,
)
from rocke.core.isa.wmma_scale import ScaledWmmaLLVM
from rocke.tests.engines import assert_both_engines_reject, lower_with_both_engines
from rocke.tests.ir_validity import target_instructions

_FLAVOR = "llvm23"
_OPS = [
    op.op_id
    for op in ArchTarget.from_gfx("gfx1250").mma.ops
    if op.family == "wmma_scaled"
]


def _bits(dtype: str) -> int:
    return dtype_info(dtype).encoded_bits


def _words(b: IRBuilder, name: str, count: int):
    """``count`` 32-bit words per lane, loaded from kernel parameter ``name``
    four at a time and joined."""
    ptr = b.param(name, PtrType(I32, "global"))
    base = b.mul(b.thread_id_x(), b.const_i32(count))
    parts = [
        b.global_load_vN(ptr, b.add(base, b.const_i32(first)), I32, 4)
        for first in range(0, count, 4)
    ]
    return reduce(b.vec_concat, parts)


def _register_words(b: IRBuilder, name: str, _dtype: str):
    """The operand as the sixteen register words a caller packed by hand."""
    return _words(b, name, 16)


def _stack_form(b: IRBuilder, name: str, dtype: str):
    """The operand as the tile layer holds 64 values of ``dtype``: an 8-bit
    format as its logical values (``vec<fp8e4m3x64>``), a 6-bit or 4-bit one as
    its packed words (``vec<i32x12>``, ``vec<i32x8>``)."""
    words = _words(b, name, 64 * _bits(dtype) // 32)
    if _bits(dtype) < 8:
        return words
    return b.bitcast(words, VectorType(dtype_to_ir_type(dtype), 64))


def _kernel(op_id: str, a_operand, b_operand):
    """One scaled WMMA between operands built by ``a_operand`` / ``b_operand``
    (each called with the builder, a parameter name and the operand's format)
    and scales loaded from memory, every result element stored."""
    atom = gfx1250_scaled_wmma(op_id).atom
    scale = (
        I64 if ScaledWmmaLLVM(gfx1250_scaled_wmma(op_id)).scale_type == "i64" else I32
    )
    b = IRBuilder(f"gfx1250_{op_id}")
    b.kernel.attrs["max_workgroup_size"] = 32
    a = a_operand(b, "A", atom.a_dtype)
    bv = b_operand(b, "B", atom.b_dtype)
    scales = b.param("scales", PtrType(scale, "global"))
    lane = b.thread_id_x()
    a_scale = b.global_load(scales, lane, scale)
    b_scale = b.global_load(scales, b.add(lane, b.const_i32(32)), scale)
    out = b.param("D", PtrType(F32, "global"))
    d = b.mma(atom, a, bv, b.zero_vec(F32, 8), a_scale, b_scale)
    base = b.mul(lane, b.const_i32(8))
    for slot in range(8):
        b.global_store(out, b.add(base, b.const_i32(slot)), b.vec_extract(d, slot))
    b.ret()
    return b.kernel


def _call_line(text: str, op_id: str) -> str:
    """The one call to ``op_id``'s intrinsic."""
    intrinsic = ScaledWmmaLLVM(gfx1250_scaled_wmma(op_id)).intrinsic
    (line,) = [
        line
        for line in text.splitlines()
        if f"@{intrinsic}(" in line and " call " in line
    ]
    return line


def _expected_call(op_id: str) -> re.Pattern:
    """The call every develop caller's word-form operands have always produced,
    built from the op's contract: format, A, format, B, C modifier, C, then each
    operand's scale format and scale, and the two reuse flags. The names of A,
    B, C and the scales are captured."""
    spec = gfx1250_scaled_wmma(op_id)
    signature = ScaledWmmaLLVM(spec)
    fa, fb = spec.matrix_formats
    sa, sb = spec.scale_formats
    st = signature.scale_type
    name = r"(%\S+)"
    return re.compile(
        rf"  %\S+ = call <8 x float> @{re.escape(signature.intrinsic)}\("
        rf"i32 {fa}, <16 x i32> {name}, i32 {fb}, <16 x i32> {name}, "
        rf"i16 0, <8 x float> {name}, "
        rf"i32 0, i32 {sa}, {st} {name}, i32 0, i32 {sb}, {st} {name}, "
        rf"i1 false, i1 false\)"
    )


def _definition(text: str, name: str) -> str:
    """The right-hand side of the line that defines ``name``."""
    (line,) = [
        line for line in text.splitlines() if line.strip().startswith(f"{name} = ")
    ]
    return line.split(" = ", 1)[1]


def _padding(words: int) -> str:
    """The shufflevector mask that widens ``words`` words to sixteen with zeros."""
    return ", ".join(
        [f"i32 {i}" for i in range(words)] + [f"i32 {words}"] * (16 - words)
    )


def test_the_catalog_lists_86_scaled_ops_each_with_16_word_operands():
    assert len(_OPS) == 86
    for op_id in _OPS:
        signature = ScaledWmmaLLVM(gfx1250_scaled_wmma(op_id))
        assert signature.matrix_types == ("<16 x i32>", "<16 x i32>"), op_id
        assert signature.matrix_registers == (VectorType(I32, 16),) * 2, op_id


@pytest.mark.parametrize("op_id", _OPS)
def test_register_words_pass_unchanged(op_id):
    """Sixteen register words for A and B reach the call as they are: the
    call's A and B are the joined loads (the last element inserted), with no
    bitcast and no padding."""
    text = lower_with_both_engines(
        _kernel(op_id, _register_words, _register_words), "gfx1250", _FLAVOR
    )
    match = _expected_call(op_id).fullmatch(_call_line(text, op_id))
    assert match, _call_line(text, op_id)
    for value in match.group(1, 2):
        assert re.fullmatch(
            r"insertelement <16 x i32> %\S+, i32 %\S+, i32 15",
            _definition(text, value),
        ), value
    assert "zeroinitializer, <16 x i32>" not in text
    assert "to <16 x i32>" not in text


@pytest.mark.parametrize("op_id", _OPS)
def test_the_stack_forms_reach_the_16_word_register(op_id):
    """64 logical fp8 / bf8 values pass with one bitcast; the packed words of
    64 fp6 / bf6 (twelve) or fp4 (eight) values gain zero words up to sixteen.
    The call is otherwise the word-form call."""
    atom = gfx1250_scaled_wmma(op_id).atom
    text = lower_with_both_engines(
        _kernel(op_id, _stack_form, _stack_form), "gfx1250", _FLAVOR
    )
    match = _expected_call(op_id).fullmatch(_call_line(text, op_id))
    assert match, _call_line(text, op_id)
    for value, dtype in zip(match.group(1, 2), (atom.a_dtype, atom.b_dtype)):
        produced = _definition(text, value)
        if _bits(dtype) == 8:
            assert re.fullmatch(r"bitcast <64 x i8> %\S+ to <16 x i32>", produced)
            continue
        words = 64 * _bits(dtype) // 32
        assert re.fullmatch(
            rf"shufflevector <{words} x i32> %\S+, <{words} x i32> "
            rf"zeroinitializer, <16 x i32> <{_padding(words)}>",
            produced,
        ), produced


def _wrong_shape(dtype: str):
    """An operand of the wrong shape for its format: twenty packed words for
    fp4 (more than the register holds), 32 logical values for fp8 (half the
    register)."""

    def build(b: IRBuilder, name: str, operand_dtype: str):
        if operand_dtype != dtype:
            return _stack_form(b, name, operand_dtype)
        if dtype == "fp4e2m1":
            return _words(b, name, 20)
        return b.bitcast(_words(b, name, 8), VectorType(dtype_to_ir_type(dtype), 32))

    return build


_WRONG_SHAPES = [
    (op_id, role, dtype)
    for op_id in _OPS
    for role, dtype in zip(
        ("a", "b"),
        (
            gfx1250_scaled_wmma(op_id).atom.a_dtype,
            gfx1250_scaled_wmma(op_id).atom.b_dtype,
        ),
    )
    if dtype in ("fp4e2m1", "fp8e4m3")
]


@pytest.mark.parametrize(("op_id", "role", "dtype"), _WRONG_SHAPES)
def test_an_operand_of_another_shape_is_refused(op_id, role, dtype):
    """Twenty fp4 words do not fit the sixteen-word register (the zero-padding
    form's error); 32 fp8 values are half its bits (the bitcast form's error)."""
    wrong, right = _wrong_shape(dtype), _stack_form
    kernel = _kernel(op_id, *((wrong, right) if role == "a" else (right, wrong)))
    if dtype == "fp4e2m1":
        message = (
            f"MMA operand {role} of '{op_id}' is vec<i32x20>; the instruction takes "
            "a packed payload of whole 32-bit words in <16 x i32> -- pass the "
            "operand's packed words"
        )
    else:
        message = (
            f"MMA operand {role} of '{op_id}' is vec<fp8e4m3x32> (256 bits); the "
            "instruction takes vec<i32x16> (512 bits) -- pass a 512-bit value"
        )
    assert_both_engines_reject(kernel, "gfx1250", ValueError, message, _FLAVOR)


# The format name the disassembler prints for each matrix format.
_ISA_FORMAT = {
    "fp8e4m3": "MATRIX_FMT_FP8",
    "bf8e5m2": "MATRIX_FMT_BF8",
    "fp6e2m3": "MATRIX_FMT_FP6",
    "fp6e3m2": "MATRIX_FMT_BF6",
    "fp4e2m1": "MATRIX_FMT_FP4",
}


def _scaled_op(a: str, b: str, block_k: int) -> str:
    (op_id,) = [
        op.op_id
        for op in ArchTarget.from_gfx("gfx1250").mma.ops
        if op.family == "wmma_scaled"
        and (op.a_dtype, op.b_dtype, op.scale_block_k) == (a, b, block_k)
        and (op.a_scale_dtype, op.b_scale_dtype) == ("e8m0", "e8m0")
    ]
    return op_id


@pytest.mark.parametrize(
    ("a", "b", "block_k"),
    [
        ("fp8e4m3", "fp4e2m1", 32),
        ("fp6e2m3", "fp6e2m3", 16),
        ("bf8e5m2", "fp6e3m2", 32),
        ("fp4e2m1", "fp4e2m1", 16),
    ],
)
def test_gfx1250_compiles_links_and_selects_the_scaled_instruction(a, b, block_k):
    """The stack forms lowered at LLVM 23 compile and link for gfx1250, and the
    selected instruction carries each operand's format. Compile-verified, not
    run; skips, naming the reason, where the compiler does not accept gfx1250."""
    op_id = _scaled_op(a, b, block_k)
    text = lower_with_both_engines(
        _kernel(op_id, _stack_form, _stack_form), "gfx1250", _FLAVOR
    )
    (wmma,) = [i for i in target_instructions(text, "gfx1250") if "v_wmma" in i]
    mode = "scale16" if block_k == 16 else "scale"
    assert f"v_wmma_{mode}_f32_16x16x128_f8f6f4" in wmma, wmma
    for operand, dtype in (("a", a), ("b", b)):
        fmt = re.search(rf"matrix_{operand}_fmt:(\w+)", wmma)
        # The disassembler leaves out a format that is the default, fp8.
        assert (fmt.group(1) if fmt else "MATRIX_FMT_FP8") == _ISA_FORMAT[dtype], wmma
