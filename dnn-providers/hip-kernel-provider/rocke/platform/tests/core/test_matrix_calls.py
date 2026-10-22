# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Matrix intrinsic calls described as data, and the operand forms they use.

A ``MatrixCall`` names an intrinsic, the order of its arguments, the form each
operand and the result take, and its declaration key. Its ``declare`` line is
derived from the same forms the call is rendered from, so the two cannot
disagree at any LLVM flavor.

The form and error cases below lower calls a test builds, through the Python
lowerer, against real MMA catalog op_ids where the rule consults the catalog.
The C++ engine's twin of the same cases is ``tests/core/matrix_call_lowering.cpp``.
The cases that loop over ``MATRIX_CALLS`` hold every table entry, in both
engines, as later changes add them.
"""

from __future__ import annotations

import pytest
from rocke.core.arch import ArchTarget, known_arches
from rocke.core.arch.mfma_scale import gfx950_scaled_mfma
from rocke.core.arch.wmma_scale import gfx1250_scaled_wmma
from rocke.core.ir import (
    BF16,
    F16,
    F32,
    F64,
    FP8E4M3,
    I1,
    I16,
    I32,
    I64,
    IRBuilder,
    Op,
    PtrType,
    Type,
    Value,
    VectorType,
    dtype_to_ir_type,
)
from rocke.core.isa import backend as isa_backend
from rocke.core.isa.matrix_call_lowering import emit_matrix_call
from rocke.core.isa.matrix_calls import (
    MATRIX_CALLS,
    Imm,
    MatrixCall,
    MatrixOperand,
    Signedness,
)
from rocke.core.isa.matrix_operand_forms import AsIs, EvenHalves, FlavorType, ZeroPadded
from rocke.core.lower_llvm import (
    LLVM_FLAVOR_LLVM20,
    LLVM_FLAVOR_LLVM22,
    LLVM_FLAVOR_LLVM23,
    LLVM_FLAVORS,
    ArchDomainWarning,
    _decls_for_flavor,
    _llvm_type,
    _Lowerer,
)
from rocke.tests.engines import assert_both_engines_reject, lower_with_both_engines

A, B, C = MatrixOperand.A, MatrixOperand.B, MatrixOperand.C

# The gfx1250 K=64 fp8 WMMA, described as data. Its hand-written declaration
# exists, so the derived one must equal it at every flavor.
_FP8_WMMA = MatrixCall(
    intrinsic="llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8.v8f32.v8i32",
    args=(A, B, Imm(I16, 0), C, Imm(I1, 0), Imm(I1, 0)),
    a=AsIs(VectorType(I32, 8)),
    b=AsIs(VectorType(I32, 8)),
    c=AsIs(VectorType(F32, 8)),
    result=AsIs(VectorType(F32, 8)),
    decl_key="wmma.gfx1250.f32.16x16x64.fp8.fp8",
)

# An operand whose register type changed at LLVM 22 (the CDNA fp8 MFMAs take
# i64 there and <2 x i32> before).
_PER_FLAVOR = MatrixCall(
    intrinsic="llvm.amdgcn.example.per.flavor",
    args=(A, B, C, Imm(I32, 0), Imm(I32, 0), Imm(I32, 0)),
    a=AsIs(FlavorType(VectorType(I32, 2), ((LLVM_FLAVOR_LLVM22, I64),))),
    b=AsIs(FlavorType(VectorType(I32, 2), ((LLVM_FLAVOR_LLVM22, I64),))),
    c=AsIs(VectorType(F32, 4)),
    result=AsIs(VectorType(F32, 4)),
)


def _lowerer(arch: str, flavor: str = LLVM_FLAVOR_LLVM20) -> _Lowerer:
    return _Lowerer(IRBuilder("matrix_call").kernel, llvm_flavor=flavor, arch=arch)


def _mma_op(op_id: str, a: Type, b: Type, c: Type, result: Type) -> Op:
    """An MMA op named ``tile.<op_id>`` on values ``%a``, ``%b``, ``%c``."""
    return Op(
        name=f"tile.{op_id}",
        operands=[Value("%a", a), Value("%b", b), Value("%c", c)],
        results=[Value("%d", result)],
    )


def _emitted(lowerer: _Lowerer) -> list[str]:
    return [line.strip() for line in lowerer._current().lines]


# ----------------------------------------------------------------- per flavor


def test_a_flavor_type_takes_the_newest_override_at_or_below_the_flavor():
    t = FlavorType(VectorType(I32, 2), ((LLVM_FLAVOR_LLVM22, I64),))
    assert t.for_flavor(LLVM_FLAVOR_LLVM20) == VectorType(I32, 2)
    assert t.for_flavor(LLVM_FLAVOR_LLVM22) == I64
    assert t.for_flavor(LLVM_FLAVOR_LLVM23) == I64


def test_a_flavor_type_refuses_an_unknown_flavor():
    choices = r"; expected one of \('llvm20', 'llvm22', 'llvm23', 'llvm24'\)$"
    with pytest.raises(
        ValueError, match=r"^FlavorType names unknown LLVM flavor 'llvm99'" + choices
    ):
        FlavorType(I32, (("llvm99", I64),))
    with pytest.raises(ValueError, match=r"^unknown LLVM flavor 'llvm99'" + choices):
        FlavorType(I32).for_flavor("llvm99")


def test_the_declaration_spells_each_flavors_register_type():
    assert _PER_FLAVOR.declaration(LLVM_FLAVOR_LLVM20) == (
        "declare <4 x float> @llvm.amdgcn.example.per.flavor(<2 x i32>, <2 x i32>, "
        "<4 x float>, i32 immarg, i32 immarg, i32 immarg)"
    )
    assert _PER_FLAVOR.declaration(LLVM_FLAVOR_LLVM23) == (
        "declare <4 x float> @llvm.amdgcn.example.per.flavor(i64, i64, "
        "<4 x float>, i32 immarg, i32 immarg, i32 immarg)"
    )


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_a_derived_declaration_equals_the_hand_written_one(flavor):
    """Describing an existing instruction as data reproduces its declaration."""
    assert _FP8_WMMA.declaration(flavor) == _decls_for_flavor(flavor)[_FP8_WMMA.key]


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_the_call_and_its_declaration_take_the_same_types(flavor):
    """The lowered call passes each operand as the type the declaration names,
    and the module declares the call with the derived text."""
    lowerer = _lowerer("gfx942", flavor)
    op = _mma_op(
        "example_per_flavor",
        *(_PER_FLAVOR.a.register_type_at(flavor),) * 2,
        VectorType(F32, 4),
        VectorType(F32, 4),
    )
    emit_matrix_call(lowerer, op, _PER_FLAVOR)
    operand = _llvm_type(_PER_FLAVOR.a.register_type_at(flavor))
    assert _emitted(lowerer) == [
        (
            "%d = call <4 x float> @llvm.amdgcn.example.per.flavor("
            f"{operand} %a, {operand} %b, <4 x float> %c, i32 0, i32 0, i32 0)"
        ),
    ]
    assert _PER_FLAVOR.declaration(flavor) in lowerer.finalize().splitlines()


# ---------------------------------------------------------------------- forms


def test_as_is_passes_an_operand_of_the_register_type_unchanged():
    lowerer = _lowerer("gfx1250")
    op = _mma_op(
        "wmma_gfx1250_f32_16x16x64_fp8_fp8",
        VectorType(I32, 8),
        VectorType(I32, 8),
        VectorType(F32, 8),
        VectorType(F32, 8),
    )
    emit_matrix_call(lowerer, op, _FP8_WMMA)
    assert _emitted(lowerer) == [
        (
            "%d = call <8 x float> @llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8.v8f32.v8i32("
            "<8 x i32> %a, <8 x i32> %b, i16 0, <8 x float> %c, i1 0, i1 0)"
        ),
    ]


def test_as_is_reinterprets_the_catalog_element_with_one_bitcast():
    """fp8 bytes are the instruction's packed operand: one bitcast per operand."""
    lowerer = _lowerer("gfx1250")
    op = _mma_op(
        "wmma_gfx1250_f32_16x16x64_fp8_fp8",
        VectorType(FP8E4M3, 32),
        VectorType(FP8E4M3, 32),
        VectorType(F32, 8),
        VectorType(F32, 8),
    )
    emit_matrix_call(lowerer, op, _FP8_WMMA)
    lines = _emitted(lowerer)
    assert lines[:2] == [
        "%reg.1 = bitcast <32 x i8> %a to <8 x i32>",
        "%reg.2 = bitcast <32 x i8> %b to <8 x i32>",
    ]
    assert "(<8 x i32> %reg.1, <8 x i32> %reg.2, i16 0," in lines[2]


_I16_ACCUMULATOR = MatrixCall(
    intrinsic="llvm.amdgcn.example.i16.acc",
    args=(A, B, C),
    a=AsIs(VectorType(I32, 8)),
    b=AsIs(VectorType(I32, 8)),
    c=AsIs(VectorType(I16, 8)),
    result=AsIs(VectorType(F32, 4)),
)


def test_as_is_reinterprets_an_accumulator_of_the_catalog_element():
    """C follows the same rule as A and B, with the catalog's accumulator dtype
    (fp32 for this op)."""
    lowerer = _lowerer("gfx1250")
    i32x8, f32x4 = VectorType(I32, 8), VectorType(F32, 4)
    op = _mma_op("wmma_gfx1250_f32_16x16x64_fp8_fp8", i32x8, i32x8, f32x4, f32x4)
    emit_matrix_call(lowerer, op, _I16_ACCUMULATOR)
    assert _emitted(lowerer)[0] == "%reg.1 = bitcast <4 x float> %c to <8 x i16>"


def test_as_is_refuses_an_accumulator_of_another_number_format():
    i32x8 = VectorType(I32, 8)
    op = _mma_op(
        "wmma_gfx1250_f32_16x16x64_fp8_fp8",
        i32x8,
        i32x8,
        VectorType(F16, 8),
        VectorType(F32, 4),
    )
    with pytest.raises(ValueError) as error:
        emit_matrix_call(_lowerer("gfx1250"), op, _I16_ACCUMULATOR)
    assert str(error.value) == (
        "MMA operand c of 'wmma_gfx1250_f32_16x16x64_fp8_fp8' holds f16 values; the "
        "instruction takes f32 or i16 -- convert the values first"
    )


def test_as_is_bitcasts_a_result_of_the_same_bits_back():
    call = MatrixCall(
        intrinsic="llvm.amdgcn.example.i16.result",
        args=(A, B, C),
        a=AsIs(VectorType(I16, 8)),
        b=AsIs(VectorType(I16, 8)),
        c=AsIs(VectorType(I16, 8)),
        result=AsIs(VectorType(I16, 8)),
    )
    lowerer = _lowerer("gfx1201")
    i16x8 = VectorType(I16, 8)
    emit_matrix_call(
        lowerer, _mma_op("example", i16x8, i16x8, i16x8, VectorType(BF16, 8)), call
    )
    assert _emitted(lowerer) == [
        (
            "%mma_d.1 = call <8 x i16> @llvm.amdgcn.example.i16.result("
            "<8 x i16> %a, <8 x i16> %b, <8 x i16> %c)"
        ),
        "%d = bitcast <8 x i16> %mma_d.1 to <8 x bfloat>",
    ]


def test_as_is_refuses_a_result_of_another_bit_width():
    call = MatrixCall(
        intrinsic="llvm.amdgcn.example",
        args=(A, B, C),
        a=AsIs(VectorType(F32, 4)),
        b=AsIs(VectorType(F32, 4)),
        c=AsIs(VectorType(F32, 4)),
        result=AsIs(VectorType(F32, 4)),
    )
    f32x4 = VectorType(F32, 4)
    with pytest.raises(
        ValueError,
        match=r"^MMA result of 'example' is vec<f32x8> \(256 bits\); the instruction "
        r"returns vec<f32x4> \(128 bits\)$",
    ):
        emit_matrix_call(
            _lowerer("gfx942"),
            _mma_op("example", f32x4, f32x4, f32x4, VectorType(F32, 8)),
            call,
        )


_PADDED = MatrixCall(
    intrinsic="llvm.amdgcn.example.padded",
    args=(A, B, C),
    a=ZeroPadded(VectorType(I32, 8)),
    b=ZeroPadded(VectorType(I32, 8)),
    c=AsIs(VectorType(F32, 4)),
    result=AsIs(VectorType(F32, 4)),
)


def test_zero_padded_widens_a_packed_payload_with_zero_words():
    lowerer = _lowerer("gfx950")
    op = _mma_op(
        "example",
        VectorType(I32, 4),
        VectorType(I32, 8),
        VectorType(F32, 4),
        VectorType(F32, 4),
    )
    emit_matrix_call(lowerer, op, _PADDED)
    assert _emitted(lowerer) == [
        (
            "%mma_pad.1 = shufflevector <4 x i32> %a, <4 x i32> zeroinitializer, "
            "<8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4>"
        ),
        (
            "%d = call <4 x float> @llvm.amdgcn.example.padded("
            "<8 x i32> %mma_pad.1, <8 x i32> %b, <4 x float> %c)"
        ),
    ]


@pytest.mark.parametrize(
    "operand_b",
    [VectorType(F16, 8), VectorType(I32, 12), I32],
    ids=["another_format", "too_many_words", "not_a_vector"],
)
def test_zero_padded_refuses_anything_but_whole_words_that_fit(operand_b):
    message = (
        "MMA operand b of 'mfma_f32_16x16x128_f8f6f4_fp8e4m3_fp4e2m1' is "
        f"{operand_b.name}; the instruction takes a packed payload of whole 32-bit "
        "words in <8 x i32> -- pass the operand's packed words"
    )
    op = _mma_op(
        "mfma_f32_16x16x128_f8f6f4_fp8e4m3_fp4e2m1",
        VectorType(I32, 8),
        operand_b,
        VectorType(F32, 4),
        VectorType(F32, 4),
    )
    with pytest.raises(ValueError) as error:
        emit_matrix_call(_lowerer("gfx950"), op, _PADDED)
    assert str(error.value) == message


_EVEN_HALVES = MatrixCall(
    intrinsic="llvm.amdgcn.wmma.f16.16x16x16.f16.v16f16.v16f16",
    args=(A, B, C, Imm(I1, 0)),
    a=AsIs(VectorType(F16, 16)),
    b=AsIs(VectorType(F16, 16)),
    c=EvenHalves(VectorType(F16, 16)),
    result=EvenHalves(VectorType(F16, 16)),
)


def test_even_halves_spreads_the_accumulator_and_gathers_the_result():
    lowerer = _lowerer("gfx1151")
    f16x16, f16x8 = VectorType(F16, 16), VectorType(F16, 8)
    emit_matrix_call(
        lowerer,
        _mma_op("wmma_f16_16x16x16_f16", f16x16, f16x16, f16x8, f16x8),
        _EVEN_HALVES,
    )
    assert _emitted(lowerer) == [
        (
            "%mma_spread.1 = shufflevector <8 x half> %c, <8 x half> zeroinitializer, "
            "<16 x i32> <i32 0, i32 8, i32 1, i32 8, i32 2, i32 8, i32 3, i32 8, "
            "i32 4, i32 8, i32 5, i32 8, i32 6, i32 8, i32 7, i32 8>"
        ),
        (
            "%mma_d.2 = call <16 x half> @llvm.amdgcn.wmma.f16.16x16x16.f16.v16f16.v16f16("
            "<16 x half> %a, <16 x half> %b, <16 x half> %mma_spread.1, i1 0)"
        ),
        (
            "%d = shufflevector <16 x half> %mma_d.2, <16 x half> poison, "
            "<8 x i32> <i32 0, i32 2, i32 4, i32 6, i32 8, i32 10, i32 12, i32 14>"
        ),
    ]


def test_even_halves_refuses_an_accumulator_of_the_register_width():
    f16x16 = VectorType(F16, 16)
    op = _mma_op("wmma_f16_16x16x16_f16", f16x16, f16x16, f16x16, VectorType(F16, 8))
    with pytest.raises(ValueError) as error:
        emit_matrix_call(_lowerer("gfx1151"), op, _EVEN_HALVES)
    assert str(error.value) == (
        "MMA operand c of 'wmma_f16_16x16x16_f16' is vec<f16x16>; the instruction "
        "keeps 8 results in the even halves of <16 x half> -- pass the 8 logical "
        "accumulator values"
    )


# ---------------------------------------------------------------- the schema


def test_a_call_passes_each_operand_exactly_once():
    with pytest.raises(
        ValueError,
        match=r"MatrixCall 'llvm.amdgcn.x' must pass each of A, B and C exactly once, "
        r"got \['A', 'B', 'B'\]",
    ):
        MatrixCall(
            "llvm.amdgcn.x", (A, B, B), AsIs(F32), AsIs(F32), AsIs(F32), AsIs(F32)
        )


def test_a_call_produces_exactly_one_result():
    op = Op(
        name="tile.mfma_f32_16x16x32_fp8",
        operands=[
            Value("%a", VectorType(I32, 2)),
            Value("%b", VectorType(I32, 2)),
            Value("%c", VectorType(F32, 4)),
        ],
        results=[],
    )
    with pytest.raises(
        ValueError,
        match=r"^tile.mfma_f32_16x16x32_fp8: expected exactly one result, got 0$",
    ):
        emit_matrix_call(_lowerer("gfx942"), op, _PER_FLAVOR)


def test_a_result_is_never_zero_padded():
    with pytest.raises(TypeError, match=r"result form must be AsIs or EvenHalves"):
        MatrixCall(
            "llvm.amdgcn.x",
            (A, B, C),
            AsIs(F32),
            AsIs(F32),
            AsIs(F32),
            ZeroPadded(VectorType(I32, 4)),
        )


def test_forms_refuse_a_register_type_they_cannot_fill():
    with pytest.raises(TypeError, match=r"vector of i32 words, got vec<f16x8>"):
        ZeroPadded(VectorType(F16, 8))
    with pytest.raises(
        TypeError, match=r"even number of 16-bit elements, got vec<f32x8>"
    ):
        EvenHalves(VectorType(F32, 8))


def test_an_immediate_is_a_fitting_integer():
    assert Imm(I1, 1).argument() == "i1 1"
    assert Imm(I32, 3, immarg=False).parameter() == "i32"
    with pytest.raises(TypeError, match=r"an immediate is one of i1, i16, i32, i64"):
        Imm(F32, 0)
    with pytest.raises(ValueError, match=r"2 does not fit an i1 immediate"):
        Imm(I1, 2)


def test_a_signedness_flag_follows_the_catalog_rows_operand_dtype():
    """1 before a signed (i8, i4) operand, 0 before an unsigned (u8, u4) one,
    read from the row the op_id names; the declaration says only i1."""
    gfx12 = ArchTarget.from_gfx("gfx12-generic").mma
    signed = gfx12.by_op_id("wmma_gfx12_i32_16x16x16_iu8")
    mixed = gfx12.by_op_id("wmma_gfx12_i32_16x16x32_iu4_u4_i4")
    assert Signedness(A).argument(signed) == "i1 1"
    assert Signedness(B).argument(signed) == "i1 1"
    assert Signedness(A).argument(mixed) == "i1 0"
    assert Signedness(B).argument(mixed) == "i1 1"
    assert Signedness(A).parameter() == "i1 immarg"
    assert Signedness(A, immarg=False).parameter() == "i1"


def test_a_signedness_flag_precedes_a_or_b_and_needs_an_integer_operand():
    with pytest.raises(
        ValueError, match=r"^a signedness flag precedes the A or B operand, got C$"
    ):
        Signedness(C)
    f16 = ArchTarget.from_gfx("gfx12-generic").mma.by_op_id(
        "wmma_gfx12_f32_16x16x16_f16"
    )
    with pytest.raises(
        ValueError,
        match=r"^tile.wmma_gfx12_f32_16x16x16_f16 operand a is fp16; a signedness "
        r"flag is derived from an integer operand \(i4, i8, u4, u8, i16, i32, i64\)$",
    ):
        Signedness(A).argument(f16)


def test_a_signedness_flag_is_refused_where_the_catalog_does_not_list_the_op():
    """The flag needs the target's row for the op_id; a target without one
    refuses the call with the catalog error, before any flag is printed. (The
    declaration it registers first draws the target's arch-domain warning.)"""
    call = MATRIX_CALLS["wmma_gfx12_i32_16x16x16_iu8"]
    i32x2, i32x8 = VectorType(I32, 2), VectorType(I32, 8)
    op = _mma_op("wmma_gfx12_i32_16x16x16_iu8", i32x2, i32x2, i32x8, i32x8)
    with pytest.warns(
        ArchDomainWarning, match=r"not available on gfx1151"
    ), pytest.raises(
        NotImplementedError,
        match=r"^tile.wmma_gfx12_i32_16x16x16_iu8 is not available on gfx1151: "
        r"its MMA catalog does not list it$",
    ):
        emit_matrix_call(_lowerer("gfx1151"), op, call)


def test_every_description_is_hashable():
    """Calls are table values and dictionary keys elsewhere; a field that
    cannot be hashed would break both."""
    for call in (_FP8_WMMA, _PER_FLAVOR, _PADDED, _EVEN_HALVES, *MATRIX_CALLS.values()):
        hash(call)
        for operand in (A, B, C):
            hash(call.form(operand))
        for arg in call.args:
            hash(arg)


def test_a_table_entry_must_match_the_hand_written_declaration_it_names():
    """A decl_key that names an existing entry with other text is refused
    rather than lowered against a declaration it does not match."""
    wrong = MatrixCall(
        intrinsic="llvm.amdgcn.wmma.f32.16x16x64.fp8.fp8.v8f32.v8i32",
        args=(A, B, C),
        a=AsIs(VectorType(I32, 8)),
        b=AsIs(VectorType(I32, 8)),
        c=AsIs(VectorType(F32, 8)),
        result=AsIs(VectorType(F32, 8)),
        decl_key="wmma.gfx1250.f32.16x16x64.fp8.fp8",
    )
    i32x8, f32x8 = VectorType(I32, 8), VectorType(F32, 8)
    op = _mma_op("wmma_gfx1250_f32_16x16x64_fp8_fp8", i32x8, i32x8, f32x8, f32x8)
    with pytest.raises(
        ValueError,
        match=r"matrix call for 'wmma_gfx1250_f32_16x16x64_fp8_fp8' declares "
        r"'wmma.gfx1250.f32.16x16x64.fp8.fp8' as ",
    ):
        emit_matrix_call(_lowerer("gfx1250"), op, wrong)


# ------------------------------------------------------- the table, entry by entry


def _hand_written_op_ids() -> set[str]:
    """Every op_id a hand-written handler lowers."""
    handled = {
        name.removeprefix("_op_tile_")
        for name in dir(_Lowerer)
        if name.startswith("_op_tile_")
    }
    for table in (
        isa_backend._RDNA_WMMA,
        isa_backend._RDNA_GFX12_WMMA,
        isa_backend._GFX1250_WMMA,
        isa_backend._GFX1250_WMMA_FP8,
    ):
        handled |= {name.removeprefix("tile.") for name in table}
    for arch in known_arches():
        for op in ArchTarget.from_gfx(arch).mma.ops:
            if (
                gfx1250_scaled_wmma(f"tile.{op.op_id}") is not None
                or gfx950_scaled_mfma(op.op_id) is not None
            ):
                handled.add(op.op_id)
    return handled


def test_an_op_id_has_a_call_description_or_a_handler_never_both():
    assert sorted(set(MATRIX_CALLS) & _hand_written_op_ids()) == []


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
def test_every_table_declaration_is_in_the_flavors_declaration_table(flavor):
    """Entries sharing a key (one intrinsic, several op_ids) derive one text."""
    table = _decls_for_flavor(flavor)
    for op_id, call in MATRIX_CALLS.items():
        assert table[call.key] == call.declaration(flavor), op_id


def _value_of(b: IRBuilder, t: Type):
    """A zero of type ``t``, the type a form takes its operand in."""
    if isinstance(t, VectorType):
        return b.zero_vec(t.elem, t.count)
    constant = {F32: b.const_f32, F64: b.const_f64, I32: b.const_i32, I64: b.const_i64}
    return constant[t](0)


def _operand_type(form, flavor: str) -> Type:
    register = form.register_type_at(flavor)
    if isinstance(form, EvenHalves):
        return VectorType(register.elem, register.count // 2)
    return register


def _call_argument_types(text: str, intrinsic: str) -> list[str]:
    line = next(
        line
        for line in text.splitlines()
        if f"@{intrinsic}(" in line and "call" in line
    )
    arguments = line.split(f"@{intrinsic}(", 1)[1].rsplit(")", 1)[0]
    return [argument.rsplit(" ", 1)[0] for argument in arguments.split(", ")]


@pytest.mark.parametrize("flavor", LLVM_FLAVORS)
@pytest.mark.parametrize("op_id", sorted(MATRIX_CALLS))
def test_every_entry_lowers_to_its_declaration_in_both_engines(op_id, flavor):
    """For every table entry: the module declares the derived text and the call
    passes each argument as the type that declaration names."""
    call = MATRIX_CALLS[op_id]
    arch = next(a for a in known_arches() if ArchTarget.from_gfx(a).mma.by_op_id(op_id))
    op = ArchTarget.from_gfx(arch).mma.by_op_id(op_id)
    b = IRBuilder(f"matrix_call_{op_id}")
    out = b.param("D", PtrType(dtype_to_ir_type(op.c_dtype), "global"))
    result = b.mma(
        op,
        *(_value_of(b, _operand_type(call.form(x), flavor)) for x in (A, B, C)),
    )
    b.global_store(out, b.thread_id_x(), b.vec_extract(result, 0))
    text = lower_with_both_engines(b.kernel, arch, flavor)

    assert call.declaration(flavor) in text.splitlines()
    assert _call_argument_types(text, call.intrinsic) == [
        (
            arg.llvm_type.name
            if isinstance(arg, Imm)
            else (
                "i1"
                if isinstance(arg, Signedness)
                else _llvm_type(call.form(arg).register_type_at(flavor))
            )
        )
        for arg in call.args
    ]


@pytest.mark.parametrize("op_id", sorted(MATRIX_CALLS))
def test_every_entry_is_refused_where_the_catalog_does_not_list_it(op_id):
    """A table entry lowers only for a target whose MMA catalog lists it: every
    other known target refuses it in both engines, as the hand-written MMA
    handlers and the scaled WMMA path do."""
    call = MATRIX_CALLS[op_id]
    op = next(
        ArchTarget.from_gfx(a).mma.by_op_id(op_id)
        for a in known_arches()
        if ArchTarget.from_gfx(a).mma.by_op_id(op_id)
    )
    b = IRBuilder(f"matrix_call_{op_id}")
    out = b.param("D", PtrType(dtype_to_ir_type(op.c_dtype), "global"))
    result = b.mma(
        op,
        *(
            _value_of(b, _operand_type(call.form(x), LLVM_FLAVOR_LLVM22))
            for x in (A, B, C)
        ),
    )
    b.global_store(out, b.thread_id_x(), b.vec_extract(result, 0))
    for arch in known_arches():
        if ArchTarget.from_gfx(arch).mma.by_op_id(op_id) is None:
            assert_both_engines_reject(
                b.kernel,
                arch,
                NotImplementedError,
                f"tile.{op_id} is not available on {arch}: "
                "its MMA catalog does not list it",
                llvm_flavor=LLVM_FLAVOR_LLVM22,
            )
