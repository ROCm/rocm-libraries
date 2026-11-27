# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``arith.lshr`` and ``vector.lshr`` are logical shifts in the element's own width.

LLVM's ``lshr iW`` (and ``lshr <N x iW>``, lane by lane) shifts zeros in at
bit ``W - 1``. The HIP lowering reads the value, or each lane, through the
unsigned type of its width before shifting, so a negative i8 or i16 does not
shift its sign bits in and an i64 keeps its high word. Covers each engine's
HIP source (the C++ engine's must equal the Python engine's) and its refusal
of a non-integer shift, that hipcc accepts the source, and, when a GPU is
present, that the LLVM route and hipcc on the HIP source compute the same
values as numpy, on negative and positive values alike.

LLVM's ``lshr`` takes two integers of one width, so IRBuilder refuses a shift
of a float and a shift amount of another type when the op is built, and the
verifier reports both and every lowering refuses both for IR written past the
builder. The element rule is the one every bitwise op and shift shares
(``and``, ``or``, ``xor``, ``not``, ``shl``, ``lshr``, scalar and vector), and
the operand rule the one every same-type binary op (the arithmetic ones too)
and the lane-wise FMA share, so each is pinned here for every op it governs.
The C++ engine's builders, verifier and lowerings, with the same texts, are
checked by test_unsigned_int_ops.cpp.
"""

from __future__ import annotations

import re
import shutil
from typing import NamedTuple

import numpy as np
import pytest
from rocke.core.ir import (
    _SAME_TYPE_BINARY_OPS,
    F16,
    F32,
    I1,
    I8,
    I16,
    I32,
    I64,
    U8,
    IRBuilder,
    PtrType,
    Type,
    VectorType,
    _operand_types_error,
)
from rocke.core.verify import verify
from rocke.tests.engines import (
    ENGINES,
    assert_engine_rejects,
    assert_engine_rejects_hip,
    lower_hip_with_engine,
)
from rocke.tests.gpu import current_device_arch, launch_with_buffers

_LANES = 64


class LshrCase(NamedTuple):
    """One shift under test: ``vector_lanes`` is ``None`` for a scalar
    ``arith.lshr``, else the lane count of a ``vector.lshr``."""

    dtype: Type
    vector_lanes: int | None
    signed: str  # the HIP type of the element
    unsigned: str  # the unsigned C type of the element's width
    numpy_dtype: type[np.signedinteger]

    @property
    def elements_per_thread(self) -> int:
        return self.vector_lanes or 1

    def shift_line(self, lane: int) -> str:
        """The pattern of the HIP line that shifts ``lane`` (0 for a scalar)."""
        signed, unsigned = self.signed, self.unsigned
        if self.vector_lanes is None:
            return rf"{signed} \w+ = \({signed}\)\(\({unsigned}\)\w+ >> \w+\);"
        return rf"\w+\[{lane}\] = \(\({unsigned}\)\w+\[{lane}\]\) >> \w+\[{lane}\];"


CASES = (
    LshrCase(I8, None, "int8_t", "uint8_t", np.int8),
    LshrCase(I16, None, "int16_t", "uint16_t", np.int16),
    LshrCase(I32, None, "int", "uint32_t", np.int32),
    LshrCase(I64, None, "int64_t", "uint64_t", np.int64),
    LshrCase(I8, 4, "int8_t", "uint8_t", np.int8),
    LshrCase(I16, 2, "int16_t", "uint16_t", np.int16),
    LshrCase(I32, 2, "int", "uint32_t", np.int32),
)


def _id(case):
    if case.vector_lanes is None:
        return case.dtype.name
    return f"{case.vector_lanes}x{case.dtype.name}"


def _lshr_kernel(case):
    """Thread ``t`` shifts ``X`` right by ``S`` and stores the result to ``Y``:
    element ``t`` with a scalar ``arith.lshr``, or the ``vector_lanes``
    elements from ``t * vector_lanes`` on with one ``vector.lshr``."""
    b = IRBuilder(f"lshr_{_id(case)}")
    b.kernel.attrs["max_workgroup_size"] = _LANES
    if case.vector_lanes is None:
        x, s, y = (b.param(name, PtrType(case.dtype, "global")) for name in "XSY")
        thread = b.thread_id_x()
        value = b.global_load(x, thread, case.dtype)
        shift = b.global_load(s, thread, case.dtype)
        b.global_store(y, thread, b.lshr(value, shift))
        b.ret()
        return b.kernel
    x, s, y = (b.param(name, PtrType(I8, "global")) for name in "XSY")
    nbytes = case.vector_lanes * np.dtype(case.numpy_dtype).itemsize
    total = b.const_i32(_LANES * nbytes)
    offset = b.mul(b.thread_id_x(), b.const_i32(nbytes))
    zero = b.const_i32(0)
    value = b.buffer_load_vN(
        b.buffer_rsrc(x, total), offset, zero, case.dtype, case.vector_lanes
    )
    shift = b.buffer_load_vN(
        b.buffer_rsrc(s, total), offset, zero, case.dtype, case.vector_lanes
    )
    b.buffer_store_vN(
        b.buffer_rsrc(y, total), offset, zero, b.vector_lshr(value, shift)
    )
    b.ret()
    return b.kernel


@pytest.mark.parametrize("engine", ENGINES)
def test_hip_shifts_through_the_unsigned_type_of_the_width(engine):
    for case in CASES:
        hip = lower_hip_with_engine(_lshr_kernel(case), "gfx942", engine=engine)
        shifts = [line.strip() for line in hip.splitlines() if " >> " in line]
        assert len(shifts) == case.elements_per_thread, (_id(case), shifts)
        for lane, line in enumerate(shifts):
            assert re.fullmatch(case.shift_line(lane), line), (_id(case), line)


# Every op whose element type must be an integer or i1: its dotted name and the
# IRBuilder method that builds it (``lnot`` from one operand, the others from
# two).
_INTEGER_ELEMENT_BUILDERS = {
    "arith.and": "land",
    "arith.or": "lor",
    "arith.xor": "xor",
    "arith.not": "lnot",
    "arith.shl": "shl",
    "arith.lshr": "lshr",
    "vector.and": "vector_and",
    "vector.or": "vector_or",
    "vector.shl": "vector_shl",
    "vector.lshr": "vector_lshr",
}
_UNARY_OPS = frozenset({"arith.not"})


def _operand_type(op_name, element):
    """An ``element`` scalar for an ``arith`` op, two ``element`` lanes for a
    ``vector`` op."""
    return element if op_name.startswith("arith.") else VectorType(element, 2)


def _build_integer_element_op(b, op_name, value):
    """``op_name`` built by its IRBuilder method on ``value`` (as both operands
    of a binary op)."""
    operands = [value] if op_name in _UNARY_OPS else [value, value]
    return getattr(b, _INTEGER_ELEMENT_BUILDERS[op_name])(*operands)


def _kernel_past_the_builder(op_name, *operand_types):
    """One ``op_name`` of one parameter per type in ``operand_types``, of the
    first one's type, written past IRBuilder (which refuses a non-integer
    operand of a bitwise op or shift, and operands of different types), as
    hand-assembled or deserialized IR can reach the verifier and the
    lowerings."""
    b = IRBuilder(op_name.replace(".", "_") + "_past_the_builder")
    operands = [
        b.param(f"operand{index}", operand_type)
        for index, operand_type in enumerate(operand_types)
    ]
    b._op(op_name, operands, [operand_types[0]])
    b.ret()
    return b.kernel


def _f32_kernel(op_name):
    """One ``op_name`` of f32 values: an f32 scalar, or two f32 lanes."""
    f32_type = _operand_type(op_name, F32)
    arity = 1 if op_name in _UNARY_OPS else 2
    return _kernel_past_the_builder(op_name, *[f32_type] * arity)


def _integer_element_message(op_name):
    return f"{op_name} failed - element type='f32', expected an integer or i1 type"


@pytest.mark.parametrize("op_name", sorted(_INTEGER_ELEMENT_BUILDERS))
def test_builder_refuses_a_bitwise_op_or_shift_of_a_non_integer(op_name):
    """LLVM's bitwise ops and shifts have no float form (``not`` lowers to an
    ``xor``): IRBuilder refuses an f32 operand, scalar or lane by lane, with a
    ``TypeError`` naming the op and the element type, instead of building IR
    the compiler rejects."""
    b = IRBuilder("integer_op_f32")
    value = b.param("value", _operand_type(op_name, F32))
    with pytest.raises(TypeError) as refusal:
        _build_integer_element_op(b, op_name, value)
    assert str(refusal.value) == _integer_element_message(op_name)


@pytest.mark.parametrize("op_name", sorted(_INTEGER_ELEMENT_BUILDERS))
def test_builder_accepts_an_i1_bitwise_op_or_shift(op_name):
    """``i1`` is an LLVM integer, so a bitwise op or shift of it, scalar or
    lane by lane, is built (and verifies): ``land`` / ``lor`` / ``lnot`` of i1
    are the predicate conjunction, disjunction and negation."""
    b = IRBuilder("integer_op_i1")
    flag = b.param("flag", _operand_type(op_name, I1))
    _build_integer_element_op(b, op_name, flag)
    b.ret()
    assert verify(b.kernel) == []


@pytest.mark.parametrize("op_name", sorted(_INTEGER_ELEMENT_BUILDERS))
def test_verifier_refuses_a_bitwise_op_or_shift_of_a_non_integer(op_name):
    """IR written past the builder still meets the rule: the verifier reports
    an f32 bitwise op or shift with the builder's text."""
    messages = [diagnostic.message for diagnostic in verify(_f32_kernel(op_name))]
    assert messages == [_integer_element_message(op_name)]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("op_name", sorted(_INTEGER_ELEMENT_BUILDERS))
def test_lowerings_refuse_a_bitwise_op_or_shift_of_a_non_integer(op_name, engine):
    """IR written past the builder reaches the lowerings without the verifier:
    each engine's LLVM and HIP lowerings refuse an f32 bitwise op or shift,
    scalar or lane by lane, with the builder's ``TypeError`` and text, instead
    of emitting IR or source the compiler rejects."""
    for assert_rejects in (assert_engine_rejects, assert_engine_rejects_hip):
        assert_rejects(
            _f32_kernel(op_name),
            "gfx942",
            engine=engine,
            error_type=TypeError,
            message=_integer_element_message(op_name),
        )


# Every same-type binary op (_SAME_TYPE_BINARY_OPS), the IRBuilder method that
# builds it, and two operand types that differ: scalars for an ``arith`` op,
# vectors (of another element or lane count) for a ``vector`` op.
_MIXED_TYPE_OPERANDS = {
    "arith.add": ("add", I32, I64),
    "arith.sub": ("sub", I64, I32),
    "arith.mul": ("mul", I64, I32),
    "arith.div": ("div", I32, I16),
    "arith.mod": ("mod", I32, I64),
    "arith.fadd": ("fadd", F32, F16),
    "arith.fsub": ("fsub", F16, F32),
    "arith.fmul": ("fmul", F32, F16),
    "arith.fdiv": ("fdiv", F32, F16),
    "arith.fmax": ("fmax", F16, F32),
    "arith.fmin": ("fmin", F32, F16),
    "arith.and": ("land", I32, I16),
    "arith.or": ("lor", I1, I32),
    "arith.xor": ("xor", I32, I64),
    "arith.shl": ("shl", I32, I64),
    "arith.lshr": ("lshr", I64, I32),
    "arith.smax": ("smax", I32, I64),
    "arith.smin": ("smin", I64, I32),
    "arith.umax": ("umax", U8, I32),
    "arith.umin": ("umin", U8, I32),
    "vector.add": ("vector_add", VectorType(I32, 2), VectorType(I32, 4)),
    "vector.sub": ("vector_sub", VectorType(I32, 2), VectorType(I16, 2)),
    "vector.mul": ("vector_mul", VectorType(F32, 2), VectorType(F16, 2)),
    "vector.and": ("vector_and", VectorType(I32, 2), VectorType(I32, 4)),
    "vector.or": ("vector_or", VectorType(I32, 2), VectorType(I16, 2)),
    "vector.shl": ("vector_shl", VectorType(I32, 2), VectorType(I16, 2)),
    "vector.lshr": ("vector_lshr", VectorType(I32, 2), VectorType(I16, 2)),
    "vector.smax": ("vector_smax", VectorType(I32, 2), VectorType(I32, 4)),
    "vector.smin": ("vector_smin", VectorType(I16, 2), VectorType(I32, 2)),
    "vector.max": ("vector_max", VectorType(F32, 2), VectorType(F32, 4)),
}


def _operand_types_message(op_name):
    _, lhs_type, rhs_type = _MIXED_TYPE_OPERANDS[op_name]
    return (
        f"{op_name} failed - second operand type='{rhs_type.name}', expected the "
        f"first operand's type '{lhs_type.name}'"
    )


def _mixed_type_kernel(op_name):
    """One ``op_name`` of its two differing operand types, written past
    IRBuilder."""
    _, lhs_type, rhs_type = _MIXED_TYPE_OPERANDS[op_name]
    return _kernel_past_the_builder(op_name, lhs_type, rhs_type)


def test_the_mixed_type_cases_cover_every_same_type_binary_op():
    """The cases below cover the whole operand rule: one per op it governs."""
    assert set(_MIXED_TYPE_OPERANDS) == _SAME_TYPE_BINARY_OPS


@pytest.mark.parametrize("op_name", sorted(_MIXED_TYPE_OPERANDS))
def test_builder_refuses_operands_of_different_types(op_name):
    """The lowerings emit both operands at the first one's type (``lshr i64``,
    ``add i32`` ...), so IRBuilder refuses a second operand of another type,
    scalar or lane by lane, with a ``TypeError`` naming both types, and adds no
    op."""
    builder, lhs_type, rhs_type = _MIXED_TYPE_OPERANDS[op_name]
    b = IRBuilder("mixed_operand_types")
    lhs = b.param("lhs", lhs_type)
    rhs = b.param("rhs", rhs_type)
    with pytest.raises(TypeError) as refusal:
        getattr(b, builder)(lhs, rhs)
    assert str(refusal.value) == _operand_types_message(op_name)
    assert b.kernel.body.ops == []


@pytest.mark.parametrize("untyped_index", [0, 1])
def test_builder_refuses_an_operand_with_no_type(untyped_index):
    """A value built by hand can lack a type: the operand rule refuses it as a
    ``TypeError`` naming the operand, before it reads any type's name."""
    b = IRBuilder("untyped_operand")
    operands = [b.param("typed", I32), b.param("untyped", None)]
    if untyped_index == 0:
        operands.reverse()
    with pytest.raises(TypeError) as refusal:
        b.add(*operands)
    assert str(refusal.value) == (
        f"arith.add operand {untyped_index} failed - type=None, expected an IR type"
    )


@pytest.mark.parametrize("op_name", sorted(_MIXED_TYPE_OPERANDS))
def test_verifier_refuses_operands_of_different_types(op_name):
    """The verifier reports a binary op whose second operand's type differs
    from the first's, scalar or lane by lane, with the builder's text."""
    messages = [
        diagnostic.message for diagnostic in verify(_mixed_type_kernel(op_name))
    ]
    assert messages == [_operand_types_message(op_name)]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("op_name", sorted(_MIXED_TYPE_OPERANDS))
def test_lowerings_refuse_operands_of_different_types(op_name, engine):
    """IR written past the builder reaches the lowerings without the verifier:
    each engine's LLVM and HIP lowerings refuse a binary op whose operands
    differ in type with the builder's ``TypeError`` and text, instead of
    emitting both at the first one's type."""
    for assert_rejects in (assert_engine_rejects, assert_engine_rejects_hip):
        assert_rejects(
            _mixed_type_kernel(op_name),
            "gfx942",
            engine=engine,
            error_type=TypeError,
            message=_operand_types_message(op_name),
        )


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda b, scalar: b.vector_and(scalar, scalar),
            "vector.and failed - first operand type='i32', expected a vector type",
        ),
        (
            lambda b, scalar: b.vector_cmp("lt", scalar, scalar),
            "vector.cmp failed - first operand type='i32', expected a vector type",
        ),
        (
            lambda b, scalar: b.vector_fma(scalar, scalar, scalar),
            "vector.fma failed - first operand type='i32', expected a vector type",
        ),
    ],
    ids=["vector_and", "vector_cmp", "vector_fma"],
)
def test_vector_builders_refuse_a_scalar_first_operand(build, message):
    """A lane-wise op takes vectors: its builder refuses a scalar first
    operand by its type, a ``TypeError``, and adds no op."""
    b = IRBuilder("vector_op_of_a_scalar")
    scalar = b.param("scalar", I32)
    with pytest.raises(TypeError) as refusal:
        build(b, scalar)
    assert str(refusal.value) == message
    assert b.kernel.body.ops == []


# A lane-wise FMA of ``vec<f32x2>`` operands but one, of another type, named by
# its ordinal; its operand types, and the operand rule's text for it.
_VECTOR_FMA_OTHER_OPERANDS = [
    (VectorType(F16, 2), "second"),
    (VectorType(F32, 4), "third"),
]
_VECTOR_FMA_IDS = [ordinal for _, ordinal in _VECTOR_FMA_OTHER_OPERANDS]


def _vector_fma_operand_types(other_type, ordinal):
    first_type = VectorType(F32, 2)
    if ordinal == "second":
        return (first_type, other_type, first_type)
    return (first_type, first_type, other_type)


def _vector_fma_operand_types_message(other_type, ordinal):
    return (
        f"vector.fma failed - {ordinal} operand type='{other_type.name}', "
        "expected the first operand's type 'vec<f32x2>'"
    )


@pytest.mark.parametrize(
    ("other_type", "ordinal"), _VECTOR_FMA_OTHER_OPERANDS, ids=_VECTOR_FMA_IDS
)
def test_vector_fma_refuses_operands_of_more_than_one_type(other_type, ordinal):
    """The lane-wise FMA takes three vectors of one type: its builder refuses
    the first operand that differs, by its ordinal, with the operand rule's
    ``TypeError`` and text, and adds no op."""
    b = IRBuilder("vector_fma_mixed")
    operands = [
        b.param(f"operand{index}", operand_type)
        for index, operand_type in enumerate(
            _vector_fma_operand_types(other_type, ordinal)
        )
    ]
    with pytest.raises(TypeError) as refusal:
        b.vector_fma(*operands)
    assert str(refusal.value) == _vector_fma_operand_types_message(other_type, ordinal)
    assert b.kernel.body.ops == []


@pytest.mark.parametrize(
    ("other_type", "ordinal"), _VECTOR_FMA_OTHER_OPERANDS, ids=_VECTOR_FMA_IDS
)
def test_verifier_refuses_a_vector_fma_of_more_than_one_type(other_type, ordinal):
    """The verifier reports a lane-wise FMA written past the builder whose
    operands differ in type with the builder's text: an f32 and an f16 vector
    would lower to an ``llvm.fmuladd.v2f32`` of a ``<2 x half>``."""
    kernel = _kernel_past_the_builder(
        "vector.fma", *_vector_fma_operand_types(other_type, ordinal)
    )
    messages = [diagnostic.message for diagnostic in verify(kernel)]
    assert messages == [_vector_fma_operand_types_message(other_type, ordinal)]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    ("other_type", "ordinal"), _VECTOR_FMA_OTHER_OPERANDS, ids=_VECTOR_FMA_IDS
)
def test_lowerings_refuse_a_vector_fma_of_more_than_one_type(
    other_type, ordinal, engine
):
    """IR written past the builder reaches the lowerings without the verifier:
    each engine's LLVM and HIP lowerings refuse a lane-wise FMA whose operands
    differ in type with the builder's ``TypeError`` and text, instead of
    emitting every operand at the first one's type."""
    kernel = _kernel_past_the_builder(
        "vector.fma", *_vector_fma_operand_types(other_type, ordinal)
    )
    for assert_rejects in (assert_engine_rejects, assert_engine_rejects_hip):
        assert_rejects(
            kernel,
            "gfx942",
            engine=engine,
            error_type=TypeError,
            message=_vector_fma_operand_types_message(other_type, ordinal),
        )


_VECTOR_FMA_ARITY_MESSAGE = (
    "vector.fma failed - operands / results=2 / 1, expected 3 / 1"
)


def test_verifier_refuses_a_vector_fma_without_three_operands():
    """The verifier reports the arity rule's text, and nothing else, for a
    lane-wise FMA with two operands. The ctest checks the C++ verifier."""
    kernel = _kernel_past_the_builder("vector.fma", *[VectorType(F32, 2)] * 2)
    messages = [diagnostic.message for diagnostic in verify(kernel)]
    assert messages == [_VECTOR_FMA_ARITY_MESSAGE]


@pytest.mark.parametrize("engine", ENGINES)
def test_lowerings_refuse_a_vector_fma_without_three_operands(engine):
    """Each engine's LLVM and HIP lowerings check a lane-wise FMA's operand
    count before they read its third operand, and refuse two operands with the
    arity rule's ``ValueError`` and text."""
    kernel = _kernel_past_the_builder("vector.fma", *[VectorType(F32, 2)] * 2)
    for assert_rejects in (assert_engine_rejects, assert_engine_rejects_hip):
        assert_rejects(
            kernel,
            "gfx942",
            engine=engine,
            error_type=ValueError,
            message=_VECTOR_FMA_ARITY_MESSAGE,
        )


def test_the_operand_rule_refuses_more_operands_than_it_names():
    """The operand rule names operands by ordinal, so it refuses more operands
    than it names, before it reads any type. The ctest checks the C++ twin."""
    assert _operand_types_error("vector.fma", [F32] * 4) == (
        "vector.fma operand rule failed - operands=4, expected at most 3"
    )


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    ("operand_type", "negation"), [(I1, "!"), (I32, "~")], ids=["i1", "i32"]
)
def test_hip_not_is_logical_on_i1_and_bitwise_otherwise(operand_type, negation, engine):
    """An i1 is a C++ ``bool`` in HIP, where ``~`` promotes to int (``~true``
    is -2, still true): each engine's HIP lowering negates an i1 with ``!``
    and keeps ``~`` for the other integers, as LLVM's ``xor`` with all ones
    does."""
    b = IRBuilder("hip_not")
    value = b.param("value", operand_type)
    negated = b.lnot(value)
    b.ret()
    hip = lower_hip_with_engine(b.kernel, "gfx942", engine=engine)
    result = negated.name.lstrip("%")
    hip_type = "bool" if operand_type == I1 else "int"
    assert f"{hip_type} {result} = {negation}value;" in hip


def _compile_via_hipcc(kernel, arch):
    """``kernel`` compiled from its HIP source by hipcc; skip without hipcc."""
    if shutil.which("hipcc") is None:
        pytest.skip("hipcc not found")
    from rocke.helpers.compile import compile_kernel_via_hipcc

    return compile_kernel_via_hipcc(kernel, arch=arch)


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_hip_source_compiles(arch):
    for case in CASES:
        assert _compile_via_hipcc(_lshr_kernel(case), arch).hsaco, _id(case)


@pytest.mark.gpu
@pytest.mark.parametrize("route", ["llvm", "hipcc"])
def test_gpu_matches_a_logical_shift(route):
    arch = current_device_arch()
    from rocke.helpers.compile import compile_kernel
    from rocke.helpers.spec import sig_param

    rng = np.random.default_rng(11)
    for case in CASES:
        kernel = _lshr_kernel(case)
        if route == "llvm":
            artifact = compile_kernel(kernel, arch=arch)
        else:
            artifact = _compile_via_hipcc(kernel, arch)
        np_dtype = case.numpy_dtype
        bits = np.dtype(np_dtype).itemsize * 8
        unsigned_dtype = np.dtype(f"u{bits // 8}")
        count = _LANES * case.elements_per_thread
        raw = rng.integers(0, 256, count * bits // 8, dtype=np.uint8)
        x = raw.view(np_dtype).copy()
        x.view(unsigned_dtype)[0] = 1 << (bits - 1)  # the top bit alone
        assert (x < 0).any() and (x > 0).any()
        s = rng.integers(0, bits, x.size).astype(np_dtype)
        y = np.zeros_like(x)
        signature = [sig_param(p.name, p.type.pointee.name) for p in kernel.params]
        launch_with_buffers(
            artifact,
            signature,
            {"X": x, "S": s, "Y": y},
            grid=(1, 1, 1),
            block=(_LANES, 1, 1),
        )
        expected = (x.view(unsigned_dtype) >> s.astype(unsigned_dtype)).view(np_dtype)
        np.testing.assert_array_equal(y, expected, err_msg=_id(case))
