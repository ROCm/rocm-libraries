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
verifier reports both for IR written past the builder (the C++ engine's builder
and verifier, with the same texts, are checked by test_unsigned_int_ops.cpp).
"""

from __future__ import annotations

import re
import shutil
from typing import NamedTuple

import numpy as np
import pytest
from rocke.core.ir import (
    F32,
    I1,
    I8,
    I16,
    I32,
    I64,
    IRBuilder,
    PtrType,
    Type,
    VectorType,
)
from rocke.core.verify import verify
from rocke.tests.engines import (
    ENGINES,
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


def _shift_kernel_past_the_builder(op_name, value_type, amount_type):
    """One ``op_name`` (``arith.lshr`` or ``vector.lshr``) of a ``value_type``
    parameter by an ``amount_type`` parameter, written past IRBuilder (which
    refuses both a non-integer shift and a mismatched shift amount), as
    hand-assembled or deserialized IR can reach the verifier and the lowerings."""
    b = IRBuilder(op_name.replace(".", "_") + "_past_the_builder")
    value = b.param("value", value_type)
    amount = b.param("amount", amount_type)
    b._op(op_name, [value, amount], [value_type])
    b.ret()
    return b.kernel


def _f32_shift_kernel(op_name):
    """One ``op_name`` of f32 values: an f32 scalar, or two f32 lanes."""
    f32_type = F32 if op_name == "arith.lshr" else VectorType(F32, 2)
    return _shift_kernel_past_the_builder(op_name, f32_type, f32_type)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("op_name", ["arith.lshr", "vector.lshr"])
def test_hip_refuses_a_shift_of_a_non_integer(op_name, engine):
    """An f32 shift, scalar or lane by lane, has no unsigned type of its width
    to be read through: both engines refuse it with a ``TypeError`` naming
    the op and the element type."""
    assert_engine_rejects_hip(
        _f32_shift_kernel(op_name),
        "gfx942",
        engine=engine,
        error_type=TypeError,
        message=(
            f"HIP lowering of {op_name} failed - element type='f32', expected an "
            "integer type, one of ('i1', 'i8', 'u8', 'i16', 'i32', 'i64')"
        ),
    )


@pytest.mark.parametrize("op_name", ["arith.lshr", "vector.lshr"])
def test_builder_refuses_a_shift_of_a_non_integer(op_name):
    """LLVM's ``lshr`` has no float form: IRBuilder refuses an f32 shift,
    scalar or lane by lane, with a ``TypeError`` naming the op and the
    element type, instead of building IR the compiler rejects."""
    b = IRBuilder("lshr_f32")
    if op_name == "arith.lshr":
        value = b.param("value", F32)
        shift = b.lshr
    else:
        value = b.param("value", VectorType(F32, 2))
        shift = b.vector_lshr
    with pytest.raises(TypeError) as refusal:
        shift(value, value)
    assert str(refusal.value) == (
        f"{op_name} failed - element type='f32', expected an integer type"
    )


def test_builder_refuses_a_shift_amount_of_another_width():
    """``lshr`` shifts two integers of one width: IRBuilder refuses an i64
    shifted by an i32 with a ``TypeError`` naming both types."""
    b = IRBuilder("lshr_mixed_widths")
    value = b.param("value", I64)
    amount = b.param("amount", I32)
    with pytest.raises(TypeError) as refusal:
        b.lshr(value, amount)
    assert str(refusal.value) == (
        "arith.lshr failed - shift amount type='i32', expected the shifted "
        "value's type 'i64'"
    )


def test_builder_accepts_an_i1_shift():
    """``i1`` is an LLVM integer, so a shift of it is built (and verifies)."""
    b = IRBuilder("lshr_i1")
    flag = b.param("flag", I1)
    b.lshr(flag, flag)
    b.ret()
    assert verify(b.kernel) == []


@pytest.mark.parametrize("op_name", ["arith.lshr", "vector.lshr"])
def test_verifier_refuses_a_shift_of_a_non_integer(op_name):
    """IR written past the builder still meets the rule: the verifier reports
    an f32 shift with the builder's text."""
    messages = [diagnostic.message for diagnostic in verify(_f32_shift_kernel(op_name))]
    assert messages == [
        f"{op_name} failed - element type='f32', expected an integer type"
    ]


@pytest.mark.parametrize(
    ("op_name", "value_type", "amount_type"),
    [
        ("arith.lshr", I64, I32),
        ("vector.lshr", VectorType(I32, 2), VectorType(I16, 2)),
    ],
)
def test_verifier_refuses_a_shift_amount_of_another_width(
    op_name, value_type, amount_type
):
    """The verifier reports a shift amount whose type differs from the
    shifted value's, scalar or lane by lane."""
    kernel = _shift_kernel_past_the_builder(op_name, value_type, amount_type)
    messages = [diagnostic.message for diagnostic in verify(kernel)]
    assert messages == [
        f"{op_name}: operand types differ ({value_type.name} vs {amount_type.name})"
    ]


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
