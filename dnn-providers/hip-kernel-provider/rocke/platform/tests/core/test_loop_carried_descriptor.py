# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A buffer descriptor cannot be a loop-carried value.

``tile.buffer_rsrc`` gives a ``<4 x i32>`` in the IR, but the lowered descriptor
is an opaque ``ptr addrspace(8)`` (HIP ``rsrc_t``). An ``scf.for`` iter_arg
becomes a phi (LLVM) or an ``i32x4`` variable (HIP), and neither can hold one,
so the compiler refuses the module with an error that does not point at the
loop. Carrying a descriptor is never needed: no op edits an existing one, so a
loop either uses the descriptor built before it and advances the offset, or
builds a new one in its body.

These tests check that the builder, the verifier and every lowering of
``scf.for`` (both engines, LLVM and HIP, the loop and the unrolled form) refuse
a descriptor as an iter_arg's initial value and as a yielded value with one
text, and that the two supported forms still lower identically in both engines
and compile. The refusals reach the verifier and the lowerings through
serialized IR, which does not pass the builder.
"""

from __future__ import annotations

import os
import re
import shutil

import pytest
from rocke.core.ir import F16, I32, IRBuilder, PtrType, VectorType
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python, lower_kernel_to_llvm
from rocke.core.verify import verify
from rocke.tests.ir_validity import assert_target_accepts

_ARCH = "gfx90a"


def _message(index: int) -> str:
    return (
        f"scf.for iter_arg {index} is a buffer descriptor, which a loop cannot "
        "carry; build the descriptor before the loop and advance the offset, or "
        "rebuild it inside the loop body"
    )


def _engine():
    """The C++ engine binding; a strict both-engine run fails without it."""
    try:
        import rocke_engine
    except ImportError as exc:
        if os.environ.get("ROCKE_CPP_STRICT") == "1":
            raise
        pytest.skip(f"rocke_engine extension not built: {exc}")
    return rocke_engine


def _bounds(b: IRBuilder):
    return b.const_i32(0), b.const_i32(2), b.const_i32(1)


def _vector_carry_loop(*, unroll: bool):
    """A loop carrying a plain ``<4 x i32>`` (not a descriptor), with a
    descriptor built before it. Tests swap the descriptor into the loop's init
    operand or its yield, the way serialized IR can arrive without the
    builder."""
    b = IRBuilder("descriptor_carry")
    x = b.param("X", PtrType(F16, "global"))
    descriptor = b.buffer_rsrc(x, b.const_i32(64))
    carry = b.vector_splat(b.const_i32(0), 4)
    loop = b.scf_for_iter(*_bounds(b), [("carry", carry)], unroll=unroll)
    with loop as (_, (value,)):
        b.scf_yield(value)
    b.ret()
    return b.kernel, loop.op, descriptor


def _carrying_kernel(where: str, *, unroll: bool = False):
    """Serialized-and-parsed IR whose loop carries a descriptor ``where``
    ("init" or "yield")."""
    kernel, loop_op, descriptor = _vector_carry_loop(unroll=unroll)
    assert descriptor.type == VectorType(I32, 4)
    if where == "init":
        loop_op.operands[3] = descriptor
    else:
        loop_op.regions[0].ops[-1].operands[0] = descriptor
    return parse(serialize(kernel))


# ---------------------------------------------------------------- builder


def test_builder_refuses_a_descriptor_as_an_iter_arg_init():
    b = IRBuilder("descriptor_init")
    x = b.param("X", PtrType(F16, "global"))
    count = b.const_i32(0)
    descriptor = b.buffer_rsrc(x, b.const_i32(64))
    with pytest.raises(ValueError, match=re.escape(_message(1))):
        b.scf_for_iter(*_bounds(b), [("count", count), ("r", descriptor)])


def test_builder_refuses_a_yielded_descriptor():
    b = IRBuilder("descriptor_yield")
    x = b.param("X", PtrType(F16, "global"))
    loop = b.scf_for_iter(*_bounds(b), [("carry", b.vector_splat(b.const_i32(0), 4))])
    with loop, pytest.raises(ValueError, match=re.escape(_message(0))):
        b.scf_yield(b.buffer_rsrc(x, b.const_i32(32)))


# --------------------------------------------------------------- verifier


@pytest.mark.parametrize("where", ["init", "yield"])
def test_verifier_refuses_a_carried_descriptor(where):
    messages = [d.message for d in verify(_carrying_kernel(where))]
    assert _message(0) in messages


# -------------------------------------------------------------- lowerings


@pytest.mark.parametrize("unroll", [False, True], ids=["loop", "unrolled"])
@pytest.mark.parametrize("where", ["init", "yield"])
def test_python_llvm_lowering_refuses_a_carried_descriptor(where, unroll):
    kernel = _carrying_kernel(where, unroll=unroll)
    with pytest.raises(ValueError, match=re.escape(_message(0))):
        _lower_kernel_to_llvm_python(kernel, arch=_ARCH, llvm_flavor="llvm20")


@pytest.mark.parametrize("unroll", [False, True], ids=["loop", "unrolled"])
@pytest.mark.parametrize("where", ["init", "yield"])
def test_cpp_llvm_lowering_refuses_a_carried_descriptor(where, unroll):
    engine = _engine()
    text = serialize(_carrying_kernel(where, unroll=unroll))
    with pytest.raises(RuntimeError, match=re.escape(_message(0))):
        engine.lower_serialized_ir(text, arch=_ARCH, flavor="llvm20")


@pytest.mark.parametrize("where", ["init", "yield"])
def test_python_hip_lowering_refuses_a_carried_descriptor(where):
    with pytest.raises(ValueError, match=re.escape(_message(0))):
        lower_kernel_to_hip(_carrying_kernel(where), arch=_ARCH)


@pytest.mark.parametrize("where", ["init", "yield"])
def test_cpp_hip_lowering_refuses_a_carried_descriptor(where):
    engine = _engine()
    text = serialize(_carrying_kernel(where))
    with pytest.raises(RuntimeError, match=re.escape(_message(0))):
        engine.lower_serialized_ir_hip(text, arch=_ARCH)


# ------------------------------------------------- the forms that do work


def _built_before_loop(*, unroll: bool):
    """Copy two dwords per lane through descriptors built before the loop; the
    loop carries the byte offset and advances it."""
    b = IRBuilder("descriptor_before_loop")
    x = b.param("X", PtrType(F16, "global"))
    y = b.param("Y", PtrType(F16, "global"))
    src = b.buffer_rsrc(x, b.const_i32(512))
    dst = b.buffer_rsrc(y, b.const_i32(512))
    soffset = b.const_i32(0)
    start = b.mul(b.thread_id_x(), b.const_i32(4))
    loop = b.scf_for_iter(*_bounds(b), [("offset", start)], unroll=unroll)
    with loop as (_, (offset,)):
        value = b.buffer_load_vN_f16(src, offset, soffset, 1)
        b.buffer_store_vN_f16(dst, offset, soffset, value, 1)
        b.scf_yield(b.add(offset, b.const_i32(256)))
    b.ret()
    return b.kernel


def _rebuilt_in_body(*, unroll: bool):
    """The same copy, building both descriptors again in every iteration."""
    b = IRBuilder("descriptor_in_body")
    x = b.param("X", PtrType(F16, "global"))
    y = b.param("Y", PtrType(F16, "global"))
    soffset = b.const_i32(0)
    lane = b.mul(b.thread_id_x(), b.const_i32(4))
    loop = b.scf_for(*_bounds(b), iv_name="i")
    loop.op.attrs["unroll"] = unroll
    with loop as i:
        offset = b.add(lane, b.mul(i, b.const_i32(256)))
        value = b.buffer_load_vN_f16(
            b.buffer_rsrc(x, b.const_i32(512)), offset, soffset, 1
        )
        b.buffer_store_vN_f16(
            b.buffer_rsrc(y, b.const_i32(512)), offset, soffset, value, 1
        )
    b.ret()
    return b.kernel


_SUPPORTED = {
    "built_before_loop": _built_before_loop,
    "rebuilt_in_body": _rebuilt_in_body,
}


@pytest.mark.parametrize("unroll", [False, True], ids=["loop", "unrolled"])
@pytest.mark.parametrize("form", sorted(_SUPPORTED))
def test_supported_forms_lower_identically_and_compile(form, unroll):
    kernel = _SUPPORTED[form](unroll=unroll)
    assert verify(kernel) == []
    python_llvm = _lower_kernel_to_llvm_python(kernel, arch=_ARCH, llvm_flavor="llvm20")
    cpp_llvm = _engine().lower_serialized_ir(
        serialize(kernel), arch=_ARCH, flavor="llvm20"
    )
    assert cpp_llvm == python_llvm
    assert_target_accepts(lower_kernel_to_llvm(kernel, arch=_ARCH), _ARCH)


@pytest.mark.parametrize("form", sorted(_SUPPORTED))
def test_supported_forms_lower_identically_to_hip_and_compile(form):
    kernel = _SUPPORTED[form](unroll=False)
    hip = lower_kernel_to_hip(kernel, arch=_ARCH)
    assert _engine().lower_serialized_ir_hip(serialize(kernel), arch=_ARCH) == hip
    if shutil.which("hipcc") is None:
        pytest.skip("hipcc not found on PATH")
    from rocke.helpers.compile import compile_kernel_via_hipcc

    assert compile_kernel_via_hipcc(kernel, arch=_ARCH).hsaco
