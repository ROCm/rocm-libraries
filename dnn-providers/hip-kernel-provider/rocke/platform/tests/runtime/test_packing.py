# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for ``runtime.packing``: the i32 kernarg wrap boundary (host only), and
the packed layout of a mixed signature against the argument offsets the
compiler records in the code object (COMGR and llvm-readelf, no GPU)."""

from __future__ import annotations

import struct

import pytest
from rocke.analysis import parse_kernel_arguments
from rocke.core.ir import F32, I32, I64, IRBuilder, PtrType
from rocke.helpers.compile import compile_kernel
from rocke.helpers.spec import sig_param, sig_scalar
from rocke.runtime.packing import compile_packer, pack_args
from rocke.tests.comgr import require_comgr_target
from rocke.tests.llc import code_object_notes

_SIG = [{"name": "x", "type": "i32"}]


def test_a_value_at_the_top_of_the_signed_range_packs_unchanged():
    assert pack_args(_SIG, {"x": 2**31 - 1}) == struct.pack("<i", 2**31 - 1)


def test_a_value_at_the_unsigned_wrap_boundary_packs_as_the_most_negative_i32():
    assert pack_args(_SIG, {"x": 1 << 31}) == struct.pack("<i", -(2**31))


def test_a_value_at_the_top_of_the_unsigned_range_packs_as_minus_one():
    assert pack_args(_SIG, {"x": 2**32 - 1}) == struct.pack("<i", -1)


def test_a_value_past_32_bits_raises_struct_error_not_value_error():
    with pytest.raises(struct.error):
        pack_args(_SIG, {"x": 1 << 32})


def test_compile_packer_agrees_with_pack_args_at_every_boundary():
    packer = compile_packer(_SIG)
    for v in (0, 2**31 - 1, 1 << 31, 2**32 - 1):
        assert packer({"x": v}) == pack_args(_SIG, {"x": v})
    with pytest.raises(struct.error):
        packer({"x": 1 << 32})


# A signature that needs padding: an i64 and a pointer after an odd number of
# 4-byte arguments, an f32 between two 8-byte ones.
_MIXED = [
    ("X", "ptr", 0x1111_2222_3333_4444),
    ("N", "i32", -7),
    ("Big", "i64", 0x0123_4567_89AB_CDEF),
    ("alpha", "f32", 1.5),
    ("Y", "ptr", 0x5555_6666_7777_8888),
    ("M", "i32", 123456),
    ("beta", "f32", -0.25),
    ("Count", "i64", -(2**40)),
    ("Z", "ptr", 0x0000_7FFF_0000_1000),
]
_IR_TYPES = {"i32": I32, "i64": I64, "f32": F32}
_STRUCT_FORMAT = {"ptr": "<Q", "i32": "<i", "i64": "<q", "f32": "<f"}


def _mixed_kernel():
    """Stores ``alpha`` to ``Y[tid]``; the other arguments are only declared."""
    b = IRBuilder("kernarg_layout_mixed")
    params = {
        name: b.param(
            name, PtrType(F32, "global") if kind == "ptr" else _IR_TYPES[kind]
        )
        for name, kind, _ in _MIXED
    }
    b.global_store(params["Y"], b.thread_id_x(), params["alpha"], align=4)
    b.ret()
    return b.kernel


@pytest.mark.parametrize("arch", ["gfx942", "gfx950"])
def test_packed_arguments_sit_at_the_offsets_the_compiler_records(arch):
    """Every explicit argument of the code object's metadata is the signature's
    argument, in order, with its size, and the packer put its value at the
    metadata's offset."""
    require_comgr_target(arch)
    kernel = _mixed_kernel()
    notes = code_object_notes(compile_kernel(kernel, arch=arch).hsaco)
    recorded = [
        arg
        for arg in parse_kernel_arguments(notes)[kernel.name]
        if not arg.value_kind.startswith("hidden_")
    ]

    signature = [
        sig_param(name, "f32") if kind == "ptr" else sig_scalar(name, kind)
        for name, kind, _ in _MIXED
    ]
    values = {name: value for name, _, value in _MIXED}
    packed = pack_args(signature, values)
    assert compile_packer(signature)(values) == packed
    assert [arg.name for arg in recorded] == [name for name, _, _ in _MIXED]
    for arg, (name, kind, value) in zip(recorded, _MIXED):
        assert arg.size == struct.calcsize(_STRUCT_FORMAT[kind]), name
        (unpacked,) = struct.unpack_from(_STRUCT_FORMAT[kind], packed, arg.offset)
        assert unpacked == value, f"{name} at offset {arg.offset}"
    last = recorded[-1]
    assert len(packed) == last.offset + last.size
