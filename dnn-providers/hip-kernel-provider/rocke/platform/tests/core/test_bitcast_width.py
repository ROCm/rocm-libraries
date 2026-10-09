# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A bitcast keeps every bit, so ``bitcast`` and ``vec_bitcast`` refuse a width change.

``type_bits`` is the one place that answers how many bits a value of an IR type
occupies. Equal-width casts lower to one LLVM ``bitcast`` exactly as before; a cast
that changes the width fails when the kernel is built instead of inside clang.
"""

import re

import pytest
from rocke.core import ir
from rocke.core.dtypes import dtype_info
from rocke.core.ir import (
    F16,
    F32,
    I32,
    I64,
    IRBuilder,
    PtrType,
    SmemType,
    VectorType,
    type_bits,
)
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.tests.ir_validity import assert_target_accepts


def _lower(b: IRBuilder) -> str:
    return _lower_kernel_to_llvm_python(b.kernel, arch="gfx90a")


def _scalar_types() -> list:
    """Every scalar IR type constant the module defines, so later types are covered."""
    return [t for t in vars(ir).values() if type(t) is ir.Type]


@pytest.mark.parametrize("t", _scalar_types(), ids=lambda t: t.name)
def test_a_scalar_occupies_its_format_width(t):
    assert type_bits(t) == dtype_info(t.name).encoded_bits


def test_a_vector_occupies_count_times_its_element_width():
    assert type_bits(VectorType(F16, 8)) == 128
    assert type_bits(VectorType(I32, 4)) == 128
    assert type_bits(VectorType(ir.FP4E2M1, 32)) == 128
    assert type_bits(VectorType(ir.FP6E2M3, 32)) == 192


@pytest.mark.parametrize(
    ("space", "bits"), [("global", 64), ("constant", 64), ("lds", 32)]
)
def test_a_pointer_occupies_its_address_space_width(space, bits):
    assert type_bits(PtrType(F32, space)) == bits


def test_an_lds_allocation_is_not_a_value():
    t = SmemType(F32, (16, 16))
    message = f"{t.name} is an LDS allocation, not a value with a bit width"
    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        type_bits(t)


def test_a_bitcast_refuses_pointers():
    b = IRBuilder("pointer_bitcast")
    p = b.param("P", PtrType(F32, "global"))
    message = "bitcast of ptr<f32,global> to i64: a bitcast does not convert pointers"
    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        b.bitcast(p, I64)


def test_equal_width_casts_lower_to_one_bitcast_each():
    b = IRBuilder("equal_width_bitcasts")
    p = b.param("P", PtrType(I32, "global"))
    q = b.param("Q", PtrType(F32, "global"))
    tid = b.thread_id_x()
    words = b.global_load_vN(p, tid, I32, 4)
    halves = b.vec_bitcast(words, VectorType(F16, 8))
    b.global_store_vN(p, tid, b.vec_bitcast(halves, VectorType(I32, 4)), 4)
    single = b.vec_pack([b.global_load(q, tid, F32, align=4)], F32)
    b.global_store(q, tid, b.bitcast(single, F32), align=4)
    text = _lower(b)
    assert re.search(r"bitcast <4 x i32> %\S+ to <8 x half>", text)
    assert re.search(r"bitcast <8 x half> %\S+ to <4 x i32>", text)
    assert re.search(r"bitcast <1 x float> %\S+ to float", text)
    assert_target_accepts(text, "gfx90a")


def test_a_vector_bitcast_that_changes_the_width_raises():
    b = IRBuilder("widening_vec_bitcast")
    halves = b.zero_vec(F16, 8)
    message = (
        "vec_bitcast of vec<f16x8> (128 bits) to vec<i32x8> (256 bits) changes the bit "
        "width -- a bitcast keeps every bit; slice or concatenate the vector first"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        b.vec_bitcast(halves, VectorType(I32, 8))


def test_a_scalar_bitcast_that_changes_the_width_raises():
    b = IRBuilder("widening_bitcast")
    message = (
        "bitcast of f32 (32 bits) to i64 (64 bits) changes the bit width -- a bitcast "
        "keeps every bit; convert the value to change its width"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        b.bitcast(b.const_f32(1.0), I64)


def test_bitcast_then_slice_then_bitcast_back_links():
    """Reading and replacing one f16 inside raw 32-bit words is done with core ops."""
    b = IRBuilder("bitcast_and_slice")
    p = b.param("P", PtrType(I32, "global"))
    tid = b.thread_id_x()
    halves = b.vec_bitcast(b.global_load_vN(p, tid, I32, 4), VectorType(F16, 8))
    third = b.vec_extract(halves, 3)
    replaced = b.vec_insert(halves, third, 0)
    b.global_store_vN(p, tid, b.vec_bitcast(replaced, VectorType(I32, 4)), 4)
    text = _lower(b)
    assert "extractelement <8 x half>" in text
    assert "insertelement <8 x half>" in text
    assert_target_accepts(text, "gfx90a")
