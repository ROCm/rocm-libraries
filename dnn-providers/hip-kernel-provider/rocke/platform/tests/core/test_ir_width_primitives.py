# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The IR layer's width and alignment primitives, called directly.

``type_bytes``, ``element_type`` and ``lane_count`` answer how wide a value of
an IR type is and what it is made of. ``vector_access_align`` is the natural
alignment of one access of a vector, and ``smem_vector_access_align`` bounds
it by what every row start of an LDS tile guarantees. ``align_attr`` reads and
checks the ``align`` a memory op carries, and ``claimed_align`` combines it
with the natural alignment into the alignment the op claims. The lowerers size
and check accesses with them, so their answers and refusals are pinned here
apart from any kernel. The C++ twins (``rocke_i_type_bytes``,
``rocke_i_elem_of``, ``rocke_i_count_of``, ``rocke_i_vector_access_align``,
``rocke_i_smem_vector_access_align``, ``rocke_i_align_attr``,
``rocke_i_claimed_align``) are pinned in ``test_ir_types.cpp``.
"""

from __future__ import annotations

import re

import pytest
from rocke.core import ir
from rocke.core.ir import (
    F16,
    F32,
    IRBuilder,
    PtrType,
    SmemType,
    VectorType,
    align_attr,
    claimed_align,
    element_type,
    lane_count,
    smem_vector_access_align,
    type_bytes,
    vector_access_align,
)


def test_a_whole_byte_type_occupies_its_bits_over_eight_bytes():
    assert type_bytes(ir.F64) == 8
    assert type_bytes(ir.FP8E4M3) == 1
    assert type_bytes(VectorType(F16, 8)) == 16


def test_a_sub_byte_type_has_no_byte_width():
    message = (
        "type_bytes failed - type=fp6e2m3 (6 bits), expected a whole number of bytes"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        type_bytes(ir.FP6E2M3)


def test_a_vector_is_made_of_its_element_type_and_lanes():
    vector = VectorType(F16, 8)
    assert element_type(vector) == F16
    assert lane_count(vector) == 8


def test_a_scalar_is_its_own_element_and_one_lane():
    assert element_type(F32) == F32
    assert lane_count(F32) == 1


@pytest.mark.parametrize(
    ("lanes", "elem", "natural"),
    [(4, F32, 16), (16, ir.FP8E4M3, 16), (1, ir.F64, 8), (3, F32, 4), (6, F16, 2)],
)
def test_a_vector_access_is_aligned_to_its_width_or_else_its_element(
    lanes, elem, natural
):
    """A power-of-two width is the natural alignment; a 12-byte <3 x float>
    or a 12-byte <6 x half> guarantees only its element's."""
    assert vector_access_align(lanes, elem) == natural


@pytest.mark.parametrize(
    ("elem", "shape", "lanes", "claim"),
    [
        # Rows a multiple of the width: the natural alignment.
        (F32, (4, 8), 4, 16),
        # 12-element f16 rows: row 1 starts 24 bytes in, gcd(8, 12) = 4 lanes.
        (F16, (2, 12), 8, 8),
        # 24-byte fp8 rows: gcd(16, 24) = 8 one-byte lanes.
        (ir.FP8E4M3, (2, 24), 16, 8),
        # A 12-byte access already claims only its element, whatever the rows.
        (F16, (2, 9), 6, 2),
    ],
)
def test_an_lds_vector_access_claims_what_every_row_start_guarantees(
    elem, shape, lanes, claim
):
    assert smem_vector_access_align(SmemType(elem, shape), lanes, elem) == claim


def test_a_type_with_no_shape_gives_no_row_bound():
    assert smem_vector_access_align(F32, 4, F32) == vector_access_align(4, F32)


def _vector_store_op():
    """A ``memref.global_store_vN`` of four f32 lanes carrying ``align`` 8."""
    builder = IRBuilder("align_attr_probe")
    out = builder.param("out", PtrType(F32, "global"))
    zero = builder.const_i32(0)
    values = builder.vec_pack([builder.const_f32(0.0)] * 4, F32)
    builder.global_store_vN(out, zero, values, 4, align=8)
    (store,) = [op for op in builder.kernel.body.ops if op.name.endswith("store_vN")]
    return store


@pytest.mark.parametrize("align", [1, 8, 2**40])
def test_an_integer_power_of_two_alignment_is_claimed_as_given(align):
    store = _vector_store_op()
    store.attrs["align"] = align
    assert align_attr(store) == align
    assert claimed_align(store, 4, F32) == align


def test_an_absent_alignment_claims_the_natural_vector_alignment():
    store = _vector_store_op()
    del store.attrs["align"]
    assert align_attr(store) is None
    assert claimed_align(store, 4, F32) == 16
    assert claimed_align(store, 3, F32) == 4


@pytest.mark.parametrize("align", [24, 12, 0, -16])
def test_an_alignment_that_is_not_a_positive_power_of_two_is_refused(align):
    store = _vector_store_op()
    store.attrs["align"] = align
    message = (
        f"memref.global_store_vN failed - align={align}, "
        "expected a positive power of two"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        align_attr(store)


@pytest.mark.parametrize(
    ("align", "kind"), [(True, "bool"), (8.0, "float"), ("16", "str")]
)
def test_an_alignment_that_is_not_an_integer_is_refused(align, kind):
    """Never read as an integer: ``True`` is not 1 and ``8.0`` is not 8."""
    store = _vector_store_op()
    store.attrs["align"] = align
    message = f"memref.global_store_vN failed - align={kind}, expected an integer"
    with pytest.raises(TypeError, match=f"^{re.escape(message)}$"):
        align_attr(store)
