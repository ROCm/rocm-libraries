# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LLVM-text checks for ``CoalescedTileLoader.load_async_to_lds`` (gfx1250).

The async global->LDS copy has no buffer-resource range check, so the loader
makes the one the resource would make itself -- byte offset below
``num_bytes`` -- and zero-fills a failing chunk with an LDS store.

* The compare must be unsigned, as the resource's is: a byte offset at or past
  2 GiB is negative as an i32, and a signed compare would read it as in range
  and issue an out-of-bounds copy instead of zero-filling.
* The copy and the zero-fill move ``load_vec`` elements of the loader's
  element type, for every type the loader admits: bf16 zeros into a bf16 tile,
  four f32 or i32 lanes in one 16-byte copy.
"""

from __future__ import annotations

import re

import pytest

from rocke.core.ir import BF16, F16, F32, I32, IRBuilder, PtrType
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers.loads import CoalescedTileLoader


def _build(dtype, load_vec):
    b = IRBuilder("load_async_to_lds_check")
    src = b.param("src", PtrType(dtype, "global"), readonly=True, align=16)
    num_bytes = b.param("num_bytes", I32)
    tile = b.smem_alloc(dtype, [16, 32], name_hint="tile")
    loader = CoalescedTileLoader(
        tile_rows=16, tile_cols=32, block_size=64, load_vec=load_vec, elem_dtype=dtype
    )
    c32 = b.const_i32(32)

    def descriptor(b, row, col):
        return b.add(b.mul(row, c32), col), None

    loader.load_async_to_lds(
        b,
        tid=b.thread_id_x(),
        smem_dst=tile,
        descriptor=descriptor,
        ptr=src,
        num_bytes=num_bytes,
    )
    b.ret()
    return b.kernel


# (element type, its LLVM spelling, the load_vec of one 16-byte copy)
ADMITTED = [(F16, "half", 8), (BF16, "bfloat", 8), (F32, "float", 4), (I32, "i32", 4)]


@pytest.mark.parametrize("dtype, llvm_elem, vec", ADMITTED)
def test_range_check_is_unsigned_and_zero_fill_matches_the_tile(dtype, llvm_elem, vec):
    kernel = _build(dtype, vec)
    ll = _lower_kernel_to_llvm_python(kernel, arch="gfx1250", llvm_flavor="llvm23")
    assert "global.load.async.to.lds.b128" in ll
    # The range check: unsigned against num_bytes, never signed.
    assert re.search(r"icmp ult i32 %[\w.]+, %num_bytes", ll), ll
    assert not re.search(r"icmp s(lt|le|gt|ge) i32 %[\w.]+, %num_bytes", ll), ll
    # The out-of-range chunks store zeros of the tile's element type.
    assert re.search(rf"store <{vec} x {llvm_elem}> %cz", ll), ll
    for _, other, _ in ADMITTED:
        if other != llvm_elem:
            assert f"x {other}>" not in ll


@pytest.mark.parametrize(
    "loader_dtype, tile_dtype, src_dtype, message",
    [
        (
            F16,
            F32,
            F16,
            "smem_dst must hold 2-byte elements to match elem_dtype 'f16' (got f32)",
        ),
        (
            F32,
            F32,
            F16,
            "ptr must hold 4-byte elements to match elem_dtype 'f32' (got f16)",
        ),
    ],
)
def test_an_operand_of_another_width_is_refused(
    loader_dtype, tile_dtype, src_dtype, message
):
    """The slot plan and the copy count the loader's elements; the tile and the
    source are addressed in their own, so a width mismatch would scatter the
    chunks."""
    b = IRBuilder("load_async_to_lds_width")
    src = b.param("src", PtrType(src_dtype, "global"))
    num_bytes = b.param("num_bytes", I32)
    tile = b.smem_alloc(tile_dtype, [16, 32], name_hint="tile")
    loader = CoalescedTileLoader(
        tile_rows=16, tile_cols=32, block_size=64, load_vec=4, elem_dtype=loader_dtype
    )
    with pytest.raises(ValueError) as refusal:
        loader.load_async_to_lds(
            b,
            tid=b.thread_id_x(),
            smem_dst=tile,
            descriptor=lambda b_, row, col: (col, None),
            ptr=src,
            num_bytes=num_bytes,
        )
    assert str(refusal.value) == f"CoalescedTileLoader.load_async_to_lds: {message}"
