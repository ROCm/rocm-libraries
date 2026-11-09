# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""LLVM-text checks for ``CoalescedTileLoader.load_async_to_lds`` (gfx1250).

The async global->LDS copy has no buffer-resource range check, so the loader
makes the one the resource would make itself -- byte offset below
``num_bytes`` -- and zero-fills a failing chunk with an LDS store.

* The compare must be unsigned, as the resource's is: a byte offset at or past
  2 GiB is negative as an i32, and a signed compare would read it as in range
  and issue an out-of-bounds copy instead of zero-filling.
* The zero-fill must have the LDS tile's element type (bf16 zeros into a bf16
  tile), as the C++ engine's mirror does.
"""

from __future__ import annotations

import re

import pytest

from rocke.core.ir import BF16, F16, I32, IRBuilder, PtrType
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers.loads import CoalescedTileLoader


def _build(dtype):
    b = IRBuilder("load_async_to_lds_check")
    src = b.param("src", PtrType(dtype, "global"), readonly=True, align=16)
    num_bytes = b.param("num_bytes", I32)
    tile = b.smem_alloc(dtype, [16, 32], name_hint="tile")
    loader = CoalescedTileLoader.from_tile(
        tile_rows=16, tile_cols=32, block_size=64, elem_dtype=dtype
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
    return b.kernel, loader.load_vec


@pytest.mark.parametrize("dtype, llvm_elem", [(F16, "half"), (BF16, "bfloat")])
def test_range_check_is_unsigned_and_zero_fill_matches_the_tile(dtype, llvm_elem):
    kernel, vec = _build(dtype)
    ll = _lower_kernel_to_llvm_python(kernel, arch="gfx1250", llvm_flavor="llvm23")
    assert "global.load.async.to.lds" in ll
    # The range check: unsigned against num_bytes, never signed.
    assert re.search(r"icmp ult i32 %[\w.]+, %num_bytes", ll), ll
    assert not re.search(r"icmp s(lt|le|gt|ge) i32 %[\w.]+, %num_bytes", ll), ll
    # The out-of-range chunks store zeros of the tile's element type.
    assert re.search(rf"store <{vec} x {llvm_elem}> %cz", ll), ll
    other = "bfloat" if llvm_elem == "half" else "half"
    assert f"x {other}>" not in ll
