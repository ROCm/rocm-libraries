# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shared complete-storage-run load policy, independent of tensor coordinates."""

import pytest

from rocke.core.ir import I8, I32, IRBuilder, PtrType
from rocke.core.ir_serialize import serialize
from rocke.helpers.tiling.transport import load_storage_chunks


@pytest.mark.parametrize(
    "element,count,byte_addressed,word_vectors,widths",
    [
        (I32, 3, True, False, [3]),
        (I32, 6, True, False, [4, 2]),
        (I32, 3, False, True, [3]),
        (I32, 6, False, True, [4, 2]),
        (I8, 20, False, False, [16, 4]),
        (I8, 36, False, False, [16, 16, 4]),
    ],
)
def test_supported_word_runs_and_byte_tail_fallbacks(
    element, count, byte_addressed, word_vectors, widths
):
    b = IRBuilder("storage_chunks")
    ptr = b.param("A", PtrType(I8, "global"))
    offset = b.const_i32(0)
    pieces = load_storage_chunks(
        b,
        ptr,
        offset,
        element,
        count,
        alignment_bytes=1,
        byte_addressed=byte_addressed,
        word_vectors=word_vectors,
    )
    assert [piece.type.count for piece in pieces] == widths
    loads = [op for op in b.kernel.body.ops if op.name.startswith("memref.global_load")]
    assert len(loads) == len(widths)
    assert all(op.attrs["align"] == 1 for op in loads)


def test_byte_word_addresses_keep_signed_offsets():
    b = IRBuilder("signed_word_address")
    ptr = b.param("A", PtrType(I8, "global"))
    offset = b.param("offset", I32)
    load_storage_chunks(b, ptr, offset, I32, 6, alignment_bytes=8, byte_addressed=True)
    pointers = [op for op in b.kernel.body.ops if op.name == "tile.global_ptr_add"]
    assert len(pointers) == 2
    assert all(op.operands[1].op.name == "arith.sext" for op in pointers)


@pytest.mark.parametrize("count", (0, -1))
def test_empty_or_negative_runs_are_refused_without_emission(count):
    b = IRBuilder("invalid_storage_run")
    ptr = b.param("A", PtrType(I8, "global"))
    offset = b.const_i32(0)
    before = serialize(b.kernel)
    with pytest.raises(ValueError, match="count must be positive"):
        load_storage_chunks(b, ptr, offset, I32, count, alignment_bytes=1)
    assert serialize(b.kernel) == before
