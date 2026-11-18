# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Global fragment accesses preserve caller-guaranteed conservative alignment."""

import pytest

from rocke.core.ir import FP4E2M1, FP6E2M3, I8, I32, IRBuilder, PtrType
from rocke.core.ir_serialize import serialize
from rocke.helpers.tiling import (
    load_fragment,
    make_fragment,
    make_tensor_desc,
    make_tile_desc,
    make_window,
    store_fragment,
)


def _setup(dtype=FP6E2M3):
    count = 16 if dtype == FP6E2M3 else 32
    tensor = make_tensor_desc((64, count), (count, 1), dtype)
    tile = make_tile_desc(
        shape=(64, count),
        thread_dist=(64, 1),
        thread_tile=(1, count),
        wave_size=64,
    )
    b = IRBuilder("fragment_alignment")
    ptr = b.param("A", PtrType(I8, "global"), align=1)
    lane = b.thread_id_x()
    return b, ptr, lane, tile, make_window(tensor, (0, 0))


@pytest.mark.parametrize("dtype", (FP4E2M1, FP6E2M3))
@pytest.mark.parametrize("alignment", (1, 2, 4, 8, 16))
def test_packed_load_and_store_cap_global_alignment(dtype, alignment):
    b, ptr, lane, tile, window = _setup(dtype)
    fragment = load_fragment(b, ptr, window, tile, lane, alignment_bytes=alignment)
    store_fragment(b, ptr, window, fragment, lane, alignment_bytes=alignment)
    accesses = [op for op in b.kernel.body.ops if op.name.startswith("memref.global_")]
    assert accesses
    assert all(0 < op.attrs["align"] <= alignment for op in accesses)


def test_byte_rebased_fp6_load_remains_unaligned_safe():
    b, ptr, lane, tile, window = _setup()
    shifted = b.global_ptr_add(ptr, b.const_i64(97))
    load_fragment(b, shifted, window, tile, lane, alignment_bytes=1)
    loads = [op for op in b.kernel.body.ops if op.name == "memref.global_load_vN"]
    assert loads
    assert all(op.result.type.elem == I32 and op.attrs["align"] == 1 for op in loads)


@pytest.mark.parametrize("verb", ("load", "store"))
@pytest.mark.parametrize(
    "alignment,error",
    [
        (0, ValueError),
        (-1, ValueError),
        (3, ValueError),
        (True, TypeError),
        (1.0, TypeError),
    ],
)
def test_invalid_alignment_is_refused_before_emission(verb, alignment, error):
    b, ptr, lane, tile, window = _setup()
    fragment = make_fragment(tile, window.tensor.dtype)
    before = serialize(b.kernel)
    with pytest.raises(error, match="alignment_bytes must be"):
        if verb == "load":
            load_fragment(b, ptr, window, tile, lane, alignment_bytes=alignment)
        else:
            store_fragment(b, ptr, window, fragment, lane, alignment_bytes=alignment)
    assert serialize(b.kernel) == before


def test_global_alignment_argument_is_refused_for_lds():
    b, _, lane, tile, window = _setup()
    lds = b.smem_alloc(I32, (64 * 3,))
    before = serialize(b.kernel)
    with pytest.raises(ValueError, match="global fragment transport only"):
        load_fragment(b, lds, window, tile, lane, alignment_bytes=4)
    assert serialize(b.kernel) == before


def test_omitted_alignment_preserves_layout_emission():
    b, ptr, lane, tile, window = _setup()
    load_fragment(b, ptr, window, tile, lane)
    loads = [op for op in b.kernel.body.ops if op.name == "memref.global_load_vN"]
    assert loads
    assert all(op.attrs["align"] == 4 for op in loads)
