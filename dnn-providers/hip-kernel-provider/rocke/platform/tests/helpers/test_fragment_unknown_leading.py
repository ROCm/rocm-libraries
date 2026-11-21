# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""An ABI-absent batch extent remains unknown, never a fabricated allocation size."""

import pytest

from rocke.core.ir import F32, I32, IRBuilder, PtrType
from rocke.core.ir_serialize import serialize
from rocke.helpers.tiling import (
    lds_allocation,
    load_fragment,
    make_tensor_desc,
    make_tile_desc,
    make_window,
)


def test_unknown_leading_extent_does_not_claim_static_allocation_size():
    tensor = make_tensor_desc((None, 16, 8), (128, 8, 1), F32)
    assert tensor.lengths[0] is None
    with pytest.raises(ValueError, match="needs known lengths"):
        _ = tensor.element_span
    with pytest.raises(ValueError, match="needs known lengths"):
        lds_allocation(tensor)


def test_unknown_leading_extent_still_addresses_the_batch_origin():
    tensor = make_tensor_desc((None, 64, 16), (1024, 16, 1), F32)
    tile = make_tile_desc(
        shape=(64, 16), thread_dist=(64, 1), thread_tile=(1, 16), wave_size=64
    )
    b = IRBuilder("unknown_batch")
    ptr = b.param("A", PtrType(F32, "global"))
    batch = b.param("batch", I32)
    fragment = load_fragment(
        b,
        ptr,
        make_window(tensor, (batch, 0, 0)),
        tile,
        b.thread_id_x(),
        alignment_bytes=16,
    )
    assert fragment.value.type.count == 16
    assert any(
        op.name == "arith.mul" and op.operands[0] is batch for op in b.kernel.body.ops
    )
    assert not any(op.name == "tile.global_ptr_add" for op in b.kernel.body.ops)


def test_unknown_tile_covered_extent_is_refused_before_emission():
    tensor = make_tensor_desc((64, None), (16, 1), F32)
    tile = make_tile_desc(
        shape=(64, 16), thread_dist=(64, 1), thread_tile=(1, 16), wave_size=64
    )
    b = IRBuilder("unknown_tile_length")
    ptr = b.param("A", PtrType(F32, "global"))
    lane = b.thread_id_x()
    before = serialize(b.kernel)
    with pytest.raises(ValueError, match="supported on leading axes only"):
        load_fragment(b, ptr, make_window(tensor, (0, 0)), tile, lane)
    assert serialize(b.kernel) == before


def test_unknown_length_does_not_admit_negative_known_lengths():
    with pytest.raises(ValueError, match="lengths must be positive"):
        make_tensor_desc((None, -1, 16), (16, 16, 1), F32)
