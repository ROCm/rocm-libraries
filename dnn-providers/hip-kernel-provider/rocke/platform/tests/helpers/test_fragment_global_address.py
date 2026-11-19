# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Descriptor byte-base decomposition equals independent logical addressing."""

from itertools import product

import pytest

from rocke.core.arch.mfma_scale import MATRIX_FORMATS, matrix_layout
from rocke.core.arch.wmma_scale import scaled_matrix_layout
from rocke.core.dtypes import dtype_info
from rocke.core.ir import F32, I8, IRBuilder, PtrType, dtype_to_ir_type
from rocke.core.storage import storage_group
from rocke.helpers.tiling import (
    make_matrix_fragment_desc,
    make_tensor_desc,
    make_tile_desc,
    make_window,
)
from rocke.helpers.tiling.global_address import (
    byte_address,
    can_rebase_global,
    split_window,
    storage_displacement,
)
from rocke.helpers.tiling.register_mapper import RegisterMapper
from rocke.tests.lane_maps import LaneMapEvaluator


@pytest.mark.parametrize("dtype,words", list(product(MATRIX_FORMATS, (8, 16))))
def test_byte_base_plus_constant_run_offset_matches_logical_coordinates(dtype, words):
    layout = matrix_layout(dtype) if words == 8 else scaled_matrix_layout(dtype, words)
    tile = make_matrix_fragment_desc(layout)
    mapper = RegisterMapper(tile.layout)
    rows, k = tile.shape
    tensor = make_tensor_desc((rows * 3, k * 3), (k * 3, 1), dtype_to_ir_type(dtype))
    window = make_window(tensor, (rows, k))
    runtime, constant = split_window(window)
    group, _ = storage_group(dtype)
    bits = dtype_info(dtype).encoded_bits
    unit_bytes = 4 if bits < 8 else 1
    for lane in range(mapper.num_lanes):
        base = tuple(mapper.matrix_coordinates(lane, 0))
        base_byte = byte_address(LaneMapEvaluator, runtime, list(base))
        for reg in range(0, layout.fragment.count, group):
            coordinate = tuple(mapper.matrix_coordinates(lane, reg))
            delta = tuple(value - start for value, start in zip(coordinate, base))
            actual = (
                base_byte
                + storage_displacement(window, tile, delta, constant) * unit_bytes
            )
            expected = (
                ((rows + coordinate[0]) * tensor.strides[0] + k + coordinate[1])
                * bits
                // 8
            )
            assert actual == expected


def test_rebase_needs_an_explicit_bounded_global_contract():
    b = IRBuilder("bounded_rebase")
    ptr = b.param("A", PtrType(I8, "global"))
    constant_ptr = b.param("B", PtrType(I8, "constant"))
    tile = make_tile_desc(
        shape=(64, 16), thread_dist=(64, 1), thread_tile=(1, 16), wave_size=64
    )
    tensor = make_tensor_desc((64, 16), (16, 1), dtype_to_ir_type("fp8e4m3"))
    window = make_window(tensor, (0, 0))
    assert can_rebase_global(ptr, window, tile, 16, True)
    assert not can_rebase_global(ptr, window, tile, None, True)
    assert not can_rebase_global(ptr, window, tile, 16, False)
    assert not can_rebase_global(constant_ptr, window, tile, 16, True)
    negative = make_window(make_tensor_desc((64, 16), (-16, 1), tensor.dtype), (0, 0))
    assert not can_rebase_global(ptr, negative, tile, 16, True)


def test_byte_index_range_does_not_narrow_a_valid_large_element_view():
    b = IRBuilder("large_element_view")
    ptr = b.param("A", PtrType(F32, "global"))
    tile = make_tile_desc(
        shape=(64, 16), thread_dist=(64, 1), thread_tile=(1, 16), wave_size=64
    )
    tensor = make_tensor_desc((64, 1 << 24), (1 << 24, 1), F32)
    assert not can_rebase_global(ptr, make_window(tensor, (0, 0)), tile, 16, True)
