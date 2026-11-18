# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Atom chunk layouts compose with the public fragment transport descriptor."""

from itertools import product

import pytest

from rocke.core.arch.mfma_scale import MATRIX_FORMATS, matrix_layout
from rocke.core.arch.wmma_scale import scaled_matrix_layout
from rocke.core.storage import BitPacking, FragmentPacking, MatrixFragmentLayout
from rocke.helpers.tiling import make_matrix_fragment_desc
from rocke.helpers.tiling.register_mapper import RegisterMapper


@pytest.mark.parametrize("dtype,words", list(product(MATRIX_FORMATS, (8, 16))))
def test_matrix_descriptor_preserves_every_lane_and_logical_slot(dtype, words):
    layout = matrix_layout(dtype) if words == 8 else scaled_matrix_layout(dtype, words)
    desc = make_matrix_fragment_desc(layout)
    mapper = RegisterMapper(desc.layout)
    assert desc.register_count == layout.fragment.count
    for lane in range(layout.lane_groups * layout.lanes_per_group):
        for slot in range(layout.fragment.count):
            assert tuple(mapper.matrix_coordinates(lane, slot)) == layout.coord(
                lane, slot
            )


@pytest.mark.parametrize("m", (16, 32))
def test_tf32_descriptor_preserves_the_probe_layout(m):
    layout = MatrixFragmentLayout(
        FragmentPacking(BitPacking(32), 2, 32, 2), 2, 64 // m, m
    )
    mapper = RegisterMapper(make_matrix_fragment_desc(layout).layout)
    for lane in range(64):
        for slot in range(2):
            assert tuple(mapper.matrix_coordinates(lane, slot)) == layout.coord(
                lane, slot
            )


def test_single_group_layout_drops_non_lane_axes_from_thread_order():
    layout = MatrixFragmentLayout(FragmentPacking(BitPacking(8), 16, 32, 4), 16, 1, 32)
    desc = make_matrix_fragment_desc(layout)
    assert desc.shape == (32, 16)
    mapper = RegisterMapper(desc.layout)
    assert tuple(mapper.matrix_coordinates(31, 15)) == (31, 15)


def test_abi_padding_does_not_change_spatial_descriptor():
    a = MatrixFragmentLayout(FragmentPacking(BitPacking(6), 32, 32, 8), 32, 2, 32)
    b = MatrixFragmentLayout(FragmentPacking(BitPacking(6), 32, 32, 6), 32, 2, 32)
    assert make_matrix_fragment_desc(a) == make_matrix_fragment_desc(b)


def test_matrix_descriptor_refuses_a_non_layout_input():
    with pytest.raises(TypeError, match="requires a MatrixFragmentLayout"):
        make_matrix_fragment_desc(None)
