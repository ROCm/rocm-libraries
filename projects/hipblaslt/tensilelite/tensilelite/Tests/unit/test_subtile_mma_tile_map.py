# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for ABGRGeometry.subtileForMmaTile, the MMA-tile to subtile mapping.

The mapping has two shapes. With a single subtile per group, or no stride, a
subtile owns a contiguous run of MMA tile rows. With several subtiles spaced by
a stride, one subtile's rows are interleaved with its neighbours', so the rows
it owns are not contiguous -- that is the case worth pinning, because getting it
wrong still produces a plausible-looking contiguous answer.

Pure geometry: no writer, no kernel emit, so the expectations are computed by
hand from the documented contract.
"""

from dataclasses import replace

import pytest

from Tensile.Components.Subtile.Kernel import AB_B16

pytestmark = pytest.mark.unit


def _geom(shape=(2, 1), count=1, stride=0):
    """An ABGRGeometry with the subtile fields for_kernel() would have set."""
    return replace(AB_B16.gr, subtileShape=shape,
                   subtileCount=count, subtileStride=stride)


def test_requires_for_kernel_to_have_materialized_the_subtile_fields():
    """Unmaterialized geometry must say so, not silently map to tile zero."""
    geom = replace(AB_B16.gr, subtileCount=None, subtileStride=None)
    with pytest.raises(RuntimeError, match="for_kernel"):
        geom.subtileForMmaTile(0, 0)


def test_single_subtile_owns_a_contiguous_row_run():
    subtile_id, block_shape, mma_tiles = _geom(count=1, stride=0).subtileForMmaTile(5, 3)
    assert subtile_id == (2, 3)          # row 5 // bM 2, col 3 // bK 1
    assert block_shape == (2, 1)
    assert mma_tiles == [(4, 3), (5, 3)]


def test_zero_stride_falls_back_to_the_contiguous_mapping():
    """subtileCount > 1 with no stride is still the contiguous case."""
    _, _, mma_tiles = _geom(count=4, stride=0).subtileForMmaTile(5, 0)
    assert mma_tiles == [(4, 0), (5, 0)]


def test_strided_subtiles_interleave_their_rows():
    """With a stride the owned rows are spread, not contiguous."""
    subtile_id, _, mma_tiles = _geom(count=2, stride=4).subtileForMmaTile(5, 0)
    assert subtile_id == (0, 0)
    assert mma_tiles == [(0, 0), (1, 0), (4, 0), (5, 0)]


def test_every_row_maps_into_a_subtile_that_contains_it():
    """The mapping must be self-consistent: r is always among the rows returned."""
    geom = _geom(count=2, stride=4)
    for r in range(16):
        _, _, mma_tiles = geom.subtileForMmaTile(r, 0)
        assert any(row == r for row, _ in mma_tiles), f"row {r} not in its own subtile"


def test_k_columns_come_from_the_subtile_k_extent():
    """bK > 1 widens the K span of every returned tile."""
    _, block_shape, mma_tiles = _geom(shape=(2, 2), count=1, stride=0).subtileForMmaTile(0, 3)
    assert block_shape == (2, 2)
    # col 3 lands in K subtile 1, which spans columns 2 and 3.
    assert sorted({col for _, col in mma_tiles}) == [2, 3]
    assert len(mma_tiles) == 4
