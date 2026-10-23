# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""A `TileDesc`'s shape must equal the extent its layout addresses, on every axis.

The verbs bound-check a tile by its `shape` but place each register by its `layout`. A shape
smaller than the layout's extent lets registers land outside the region the bound checks cover:
an out-of-bounds read and write with no error. Construction now refuses any shape that differs
from the layout's extent, with one message naming both.

The census half builds every tracked in-tree demo configuration (the same list the IR snapshot
pins): each one constructs its TileDescs through the checked constructor, so a demo that relied on
a mismatched shape fails to build here.

Offline (no GPU): layouts come from the real authoring helpers.
"""

from __future__ import annotations

import pytest

from rocke.helpers.tiling import TileDesc, TileMma, Tiling, make_tile_desc

from .test_demo_ir_snapshot import _CONFIGS

# A 16x16 tile, 4x4 lanes each holding a 4x4 block: the layout addresses (16, 16).
_DESC_16x16 = make_tile_desc(
    shape=[16, 16], thread_tile=[4, 4], thread_dist=[4, 4], wave_size=16
)


def test_a_shape_smaller_than_the_layout_extent_is_refused() -> None:
    with pytest.raises(ValueError) as error:
        TileDesc((8, 16), _DESC_16x16.layout)
    assert str(error.value) == (
        "TileDesc shape (8, 16) does not match the extent its layout addresses (16, 16) "
        "-- shape must equal the layout extent on every axis"
    )


@pytest.mark.parametrize(
    "shape",
    [(16, 32), (32, 16), (16,), (16, 16, 1)],
    ids=["larger-cols", "larger-rows", "fewer-axes", "more-axes"],
)
def test_any_other_mismatch_is_refused(shape: tuple[int, ...]) -> None:
    with pytest.raises(
        ValueError, match="does not match the extent its layout addresses"
    ):
        TileDesc(shape, _DESC_16x16.layout)


def test_a_matching_shape_is_accepted_and_survives_the_rewrites() -> None:
    desc = TileDesc((16, 16), _DESC_16x16.layout)
    assert desc == _DESC_16x16
    # The two in-place rewrites build new TileDescs through the same check.
    assert desc.swap_dims(0, 1).shape == (16, 16)
    order = tuple(reversed(range(len(desc.layout.register_to_rh_major))))
    assert desc.reorder_registers(order).shape == (16, 16)


def test_a_non_square_tile_keeps_its_axis_order_through_swap_dims() -> None:
    desc = make_tile_desc(
        shape=[8, 32], thread_tile=[1, 4], thread_dist=[8, 8], wave_size=64
    )
    assert desc.swap_dims(0, 1).shape == (32, 8)


def test_every_mma_operand_and_accumulator_layout_matches_its_shape() -> None:
    mma = TileMma(
        (32, 32, 16),
        a="f16",
        b="f16",
        c="f32",
        target="gfx90a",
        tiling=Tiling(atom_shape=(16, 16, 16)),
    )
    assert mma.a_desc.shape == (32, 16)
    assert mma.c_desc.shape == (32, 32)


@pytest.mark.parametrize("label", list(_CONFIGS))
def test_every_tracked_demo_builds_only_matching_tile_descs(label: str) -> None:
    _CONFIGS[label]()
