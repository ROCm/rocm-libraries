# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The register-side value objects: a :class:`TileDesc` (logical layout) and the
:class:`Fragment` (its realized per-lane registers).

This is the REGISTER side of the surface, the counterpart to
:mod:`rocke.helpers.tiling.descriptors` (the memory side). It is pure data (no IRBuilder): a
`TileDesc` says *where each element lives* in lanes/registers (a shape + a
:class:`~rocke.helpers.tiling.encoding.WarpDistributionEncoding`), and a `Fragment` binds that layout to
an element `dtype` and the SSA value holding the registers. The IR verbs in
:mod:`rocke.helpers.tiling.emit` fill / load / store the `Fragment`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .encoding import WarpDistributionEncoding
from .register_mapper import RegisterMapper

__all__ = ["TileDesc", "Fragment", "make_fragment", "fragment_length"]

def fragment_length(encoding: WarpDistributionEncoding) -> int:
    """Per-lane register count for an encoding (= num_vector_items)."""
    return RegisterMapper(encoding).num_vector_items

@dataclass(frozen=True)
class TileDesc:
    """A logical-matrix -> per-lane-register layout descriptor.

    DTYPE-FREE and MEMORY-FREE: it says *where each element lives* (shape + the warp
    distribution `layout`), never *what type it is* or *which buffer*. The same TileDesc is
    reusable across dtypes -- the type is bound only when a `Fragment` is realized (load) or
    written (store). `shape` is the logical (rows, cols) of the tile.
    """

    shape: tuple[int, ...]
    layout: WarpDistributionEncoding

    @property
    def register_count(self) -> int:
        """Per-lane register count implied by the layout."""
        return fragment_length(self.layout)

@dataclass
class Fragment:
    """Per-lane register data for a tile: the `tile_desc` that lays it out, its element
    `dtype`, and the SSA `value` holding the registers. Build it from ``(tile_desc, dtype)``;
    `fill` or `load` sets `value`."""

    tile_desc: TileDesc
    dtype: Any
    value: Any = None

def make_fragment(tile_desc: TileDesc, dtype: Any, value: Any = None) -> Fragment:
    """Free factory: a `Fragment` for `tile_desc` at element `dtype` (registers set by
    `fill_fragment` / `load_fragment`)."""
    return Fragment(tile_desc, dtype, value)
