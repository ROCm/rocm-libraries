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

    def swap_dims(self, i: int, j: int) -> "TileDesc":
        """Swap two X-dims -- a free-symmetry REPOSITION (coordinate transpose), register-identity
        and label-invariant. The same elements on the same lanes, addressed in transposed axis order
        (e.g. index a free-innermost memref in (K, free) order while the MMA side keeps (free, K)).

        Rank-generic: ``i``/``j`` are X-dim indices. An ``rh_major`` value ``v`` names replication
        (``v==0``) or X-dim ``v-1``, so swapping X-dims ``i`` and ``j`` remaps majors ``i+1<->j+1``
        and leaves replication + all minors untouched.
        """
        e = self.layout
        n = len(e.hierarchical_lengths)
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError(f"swap_dims axes out of range -- i={i}, j={j}, rank={n}")

        def remap(v: int) -> int:
            if v == i + 1:
                return j + 1
            if v == j + 1:
                return i + 1
            return v

        hl = list(e.hierarchical_lengths)
        hl[i], hl[j] = hl[j], hl[i]
        shape = list(self.shape)
        shape[i], shape[j] = shape[j], shape[i]
        return TileDesc(
            shape=tuple(shape),
            layout=WarpDistributionEncoding(
                replication_lengths=e.replication_lengths,
                hierarchical_lengths=tuple(hl),
                lane_to_rh_major=tuple(tuple(remap(m) for m in row) for row in e.lane_to_rh_major),
                lane_to_rh_minor=e.lane_to_rh_minor,
                register_to_rh_major=tuple(remap(m) for m in e.register_to_rh_major),
                register_to_rh_minor=e.register_to_rh_minor,
            ),
        )

    def reorder_registers(self, order: tuple[int, ...]) -> "TileDesc":
        """Permute the REGISTER-bucket significance (outer = most-significant slot), keeping the lane
        map and every label identical -- the same elements on the same lanes, in a different register
        order. ``order`` is a permutation of the existing register buckets; callers name their buckets
        and compute the permutation (a bucket that collapses to extent 1 is absent from both the
        encoding and the permutation, so index math stays stable). ``make_tile_desc`` hardcodes
        ``block_repeat`` major / ``thread_tile`` minor, which cannot express the MMA-ready or the
        free-dim-fastest LDS-read order; this is the minimal escape.
        """
        e = self.layout
        if sorted(order) != list(range(len(e.register_to_rh_major))):
            raise ValueError(
                f"reorder_registers order must be a permutation of the "
                f"{len(e.register_to_rh_major)} register buckets -- got {order!r}"
            )
        return TileDesc(
            shape=self.shape,
            layout=WarpDistributionEncoding(
                replication_lengths=e.replication_lengths,
                hierarchical_lengths=e.hierarchical_lengths,
                lane_to_rh_major=e.lane_to_rh_major,
                lane_to_rh_minor=e.lane_to_rh_minor,
                register_to_rh_major=tuple(e.register_to_rh_major[i] for i in order),
                register_to_rh_minor=tuple(e.register_to_rh_minor[i] for i in order),
            ),
        )

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
