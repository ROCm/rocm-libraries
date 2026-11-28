# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Spatial tile descriptors from atom-owned contiguous-chunk operand layouts."""

from __future__ import annotations

from ....core.storage import MatrixFragmentLayout
from ..fragments import TileDesc
from .tile_distribution import make_tile_desc

__all__ = ["make_matrix_fragment_desc"]


def make_matrix_fragment_desc(layout: MatrixFragmentLayout) -> TileDesc:
    """Describe the logical elements an atom assigns to each lane/register.

    Rows come from ``lane % lanes_per_group``. Lane groups alternate complete
    contiguous chunks along K. Packing and carrier padding do not enter this
    spatial descriptor; the memory dtype and the operand ABI bind separately.
    """
    if not isinstance(layout, MatrixFragmentLayout):
        raise TypeError("matrix fragment descriptor requires a MatrixFragmentLayout")
    return make_tile_desc(
        shape=(layout.lanes_per_group, layout.fragment.count * layout.lane_groups),
        thread_dist=(layout.lanes_per_group, layout.lane_groups),
        thread_tile=(1, layout.chunk_elements),
        block_repeat=(1, layout.chunks_per_lane),
        thread_order=tuple(
            axis
            for axis in (1, 0)
            if (layout.lanes_per_group, layout.lane_groups)[axis] > 1
        ),
        wave_size=layout.lane_groups * layout.lanes_per_group,
    )
