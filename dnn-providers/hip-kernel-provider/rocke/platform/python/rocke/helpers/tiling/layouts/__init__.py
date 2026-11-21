# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""rocke.helpers.tiling.layouts -- human-approachable tile-distribution authoring.

Author a tile as one axes-ordered list per geometric quantity (columns = axes) via
:func:`make_tile_desc`, which returns a ready ``TileDesc`` (shape + derived layout) -- never by
hand-writing raw encoding integer sequences.
"""
from __future__ import annotations

from .accumulator import make_mma_accumulator_desc
from .matrix_fragment import make_matrix_fragment_desc
from .scale import make_mma_scale_desc
from .tile_distribution import make_tile_desc

__all__ = [
    "make_tile_desc",
    "make_matrix_fragment_desc",
    "make_mma_accumulator_desc",
    "make_mma_scale_desc",
]
