# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GIR analyses used through the package interface."""

from .cfg import BackEdges
from .swap_regions import SwapRegions
from .gr_increment import GrIncrementRegions
from .gl2_prefetch import Gl2PrefetchRegions
from .region_increment import RegionIncrementRegions

__all__ = [
    "BackEdges", "SwapRegions", "GrIncrementRegions", "Gl2PrefetchRegions",
    "RegionIncrementRegions",
]
