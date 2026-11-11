# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
CollectPendingMarksPass -- run the region analyses and stash their PendingMarks on
`prog.pending`. IR-UNTOUCHED (it only reads analyses and records their plan); the placement +
apply passes downstream turn the plan into body Marks.
"""

from __future__ import annotations

from .base import Pass
from ..analyses import (SwapRegions, GrIncrementRegions, RegionIncrementRegions,
                        Gl2PrefetchRegions)


class CollectPendingMarksPass(Pass):
    """Run the region analyses and stash every PendingMark they produce on the program."""
    def __init__(self, region_analyses=None):
        # region analyses to run; default = the R3 set.  R4 extends this list (GuardSite).
        self._analyses = (region_analyses if region_analyses is not None
                          else [SwapRegions(), GrIncrementRegions(),
                                RegionIncrementRegions(),
                                #: inert unless meta['prefetch_gl2'] > 0.
                                Gl2PrefetchRegions()])

    def run(self, prog, am):
        pending = []
        for analysis in self._analyses:
            pending.extend(am.get(analysis, prog))
        prog.pending = pending
