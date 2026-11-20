# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
PlacementPass -- resolve each PendingMark's Region to a concrete anchor.

Policy travels with the Region (design "placement by swap TYPE -- a user directive, not just
sigma"), because it differs by what the Mark mutates.

"""

from __future__ import annotations

from .base import Pass
from ..nodes import BLOCK_EXIT, MIDPOINT, region_slot


class PlacementPass(Pass):
    """Resolve each PendingMark's legal window to one concrete anchor. Mutates no IR."""
    def run(self, prog, am):
        for pm in prog.pending:
            pm.anchor = self._resolve(prog, pm.region)

    @classmethod
    def _resolve(cls, prog, region):
        body = prog.block(region.block).body
        lo = region_slot(body, region.after, True)    # first legal slot
        hi = region_slot(body, region.before, False)  # one past the last legal slot
        if hi < lo:
            return (region.block, region.before)             # malformed window: stated bound
        idx = lo + (hi - lo) // 2 if region.policy == MIDPOINT else lo
        return (region.block, body[idx] if idx < len(body) else BLOCK_EXIT)
