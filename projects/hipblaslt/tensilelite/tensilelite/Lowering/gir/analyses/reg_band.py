# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
RegBand -- VALIDATE the register rotation width W; never decide it.
"""

from __future__ import annotations

from ..nodes import Move
from ..analysis import Analysis


class RegBand:
    def __init__(self, widths):
        self._widths = widths     # (operand, group) -> W

    def items(self):
        return dict(self._widths)


class RegBandAnalysis(Analysis):
    @property
    def cache_key(self):
        return "RegBand"

    def run(self, prog, am):
        widths = {}
        for blk in prog.blocks.values():
            for inst in blk.body:
                if not isinstance(inst, Move):
                    continue
                for ref in inst.dsts:
                    if ref.tile.space != "register" or ref.reg_ring is None:
                        continue
                    key = (ref.tile.operand, ref.group)
                    if key in widths and widths[key] != ref.reg_ring:
                        raise RuntimeError(
                            f"RegBand: operand {ref.tile.operand} group {ref.group} has "
                            f"conflicting register widths {widths[key]} vs {ref.reg_ring} "
                            f"(LoopIR slot moduli inconsistent)")
                    widths[key] = ref.reg_ring
        return RegBand(widths)
