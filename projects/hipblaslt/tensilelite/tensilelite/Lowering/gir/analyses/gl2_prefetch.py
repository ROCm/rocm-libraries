# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Gl2PrefetchRegions -- WHERE the GL2 cache prefetch goes, once per summation chunk.
"""

from __future__ import annotations

from ..nodes import Mark, Move
from ..nodes import Region, PendingMark, BLOCK_ENTRY, BLOCK_EXIT, MIDPOINT
from ..analysis import Analysis


class Gl2PrefetchRegions(Analysis):
    """`[PendingMark]` -- one `gl2_prefetch` Mark per steady chunk, or none when GL2 is off.

    THE DEPTH IS A PRESET, NOT A theta FACT, so it arrives on `Program.params`, not on the
    theta meta.
    """

    #: the `Program.params` key.  Named once so the producer (KernelWriter) and this consumer
    #: cannot drift apart silently -- a mistyped string here is an analysis that is inert forever.
    PARAM = "PrefetchGL2"

    @property
    def cache_key(self):
        return "Gl2PrefetchRegions"

    def run(self, prog, am=None):
        depth = int((prog.params or {}).get(self.PARAM, 0) or 0)
        if depth <= 0:
            return []
        out = []
        for name, block in prog.blocks.items():
            if not self._is_steady(name):
                continue
            out.append(PendingMark(
                mark=Mark("gl2_prefetch", {"depth": depth, "block": name}),
                region=Region(name, after=BLOCK_ENTRY, before=self._first_fetch(block),
                              policy=MIDPOINT)))
        return out

    @staticmethod
    def _first_fetch(block):
        """The block's first Move that READS GLOBAL, or BLOCK_EXIT when it has none.

        The prefetch warms a line some chunks ahead, so nothing in this block consumes it and the
        window is a spacing choice: it ends at the first fetch because that fetch and the prefetch
        issue down the same memory path, and bisecting the whole block would leave them adjacent.
        Keyed on the SOURCE, as `gr_increment` is: a register->shared write also fills shared but
        issues nowhere near global, so ending the window on it would spread against the wrong run.
        """
        for inst in block.body:
            if isinstance(inst, Move) and any(r.tile.space == "global" for r in inst.srcs):
                return inst
        return BLOCK_EXIT

    @staticmethod
    def _is_steady(name):
        """The steady body only, and ONE mark per steady block -- which is one per summation chunk,
        the same rate as `gr_increment`, because the scaffold's own placement is one per
        `_loopBody` call and `_loopBody` is called once per loop copy (`KernelWriter.py` ~6288).
        """
        return name.startswith("steady")
