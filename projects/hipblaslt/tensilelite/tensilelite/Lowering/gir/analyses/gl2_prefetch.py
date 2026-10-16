# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Gl2PrefetchRegions -- WHERE the GL2 cache prefetch goes, once per summation chunk.
"""

from __future__ import annotations

from ..nodes import Mark
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
        for name in prog.blocks:
            if not self._is_steady(name):
                continue
            out.append(PendingMark(
                mark=Mark("gl2_prefetch", {"depth": depth, "block": name}),
                region=Region(name, after=BLOCK_ENTRY, before=BLOCK_EXIT, policy=MIDPOINT)))
        return out

    @staticmethod
    def _is_steady(name):
        """The steady body only, and ONE mark per steady block -- which is one per summation chunk,
        the same rate as `gr_increment`, because the scaffold's own placement is one per
        `_loopBody` call and `_loopBody` is called once per loop copy (`KernelWriter.py` ~6288).
        """
        return name.startswith("steady")
