# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
Gl2PrefetchRegions -- WHERE the GL2 cache prefetch goes, once per summation chunk and movement.
"""

from __future__ import annotations

from ..nodes import Mark, Move, copy_unit
from ..nodes import Region, PendingMark, BLOCK_ENTRY, BLOCK_EXIT, MIDPOINT
from ..analysis import Analysis
# IMPORTED, not re-derived: `gr_increment` reads a movement's identity the same way, and a
# second reading here is exactly how the two would drift apart.
from .region_increment import _canonical_unit


class Gl2PrefetchRegions(Analysis):
    """`[PendingMark]` -- one `gl2_prefetch` Mark per steady chunk and MOVEMENT, or none when
    GL2 is off.

    Per movement, as `gr_increment` is: a movement owns one descriptor and issues one fetch, so
    it owns the warm-up for that fetch. One mark for the whole block instead would put every
    operand's prefetch in front of whichever fetch happens to come first, and the scheduler --
    which pairs a prefetch with the next fetch after it -- would anchor them all there.

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
            fetches = self._fetches(prog, block)
            if not fetches:
                # No movement to key on (no global read in the block): one unowned mark, the
                # window the whole block, as before movements were distinguished.
                out.append(PendingMark(
                    mark=Mark("gl2_prefetch", {"depth": depth, "block": name, "unit": ()}),
                    region=Region(name, after=BLOCK_ENTRY, before=BLOCK_EXIT, policy=MIDPOINT)))
                continue
            after = BLOCK_ENTRY
            for unit, fetch in fetches:
                out.append(PendingMark(
                    mark=Mark("gl2_prefetch", {"depth": depth, "block": name, "unit": unit}),
                    region=Region(name, after=after, before=fetch, policy=MIDPOINT)))
                after = fetch
        return out

    @staticmethod
    def _fetches(prog, block):
        """`[(movement, its first fetch)]` in program order -- the block's global reads.

        The window of each mark runs from the previous movement's fetch to its own, so the
        prefetches come out interleaved with the fetches they warm rather than bunched in front
        of the first one. Keyed on the SOURCE, as `gr_increment` is: a register->shared write
        also fills shared but issues nowhere near global, so ending a window on it would spread
        against the wrong run.
        """
        out, seen = [], []
        for inst in block.body:
            if not isinstance(inst, Move):
                continue
            if not any(r.tile.space == "global" for r in inst.srcs):
                continue
            members, _refs = copy_unit(inst)
            if members is None:
                continue
            unit = _canonical_unit(prog, members)
            if unit in seen:
                continue
            seen.append(unit)
            out.append((unit, inst))
        return out

    @staticmethod
    def _is_steady(name):
        """The steady body only, and one mark per steady block and movement -- which is one per
        summation chunk and movement, the same rate as `gr_increment`, because the scaffold's own
        placement is one per `_loopBody` call and `_loopBody` is called once per loop copy
        (`KernelWriter.py` ~6288).
        """
        return name.startswith("steady")
