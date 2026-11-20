# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""
The GIR backend: pass ORDER as data, and the driver that runs it.

Everything that only READS the program is an Analysis (under `gir/analyses/`) or a check in
`gir/verify.py`; only the passes listed here may change it.
"""

from __future__ import annotations

from ..analysis import AnalysisManager
from ..verify import verify_gir
from ..verify_dataflow import check_register_dataflow, check_ring_rotation
from .scaffold_shape import ScaffoldShapePass
from .entrance_frames import EntranceFramePhisPass
from .hoist_copies import HoistCopiesPass
from .collect_pending import CollectPendingMarksPass
from .placement import PlacementPass
from .apply_marks import ApplyMarksPass
from .record_unemitted import RecordUnemittedPass


def pipeline():
    """The backend pass order.

    ScaffoldShapePass is the ONLY stage allowed to change the CFG: it merges the correct short-arm
    register order into the prologue/drain shape, proves equivalence, adds early-exit edges, and
    labels the terminators.

    Fences, memory tokens and wait counts are all StinkyTofu's, derived after scheduling from the
    frame contract this pipeline exports; GIR states the dataflow and stops there.
    """
    return [
        ScaffoldShapePass(),
        EntranceFramePhisPass(),
        HoistCopiesPass(reads=True),
        CollectPendingMarksPass(),
        PlacementPass(),
        ApplyMarksPass(),
        RecordUnemittedPass(),
    ]


def run_pipeline(prog, passes=None, verify=True):
    """Run `passes` (default: pipeline()) over `prog`, then verify.  Returns `prog`."""
    passes = passes if passes is not None else pipeline()
    am = AnalysisManager()
    for p in passes:
        before = prog.version
        p.run(prog, am)
        # ONLY a mutation drops the cache, and `bump()` is what declares one.  Clearing
        # unconditionally discarded results a later pass in this same run still wanted.
        if prog.version != before:
            am.invalidate()
    if verify:
        verify_gir(prog)
    # Unconditional: a wmma reading another source is a wrong answer, not a diagnostic.
    lost = [v for v in check_register_dataflow(prog, raise_wrong_source=True)
            if v.lstrip("[loop-carried] ").startswith("OVERWRITE-BEFORE-USE")]
    if lost:
        # A FOLD'S MEMBERS NEED NOT BE CONSUMED TOGETHER.  One read lands tiles that are adjacent
        # in LDS; the loop order may put the ring between two of them, and the refill then takes
        # the slot back before the later member is read.  That is this check, said directly --
        # leaving it to the WRONG SOURCE a consumer raises later names the register and not the
        # reason.  REDUNDANT-RELOAD stays a diagnostic: the same value back is no loss.
        raise RuntimeError(
            "OVERWRITE-BEFORE-USE: a refill took a register back before its value was read.\n"
            + "\n".join(lost))
    # Also unconditional, and for the same reason: a ring that does not advance by its period
    # hands one buffer two of a trip's coordinates, which no consumer-side check can see.
    stuck = check_ring_rotation(prog)
    if stuck:
        raise RuntimeError(
            "RING-DOES-NOT-ROTATE: a copy of the loop body did not advance a ring.\n"
            + "\n".join(stuck))
    return prog
