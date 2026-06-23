# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Cluster-scope barrier handshake for the subtile mainloop.

Subtile-specific equivalent of the StinkyTofu InsertClusterBarrierPass (which
does not run at OptLevel 0 / ScheduleIterAlg=3). Free functions take the writer
explicitly to keep cluster logic in Subtile/ rather than KernelWriterAssembly.

The handshake is split into a *signal* half and a *wait* half so the wait can be
moved away from the signal: the cluster barrier's cross-CU latency is then hidden
behind WMMAs that issue in the gap instead of being exposed as a stall. See
``spliceClusterBarrierWait``.
"""

from __future__ import annotations

from rocisa.code import Label, Module
from rocisa.instruction import (
    MFMAInstruction, MXMFMAInstruction, SBarrier, SBarrierSignalIsFirst,
    SCBranchSCC0,
)

# Number of WMMAs to issue between the cluster_barrier signal and its wait so the
# barrier's cross-CU latency is hidden behind useful work rather than stalled on.
CLUSTER_BARRIER_WMMA_GAP = 4

_isMMA = lambda x: isinstance(x, (MFMAInstruction, MXMFMAInstruction))


def subtileClusterBarrierSignal(writer, kernel, label="") -> Module:
    """Workgroup barrier + the first-arriving wave's cluster_barrier signal.

    Ends at the ``skipPreSignal`` label so all waves fall through to whatever work
    follows; the matching wait is emitted later by ``subtileClusterBarrierWait``.
    """
    mod = Module("subtile_cluster_barrier_signal")
    # Workgroup barrier via isfirst: the first wave to arrive gets SCC=1 and so
    # is the single wave that signals the cluster barrier (one arrival per WG).
    skipPreSignal = Label(writer.labels.getUniqueNamePrefix("skipCBPreSignal"), "", 16)
    mod.add(SBarrierSignalIsFirst(False, "workgroup barrier signal (isfirst)"))
    mod.add(SBarrier(True, True, False, "workgroup barrier wait"))
    mod.add(SCBranchSCC0(skipPreSignal.getLabelName(), "only the first-arriving wave signals the cluster"))
    mod.add(SBarrier(True, False, True, "cluster_barrier signal"))
    mod.add(skipPreSignal)
    return mod


def subtileClusterBarrierWait(writer, kernel, label="") -> Module:
    """The all-waves cluster_barrier wait that closes the handshake."""
    mod = Module("subtile_cluster_barrier_wait")
    mod.add(SBarrier(True, True, True, "cluster_barrier wait"))
    return mod


def subtileClusterBarrier(writer, kernel, label="") -> Module:
    """Signal + wait emitted back-to-back (no latency hiding).

    Retained for callers that do not splice the wait away from the signal.
    """
    mod = Module("subtile_cluster_barrier")
    mod.appendModule(subtileClusterBarrierSignal(writer, kernel, label))
    mod.appendModule(subtileClusterBarrierWait(writer, kernel, label))
    return mod


def spliceClusterBarrierWait(module, waitMod, gap=CLUSTER_BARRIER_WMMA_GAP) -> Module:
    """Insert the cluster_barrier wait ``gap`` WMMAs after the cluster signal.

    Walks ``module`` in program order and inserts the wait instruction(s)
    immediately after the ``gap``-th WMMA, so the cluster barrier's cross-CU
    latency overlaps those WMMAs instead of stalling. If fewer than ``gap`` WMMAs
    are present (e.g. PRELOOP), the wait is appended at the end so the handshake
    is always closed (correctness over latency hiding).

    Returns a rebuilt flat Module; the input is left untouched. Mirrors the
    flatitems()-rebuild idiom of WaitAluInsertion's post-schedule passes.
    """
    waitItems = waitMod.flatitems()
    result = Module(module.name)
    count = 0
    done = False
    for inst in module.flatitems():
        result.add(inst)
        if not done and _isMMA(inst):
            count += 1
            if count >= gap:
                for w in waitItems:
                    result.add(w)
                done = True
    if not done:  # fewer than `gap` WMMAs in this section: close the handshake at the end
        for w in waitItems:
            result.add(w)
    return result
