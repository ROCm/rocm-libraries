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
from rocisa.container import sgpr, vgpr
from rocisa.instruction import (
    MFMAInstruction, MXMFMAInstruction, SBarrier, SCBranchSCC0, SCmpEQU32,
    VReadfirstlaneB32,
)

# Number of WMMAs to issue between the cluster_barrier signal and its wait so the
# barrier's cross-CU latency is hidden behind useful work rather than stalled on.
CLUSTER_BARRIER_WMMA_GAP = 8

_isMMA = lambda x: isinstance(x, (MFMAInstruction, MXMFMAInstruction))


_isWgBarrier = lambda x: isinstance(x, SBarrier) and "-3" not in str(x)


def subtileClusterBarrierSignal(writer, kernel, label="") -> Module:
    """Wave-0-only cluster_barrier signal (no workgroup barrier).

    Wave 0 alone issues the cluster_barrier signal; all other waves branch over
    it. Ends at the ``skipPreSignal`` label so all waves fall through to whatever
    work follows; the matching wait is emitted later by ``subtileClusterBarrierWait``.

    This module carries no workgroup barrier of its own; ``spliceClusterBarrierSignal``
    places it immediately after the mainloop's existing workgroup barrier so that
    sync is reused rather than duplicated.
    """
    mod = Module("subtile_cluster_barrier_signal")
    skipPreSignal = Label(writer.labels.getUniqueNamePrefix("skipCBPreSignal"), "", 16)
    # No workgroup barrier here: this signal is spliced in immediately after the
    # mainloop's existing workgroup barrier (s_barrier_signal -1/s_barrier_wait -1)
    # by spliceClusterBarrierSignal, so all waves are already synced before wave 0
    # announces the workgroup's arrival to the cluster.
    # Elect wave 0 to issue the single cluster_barrier signal. readfirstlane of
    # Serial returns the wave's lowest lane id (= waveId * wavesize), which is 0
    # only for wave 0.
    with writer.allocTmpSgpr(1) as tmpSgpr:
        s = tmpSgpr.idx
        mod.add(VReadfirstlaneB32(sgpr(s), vgpr("Serial"), "first lane tId (= waveId * wavesize)"))
        mod.add(SCmpEQU32(sgpr(s), 0, "wave 0?"))
        mod.add(SCBranchSCC0(skipPreSignal.getLabelName(), "only wave 0 signals the cluster"))
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


def spliceClusterBarrierSignal(module, signalMod) -> Module:
    """Insert the cluster_barrier signal after the first workgroup barrier.

    Walks ``module`` in program order and inserts the signal instruction(s)
    immediately after the first workgroup barrier (the mainloop's existing
    ``s_barrier_signal -1``/``s_barrier_wait -1``, emitted as a combined
    ``SBarrier()`` by ``emit_sync``), reusing that workgroup sync instead of
    emitting a second barrier. At this point no cluster (``-3``) barriers are in
    the module yet, so the first ``SBarrier`` is the workgroup one.

    If no workgroup barrier is found in this section, the signal is prepended at
    the start so the handshake is still opened (correctness over reuse).

    Returns a rebuilt flat Module; the input is left untouched.
    """
    signalItems = signalMod.flatitems()
    result = Module(module.name)
    done = False
    for inst in module.flatitems():
        result.add(inst)
        if not done and _isWgBarrier(inst):
            for s in signalItems:
                result.add(s)
            done = True
    if not done:  # no workgroup barrier in this section: open the handshake at the start
        head = Module(module.name)
        for s in signalItems:
            head.add(s)
        for inst in module.flatitems():
            head.add(inst)
        return head
    return result


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
