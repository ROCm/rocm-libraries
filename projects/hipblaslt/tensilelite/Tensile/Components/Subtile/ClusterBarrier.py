# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Cluster-scope barrier handshake for the subtile mainloop.

The handshake is split into a *signal* half and a *wait* half so the wait can be
moved away from the signal, hiding the cluster barrier's cross-CU latency behind
the WMMAs that issue in between instead of exposing it as a stall.
"""

from __future__ import annotations

from rocisa.code import Label, Module
from rocisa.container import sgpr
from rocisa.instruction import (SBarrier, BranchInstruction, SCBranchSCC0,
                                SCmpEQU32,
                                MFMAInstruction, MXMFMAInstruction)

_isWgBarrier = lambda x: isinstance(x, SBarrier) and "s_barrier_wait -1" in str(x)


def _findNextMFMA(items, start):
    """Index of the first MFMA at/after ``start``, or ``None`` if none follows."""
    for j in range(start, len(items)):
        if isinstance(items[j], (MFMAInstruction, MXMFMAInstruction)):
            return j
    return None


def subtileClusterBarrierSignal(writer, kernel) -> Module:
    """Wave-0-only cluster_barrier signal.

    Wave 0 alone issues the cluster_barrier signal; all other waves branch over
    it. Ends at the ``skipPreSignal`` label so all waves fall through to whatever
    work follows; the matching wait is emitted later by ``subtileClusterBarrierWait``.
    """
    mod = Module("subtile_cluster_barrier_signal")
    skipPreSignal = Label(writer.labels.getUniqueNamePrefix("skipCBPreSignal"), "", 16)
    # Elect wave 0 to issue the single cluster_barrier signal.
    mod.add(SCmpEQU32(sgpr("WaveIdx"), 0, "wave 0?"))
    mod.add(SCBranchSCC0(skipPreSignal.getLabelName(), "only wave 0 signals the cluster"))
    mod.add(SBarrier(True, False, True, "cluster_barrier signal"))
    mod.add(skipPreSignal)
    return mod


def subtileClusterBarrierWait(writer, kernel) -> Module:
    """The all-waves cluster_barrier wait that closes the handshake."""
    mod = Module("subtile_cluster_barrier_wait")
    mod.add(SBarrier(True, True, True, "cluster_barrier wait"))
    return mod


def insertClusterBarrier(module, writer, kernel):
    """Splice the cluster-scope barrier handshake into the post-schedule order.

    No-op unless ``ClusterBarrier`` is enabled. The signal is spliced in right
    after the mainloop's existing workgroup barrier (reusing that sync instead of
    emitting a second one); the wait is appended at the end of the section, so the
    barrier's cross-CU latency overlaps the whole macro tile's WMMAs before the
    handshake is closed.

    If no workgroup barrier is found in this section, the signal is prepended at
    the start so the handshake is still opened (correctness over reuse).

    Returns a rebuilt Module; the input is left untouched.
    """
    if not kernel.get("ClusterBarrier"):
        return module

    signalItems = subtileClusterBarrierSignal(writer, kernel).flatitems()
    waitItems = subtileClusterBarrierWait(writer, kernel).flatitems()

    # ClusterBarrier is only supported on gfx1250.
    assert writer.states.asmCaps.get("HasClusterBarrier", False), \
        "ClusterBarrier requires the HasClusterBarrier asm capability"

    # Place the wave-0-election branch right after a WMMA to hide branching
    # latency: keep s_cmp before the next scheduled MFMA and emit the branch
    # after it.

    items = module.flatitems()
    result = Module(module.name)
    done = False
    skip = set()
    for i, inst in enumerate(items):
        if i in skip:
            continue
        result.add(inst)
        if not done and _isWgBarrier(inst):
            done = True
            mfmaIdx = _findNextMFMA(items, i + 1)
            if mfmaIdx is None:
                # No following MFMA to pin the branch to: emit the block intact
                # (best-effort).
                for s in signalItems:
                    result.add(s)
            else:
                # Split the signal block at the wave-0 election branch. The
                # block is authored with exactly one conditional branch; assert
                # it so a future change that adds another fails loudly here.
                brIdxs = [k for k, s in enumerate(signalItems)
                          if isinstance(s, SCBranchSCC0)]
                assert len(brIdxs) == 1, \
                    "signal block must contain exactly one wave-0 election branch"
                brIdx = brIdxs[0]
                pre, post = signalItems[:brIdx], signalItems[brIdx:]
                # Everything up to the MFMA (incl. its s_set_vgpr_msb primer)
                # keeps its order, then s_cmp, the MFMA, and the branch. SCC
                # survives the MFMA and vgpr-msb is a persistent mode, so the
                # intervening compare disturbs neither.
                for k in range(i + 1, mfmaIdx):
                    result.add(items[k])
                    skip.add(k)
                for s in pre:
                    result.add(s)
                result.add(items[mfmaIdx])
                skip.add(mfmaIdx)
                for s in post:
                    result.add(s)
    if not done:  # no workgroup barrier: open the handshake at the start
        head = Module(module.name)
        head.add(SBarrier(True, False, False))
        head.add(SBarrier(True, True, False, "workgroup barrier wait"))
        for s in signalItems:
            head.add(s)
        for inst in result.flatitems():
            head.add(inst)
        result = head

    # Second pass: place the wait before the first branch after the signal,
    # so no exit path can skip it.  Falls back to end-of-module if no branch follows.
    signalInst = next(s for s in signalItems if isinstance(s, SBarrier))
    items = result.flatitems()
    patched = Module(result.name)
    signalSeen = False
    waitPlaced = False
    for inst in items:
        if inst is signalInst:
            signalSeen = True
            waitPlaced = False
        if signalSeen and not waitPlaced and isinstance(inst, BranchInstruction):
            for w in waitItems:
                patched.add(w)
            waitPlaced = True
            signalSeen = False
        patched.add(inst)
    # Trailing wait for the last signal if no exit branch followed it.
    if signalSeen and not waitPlaced:
        for w in waitItems:
            patched.add(w)
    return patched
