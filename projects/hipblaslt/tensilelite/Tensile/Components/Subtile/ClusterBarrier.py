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

# Overlap budget: how many MFMAs should follow the signal before the wait.
# At least 16, or ~5% of total MFMAs for large tiles.
_MIN_MFMAS_AFTER_SIGNAL = 16
_MFMAS_AFTER_SIGNAL_DIVISOR = 20

_isWgBarrier = lambda x: isinstance(x, SBarrier) and "s_barrier_wait -1" in str(x)
_isMFMA = lambda x: isinstance(x, (MFMAInstruction, MXMFMAInstruction))


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

    No-op unless ``ClusterBarrier`` is enabled.  The signal is placed near the
    end of the MFMA stream — only a small overlap budget of MFMAs separates it
    from the trailing wait.  The signal must still follow the workgroup barrier
    (LDS-write visibility); when the computed target falls before the barrier
    the signal is clamped to the first MFMA after it.

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

    items = module.flatitems()

    # Pre-scan: locate MFMAs and the workgroup barrier so we can pick a
    # target MFMA near the end of the stream.
    mfma_positions = [i for i, inst in enumerate(items) if _isMFMA(inst)]
    wg_barrier_pos = next(
        (i for i, inst in enumerate(items) if _isWgBarrier(inst)), None)

    total_mfmas = len(mfma_positions)
    overlap = max(_MIN_MFMAS_AFTER_SIGNAL, total_mfmas // _MFMAS_AFTER_SIGNAL_DIVISOR)

    # Pick the target MFMA: ``overlap`` positions from the end, clamped to
    # the first MFMA after the workgroup barrier.
    targetMfmaIdx = None
    if total_mfmas > 0 and wg_barrier_pos is not None:
        target_rank = max(0, total_mfmas - overlap - 1)
        first_after_barrier = next(
            (j for j, p in enumerate(mfma_positions) if p > wg_barrier_pos),
            None)
        if first_after_barrier is not None:
            target_rank = max(target_rank, first_after_barrier)
            targetMfmaIdx = mfma_positions[target_rank]

    # Place the wave-0-election branch right after the target WMMA to hide
    # branching latency: keep s_cmp before the scheduled MFMA and emit the
    # branch after it.

    result = Module(module.name)
    done = False
    for i, inst in enumerate(items):
        if not done and targetMfmaIdx is not None and i == targetMfmaIdx:
            done = True
            # Split the signal block at the wave-0 election branch. The
            # block is authored with exactly one conditional branch; assert
            # it so a future change that adds another fails loudly here.
            brIdxs = [k for k, s in enumerate(signalItems)
                      if isinstance(s, SCBranchSCC0)]
            assert len(brIdxs) == 1, \
                "signal block must contain exactly one wave-0 election branch"
            brIdx = brIdxs[0]
            pre, post = signalItems[:brIdx], signalItems[brIdx:]
            # Emit s_cmp (pre), then the target MFMA, then branch+signal
            # (post).  SCC survives the MFMA, so the branch reads the
            # correct comparison result.
            for s in pre:
                result.add(s)
            result.add(inst)
            for s in post:
                result.add(s)
        else:
            result.add(inst)
    if not done:
        if wg_barrier_pos is not None:
            # WG barrier exists but no MFMA follows it: emit the signal
            # block intact right after the barrier (best-effort).
            rebuilt = Module(module.name)
            for i2, inst2 in enumerate(items):
                rebuilt.add(inst2)
                if i2 == wg_barrier_pos:
                    for s in signalItems:
                        rebuilt.add(s)
            result = rebuilt
        else:
            # No workgroup barrier at all: open the handshake at the start.
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
