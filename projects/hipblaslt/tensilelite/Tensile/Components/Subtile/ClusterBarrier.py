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
from rocisa.instruction import SBarrier, SCBranchSCC0, SCmpEQU32

_isWgBarrier = lambda x: isinstance(x, SBarrier) and "s_barrier_wait -1" in str(x)


def subtileClusterBarrierSignal(writer, kernel, label="") -> Module:
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


def subtileClusterBarrierWait(writer, kernel, label="") -> Module:
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

    # Signal: immediately after the workgroup wait.
    result = Module(module.name)
    done = False
    for inst in module.flatitems():
        result.add(inst)
        if not done and _isWgBarrier(inst):
            for s in signalItems:
                result.add(s)
            done = True
    if not done:  # no workgroup barrier: open the handshake at the start
        head = Module(module.name)
        for s in signalItems:
            head.add(s)
        for inst in result.flatitems():
            head.add(inst)
        result = head

    # Wait: append at the end of the section so cluster latency hides behind the
    # whole macro tile's WMMAs before the handshake is closed.
    for w in waitItems:
        result.add(w)
    return result
