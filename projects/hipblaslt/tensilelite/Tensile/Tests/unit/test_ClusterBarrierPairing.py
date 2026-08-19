# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression tests for cluster-barrier signal/wait pairing.

The cluster-scope barrier handshake is split into a ``signal`` half and a
``wait`` half so the cross-CU latency can hide behind intervening WMMAs. The
wait must nonetheless dominate every early-exit branch (``SkipToNLL``,
``SkipToNGLL``, ``SkipTailLoopL`` etc.) that follows the signal; otherwise a
taken branch leaves the barrier signalled but never waited, desyncing the
cluster handshake (FFM assertion / incorrect on hardware).

These tests exercise ``insertClusterBarrier`` directly with a mock writer so
they stay fast and hardware-free.
"""

from unittest.mock import MagicMock

from rocisa.code import Module
from rocisa.container import sgpr
from rocisa.instruction import SBarrier, SCBranchSCC1, SCmpEQU32

from Tensile.Components.Subtile.ClusterBarrier import insertClusterBarrier


def _mockWriter():
    writer = MagicMock()
    # ClusterBarrier requires the HasClusterBarrier asm capability.
    writer.states.asmCaps.get.return_value = True
    writer.labels.getUniqueNamePrefix.side_effect = lambda prefix: prefix + "_TEST"
    return writer


def _kernel():
    return {"ClusterBarrier": True}


def _isSignal(inst):
    return isinstance(inst, SBarrier) and "cluster_barrier signal" in str(inst)


def _isClusterWait(inst):
    return (isinstance(inst, SBarrier)
            and "cluster_barrier wait" in str(inst)
            and "workgroup" not in str(inst))


def _indices(items):
    sig = next(i for i, x in enumerate(items) if _isSignal(x))
    wait = next(i for i, x in enumerate(items) if _isClusterWait(x))
    return sig, wait


def test_wait_dominates_early_exit_branch():
    """The cluster wait must sit after the signal but before the first exit."""
    module = Module("section")
    module.add(SCmpEQU32(sgpr("LoopCounterL"), 1, "LoopCounter LE 1?"))
    module.add(SCBranchSCC1("label_SkipToNLL", "skip to NLL"))
    module.add(SCmpEQU32(sgpr("LoopCounterL"), 2, "LoopCounter LE 2?"))
    module.add(SCBranchSCC1("label_SkipToNGLL", "skip to NGLL"))

    items = insertClusterBarrier(module, _mockWriter(), _kernel()).flatitems()
    sig, wait = _indices(items)
    firstExit = next(i for i, x in enumerate(items) if isinstance(x, SCBranchSCC1))

    assert sig < wait < firstExit, (
        "cluster wait must follow the signal and precede the first early-exit "
        "branch so no taken branch can skip it"
    )


def test_wait_never_placed_after_exit_branch():
    """Anti-regression: no early-exit branch may precede the cluster wait."""
    module = Module("section")
    module.add(SCmpEQU32(sgpr("LoopCounterL"), 32, "LoopCounter LE 32?"))
    module.add(SCBranchSCC1("label_SkipTailLoopL", "early-exit tail"))

    items = insertClusterBarrier(module, _mockWriter(), _kernel()).flatitems()
    _, wait = _indices(items)

    exitsBeforeWait = [i for i, x in enumerate(items)
                       if isinstance(x, SCBranchSCC1) and i < wait]
    assert not exitsBeforeWait, (
        "an early-exit branch precedes the cluster wait; the barrier can be "
        "signalled but never waited on that path"
    )
