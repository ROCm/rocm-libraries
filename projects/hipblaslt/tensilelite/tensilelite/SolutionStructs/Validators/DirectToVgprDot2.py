# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Validator for ``DirectToVgprDot2A``: dot2 WaveSplitK kernels that feed A
from the global-read registers instead of staging it through LDS.

Each WaveSplitK lane owns one contiguous block of DepthU/NumWaveSplitK
elements of K. A single A global load covers exactly that block, so the
loaded (and, for int4, dequantized) registers are the v_dot2 operands; B is
read from LDS over the same block.

Public entry point: :func:`validateDirectToVgprDot2A`.
"""

from ..Utilities import reject


#: Name prefix of A's local-write items under DirectToVgprDot2A. They contain no
#: LDS store, but each consumes one global load and needs the matching vmcnt wait.
DOT2_DTVA_ITEM = "Dot2DtvA "


def countDot2DtvAItem(item):
    """1 for an A dequant-only local-write item, else 0."""
    return 1 if getattr(item, "name", "").startswith(DOT2_DTVA_ITEM) else 0


def dot2DtvASlotPipelined(kernel):
    """Without PrefetchGlobalRead the A loads of an iteration are consumed in the
    order they were issued: each lane's NumLoadsCoalescedA loads ("slots") are
    dequantized and multiplied one after the other, each behind its own vmcnt."""
    return kernel.get("DirectToVgprDot2A", False) and kernel["PrefetchGlobalRead"] == 0


def dot2DtvADoubleBuffered(kernel):
    """With PrefetchGlobalRead the next A load lands while the current v_dot2 still
    read A, so the A registers alternate between two sets (G2LA, G2LA2)."""
    return kernel.get("DirectToVgprDot2A", False) and kernel["PrefetchGlobalRead"] >= 1


def validateDirectToVgprDot2A(state, printRejectionReason):
    """Return ``True`` when the mode is off or supported, ``False`` after a reject.

    Must run after ``GlobalReadVectorWidthA``, ``NumLoadsCoalescedA`` and the
    dot2 parameters are derived.
    """
    if not state["DirectToVgprDot2A"]:
        return True

    def fail(msg):
        reject(state, printRejectionReason, "DirectToVgprDot2A: " + msg)
        return False

    if not (state["UseDotInstruction"] and state["WaveSplitK"]):
        return fail("requires a dot2 WaveSplitK kernel")
    if state["DirectToVgprA"] or state["DirectToLdsA"]:
        return fail("is exclusive with DirectToVgprA/DirectToLdsA")
    if state["ProblemType"]["TLUA"]:
        return fail("requires A K-contiguous (TLUA=False)")
    # Global-read row t / NumWaveSplitK must be the dot2 lane's row.
    if state["WorkGroup"][1] != 1:
        return fail("requires WorkGroup[1] == 1")
    # Lane kidx loads, per row, NumLoadsCoalescedA blocks of GlobalReadVectorWidthA
    # elements at K = kidx*GRVWA + slot*NumWaveSplitK*GRVWA; B is read at the same K.
    if state["DepthU"] != (state["GlobalReadVectorWidthA"] * state["NumWaveSplitK"]
                           * state["NumLoadsCoalescedA"]):
        return fail("requires DepthU (%d) == GlobalReadVectorWidthA (%d) * NumWaveSplitK (%d)"
                    " * NumLoadsCoalescedA (%d)"
                    % (state["DepthU"], state["GlobalReadVectorWidthA"], state["NumWaveSplitK"],
                       state["NumLoadsCoalescedA"]))
    if state["InnerUnroll"] != 1:
        return fail("requires InnerUnroll == 1")
    # Two A register sets cover one load in flight; a third would be needed for PGR2.
    if state["PrefetchGlobalRead"] > 1:
        return fail("requires PrefetchGlobalRead <= 1")
    # The tail loop reads A through LDS.
    if state["AssertSummationElementMultiple"] % state["DepthU"] != 0:
        return fail("requires AssertSummationElementMultiple to be a multiple of DepthU")
    return validateDirectToVgprDot2B(state, printRejectionReason)


def validateDirectToVgprDot2B(state, printRejectionReason):
    """B bypasses LDS when its global read already gives every lane the B values
    of its own A K slots: one wave per workgroup and the same load geometry."""
    if not state["DirectToVgprDot2B"]:
        return True

    def fail(msg):
        reject(state, printRejectionReason, "DirectToVgprDot2B: " + msg)
        return False

    if not state["DirectToVgprDot2A"]:
        return fail("requires DirectToVgprDot2A")
    if state["ProblemType"]["TLUB"]:
        return fail("requires B K-contiguous (TLUB=False)")
    # Each wave needs all of B for its K blocks; the global read spreads B over the workgroup.
    if state["WorkGroup"][0] != 1 or state["NumThreads"] != state["NumWaveSplitK"]:
        return fail("requires one row group per workgroup (WorkGroup[0] == 1)")
    if state["GlobalReadVectorWidthB"] != state["GlobalReadVectorWidthA"] \
       or state["NumLoadsCoalescedB"] != state["NumLoadsCoalescedA"]:
        return fail("requires B global loads shaped like A's (GRVW and NumLoadsCoalesced)")
    if state["PrefetchGlobalRead"] != 0:
        return fail("requires PrefetchGlobalRead == 0")
    return True
