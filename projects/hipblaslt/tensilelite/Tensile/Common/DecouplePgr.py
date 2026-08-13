################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Semantics of PrefetchGlobalReadA/B, the per-tensor LDS block counts.

In Common rather than beside the derivation in SolutionStructs.Solution because
Components.SIA needs these too, and SolutionStructs imports Component which
imports Components, so anything SIA reaches for has to live below both.
"""


def pgrLevelsForTensors(ks):
    """(decoupled, pgrA, pgrB) -- the per-tensor levels, scalar-filled.

    An absent key is the "not specified" sentinel and falls back to the scalar
    PrefetchGlobalRead, which is why 0 is free to be a real value.
    """
    pgr = ks.get("PrefetchGlobalRead", 0)
    pgrA = ks.get("PrefetchGlobalReadA")
    pgrB = ks.get("PrefetchGlobalReadB")
    if pgrA is None and pgrB is None:
        return False, pgr, pgr
    return True, pgr if pgrA is None else pgrA, pgr if pgrB is None else pgrB


def ldsBlocksForPgrLevel(pgr):
    """LDS blocks one per-tensor level allocates.

    Level 1 is the rung that does not agree with the scalar, which allocates TWO
    blocks for PrefetchGlobalRead=1: the buffer_load path also holds a VGPR
    staging buffer and sizes LDS against a three-buffer pipeline. Under TDM and
    DirectToLds there is no VGPR buffer, so N blocks is what depth N needs.
    """
    if pgr <= 1:
        return 1
    return 2 if pgr == 2 else pgr


def tdmBothTensors(ks):
    """True when the TDM moves both tensors, which the per-tensor levels need.

    TDMInst is a per-tensor bitmask, bit 0 for A and bit 1 for B, read that way
    rather than compared against 3 so this does not depend on the separate
    reject that pins the parameter to 0 or 3.

    A block count is a prefetch depth only where nothing stages the tile in
    VGPRs first, which is why this is the precondition for the whole feature and
    not just for the shapes that misbuild without it.
    """
    tdmInst = ks.get("TDMInst", 0)
    return bool(tdmInst & 0x01) and bool(tdmInst & 0x02)


def decouplePgrBlocks(ks):
    """(decoupled, numLdsBlkA, numLdsBlkB).

    Derived on demand rather than stored in solution state, so nothing derived
    reaches the serialized library.
    """
    decoupled, pgrA, pgrB = pgrLevelsForTensors(ks)
    return decoupled, ldsBlocksForPgrLevel(pgrA), ldsBlocksForPgrLevel(pgrB)


def equalPairDegeneratesToScalar(ks):
    """True when an equal per-tensor pair is exactly its legacy scalar spelling.

    One block count on both tensors is the legacy PrefetchGlobalRead=k layout,
    loop and kernel, so resolving the pair away lets everything downstream see an
    ordinary scalar solution and keeps the A/B asynchrony machinery out of a case
    that never uses it.

    (1,1) is excluded and stays decoupled: its only byte-identical legacy
    spelling sets 1LDSBuffer, which this path must not do implicitly.

    An explicit 1LDSBuffer=1 also blocks it. Resolving the pair away deletes both
    keys, and a deleted key cannot be compared against the one shared LDS block
    1LDSBuffer asks for, so the pair has to stay alive long enough for the reject
    in Solution.depthUIteration to see the contradiction -- otherwise a pair
    asking for two blocks quietly builds one. Only an explicit 1 blocks it: an
    unresolved -1 is the auto rule, and getting the auto rule's answer is part of
    what "this is legacy PrefetchGlobalRead=k exactly" means.
    """
    decoupled, pgrA, pgrB = pgrLevelsForTensors(ks)
    if not (decoupled and pgrA == pgrB and pgrA != 1):
        return False
    if ks.get("1LDSBuffer") == 1 and ldsBlocksForPgrLevel(pgrA) > 1:
        return False
    return True


def divergentPairUnsupportedReason(ks):
    """Why a divergent pair has nowhere to put the single-buffered fill, or None.

    A divergent pair is legal only because
    KernelWriter._dcpScheduleSingleBufferedFillLate can move the single-buffered
    tensor's fill into a sub-iteration between the last local read of its block
    and the pre-read sync. These are the conditions under which that slot exists
    and can be reached; outside them the kernel computes wrong results from
    K = 2*DepthU, or does not build at all.

    Assumes the pair is divergent and both tensors are on the TDM, which the
    caller has already established. Returns the clause its reject frames, so the
    message stays with the reject. Here rather than inline in that reject so a
    unit test can reach it without standing up an entire solution-derivation
    pipeline.
    """
    _, numLdsBlkA, numLdsBlkB = decouplePgrBlocks(ks)
    if max(numLdsBlkA, numLdsBlkB) > 2:
        return "more than two LDS blocks for a tensor is not supported"
    if ks["ScheduleIterAlg"] != 0:
        return "only ScheduleIterAlg=0 places the fill where it can be moved"
    if ks["PrefetchLocalRead"] < 1:
        return ("PrefetchLocalRead must be at least 1 so a sub-iteration exists "
                "between the last local read and the pre-read sync")
    # PrefetchLocalRead >= LoopIters reaches that same state by another route:
    # Solution.assignDerivedParameters rewrites PrefetchLocalRead to 0 when
    # ClusterLocalRead is set, and that rewrite runs after this call. So the
    # clause above sees the value the user wrote, passes it, and the emitter
    # then meets the 0. Supplying 0 rejects; arriving at 0 rewrote and asserted.
    #
    # LoopIters is recomputed rather than read because state["LoopIters"] is
    # assigned after this call -- absent on the first DepthU tried, and stale on
    # every one after. This mirrors that derivation.
    #
    # ScheduleIterAlg=0 above already excludes the _ScheduleIterAlg == 2 arm of
    # the rewrite's condition, so only the other two are re-tested here.
    #
    # The block-count clause at the top of this function currently hides this
    # for aggressive pairs: (1,4) and (4,1) at DepthU 128 also have
    # PrefetchLocalRead >= LoopIters, and are rejected there before reaching
    # here. Relaxing that clause without keeping this one widens the assertion.
    loopIters = ks["DepthU"] // ks["LocalSplitU"] // ks["InnerUnroll"]
    if ks.get("EnableMatrixInstruction", True):
        loopIters //= ks["MatrixInstK"]
    if (ks["PrefetchLocalRead"] >= loopIters
            and ks.get("ClusterLocalRead", 1)
            and not ks.get("ForceUnrollSubIter", False)):
        return ("PrefetchLocalRead=%u is not below LoopIters=%u, and is rewritten to 0 "
                "after this check, leaving no sub-iteration between the last local read "
                "and the pre-read sync" % (ks["PrefetchLocalRead"], loopIters))
    # The relocated fill and the one it replaces are emitted under complementary
    # wave-parity guards, and parity only selects a tensor on the wave-separated
    # descriptor -- KernelWriterAssembly.isTdmWaveSeparated, which is both
    # tensors on the TDM AND more than one wave. The caller requires the first.
    # Nothing required the second, so a one-wave divergent pair used to pass
    # validation and then die on that emitter's assertion.
    if ks["NumWaves"] <= 1:
        return ("the fill is re-slotted under a wave-parity guard, which needs the "
                "wave-separated TDM descriptor (NumWaves > 1); this solution has "
                "NumWaves=%u" % ks["NumWaves"])
    return None


def decoupledSingleBuffered(ks):
    """True when exactly one tensor is left on a single LDS block.

    That tensor has nowhere to put its next tile except on top of the copy the
    current iteration is still reading, so the write-after-read barriers have to
    fire even though the scalar PrefetchGlobalRead is nonzero.
    """
    decoupled, numLdsBlkA, numLdsBlkB = decouplePgrBlocks(ks)
    return decoupled and min(numLdsBlkA, numLdsBlkB) == 1 and max(numLdsBlkA, numLdsBlkB) > 1


def tdmDealiasAB(ks):
    """True when A and B get their own TDM descriptor sets instead of sharing one.

    Costs 12 SGPRs -- Group0 is 4 and Group1 is 8, the fixed tuple widths of
    tensor_load_to_lds -- against an architectural ceiling of 106.

    Never derived, only selected by TDMFuse=6, so that 0 stays inert. Equal
    block counts keep the alias, because their byte-identity with a legacy
    configuration is the evidence the feature rests on. MXSA/MXSB stay
    parity-aliased; de-aliasing all four costs another 24 SGPRs. TDMSplit keeps
    the alias, because its multi-wave increment recomputes one parity-selected
    split stride for one shared descriptor.
    """
    if ks.get("TDMFuse") != 6:
        return False
    decoupled, numLdsBlkA, numLdsBlkB = decouplePgrBlocks(ks)
    if not (decoupled and numLdsBlkA != numLdsBlkB):
        return False
    if not tdmBothTensors(ks):
        return False
    if ks.get("TDMSplit"):
        return False
    return ks.get("NumWaves", 1) > 1 and not ks.get("UseSubtileImpl")


def decoupledOneBlockBoth(ks):
    """True when both tensors are on a single LDS block inside a prefetch loop.

    Same emit shape as 1LDSBuffer=1 but reached from the per-tensor block counts,
    which is what lets it exist at every ScheduleIterAlg -- 1LDSBuffer=1 is
    rejected outside SIA 2 and 3 -- and what avoids needing a per-tensor
    1LDSBufferA/B alongside PrefetchGlobalReadA/B.

    PrefetchGlobalRead must be nonzero: at level 0 the no-prefetch branch already
    allocates a single block and NumLdsBlk stays at the 2 that legacy
    PrefetchGlobalRead=0 also reports.
    """
    decoupled, numLdsBlkA, numLdsBlkB = decouplePgrBlocks(ks)
    return decoupled and max(numLdsBlkA, numLdsBlkB) == 1 and bool(ks["PrefetchGlobalRead"])
