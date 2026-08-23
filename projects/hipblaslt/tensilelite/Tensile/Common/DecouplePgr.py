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
    K = 2*DepthU, or does not build at all. That threshold is derived rather
    than sampled: two trips through the unrolled loop are needed before a fill
    can overwrite a block a previous trip is still reading, so it is
    deterministic by construction. It is also the threshold an independent
    measurement lands on -- the one-block-both FFM sweep cited at the reject in
    Solution.depthUIteration is clean at K=992 and wrong at K=1024 with
    DepthU 512, exactly 2*DepthU -- so prediction and measurement agree here.
    Do not read it as the same kind of claim as the SIA4 StinkyTofu barrier
    defect, which is a race, has no K threshold at all, and whose empirical
    thresholds were withdrawn because the same tree failed then passed at the
    same K.

    Assumes the pair is divergent and both tensors are on the TDM, which the
    caller has already established. Returns the clause its reject frames, so the
    message stays with the reject. Here rather than inline in that reject so a
    unit test can reach it without standing up an entire solution-derivation
    pipeline.
    """
    _, numLdsBlkA, numLdsBlkB = decouplePgrBlocks(ks)
    if max(numLdsBlkA, numLdsBlkB) > 2:
        return "more than two LDS blocks for a tensor is not supported"
    # The derived key, not the one the user wrote. ScheduleIterAlg=4 is remapped
    # in assignProblemIndependentDerivedParameters to _ScheduleIterAlg=0 plus
    # _StinkyTofuOptLevel=3, so its unrolled loop is scheduled exactly as SIA0
    # and the sub-iteration the relocated fill needs is present. Reading the raw
    # key refused those solutions for a scheduler they never run; HalfPLR
    # already rejects on this same derived key for this same reason. 1, 2 and 3
    # are different schedulers and the reject stays load-bearing for them.
    if ks["_ScheduleIterAlg"] != 0:
        return ("only ScheduleIterAlg=0 places the fill where it can be moved, "
                "and ScheduleIterAlg=4 derives to it")
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
    # _ScheduleIterAlg=0 above already excludes the _ScheduleIterAlg == 2 arm of
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

    Selected only by TDMFuse=6 (never derived; 0 stays inert). Equal pairs
    spend the extra SGPRs without using the cadence split. MXSA/MXSB stay
    parity-aliased. TDMSplit keeps the shared descriptor.
    """
    if ks.get("TDMFuse") != 6:
        return False
    if not tdmBothTensors(ks):
        return False
    # Unreachable while upstream's temporary blanket TDMSplit reject stands
    # (97e1223a3f9, PR #10911): no solution carrying TDMSplit=True now reaches
    # any writer predicate. Kept so this row's exclusion outlives that reject.
    if ks.get("TDMSplit"):
        return False
    return ks.get("NumWaves", 1) > 1 and not ks.get("UseSubtileImpl")


def tdmFuseAMx(ks):
    """True when {A,MXSA,MXSB} share one TDM descriptor set and B owns its own.

    TDMFuse=2, the grouping the manual reference kernel uses. Two sets of 4+8 is
    the same 24 SGPRs the default {A,B}+{MXSA,MXSB} pairing already spends.

    Selected only by TDMFuse=2, never derived, so 0 stays inert.

    NumWaves == 4 exactly. The dispatch is 1/1/2 -- one wave for MXSA, one for
    MXSB, two for A -- and A's share is numWaves - 2, which is a power of two
    only at 4. 4 does not divide by 3, so the 1/1/2 split is the remainder
    policy for three group members rather than an even partition, and there is
    no arithmetic that generalises it to another wave count.

    TDMSplit is excluded for the same reason it is excluded from tdmDealiasAB:
    its multi-wave increment recomputes one parity-selected split stride for one
    shared descriptor, and this row has no parity pairing left to select on.
    """
    if ks.get("TDMFuse") != 2:
        return False
    if not tdmBothTensors(ks):
        return False
    # Unreachable while upstream's temporary blanket TDMSplit reject stands
    # (97e1223a3f9, PR #10911): no solution carrying TDMSplit=True now reaches
    # any writer predicate. Kept so this row's exclusion outlives that reject.
    if ks.get("TDMSplit"):
        return False
    if not (ks["ProblemType"].get("MXBlockA") and ks["ProblemType"].get("MXBlockB")):
        return False
    return ks.get("NumWaves", 1) == 4 and not ks.get("UseSubtileImpl")


def tdmFusePaired(ks):
    """True when {MXSA,A} and {MXSB,B} each share one TDM descriptor set.

    TDMFuse=5. Two sets of 4+8 is the same 24 SGPRs the default {A,B}+{MXSA,MXSB}
    pairing already spends.

    Selected only by TDMFuse=5, never derived, so 0 stays inert.

    The wave division is crossed -- A and MXSB on the even waves, MXSA and B on
    the odd (see tdmWavePartition) -- so A and B keep the parity the default
    pairing gives them and their global-address arithmetic does not move.

    Any power-of-two NumWaves above one works: both sets take the same two-way
    parity split every pre-existing row uses, so unlike tdmFuseAMx there is no
    remainder policy to pin the wave count.

    TDMSplit is excluded for the same reason it is excluded from tdmDealiasAB and
    tdmFuseAMx: its multi-wave increment recomputes one parity-selected split
    stride for one shared descriptor, and this row has two.
    """
    if ks.get("TDMFuse") != 5:
        return False
    if not tdmBothTensors(ks):
        return False
    # Unreachable while upstream's temporary blanket TDMSplit reject stands
    # (97e1223a3f9, PR #10911): no solution carrying TDMSplit=True now reaches
    # any writer predicate. Kept so this row's exclusion outlives that reject.
    if ks.get("TDMSplit"):
        return False
    if not (ks["ProblemType"].get("MXBlockA") and ks["ProblemType"].get("MXBlockB")):
        return False
    return ks.get("NumWaves", 1) > 1 and not ks.get("UseSubtileImpl")


def tdmWavePartition(ks, tc):
    """(numComp, waves) for tensor `tc` on the wave-separated TDM path.

    THE single source of truth for how a tensor is divided across waves. numComp
    is how many waves divide it between them; waves is the tuple of wave indices
    that actually move it, so the offset arithmetic and the dispatch guard are
    derived from one statement rather than agreeing by coincidence. A tensor's
    component id is its position within `waves`.

    The default is the two-way parity split every pre-existing row uses: numComp
    = numWaves // 2 with component id waveIdx >> 1, so waves {0,2,..} take the
    A/MXSA arm and {1,3,..} the B/MXSB arm.
    """
    numWaves = ks.get("NumWaves", 1)
    if tdmFuseAMx(ks):
        # A on the low two waves, one scale tensor each on the top two. B is alone in
        # its set, so it has no partner to divide against and every wave carries it.
        if tc.endswith("A") and not tc.startswith("MX"):
            return 2, (0, 1)
        if tc == "MXSA":
            return 1, (2,)
        if tc == "MXSB":
            return 1, (3,)
        if tc.endswith("B"):
            return numWaves, tuple(range(numWaves))
    if tdmFusePaired(ks):
        # Crossed: each set keeps its data tensor on the parity the default
        # pairing gives it and takes the OTHER tensor's scales on the other
        # parity, so A and B divide exactly as they do at TDMFuse=0.
        isEvenArm = tc in ("A", "MXSB")
        return numWaves // 2, tuple(
            w for w in range(numWaves) if (w % 2 == 0) == isEvenArm)
    numComp = numWaves // 2
    isAArm = tc.endswith("A")
    return numComp, tuple(w for w in range(numWaves) if (w % 2 == 0) == isAArm)


def tdmWaveCompIdMode(ks, tc):
    """How a tensor's TDM component id is derived from the wave index.

    Both the global-address arithmetic (Components/TensorDataMover) and the LDS
    arithmetic (KernelWriterAssembly.initTDMDescriptorWaveSeparatedImpl) have to
    pick the same component id for the same tensor, or a wave reads one tile from
    memory and writes a different tile's LDS slot -- which validates or not by
    luck rather than failing to build. They read this.

      "zero"    -- one wave carries the whole tensor, so its component id is 0
      "waveIdx" -- component id IS the wave index; participants are waves
                   0..numComp-1, which is how an uneven split addresses itself
      "parity"  -- component id is waveIdx >> 1, two waves per component; the
                   two-way split every pre-existing row uses
    """
    numComp, waves = tdmWavePartition(ks, tc)
    if numComp == 1:
        return "zero"
    if waves == tuple(range(numComp)):
        return "waveIdx"
    return "parity"


def tdmWaveComponents(ks, tc):
    """(numComp, compShift) for tensor `tc` -- the shift form of the partition.

    A thin adapter over tdmWavePartition/tdmWaveCompIdMode so that the
    global-address side, the LDS side, both tail resets and the solution-level
    divisibility guard all derive their component count and component id from
    ONE function. compShift is the right shift applied to WaveIdx: 0 means the
    component id IS the wave index, and None means one wave carries the tensor,
    so the id is a constant zero and no register arithmetic is emitted.

    Keeping the shift form is what makes TDMFuse=2 correct on the global-address
    side for free: that side already asks this question per tensor, so a new
    grouping answers it without a new call site to keep in step.
    """
    numComp, _waves = tdmWavePartition(ks, tc)
    mode = tdmWaveCompIdMode(ks, tc)
    return numComp, {"zero": None, "waveIdx": 0, "parity": 1}[mode]


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
