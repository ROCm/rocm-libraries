# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TDM descriptor grouping (TDMFuse) and the wave assignment it selects.

Two separate questions live here and must not be confused:

  grouping        which tensors share one descriptor set, i.e. ride one fused
                  `tensor_load_to_lds`.  Selected by TDMFuse.
  wave assignment which wave issues which member of a group.  Selected by
                  TDMCross.

TDMFuse=0  {A,B} + {MXSA,MXSB}   default two-way parity
TDMFuse=1  {A,MXSA} + {MXSB,B}   crossed parity (tdmFusePaired)
TDMFuse=2  {A,MXSA,MXSB} + {B}   1/1/2 remainder split, NumWaves==4 (tdmFuseAMx)

The groupings are rows of TDM_GROUPS below, keyed on name.  The TDMFuse integer
is a stable-but-arbitrary index into that table: our shipped values are baked
into solution names and cannot move, so a name is the only stable key.
"""

import os
from typing import NamedTuple

from .DecouplePGR import (decouplePGRBlocks, dcpThickThinIssueOrder,
                          ldsAlignedBytesForTensor)

# The four tensors a TDM descriptor group can carry. Sparse metadata rides
# tdmMetadataGroup0, which no grouping names, and is excluded everywhere.
TDM_TENSORS = ("A", "MXSA", "MXSB", "B")

# TDMCross values. Int enum, not a bool: a grouping with more than two
# partitioned groups admits more than one non-trivial arrangement, and growing
# a bool later would churn every solution name.
TDM_CROSS_DEFAULT = 0
TDM_CROSS_CROSSED = 1


class TdmGrouping:
    """One row of the grouping table.

    groups     tuple of member tuples. Each group is one descriptor set, i.e.
               one fused `tensor_load_to_lds` whose members are dispatched
               across waves. Member order IS the wave order (see `layout`).
    layout     "parity" splits a k-member group by wave index modulo k, so a
               two-member group lands member 0 on the even waves and member 1
               on the odd. "block" splits the waves into k contiguous shares,
               remainder to the leading members, which is the 1/1/2 policy
               TDMFuse=2 names.
    index      the historical integer. Arbitrary and non-contiguous with
               anybody else's numbering; never key on it.
    """

    __slots__ = ("name", "index", "groups", "layout")

    def __init__(self, name, index, groups, layout="parity"):
        self.name = name
        self.index = index
        self.groups = groups
        self.layout = layout


# Declarative grouping table, keyed on name.  Nothing below branches on a
# specific name or index, so adding a grouping is adding a row plus one entry in
# TDM_FUSE_GROUPING pointing an integer at it -- no new rejection, no new wave
# rule, no new naming case.
#
#   name     groups (each = one fused tensor_load_to_lds)
#   None     none -- every part loads on its own instruction
#   AB       {A,B}; scales, if any, unfused
#   A_MX     {A,MXSA,MXSB} on one; B (and its scale share) separate
#   B_MX     {B,MXSA,MXSB} on one; A separate
#   MX_AB    {MXSA,MXSB} and {A,B}
#   paired   {MXSA,A} and {MXSB,B} -- each scale rides its own data
#
# Member order inside a group is the wave order, so it records the arrangement
# as shipped, not the order the grouping is spoken aloud in.  "paired" is
# written (A, MXSA) + (MXSB, B) because that is the arrangement TDMFuse=1
# emits -- the same grouping the parallel branch writes MXSA/A + MXSB/B, laid
# onto waves already crossed, which is what its "crossed parity" name records.
TDM_GROUPS = {
    "MX_AB": TdmGrouping("MX_AB", 4, (("A", "B"), ("MXSA", "MXSB"))),
    "paired": TdmGrouping("paired", 5, (("A", "MXSA"), ("MXSB", "B"))),
    "A_MX": TdmGrouping("A_MX", 2, (("A", "MXSA", "MXSB"), ("B",)), layout="block"),
    # Rows no TDMFuse integer selects yet. They are data, and they already get
    # the right answers out of every function here -- B_MX in particular
    # inherits A_MX's "nothing to cross" rejection mirrored, for free, because
    # the rule counts partitioned groups rather than testing a value.
    "B_MX": TdmGrouping("B_MX", 3, (("B", "MXSA", "MXSB"), ("A",)), layout="block"),
    "AB": TdmGrouping("AB", 1, (("A", "B"), ("MXSA",), ("MXSB",))),
    "None": TdmGrouping("None", 0, (("A",), ("B",), ("MXSA",), ("MXSB",))),
}

# Our TDMFuse integers, resolved to a row name. The integers are baked into
# shipped solution names, the frozen 00_Final.yaml and the silicon evidence, so
# they cannot move; the parallel branch numbers the same groupings differently.
# Keying the table on the name is what lets both numberings coexist.
TDM_FUSE_GROUPING = {0: "MX_AB", 1: "paired", 2: "A_MX"}


def tdmBothTensors(ks):
    """True when TDMInst moves both A and B (bits 0 and 1)."""
    tdmInst = ks.get("TDMInst", 0)
    return bool(tdmInst & 0x01) and bool(tdmInst & 0x02)


def _tdmFuseCanShareDescriptors(ks):
    """Guards shared by TDMFuse=1 and 2.

    TDMSplit is kept here even while Solution rejects it globally: writer
    predicates and unit tests still see TDMSplit=True kernels.
    """
    if not tdmBothTensors(ks):
        return False
    if ks.get("TDMSplit") or ks.get("UseSubtileImpl"):
        return False
    pt = ks.get("ProblemType") or {}
    return bool(pt.get("MXBlockA") and pt.get("MXBlockB"))


def tdmFuseAMx(ks):
    """TDMFuse=2: {A,MXSA,MXSB} share one set, B owns its own. NumWaves==4."""
    return (ks.get("TDMFuse") == 2
            and _tdmFuseCanShareDescriptors(ks)
            and ks.get("NumWaves", 1) == 4)


def tdmFusePaired(ks):
    """TDMFuse=1: {MXSA,A} and {MXSB,B}, crossed parity. NumWaves>1."""
    return (ks.get("TDMFuse") == 1
            and _tdmFuseCanShareDescriptors(ks)
            and ks.get("NumWaves", 1) > 1)


def tdmGrouping(ks):
    """The grouping row this solution actually gets.

    TDMFuse names a grouping, but tdmFuseAMx / tdmFusePaired can decline it --
    on TDMSplit, subtile, a missing scale, the wrong wave count. A declined
    grouping falls back to the default, exactly as the writer does.
    """
    if tdmFuseAMx(ks):
        return TDM_GROUPS["A_MX"]
    if tdmFusePaired(ks):
        return TDM_GROUPS["paired"]
    return TDM_GROUPS[TDM_FUSE_GROUPING[0]]


def tdmSeparateABDescriptors(ks):
    """True when the grouping puts A and B in different descriptor sets.

    Two tensors sharing one set ride one fused `tensor_load_to_lds` and so share
    one TDM tensor token: nothing can drain one without draining the other. Two
    sets can be handed disjoint tokens and drained independently. That is the
    only property the decoupled PGR thick gate turns on.

    Read off the resolved grouping row, never off the TDMFuse integer, so a
    grouping TDMFuse asked for but `_tdmFuseCanShareDescriptors` declined
    answers with the fallback the writer actually gets. Adding a row therefore
    needs no branch here: `A_MX` and `B_MX` separate A from B, `MX_AB` and `AB`
    do not, and `None` fuses nothing so every tensor is its own set.
    """
    return not any({"A", "B"} <= set(group) for group in tdmGrouping(ks).groups)


def tdmMemberIsLive(ks, tc):
    """True when tensor `tc` exists on this problem.

    A scale-less type has no MXSA/MXSB, so its groups degenerate: MX_AB's
    {A,B} + {MXSA,MXSB} becomes {A,B} alone. That is what the TDMFuse=2 reject
    already says in prose -- "without them the group is just {A}" -- and it is
    why no rejection here has to test the data type: the group count derives it.
    """
    if tc not in ("MXSA", "MXSB"):
        return True
    pt = ks.get("ProblemType") or {}
    return bool(pt.get("MXBlock%s" % tc[-1]))


def liveGroups(ks, grouping=None):
    """`grouping.groups` with dead members dropped and empty groups removed.

    Used to decide what can be crossed and to report it. The wave *layout* is
    deliberately computed from the full group instead (see _waveShares): a
    descriptor set keeps its shape when one member is absent, the absent member
    simply never issues. Sizing the shares off the live members would move the
    surviving one onto different waves.
    """
    grouping = grouping if grouping is not None else tdmGrouping(ks)
    live = []
    for group in grouping.groups:
        members = tuple(tc for tc in group if tdmMemberIsLive(ks, tc))
        if members:
            live.append(members)
    return tuple(live)


def partitionedGroups(ks, grouping=None):
    """Live groups that are actually split across waves (two or more members).

    A one-member group is issued by every wave, so it pins nothing: there is no
    choice of which wave carries it and therefore nothing to cross it against.
    """
    return tuple(g for g in liveGroups(ks, grouping) if len(g) > 1)


def _waveShares(numWaves, numMembers, layout):
    """(waves, numComp) for each member of a group of `numMembers`."""
    if numMembers <= 1:
        return ((tuple(range(numWaves)), numWaves),)
    if layout == "parity":
        # numComp is numWaves // numMembers, not the share length: at
        # NumWaves=1 a two-member group leaves the odd member on no wave at
        # all, and the shipped partition reports 0 components there.
        numComp = numWaves // numMembers
        return tuple(
            (tuple(w for w in range(numWaves) if w % numMembers == i), numComp)
            for i in range(numMembers))
    # "block": contiguous shares, remainder handed to the leading members.
    base, extra = divmod(numWaves, numMembers)
    shares, start = [], 0
    for i in range(numMembers):
        width = base + (1 if i < extra else 0)
        shares.append((tuple(range(start, start + width)), width))
        start += width
    return tuple(shares)


def _crossGroups(groups):
    """Reverse every partitioned group after the first.

    With two partitioned groups this is the one non-trivial arrangement: it
    pairs the first member of one group with the last member of the other, so
    each wave gets one large item and one small one instead of two of a kind.
    Global wave relabelling is not a distinct arrangement, which is why only
    the groups after the first are reversed.
    """
    crossed, seen = [], 0
    for group in groups:
        if len(group) > 1:
            seen += 1
            if seen > 1:
                group = tuple(reversed(group))
        crossed.append(group)
    return tuple(crossed)


def tdmCross(ks):
    """The requested wave arrangement. Missing key means default."""
    return ks.get("TDMCross", TDM_CROSS_DEFAULT) or TDM_CROSS_DEFAULT


def _arrangedGroups(ks):
    """The grouping's groups, in the member order TDMCross selected.

    Member order is the wave order, so this is the whole of what the parameter
    does: crossing rewrites nothing except the sequence a group's members are
    laid onto waves in.
    """
    groups = tdmGrouping(ks).groups
    if tdmCross(ks) == TDM_CROSS_DEFAULT:
        return groups
    return _crossGroups(groups)


def tdmWaveAssignment(ks):
    """Which tensors each wave issues, as {wave: (tc, ...)}.

    LAYER 1. The only consumer of TDMCross in the whole codebase; every
    other site reads the assignment (or tdmWavePartition, its per-tensor view)
    rather than the parameter, so a new arrangement cannot half-apply.

    Pure: depends on the grouping row, TDMCross, NumWaves and member
    liveness, and on nothing the writer mutates. The decoupled PGR pair reaches
    it through tdmWaveLdsBytes, which is where block counts matter.

    At TDMCross=0 this is the identity on the shipped wave partition --
    asserted directly in test_TDMCross.py rather than inferred from a diff.
    """
    numWaves = ks.get("NumWaves", 1)
    layout = tdmGrouping(ks).layout
    assignment = {w: [] for w in range(numWaves)}
    for group in _arrangedGroups(ks):
        for member, (waves, _) in zip(group, _waveShares(numWaves, len(group), layout)):
            if not tdmMemberIsLive(ks, member):
                continue
            for w in waves:
                assignment[w].append(member)
    return {w: tuple(members) for w, members in assignment.items()}


def tdmWavePartition(ks, tc):
    """(numComp, waves) for tensor `tc`.

    numComp is how many waves divide the tensor; waves is which wave indices
    actually move it. Component id is the index within `waves`.

    The per-tensor view of tdmWaveAssignment, so the descriptor init, the
    stagger gate and the fill guards all follow one arrangement automatically.
    """
    numWaves = ks.get("NumWaves", 1)
    layout = tdmGrouping(ks).layout
    for group in _arrangedGroups(ks):
        if tc in group:
            waves, numComp = _waveShares(numWaves, len(group), layout)[group.index(tc)]
            return numComp, waves
    # A tensor no row names: answer with the default two-way parity, which is
    # what every caller saw before the table existed.
    numComp = numWaves // 2
    isAArm = tc.endswith("A")
    return numComp, tuple(w for w in range(numWaves) if (w % 2 == 0) == isAArm)


def tdmGroupPartner(ks, tc, fallback):
    """The other member of `tc`'s descriptor set, or `fallback`.

    Only a two-member set has one unambiguous partner; a three-member set (the
    TDMFuse=2 shared group) does not, and the caller's own pair is the right
    answer there.
    """
    for group in tdmGrouping(ks).groups:
        if tc in group and len(group) == 2:
            return group[0] if group[1] == tc else group[1]
    return fallback


def tdmWaveComponents(ks, tc):
    """(numComp, compShift) for tensor `tc`.

    compShift is a right-shift on WaveIdx, or None when WaveIdx is not used:

      None  SMov  id, 0
      0     SMov  id, WaveIdx
      1     SLshr id, WaveIdx, 1

    None is the sentinel because 0 already means "shift by zero". TensorDataMover
    and KernelWriterAssembly both branch on these three values.
    """
    numComp, waves = tdmWavePartition(ks, tc)
    if numComp == 1:
        return numComp, None
    if waves == tuple(range(numComp)):
        return numComp, 0
    return numComp, 1


def tdmDataTensorsShareAWave(ks):
    """True when some wave issues both A and B.

    Only possible when one of them rides an unpartitioned group -- the
    TDMFuse=2 shape, where B is a one-member group and so is issued by every
    wave. When every group is partitioned, A and B sit on opposite waves under
    every arrangement the table can produce, crossed or not.
    """
    return any({"A", "B"} <= set(members)
               for members in tdmWaveAssignment(ks).values())


def tdmWaveIssueOrder(ks, itemA, itemB):
    """(first, second) of the two data tensors' items, thick tensor first.

    LAYER 3. Reads the wave assignment, weighs each data tensor by the LDS
    blocks it holds, and hands both to dcpThickThinIssueOrder.

    Weighing each tensor by its own block count rather than by its wave's total
    is deliberate. The two are the same number whenever A and B sit on separate
    waves, which tdmDataTensorsShareAWave shows is every arrangement except the
    TDMFuse=2 shape -- and that shape is equal-pair-only, so its two weights
    tie and the stable sort keeps A first. The per-wave question therefore has
    the same answer as the global one everywhere it is currently asked, which
    is why routing it through the assignment costs no change in emitted code.
    """
    _, numLdsBlkA, numLdsBlkB = decouplePGRBlocks(ks)
    return dcpThickThinIssueOrder(((numLdsBlkA, itemA), (numLdsBlkB, itemB)))


DCP_THICK_GATE_TOKENS = "tokens"
DCP_THICK_GATE_TEXT = "text"

# The count each mechanism's grouping actually supports. Not a tunable: it is
# how many independent tensor ops the grouping leaves outstanding.
DCP_THICK_GATE_SUPPORTED = {DCP_THICK_GATE_TEXT: 2, DCP_THICK_GATE_TOKENS: 1}


class DcpThickGate(NamedTuple):
    """How a divergent decoupled pair's thick-tensor gate gets relaxed."""
    mechanism: str
    tensorcnt: int


def _parseThickGateCountOverride(spec):
    """Parse the gate-pricing hook, e.g. "text=1" or "text=0,tokens=0".

    DOWNWARD ONLY, and this is where that is enforced. `s_wait_tensorcnt N`
    retires while N tensor ops are still outstanding, so lowering N is strictly
    more conservative -- at 0 it is a full drain, which is always correct -- but
    raising N above what the grouping supports lets LDS reads begin before their
    data has landed. An upward request is refused here rather than measured.
    """
    out = {}
    for item in (spec or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        mechanism, _, value = item.partition("=")
        mechanism = mechanism.strip()
        if mechanism not in DCP_THICK_GATE_SUPPORTED:
            raise ValueError(
                "TENSILE_DCP_THICK_GATE_COUNT=%r: unknown mechanism %r, expected "
                "one of %s" % (spec, mechanism, sorted(DCP_THICK_GATE_SUPPORTED)))
        count = int(value.strip())
        supported = DCP_THICK_GATE_SUPPORTED[mechanism]
        if not 0 <= count <= supported:
            raise ValueError(
                "TENSILE_DCP_THICK_GATE_COUNT=%r sweeps %s upward: %d is outside "
                "[0, %d]. s_wait_tensorcnt N retires with N tensor ops still "
                "outstanding, so a count above what the grouping supports starts "
                "LDS reads before their data lands. Sweep downward only."
                % (spec, mechanism, count, supported))
        out[mechanism] = count
    return out


# Read once at import so decoupledThickGateRelaxation stays pure for the life of
# the process: two calls in one build can never disagree.
_THICK_GATE_COUNT_OVERRIDE = _parseThickGateCountOverride(
    os.environ.get("TENSILE_DCP_THICK_GATE_COUNT", ""))


def dcpThickGateCountOverridden(mechanism):
    """True when the gate-pricing hook is overriding `mechanism`'s count."""
    return mechanism in _THICK_GATE_COUNT_OVERRIDE


def decoupledThickGateRelaxation(ks):
    """The relaxed thick-tensor gate a decoupled pair earns, or None.

    LAYER 3. Sole owner of that decision: both emission paths read it and
    neither decides anything itself.

    `s_wait_tensorcnt N` is an age-ordered drain on one counter -- N is how many
    tensor ops may still be outstanding when it retires, so 0 is a full drain
    and a larger N is a WEAKER gate. A divergent pair leaves one tensor holding
    more LDS blocks than the other, so the thin tensor's reads can begin while
    the thick tensor's remaining block is still landing.

      `(DCP_THICK_GATE_TOKENS, 1)`
          The grouping gives A and B a descriptor set each, so they can be
          handed disjoint TDM tensor tokens (`memTokenLdsDcp`) and the
          wait-count insertion pass derives the gate from dataflow, emitting
          `s_wait_tensorcnt 1` itself at the token-transition barriers. One
          sibling fill is left outstanding, hence 1.
      `(DCP_THICK_GATE_TEXT, 2)`
          The grouping puts A and B in one set, so their tensor ops share a
          token and the insertion pass can only emit a full drain.
          `KernelWriter._dcpApplyThickWait1` relaxes the gate after the thick
          fill's own label afterwards. That grouping leaves the thick tensor's
          second block outstanding as well as the sibling fill, hence 2.
      `None`
          No relaxation. An equal pair or scalar PrefetchGlobalRead has no thin
          side to start early. A separate-descriptor pair the TDM does not move
          on both tensors, or whose thin side is not on a single block, gets
          none either: there is no second tensor-token stream for a relaxed
          count to skip past.

    The *presence* of a relaxation follows from the PGR pair alone and is the
    same under every grouping: every divergent pair gets one. The grouping
    decides the *mechanism and count*. A census that counts `s_wait_tensorcnt 1`
    alone therefore counts only the separate-descriptor spelling and misreports
    the shared-descriptor grouping as unrelaxed, even though its gate is the
    weaker of the two.

    Why this asks the grouping and not TDMFuse
    ------------------------------------------
    `_tdmFuseCanShareDescriptors` can decline the grouping TDMFuse names -- on
    TDMSplit, subtile, a missing MX scale, the wrong wave count -- and the
    writer then falls back to the default, where A and B share one descriptor
    set. Both emission sites used to test `TDMFuse == 1` directly, so a declined
    solution would have been handed disjoint tokens for descriptors that are in
    fact shared. The guard was not where the decision was, and it failed open.

    That is the sixth instance in this project of a single root cause: a guard,
    or a path resolution, decoupled from the thing it governs, so that when the
    two disagree the guard permits instead of refusing. The fix is placement,
    not another check -- `tdmSeparateABDescriptors` routes the question through
    `tdmGrouping`, the same function the writer's own grouping comes from, so
    the two cannot diverge by construction.
    """
    decoupled, numLdsBlkA, numLdsBlkB = decouplePGRBlocks(ks)
    if not (decoupled and numLdsBlkA != numLdsBlkB):
        return None
    if tdmSeparateABDescriptors(ks):
        if not (ks.get("enableTDMA") and ks.get("enableTDMB")):
            return None
        if min(numLdsBlkA, numLdsBlkB) != 1:
            return None
        mechanism = DCP_THICK_GATE_TOKENS
    else:
        mechanism = DCP_THICK_GATE_TEXT
    return DcpThickGate(mechanism, _THICK_GATE_COUNT_OVERRIDE.get(
        mechanism, DCP_THICK_GATE_SUPPORTED[mechanism]))


def tdmWaveLdsBytes(ks, problemType=None):
    """LDS bytes each wave is responsible for filling, as {wave: bytes}.

    Crossing moves items between waves, so a wave's share is only meaningful
    per wave once TDMCross is in play -- which is the whole load-balance
    argument for crossing in the first place.

    Returns None when an element size cannot be resolved, matching
    decouplePGRLdsBytesEstimate. Block counts come from the decoupled PGR pair,
    so a tensor held at two LDS blocks counts twice.
    """
    _, blkA, blkB = decouplePGRBlocks(ks)
    blocks = {"A": blkA, "MXSA": blkA, "B": blkB, "MXSB": blkB}
    perTensor = {}
    for tc in TDM_TENSORS:
        if not tdmMemberIsLive(ks, tc):
            perTensor[tc] = 0
            continue
        size = ldsAlignedBytesForTensor(ks, tc, problemType)
        if size is None:
            return None
        perTensor[tc] = size * blocks[tc]
    return {w: sum(perTensor[tc] for tc in members)
            for w, members in tdmWaveAssignment(ks).items()}


def tdmCrossRejectReason(ks):
    """Why this solution cannot take a non-default TDMCross, or None.

    Derived from the group structure, never from a TDMFuse integer, so a new
    grouping row inherits the right answer without a new branch. In particular
    B_MX inherits A_MX's rejection mirrored, for free.

    Interactions this rejection has with the rest of the pipeline:

      TDMFuse       decides the grouping; crossing only rearranges it. A
                    grouping that TDMFuse itself rejects never reaches here,
                    so TDMFuse's reason is reported in preference to this one.
                    That is deliberate: an arrangement of a grouping that
                    cannot be built is not a separate defect.
      NumWaves      crossing needs distinct waves to cross between.
      MX scaling    a scale-less type degenerates to one live group, so it is
                    rejected by the group-count rule rather than by a type test.
      decoupled PGR block counts do not gate crossing -- crossing preserves
                    total LDS exactly (see tdmWaveLdsBytes), it only moves which
                    wave fills which part.
    """
    if tdmCross(ks) == TDM_CROSS_DEFAULT:
        return None
    numWaves = ks.get("NumWaves", 1)
    if numWaves <= 1:
        return ("TDMCross=%d rearranges which wave issues which member of a "
                "descriptor group, which needs wave-separated TDM (NumWaves > 1); "
                "got NumWaves=%d" % (tdmCross(ks), numWaves))
    groups = partitionedGroups(ks)
    if len(groups) < 2:
        return ("TDMCross=%d crosses one descriptor group against another, and "
                "the %s grouping leaves %d group(s) split across waves (%s); with "
                "fewer than two there is nothing to cross"
                % (tdmCross(ks), tdmGrouping(ks).name, len(groups),
                   " + ".join("{%s}" % ",".join(g) for g in liveGroups(ks)) or "none"))
    return None
