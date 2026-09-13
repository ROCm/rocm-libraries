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
TDMFuse=2  {A,MXSA,MXSB} + {B}   2/1/1 remainder split, NumWaves==4 (tdmFuseAMx)
TDMFuse=3  {B,MXSA,MXSB} + {A}   the mirror of 2, same split and wave count

The groupings are rows of TDM_GROUPS below, keyed on name.  The TDMFuse integer
is not an index into that table.  It is a key into TDM_FUSE_GROUPING, which maps
it to a row name; a row's own `.index` is a third, unrelated numbering (4, 5, 2,
3, 1, 0) carried for history, while the integers we ship are 0, 1, 2.  Our values
are baked into solution names and cannot move, so a name is the only stable key.
"""

import os
import re
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


class TdmArrangementNotEmittable(ValueError):
    """A grouping row's wave partition has no spelling in the writer.

    Raised rather than answered approximately. A mis-derived wave rule does not
    fault: it points two waves at one LDS block, or advances a descriptor by
    another tensor's increment, and the assembly looks well formed either way.
    """


class TdmGrouping:
    """One row of the grouping table.

    groups     tuple of member tuples. Each group is one descriptor set, i.e.
               one fused `tensor_load_to_lds` whose members are dispatched
               across waves. Member order IS the wave order (see `layout`).
    layout     "parity" splits a k-member group by wave index modulo k, so a
               two-member group lands member 0 on the even waves and member 1
               on the odd. "block" splits the waves into k contiguous shares,
               remainder to the leading members, which is the 2/1/1 policy
               TDMFuse=2 names: at NumWaves=4 its three-member group puts A on
               waves 0-1, MXSA on wave 2 and MXSB on wave 3.
    index      the historical integer. Arbitrary and non-contiguous with
               anybody else's numbering; never key on it.
    """

    __slots__ = ("name", "index", "groups", "layout")

    def __init__(self, name, index, groups, layout="parity"):
        self.name = name
        self.index = index
        self.groups = groups
        self.layout = layout


# Declarative grouping table, keyed on name.
#
# What adding a row costs, measured by wiring `B_MX` end to end rather than
# argued from the shape of the code.  Three edits are REQUIRED and they are the
# whole of it:
#
#   1. the row here (pure data: name, index, groups, layout);
#   2. an entry in `_GROUPING_ACCEPTED` -- the preconditions the row needs.  A
#      mirror row shares its mirror's entry, so `B_MX` reuses `A_MX`'s;
#   3. an entry in `TDM_FUSE_GROUPING` pointing an integer at the row, plus the
#      same integer in `ValidParameters` and in any yaml that sweeps it.
#
# The three are order-free.  A row with no acceptance entry is not selectable --
# `tdmGrouping` raises rather than resolving it -- so the window in which an
# integer is reachable but unguarded cannot be opened.  Before that, step 3
# ahead of step 2 built kernels named for a grouping they did not have.
#
# Everything else is derived and inherited: the wave layout (`_waveShares`), the
# crossing rejection (`tdmCrossRejectReason`), the naming (`Naming.py`), the
# descriptor-set owner (`tdmSetOwner`), the SGPR aliasing `defineTdmSgprs`
# emits, the PAP rejection (`tdmPapRejectReason`), the shared-set increment
# dispatch and the thick gate.  None of them branch on a name or an index.
#
# Two rules stay only as general as the wave shapes they can spell:
# `tdmWaveComponents` expresses a one-wave share, a contiguous share starting at
# wave 0, and a stride-two share; `tdmSoleWave` expresses a single wave.  Both
# refuse anything else by raising, so a row whose partition falls outside them
# fails the build loudly instead of miscompiling -- but it needs work there
# before an integer may point at it.  That is the one thing on this list a new
# row can still owe.
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
    "B_MX": TdmGrouping("B_MX", 3, (("B", "MXSA", "MXSB"), ("A",)), layout="block"),
    # Rows no TDMFuse integer selects, and which no _GROUPING_ACCEPTED entry
    # covers either, so they are unreachable rather than merely unswept. They
    # are data: every derived function here already answers correctly for them.
    "AB": TdmGrouping("AB", 1, (("A", "B"), ("MXSA",), ("MXSB",))),
    "None": TdmGrouping("None", 0, (("A",), ("B",), ("MXSA",), ("MXSB",))),
}

# Our TDMFuse integers, resolved to a row name. The integers are baked into
# shipped solution names, the frozen 00_Final.yaml and the silicon evidence, so
# they cannot move; the parallel branch numbers the same groupings differently.
# Keying the table on the name is what lets both numberings coexist.
#
# 3 is the one integer that means the same grouping on both sides, which is why
# it is the one that was wired here. It is not free of a caveat: the two wave
# policies coincide at four waves and diverge above it, so `_acceptSharedScaleSet`
# and Solution.py both pin NumWaves == 4. `AB` needs a number from 6 up if it is
# ever wired -- its `.index` is 1, and 1 is occupied on both sides with
# different meanings. Do not read `.index` as an integer assignment.
TDM_FUSE_GROUPING = {0: "MX_AB", 1: "paired", 2: "A_MX", 3: "B_MX"}


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


def _acceptSharedScaleSet(ks):
    """Preconditions for a row seating both MX scales on one data tensor's set.

    `A_MX` and `B_MX` are mirror images -- {A,MXSA,MXSB}+{B} against
    {B,MXSA,MXSB}+{A} -- so one entry serves both. NumWaves==4 is not a
    conservative choice: the 2/1/1 split is `divmod` remainder-to-leading-member,
    which at eight waves is 3/3/2, a trailing two-wave share that
    `tdmWaveComponents` cannot spell. Four waves is the only count where the
    remainder policy and an emittable partition coincide.
    """
    return _tdmFuseCanShareDescriptors(ks) and ks.get("NumWaves", 1) == 4


def _acceptPairedSets(ks):
    """Preconditions for a row giving each data tensor its own scale."""
    return _tdmFuseCanShareDescriptors(ks) and ks.get("NumWaves", 1) > 1


def _acceptAlways(ks):
    """The default row. `defineTdmSgprs` programs it for any solution, which is
    what makes it the fallback a declined grouping can land on."""
    return True


# Acceptance predicate per row name. A row is selectable only with an entry
# here: `tdmGrouping` raises on a row that has none rather than resolving it, so
# an integer can never be reachable while its preconditions are unwritten. Rows
# absent from this table (`AB`, `None`) are therefore unreachable, not merely
# unswept.
_GROUPING_ACCEPTED = {
    "MX_AB": _acceptAlways,
    "paired": _acceptPairedSets,
    "A_MX": _acceptSharedScaleSet,
    "B_MX": _acceptSharedScaleSet,
}

# The row a declined grouping falls back to, which is also TDMFuse=0's row.
TDM_GROUPING_DEFAULT = TDM_FUSE_GROUPING[0]


def tdmGroupingName(ks):
    """The row name TDMFuse asks for, before acceptance.

    Fail-closed on both halves of the lookup, because both used to fail open in
    the same direction -- towards the default grouping, under the name of a
    different one:

      unmapped integer   resolved to the default row silently, so a kernel built
                         with it was named `_TDMF<n>` and was byte-for-byte the
                         default grouping. No error, no warning, no rejection.
      row with no
      acceptance entry   would resolve as soon as a mapping named it, with
                         nothing left to enforce the preconditions it needs.

    Raising here rather than returning a default is what lets the three wiring
    edits be done in any order.
    """
    fuse = ks.get("TDMFuse", 0)
    name = TDM_FUSE_GROUPING.get(fuse)
    if name is None:
        raise ValueError(
            "TDMFuse=%r names no grouping; TDM_FUSE_GROUPING maps %s. Add the "
            "integer there (and to ValidParameters) rather than relying on a "
            "fallback: an unmapped integer resolving to the %s row would name "
            "the kernel for a grouping it does not have."
            % (fuse, sorted(TDM_FUSE_GROUPING), TDM_GROUPING_DEFAULT))
    if name not in TDM_GROUPS:
        raise ValueError(
            "TDMFuse=%r maps to %r, which is not a row of TDM_GROUPS (%s)"
            % (fuse, name, sorted(TDM_GROUPS)))
    if name not in _GROUPING_ACCEPTED:
        raise ValueError(
            "TDMFuse=%r maps to the %r row, which has no _GROUPING_ACCEPTED "
            "entry, so nothing would enforce the preconditions it needs. Add "
            "the entry -- a mirror row may share its mirror's -- before "
            "pointing an integer at the row." % (fuse, name))
    return name


def tdmGroupingAccepted(ks):
    """True when the row TDMFuse asks for can actually be produced here.

    Solution.py's per-integer guards must reject exactly where this is False:
    a declined grouping falls back to the default while the kernel name still
    carries the integer, so a missing rejection is a kernel that lies.
    """
    return bool(_GROUPING_ACCEPTED[tdmGroupingName(ks)](ks))


def tdmGrouping(ks):
    """The grouping row this solution actually gets.

    TDMFuse names a grouping, but its acceptance predicate can decline it -- on
    TDMSplit, subtile, a missing scale, the wrong wave count. A declined
    grouping falls back to the default, exactly as the writer does.
    """
    name = tdmGroupingName(ks)
    if not _GROUPING_ACCEPTED[name](ks):
        return TDM_GROUPS[TDM_GROUPING_DEFAULT]
    return TDM_GROUPS[name]


def tdmFuseAMx(ks):
    """TDMFuse=2: {A,MXSA,MXSB} share one set, B owns its own. NumWaves==4.

    Asks which row resolved rather than restating the integer and its guards, so
    it cannot drift from the resolver the way a second copy of the preconditions
    could. Only TDMFuse=2 maps to `A_MX`, so this is the same answer.
    """
    return tdmGrouping(ks).name == "A_MX"


def tdmFusePaired(ks):
    """TDMFuse=1: {MXSA,A} and {MXSB,B}, crossed parity. NumWaves>1."""
    return tdmGrouping(ks).name == "paired"


def tdmWaveSeparated(ks):
    """True when the TDM moves both tensors and there is more than one wave.

    The writer's precondition for every wave-separated arrangement: with a
    single wave, or with the TDM moving only one of A and B, there is no wave
    partition to lay a grouping onto and the default shared descriptor set is
    programmed whatever TDMFuse asked for.

    `KernelWriterAssembly.isTdmWaveSeparated` delegates here rather than
    restating it, so the writer and the thick gate cannot drift apart on the
    precondition the way they could on the grouping question below.
    """
    return bool(ks.get("enableTDMA") and ks.get("enableTDMB")
                and ks.get("NumWaves", 1) > 1)


def tdmGroupingSeparatesAB(ks):
    """True when the resolved grouping row puts A and B in different sets.

    The table question alone, without the wave-separation precondition. Exposed
    because the thick gate needs to tell "the grouping would have separated them
    but the writer is not wave separated", which earns no relaxation at all,
    from "the grouping shares them", which earns the text mechanism.
    """
    return not any({"A", "B"} <= set(group) for group in tdmGrouping(ks).groups)


def tdmSeparateABDescriptors(ks):
    """True when A's and B's TDM descriptors are distinct register sets.

    Two tensors sharing one set ride one fused `tensor_load_to_lds` and so share
    one TDM tensor token: nothing can drain one without draining the other. Two
    sets can be handed disjoint tokens and drained independently. That is the
    only property the decoupled PGR thick gate turns on.

    Read off the resolved grouping row, never off the TDMFuse integer, so a
    grouping TDMFuse asked for but `_tdmFuseCanShareDescriptors` declined
    answers with the fallback the writer actually gets. Adding a row therefore
    needs no branch here: `A_MX` and `B_MX` separate A from B, `MX_AB` and `AB`
    do not, and `None` fuses nothing so every tensor is its own set.

    Single owner. `KernelWriterAssembly.tdmSeparateABDescriptors` used to spell
    this `tdmFuseAMx or tdmFusePaired`, naming the two wired TDMFuse values
    instead of asking the table. The two agreed on every shipped kernel, so
    nothing miscompiled, but they are not the same question: `B_MX` separates A
    from B and neither predicate names it, so the writer answered False where
    this answers True. That method now delegates here, leaving one definition
    rather than two that happen to match.
    """
    return tdmWaveSeparated(ks) and tdmGroupingSeparatesAB(ks)


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


# The descriptor-set ownership every PrefetchAcrossPersistent helper is written
# against: the data tensors on one register range, the scale tensors on another.
TDM_DATA_TENSORS = ("A", "B")
TDM_SCALE_TENSORS = ("MXSA", "MXSB")


def tdmScaleSharesDataSet(ks, grouping=None):
    """Live groups that carry a data tensor and a scale tensor on one set.

    Read off the resolved grouping rather than a TDMFuse integer, so a grouping
    TDMFuse asked for but `_tdmFuseCanShareDescriptors` declined answers with
    the fallback the writer actually gets, and a scale-less type answers with
    its degenerate groups -- it has no MXSA/MXSB left to share anything.
    """
    return tuple(g for g in liveGroups(ks, grouping)
                 if any(tc in TDM_DATA_TENSORS for tc in g)
                 and any(tc in TDM_SCALE_TENSORS for tc in g))


def tdmSetGroup(ks, tc):
    """The group `tc` rides, or None when no row names it.

    The full group, not the live one: a set keeps its shape when a member is
    absent (see liveGroups).
    """
    for group in tdmGrouping(ks).groups:
        if tc in group:
            return group
    return None


def tdmSetOwner(ks, tc):
    """The member whose name programs the descriptor set that carries `tc`.

    One member of each set is allocated by `defineTdmSgprs` and the rest are
    RegSet aliases of it, so per-set mutations -- the LDS buffer swap above all
    -- must be applied under the owner only: applied under an alias too, the
    even count cancels silently instead of failing to build.

    The owner is the set's data tensor, or its first member when it has none.
    That is a reading of the table, not a new convention: it reproduces every
    allocation `defineTdmSgprs` already emits -- `MX_AB` allocates A and aliases
    B onto it, and allocates MXSA and aliases MXSB; `paired` allocates A and B
    and aliases each scale onto its own data tensor, so the {MXSB,B} set is
    owned by B even though the table writes MXSB first; `A_MX` allocates A and
    B and puts both scales on A. `B_MX` inherits the mirror of the last one with
    no branch here.

    A set cannot have two data tensors and an alias to resolve at the same time:
    {A,B} is one register range whichever name allocates it, and A is the name
    that does.
    """
    group = tdmSetGroup(ks, tc)
    if group is None:
        return tc
    for member in group:
        if member in TDM_DATA_TENSORS:
            return member
    return group[0]


def tdmSharedScaleSet(ks):
    """The set seating both MX scales on one data tensor's set, or None.

    The shape `A_MX` names and `B_MX` mirrors: three members, one data tensor
    and both scales, dispatched 2/1/1 across four waves. The writer needs to ask
    for it by structure rather than by TDMFuse value, because everything it then
    does -- the three-way dispatch guard, which increment the shared register
    falls through to, which name gets the RegSet alias -- is the same code with
    the owner substituted.
    """
    for group in tdmGrouping(ks).groups:
        if len(group) < 2 or not set(TDM_SCALE_TENSORS) <= set(group):
            continue
        if len([tc for tc in group if tc in TDM_DATA_TENSORS]) == 1:
            return group
    return None


def tdmSharedScaleSetOwner(ks):
    """The data tensor whose descriptor set both MX scales ride, or None."""
    group = tdmSharedScaleSet(ks)
    if group is None:
        return None
    return next(tc for tc in group if tc in TDM_DATA_TENSORS)


def tdmFuseSharedScales(ks):
    """True when both MX scales ride one data tensor's descriptor set.

    True for `A_MX` and for `B_MX`. The writer's own `tdmFuseAMx` asks the
    narrower question -- is this specifically A's set -- and the difference is
    exactly the set of sites that had A hardcoded as the owner.
    """
    return tdmSharedScaleSet(ks) is not None


def tdmSharedSetOrder(ks, tcA, tcB):
    """(shared, separate) of a data-tensor pair, the shared set's owner first.

    Exchanged for `B_MX`, where B holds the shared set. A function of the two
    names so the writer's callers need no `if owner == "A"`, and a module
    function for the reason `tdmSharedScaleSetActive` gives.
    """
    owner = tdmSharedScaleSetOwner(ks)
    return (tcA, tcB) if owner is None or tcA == owner else (tcB, tcA)


def tdmSharedScaleSetActive(ks):
    """True when the writer emits a shared-scale-set dispatch for this solution.

    The grouping question above plus the wave-separation precondition: with one
    wave, or with the TDM moving only one of A and B, there is no wave partition
    to dispatch over and the default shared set is programmed whatever TDMFuse
    asked for.

    A module function rather than a writer method, deliberately, and the reason
    is a measurement. Several writer tests drive real method bodies through
    hand-rolled stub objects that forward a FIXED LIST of method names. Adding
    this predicate as a `KernelWriterAssembly` method broke 44 of them with
    `AttributeError` while the generated corpus stayed byte-identical -- the
    stubs do not fake the answers, they forward to the real bodies, so nothing
    about the behaviour had changed. A method is a name every stub must know; a
    module function is not. `tdmWaveRangeText` is a plain function for the same
    reason, and says so.
    """
    return tdmWaveSeparated(ks) and tdmFuseSharedScales(ks)


def tdmPapRejectReason(ks):
    """Why PrefetchAcrossPersistent cannot ride this grouping, or None.

    PAP hands a persistent tile over by rebuilding the TDM descriptors and by
    saving and restoring the LDS bank the next-tile prefetch landed in. Every
    one of those helpers walks the tensors as the two pairs (A, B) and
    (MXSA, MXSB) and applies exactly one offset per pair, which is correct only
    for the ownership the default grouping has: {A,B} on one descriptor set and
    {MXSA,MXSB} on another, allocated as two distinct register ranges.

    A grouping that seats a scale tensor on a data tensor's set makes
    `tdmMXSAGroup0` / `tdmMXSBGroup0` RegSet aliases of `tdmAGroup0` /
    `tdmBGroup0`. The two per-pair offsets then land on one physical range
    twice while its sibling range is never offset at all, and the tail reset
    reads `tdmMXSAMXSBIncs`, which is not allocated once both scales ride A's
    set. The two failures are not alike: the first is silent, the second
    reaches the assembler as an undefined symbol.

    Derived from group membership, so a new grouping row inherits the right
    answer with no branch here, and the reason names the sharing group rather
    than the parameter value that happened to select it.
    """
    shared = tdmScaleSharesDataSet(ks)
    if not shared:
        return None
    return ("PrefetchAcrossPersistent requires every TDM scale tensor to own its "
            "own descriptor set, and the %s grouping seats one on a data "
            "tensor's set (%s). That aliases tdmMXSAGroup0/tdmMXSBGroup0 onto "
            "tdmAGroup0/tdmBGroup0, so the persistent-tile handoff offsets one "
            "register range twice and its sibling range not at all, and the "
            "tail reset reads the unallocated tdmMXSAMXSBIncs"
            % (tdmGrouping(ks).name,
               " + ".join("{%s}" % ",".join(g) for g in shared)))


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
    """The requested wave arrangement. Missing key means default.

    Boolean in effect, not an enum. Every caller compares against
    TDM_CROSS_DEFAULT rather than dispatching on the value, so any nonzero
    selects the one crossed arrangement and TDMCross=2 is TDMCross=1, not a
    third shape. A third arrangement means teaching `_arrangedGroups` to
    dispatch on the value, not just widening ValidParameters.
    """
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

    LAYER 1. The only consumer of TDMCross that decides codegen, which it
    reaches through `_arrangedGroups`; `tdmCrossRejectReason` also reads the
    parameter, to name the value it refuses. Every other site reads the
    assignment, or tdmWavePartition, its per-tensor view.

    That routing is necessary for an arrangement to apply whole, not sufficient.
    A consumer that reads the partition can still be unable to spell it, so an
    arrangement can half-apply by being refused downstream instead of being
    applied inconsistently -- `tdmWaveComponents` is the live example.

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


def tdmSoleWave(ks, tc):
    """The one wave that carries `tc`, read off the partition.

    A chained s_cselect dispatch can only test WaveIdx against a single value per
    member, so a row that gives `tc` more than one wave needs a different
    spelling in the writer. Refuse rather than compare against the first and
    leave the rest of the share advancing by another tensor's increment.
    """
    _, waves = tdmWavePartition(ks, tc)
    if len(waves) != 1:
        raise TdmArrangementNotEmittable(
            "the shared-set increment selects one wave per member, but the "
            "grouping puts %s on waves %s; extend that dispatch to a wave range "
            "before wiring a row with that partition" % (tc, waves))
    return waves[0]


def tdmWaveRangeText(ks, tc):
    """`tc`'s wave share as comment text: "wave 2", or "waves 0-1".

    A plain function rather than a writer method on purpose: it needs nothing but
    the solution, and several tests drive the writer through hand-rolled stubs
    that carry only the methods they expect. New instance methods break those
    stubs; module functions do not.
    """
    _, waves = tdmWavePartition(ks, tc)
    if len(waves) == 1:
        return "wave %d" % waves[0]
    return "waves %d-%d" % (waves[0], waves[-1])


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

    Those three are the whole vocabulary, so a share outside them raises
    TdmArrangementNotEmittable rather than being rounded to the nearest one. A
    two-wave share of (2, 3) is the case that used to fall through to a shift of
    one and land both waves on component 1, pointing two waves at a single LDS
    block -- a data race with no trace in the assembly.

    A share of zero components is not such a case and is not refused: at
    NumWaves=1 a multi-member group leaves its later members on no wave, which
    _waveShares reports as zero components, and no wave reads the id.
    """
    numComp, waves = tdmWavePartition(ks, tc)
    if numComp == 1:
        return numComp, None
    if waves == tuple(range(numComp)):
        return numComp, 0
    if numComp < 2:
        # numComp == 0: a multi-member group on a single wave leaves this member
        # on no wave at all, and _waveShares reports zero components for it by
        # design. There is no wave rule here to be unable to spell -- no wave
        # carries the member, so the id is never read -- so keep the shipped
        # answer instead of refusing.
        return numComp, 1
    if tuple(w >> 1 for w in waves) == tuple(range(numComp)):
        return numComp, 1
    raise TdmArrangementNotEmittable(
        "tensor %s rides waves %s over %d components, and no right-shift of "
        "WaveIdx maps that onto components 0..%d: shifting by one gives %s. A "
        "component id is emittable as a constant zero (one wave), WaveIdx "
        "itself (a contiguous share from wave 0) or WaveIdx >> 1 (a stride-two "
        "share). Teach this function the shape before pointing a TDMFuse "
        "integer at a row that produces it."
        % (tc, waves, numComp, numComp - 1, tuple(w >> 1 for w in waves)))


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
