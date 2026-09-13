# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TDM descriptor grouping (TDMFuse): which tensors share one descriptor set.

  TDMFuse=0  {A,B} + {MXSA,MXSB}   default two-way parity
  TDMFuse=1  {A,MXSA} + {MXSB,B}   crossed parity
  TDMFuse=2  {A,MXSA,MXSB} + {B}   2/1/1 remainder split, NumWaves==4
  TDMFuse=3  {B,MXSA,MXSB} + {A}   the mirror of 2, same split and wave count

Which wave issues which member of a set is a separate question, and lives in
TDMCross.py.

The groupings are rows of TDM_GROUPS below, keyed on name. The TDMFuse integer
is not an index into that table: TDM_FUSE_GROUPING maps it to a row name, and a
row's own `.index` is an unrelated numbering. TDMFuse values are baked into
solution names and cannot move, so the name is the only stable key.
"""



# The four tensors a TDM descriptor group can carry. Sparse metadata rides
# tdmMetadataGroup0, which no grouping names, and is excluded everywhere.
TDM_TENSORS = ("A", "MXSA", "MXSB", "B")


class TdmGrouping:
    """One row of the grouping table.

    groups  member tuples, one per descriptor set. Member order is wave order.
    layout  "parity" splits a k-member group by wave index modulo k; "block"
            gives contiguous shares with the remainder to the leading members.
    index   a separate numbering. Never key on it.
    """

    __slots__ = ("name", "index", "groups", "layout")

    def __init__(self, name, index, groups, layout="parity"):
        self.name = name
        self.index = index
        self.groups = groups
        self.layout = layout


# Adding a row is three data edits, in any order: the row here, an acceptance
# entry below (a mirror row may share its mirror's), and a TDM_FUSE_GROUPING
# entry plus the same integer in ValidParameters and any yaml that sweeps it.
# Everything else derives from group membership. A row with no acceptance entry
# is unreachable: tdmGroupingName raises rather than resolving it.
#
# TDMCross.tdmWaveComponents and tdmSoleWave bound what a row may be -- a
# one-wave share, a contiguous share from wave 0, or a stride-two share -- and
# raise on anything else.
TDM_GROUPS = {
    "MX_AB": TdmGrouping("MX_AB", 4, (("A", "B"), ("MXSA", "MXSB"))),
    "paired": TdmGrouping("paired", 5, (("A", "MXSA"), ("MXSB", "B"))),
    "A_MX": TdmGrouping("A_MX", 2, (("A", "MXSA", "MXSB"), ("B",)), layout="block"),
    "B_MX": TdmGrouping("B_MX", 3, (("B", "MXSA", "MXSB"), ("A",)), layout="block"),
    # No integer selects these and no acceptance entry covers them, so they are
    # unreachable data; every derived function already answers for them.
    "AB": TdmGrouping("AB", 1, (("A", "B"), ("MXSA",), ("MXSB",))),
    "None": TdmGrouping("None", 0, (("A",), ("B",), ("MXSA",), ("MXSB",))),
}

# Baked into shipped solution names, so these integers cannot move.
TDM_FUSE_GROUPING = {0: "MX_AB", 1: "paired", 2: "A_MX", 3: "B_MX"}


def tdmBothTensors(ks):
    """True when TDMInst moves both A and B (bits 0 and 1)."""
    tdmInst = ks.get("TDMInst", 0)
    return bool(tdmInst & 0x01) and bool(tdmInst & 0x02)


def _tdmFuseCanShareDescriptors(ks):
    """Preconditions every descriptor-sharing row needs.

    TDMSplit is tested here even while Solution rejects it globally: writer
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

    `A_MX` and `B_MX` are mirror images, so one entry serves both. NumWaves==4
    is not conservatism: the 2/1/1 split is remainder-to-leading-member, which
    at eight waves is 3/3/2 -- a trailing two-wave share `tdmWaveComponents`
    cannot spell. Four is the only count where the policy and an emittable
    partition coincide.
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

    Fail-closed: resolving an unmapped integer, or a row with no acceptance
    entry, would name the kernel for a grouping the writer does not build.
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

    Solution.py must reject exactly where this is False, or the kernel name
    carries an integer the writer did not honour.
    """
    return bool(_GROUPING_ACCEPTED[tdmGroupingName(ks)](ks))


def tdmGrouping(ks):
    """The grouping row this solution actually gets.

    A declined grouping falls back to the default, exactly as the writer does.
    """
    name = tdmGroupingName(ks)
    if not _GROUPING_ACCEPTED[name](ks):
        return TDM_GROUPS[TDM_GROUPING_DEFAULT]
    return TDM_GROUPS[name]


def tdmFuseAMx(ks):
    """TDMFuse=2: {A,MXSA,MXSB} share one set, B owns its own. NumWaves==4."""
    return tdmGrouping(ks).name == "A_MX"


def tdmFusePaired(ks):
    """TDMFuse=1: {MXSA,A} and {MXSB,B}, crossed parity. NumWaves>1."""
    return tdmGrouping(ks).name == "paired"


def tdmWaveSeparated(ks):
    """True when the TDM moves both tensors and there is more than one wave.

    Otherwise there is no wave partition to lay a grouping onto, and the
    default shared descriptor set is programmed whatever TDMFuse asked for.
    """
    return bool(ks.get("enableTDMA") and ks.get("enableTDMB")
                and ks.get("NumWaves", 1) > 1)


def tdmGroupingSeparatesAB(ks):
    """True when the resolved grouping row puts A and B in different sets.

    The table question alone. The thick gate needs it separately from
    tdmSeparateABDescriptors to tell "separated but not wave separated", which
    earns no relaxation, from "shared", which earns the text mechanism.
    """
    return not any({"A", "B"} <= set(group) for group in tdmGrouping(ks).groups)


def tdmSeparateABDescriptors(ks):
    """True when A's and B's TDM descriptors are distinct register sets.

    One shared set means one TDM tensor token, so nothing can drain one tensor
    without draining the other; two sets take disjoint tokens. That is the only
    property the decoupled PGR thick gate turns on.
    """
    return tdmWaveSeparated(ks) and tdmGroupingSeparatesAB(ks)


def tdmMemberIsLive(ks, tc):
    """True when tensor `tc` exists on this problem.

    A scale-less type degenerates its groups, which is why no rejection here
    tests the data type: the group count derives it.
    """
    if tc not in ("MXSA", "MXSB"):
        return True
    pt = ks.get("ProblemType") or {}
    return bool(pt.get("MXBlock%s" % tc[-1]))


def liveGroups(ks, grouping=None):
    """`grouping.groups` with dead members dropped and empty groups removed.

    The wave layout is computed from the FULL group instead: a set keeps its
    shape when a member is absent, and sizing shares off the live members would
    move the survivor onto different waves.
    """
    grouping = grouping if grouping is not None else tdmGrouping(ks)
    live = []
    for group in grouping.groups:
        members = tuple(tc for tc in group if tdmMemberIsLive(ks, tc))
        if members:
            live.append(members)
    return tuple(live)


def partitionedGroups(ks, grouping=None):
    """Live groups actually split across waves (two or more members).

    A one-member group is issued by every wave, so there is nothing to cross.
    """
    return tuple(g for g in liveGroups(ks, grouping) if len(g) > 1)


# The descriptor-set ownership every PrefetchAcrossPersistent helper is written
# against: the data tensors on one register range, the scale tensors on another.
TDM_DATA_TENSORS = ("A", "B")
TDM_SCALE_TENSORS = ("MXSA", "MXSB")


def tdmScaleSharesDataSet(ks, grouping=None):
    """Live groups carrying a data tensor and a scale tensor on one set."""
    return tuple(g for g in liveGroups(ks, grouping)
                 if any(tc in TDM_DATA_TENSORS for tc in g)
                 and any(tc in TDM_SCALE_TENSORS for tc in g))


def tdmSetGroup(ks, tc):
    """The full group `tc` rides, or None. Not the live one: see liveGroups."""
    for group in tdmGrouping(ks).groups:
        if tc in group:
            return group
    return None


def tdmSetOwner(ks, tc):
    """The member whose name programs the descriptor set that carries `tc`.

    The rest are RegSet aliases of it, so per-set mutations must be applied
    under the owner only: applied under an alias too, the even count cancels
    silently instead of failing to build. The owner is the set's data tensor,
    or its first member when it has none.
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

    The shape A_MX names and B_MX mirrors. Asked by structure so the writer's
    dispatch, increment fallthrough and RegSet alias are one body with the
    owner substituted.
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


def tdmSharedSetOrder(ks, tcA, tcB):
    """(shared, separate) of a data-tensor pair, the shared set's owner first.

    Exchanged for B_MX, so the writer's callers need no `if owner == "A"`.
    """
    owner = tdmSharedScaleSetOwner(ks)
    return (tcA, tcB) if owner is None or tcA == owner else (tcB, tcA)


def tdmSharedScaleSetActive(ks):
    """True when the writer emits a shared-scale-set dispatch for this solution.

    A module function, not a writer method: several writer tests drive real
    method bodies through stubs that forward a fixed list of method names.
    """
    return tdmWaveSeparated(ks) and tdmSharedScaleSet(ks) is not None


def tdmPapRejectReason(ks):
    """Why PrefetchAcrossPersistent cannot ride this grouping, or None.

    PAP's handoff helpers offset the pairs (A, B) and (MXSA, MXSB) once each,
    which holds only while the scales own their own range. A grouping seating a
    scale on a data tensor's set aliases the two together, so one range is
    offset twice and its sibling not at all, and the tail reset names an
    unallocated increment.
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


class TdmArrangementNotEmittable(ValueError):
    """A grouping row's wave partition has no spelling in the writer.

    Raised rather than answered approximately. A mis-derived wave rule does not
    fault: it points two waves at one LDS block, or advances a descriptor by
    another tensor's increment, and the assembly looks well formed either way.
    """


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


def _arrangedGroups(ks):
    """The grouping's groups in the member order TDMCross selected.

    The import is function-local on purpose: TDMCross is a later feature, and
    with the key absent this never runs, so the grouping works without it.
    """
    groups = tdmGrouping(ks).groups
    if not ks.get("TDMCross", 0):
        return groups
    from .TDMCross import crossGroups
    return crossGroups(groups)


def tdmWaveAssignment(ks):
    """Which tensors each wave issues, as {wave: (tc, ...)}.

    The only consumer of TDMCross that decides codegen; every other site reads
    this or `tdmWavePartition`, its per-tensor view. A consumer can still be
    unable to spell a partition, so an arrangement is refused downstream rather
    than applied inconsistently -- `tdmWaveComponents` is the live example.
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


def tdmWaveComponents(ks, tc):
    """(numComp, compShift) for tensor `tc`.

    compShift is a right-shift on WaveIdx, or None when WaveIdx is unused:
    None emits `SMov id, 0`, 0 emits `SMov id, WaveIdx`, n emits `SLshr id,
    WaveIdx, n`. None is the sentinel because 0 already means shift-by-zero.

    Those three are the whole vocabulary, so a share outside them raises rather
    than being rounded: rounding a share of (2, 3) to a shift of one lands both
    waves on component 1, pointing two waves at one LDS block with no trace in
    the assembly. Zero components is not such a case and is allowed -- no wave
    carries the member, so the id is never read.
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
    """`tc`'s wave share as comment text: "wave 2", or "waves 0-1"."""
    _, waves = tdmWavePartition(ks, tc)
    if len(waves) == 1:
        return "wave %d" % waves[0]
    return "waves %d-%d" % (waves[0], waves[-1])
