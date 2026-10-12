# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TDM descriptor grouping (TDMFuse): which tensors share one descriptor set.

  TDMFuse=0  {A,B} + {MXSA,MXSB}   default two-way parity
  TDMFuse=1  {A,MXSA} + {MXSB,B}   paired: each scale on a data tensor
  TDMFuse=2  {A,MXSA,MXSB} + {B}   2/1/1 remainder split, NumWaves==4
  TDMFuse=3  {B,MXSA,MXSB} + {A}   the mirror of 2, same split and wave count

TDM_FUSE_GROUPING maps stable, name-bearing integers to TDM_GROUPS rows.
"""


class TdmGrouping:
    """One row of the grouping table.

    groups  member tuples, one per descriptor set; member order is wave order.
    layout  "parity" splits a k-member group by wave index modulo k; "block"
            gives contiguous shares, remainder to the leading members.
    """

    __slots__ = ("name", "groups", "layout")

    def __init__(self, name, groups, layout="parity"):
        self.name = name
        self.groups = groups
        self.layout = layout


TDM_GROUPS = {
    "MX_AB": TdmGrouping("MX_AB", (("A", "B"), ("MXSA", "MXSB"))),
    "paired": TdmGrouping("paired", (("A", "MXSA"), ("MXSB", "B"))),
    "A_MX": TdmGrouping("A_MX", (("A", "MXSA", "MXSB"), ("B",)), layout="block"),
    "B_MX": TdmGrouping("B_MX", (("B", "MXSA", "MXSB"), ("A",)), layout="block"),
}

# Baked into shipped solution names, so these integers cannot move.
TDM_FUSE_GROUPING = {0: "MX_AB", 1: "paired", 2: "A_MX", 3: "B_MX"}


def tdmBothTensors(ks):
    """True when TDMInst moves both A and B (bits 0 and 1)."""
    tdmInst = ks.get("TDMInst", 0)
    return bool(tdmInst & 0x01) and bool(tdmInst & 0x02)


def _tdmFuseCanShareDescriptors(ks):
    """Shared-descriptor preconditions, including writer-visible TDMSplit states."""
    if not tdmBothTensors(ks):
        return False
    if ks.get("TDMSplit") or ks.get("UseSubtileImpl"):
        return False
    pt = ks.get("ProblemType") or {}
    return bool(pt.get("MXBlockA") and pt.get("MXBlockB"))


def _acceptSharedScaleSet(ks):
    """Accept A_MX/B_MX only for the emittable four-wave 2/1/1 partition."""
    return _tdmFuseCanShareDescriptors(ks) and ks.get("NumWaves", 1) == 4


def _acceptPairedSets(ks):
    """Preconditions for a row giving each data tensor its own scale."""
    return _tdmFuseCanShareDescriptors(ks) and ks.get("NumWaves", 1) > 1


def _acceptAlways(ks):
    """The default row: defineTdmSgprs programs it for any solution."""
    return True


# Every mapped row needs an acceptance predicate; tdmGroupingName rejects omissions.
_GROUPING_ACCEPTED = {
    "MX_AB": _acceptAlways,
    "paired": _acceptPairedSets,
    "A_MX": _acceptSharedScaleSet,
    "B_MX": _acceptSharedScaleSet,
}

# The row a declined grouping falls back to, which is also TDMFuse=0's row.
TDM_GROUPING_DEFAULT = TDM_FUSE_GROUPING[0]


def tdmGroupingName(ks):
    """Return the requested row name, failing closed on incomplete mappings."""
    fuse = ks.get("TDMFuse", 0)
    name = TDM_FUSE_GROUPING.get(fuse)
    if name is None:
        raise ValueError("TDMFuse=%r names no grouping; mapped values are %s"
                         % (fuse, sorted(TDM_FUSE_GROUPING)))
    if name not in TDM_GROUPS:
        raise ValueError("TDMFuse=%r maps to %r, which is not a row of TDM_GROUPS (%s)"
                         % (fuse, name, sorted(TDM_GROUPS)))
    if name not in _GROUPING_ACCEPTED:
        raise ValueError("TDMFuse=%r maps to the %r row, which has no _GROUPING_ACCEPTED entry"
                         % (fuse, name))
    return name


def tdmGroupingAccepted(ks):
    """True when the requested grouping can be produced."""
    return bool(_GROUPING_ACCEPTED[tdmGroupingName(ks)](ks))


def tdmGrouping(ks):
    """Return the accepted grouping, or the default when declined."""
    name = tdmGroupingName(ks)
    if not _GROUPING_ACCEPTED[name](ks):
        return TDM_GROUPS[TDM_GROUPING_DEFAULT]
    return TDM_GROUPS[name]


def tdmFusePaired(ks):
    """TDMFuse=1: {A,MXSA} and {MXSB,B}, one scale per data tensor. NumWaves>1."""
    return tdmGrouping(ks).name == "paired"


def tdmWaveSeparated(ks):
    """True when TDM moves both tensors across multiple waves."""
    return bool(ks.get("enableTDMA") and ks.get("enableTDMB")
                and ks.get("NumWaves", 1) > 1)


def tdmGroupingSeparatesAB(ks):
    """True when the resolved row puts A and B in different sets."""
    return not any({"A", "B"} <= set(group) for group in tdmGrouping(ks).groups)


def tdmSeparateABDescriptors(ks):
    """True when A/B have distinct descriptor registers and tensor tokens."""
    return tdmWaveSeparated(ks) and tdmGroupingSeparatesAB(ks)


def tdmMemberIsLive(ks, tc):
    """True when tensor `tc` exists on this problem; only MX scales can be absent."""
    if tc not in ("MXSA", "MXSB"):
        return True
    pt = ks.get("ProblemType") or {}
    return bool(pt.get("MXBlock%s" % tc[-1]))


def liveGroups(ks, grouping=None):
    """Drop dead members, but preserve full-group wave partitioning."""
    grouping = grouping if grouping is not None else tdmGrouping(ks)
    live = []
    for group in grouping.groups:
        members = tuple(tc for tc in group if tdmMemberIsLive(ks, tc))
        if members:
            live.append(members)
    return tuple(live)


# Descriptor ownership expected by PrefetchAcrossPersistent.
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

    Other members are RegSet aliases; mutate the set through its owner only.
    """
    group = tdmSetGroup(ks, tc)
    if group is None:
        return tc
    for member in group:
        if member in TDM_DATA_TENSORS:
            return member
    return group[0]


def tdmSharedScaleSet(ks):
    """Return the set containing both MX scales and one data tensor, or None."""
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
    """Return (shared-set owner, separate tensor)."""
    owner = tdmSharedScaleSetOwner(ks)
    return (tcA, tcB) if owner is None or tcA == owner else (tcB, tcA)


def tdmSharedScaleSetActive(ks):
    """True when codegen emits a shared-scale-set dispatch."""
    return tdmWaveSeparated(ks) and tdmSharedScaleSet(ks) is not None


def tdmPapRejectReason(ks):
    """Why PrefetchAcrossPersistent cannot ride this grouping, or None."""
    shared = tdmScaleSharesDataSet(ks)
    if not shared:
        return None
    return ("PrefetchAcrossPersistent needs each TDM scale on its own descriptor set; the %s "
            "grouping (%s) aliases tdmMXSAGroup0/tdmMXSBGroup0 onto tdmAGroup0/tdmBGroup0 and "
            "leaves tdmMXSAMXSBIncs unallocated. Use TDMFuse=0 or disable PrefetchAcrossPersistent."
            % (tdmGrouping(ks).name,
               " + ".join("{%s}" % ",".join(g) for g in shared)))


def tdmGroupPartner(ks, tc, fallback):
    """Return the other member of a two-member set, or `fallback`."""
    for group in tdmGrouping(ks).groups:
        if tc in group and len(group) == 2:
            return group[0] if group[1] == tc else group[1]
    return fallback


class TdmArrangementNotEmittable(ValueError):
    """The writer cannot encode a grouping's wave partition."""


def _waveShares(numWaves, numMembers, layout):
    """(waves, numComp) for each member of a group of `numMembers`."""
    if numMembers <= 1:
        return ((tuple(range(numWaves)), numWaves),)
    if layout == "parity":
        # numComp is numWaves // numMembers, not the share length: at NumWaves=1
        # the odd member of a two-member group gets 0 components.
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


def tdmWavePartition(ks, tc):
    """Return (`numComp`, wave indices) for tensor `tc`."""
    numWaves = ks.get("NumWaves", 1)
    grouping = tdmGrouping(ks)
    layout = grouping.layout
    for group in grouping.groups:
        if tc in group:
            waves, numComp = _waveShares(numWaves, len(group), layout)[group.index(tc)]
            return numComp, waves
    # A tensor no row names: the default two-way parity. Metadata rides the
    # even arm, the one issueLoad admits it on.
    numComp = numWaves // 2
    isAArm = tc.endswith("A") or tc == "Metadata"
    return numComp, tuple(w for w in range(numWaves) if (w % 2 == 0) == isAArm)


def tdmWaveComponents(ks, tc):
    """Return (`numComp`, WaveIdx right shift); None means WaveIdx is unused."""
    numComp, waves = tdmWavePartition(ks, tc)
    if numComp == 1:
        return numComp, None
    if waves == tuple(range(numComp)):
        return numComp, 0
    if numComp < 2:
        # numComp == 0: no wave carries this member, so the id is never read.
        return numComp, 1
    if tuple(w >> 1 for w in waves) == tuple(range(numComp)):
        return numComp, 1
    raise TdmArrangementNotEmittable(
        "tensor %s rides waves %s over %d components, and no right-shift of WaveIdx maps "
        "that onto components 0..%d: shifting by one gives %s"
        % (tc, waves, numComp, numComp - 1, tuple(w >> 1 for w in waves)))


def tdmSoleWave(ks, tc):
    """Return the sole wave carrying `tc`."""
    _, waves = tdmWavePartition(ks, tc)
    if len(waves) != 1:
        raise TdmArrangementNotEmittable(
            "the shared-set increment selects one wave per member; %s rides waves %s"
            % (tc, waves))
    return waves[0]


def tdmWaveRangeText(ks, tc):
    """`tc`'s wave share as comment text: "wave 2", "waves 0-1", "waves 0,2"."""
    _, waves = tdmWavePartition(ks, tc)
    if len(waves) == 1:
        return "wave %d" % waves[0]
    if waves != tuple(range(waves[0], waves[-1] + 1)):
        return "waves %s" % ",".join(str(w) for w in waves)
    return "waves %d-%d" % (waves[0], waves[-1])
