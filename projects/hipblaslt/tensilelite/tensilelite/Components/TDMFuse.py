# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TDM descriptor grouping (TDMFuse): which tensors share one descriptor set.

  TDMFuse=0  {A,B} + {MXSA,MXSB}   default two-way parity
  TDMFuse=1  {A,MXSA} + {MXSB,B}   paired: each scale on a data tensor
  TDMFuse=2  {A,MXSA,MXSB} + {B}   block split; with both scales 2/1/1 on 4 waves,
                                   and on 2 waves MXSA and MXSB share wave 1
  TDMFuse=3  {B,MXSA,MXSB} + {A}   the mirror of 2, same split and wave counts

A set splits its waves over the members the problem carries (tdmWaveShares), so
the waves of a missing MX scale go to the rest of its set. Without MX scales
TDMFuse 1-3 all reduce to {A} + {B}: A and B each ride every wave and the three
rows issue the same loads (tdmWaveMask). TDMFuse=0 keeps A and B on one set,
split by wave parity. With one scale, that scale takes every wave of the
TDMFuse=0 scale set, shares its data tensor's set with it on TDMFuse 1-3, and
the other data tensor of TDMFuse 1-3 rides every wave of a set of its own.
TDMFuse 1-3 need more than one wave: a one-wave kernel, like a subtile kernel at
every wave count, keeps one descriptor per tensor.

TDM_FUSE_GROUPING maps stable, name-bearing integers to TDM_GROUPS rows.
"""

import functools
import itertools


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
    """Preconditions of every row but the default, with or without MX scales.

    Includes writer-visible TDMSplit states; the sets need more than one wave
    to split over.
    """
    if not tdmBothTensors(ks):
        return False
    if ks.get("TDMSplit") or ks.get("UseSubtileImpl"):
        return False
    return ks.get("NumWaves", 1) > 1


def _acceptSharedScaleSet(ks):
    """Accept A_MX/B_MX; with both scales only the emittable 2/1/1 partitions on 2 and 4 waves."""
    bothScales = tdmLiveScales(ks) == TDM_SCALE_TENSORS
    return _tdmFuseCanShareDescriptors(ks) and (not bothScales or ks.get("NumWaves", 1) in (2, 4))


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
    """TDMFuse=1 with both MX scales: {A,MXSA} and {MXSB,B}. NumWaves>1.

    The paired writer arms assume both scales. With one scale the row is the
    scale's data set plus a lone data set and takes the shared-scale arms
    (tdmSharedScaleSet); without scales it is {A} + {B} on the every-wave arms
    (tdmPairOnEveryWave).
    """
    return tdmGrouping(ks).name == "paired" and tdmLiveScales(ks) == TDM_SCALE_TENSORS


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
    """Each set's live members, dropping sets with none; waves come from tdmWaveShares."""
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


def tdmLiveScales(ks):
    """The MX scales this problem carries, in TDM_SCALE_TENSORS order."""
    return tuple(tc for tc in TDM_SCALE_TENSORS if tdmMemberIsLive(ks, tc))


def tdmScaleSharesDataSet(ks, grouping=None):
    """Live groups carrying a data tensor and a scale tensor on one set."""
    return tuple(g for g in liveGroups(ks, grouping)
                 if any(tc in TDM_DATA_TENSORS for tc in g)
                 and any(tc in TDM_SCALE_TENSORS for tc in g))


def tdmHintDeciders(members):
    """Positions in `members` whose cache hints decide one tensor_load_to_lds.

    One instruction serves every member of a set, so its th/scope/nv bits are
    instruction-wide. A set that holds a data tensor (A or B) follows its data
    tensors: the scale tensors riding along (MXSA, MXSB) and any other member are
    ignored, so the generated text follows A/B. A set without one, the
    {MXSA, MXSB} set of TDMFuse=0, lets every member decide.
    """
    data = tuple(i for i, tc in enumerate(members) if tc in TDM_DATA_TENSORS)
    return data or tuple(range(len(members)))


def tdmSetGroup(ks, tc):
    """The full group `tc` rides, or None. Not the live one: see liveGroups."""
    for group in tdmGrouping(ks).groups:
        if tc in group:
            return group
    return None


def tdmSetOwner(ks, tc):
    """The member whose name programs the descriptor set that carries `tc`.

    Other members are RegSet aliases; mutate the set through its owner only. A stashed
    member (tdmStashedMembers) answers for itself: its name holds the Group0 words it
    issues with, through the set's Group1.
    """
    group = tdmSetGroup(ks, tc)
    if group is None or tc in tdmStashedMembers(ks):
        return tc
    for member in group:
        if member in TDM_DATA_TENSORS:
            return member
    return group[0]


def tdmSharedScaleSet(ks):
    """Return the one live set carrying a data tensor and the live MX scales, or None.

    TDMFuse 2/3 seat every live scale on the owner's set, and TDMFuse=1 with one
    scale seats it on its data tensor's set; the other data tensor rides a set
    of its own. TDMFuse=1 with both scales puts a scale on each data set and
    takes the paired arms instead (tdmFusePaired), so it has no such set.
    """
    shared = tdmScaleSharesDataSet(ks)
    if len(shared) != 1:
        return None
    group = shared[0]
    if len([tc for tc in group if tc in TDM_DATA_TENSORS]) != 1:
        return None
    return group


def tdmSharedScaleSetOwner(ks):
    """The data tensor whose descriptor set the live MX scales ride, or None."""
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
    """Why PrefetchAcrossPersistent cannot ride this grouping, or None.

    PAP refreshes and re-banks each descriptor set once, through its owner, and refreshes
    the MX scales only as a pair: a lone scale on a data tensor's set has no arm there.
    """
    shared = tdmScaleSharesDataSet(ks)
    if not shared or len(tdmLiveScales(ks)) != 1:
        return None
    return ("PrefetchAcrossPersistent refreshes MX scale descriptors only as a pair, and the %s "
            "grouping (%s) seats the one live MX scale on a data tensor's descriptor set. Put "
            "MX scales on both tensors or on neither, or disable PrefetchAcrossPersistent."
            % (tdmGrouping(ks).name,
               " + ".join("{%s}" % ",".join(g) for g in shared)))


def tdmGroupPartner(ks, tc, fallback):
    """Return the other member of a two-member set whose members are both live, or `fallback`."""
    for group in tdmGrouping(ks).groups:
        if tc in group and len(group) == 2:
            if not all(tdmMemberIsLive(ks, member) for member in group):
                return fallback
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
    if numMembers == 3:
        # The owner takes quarters 1-2 and each scale one more; on two waves both scales
        # land on wave 1, which issues MXSB after MXSA through one set (tdmStashedMembers).
        bounds = ((0, numWaves // 2), (numWaves // 2, numWaves * 3 // 4),
                  (numWaves * 3 // 4, numWaves))
        runs = tuple(tuple(range(lo, max(hi, lo + 1))) for lo, hi in bounds)
        return tuple((run, len(run)) for run in runs)
    # "block": contiguous shares, remainder handed to the leading members.
    base, extra = divmod(numWaves, numMembers)
    widths = tuple(base + (1 if i < extra else 0) for i in range(numMembers))
    starts = itertools.accumulate(widths, initial=0)
    return tuple((tuple(range(start, start + width)), width)
                 for start, width in zip(starts, widths))


def _setWaveShares(group, layout, numWaves, isLive):
    """(waves, numComp) for each member of one descriptor set, in member order.

    The set's waves split over its live members alone, by `layout` and in member
    order, so no wave of a set with a live member idles. A dead member gets no
    wave; with every member live this is the row's full split.
    """
    live = tuple(tc for tc in group if isLive(tc))
    shares = dict(zip(live, _waveShares(numWaves, len(live), layout)))
    return tuple(shares.get(tc, ((), 0)) for tc in group)


def tdmWaveShares(grouping, numWaves, isLive):
    """{member: (numComp, waves)} over every set of `grouping`.

    Reads nothing but its arguments: `isLive(tc)` says which members the
    problem carries.
    """
    return {tc: (numComp, waves)
            for group in grouping.groups
            for tc, (waves, numComp) in zip(
                group, _setWaveShares(group, grouping.layout, numWaves, isLive))}


def tdmWavePartition(ks, tc):
    """Return (`numComp`, wave indices) for tensor `tc`: (0, ()) when no wave issues it."""
    numWaves = ks.get("NumWaves", 1)
    shares = tdmWaveShares(tdmGrouping(ks), numWaves, functools.partial(tdmMemberIsLive, ks))
    if tc in shares:
        return shares[tc]
    # A tensor no row names: the default two-way parity. Metadata rides the
    # even arm, the one issueLoad admits it on.
    numComp = numWaves // 2
    isAArm = tc.endswith("A") or tc == "Metadata"
    return numComp, tuple(w for w in range(numWaves) if (w % 2 == 0) == isAArm)


def tdmWaveMask(ks, tc):
    """The waves that issue tensor `tc`'s TDM load: bit w is set for wave w, 0 for none."""
    return sum(1 << w for w in tdmWavePartition(ks, tc)[1])


def tdmWaveMasks(ks):
    """{tensor: tdmWaveMask} for every live member of the resolved grouping."""
    return {tc: tdmWaveMask(ks, tc) for group in liveGroups(ks) for tc in group}


def tdmStashedMembers(ks):
    """Live members that share a wave with an earlier member of their set, in set order.

    The set stays with the first member on that wave (tdmWaveResident). A stashed member keeps
    only a Group0-shaped stash: its enable word, LDS address and global address, programmed
    on every wave and disabled outside its share. It issues after the resident with the stash
    as its Group0 and the set's Group1 (and Group2), whose member fields are rewritten before
    the load and restored after it (KernelWriterAssembly._tdmStashedIssue). Only TDMFuse 2/3
    with both scales on two waves has one: MXSA and MXSB share wave 1.
    """
    if not tdmWaveSeparated(ks):
        return ()
    own = []
    for group in liveGroups(ks):
        taken = set()
        for tc in group:
            waves = set(tdmWavePartition(ks, tc)[1])
            if waves & taken:
                own.append(tc)
            taken |= waves
    return tuple(own)


def tdmWaveResident(ks, tc):
    """The member of `tc`'s set whose descriptor the set holds on `tc`'s waves.

    That is the first live member, in set order, whose waves include them; a stashed member
    issues after it on those waves. Any other member is its own resident.
    """
    waves = set(tdmWavePartition(ks, tc)[1])
    for member in tdmSetGroup(ks, tc) or ():
        if tdmMemberIsLive(ks, member) and waves <= set(tdmWavePartition(ks, member)[1]):
            return member
    return tc


def tdmScaleKGroups(ks, tc):
    """MatrixInstK-wide K-groups in scale `tc`'s DepthU tile, as the writer counts them; None for data."""
    if tc not in ("MXSA", "MXSB"):
        return None
    mxBlock = (ks.get("ProblemType") or {}).get("MXBlock%s" % tc[-1])
    instK, depthU = ks.get("MatrixInstK"), ks.get("DepthU")
    if not (mxBlock and instK and depthU):
        return None
    return (depthU // mxBlock) // (instK // mxBlock)


def tdmWaveComponents(ks, tc):
    """Return (`numComp`, WaveIdx right shift, wave offset) for tensor `tc`.

    A wave's component is (WaveIdx - offset) >> shift; a None shift means one
    component, so WaveIdx is unused. The offset is nonzero only for a run of
    waves that starts above wave 0, which no shift alone reaches. A scale on more
    waves than it has K-groups (tdmScaleKGroups) gives each K-group 1 << shift waves.
    """
    numComp, waves = tdmWavePartition(ks, tc)
    kGroups = tdmScaleKGroups(ks, tc)
    if (kGroups and numComp > kGroups and numComp % kGroups == 0
            and waves == tuple(range(waves[0], waves[0] + numComp))):
        repeat = numComp // kGroups
        if repeat & (repeat - 1) == 0:
            # The writer's row split of a scale tile (numMxKGroups < numComp) is wrong wherever the
            # K tail or a partial tile cuts a piece, so the extra waves repeat a K-group instead.
            return kGroups, repeat.bit_length() - 1, waves[0]
    if numComp == 1:
        return numComp, None, 0
    if waves == tuple(range(numComp)):
        return numComp, 0, 0
    if numComp < 2:
        # numComp == 0: no wave carries this member, so the id is never read.
        return numComp, 1, 0
    if tuple(w >> 1 for w in waves) == tuple(range(numComp)):
        return numComp, 1, 0
    if waves == tuple(range(waves[0], waves[0] + numComp)):
        return numComp, 0, waves[0]
    raise TdmArrangementNotEmittable(
        "tensor %s rides waves %s over %d components, and no offset and right-shift of "
        "WaveIdx maps that onto components 0..%d: shifting by one gives %s"
        % (tc, waves, numComp, numComp - 1, tuple(w >> 1 for w in waves)))


def tdmRidesEveryWave(ks, tc):
    """True when wave-separated TDM spreads tensor `tc` over every wave."""
    if not tdmWaveSeparated(ks):
        return False
    _, waves = tdmWavePartition(ks, tc)
    return waves == tuple(range(ks.get("NumWaves", 1)))


def tdmPairOnEveryWave(ks, tcA, tcB):
    """True when both tensors of a writer pair ride every wave, so neither needs a wave guard.

    A data tensor rides every wave when it is its set's only live member, and
    TDMFuse 1-3 without MX scales give both A and B such a set; the scale pair
    and every other row keep their guarded paths.
    """
    return tdmRidesEveryWave(ks, tcA) and tdmRidesEveryWave(ks, tcB)


def tdmHalfPlrMaskedIncs(ks):
    """Tensors whose own GlobalReadIncs HalfPLR's last-iteration mask zeroes.

    Parity and shared sets advance through a shadow increment, which the mask zeroes instead. A
    tensor that advances by its own increment loses it, while removeStagger still needs it to
    rebase that tensor's descriptor for the K tail: every live tensor on one wave, where each
    keeps its own Group0, and on more waves both members of an every-wave pair, the data
    tensor beside a shared scale set and a stashed scale there, and a lone scale on
    a parity row. Empty for HalfPLR 0.
    """
    if not ks.get("HalfPLR"):
        return ()
    if ks.get("NumWaves", 1) <= 1:
        return ("A", "B") + tdmLiveScales(ks)
    if tdmSharedScaleSetActive(ks):
        return ("B" if tdmSharedScaleSetOwner(ks) == "A" else "A",) + tdmStashedMembers(ks)
    if tdmPairOnEveryWave(ks, "A", "B"):
        return ("A", "B")
    live = tdmLiveScales(ks)
    return tuple(live) if len(live) == 1 else ()


def tdmSoleWave(ks, tc):
    """Return the sole wave carrying `tc`."""
    _, waves = tdmWavePartition(ks, tc)
    if len(waves) != 1:
        raise TdmArrangementNotEmittable(
            "the shared-set increment selects one wave per member; %s rides waves %s"
            % (tc, waves))
    return waves[0]


def tdmWaveGuard(ks, tc):
    """How a writer admits exactly the waves that carry `tc`.

    ("all",) every wave; ("none",) no wave; ("wave", w) wave w alone; ("below", n)
    waves 0..n-1; ("from", w) waves w..NumWaves-1; ("parity", r) the waves whose
    index is r modulo 2.
    """
    numWaves = ks.get("NumWaves", 1)
    _, waves = tdmWavePartition(ks, tc)
    if not waves:
        return ("none",)
    if waves == tuple(range(numWaves)):
        return ("all",)
    if len(waves) == 1:
        return ("wave", waves[0])
    if waves == tuple(range(len(waves))):
        return ("below", len(waves))
    if waves == tuple(range(waves[0], numWaves)):
        return ("from", waves[0])
    if waves == tuple(range(waves[0] % 2, numWaves, 2)):
        return ("parity", waves[0] % 2)
    raise TdmArrangementNotEmittable(
        "tensor %s rides waves %s of %d, which is neither one wave, a run touching the first "
        "or last wave, nor a parity class" % (tc, waves, numWaves))


def tdmWaveRangeText(ks, tc):
    """`tc`'s wave share as comment text: "wave 2", "waves 0-1", "waves 0,2"."""
    _, waves = tdmWavePartition(ks, tc)
    if len(waves) == 1:
        return "wave %d" % waves[0]
    if waves != tuple(range(waves[0], waves[-1] + 1)):
        return "waves %s" % ",".join(str(w) for w in waves)
    return "waves %d-%d" % (waves[0], waves[-1])
