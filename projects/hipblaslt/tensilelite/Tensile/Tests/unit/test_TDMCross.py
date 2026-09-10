################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for TDMCross -- which wave issues which member of a TDM group.

TDMCross is a knob on the wave *assignment*. TDMFuse picks the grouping
(which tensors share one fused `tensor_load_to_lds`); this picks how a group's
members are laid onto waves. The two are tested apart on purpose: test_TDMFuse.py
owns the groupings and their rejections, this file owns the arrangement over them.

The load-bearing test here is test_default_is_the_identity_on_the_shipped_partition.
Every kernel of the b8-3 silicon evidence chain was generated at TDMCross=0,
and that evidence cannot be regenerated, so byte-identity at the default is a
gate rather than a nice-to-have. Asserting it against a frozen copy of the
pre-refactor formula makes the guarantee a precondition of the pure function
instead of an after-the-fact corpus diff.

Related coverage that is not this module:
  the groupings themselves and their rejects -> test_TDMFuse.py
  thick/thin order in emitted assembly       -> test_DecouplePGR.py
  kernel-name tokens                         -> characterization/Naming/
"""
import copy
import itertools

import pytest

from Tensile.Components.DecouplePGR import dcpThickThinIssueOrder
from Tensile.Components.TDMFuse import (
    TDM_GROUPS,
    TDM_TENSORS,
    TDM_CROSS_CROSSED,
    TDM_CROSS_DEFAULT,
    _crossGroups,
    liveGroups,
    tdmMemberIsLive,
    partitionedGroups,
    tdmDataTensorsShareAWave,
    tdmFuseAMx,
    tdmFusePaired,
    tdmGroupPartner,
    tdmGrouping,
    tdmWaveAssignment,
    tdmWaveComponents,
    tdmCrossRejectReason,
    tdmWaveIssueOrder,
    tdmWaveLdsBytes,
    tdmWavePartition,
)
from Tensile.Common.GlobalParameters import defaultSolution

pytestmark = pytest.mark.unit

_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))
_NO_MX_ON_B = {"MacDataTypeB": "F8", "DataTypeMXSB": "E8", "MXBlockB": 0}
_ONE_WAVE_MI = [16, 16, 128, 1, 1, 2, 16, 1, 1]
_ONE_WAVE_WG = [32, 1, 1]


def _ks(fuse=0, cross=TDM_CROSS_DEFAULT, numWaves=4, pgrA=2, pgrB=2,
        mxBlockA=32, mxBlockB=32, **overrides):
    ks = {
        "TDMFuse": fuse,
        "TDMCross": cross,
        "TDMInst": 3,
        "TDMSplit": False,
        "NumWaves": numWaves,
        "UseSubtileImpl": False,
        "PrefetchGlobalRead": max(pgrA, pgrB),
        "PrefetchGlobalReadA": pgrA,
        "PrefetchGlobalReadB": pgrB,
        "ProblemType": {"MXBlockA": mxBlockA, "MXBlockB": mxBlockB},
    }
    ks.update(overrides)
    return ks


# ---------------------------------------------------------------------------
# The identity-at-default invariant.
# ---------------------------------------------------------------------------
def _shippedWavePartition(ks, tc):
    """A frozen copy of the wave partition as it shipped, before the table.

    Deliberately duplicated rather than imported: an oracle that is the code
    under test proves nothing. If the table is ever edited so that the default
    arrangement moves, this is the assertion that has to be consciously
    rewritten, and rewriting it is the moment to notice that 218 kernels and
    the whole b8-3 evidence chain move with it.
    """
    numWaves = ks.get("NumWaves", 1)
    if tdmFuseAMx(ks):
        if tc == "A":
            return 2, (0, 1)
        if tc == "MXSA":
            return 1, (2,)
        if tc == "MXSB":
            return 1, (3,)
        if tc == "B":
            return numWaves, tuple(range(numWaves))
    if tdmFusePaired(ks):
        isEvenArm = tc in ("A", "MXSB")
        return numWaves // 2, tuple(
            w for w in range(numWaves) if (w % 2 == 0) == isEvenArm)
    numComp = numWaves // 2
    isAArm = tc.endswith("A")
    return numComp, tuple(w for w in range(numWaves) if (w % 2 == 0) == isAArm)


def _everyShape():
    """The cross-product of everything the partition can key on."""
    for fuse, numWaves, mxA, mxB, pgrA, pgrB, split, subtile, inst in itertools.product(
            (0, 1, 2), (1, 2, 4, 8), (0, 32), (0, 32), (1, 2), (1, 2),
            (False, True), (False, True), (1, 2, 3)):
        yield _ks(fuse=fuse, numWaves=numWaves, mxBlockA=mxA, mxBlockB=mxB,
                  pgrA=pgrA, pgrB=pgrB, TDMSplit=split, UseSubtileImpl=subtile,
                  TDMInst=inst)


@pytest.mark.parametrize("tc", TDM_TENSORS)
def test_default_is_the_identity_on_the_shipped_partition(tc):
    """THE gate. At TDMCross=0 the assignment reproduces the shipped wave
    partition exactly, for every shape the partition keys on -- so
    _dcpThickThinIssueOrder is handed exactly today's inputs and codegen is
    byte-identical."""
    for ks in _everyShape():
        assert tdmWavePartition(ks, tc) == _shippedWavePartition(ks, tc), ks


@pytest.mark.parametrize("tc", TDM_TENSORS)
def test_default_wave_components_are_unchanged(tc):
    """tdmWaveComponents is what the writer branches on (SMov 0 / SMov WaveIdx /
    SLshr), so pinning the partition alone would leave the emitted instruction
    unpinned at NumWaves=1, where a share of length one and a component count
    of zero disagree."""
    for ks in _everyShape():
        numComp, waves = _shippedWavePartition(ks, tc)
        expected = (numComp, None) if numComp == 1 else (
            numComp, 0 if waves == tuple(range(numComp)) else 1)
        assert tdmWaveComponents(ks, tc) == expected, ks


def test_default_assignment_is_the_inverse_of_the_shipped_partition():
    """The per-wave view and the per-tensor view are the same fact."""
    for ks in _everyShape():
        ordered = [tc for group in tdmGrouping(ks).groups for tc in group]
        expected = {
            w: tuple(tc for tc in ordered
                     if tdmMemberIsLive(ks, tc)
                     and w in _shippedWavePartition(ks, tc)[1])
            for w in range(ks["NumWaves"])
        }
        assert tdmWaveAssignment(ks) == expected, ks


# ---------------------------------------------------------------------------
# What crossing actually does.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("fuse, expected", [
    (0, {0: ("A", "MXSB"), 1: ("B", "MXSA"), 2: ("A", "MXSB"), 3: ("B", "MXSA")}),
    (1, {0: ("A", "B"), 1: ("MXSA", "MXSB"), 2: ("A", "B"), 3: ("MXSA", "MXSB")}),
])
def test_crossing_swaps_the_second_group_against_the_first(fuse, expected):
    """TDMFuse=0's groups are {A,B} and {MXSA,MXSB}, so crossing hands each wave
    one tensor and the *other* tensor's scale -- one large item and one small,
    which is the load-balance arrangement.

    TDMFuse=1's groups are already {A,MXSA} and {MXSB,B} laid down crossed, so
    crossing it *un*-crosses: both data tensors land on the even waves and both
    scales on the odd. That is the arm where a wave issues only scales, finishes
    early and waits -- kept because it is the control the load-balance claim
    needs, not because it is expected to win."""
    assert tdmWaveAssignment(_ks(fuse=fuse, cross=TDM_CROSS_CROSSED)) == expected


def test_crossing_is_its_own_inverse():
    """Crossing twice is not expressible through the parameter -- it has one
    crossed value -- but the operation it names is an involution, which is what
    makes {0, 1} the whole space for a two-group grouping and why widening to a
    third value would need a third arrangement, not a third flag."""
    for grouping in TDM_GROUPS.values():
        assert _crossGroups(_crossGroups(grouping.groups)) == grouping.groups
    for fuse in (0, 1):
        assert (tdmWaveAssignment(_ks(fuse=fuse, cross=TDM_CROSS_CROSSED))
                != tdmWaveAssignment(_ks(fuse=fuse)))


def test_crossing_moves_no_tensor_off_the_waves_altogether():
    """Every live member is still issued by exactly the same number of waves."""
    for fuse in (0, 1):
        plain = tdmWaveAssignment(_ks(fuse=fuse))
        crossed = tdmWaveAssignment(_ks(fuse=fuse, cross=TDM_CROSS_CROSSED))
        for tc in TDM_TENSORS:
            assert (sum(tc in m for m in plain.values())
                    == sum(tc in m for m in crossed.values()) > 0)


def test_crossed_default_grouping_balances_every_wave():
    """The point of the exercise: at TDMFuse=0 crossed, no wave is handed two
    data tensors or two scales."""
    for members in tdmWaveAssignment(_ks(fuse=0, cross=TDM_CROSS_CROSSED)).values():
        assert sum(1 for tc in members if tc.startswith("MXS")) == 1
        assert sum(1 for tc in members if not tc.startswith("MXS")) == 1


# ---------------------------------------------------------------------------
# Rejections, all derived from the group structure.
# ---------------------------------------------------------------------------
def test_default_is_never_rejected():
    for ks in _everyShape():
        assert tdmCrossRejectReason(ks) is None, ks


def test_two_partitioned_groups_admit_crossing():
    """MX_AB is the grouping whose crossed arrangement is implemented."""
    ks = _ks(fuse=0, cross=TDM_CROSS_CROSSED)
    assert len(partitionedGroups(ks)) == 2
    assert tdmCrossRejectReason(ks) is None


def test_paired_has_two_groups_but_its_crossing_is_not_implemented():
    """Two partitioned groups is necessary for crossing but not sufficient.

    The paired sets are {A,MXSA} and {MXSB,B}, one member of each on either
    parity. Crossing asks for both data tensors on one parity and both scales
    on the other, and _tdmPairedParityOrder answers within its own argument
    pair, so it cannot express that arrangement -- it invents an even/odd pair
    from argument position instead. Initialisation and the tail then program
    different descriptors, which shows up as wrong results at every K that
    leaves a tail while K dividing DepthU stays clean.
    """
    ks = _ks(fuse=1, cross=TDM_CROSS_CROSSED)
    assert len(partitionedGroups(ks)) == 2
    assert "paired grouping is not implemented" in tdmCrossRejectReason(ks)


def test_crossing_needs_the_two_data_tensors_at_one_prefetch_depth():
    """Decoupled prefetch depths pair each scale with a data tensor before the
    crossing is applied, so the two disagree about which scale belongs to
    which tensor. Wrong at every size, unlike the tail-only paired defect."""
    ks = _ks(fuse=0, cross=TDM_CROSS_CROSSED, pgrA=1, pgrB=2)
    assert "decoupled prefetch depths" in tdmCrossRejectReason(ks)
    assert tdmCrossRejectReason(_ks(fuse=0, cross=TDM_CROSS_CROSSED,
                                    pgrA=2, pgrB=2)) is None


def test_amx_has_one_partitioned_group_so_nothing_to_cross():
    """TDMFuse=2 is {A,MXSA,MXSB} plus B on its own. B is a one-member group,
    issued by every wave, so it pins nothing -- there is no second group to
    cross the shared set against. The mirrored grouping B_MX inherits this
    without a line of code, because the count is taken over the table row."""
    ks = _ks(fuse=2, cross=TDM_CROSS_CROSSED)
    assert partitionedGroups(ks) == (("A", "MXSA", "MXSB"),)
    assert "leaves 1 group(s) split across waves" in tdmCrossRejectReason(ks)


@pytest.mark.parametrize("mxBlockA, mxBlockB", [(0, 32), (32, 0), (0, 0)])
def test_a_scaleless_type_degenerates_and_is_rejected(mxBlockA, mxBlockB):
    """Without MX scales the {MXSA,MXSB} group has no live member left, so the
    grouping is {A,B} alone -- exactly what the TDMFuse=2 reject already says
    in prose, "without them the group is just {A}". No type test is needed: the
    group count derives it."""
    ks = _ks(fuse=0, cross=TDM_CROSS_CROSSED,
             mxBlockA=mxBlockA, mxBlockB=mxBlockB)
    assert len(partitionedGroups(ks)) < 2
    assert "nothing to cross" in tdmCrossRejectReason(ks)


@pytest.mark.parametrize("numWaves", [0, 1])
def test_one_wave_has_no_waves_to_cross_between(numWaves):
    reason = tdmCrossRejectReason(
        _ks(fuse=0, cross=TDM_CROSS_CROSSED, numWaves=numWaves))
    assert "needs wave-separated TDM" in reason


def test_reject_reason_names_the_grouping_not_the_tdmfuse_integer():
    """Our integers collide with the parallel branch's, so a message that named
    one would be wrong the moment the tables merge."""
    reason = tdmCrossRejectReason(_ks(fuse=2, cross=TDM_CROSS_CROSSED))
    assert "A_MX" in reason
    assert "TDMFuse=2" not in reason


def test_a_declined_grouping_falls_back_before_the_cross_is_judged():
    """TDMFuse=1 on a subtile solution is declined by tdmFusePaired, so the
    grouping really is the default one and crossing is judged against that --
    not against the paired grouping the parameter asked for. Precedence: the
    grouping settles first, the arrangement second."""
    ks = _ks(fuse=1, cross=TDM_CROSS_CROSSED, UseSubtileImpl=True)
    assert tdmGrouping(ks).name == "MX_AB"


# ---------------------------------------------------------------------------
# Per-wave LDS accounting.
# ---------------------------------------------------------------------------
def _ldsKs(fuse=0, cross=TDM_CROSS_DEFAULT, depthU=256, mt0=64, mt1=512,
           pgrA=2, pgrB=2):
    ks = _ks(fuse=fuse, cross=cross, pgrA=pgrA, pgrB=pgrB)
    ks.update({
        "DepthU": depthU, "MacroTile0": mt0, "MacroTile1": mt1,
        "LdsPadA": 0, "LdsPadB": 0, "LdsPadMXSA": 0, "LdsPadMXSB": 0,
        "LdsBlockSizePerPadA": 0, "LdsBlockSizePerPadB": 0,
        "LdsBlockSizePerPadMXSA": 0, "LdsBlockSizePerPadMXSB": 0,
        "MaxLDS": 327680,
    })
    ks["ProblemType"].update({
        "MacDataTypeA": "F8", "MacDataTypeB": "F4",
        "DataTypeA": "F8", "DataTypeB": "F4", "DataType": "F8",
    })
    return ks


def test_per_wave_lds_is_reported_per_wave():
    perWave = tdmWaveLdsBytes(_ldsKs())
    assert set(perWave) == {0, 1, 2, 3}
    assert all(v > 0 for v in perWave.values())


@pytest.mark.parametrize("fuse", [0, 1])
def test_crossing_conserves_total_lds(fuse):
    """The capacity limit is per workgroup, and crossing only moves which wave
    fills which part, so the total cannot move. This is why the 327680 B/CU
    check is not re-derived per wave: re-checking it per arrangement can only
    find what the whole-workgroup check already found."""
    plain = tdmWaveLdsBytes(_ldsKs(fuse=fuse))
    crossed = tdmWaveLdsBytes(_ldsKs(fuse=fuse, cross=TDM_CROSS_CROSSED))
    assert sum(plain.values()) == sum(crossed.values())


def test_crossing_evens_out_the_per_wave_share_at_the_default_grouping():
    """The measurable content of the load-balance claim, stated as an
    invariant: crossing narrows the spread between the busiest and the idlest
    wave. Asymmetric tiles are the case that matters -- MT0 64 against MT1 512
    is where a scale-only wave finishes early and then sits on the barrier."""
    def spread(ks):
        v = tdmWaveLdsBytes(ks).values()
        return max(v) - min(v)
    assert spread(_ldsKs(fuse=0, cross=TDM_CROSS_CROSSED)) \
        < spread(_ldsKs(fuse=0))


def test_block_size_is_not_macrotile_alone():
    """Guard against anyone re-deriving the per-wave share from the tile shape:
    element size, MX scale block, pad and alignment all enter it."""
    base = tdmWaveLdsBytes(_ldsKs())
    padded = _ldsKs()
    padded["LdsPadA"] = 8
    assert tdmWaveLdsBytes(padded) != base
    deeper = _ldsKs(depthU=512)
    assert sum(tdmWaveLdsBytes(deeper).values()) > sum(base.values())


def test_lds_share_scales_with_the_decoupled_block_count():
    """A tensor held at two LDS blocks costs twice, which is what makes the
    divergent pair visible in the per-wave numbers at all."""
    equal = tdmWaveLdsBytes(_ldsKs(pgrA=2, pgrB=2))
    thinA = tdmWaveLdsBytes(_ldsKs(pgrA=1, pgrB=2))
    assert sum(thinA.values()) < sum(equal.values())


def test_pgr2_at_mt64x512x512_exceeds_the_cu_limit():
    """Calibration against a measured figure: PGR2 is LDS-unbuildable at
    MT64x512 DepthU 512 because the requirement lands above 327680."""
    total = sum(tdmWaveLdsBytes(_ldsKs(depthU=512, mt0=64, mt1=512)).values())
    assert total > 327680


# ---------------------------------------------------------------------------
# Layer 2: the pure ordering function.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("blocks, expected", [
    ((1, 2), ("B", "A")),
    ((2, 1), ("A", "B")),
    ((2, 2), ("A", "B")),
    ((1, 1), ("A", "B")),
])
def test_layer2_orders_thick_first_and_breaks_ties_by_input_order(blocks, expected):
    """The (2,2) and (1,1) ties are the cases the old `>=` comparison pinned;
    a stable sort keeps them, for any number of items rather than exactly two."""
    blkA, blkB = blocks
    assert dcpThickThinIssueOrder(((blkA, "A"), (blkB, "B"))) == expected


def test_layer2_is_pure_and_takes_a_set_of_items_not_a_pair():
    """It orders whatever one wave was handed. Nothing about tensors, waves or
    groupings reaches it, which is what makes it testable without a writer."""
    items = ((1, "MXSA"), (2, "A"), (2, "B"), (1, "MXSB"))
    assert dcpThickThinIssueOrder(items) == ("A", "B", "MXSA", "MXSB")
    assert dcpThickThinIssueOrder(()) == ()
    assert dcpThickThinIssueOrder(((3, "solo"),)) == ("solo",)


@pytest.mark.parametrize("pgrA, pgrB, expected", [
    (1, 2, ("B", "A")),
    (2, 1, ("A", "B")),
    (2, 2, ("A", "B")),
    (1, 1, ("A", "B")),
])
def test_layer3_reproduces_the_shipped_issue_order(pgrA, pgrB, expected):
    assert tdmWaveIssueOrder(_ks(pgrA=pgrA, pgrB=pgrB), "A", "B") == expected


def test_layer3_reorders_its_arguments_rather_than_returning_literals():
    tpA, tpB = object(), object()
    assert tdmWaveIssueOrder(_ks(pgrA=1, pgrB=2), tpA, tpB) == (tpB, tpA)
    assert tdmWaveIssueOrder(_ks(pgrA=2, pgrB=1), tpA, tpB) == (tpA, tpB)


def test_issue_order_is_unchanged_by_crossing():
    """Follows from the assignment, and worth pinning because it is the reason
    crossing is safe to land: A and B share a wave only under a grouping whose
    second group is unpartitioned, and that grouping rejects crossing anyway."""
    for fuse, pgrA, pgrB in itertools.product((0, 1), (1, 2), (1, 2)):
        plain = tdmWaveIssueOrder(_ks(fuse=fuse, pgrA=pgrA, pgrB=pgrB), "A", "B")
        crossed = tdmWaveIssueOrder(
            _ks(fuse=fuse, cross=TDM_CROSS_CROSSED, pgrA=pgrA, pgrB=pgrB), "A", "B")
        assert plain == crossed


@pytest.mark.parametrize("fuse, shared", [(0, False), (1, False), (2, True)])
def test_data_tensors_share_a_wave_only_under_an_unpartitioned_group(fuse, shared):
    """The finding that makes the per-wave question equal the global one
    everywhere it is currently asked: only TDMFuse=2 puts A and B on one wave,
    and TDMFuse=2 is equal-pair-only, so its two weights tie."""
    assert tdmDataTensorsShareAWave(_ks(fuse=fuse)) is shared


# ---------------------------------------------------------------------------
# Grouping table hygiene.
# ---------------------------------------------------------------------------
def test_every_row_covers_every_tensor_exactly_once():
    """A member in two groups would be issued twice and would inflate the
    per-wave LDS share, which is the one way an arrangement could break the
    conservation the Solution-level guard relies on."""
    for name, grouping in TDM_GROUPS.items():
        members = [tc for group in grouping.groups for tc in group]
        assert sorted(members) == sorted(TDM_TENSORS), name


def test_rows_are_keyed_on_name_and_their_indices_are_only_labels():
    """Our TDMFuse integers are baked into shipped solution names and the frozen
    00_Final.yaml, and they collide with the parallel branch's numbering, so the
    name is the only stable key. Indices must stay unique but nothing may sort
    or index by them."""
    indices = [g.index for g in TDM_GROUPS.values()]
    assert len(set(indices)) == len(indices)
    assert all(name == grouping.name for name, grouping in TDM_GROUPS.items())


def test_group_partner_is_defined_only_for_two_member_sets():
    paired = _ks(fuse=1)
    assert tdmGroupPartner(paired, "A", "fallback") == "MXSA"
    assert tdmGroupPartner(paired, "MXSB", "fallback") == "B"
    assert tdmGroupPartner(_ks(fuse=0), "A", "fallback") == "B"
    # The three-member shared set has no unambiguous partner.
    assert tdmGroupPartner(_ks(fuse=2), "A", "fallback") == "fallback"


def test_live_groups_drop_dead_members_but_keep_the_wave_layout():
    """A descriptor set keeps its shape when a member is absent; the absent
    member simply never issues. Sizing shares off the live members instead
    would move the survivor onto different waves and break byte-identity."""
    ks = _ks(fuse=0, mxBlockB=0)
    assert liveGroups(ks) == (("A", "B"), ("MXSA",))
    assert tdmWavePartition(ks, "MXSA") == (2, (0, 2))
    assert "MXSB" not in {tc for m in tdmWaveAssignment(ks).values() for tc in m}


# ---------------------------------------------------------------------------
# Naming. The default must add no token: a token would rename all 218 kernels
# and shift dedup.
# ---------------------------------------------------------------------------
def _nameState(**overrides):
    """A solution state complete enough to drive _getName and the dedup key.

    Same shape as the Naming characterization suite's fixture; kept local so
    this file's naming assertions do not depend on a conftest in another
    directory.
    """
    from Tensile.SolutionStructs.Problem import ProblemType

    state = {
        "ProblemType": ProblemType({"DataType": 0}, False),
        "GlobalSplitU": 1,
        "UseCustomMainLoopSchedule": False,
        "WorkGroupMapping": 1,
        "WorkGroupMappingXCC": 1,
        "WorkGroupMappingXCCGroup": 0,
        "StaggerU": 0,
        "StaggerUStride": 0,
        "StaggerUMapping": 0,
        "GlobalSplitUCoalesced": False,
        "GlobalSplitUWorkGroupMappingRoundRobin": False,
        "SFCWGM": [],
        "SpaceFillingAlgo": [],
        "MacroTile0": 128,
        "MacroTile1": 128,
        "DepthU": 16,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstB": 1,
        "MIWaveTile": [2, 2],
    }
    state.update(overrides)
    return state


def test_default_adds_no_token_to_the_kernel_name():
    """A token at the default would rename all 218 kernels and shift dedup.
    Precedent for omission: no shipped row carries _TDMF0_."""
    from Tensile.SolutionStructs import Naming as N

    assert "TDMC" not in N.getKernelNameMin(
        _nameState(TDMCross=TDM_CROSS_DEFAULT), splitGSU=False)


def test_crossed_adds_a_token_so_it_cannot_dedup_onto_the_default():
    from Tensile.SolutionStructs import Naming as N

    plain = N.getKernelNameMin(
        _nameState(TDMCross=TDM_CROSS_DEFAULT), splitGSU=False)
    crossed = N.getKernelNameMin(
        _nameState(TDMCross=TDM_CROSS_CROSSED), splitGSU=False)
    assert "TDMC1" in crossed
    assert plain != crossed


def test_the_default_name_is_identical_to_one_with_the_key_absent():
    """The strongest form of the naming guarantee: adding the parameter to
    validParameters must not perturb a solution that never mentions it. The
    dedup key is built from the Full required set, which is validParameters'
    keys, so a token here would move 20 kernels that currently vanish by
    dedup."""
    from Tensile.SolutionStructs import Naming as N

    absent = _nameState()
    assert "TDMCross" not in absent
    assert (N.getKernelNameMin(absent, splitGSU=False)
            == N.getKernelNameMin(
                _nameState(TDMCross=TDM_CROSS_DEFAULT), splitGSU=False))
    assert (N.getSolutionNameFull(absent, splitGSU=False)
            == N.getSolutionNameFull(
                _nameState(TDMCross=TDM_CROSS_DEFAULT), splitGSU=False))
