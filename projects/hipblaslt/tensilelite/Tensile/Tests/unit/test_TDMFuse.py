################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for Tensile.Components.TDMFuse.

Related coverage that is not this module:
  HalfPLR + TDMFuse=1 at a divergent pair -> test_halfplr_streamk_rejects.py
  TDMFuse kernel-name tokens              -> characterization/Naming/test_mut_Naming_char.py
"""
import copy

import pytest

from Tensile.Components.DecouplePGR import decouplePGRBlocks
from Tensile.Components.TDMFuse import tdmBothTensors, tdmFuseAMx, tdmFusePaired
from Tensile.Components.TDMFuse import tdmWaveComponents, tdmWavePartition
from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Common.ValidParameters import validParameters
from Tensile.Components import TDMFuse as TF
from Tensile.Components import DecouplePGR as DP
from Tensile.Components import TDMCross as TC
import Tensile.KernelWriterAssembly as KWAMod
from Tensile.KernelWriterAssembly import KernelWriterAssembly

pytestmark = pytest.mark.unit

_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))
_TENSORS = ("A", "MXSA", "MXSB", "B")
_NO_MX_ON_B = {"MacDataTypeB": "F8", "DataTypeMXSB": "E8", "MXBlockB": 0}
_ONE_WAVE_MI = [16, 16, 128, 1, 1, 2, 16, 1, 1]
_ONE_WAVE_WG = [32, 1, 1]
# strict, matching pytest.ini's xfail_strict: TDMSplit is rejected
# unconditionally today, so these must fail. If it is ever re-enabled they XPASS
# and strict turns that into a failure, which is the notification we want.
_TDMSPLIT_DISABLED = pytest.mark.xfail(
    reason="TDMSplit is currently disabled upstream (PR #10911)", strict=True)


def _ks(fuse=1, pgrA=1, pgrB=2, **overrides):
    ks = {
        "TDMFuse": fuse,
        "TDMInst": 3,
        "TDMSplit": False,
        "enableTDMA": True,
        "enableTDMB": True,
        "NumWaves": 4,
        "UseSubtileImpl": False,
        "PrefetchGlobalRead": max(pgrA, pgrB),
        "PrefetchGlobalReadA": pgrA,
        "PrefetchGlobalReadB": pgrB,
        "ProblemType": {"MXBlockA": 32, "MXBlockB": 32},
    }
    ks.update(overrides)
    return ks


def test_tdm_both_tensors():
    assert tdmBothTensors({"TDMInst": 3}) is True
    assert tdmBothTensors({"TDMInst": 1}) is False
    assert tdmBothTensors({"TDMInst": 2}) is False
    assert tdmBothTensors({"TDMInst": 0}) is False


def test_fuse_predicates_are_exclusive():
    assert tdmFusePaired(_ks(fuse=1)) is True
    assert tdmFuseAMx(_ks(fuse=1)) is False
    assert tdmFuseAMx(_ks(fuse=2)) is True
    assert tdmFusePaired(_ks(fuse=2)) is False
    assert tdmFusePaired(_ks(fuse=0)) is False
    assert tdmFuseAMx(_ks(fuse=0)) is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"NumWaves": 1},
        {"UseSubtileImpl": True},
        {"TDMSplit": True},
        {"TDMInst": 1},
        {"TDMInst": 2},
        {"ProblemType": {"MXBlockA": 0, "MXBlockB": 32}},
        {"ProblemType": {"MXBlockA": 32, "MXBlockB": 0}},
    ],
)
def test_paired_declines_outside_its_envelope(overrides):
    assert tdmFusePaired(_ks(**overrides)) is False
    assert tdmFuseAMx(_ks(fuse=2, **overrides)) is False


@pytest.mark.parametrize("pgrA, pgrB", [(1, 2), (2, 1), (2, 2), (0, 2), (1, 1)])
def test_fuse_predicates_do_not_key_on_block_counts(pgrA, pgrB):
    assert tdmFusePaired(_ks(pgrA=pgrA, pgrB=pgrB)) is True
    assert tdmFuseAMx(_ks(fuse=2, pgrA=pgrA, pgrB=pgrB)) is True


@pytest.mark.parametrize("numWaves", [2, 4, 8])
def test_paired_holds_at_every_wave_count_parity_can_split(numWaves):
    assert tdmFusePaired(_ks(NumWaves=numWaves)) is True
    assert tdmFuseAMx(_ks(fuse=2, NumWaves=numWaves)) is (numWaves == 4)


@pytest.mark.parametrize(
    "tc, waves",
    [("A", (0, 2)), ("MXSA", (1, 3)), ("MXSB", (0, 2)), ("B", (1, 3))],
)
def test_paired_wave_assignment(tc, waves):
    numComp, got = tdmWavePartition(_ks(), tc)
    assert (numComp, got) == (2, waves)
    assert tdmWaveComponents(_ks(), tc) == (2, 1)


@pytest.mark.parametrize(
    "tc, numComp, waves, shift",
    [
        ("A", 2, (0, 1), 0),
        ("MXSA", 1, (2,), None),
        ("MXSB", 1, (3,), None),
        ("B", 4, (0, 1, 2, 3), 0),
    ],
)
def test_amx_wave_assignment(tc, numComp, waves, shift):
    ks = _ks(fuse=2)
    assert tdmWavePartition(ks, tc) == (numComp, waves)
    assert tdmWaveComponents(ks, tc) == (numComp, shift)


def test_paired_only_swaps_scale_parity_against_default():
    paired, default = _ks(fuse=1), _ks(fuse=0)
    assert tdmWavePartition(paired, "A") == tdmWavePartition(default, "A")
    assert tdmWavePartition(paired, "B") == tdmWavePartition(default, "B")
    assert tdmWavePartition(paired, "MXSA") == tdmWavePartition(default, "MXSB")
    assert tdmWavePartition(paired, "MXSB") == tdmWavePartition(default, "MXSA")
    for wave in range(4):
        carried = [tc for tc in _TENSORS if wave in tdmWavePartition(paired, tc)[1]]
        assert len(carried) == 2
        assert sum(1 for tc in carried if "MXS" in tc) == 1


# ---------------------------------------------------------------------------
# Solution wiring. Needs amdclang++ gfx1250; skipped otherwise.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gfx1250_iim():
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.Common.Capabilities import makeIsaInfoMap
    from Tensile.Toolchain.Validators import validateToolchain

    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa("gfx1250")
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx1250")
    return iim


@pytest.fixture(scope="module")
def assembler():
    from Tensile.Toolchain.Assembly import makeAssemblyToolchain
    from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _gp_gfx1250(gfx1250_iim):
    from Tensile.Common.GlobalParameters import globalParameters, assignGlobalParameters
    from Tensile.Common.ValidParameters import validParameters

    saved_gp = copy.deepcopy(dict(globalParameters))
    saved_vp = copy.deepcopy(dict(validParameters))
    saved_ds = copy.deepcopy(dict(defaultSolution))
    defaultSolution.clear()
    defaultSolution.update(copy.deepcopy(_PRISTINE_DEFAULT_SOLUTION))
    assignGlobalParameters({}, gfx1250_iim)
    yield
    globalParameters.clear()
    globalParameters.update(saved_gp)
    validParameters.clear()
    validParameters.update(saved_vp)
    defaultSolution.clear()
    defaultSolution.update(saved_ds)


def _derive(gfx1250_iim, assembler, capsys, **overrides):
    from Tensile.Common.Architectures import gfxToIsa
    from Tensile.SolutionStructs.Solution import Solution
    from Tensile.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    isa = gfxToIsa("gfx1250")
    mi = overrides.pop("MatrixInstruction", [16, 16, 128, 1, 1, 2, 16, 2, 2])
    workGroup = overrides.pop("WorkGroup", [32, 4, 1])
    problemType = {
        "OperationType": "GEMM",
        "MacDataTypeA": "F8",
        "MacDataTypeB": "F4",
        "DataType": "F8",
        "DestDataType": "s",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "MXBlockA": 32,
        "MXBlockB": 32,
        "DataTypeMXSA": "E8",
        "DataTypeMXSB": "E8",
    }
    problemType.update(overrides.pop("ProblemType", {}))
    params = {
        "ProblemType": problemType,
        "ISA": isa,
        "MatrixInstruction": mi,
        "WorkGroup": workGroup,
        "WavefrontSize": 32,
        "DepthU": 256,
        "MaxLDS": 327680,
        "KernelLanguage": "Assembly",
        "TDMInst": 3,
        "MXScaleFormat": "InMemorySwizzle",
        "LDSTrInst": True,
        "TDMFuse": 1,
        "TDMSplit": False,
        "PrefetchGlobalRead": 2,
        "PrefetchGlobalReadA": 2,
        "PrefetchGlobalReadB": 2,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 0,
        "StaggerU": 0,
        "GlobalSplitU": 1,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "InnerUnroll": 1,
        "TransposeLDS": -1,
        "LdsPadA": -1,
        "LdsPadB": -1,
        "LdsBlockSizePerPadA": -1,
        "LdsBlockSizePerPadB": -1,
        "LdsPadMetadata": 0,
        "1LDSBuffer": 0,
        "VectorWidthA": -1,
        "VectorWidthB": -1,
        "StoreVectorWidth": -1,
        "GlobalReadVectorWidthA": -1,
        "GlobalReadVectorWidthB": -1,
        "LocalReadVectorWidth": -1,
        "SourceSwap": False,
        "ExpandPointerSwap": False,
        "StoreRemapVectorWidth": 0,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToVgprSparseMetadata": False,
        "WorkGroupMapping": 1,
    }
    params.update(overrides)
    params.update(matrixInstructionToMIParameters(
        mi, isa, params["WavefrontSize"], problemType, workGroup, gfx1250_iim))
    sol = Solution(params, False, True, False, assembler, gfx1250_iim)
    return sol, capsys.readouterr().out


@pytest.mark.parametrize("fuse", [0, 1, 2])
def test_solution_accepts_each_grouping_at_equal_pair(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse)
    assert sol.get("Valid") is True, out


@pytest.mark.parametrize("pgrA, pgrB", [(1, 2), (2, 1)])
def test_solution_accepts_paired_at_divergent_pair(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, pgrA, pgrB):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=1,
                       PrefetchGlobalReadA=pgrA, PrefetchGlobalReadB=pgrB)
    assert sol.get("Valid") is True, out


@pytest.mark.parametrize("fuse", [1, 2])
def test_solution_accepts_stagger(_gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, StaggerU=32)
    assert sol.get("Valid") is True, out
    assert "requires StaggerU=0" not in out


@pytest.mark.parametrize(
    "fuse, overrides, clause",
    [
        (2, {"PrefetchGlobalReadA": 1, "PrefetchGlobalReadB": 2},
         "TDMFuse=2 requires an equal decoupled pair"),
        (1, {"MatrixInstruction": _ONE_WAVE_MI, "WorkGroup": _ONE_WAVE_WG},
         "TDMFuse=1 splits each of its two descriptor sets by wave parity"),
        (2, {"MatrixInstruction": _ONE_WAVE_MI, "WorkGroup": _ONE_WAVE_WG},
         "2/1/1 split is a remainder policy"),
        (2, {"MatrixInstruction": [16, 16, 128, 1, 1, 2, 16, 2, 1], "WorkGroup": [32, 2, 1]},
         "got NumWaves=2"),
        (1, {"ProblemType": _NO_MX_ON_B},
         "TDMFuse=1 requires MX scales on both tensors"),
        (2, {"ProblemType": _NO_MX_ON_B},
         "TDMFuse=2 names MXSA and MXSB as the two single-wave members"),
        (1, {"ProblemType": {"Sparse": 1}},
         "TDMFuse=1 does not describe the sparse metadata tensor"),
        (2, {"ProblemType": {"Sparse": 1}},
         "TDMFuse=2 does not describe the sparse metadata tensor"),
        (1, {"TDMInst": 1}, "TDMA and TDMB must be enabled simultaneously"),
        (1, {"UseSubtileImpl": True}, "Unable to load MXSB scales using one load per wave"),
    ],
)
def test_solution_rejects_outside_the_grouping(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, overrides, clause):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, **overrides)
    assert sol.get("Valid") is False
    assert clause in out


@pytest.mark.parametrize("fuse", [0, 1, 2])
@_TDMSPLIT_DISABLED
def test_tdmsplit_across_every_grouping(_gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, TDMSplit=True)
    if fuse == 0:
        assert sol.get("Valid") is True, out
        return
    assert sol.get("Valid") is False
    assert "TDMFuse=%d is not available with TDMSplit" % fuse in out


@pytest.mark.parametrize("fuse, predicate", [(1, tdmFusePaired), (2, tdmFuseAMx)])
def test_accepted_solution_matches_the_writer_predicate(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, predicate):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse)
    assert sol.get("Valid") is True, out
    assert predicate(sol._state) is True
    if fuse == 1:
        sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=1,
                           PrefetchGlobalReadA=1, PrefetchGlobalReadB=2)
        assert sol.get("Valid") is True, out
        assert tdmFusePaired(sol._state) is True
        assert decouplePGRBlocks(sol._state)[1:] == (1, 2)


# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Adding a grouping row is three edits, in any order, and none of them silent.

Every test here fails against the code before `B_MX` was wired, and each one
names the failure mode it closes. The reason they exist as a file of their own
is that the defect class they guard is not "the table is wrong" -- the table was
right every time -- but "production never read the table". A test that asserts
`TDM_GROUPS["B_MX"]` exists passes just as well when nothing can select it.

The measured baseline these are written against: at commit a7328e108d, adding
`3` to `ValidParameters` and `3: "B_MX"` to `TDM_FUSE_GROUPING` produced a
kernel named `_TDMF3` that was byte-identical to the `f0` kernel, with no
error, no warning and no rejection -- and the two runs had the same digest, so
the mapping entry changed nothing at all.
"""


def ks(fuse=3, **over):
    """A solution the resolver accepts as TDMFuse=3 / B_MX on four waves."""
    state = dict(TDMFuse=fuse, TDMInst=3, NumWaves=4, TDMSplit=0,
                 UseSubtileImpl=False, enableTDMA=True, enableTDMB=True,
                 ProblemType={"MXBlockA": 32, "MXBlockB": 32})
    state.update(over)
    return state


@pytest.mark.parametrize("fuse", [-3, -1, 4, 5, 6, 7, 8])
def test_an_unmapped_integer_raises_instead_of_resolving(fuse):
    """Fails against the unfixed resolver, which returned the default row.

    `tdmGrouping` was `return TDM_GROUPS[TDM_FUSE_GROUPING[0]]`, so every
    integer outside {0,1,2} resolved to `MX_AB` while `Naming.py` still put the
    integer in the kernel name. Measured over -3..8: eleven of fourteen values
    silently produced an `f0` kernel under another name.
    """
    with pytest.raises(ValueError, match="names no grouping"):
        TF.tdmGrouping(ks(fuse=fuse))


def test_a_row_without_an_acceptance_entry_raises(monkeypatch):
    """A mapping entry alone must not make a row selectable.

    This is the fail-open half, and it is the one that made the wiring order
    matter: once `TDM_FUSE_GROUPING` named a row the resolver stopped raising,
    and if nothing had written the row's preconditions there was nothing left
    to enforce them. Raising here is what makes the three edits order-free.
    """
    monkeypatch.setitem(TF.TDM_FUSE_GROUPING, 6, "AB")
    with pytest.raises(ValueError, match="no _GROUPING_ACCEPTED entry"):
        TF.tdmGrouping(ks(fuse=6))


def test_every_mapped_row_has_an_acceptance_entry():
    """The invariant the test above protects, asserted over the shipped table."""
    for fuse, name in TF.TDM_FUSE_GROUPING.items():
        assert name in TF.TDM_GROUPS, (fuse, name)
        assert name in TF._GROUPING_ACCEPTED, (fuse, name)


def test_valid_parameters_and_the_mapping_name_the_same_integers():
    """Fails against the unfixed pair, where 3 was mappable but not valid.

    Either direction of drift is a defect with a different shape: listed but
    unmapped built a kernel named for a grouping it did not have, mapped but
    unlisted is merely unreachable. Pinning them together closes both.
    """
    assert sorted(validParameters["TDMFuse"]) == sorted(TF.TDM_FUSE_GROUPING)


def test_unwired_rows_are_unreachable_not_merely_unswept():
    """`AB` and `None` have no integer and no acceptance entry."""
    unreachable = set(TF.TDM_GROUPS) - set(TF.TDM_FUSE_GROUPING.values())
    assert unreachable == {"AB", "None"}
    assert not unreachable & set(TF._GROUPING_ACCEPTED)


def test_three_resolves_to_b_mx():
    """The assertion that a table-content test cannot make.

    Fails against the unfixed resolver with `B_MX` present in `TDM_GROUPS` and
    `3` present in `TDM_FUSE_GROUPING`: it still answered `MX_AB`.
    """
    assert TF.tdmGrouping(ks()).name == "B_MX"
    assert TF.tdmGrouping(ks()) is not TF.TDM_GROUPS[TF.TDM_GROUPING_DEFAULT]


def test_b_mx_is_the_wave_mirror_of_a_mx():
    """Same partition as TDMFuse=2 with A and B exchanged.

    Checked as a mirror rather than against literals so that a change to the
    2/1/1 policy cannot leave the two rows disagreeing.
    """
    swap = {"A": "B", "B": "A", "MXSA": "MXSA", "MXSB": "MXSB"}
    for tc, mirrored in swap.items():
        assert TF.tdmWavePartition(ks(fuse=2), tc) \
               == TF.tdmWavePartition(ks(fuse=3), mirrored), tc


def test_b_mx_separates_a_from_b_and_earns_the_token_gate():
    """The property the decoupled PGR thick gate turns on."""
    assert TF.tdmSeparateABDescriptors(ks()) is True
    assert TF.tdmGroupingSeparatesAB(ks()) is True


def test_b_mx_inherits_the_nothing_to_cross_rejection():
    """One partitioned group, so crossing is refused with no new branch."""
    assert TC.tdmCrossRejectReason(ks(TDMCross=1)) is not None
    assert "nothing to cross" in TC.tdmCrossRejectReason(ks(TDMCross=1))


def test_b_mx_inherits_the_pap_rejection():
    """A scale seated on a data tensor's set cannot take PAP's handoff."""
    reason = TF.tdmPapRejectReason(ks())
    assert reason is not None
    assert "B_MX" in reason and "{B,MXSA,MXSB}" in reason


@pytest.mark.parametrize("numWaves", [1, 2, 3, 5, 8, 12, 16])
def test_b_mx_is_declined_away_from_four_waves(numWaves):
    """The guard the cross-repository claim on integer 3 depends on.

    The two trees agree on membership, member order and the unfused tensor, and
    their wave dispatch coincides ONLY at four waves: this side hands the
    `divmod` remainder to the leading members (3/3/2 at eight waves), the other
    side uses a fixed 2:1:1 (4/2/2). Pinning four waves makes the divergence
    unreachable, so it has to be asserted rather than assumed.

    Independently, 3/3/2 is not even emittable here -- the trailing two-wave
    share has no right-shift onto its components -- so this guard is also what
    keeps `tdmWaveComponents` from having to refuse.
    """
    assert TF.tdmGroupingAccepted(ks(NumWaves=numWaves)) is False
    assert TF.tdmGrouping(ks(NumWaves=numWaves)).name == TF.TDM_GROUPING_DEFAULT


def test_four_waves_is_accepted_and_shares_a_mx_entry():
    assert TF.tdmGroupingAccepted(ks()) is True
    assert TF._GROUPING_ACCEPTED["B_MX"] is TF._GROUPING_ACCEPTED["A_MX"]


@pytest.mark.parametrize("overrides", [
    {"TDMInst": 1}, {"TDMSplit": True}, {"UseSubtileImpl": True},
    {"ProblemType": {"MXBlockA": 32, "MXBlockB": 0}},
    {"ProblemType": {"MXBlockA": 0, "MXBlockB": 32}},
])
def test_b_mx_inherits_every_a_mx_precondition(overrides):
    """Sharing the entry is the point: these are not written twice."""
    assert TF.tdmGroupingAccepted(ks(**overrides)) is False
    assert TF.tdmGroupingAccepted(ks(fuse=2, **overrides)) is False


@pytest.mark.parametrize("name, expected", [
    ("MX_AB", {"A": "A", "B": "A", "MXSA": "MXSA", "MXSB": "MXSA"}),
    ("paired", {"A": "A", "MXSA": "A", "B": "B", "MXSB": "B"}),
    ("A_MX", {"A": "A", "MXSA": "A", "MXSB": "A", "B": "B"}),
    ("B_MX", {"B": "B", "MXSA": "B", "MXSB": "B", "A": "A"}),
    ("AB", {"A": "A", "B": "A", "MXSA": "MXSA", "MXSB": "MXSB"}),
    ("None", {"A": "A", "B": "B", "MXSA": "MXSA", "MXSB": "MXSB"}),
])
def test_set_owner_reproduces_every_rows_allocation(monkeypatch, name, expected):
    """`defineTdmSgprs` reads this instead of carrying an arm per TDMFuse value.

    The three wired rows' expectations here are the allocations the writer
    already emitted, which is why moving `defineTdmSgprs` onto the table left
    the corpus byte-identical. The unwired rows are included so the derivation
    is pinned before somebody points an integer at one.
    """
    monkeypatch.setattr(TF, "tdmGrouping", lambda _ks: TF.TDM_GROUPS[name])
    for tc, owner in expected.items():
        assert TF.tdmSetOwner(ks(fuse=0), tc) == owner, tc


def test_the_shared_scale_set_is_recognised_by_structure():
    """A_MX and B_MX only; the paired row splits the scales between two sets."""
    assert TF.tdmSharedScaleSetOwner(ks(fuse=2)) == "A"
    assert TF.tdmSharedScaleSetOwner(ks(fuse=3)) == "B"
    assert TF.tdmSharedScaleSetOwner(ks(fuse=1)) is None
    assert TF.tdmSharedScaleSetOwner(ks(fuse=0)) is None


def test_shared_set_order_exchanges_the_pair_for_the_mirror():
    """What the writer asks instead of assuming the owner is A."""
    assert TF.tdmSharedSetOrder(ks(fuse=2), "A", "B") == ("A", "B")
    assert TF.tdmSharedSetOrder(ks(fuse=3), "A", "B") == ("B", "A")


def test_shared_scale_set_needs_wave_separation():
    """The precondition the table does not carry, kept out of the row."""
    assert TF.tdmSharedScaleSetActive(ks()) is True
    assert TF.tdmSharedScaleSetActive(ks(enableTDMA=False)) is False
    assert TF.tdmSharedScaleSetActive(ks(NumWaves=1)) is False


# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Wave rules and descriptor-set predicates have one owner, and the writer uses it.

Every test here drives a real `KernelWriterAssembly` method body, bound to a
stub instance, rather than comparing the partition model against itself. A model
test cannot see these defects: the model was already right, and the writer held
a second copy of each answer.  Where a row that no TDMFuse integer selects is
needed, `tdmGrouping` is patched to return it -- that makes the unwired row
reachable, it does not stand in for the writer.
"""


def _ksOwner(**over):
    """A solution the writer accepts as TDMFuse=2 / A_MX on four waves."""
    state = dict(TDMFuse=2, TDMInst=3, NumWaves=4, TDMSplit=0, UseSubtileImpl=False,
                 enableTDMA=True, enableTDMB=True,
                 ProblemType={"MXBlockA": 32, "MXBlockB": 32})
    state.update(over)
    return state


class Stub:
    """Carries real KernelWriterAssembly method bodies and nothing else."""

    def __init__(self, **binds):
        for name in ("isTdmWaveSeparated", "tdmFuseAMx", "tdmFusePaired",
                     "tdmSeparateABDescriptors", "tdmSetupIncrementWaveSeparated"):
                fn = getattr(KernelWriterAssembly, name, None)
                if fn is not None:
                    setattr(self, name, fn.__get__(self, Stub))
        for k, v in binds.items():
            setattr(self, k, v)


@pytest.mark.parametrize("name", sorted(TF.TDM_GROUPS))
def test_writer_predicate_matches_the_owner_on_every_row(monkeypatch, name):
    """The writer's answer must be the table's answer for every row, wired or not.

    Fails against the unfixed writer on `B_MX` and `None`: spelled
    `tdmFuseAMx or tdmFusePaired` it names two TDMFuse values, so it answers
    False for any row those values do not select -- including the two rows that
    do separate A from B. `B_MX` is the row being wired next.
    """
    row = TF.TDM_GROUPS[name]
    monkeypatch.setattr(TF, "tdmGrouping", lambda _ks: row)
    # TDMFuse=0 so neither writer predicate fires off the integer; the row is
    # supplied by the table, which is where the writer should be reading it.
    state = _ksOwner(TDMFuse=0)
    assert Stub().tdmSeparateABDescriptors(state) is TF.tdmSeparateABDescriptors(state)


def test_owner_separates_ab_for_b_mx(monkeypatch):
    """B_MX separates A from B, so both owner and writer must say so."""
    monkeypatch.setattr(TF, "tdmGrouping", lambda _ks: TF.TDM_GROUPS["B_MX"])
    state = _ksOwner(TDMFuse=0)
    assert TF.tdmSeparateABDescriptors(state) is True
    assert Stub().tdmSeparateABDescriptors(state) is True


def test_wave_separation_still_gates_the_writer():
    """Delegating must not drop the wave-separation precondition.

    TDMInst names both tensors, so the grouping resolves to A_MX, but the TDM is
    not moving A: the writer programs the default shared set and the thick gate
    must not hand out disjoint tokens for it.
    """
    assert TF.tdmSeparateABDescriptors(_ksOwner()) is True
    assert TF.tdmSeparateABDescriptors(_ksOwner(enableTDMA=False)) is False
    assert Stub().tdmSeparateABDescriptors(_ksOwner(enableTDMA=False)) is False
    assert DP.decoupledThickGateRelaxation(_ksOwner(enableTDMA=False)) is None


def _incrementText(state):
    w = Stub(tdmFuseAMx=lambda _k: True, tdmFusePaired=lambda _k: False)
    mod = w.tdmSetupIncrementWaveSeparated(
        state, {"tensorChar": "A"}, {"tensorChar": "B"})
    return str(mod)


def test_shipped_increment_selects_waves_2_and_3():
    """The shipped 2/1/1 layout: MXSA on wave 2, MXSB on wave 3, A on waves 0-1."""
    text = _incrementText(_ksOwner())
    assert "wave 2 carries MXSA" in text
    assert "wave 3 carries MXSB" in text
    assert "waves 0-1 carry A" in text


def test_shipped_increment_literals_match_the_partition():
    """Pin the increment's literal wave rule to the table's answer.

    This site is deliberately NOT routed through the partition. A_MX has a single
    partitioned group and crossing reverses groups after the first, so crossing is
    already refused for the row and no reachable arrangement makes the literals
    wrong -- a structural finding, not a defect. Routing it would also make the
    dispatch read solution keys that stub-driven writer tests do not supply.

    So the duplication is made explicit and checked. If a row ever moves MXSA or
    MXSB, tdmSoleWave's answer changes, this test fails, and it names the literal
    that has to follow. That is the guard the duplication was missing, not a
    behaviour change: the emitted assembly is unchanged.
    """
    state = _ksOwner()
    assert TF.tdmSoleWave(state, "MXSA") == 2
    assert TF.tdmSoleWave(state, "MXSB") == 3
    assert TF.tdmWaveRangeText(state, "A") == "waves 0-1"
    text = _incrementText(state)
    assert "wave %d carries MXSA" % TF.tdmSoleWave(state, "MXSA") in text, text
    assert "wave %d carries MXSB" % TF.tdmSoleWave(state, "MXSB") in text, text
    assert "%s carry A" % TF.tdmWaveRangeText(state, "A") in text, text


def test_sole_wave_refuses_a_scale_spread_over_two_waves(monkeypatch):
    """One compare cannot select two waves, so the table's answer must refuse."""
    monkeypatch.setattr(TF, "tdmWavePartition", lambda _k, _tc: (2, (2, 3)))
    with pytest.raises(TF.TdmArrangementNotEmittable, match="one wave per member"):
        TF.tdmSoleWave(_ksOwner(), "MXSA")


def test_wave_components_spells_the_three_shapes(monkeypatch):
    shapes = {(1, (2,)): None, (2, (0, 1)): 0, (2, (0, 2)): 1, (2, (1, 3)): 1}
    for part, shift in shapes.items():
        monkeypatch.setattr(TF, "tdmWavePartition", lambda _k, _tc, p=part: p)
        assert TF.tdmWaveComponents(_ksOwner(), "A") == (part[0], shift)


def test_wave_components_does_not_refuse_zero_components(monkeypatch):
    """A member on no wave at all keeps its shipped answer, and is not refused.

    At NumWaves=1 a multi-member group reports zero components for its later
    members: `_waveShares` computes `numWaves // numMembers`, so the member rides
    no wave and nothing reads its component id. Refusing here breaks every
    single-wave TDM kernel -- it is the boundary the refusal above must not cross,
    and getting it wrong failed nine codegen tests while leaving the feature
    corpus byte-identical, because that corpus has no single-wave TDM solution.
    """
    monkeypatch.setattr(TF, "tdmWavePartition", lambda _k, _tc: (0, (0,)))
    assert TF.tdmWaveComponents(_ksOwner(NumWaves=1), "A") == (0, 1)


def test_wave_components_refuses_a_trailing_two_wave_share(monkeypatch):
    """Waves (2,3) have no right-shift onto components 0..1.

    Fails against the unfixed function, which falls through to a shift of one and
    maps both waves onto component 1 -- two waves filling one LDS block, which
    the assembly gives no sign of. This is the shape the "remainder to the
    trailing members" reading of the comments would have produced.
    """
    monkeypatch.setattr(TF, "tdmWavePartition", lambda _k, _tc: (2, (2, 3)))
    with pytest.raises(TF.TdmArrangementNotEmittable, match=r"rides waves \(2, 3\)"):
        TF.tdmWaveComponents(_ksOwner(), "A")


@pytest.mark.parametrize("name", sorted(TF.TDM_GROUPS))
def test_a_set_that_b_owns_is_issued_by_every_wave(monkeypatch, name):
    """Whenever B owns a descriptor set, every wave issues that set.

    This invariant is what retired `_emitTdmDealiasedIssue`, a parity-gated
    fill carrying a literal even=A / odd=B wave rule. A set is filled once per
    wave and the per-wave descriptor programming picks the member, so a group
    is either one member -- issued by every wave -- or several, each wave
    carrying exactly one. Either way no row leaves a wave with nothing to
    issue, so there was never a row for the parity gate to fire on: it was
    unreachable, not merely unused.

    Asserted over every row including the unwired ones, because the row such a
    gate would fire on first is the next one somebody adds. `B_MX` is the
    concrete case -- it puts A in a one-member group, so the literal expected A
    on the even waves while A is on all four.

    This holds on the table before the change as well as after, and that is the
    point rather than a weakness: the invariant was always true, so the parity
    gate was always unreachable. What the change did was stop maintaining a
    wave rule that no row could ever select. Written with nothing but the
    published partition so it cannot pass merely because a new helper exists.
    """
    row = TF.TDM_GROUPS[name]
    monkeypatch.setattr(TF, "tdmGrouping", lambda _ks: row)
    state = _ksOwner(TDMFuse=0)
    if not TF.tdmSeparateABDescriptors(state):
        pytest.skip("%s seats A and B on one set, so B issues nothing" % name)
    numWaves = state["NumWaves"]
    assignment = TF.tdmWaveAssignment(state)
    for tc in ("A", "B"):
        group = next(g for g in row.groups if tc in g)
        issuing = sorted(w for w, members in assignment.items()
                         if any(m in group for m in members))
        assert issuing == list(range(numWaves)), (
            "%s: every wave must carry some member of %s's set %s, got %s"
            % (name, tc, group, issuing))


# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Wave-separated TDM parity must follow the tensor, not the argument position.

The prologue programs the descriptor sets once, always as (A, B), so the A side
lands on the even waves. The tail loop then hands its pair to the same helpers in
*issue* order, and DirectToVgpr / DirectToLds / SwapGlobalReadOrder=1 reverse that
order (KernelWriter.isSwapGlobalReadOrderForDtvOrDtl). A positional reading of the
pair therefore rebuilds B's descriptor on the even waves and A's on the odd ones,
so after a PAP handoff every wave addresses the tensor it does not read. The
kernel still assembles and every barrier is still in place, which is why only an
invariant on the helpers catches it.

Both helpers are pure functions of the pair and TDMFuse, so they run unbound
against a stub -- no toolchain, no rocisa kernel state. Reversed arrival is not
hypothetical: at (PGRA, PGRB) = (1, 2) `_dcpThickThinIssueOrder()` returns
(B, A) so that the thick tensor issues first, and the StreamK PAP handoff passes
that pair straight into the tail reset. Calling the helpers directly is simply
the cheapest way to cover every pair.
"""


class _Writer:
    """Only the TDMFuse predicates are reached, and those read the kernel."""

    def __init__(self, kernel):
        self._kernel = kernel

    def tdmFusePaired(self, kernel):
        return KernelWriterAssembly.tdmFusePaired(self, kernel)

    def tdmFuseAMx(self, kernel):
        return KernelWriterAssembly.tdmFuseAMx(self, kernel)

    def isTdmWaveSeparated(self, kernel):
        return KernelWriterAssembly.isTdmWaveSeparated(self, kernel)

    def _tdmPairedParityOrder(self, kernel, tPA, tPB):
        return KernelWriterAssembly._tdmPairedParityOrder(self, kernel, tPA, tPB)


def _kernel(tdmFuse=0, numWaves=4):
    return {
        "TDMFuse": tdmFuse,
        "NumWaves": numWaves,
        "enableTDMA": True,
        "enableTDMB": True,
        "TDMInst": 0x03,  # TDM moves both A and B
        "TDMSplit": False,
        "UseSubtileImpl": False,
        "ProblemType": {"MXBlockA": 32, "MXBlockB": 32},
    }


def _tp(tc):
    return {"tensorChar": tc}


def _parityOrder(kernel, tP1, tP2):
    even, odd = KernelWriterAssembly._tdmPairedParityOrder(_Writer(kernel), kernel, tP1, tP2)
    return even["tensorChar"], odd["tensorChar"]


def _secondIsOdd(kernel, tP1, tP2):
    return KernelWriterAssembly._tdmSecondMemberIsOdd(_Writer(kernel), kernel, tP1, tP2)


@pytest.mark.parametrize("tdmFuse", [0, 1, 2])
@pytest.mark.parametrize("pair", [("A", "B"), ("MXSA", "MXSB")])
def test_parity_order_is_independent_of_argument_order(tdmFuse, pair):
    kernel = _kernel(tdmFuse)
    first, second = _tp(pair[0]), _tp(pair[1])
    assert _parityOrder(kernel, first, second) == _parityOrder(kernel, second, first)


@pytest.mark.parametrize("pair", [("A", "B"), ("MXSA", "MXSB")])
def test_coupled_pair_puts_the_a_side_on_the_even_waves(pair):
    # What the prologue's (A, B) call programs, and therefore what every later
    # call on the same pair has to agree with.
    kernel = _kernel(tdmFuse=0)
    assert _parityOrder(kernel, _tp(pair[0]), _tp(pair[1])) == (pair[0], pair[1])


def test_tdmfuse_paired_crosses_the_scale_pair():
    # TDMFuse=1: the scale call programs the set B rides, so MXSB is its even
    # member -- in either argument order.
    kernel = _kernel(tdmFuse=1)
    assert _parityOrder(kernel, _tp("MXSA"), _tp("MXSB")) == ("MXSB", "MXSA")
    assert _parityOrder(kernel, _tp("MXSB"), _tp("MXSA")) == ("MXSB", "MXSA")
    # The A/B call keeps the pair's own order.
    assert _parityOrder(kernel, _tp("A"), _tp("B")) == ("A", "B")


@pytest.mark.parametrize("tdmFuse", [0, 1, 2])
@pytest.mark.parametrize("pair", [("A", "B"), ("MXSA", "MXSB")])
def test_second_member_is_odd_tracks_the_argument_it_is_asked_about(tdmFuse, pair):
    # This one answers a question *about* the second argument, so unlike the
    # parity order it must flip when the pair is reversed. That is what lets the
    # tail-loop reset emit its two blocks in issue order and still branch on the
    # right parity.
    kernel = _kernel(tdmFuse)
    first, second = _tp(pair[0]), _tp(pair[1])
    assert _secondIsOdd(kernel, first, second) is not _secondIsOdd(kernel, second, first)


@pytest.mark.parametrize("tdmFuse", [0, 1, 2])
@pytest.mark.parametrize("pair", [("A", "B"), ("MXSA", "MXSB")])
def test_the_two_helpers_agree_on_which_member_is_odd(tdmFuse, pair):
    kernel = _kernel(tdmFuse)
    for tP1, tP2 in ((_tp(pair[0]), _tp(pair[1])), (_tp(pair[1]), _tp(pair[0]))):
        _even, odd = _parityOrder(kernel, tP1, tP2)
        assert _secondIsOdd(kernel, tP1, tP2) == (odd == tP2["tensorChar"])
