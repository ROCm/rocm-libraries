################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for Tensile.Components.TDMFuse."""
import copy
import types

import pytest

from Tensile.Common.GlobalParameters import defaultSolution
from Tensile.Common.ValidParameters import validParameters
from Tensile.Components import DecouplePGR as DP
from Tensile.Components import TDMFuse as TF
from Tensile.Components.DecouplePGR import decouplePGRBlocks
from Tensile.Components.TDMFuse import (TDM_FUSE_GROUPING, tdmBothTensors,
                                        tdmFusePaired, tdmGrouping,
                                        tdmWaveComponents, tdmWavePartition)
from Tensile.KernelWriterAssembly import KernelWriterAssembly

pytestmark = pytest.mark.unit

# TDM descriptor-group tensors; sparse metadata rides tdmMetadataGroup0 (unnamed).
TDM_TENSORS = ("A", "MXSA", "MXSB", "B")
_PRISTINE_DEFAULT_SOLUTION = copy.deepcopy(dict(defaultSolution))
_NO_MX_ON_B = {"MacDataTypeB": "F8", "DataTypeMXSB": "E8", "MXBlockB": 0}
_ONE_WAVE_MI = [16, 16, 128, 1, 1, 2, 16, 1, 1]
_ONE_WAVE_WG = [32, 1, 1]


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
        carried = [tc for tc in TDM_TENSORS if wave in tdmWavePartition(paired, tc)[1]]
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


@pytest.mark.parametrize("fuse", sorted(TDM_FUSE_GROUPING))
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


@pytest.mark.parametrize("fuse", sorted(TDM_FUSE_GROUPING))
def test_solution_accepts_stagger(_gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, StaggerU=32)
    assert sol.get("Valid") is True, out
    assert "requires StaggerU=0" not in out


@pytest.mark.parametrize(
    "fuse, overrides, clause",
    [
        (2, {"PrefetchGlobalReadA": 1, "PrefetchGlobalReadB": 2},
         "MXSB rides A's descriptor set but follows B's LDS block count"),
        # The mirror has to name the other scale: at 3 the set sits on B, so
        # MXSB follows its own owner and MXSA is the member carrying the second
        # cadence.
        (3, {"PrefetchGlobalReadA": 1, "PrefetchGlobalReadB": 2},
         "MXSA rides B's descriptor set but follows A's LDS block count"),
        (3, {"PrefetchGlobalReadA": 2, "PrefetchGlobalReadB": 1},
         "MXSA rides B's descriptor set but follows A's LDS block count"),
        (1, {"MatrixInstruction": _ONE_WAVE_MI, "WorkGroup": _ONE_WAVE_WG},
         "TDMFuse=1 requires NumWaves > 1 for parity grouping"),
        (2, {"MatrixInstruction": _ONE_WAVE_MI, "WorkGroup": _ONE_WAVE_WG},
         "TDMFuse=2 requires NumWaves=4 for its 2/1/1 split"),
        (2, {"MatrixInstruction": [16, 16, 128, 1, 1, 2, 16, 2, 1], "WorkGroup": [32, 2, 1]},
         "got 2"),
        (1, {"ProblemType": _NO_MX_ON_B},
         "TDMFuse=1 requires MX scales on both tensors"),
        (2, {"ProblemType": _NO_MX_ON_B},
         "TDMFuse=2 requires MX scales on both tensors"),
        (1, {"ProblemType": {"Sparse": 1}},
         "TDMFuse=1 does not support sparse metadata"),
        (2, {"ProblemType": {"Sparse": 1}},
         "TDMFuse=2 does not support sparse metadata"),
        (1, {"TDMInst": 1}, "TDMA and TDMB must be enabled simultaneously"),
        # A tile whose MX scales fit one load per wave, so the subtile
        # scale-load budget does not answer first.
        (1, {"UseSubtileImpl": True,
             "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2]},
         "TDMFuse=1 requires UseSubtileImpl=0"),
    ],
)
def test_solution_rejects_outside_the_grouping(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, overrides, clause):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, **overrides)
    assert sol.get("Valid") is False
    assert clause in out


@pytest.mark.parametrize("fuse, predicate", [
    (1, tdmFusePaired),
    (2, lambda ks: tdmGrouping(ks).name == "A_MX"),
    (3, lambda ks: tdmGrouping(ks).name == "B_MX")])
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


def ks(fuse=3, **over):
    """A solution the resolver accepts as TDMFuse=3 / B_MX on four waves."""
    state = dict(TDMFuse=fuse, TDMInst=3, NumWaves=4, TDMSplit=0,
                 UseSubtileImpl=False, enableTDMA=True, enableTDMB=True,
                 ProblemType={"MXBlockA": 32, "MXBlockB": 32})
    state.update(over)
    return state


@pytest.mark.parametrize("fuse", [-3, -1, 4, 5, 6, 7, 8])
def test_an_unmapped_integer_raises_instead_of_resolving(fuse):
    """An integer no row claims must raise, not fall back."""
    with pytest.raises(ValueError, match="names no grouping"):
        TF.tdmGrouping(ks(fuse=fuse))


def test_a_row_without_an_acceptance_entry_raises(monkeypatch):
    """A mapping entry alone must not make a row selectable."""
    monkeypatch.setitem(TF.TDM_GROUPS, "probe",
                        TF.TdmGrouping("probe", (("A", "B"), ("MXSA", "MXSB"))))
    monkeypatch.setitem(TF.TDM_FUSE_GROUPING, 6, "probe")
    with pytest.raises(ValueError, match="no _GROUPING_ACCEPTED entry"):
        TF.tdmGrouping(ks(fuse=6))


def test_valid_parameters_and_the_mapping_name_the_same_integers():
    assert sorted(validParameters["TDMFuse"]) == sorted(TF.TDM_FUSE_GROUPING)


def test_every_shipped_row_is_reachable():
    assert set(TF.TDM_GROUPS) == set(TF.TDM_FUSE_GROUPING.values())
    assert set(TF.TDM_GROUPS) == set(TF._GROUPING_ACCEPTED)


def test_three_resolves_to_b_mx():
    assert TF.tdmGrouping(ks()).name == "B_MX"
    assert TF.tdmGrouping(ks()) is not TF.TDM_GROUPS[TF.TDM_GROUPING_DEFAULT]


def test_b_mx_is_the_wave_mirror_of_a_mx():
    swap = {"A": "B", "B": "A", "MXSA": "MXSA", "MXSB": "MXSB"}
    for tc, mirrored in swap.items():
        assert TF.tdmWavePartition(ks(fuse=2), tc) \
               == TF.tdmWavePartition(ks(fuse=3), mirrored), tc


def test_b_mx_separates_a_from_b():
    assert TF.tdmSeparateABDescriptors(ks()) is True
    assert TF.tdmGroupingSeparatesAB(ks()) is True


def test_b_mx_inherits_the_pap_rejection():
    """A scale seated on a data tensor's set cannot take PAP's handoff."""
    reason = TF.tdmPapRejectReason(ks())
    assert reason is not None
    assert "B_MX" in reason and "{B,MXSA,MXSB}" in reason


@pytest.mark.parametrize("numWaves", [1, 2, 3, 5, 8, 12, 16])
def test_b_mx_is_declined_away_from_four_waves(numWaves):
    """Four waves only: remainder-to-leading vs fixed 2/1/1; trailing two-wave share has no right-shift."""
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
    assert TF.tdmGroupingAccepted(ks(**overrides)) is False
    assert TF.tdmGroupingAccepted(ks(fuse=2, **overrides)) is False


@pytest.mark.parametrize("name, expected", [
    ("MX_AB", {"A": "A", "B": "A", "MXSA": "MXSA", "MXSB": "MXSA"}),
    ("paired", {"A": "A", "MXSA": "A", "B": "B", "MXSB": "B"}),
    ("A_MX", {"A": "A", "MXSA": "A", "MXSB": "A", "B": "B"}),
    ("B_MX", {"B": "B", "MXSA": "B", "MXSB": "B", "A": "A"}),
])
def test_set_owner_reproduces_every_rows_allocation(monkeypatch, name, expected):
    monkeypatch.setattr(TF, "tdmGrouping", lambda _ks: TF.TDM_GROUPS[name])
    for tc, owner in expected.items():
        assert TF.tdmSetOwner(ks(fuse=0), tc) == owner, tc


def test_the_shared_scale_set_is_recognised_by_structure():
    assert TF.tdmSharedScaleSetOwner(ks(fuse=2)) == "A"
    assert TF.tdmSharedScaleSetOwner(ks(fuse=3)) == "B"
    assert TF.tdmSharedScaleSetOwner(ks(fuse=1)) is None
    assert TF.tdmSharedScaleSetOwner(ks(fuse=0)) is None


def test_shared_set_order_exchanges_the_pair_for_the_mirror():
    assert TF.tdmSharedSetOrder(ks(fuse=2), "A", "B") == ("A", "B")
    assert TF.tdmSharedSetOrder(ks(fuse=3), "A", "B") == ("B", "A")


def test_shared_scale_set_needs_wave_separation():
    assert TF.tdmSharedScaleSetActive(ks()) is True
    assert TF.tdmSharedScaleSetActive(ks(enableTDMA=False)) is False
    assert TF.tdmSharedScaleSetActive(ks(NumWaves=1)) is False


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
        for name in ("isTdmWaveSeparated", "tdmFusePaired",
                     "tdmSeparateABDescriptors", "tdmSetupIncrementWaveSeparated",
                     "_dcpTokenSide", "_dcpCurrentToken", "_dcpTdmIssueTokens"):
            fn = getattr(KernelWriterAssembly, name, None)
            if fn is not None:
                setattr(self, name, fn.__get__(self, Stub))
        for k, v in binds.items():
            setattr(self, k, v)


def test_owner_separates_ab_for_b_mx(monkeypatch):
    monkeypatch.setattr(TF, "tdmGrouping", lambda _ks: TF.TDM_GROUPS["B_MX"])
    state = _ksOwner(TDMFuse=0)
    assert TF.tdmSeparateABDescriptors(state) is True
    assert Stub().tdmSeparateABDescriptors(state) is True


def test_wave_separation_still_gates_the_writer():
    """Delegating must not drop the wave-separation precondition."""
    assert TF.tdmSeparateABDescriptors(_ksOwner()) is True
    assert TF.tdmSeparateABDescriptors(_ksOwner(enableTDMA=False)) is False
    assert Stub().tdmSeparateABDescriptors(_ksOwner(enableTDMA=False)) is False
    divergent = _ksOwner(enableTDMA=False, PrefetchGlobalReadA=1,
                         PrefetchGlobalReadB=2)
    assert DP.decoupledThickGateRelaxation(divergent) is None


def _tokenWriter():
    """A writer carrying the per-side tensor tokens decoupled PGR installs."""
    return Stub(states=types.SimpleNamespace(ldsTensorTokenIdxA=7,
                                             ldsTensorTokenIdxB=9))


def test_an_issue_declares_every_lds_side_its_set_fills():
    """A_MX seats MXSB on A's set; MXSB's LDS is on B's side."""
    writer, state = _tokenWriter(), _ksOwner()
    assert writer._dcpTdmIssueTokens(state, "A") == [7, 9]
    assert writer._dcpTdmIssueTokens(state, "B") == [9]


def test_a_dead_set_member_contributes_no_token(monkeypatch):
    """A scale-less problem fills the set's live members, not the row's full membership."""
    monkeypatch.setattr(TF, "tdmGrouping", lambda _ks: TF.TDM_GROUPS["A_MX"])
    state = _ksOwner(ProblemType={"MXBlockA": 32, "MXBlockB": 0})
    assert _tokenWriter()._dcpTdmIssueTokens(state, "A") == [7]


def test_shipped_increment_literals_match_the_partition():
    """Increment literals are not routed through the partition; pin them to tdmSoleWave."""
    state = _ksOwner()
    assert TF.tdmSoleWave(state, "MXSA") == 2
    assert TF.tdmSoleWave(state, "MXSB") == 3
    assert TF.tdmWaveRangeText(state, "A") == "waves 0-1"
    text = _incrementText(state)
    assert "wave %d carries MXSA" % TF.tdmSoleWave(state, "MXSA") in text, text
    assert "wave %d carries MXSB" % TF.tdmSoleWave(state, "MXSB") in text, text
    assert "%s carry A" % TF.tdmWaveRangeText(state, "A") in text, text


def test_sole_wave_refuses_a_scale_spread_over_two_waves(monkeypatch):
    """One compare cannot select two waves."""
    monkeypatch.setattr(TF, "tdmWavePartition", lambda _k, _tc: (2, (2, 3)))
    with pytest.raises(TF.TdmArrangementNotEmittable, match="one wave per member"):
        TF.tdmSoleWave(_ksOwner(), "MXSA")


def test_wave_components_spells_the_three_shapes(monkeypatch):
    shapes = {(1, (2,)): None, (2, (0, 1)): 0, (2, (0, 2)): 1, (2, (1, 3)): 1}
    for part, shift in shapes.items():
        monkeypatch.setattr(TF, "tdmWavePartition", lambda _k, _tc, p=part: p)
        assert TF.tdmWaveComponents(_ksOwner(), "A") == (part[0], shift)


def test_wave_components_does_not_refuse_zero_components(monkeypatch):
    """A member on no wave keeps its shipped answer."""
    monkeypatch.setattr(TF, "tdmWavePartition", lambda _k, _tc: (0, (0,)))
    assert TF.tdmWaveComponents(_ksOwner(NumWaves=1), "A") == (0, 1)


def test_wave_components_refuses_a_trailing_two_wave_share(monkeypatch):
    """Waves (2, 3) have no right-shift onto components 0..1."""
    monkeypatch.setattr(TF, "tdmWavePartition", lambda _k, _tc: (2, (2, 3)))
    with pytest.raises(TF.TdmArrangementNotEmittable, match=r"rides waves \(2, 3\)"):
        TF.tdmWaveComponents(_ksOwner(), "A")


class _Writer:
    """Only the TDMFuse predicates are reached, and those read the kernel."""

    def __init__(self, kernel):
        self._kernel = kernel

    def tdmFusePaired(self, kernel):
        return KernelWriterAssembly.tdmFusePaired(self, kernel)

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


def _incrementText(state):
    w = Stub()
    mod = w.tdmSetupIncrementWaveSeparated(
        state, {"tensorChar": "A"}, {"tensorChar": "B"})
    return str(mod)
