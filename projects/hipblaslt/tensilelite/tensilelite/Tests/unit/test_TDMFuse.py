################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for Tensile.Components.TDMFuse."""
import copy
import itertools
import types

import pytest
from rocisa.code import RegSet

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
# MX scales on one tensor only, on F8 x F8.
_MXSA_ONLY = {"MacDataTypeB": "F8", "MXBlockB": 0}
_MXSB_ONLY = {"MacDataTypeB": "F8", "MXBlockA": 0}
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
    assert tdmWaveComponents(_ks(), tc) == (2, 1, 0)


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
    assert tdmWaveComponents(ks, tc) == (numComp, shift, 0)


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
    # One scale on TDMFuse=1 rides its data tensor's set, the shape 2 and 3 have.
    assert TF.tdmSharedScaleSetOwner(_maskState(1, "mx_a")) == "A"
    assert TF.tdmSharedScaleSetOwner(_maskState(1, "mx_b")) == "B"
    assert TF.tdmSharedScaleSetOwner(_maskState(0, "mx_a")) is None


def test_shared_set_order_exchanges_the_pair_for_the_mirror():
    assert TF.tdmSharedSetOrder(ks(fuse=2), "A", "B") == ("A", "B")
    assert TF.tdmSharedSetOrder(ks(fuse=3), "A", "B") == ("B", "A")
    assert TF.tdmSharedSetOrder(_maskState(1, "mx_b"), "A", "B") == ("B", "A")


def test_shared_scale_set_needs_wave_separation():
    assert TF.tdmSharedScaleSetActive(ks()) is True
    assert TF.tdmSharedScaleSetActive(ks(enableTDMA=False)) is False
    assert TF.tdmSharedScaleSetActive(ks(NumWaves=1)) is False
    assert TF.tdmSharedScaleSetActive(_maskState(1, "mx_a")) is True
    assert TF.tdmSharedScaleSetActive(_maskState(1, "mx_a", numWaves=1)) is False
    assert TF.tdmSharedScaleSetActive(ks(fuse=1)) is False


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
                     "tdmDescriptorSetOwner", "tdmLastIterDisableSgprs", "defineTdmSgprs",
                     "_dcpTokenSide", "_dcpCurrentToken", "_dcpTdmIssueTokens",
                     "tdmPairOnEveryWave", "removeStaggerCopies", "tdmHalfPlrReenableSgprs"):
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


def _disabledOwners(writer, state):
    names = writer.tdmLastIterDisableSgprs(state)
    return names, {writer.tdmDescriptorSetOwner(state, n[len("tdm"):-len("Group0+0")])
                   for n in names}


@pytest.mark.parametrize("name", sorted(TF.TDM_GROUPS))
@pytest.mark.parametrize("mxA, mxB", [(32, 32), (32, 0), (0, 32)])
def test_last_iter_disable_reaches_every_descriptor_set(monkeypatch, name, mxA, mxB):
    monkeypatch.setattr(TF, "tdmGrouping", lambda _ks: TF.TDM_GROUPS[name])
    writer = Stub()
    state = _ksOwner(ProblemType={"MXBlockA": mxA, "MXBlockB": mxB})
    live = [tc for tc in TDM_TENSORS if TF.tdmMemberIsLive(state, tc)]
    names, owners = _disabledOwners(writer, state)
    assert owners == {writer.tdmDescriptorSetOwner(state, tc) for tc in live}
    assert len(names) == len(set(names))


@pytest.mark.parametrize("name", sorted(TF.TDM_GROUPS))
@pytest.mark.parametrize("mxA, mxB", [(32, 32), (32, 0), (0, 32)])
def test_every_tdm_sgpr_alias_targets_a_defined_set(monkeypatch, name, mxA, mxB):
    monkeypatch.setattr(TF, "tdmGrouping", lambda _ks: TF.TDM_GROUPS[name])
    writer = Stub(states=types.SimpleNamespace(subtileIterateModeA=False,
                                               subtileIterateModeB=False, staggerUCode=True),
                  defineSgpr=lambda reg, num, align=1: RegSet("s", "sgpr" + reg, 0))
    state = _ksOwner(enableTDMMetadata=False,
                     ProblemType={"MXBlockA": mxA, "MXBlockB": mxB})
    defined = set()
    for item in writer.defineTdmSgprs(state).items():
        directive, dst, src = str(item).split()
        assert directive == ".set"
        assert src.isdigit() or src.split("+")[0] in defined, str(item)
        defined.add(dst.rstrip(","))


@pytest.mark.parametrize("name, expected", [
    ("MX_AB", ["tdmAGroup0+0", "tdmMXSAGroup0+0"]),
    ("B_MX", ["tdmAGroup0+0", "tdmMXSAGroup0+0"]),
    ("paired", ["tdmAGroup0+0", "tdmMXSAGroup0+0", "tdmBGroup0+0"]),
    ("A_MX", ["tdmAGroup0+0", "tdmMXSAGroup0+0", "tdmBGroup0+0"]),
])
def test_last_iter_disable_names_per_grouping(monkeypatch, name, expected):
    monkeypatch.setattr(TF, "tdmGrouping", lambda _ks: TF.TDM_GROUPS[name])
    assert Stub().tdmLastIterDisableSgprs(_ksOwner()) == expected


@pytest.mark.parametrize("over", [dict(enableTDMB=False), dict(UseSubtileImpl=True)])
def test_last_iter_disable_without_wave_separated_sets(monkeypatch, over):
    monkeypatch.setattr(TF, "tdmGrouping", lambda _ks: TF.TDM_GROUPS["paired"])
    assert Stub().tdmLastIterDisableSgprs(_ksOwner(**over)) == ["tdmAGroup0+0", "tdmMXSAGroup0+0"]


def _tokenWriter():
    """A writer carrying the per-side tensor tokens decoupled PGR installs."""
    return Stub(states=types.SimpleNamespace(ldsTensorTokenIdxA=7,
                                             ldsTensorTokenIdxB=9,
                                             tdmSplitTokenSides={}))


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


def test_wave_components_spells_each_shape(monkeypatch):
    """A wave's component is (WaveIdx - offset) >> shift; a None shift is one component."""
    shapes = {(1, (2,)): (None, 0), (2, (0, 1)): (0, 0), (2, (0, 2)): (1, 0),
              (2, (1, 3)): (1, 0), (2, (2, 3)): (0, 2), (4, (0, 1, 2, 3)): (0, 0)}
    for part, (shift, offset) in shapes.items():
        monkeypatch.setattr(TF, "tdmWavePartition", lambda _k, _tc, p=part: p)
        assert TF.tdmWaveComponents(_ksOwner(), "A") == (part[0], shift, offset)
        if shift is not None:
            assert [(w - offset) >> shift for w in part[1]] == list(range(part[0])), part


def test_wave_components_does_not_refuse_zero_components(monkeypatch):
    """A member on no wave keeps its shipped answer."""
    monkeypatch.setattr(TF, "tdmWavePartition", lambda _k, _tc: (0, (0,)))
    assert TF.tdmWaveComponents(_ksOwner(NumWaves=1), "A") == (0, 1, 0)


def test_wave_components_reach_a_trailing_two_wave_share_through_an_offset(monkeypatch):
    """Waves (2, 3) have no right shift onto components 0..1; subtracting 2 first does."""
    monkeypatch.setattr(TF, "tdmWavePartition", lambda _k, _tc: (2, (2, 3)))
    assert TF.tdmWaveComponents(_ksOwner(), "A") == (2, 0, 2)


def test_wave_components_refuse_a_share_no_offset_or_shift_reaches(monkeypatch):
    """Waves (0, 2, 3) are neither a run nor a right-shifted parity class."""
    monkeypatch.setattr(TF, "tdmWavePartition", lambda _k, _tc: (3, (0, 2, 3)))
    with pytest.raises(TF.TdmArrangementNotEmittable, match=r"rides waves \(0, 2, 3\)"):
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


_NO_MX = {"MacDataTypeB": "F8", "MXBlockA": 0, "MXBlockB": 0}
_TWO_WAVES = {"MatrixInstruction": [16, 16, 128, 1, 1, 2, 16, 2, 1], "WorkGroup": [32, 2, 1]}
_ONE_WAVE_64X64 = {"MatrixInstruction": [16, 16, 128, 1, 1, 4, 4, 1, 1], "WorkGroup": _ONE_WAVE_WG}
_WAVES = {1: _ONE_WAVE_64X64, 2: _TWO_WAVES, 4: {}}
_ONE_SCALE = {"mx_a": _MXSA_ONLY, "mx_b": _MXSB_ONLY}
_ONE_SIDED_MX = "MX scales on one tensor only"
_STREAMK = {"StreamK": 3, "StreamKForceDPOnly": 1}


@pytest.mark.parametrize("numWaves", [1, 2, 4])
@pytest.mark.parametrize("mx", sorted(_ONE_SCALE))
@pytest.mark.parametrize("fuse", sorted(TDM_FUSE_GROUPING))
def test_solution_on_one_mx_scale_for_every_grouping_and_wave_count(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, mx, numWaves):
    """One-sided MX derives on gfx1250 TDM kernels; TDMFuse 1-3 still need more than one wave."""
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse,
                       ProblemType=_ONE_SCALE[mx], **_WAVES[numWaves])
    if fuse != 0 and numWaves == 1:
        assert sol.get("Valid") is False
        assert "TDMFuse=%d" % fuse in out and "requires NumWaves > 1" in out, out
        return
    assert sol.get("Valid") is True, out
    assert _ONE_SIDED_MX not in out, out
    state = sol._state
    assert state["NumWaves"] == numWaves
    assert TF.tdmLiveScales(state) == (("MXSA",) if mx == "mx_a" else ("MXSB",))
    assert tdmGrouping(state).name == TDM_FUSE_GROUPING[fuse]
    assert TF.tdmWaveMasks(state) == _expectedMasks(fuse, mx, numWaves)


@pytest.mark.parametrize("overrides, found", [
    ({"UseSubtileImpl": True, "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2]},
     "UseSubtileImpl="),
    (dict(_STREAMK, PrefetchAcrossPersistent=1), "PrefetchAcrossPersistent=1"),
    (dict(_STREAMK, ReuseAcrossPersistent=1), "ReuseAcrossPersistent=1"),
    (_STREAMK, "a persistent execution policy"),
    ({"StreamK": 3, "StreamKForceDPOnly": 0}, "got a StreamK execution policy"),
    ({"PrefetchGlobalReadA": 1, "PrefetchGlobalReadB": 2}, "PGRA=1, PGRB=2"),
], ids=["subtile", "pap", "rap", "persistent_dp", "streamk", "divergent_pair"])
@pytest.mark.parametrize("mx", sorted(_ONE_SCALE))
@pytest.mark.parametrize("fuse", [0, 1])
def test_solution_rejects_one_mx_scale_outside_its_limits(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, mx, overrides, found):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse,
                       ProblemType=_ONE_SCALE[mx], **overrides)
    assert sol.get("Valid") is False
    assert _ONE_SIDED_MX in out, out
    assert found in out, out


@pytest.mark.parametrize("mx", sorted(_ONE_SCALE))
def test_solution_rejects_one_mx_scale_off_the_tdm_scale_load(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, mx):
    """gfx1250 swizzles MX scales in memory, which only the TDM scale load does, so the
    transport check can answer before the one-sided limit; either names the load."""
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=0,
                       ProblemType=_ONE_SCALE[mx], MXLoadInst="BufferLoad")
    assert sol.get("Valid") is False
    assert "MXLoadInst=BufferLoad" in out, out


@pytest.mark.parametrize("fuse", [1, 2, 3])
@pytest.mark.parametrize("waves", [_TWO_WAVES, {}], ids=["2w", "4w"])
def test_solution_accepts_rows_one_to_three_without_scales(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, waves):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, ProblemType=_NO_MX, **waves)
    assert sol.get("Valid") is True, out
    assert tdmGrouping(sol._state).name == TDM_FUSE_GROUPING[fuse]
    assert TF.tdmPairOnEveryWave(sol._state, "A", "B") is True
    assert tdmFusePaired(sol._state) is False


@pytest.mark.parametrize("overrides, clause", [
    ({"MatrixInstruction": _ONE_WAVE_MI, "WorkGroup": _ONE_WAVE_WG}, "requires NumWaves > 1"),
    ({"ProblemType": dict(_NO_MX, Sparse=1)}, "does not support sparse metadata"),
    ({"PrefetchGlobalReadA": 1, "PrefetchGlobalReadB": 2}, "requires an equal decoupled pair"),
    ({"UseSubtileImpl": True, "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2]},
     "requires UseSubtileImpl=0"),
])
@pytest.mark.parametrize("fuse", [1, 2, 3])
def test_solution_rejects_rows_one_to_three_without_scales_outside_their_limits(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, overrides, clause):
    overrides = dict(overrides)
    problemType = overrides.pop("ProblemType", _NO_MX)
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, ProblemType=problemType,
                       **overrides)
    assert sol.get("Valid") is False
    assert "TDMFuse=%d without MX scales %s" % (fuse, clause) in out, out


@pytest.mark.parametrize("fuse", sorted(TDM_FUSE_GROUPING))
def test_only_rows_without_scales_put_both_data_tensors_on_every_wave(fuse):
    assert TF.tdmPairOnEveryWave(ks(fuse=fuse), "A", "B") is False
    assert TF.tdmPairOnEveryWave(ks(fuse=fuse), "MXSA", "MXSB") is False
    unscaled = ks(fuse=fuse, ProblemType={"MXBlockA": 0, "MXBlockB": 0})
    assert TF.tdmPairOnEveryWave(unscaled, "A", "B") is (fuse != 0)


def test_a_single_member_set_alone_rides_every_wave():
    """2 and 3 each leave one data tensor alone, which is why the pair needs both."""
    assert TF.tdmRidesEveryWave(ks(fuse=2), "B") is True
    assert TF.tdmRidesEveryWave(ks(fuse=3), "A") is True
    unscaledOneWave = ks(fuse=1, NumWaves=1, ProblemType={"MXBlockA": 0, "MXBlockB": 0})
    assert TF.tdmRidesEveryWave(unscaledOneWave, "A") is False
    # One wave issues every live tensor from its own descriptor; nothing is wave-separated.
    assert TF.tdmWavePartition(unscaledOneWave, "A") == (1, (0,))
    assert TF.tdmWavePartition(unscaledOneWave, "B") == (1, (0,))


@pytest.mark.parametrize("fuse", [1, 2, 3])
def test_rows_without_scales_allocate_no_shared_increment(fuse):
    writer = Stub(states=types.SimpleNamespace(subtileIterateModeA=False,
                                               subtileIterateModeB=False, staggerUCode=True),
                  defineSgpr=lambda reg, num, align=1: RegSet("s", "sgpr" + reg, 0))
    state = _ksOwner(TDMFuse=fuse, enableTDMMetadata=False,
                     ProblemType={"MXBlockA": 0, "MXBlockB": 0})
    defined = {str(item).split()[1].rstrip(",")[len("sgpr"):]
               for item in writer.defineTdmSgprs(state).items()}
    assert {"tdmABIncs", "tdmMXSAMXSBIncs"} & defined == set()


# ---------------------------------------------------------------------------
# Wave masks: which waves issue each tensor, with and without MX scales.
# ---------------------------------------------------------------------------
_MX_STATES = {"mx": (32, 32), "no_mx": (0, 0), "mx_a": (32, 0), "mx_b": (0, 32)}


def _maskState(fuse, mx="mx", numWaves=4):
    mxA, mxB = _MX_STATES[mx]
    return ks(fuse=fuse, NumWaves=numWaves, ProblemType={"MXBlockA": mxA, "MXBlockB": mxB})


def _acceptedMaskStates(waveCounts=(1, 2, 4, 8, 16)):
    for fuse, numWaves, mx in itertools.product(sorted(TDM_FUSE_GROUPING), waveCounts,
                                                sorted(_MX_STATES)):
        state = _maskState(fuse, mx, numWaves)
        if fuse == 0 or TF.tdmGroupingAccepted(state):
            yield state


@pytest.mark.parametrize("fuse, mx, masks", [
    (0, "mx", {"A": 0b0101, "B": 0b1010, "MXSA": 0b0101, "MXSB": 0b1010}),
    (0, "no_mx", {"A": 0b0101, "B": 0b1010}),
    (1, "mx", {"A": 0b0101, "MXSA": 0b1010, "MXSB": 0b0101, "B": 0b1010}),
    (1, "no_mx", {"A": 0b1111, "B": 0b1111}),
    (2, "mx", {"A": 0b0011, "MXSA": 0b0100, "MXSB": 0b1000, "B": 0b1111}),
    (2, "no_mx", {"A": 0b1111, "B": 0b1111}),
    (3, "mx", {"B": 0b0011, "MXSA": 0b0100, "MXSB": 0b1000, "A": 0b1111}),
    (3, "no_mx", {"A": 0b1111, "B": 0b1111}),
])
def test_wave_masks_on_four_waves(fuse, mx, masks):
    assert TF.tdmWaveMasks(_maskState(fuse, mx)) == masks


# With one MX scale each set still splits its waves over its live members, so no wave
# of a set with a live member idles.
@pytest.mark.parametrize("fuse, mx, masks", [
    (0, "mx_a", {"A": 0b0101, "B": 0b1010, "MXSA": 0b1111}),
    (0, "mx_b", {"A": 0b0101, "B": 0b1010, "MXSB": 0b1111}),
    (1, "mx_a", {"A": 0b0101, "MXSA": 0b1010, "B": 0b1111}),
    (1, "mx_b", {"A": 0b1111, "MXSB": 0b0101, "B": 0b1010}),
    (2, "mx_a", {"A": 0b0011, "MXSA": 0b1100, "B": 0b1111}),
    (2, "mx_b", {"A": 0b0011, "MXSB": 0b1100, "B": 0b1111}),
    (3, "mx_a", {"B": 0b0011, "MXSA": 0b1100, "A": 0b1111}),
    (3, "mx_b", {"B": 0b0011, "MXSB": 0b1100, "A": 0b1111}),
])
def test_wave_masks_with_one_scale_on_four_waves(fuse, mx, masks):
    assert TF.tdmWaveMasks(_maskState(fuse, mx)) == masks


@pytest.mark.parametrize("numWaves", [2, 4, 8])
@pytest.mark.parametrize("fuse", [1, 2, 3])
def test_rows_one_to_three_reduce_to_a_and_b_on_every_wave_without_scales(fuse, numWaves):
    """Each data tensor is left alone in its set and takes every wave, the same in all three rows."""
    state, paired = _maskState(fuse, "no_mx", numWaves), _maskState(1, "no_mx", numWaves)
    every = (1 << numWaves) - 1
    assert TF.tdmGroupingAccepted(state) is True
    assert TF.tdmGrouping(state).name == TDM_FUSE_GROUPING[fuse]
    assert sorted(TF.liveGroups(state)) == [("A",), ("B",)]
    assert TF.tdmWaveMasks(state) == TF.tdmWaveMasks(paired) == {"A": every, "B": every}
    for tc in ("A", "B"):
        assert TF.tdmWavePartition(state, tc) == TF.tdmWavePartition(paired, tc) \
               == (numWaves, tuple(range(numWaves))), tc
        assert TF.tdmWaveComponents(state, tc) == (numWaves, 0, 0), tc
    for tc in TF.TDM_SCALE_TENSORS:
        assert TF.tdmWavePartition(state, tc) == (0, ()), tc
    # The writer picks the every-wave arms, not the paired, shared-scale or parity ones.
    assert TF.tdmPairOnEveryWave(state, "A", "B") is True
    assert TF.tdmSeparateABDescriptors(state) is True
    assert tdmFusePaired(state) is False
    assert TF.tdmSharedScaleSet(state) is None
    assert TF.tdmSharedScaleSetActive(state) is False
    assert TF.tdmScaleSharesDataSet(state) == ()
    assert TF.tdmPapRejectReason(state) is None
    assert TF.tdmGroupPartner(state, "A", "B") == "B"
    assert TF.tdmGroupPartner(state, "B", "A") == "A"


@pytest.mark.parametrize("over", [
    {"NumWaves": 1}, {"TDMSplit": True}, {"UseSubtileImpl": True}, {"TDMInst": 1},
])
@pytest.mark.parametrize("fuse", [1, 2, 3])
def test_rows_one_to_three_without_scales_decline_outside_their_limits(fuse, over):
    state = dict(_maskState(fuse, "no_mx"), **over)
    assert TF.tdmGroupingAccepted(state) is False
    assert TF.tdmGrouping(state).name == TF.TDM_GROUPING_DEFAULT


@pytest.mark.parametrize("numWaves", [2, 4, 8])
@pytest.mark.parametrize("mx", ["mx_a", "mx_b"])
@pytest.mark.parametrize("fuse", [1, 2, 3])
def test_a_lone_scale_keeps_rows_one_to_three_accepted(fuse, mx, numWaves):
    """The lone scale rides its data tensor's set; the other data tensor rides every wave of its own."""
    state = _maskState(fuse, mx, numWaves)
    assert TF.tdmGroupingAccepted(state) is True
    assert TF.tdmGrouping(state).name == TDM_FUSE_GROUPING[fuse]
    assert TF.tdmGroupingAccepted(dict(state, NumWaves=1)) is False
    # The paired arms program both scales; one scale takes the shared-scale arms.
    assert tdmFusePaired(state) is False
    owner = {1: "A" if mx == "mx_a" else "B", 2: "A", 3: "B"}[fuse]
    assert TF.tdmSharedScaleSetOwner(state) == owner
    assert TF.tdmSharedScaleSetActive(state) is True
    assert TF.tdmRidesEveryWave(state, "B" if owner == "A" else "A") is True


@pytest.mark.parametrize("mx, live, dead", [("mx_a", "MXSA", "MXSB"), ("mx_b", "MXSB", "MXSA")])
def test_a_scale_only_set_gives_its_lone_live_scale_every_wave(mx, live, dead):
    state = _maskState(0, mx)
    assert TF.tdmWaveMasks(state) == {"A": 0b0101, "B": 0b1010, live: 0b1111}
    assert TF.tdmWavePartition(state, live) == (4, (0, 1, 2, 3))
    assert TF.tdmWavePartition(state, dead) == (0, ())
    assert TF.tdmWaveMask(state, dead) == 0


def test_no_wave_idles_and_none_is_shared_in_a_set_with_a_live_member():
    """On one wave every live tensor issues from its own descriptor, so sets split from two waves."""
    states = list(_acceptedMaskStates((2, 4, 8, 16)))
    assert {len(TF.tdmLiveScales(state)) for state in states} == {0, 1, 2}
    for state in states:
        every = (1 << state["NumWaves"]) - 1
        for group in TF.liveGroups(state):
            masks = [TF.tdmWaveMask(state, tc) for tc in group]
            assert all(a & b == 0 for a, b in itertools.combinations(masks, 2)), (state, group)
            assert sum(masks) == every, (state, group, masks)


def _guardedWaves(guard, numWaves):
    """The waves a tdmWaveGuard answer admits, read from its documented meaning."""
    kind = guard[0]
    if kind == "all":
        return tuple(range(numWaves))
    if kind == "none":
        return ()
    if kind == "wave":
        return (guard[1],)
    if kind == "below":
        return tuple(range(guard[1]))
    if kind == "from":
        return tuple(range(guard[1], numWaves))
    assert kind == "parity", guard
    return tuple(range(guard[1], numWaves, 2))


def test_every_mask_the_writer_sees_is_emittable():
    """Every live member's waves map onto components 0..numComp-1 by an offset and a right
    shift of WaveIdx, and its guard admits exactly those waves, with or without MX scales."""
    for state in _acceptedMaskStates():
        for group in TF.liveGroups(state):
            for tc in group:
                numComp, waves = TF.tdmWavePartition(state, tc)
                _, shift, offset = TF.tdmWaveComponents(state, tc)
                components = [0] * len(waves) if shift is None else \
                    [(w - offset) >> shift for w in waves]
                assert components == list(range(numComp)), (state, tc, waves)
                assert _guardedWaves(TF.tdmWaveGuard(state, tc), state["NumWaves"]) == waves, \
                    (state, tc, waves)


@pytest.mark.parametrize("fuse, mx, scale", [
    (2, "mx_a", "MXSA"), (2, "mx_b", "MXSB"), (3, "mx_a", "MXSA"), (3, "mx_b", "MXSB")])
def test_a_lone_scale_on_a_block_set_rides_an_offset_run(fuse, mx, scale):
    """Waves 2-3 are no right shift of WaveIdx; the component subtracts an offset of 2 first."""
    state = _maskState(fuse, mx)
    assert TF.tdmWavePartition(state, scale) == (2, (2, 3))
    assert TF.tdmWaveComponents(state, scale) == (2, 0, 2)
    assert TF.tdmWaveGuard(state, scale) == ("from", 2)
    assert TF.tdmWaveRangeText(state, scale) == "waves 2-3"
    # The shared-set increment still selects one wave per member.
    with pytest.raises(TF.TdmArrangementNotEmittable, match=r"rides waves \(2, 3\)"):
        TF.tdmSoleWave(state, scale)


def test_wave_shares_with_every_member_live_split_each_full_set():
    for grouping, numWaves in itertools.product(TF.TDM_GROUPS.values(), range(1, 17)):
        shares = TF.tdmWaveShares(grouping, numWaves, lambda _tc: True)
        for group in grouping.groups:
            split = TF._waveShares(numWaves, len(group), grouping.layout)
            for tc, (waves, numComp) in zip(group, split):
                assert shares[tc] == (numComp, waves), (grouping.name, numWaves, tc)


@pytest.mark.parametrize("numWaves, numMembers, expected", [
    (4, 3, (((0, 1), 2), ((2,), 1), ((3,), 1))),
    (8, 3, (((0, 1, 2), 3), ((3, 4, 5), 3), ((6, 7), 2))),
    (2, 3, (((0,), 1), ((1,), 1), ((), 0))),
    (4, 2, (((0, 1), 2), ((2, 3), 2))),
])
def test_block_layout_hands_the_remainder_to_the_leading_members(numWaves, numMembers, expected):
    assert TF._waveShares(numWaves, numMembers, "block") == expected


def test_wave_shares_read_nothing_but_their_arguments():
    grouping = TF.TdmGrouping("probe", (("A", "MXSA", "MXSB"), ("B",)), layout="block")
    isLive = {"A", "B"}.__contains__
    shares = TF.tdmWaveShares(grouping, 4, isLive)
    assert shares == {"A": (4, (0, 1, 2, 3)), "MXSA": (0, ()), "MXSB": (0, ()),
                      "B": (4, (0, 1, 2, 3))}
    assert TF.tdmWaveShares(grouping, 4, isLive) == shares
    assert grouping.groups == (("A", "MXSA", "MXSB"), ("B",))


def test_wave_shares_split_each_set_over_its_live_members():
    """A set with no live member gives no member a wave; a lone live member takes them all."""
    grouping = TF.TDM_GROUPS["MX_AB"]
    assert TF.tdmWaveShares(grouping, 4, {"A", "B"}.__contains__) == {
        "A": (2, (0, 2)), "B": (2, (1, 3)), "MXSA": (0, ()), "MXSB": (0, ())}
    assert TF.tdmWaveShares(grouping, 4, {"A", "B", "MXSB"}.__contains__) == {
        "A": (2, (0, 2)), "B": (2, (1, 3)), "MXSA": (0, ()), "MXSB": (4, (0, 1, 2, 3))}


def test_wave_masks_leave_the_kernel_state_alone():
    state = _maskState(2, "no_mx")
    before = copy.deepcopy(state)
    TF.tdmWaveMasks(state)
    assert state == before


# ---------------------------------------------------------------------------
# Allocation matrix: TDMFuse 0-3 x MX state x NumWaves 1, 2, 4, on TDMInst=3 without
# TDMSplit or UseSubtileImpl.
# ---------------------------------------------------------------------------
# (TDMFuse, MX state, NumWaves): the resolved row, whether the requested row was declined,
# and the waves issuing A, MXSA, MXSB and B, () where no wave issues the tensor. Spelled
# out from the allocation rules so tdmWavePartition cannot redefine the expectation.
_ALLOCATION = {
    # TDMFuse=0 MX_AB: {A,B} + {MXSA,MXSB} by wave parity; a lone scale takes its set's waves.
    #                 row       declined  A             MXSA          MXSB          B
    (0, "no_mx", 1): ("MX_AB",  False,    (0,),         (),           (),           (0,)),
    (0, "mx_a", 1):  ("MX_AB",  False,    (0,),         (0,),         (),           (0,)),
    (0, "mx_b", 1):  ("MX_AB",  False,    (0,),         (),           (0,),         (0,)),
    (0, "mx", 1):    ("MX_AB",  False,    (0,),         (0,),         (0,),         (0,)),
    (0, "no_mx", 2): ("MX_AB",  False,    (0,),         (),           (),           (1,)),
    (0, "mx_a", 2):  ("MX_AB",  False,    (0,),         (0, 1),       (),           (1,)),
    (0, "mx_b", 2):  ("MX_AB",  False,    (0,),         (),           (0, 1),       (1,)),
    (0, "mx", 2):    ("MX_AB",  False,    (0,),         (0,),         (1,),         (1,)),
    (0, "no_mx", 4): ("MX_AB",  False,    (0, 2),       (),           (),           (1, 3)),
    (0, "mx_a", 4):  ("MX_AB",  False,    (0, 2),       (0, 1, 2, 3), (),           (1, 3)),
    (0, "mx_b", 4):  ("MX_AB",  False,    (0, 2),       (),           (0, 1, 2, 3), (1, 3)),
    (0, "mx", 4):    ("MX_AB",  False,    (0, 2),       (0, 2),       (1, 3),       (1, 3)),
    # TDMFuse=1 paired: {A,MXSA} + {MXSB,B} by wave parity; one wave declines to MX_AB.
    (1, "no_mx", 1): ("MX_AB",  True,     (0,),         (),           (),           (0,)),
    (1, "mx_a", 1):  ("MX_AB",  True,     (0,),         (0,),         (),           (0,)),
    (1, "mx_b", 1):  ("MX_AB",  True,     (0,),         (),           (0,),         (0,)),
    (1, "mx", 1):    ("MX_AB",  True,     (0,),         (0,),         (0,),         (0,)),
    (1, "no_mx", 2): ("paired", False,    (0, 1),       (),           (),           (0, 1)),
    (1, "mx_a", 2):  ("paired", False,    (0,),         (1,),         (),           (0, 1)),
    (1, "mx_b", 2):  ("paired", False,    (0, 1),       (),           (0,),         (1,)),
    (1, "mx", 2):    ("paired", False,    (0,),         (1,),         (0,),         (1,)),
    (1, "no_mx", 4): ("paired", False,    (0, 1, 2, 3), (),           (),           (0, 1, 2, 3)),
    (1, "mx_a", 4):  ("paired", False,    (0, 2),       (1, 3),       (),           (0, 1, 2, 3)),
    (1, "mx_b", 4):  ("paired", False,    (0, 1, 2, 3), (),           (0, 2),       (1, 3)),
    (1, "mx", 4):    ("paired", False,    (0, 2),       (1, 3),       (0, 2),       (1, 3)),
    # TDMFuse=2 A_MX: {A,MXSA,MXSB} + {B} in blocks; both scales need four waves.
    (2, "no_mx", 1): ("MX_AB",  True,     (0,),         (),           (),           (0,)),
    (2, "mx_a", 1):  ("MX_AB",  True,     (0,),         (0,),         (),           (0,)),
    (2, "mx_b", 1):  ("MX_AB",  True,     (0,),         (),           (0,),         (0,)),
    (2, "mx", 1):    ("MX_AB",  True,     (0,),         (0,),         (0,),         (0,)),
    (2, "no_mx", 2): ("A_MX",   False,    (0, 1),       (),           (),           (0, 1)),
    (2, "mx_a", 2):  ("A_MX",   False,    (0,),         (1,),         (),           (0, 1)),
    (2, "mx_b", 2):  ("A_MX",   False,    (0,),         (),           (1,),         (0, 1)),
    (2, "mx", 2):    ("MX_AB",  True,     (0,),         (0,),         (1,),         (1,)),
    (2, "no_mx", 4): ("A_MX",   False,    (0, 1, 2, 3), (),           (),           (0, 1, 2, 3)),
    (2, "mx_a", 4):  ("A_MX",   False,    (0, 1),       (2, 3),       (),           (0, 1, 2, 3)),
    (2, "mx_b", 4):  ("A_MX",   False,    (0, 1),       (),           (2, 3),       (0, 1, 2, 3)),
    (2, "mx", 4):    ("A_MX",   False,    (0, 1),       (2,),         (3,),         (0, 1, 2, 3)),
    # TDMFuse=3 B_MX: the mirror of 2, B owning the scale set and A on every wave.
    (3, "no_mx", 1): ("MX_AB",  True,     (0,),         (),           (),           (0,)),
    (3, "mx_a", 1):  ("MX_AB",  True,     (0,),         (0,),         (),           (0,)),
    (3, "mx_b", 1):  ("MX_AB",  True,     (0,),         (),           (0,),         (0,)),
    (3, "mx", 1):    ("MX_AB",  True,     (0,),         (0,),         (0,),         (0,)),
    (3, "no_mx", 2): ("B_MX",   False,    (0, 1),       (),           (),           (0, 1)),
    (3, "mx_a", 2):  ("B_MX",   False,    (0, 1),       (1,),         (),           (0,)),
    (3, "mx_b", 2):  ("B_MX",   False,    (0, 1),       (),           (1,),         (0,)),
    (3, "mx", 2):    ("MX_AB",  True,     (0,),         (0,),         (1,),         (1,)),
    (3, "no_mx", 4): ("B_MX",   False,    (0, 1, 2, 3), (),           (),           (0, 1, 2, 3)),
    (3, "mx_a", 4):  ("B_MX",   False,    (0, 1, 2, 3), (2, 3),       (),           (0, 1)),
    (3, "mx_b", 4):  ("B_MX",   False,    (0, 1, 2, 3), (),           (2, 3),       (0, 1)),
    (3, "mx", 4):    ("B_MX",   False,    (0, 1, 2, 3), (2,),         (3,),         (0, 1)),
}

# What a wave share means at its NumWaves: (numComp, tdmWaveMask, tdmWaveComponents,
# tdmWaveGuard). A wave's component is (WaveIdx - offset) >> shift; a None shift is one component.
_SHARE = {
    (1, (0,)):         (1, 0b1,    (1, None, 0), ("all",)),
    (2, (0,)):         (1, 0b01,   (1, None, 0), ("wave", 0)),
    (2, (1,)):         (1, 0b10,   (1, None, 0), ("wave", 1)),
    (2, (0, 1)):       (2, 0b11,   (2, 0, 0),    ("all",)),
    (4, (0, 1, 2, 3)): (4, 0b1111, (4, 0, 0),    ("all",)),
    (4, (0, 2)):       (2, 0b0101, (2, 1, 0),    ("parity", 0)),
    (4, (1, 3)):       (2, 0b1010, (2, 1, 0),    ("parity", 1)),
    (4, (0, 1)):       (2, 0b0011, (2, 0, 0),    ("below", 2)),
    (4, (2, 3)):       (2, 0b1100, (2, 0, 2),    ("from", 2)),
    (4, (2,)):         (1, 0b0100, (1, None, 0), ("wave", 2)),
    (4, (3,)):         (1, 0b1000, (1, None, 0), ("wave", 3)),
}
# tdmWaveComponents of a tensor no wave issues (a dead scale): no component, an id never read.
_NO_SHARE_COMPONENTS = (0, 0, 0)


def _expectedMasks(fuse, mx, numWaves):
    shares = _ALLOCATION[fuse, mx, numWaves][2:]
    return {tc: _SHARE[numWaves, waves][1] for tc, waves in zip(TDM_TENSORS, shares) if waves}


def test_the_allocation_matrix_covers_every_row_mx_state_and_wave_count():
    assert set(_ALLOCATION) == set(itertools.product(sorted(TDM_FUSE_GROUPING), _MX_STATES,
                                                     (1, 2, 4)))
    for (_fuse, _mx, numWaves), entry in _ALLOCATION.items():
        for waves in entry[2:]:
            if waves:
                assert _guardedWaves(_SHARE[numWaves, waves][3], numWaves) == waves, waves


@pytest.mark.parametrize("fuse, mx, numWaves", sorted(_ALLOCATION))
def test_wave_allocation_matrix(fuse, mx, numWaves):
    row, declined = _ALLOCATION[fuse, mx, numWaves][:2]
    state = _maskState(fuse, mx, numWaves)
    assert TF.tdmGrouping(state).name == row
    assert TF.tdmGroupingAccepted(state) is (not declined)
    for tc, waves in zip(TDM_TENSORS, _ALLOCATION[fuse, mx, numWaves][2:]):
        if not waves:
            assert TF.tdmWavePartition(state, tc) == (0, ()), tc
            assert TF.tdmWaveMask(state, tc) == 0, tc
            assert TF.tdmWaveGuard(state, tc) == ("none",), tc
            assert TF.tdmWaveComponents(state, tc)[0] == 0, tc
            continue
        numComp, mask, components, guard = _SHARE[numWaves, waves]
        assert TF.tdmWavePartition(state, tc) == (numComp, waves), tc
        assert TF.tdmWaveMask(state, tc) == mask, tc
        assert TF.tdmWaveComponents(state, tc) == components, tc
        assert TF.tdmWaveGuard(state, tc) == guard, tc
    assert TF.tdmWaveMasks(state) == _expectedMasks(fuse, mx, numWaves)
    if numWaves > 1:
        every = (1 << numWaves) - 1
        for group in TF.liveGroups(state):
            masks = [TF.tdmWaveMask(state, tc) for tc in group]
            assert all(a & b == 0 for a, b in itertools.combinations(masks, 2)), group
            assert sum(masks) == every, (group, masks)


def test_a_tensor_no_wave_issues_has_the_dead_member_components():
    """A dead scale carries no component, so its WaveIdx shift is never read."""
    got = {}
    for (fuse, mx, numWaves), entry in sorted(_ALLOCATION.items()):
        state = _maskState(fuse, mx, numWaves)
        for tc, waves in zip(TDM_TENSORS, entry[2:]):
            if not waves:
                components = TF.tdmWaveComponents(state, tc)
                got.setdefault(components, []).append((fuse, mx, numWaves, tc))
    assert set(got) == {_NO_SHARE_COMPONENTS}, {k: v[:3] for k, v in got.items()}


# The live set carrying a data tensor and the live scales, on each row that keeps its
# grouping; TDMFuse=1 with both scales puts one on each data set and takes the paired arms.
_SHARED_SCALE_SET = {
    (1, "mx_a"): ("A", "MXSA"), (1, "mx_b"): ("MXSB", "B"),
    (2, "mx_a"): ("A", "MXSA"), (2, "mx_b"): ("A", "MXSB"), (2, "mx"): ("A", "MXSA", "MXSB"),
    (3, "mx_a"): ("B", "MXSA"), (3, "mx_b"): ("B", "MXSB"), (3, "mx"): ("B", "MXSA", "MXSB"),
}


@pytest.mark.parametrize("fuse, mx, numWaves", sorted(_ALLOCATION))
def test_shared_scale_set_over_the_allocation_matrix(fuse, mx, numWaves):
    state = _maskState(fuse, mx, numWaves)
    declined = _ALLOCATION[fuse, mx, numWaves][1]
    group = None if declined else _SHARED_SCALE_SET.get((fuse, mx))
    owner = None if group is None else ("A" if "A" in group else "B")
    assert TF.tdmSharedScaleSet(state) == group
    assert TF.tdmSharedScaleSetOwner(state) == owner
    assert TF.tdmSharedScaleSetActive(state) is (group is not None)
    assert TF.tdmSharedSetOrder(state, "A", "B") == (("B", "A") if owner == "B" else ("A", "B"))
    assert tdmFusePaired(state) is (fuse == 1 and mx == "mx" and not declined)


# Tensors whose own GlobalReadIncs HalfPLR's last-iteration mask zeroes on more than one wave,
# by whether A and B carry MX scales, then TDMFuse.
_HALFPLR_MASKED_INCS = {
    (False, False): {0: (), 1: ("A", "B"), 2: ("A", "B"), 3: ("A", "B")},
    (True, True): {0: (), 1: (), 2: ("B",), 3: ("A",)},
    (True, False): {0: ("MXSA",), 1: ("B",), 2: ("B",), 3: ("A",)},
    (False, True): {0: ("MXSB",), 1: ("A",), 2: ("B",), 3: ("A",)},
}


@pytest.mark.parametrize("numWaves", [1, 2, 4])
@pytest.mark.parametrize("halfPLR", [0, 1, 2, 3])
@pytest.mark.parametrize("mx", sorted(_MX_STATES))
@pytest.mark.parametrize("fuse", sorted(TDM_FUSE_GROUPING))
def test_halfplr_masks_the_own_increment_of_every_set_without_a_shadow(fuse, mx, halfPLR, numWaves):
    """Parity and paired sets lose a shadow increment; every other tensor loses its own."""
    state = dict(_maskState(fuse, mx, numWaves), HalfPLR=halfPLR)
    mxA, mxB = (bool(block) for block in _MX_STATES[mx])
    want = _HALFPLR_MASKED_INCS[(mxA, mxB)][fuse]
    if not halfPLR:
        want = ()
    elif numWaves == 1:
        # One wave keeps a descriptor per tensor, and the mask zeroes each one's own increment.
        want = ("A", "B") + ("MXSA",) * mxA + ("MXSB",) * mxB
    elif fuse in (2, 3) and mxA and mxB and numWaves == 2:
        want = ()  # three members on two waves: TDMFuse declines to 0
    assert TF.tdmHalfPlrMaskedIncs(state) == want


_MX_STATE_NAMES = {blocks: name for name, blocks in _MX_STATES.items()}


def _maskStateId(state):
    blocks = (state["ProblemType"]["MXBlockA"], state["ProblemType"]["MXBlockB"])
    return "tdmfuse%d-%s-w%d" % (state["TDMFuse"], _MX_STATE_NAMES[blocks], state["NumWaves"])


@pytest.mark.parametrize("state", list(_acceptedMaskStates((1, 2, 4, 8))), ids=_maskStateId)
def test_halfplr_tail_reenables_every_set_the_last_iteration_disables(state):
    """The K tail loads through every descriptor set the last main-loop iterations disabled."""
    writer = Stub()
    if state["NumWaves"] > 1:
        _, disabled = _disabledOwners(writer, state)
    else:
        # One wave disables each live tensor's own set.
        disabled = {tc for tc in TDM_TENSORS if TF.tdmMemberIsLive(state, tc)}
    tPA = {"tensorChar": "A", "MX": {"tensorChar": "MXSA"}}
    tPB = {"tensorChar": "B", "MX": {"tensorChar": "MXSB"}}
    names = writer.tdmHalfPlrReenableSgprs(state, tPA, tPB)
    assert len(names) == len(set(names))
    assert disabled <= {writer.tdmDescriptorSetOwner(state, name[len("tdm"):-len("Group0+0")])
                        for name in names}
