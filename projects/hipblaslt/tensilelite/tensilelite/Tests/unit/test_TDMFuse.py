################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for tensilelite.Components.TDMFuse."""
import copy
import itertools
import re
import types

import pytest
import yaml
from rocisa.code import Module, RegSet
from rocisa.container import sgpr
from rocisa.enum import CacheScope, NonVolatile, TemporalHint

from tensilelite.Common.GlobalParameters import defaultSolution
from tensilelite.Common.ValidParameters import validParameters
from tensilelite.Components import DecouplePGR as DP
from tensilelite.Components import TDMFuse as TF
from tensilelite.Components.DecouplePGR import decouplePGRBlocks
from tensilelite.Components.TDMFuse import (TDM_FUSE_GROUPING, tdmBothTensors,
                                        tdmFusePaired, tdmGrouping,
                                        tdmWaveComponents, tdmWavePartition)
from tensilelite.KernelWriterAssembly import KernelWriterAssembly, _tdmEmitWaveCarriesSCC

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
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.Common.Capabilities import makeIsaInfoMap
    from tensilelite.Toolchain.Validators import validateToolchain

    cxx = validateToolchain("amdclang++")
    isa = gfxToIsa("gfx1250")
    iim = makeIsaInfoMap([isa], cxx)
    if not iim[isa].asmCaps["SupportedISA"]:
        pytest.skip("amdclang++ in this environment does not support gfx1250")
    return iim


@pytest.fixture(scope="module")
def assembler():
    from tensilelite.Toolchain.Assembly import makeAssemblyToolchain
    from tensilelite.Toolchain.Validators import validateToolchain, ToolchainDefaults

    cxx = validateToolchain("amdclang++")
    bundler = validateToolchain(ToolchainDefaults.OFFLOAD_BUNDLER)
    return makeAssemblyToolchain(cxx, bundler, "default").assembler


@pytest.fixture(scope="module")
def _gp_gfx1250(gfx1250_iim):
    from tensilelite.Common.GlobalParameters import globalParameters, assignGlobalParameters

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
    from tensilelite.Common.Architectures import gfxToIsa
    from tensilelite.SolutionStructs.Solution import Solution
    from tensilelite.SolutionStructs.Validators.MatrixInstruction import (
        matrixInstructionToMIParameters,
    )

    # The MI parameters carry "ISA" and are merged last, so an ISA override must feed them too.
    isa = overrides.pop("ISA", gfxToIsa("gfx1250"))
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
         "MXSB is in A's descriptor group but follows B's LDS block count"),
        # The mirror has to name the other scale: at 3 the set sits on B, so
        # MXSB follows its own owner and MXSA is the member carrying the second
        # cadence.
        (3, {"PrefetchGlobalReadA": 1, "PrefetchGlobalReadB": 2},
         "MXSA is in B's descriptor group but follows A's LDS block count"),
        (3, {"PrefetchGlobalReadA": 2, "PrefetchGlobalReadB": 1},
         "MXSA is in B's descriptor group but follows A's LDS block count"),
        (2, {"MatrixInstruction": [16, 16, 128, 1, 1, 2, 4, 2, 4], "WorkGroup": [32, 8, 1]},
         "TDMFuse=2 with MX scales on both tensors supports 2 or 4 waves; got NumWaves=8"),
        (1, {"TDMInst": 1}, "TDMA and TDMB must be enabled simultaneously"),
        # A tile whose MX scales fit one load per wave, so the subtile
        # scale-load budget does not answer first.
        (1, {"UseSubtileImpl": True,
             "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2]},
         "TDMFuse=1 is not implemented with UseSubtileImpl on more than one wave"),
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


def test_b_mx_takes_pap_with_scales_on_both_tensors():
    """PAP refreshes each set through its owner; only a lone scale on a data set is refused."""
    assert TF.tdmPapRejectReason(ks()) is None
    lone = dict(ks(), ProblemType=dict(ks()["ProblemType"], MXBlockB=0))
    reason = TF.tdmPapRejectReason(lone)
    assert reason is not None
    assert "B_MX" in reason and "{B,MXSA}" in reason


@pytest.mark.parametrize("numWaves", [1, 3, 5, 8, 12, 16])
def test_b_mx_is_declined_away_from_two_and_four_waves(numWaves):
    """With both scales B_MX takes the 2/1/1 split on 2 or 4 waves; every other wave count declines."""
    assert TF.tdmGroupingAccepted(ks(NumWaves=numWaves)) is False
    assert TF.tdmGrouping(ks(NumWaves=numWaves)).name == TF.TDM_GROUPING_DEFAULT


@pytest.mark.parametrize("fuse", [1, 2, 3])
@pytest.mark.parametrize("mx", ["mx", "no_mx", "mx_a", "mx_b"])
def test_one_wave_declines_rows_one_to_three(fuse, mx):
    """Rows 1-3 split their sets over the waves, so one wave declines them to TDMFuse=0's row."""
    state = _maskState(fuse, mx, numWaves=1)
    assert TF.tdmGroupingAccepted(state) is False
    assert TF.tdmGrouping(state).name == TF.TDM_GROUPING_DEFAULT
    assert TF.tdmWavePartition(state, "A") == TF.tdmWavePartition(_maskState(0, mx, numWaves=1), "A")


@pytest.mark.parametrize("mx", ["mx", "no_mx", "mx_a", "mx_b"])
def test_one_wave_tdmfuse0_shares_no_set_and_stashes_nothing(mx):
    """One wave keeps a descriptor per tensor: no set is shared over waves and no member is stashed."""
    state = _maskState(0, mx, numWaves=1)
    assert TF.tdmGroupingAccepted(state) is True
    assert TF.tdmGrouping(state).name == TF.TDM_GROUPING_DEFAULT
    assert TF.tdmWaveSeparated(state) is False
    assert TF.tdmSeparateABDescriptors(state) is False
    assert TF.tdmStashedMembers(state) == ()
    assert TF.tdmSharedScaleSet(state) is None
    assert TF.tdmSharedScaleSetActive(state) is False
    assert TF.tdmPairOnEveryWave(state, "A", "B") is False
    assert TF.tdmPapRejectReason(state) is None
    # The parity split the single-wave writer reads: no component on either data tensor.
    assert TF.tdmWavePartition(state, "A") == (0, (0,))
    assert TF.tdmWavePartition(state, "B") == (0, ())


@pytest.mark.parametrize("fuse", [0, 1, 2, 3])
@pytest.mark.parametrize("mx", ["mx", "no_mx", "mx_a", "mx_b"])
def test_one_wave_subtile_keeps_a_descriptor_per_tensor(fuse, mx):
    """Subtile keeps a descriptor per tensor; rows 1-3 decline there as on every one-wave kernel."""
    state = dict(_maskState(fuse, mx, numWaves=1), UseSubtileImpl=True)
    default = dict(_maskState(0, mx, numWaves=1), UseSubtileImpl=True)
    assert TF.tdmGroupingAccepted(state) is (fuse == 0)
    assert TF.tdmGrouping(state).name == TF.TDM_GROUPING_DEFAULT
    assert TF.tdmStashedMembers(state) == ()
    for tc in TDM_TENSORS:
        assert TF.tdmWavePartition(state, tc) == TF.tdmWavePartition(default, tc), tc


@pytest.mark.parametrize("fuse, owner, other", [(2, "A", "B"), (3, "B", "A")])
def test_two_waves_put_both_scales_on_wave_1_and_stash_mxsb(fuse, owner, other):
    """Owner on wave 0, MXSA and MXSB on wave 1 through its set, MXSB stashed; the other on both."""
    state = _maskState(fuse, "mx", numWaves=2)
    assert TF.tdmGroupingAccepted(state) is True
    assert TF.tdmGrouping(state).name == TDM_FUSE_GROUPING[fuse]
    assert TF.tdmWavePartition(state, owner) == (1, (0,))
    assert TF.tdmWavePartition(state, "MXSA") == (1, (1,))
    assert TF.tdmWavePartition(state, "MXSB") == (1, (1,))
    assert TF.tdmWavePartition(state, other) == (2, (0, 1))
    assert TF.tdmWaveGuard(state, "MXSA") == TF.tdmWaveGuard(state, "MXSB") == ("wave", 1)
    assert TF.tdmStashedMembers(state) == ("MXSB",)
    assert TF.tdmSetOwner(state, "MXSA") == owner
    assert TF.tdmSetOwner(state, "MXSB") == "MXSB"
    assert TF.tdmSharedScaleSetActive(state) is True
    assert TF.tdmHalfPlrMaskedIncs(dict(state, HalfPLR=1)) == (other, "MXSB")


def test_only_two_wave_both_scale_rows_stash_a_member():
    """Every other state, one wave included, keeps no stash."""
    for fuse, mx, numWaves in itertools.product(sorted(TDM_FUSE_GROUPING), _MX_STATES,
                                                (1, 2, 4, 8, 16)):
        state = _maskState(fuse, mx, numWaves)
        expected = ("MXSB",) if (fuse in (2, 3) and mx == "mx" and numWaves == 2) else ()
        assert TF.tdmStashedMembers(state) == expected, (fuse, mx, numWaves)


@pytest.mark.parametrize("numWaves, waves", [
    (2, ((0,), (1,), (1,))), (4, ((0, 1), (2,), (3,))),
    (8, ((0, 1, 2, 3), (4, 5), (6, 7))), (16, (tuple(range(8)), (8, 9, 10, 11), (12, 13, 14, 15)))])
def test_a_three_member_block_gives_the_owner_half_and_each_scale_a_quarter(numWaves, waves):
    shares = TF.tdmWaveShares(TF.TDM_GROUPS["A_MX"], numWaves, lambda tc: True)
    assert tuple(shares[tc][1] for tc in ("A", "MXSA", "MXSB")) == waves
    assert tuple(shares[tc][0] for tc in ("A", "MXSA", "MXSB")) == tuple(len(w) for w in waves)


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
                     "tdmPairOnEveryWave", "removeStaggerCopies", "tdmHalfPlrReenableSgprs",
                     "removeStaggerIncs", "graIncrementMask", "releaseGlobalReadIncsSgprsAfterTdmWaveSep",
                     "_emitTdmWaveSetSkipSCC"):
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
    state = _ksOwner(ProblemType={"MXBlockA": 0, "MXBlockB": 32})
    assert _tokenWriter()._dcpTdmIssueTokens(state, "A") == [7, 9]


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


def test_wave_components_encode_each_shape(monkeypatch):
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


# ---------------------------------------------------------------------------
# Cache hints of a descriptor set: the data tensor decides
# ---------------------------------------------------------------------------
HINT_CAPS = {"HasTHModifier": True}
DEV, SYS = CacheScope.SCOPE_DEV, CacheScope.SCOPE_SYS
NO_TH, NO_NV = TemporalHint.TH_NONE, NonVolatile.NV_NONE


@pytest.mark.parametrize("members, deciders", [
    (("A", "B"), (0, 1)),             # TDMFuse=0, the data pair: both decide
    (("MXSA", "MXSB"), (0, 1)),       # TDMFuse=0, the scale set: both decide
    (("A", "MXSA"), (0,)),            # TDMFuse=1
    (("MXSB", "B"), (1,)),
    (("A", "MXSA", "MXSB"), (0,)),    # TDMFuse=2
    (("B", "MXSA", "MXSB"), (0,)),    # TDMFuse=3
    (("A",), (0,)),
    (("MXSA",), (0,)),
    (("Metadata",), (0,)),
    (("A", "Metadata"), (0,)),        # a member other than A or B never decides next to a data tensor
])
def test_hint_deciders_by_set(members, deciders):
    assert TF.tdmHintDeciders(members) == deciders


def _hint_writer(kernel, caps):
    """The real tdmLoadModifiers and _tdmTensorParameters over the hints stored in `kernel`."""
    def tensor(tc):
        return {"NonTemporal": kernel.get("NonTemporal" + tc, 0),
                "TemporalHint": kernel.get("TemporalHint" + tc, 0)}

    writer = types.SimpleNamespace(
        states=types.SimpleNamespace(asmCaps=caps),
        tPA=dict(tensor("A"), MX=tensor("MXSA")),
        tPB=dict(tensor("B"), MX=tensor("MXSB")))
    writer.isTdmWaveSeparated = lambda k: KernelWriterAssembly.isTdmWaveSeparated(writer, k)
    writer._tdmTensorParameters = lambda tc: KernelWriterAssembly._tdmTensorParameters(writer, tc)
    return writer


def _issued(kernel, tc, caps=HINT_CAPS):
    """(scope, th, nv) of the tensor_load_to_lds that issues `tc`'s descriptor set, or None."""
    mods = KernelWriterAssembly.tdmLoadModifiers(_hint_writer(kernel, caps), kernel, tc)
    return None if mods is None else (mods.scope, mods.th, mods.nv)


@pytest.mark.parametrize("fuse", [1, 2, 3])
def test_fused_set_follows_its_data_tensor(fuse):
    """A scale tensor shares a data tensor's set: scope, th and nv come from A or B alone."""
    kernel = dict(_kernel(fuse),
                  NonTemporalA=2, NonTemporalB=3, NonTemporalMXSA=1, NonTemporalMXSB=0,
                  TemporalHintA=1, TemporalHintB=2, TemporalHintMXSA=3, TemporalHintMXSB=5,
                  NonVolatileA=1, NonVolatileB=0, NonVolatileMXSA=0, NonVolatileMXSB=1)
    assert _issued(kernel, "A") == (DEV, TemporalHint.TH_NT, NonVolatile.NV)
    assert _issued(kernel, "B") == (SYS, TemporalHint.TH_HT, NO_NV)


def test_default_grouping_still_needs_agreement_within_each_set():
    """TDMFuse=0: {A,B} and {MXSA,MXSB} each keep only the fields their members agree on."""
    kernel = _kernel(0)
    agree = dict(kernel, NonTemporalA=2, NonTemporalB=2, TemporalHintA=1, TemporalHintB=2,
                 NonTemporalMXSA=3, NonTemporalMXSB=3, TemporalHintMXSA=5, TemporalHintMXSB=5)
    assert _issued(agree, "A") == (DEV, NO_TH, NO_NV)                       # th differs
    assert _issued(agree, "MXSA") == (SYS, TemporalHint(5), NO_NV)
    clash = dict(kernel, NonTemporalA=2, NonTemporalB=0, TemporalHintA=1, TemporalHintB=2,
                 NonTemporalMXSA=1, NonTemporalMXSB=1)
    assert _issued(clash, "A") is None                                      # nothing agrees
    assert _issued(clash, "MXSB") == (CacheScope.SCOPE_SE, NO_TH, NO_NV)
    # The hints of A and B never reach the scale set.
    assert _issued(dict(kernel, NonTemporalA=2, NonTemporalB=2), "MXSA") is None


@pytest.mark.parametrize("fuse", [1, 2, 3])
def test_unsuffixed_nontemporal_reaches_every_fused_set(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    """NonTemporal=2 sets A and B but not MXSA/MXSB; the data tensor decides, so scope survives."""
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, NonTemporal=2)
    assert sol.get("Valid") is True, out
    state = sol._state
    assert (state["NonTemporalA"], state["NonTemporalB"]) == (2, 2)
    assert (state["NonTemporalMXSA"], state["NonTemporalMXSB"]) == (0, 0)
    caps = {"HasTHModifier": bool(state["_HasTemporalHint"])}
    assert caps["HasTHModifier"]
    assert _issued(state, "A", caps) == (DEV, NO_TH, NO_NV)
    assert _issued(state, "B", caps) == (DEV, NO_TH, NO_NV)
_NO_MX = {"MacDataTypeB": "F8", "MXBlockA": 0, "MXBlockB": 0}
_TWO_WAVES = {"MatrixInstruction": [16, 16, 128, 1, 1, 2, 16, 2, 1], "WorkGroup": [32, 2, 1]}
_ONE_WAVE_64X64 = {"MatrixInstruction": [16, 16, 128, 1, 1, 4, 4, 1, 1], "WorkGroup": _ONE_WAVE_WG}
_WAVES = {1: _ONE_WAVE_64X64, 2: _TWO_WAVES, 4: {}}
_ONE_SCALE = {"mx_a": _MXSA_ONLY, "mx_b": _MXSB_ONLY}
_ONE_SIDED_MX = "MX scales on one tensor only"
_STREAMK = {"StreamK": 3, "StreamKForceDPOnly": 1}


@pytest.mark.parametrize("numWaves", [2, 4])
@pytest.mark.parametrize("mx", sorted(_ONE_SCALE))
@pytest.mark.parametrize("fuse", sorted(TDM_FUSE_GROUPING))
def test_solution_on_one_mx_scale_for_every_grouping_and_wave_count(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, mx, numWaves):
    """One-sided MX derives on gfx1250 TDM kernels of more than one wave."""
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse,
                       ProblemType=_ONE_SCALE[mx], **_WAVES[numWaves])
    assert sol.get("Valid") is True, out
    assert _ONE_SIDED_MX not in out, out
    state = sol._state
    assert state["NumWaves"] == numWaves
    assert TF.tdmLiveScales(state) == (("MXSA",) if mx == "mx_a" else ("MXSB",))
    assert tdmGrouping(state).name == TDM_FUSE_GROUPING[fuse]
    assert TF.tdmWaveMasks(state) == _expectedMasks(fuse, mx, numWaves)


@pytest.mark.parametrize("overrides, found", [
    ({"UseSubtileImpl": True, "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2]},
     "are not implemented with UseSubtileImpl:"),
    (dict(_STREAMK, PrefetchAcrossPersistent=1), "PrefetchAcrossPersistent=1"),
    (dict(_STREAMK, ReuseAcrossPersistent=1), "ReuseAcrossPersistent=1"),
    ({"PrefetchGlobalReadA": 1, "PrefetchGlobalReadB": 2}, "PGRA=1, PGRB=2"),
    ({"PrefetchGlobalReadA": 2, "PrefetchGlobalReadB": 1}, "PGRA=2, PGRB=1"),
], ids=["subtile", "pap", "rap", "divergent_pair", "divergent_pair_ba"])
@pytest.mark.parametrize("mx", sorted(_ONE_SCALE))
@pytest.mark.parametrize("fuse", [0, 1, 2, 3])
def test_solution_rejects_one_mx_scale_outside_its_limits(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, mx, overrides, found):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse,
                       ProblemType=_ONE_SCALE[mx], **overrides)
    assert sol.get("Valid") is False
    assert _ONE_SIDED_MX in out, out
    assert found in out, out


@pytest.mark.parametrize("key", ["PrefetchAcrossPersistent", "ReuseAcrossPersistent"])
@pytest.mark.parametrize("mx", sorted(_ONE_SCALE))
def test_one_sided_pap_and_rap_limit_spares_a_non_persistent_kernel(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, mx, key):
    """The one-sided PAP/RAP refusal is a persistent kernel's. Solution's constructor already refuses or clears both
    keys on a non-persistent policy, so the limit meets one only when assignDerivedParameters re-derives a state, as
    a re-derivation does: there a non-persistent kernel is not refused and has the key cleared."""
    from tensilelite.SolutionStructs.Solution import Solution

    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=0, ProblemType=_ONE_SCALE[mx])
    assert sol.get("Valid") is True, out
    state = copy.deepcopy(sol._state)
    assert not state.get("_PersistentLoop"), "the derived kernel must be non-persistent"
    state[key] = 1
    state["AssignedDerivedParameters"] = False
    state["AssignedProblemIndependentDerivedParameters"] = False
    Solution.assignDerivedParameters(state, False, True, False, gfx1250_iim, assembler.rocm_version)
    out = capsys.readouterr().out
    assert "%s=1 on a persistent kernel" % key not in out, out
    assert state["Valid"] is True, out
    assert not state[key]


@pytest.mark.parametrize("mx", sorted(_ONE_SCALE))
def test_solution_rejects_one_mx_scale_off_the_tdm_scale_load(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, mx):
    """gfx1250 swizzles MX scales in memory, which only the TDM scale load does, so the
    transport check refuses any other scale load and names it."""
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=0,
                       ProblemType=_ONE_SCALE[mx], MXLoadInst="BufferLoad")
    assert sol.get("Valid") is False
    assert "MXLoadInst=BufferLoad" in out, out


def _isaWithGfx1250Caps(gfx1250_iim, version):
    """ISA `version` carrying gfx1250's caps, for a rule keyed on the ISA itself."""
    from tensilelite.Common.Architectures import gfxToIsa

    isa = gfxToIsa("gfx1250")
    other = type(isa)(*version) if hasattr(isa, "_fields") else tuple(version)
    iim = dict(gfx1250_iim)
    iim[other] = gfx1250_iim[isa]
    return other, iim


@pytest.mark.parametrize("mx", sorted(_ONE_SCALE))
def test_solution_rejects_one_mx_scale_on_another_isa(_gp_gfx1250, gfx1250_iim, assembler, capsys, mx):
    isa, iim = _isaWithGfx1250Caps(gfx1250_iim, (12, 6, 0))
    sol, out = _derive(iim, assembler, capsys, TDMFuse=0, ProblemType=_ONE_SCALE[mx], ISA=isa)
    assert sol.get("Valid") is False
    assert _ONE_SIDED_MX in out, out
    assert ("are implemented only for gfx1250 (ISA 12.5.x) TDM kernels; got ISA 12.6.0, whose MX "
            "scale path (UseSubtileImpl or buffer scale loads) has no one-sided support") in out, out


# ---------------------------------------------------------------------------
# One-sided MX kernels at the register level. Needs amdclang++ gfx1250; skipped otherwise.
# CPU-only: Python and rocisa codegen, nothing is assembled or launched. SGPR names resolve
# through the kernel's .set table, so a member that aliases its set owner's registers counts
# as that set.
# ---------------------------------------------------------------------------
# 64x64 on one wave and on waves [2,1]; 64x128 on waves [2,2].
_ONE_SIDED_MI = {
    1: [16, 16, 128, 1, 1, 4, 4, 1, 1],
    2: [16, 16, 128, 1, 1, 2, 4, 2, 1],
    4: [16, 16, 128, 1, 1, 2, 4, 2, 2],
}
_ONE_SIDED_CELLS = [(fuse, scale, numWaves)
                    for fuse in (0, 1, 2, 3)
                    for numWaves in (2, 4)
                    for scale in ("MXSA", "MXSB")]
_ONE_SIDED_IDS = ["tdmfuse%d-%s-w%d" % cell for cell in _ONE_SIDED_CELLS]
# The data tensor whose descriptor set the lone scale rides on TDMFuse 1-3.
_SCALE_SET_OWNER = {(1, "MXSA"): "A", (1, "MXSB"): "B", (2, "MXSA"): "A",
                    (2, "MXSB"): "A", (3, "MXSA"): "B", (3, "MXSB"): "B"}
_GROUP1_SGPRS = 8
_ONE_SIDED_FORK = {
    "DepthU": 256,
    "TDMInst": 3,
    "TDMSplit": False,
    "PrefetchGlobalRead": 2,
    "ScheduleIterAlg": 4,
    "StaggerU": 0,
    "TransposeLDS": -1,
    "LdsPadA": -1,
    "LdsPadB": -1,
    "LdsBlockSizePerPadA": -1,
    "LdsBlockSizePerPadB": -1,
    "LdsPadMetadata": 0,
    "LDSTrInst": False,
    "PrefetchLocalRead": 1,
    "StoreRemapVectorWidth": 0,
    "GlobalSplitU": 1,
    "GlobalSplitUAlgorithm": "MultipleBuffer",
    "GlobalReadVectorWidthA": -1,
    "GlobalReadVectorWidthB": -1,
    "VectorWidthA": -1,
    "VectorWidthB": -1,
    "LocalReadVectorWidth": -1,
    "1LDSBuffer": 0,
    "SourceSwap": False,
    "DirectToVgprSparseMetadata": False,
    "WorkGroupMapping": 1,
    "StoreVectorWidth": -1,
    "DirectToVgprA": False,
    "DirectToVgprB": False,
    "WavefrontSize": 32,
    "ExpandPointerSwap": False,
}


def _oneSidedConfig(fuse, scale, numWaves):
    problemType = {
        "OperationType": "GEMM",
        "MacDataTypeA": "F8",
        "MacDataTypeB": "F8",
        "DataType": "F8",
        "DestDataType": "s",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "Activation": False,
        "SupportUserArgs": True,
        "UseBias": 0,
        "MXBlockA": 32 if scale == "MXSA" else 0,
        "MXBlockB": 32 if scale == "MXSB" else 0,
        "DataTypeMXSA": "E8",
        "DataTypeMXSB": "E8",
    }
    fork = dict(_ONE_SIDED_FORK, MatrixInstruction=_ONE_SIDED_MI[numWaves], TDMFuse=fuse)
    return {
        "GlobalParameters": {"PrintSolutionRejectionReason": True},
        "BenchmarkProblems": [[problemType, {
            "BenchmarkCommonParameters": [{"KernelLanguage": ["Assembly"]}],
            "ForkParameters": [{key: [value]} for key, value in fork.items()],
        }]],
    }


_ASM_SET = re.compile(r"^\s*\.set\s+(\w+)\s*,\s*(\S+)", re.M)
_ASM_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
_ASM_SGPR_START = re.compile(r"^s\[([^\]:]+)")
# Mnemonic prefixes whose first operand is a source: they write no register operand.
_ASM_NO_DESTINATION = ("s_cmp", "s_bitcmp", "s_cbranch", "s_branch", "s_setpc", "v_cmpx",
                       "tensor_", "s_store", "buffer_store", "global_store", "flat_store",
                       "scratch_store", "ds_store", "ds_write")


class _AsmKernel:
    """One emitted kernel: its instructions, label positions and .set table."""

    def __init__(self, source):
        self.source = source
        self.prologue = source.partition("label_LoopBeginL:")[0]
        self.symbols = {}
        for name, value in _ASM_SET.findall(source):
            self.symbols.setdefault(name, value)
        self.instructions = []
        self.labels = {}
        for line in _ASM_BLOCK_COMMENT.sub("", source).splitlines():
            code = line.split("//", 1)[0].strip()
            if not code or code.startswith("."):
                continue
            if code.endswith(":"):
                self.labels.setdefault(code[:-1], len(self.instructions))
                continue
            mnemonic, _, rest = code.partition(" ")
            self.instructions.append(
                (mnemonic, [op.strip() for op in rest.split(",") if op.strip()]))

    def index(self, expr, depth=0):
        """The SGPR index `expr` names (``sgprtdmAGroup1+2``, ``52``), or None."""
        total = 0
        for term in expr.split("+"):
            term = term.strip()
            if term.isdigit():
                total += int(term)
                continue
            if term not in self.symbols or depth > 16:
                return None
            value = self.index(self.symbols[term], depth + 1)
            if value is None:
                return None
            total += value
        return total

    def operandIndex(self, operand):
        match = _ASM_SGPR_START.match(operand)
        return self.index(match.group(1)) if match else None

    def setIndex(self, *names):
        """The SGPR index of the first of `names` the kernel defines, or None."""
        for name in names:
            if "sgpr" + name in self.symbols:
                return self.index("sgpr" + name)
        return None

    def reads(self, name):
        """Instructions reading SGPR `name`, as a source or as an operand nothing writes."""
        pattern = re.compile(r"\bsgpr%s\b" % re.escape(name))
        return [(mnemonic, operands) for mnemonic, operands in self.instructions
                if any(pattern.search(op) for op in
                       (operands if mnemonic.startswith(_ASM_NO_DESTINATION) else operands[1:]))]

    def tdmWrites(self, first, count, begin=0, end=None):
        """Instructions in [begin, end) whose destination is a TDM register in first..first+count-1."""
        hits = []
        for mnemonic, operands in self.instructions[begin:end]:
            if not operands or mnemonic.startswith(_ASM_NO_DESTINATION) \
                    or "sgprtdm" not in operands[0]:
                continue
            index = self.operandIndex(operands[0])
            if index is not None and first <= index < first + count:
                hits.append((mnemonic, operands))
        return hits

    def tdmIssues(self, begin=0, end=None):
        """The Group0 SGPR index of each tensor_load_to_lds in instructions[begin:end]."""
        return [self.operandIndex(operands[0])
                for mnemonic, operands in self.instructions[begin:end]
                if mnemonic.startswith("tensor_load_to_lds") and operands]

    def mainLoop(self):
        begin = self.labels.get("label_LoopBeginL")
        end = self.labels.get("label_LoopEndL")
        assert begin is not None and end is not None and begin < end, \
            "no label_LoopBeginL ... label_LoopEndL main loop"
        return begin, end


def _oneSidedCarriers(fuse, scale, numWaves, tc):
    """Tensors whose name may program the descriptor set carrying `tc`, its own first."""
    if fuse != 0:
        return [tc, _SCALE_SET_OWNER[fuse, scale]] if tc == scale else [tc]
    return [tc, "MXSA", "MXSB"] if tc == scale else [tc, "A"]


def _oneSidedLiveSets(kernel, fuse, scale, numWaves):
    """{live tensor: Group0 SGPR index of the descriptor set carrying it}."""
    return {tc: kernel.setIndex(*("tdm%sGroup0" % name
                                  for name in _oneSidedCarriers(fuse, scale, numWaves, tc)))
            for tc in ("A", "B", scale)}


def _oneSidedDead(scale):
    return "MXSB" if scale == "MXSA" else "MXSA"


@pytest.fixture(scope="module")
def oneSidedKernel(gfx1250_iim, tmp_path_factory):
    """Derive and emit each one-sided cell's kernel once, on first use."""
    from config_harness import derive_states, emit_kernels_from_config

    root = tmp_path_factory.mktemp("tdm_onesided_mx")
    kernels = {}

    def emit(fuse, scale, numWaves):
        cell = (fuse, scale, numWaves)
        if cell not in kernels:
            path = root / ("tdmfuse%d_%s_w%d.yaml" % cell)
            path.write_text(yaml.safe_dump(_oneSidedConfig(*cell)))
            states = derive_states(path, arch="gfx1250", limit_solutions=1)
            assert len(states) == 1, "%s derives no solution" % (cell,)
            assert (states[0]["TDMFuse"], states[0]["NumWaves"]) == (fuse, numWaves), cell
            results = emit_kernels_from_config(path, limit=1, arch="gfx1250")
            assert len(results) == 1, cell
            base, source, err = results[0]
            assert err == 0, (cell, base)
            assert "gfx1250" in source, (cell, base)
            kernels[cell] = _AsmKernel(source)
        return kernels[cell]

    return emit


@pytest.mark.parametrize("fuse, scale, numWaves", _ONE_SIDED_CELLS, ids=_ONE_SIDED_IDS)
def test_onesided_mx_dead_scale_address_is_never_read(oneSidedKernel, fuse, scale, numWaves):
    kernel = oneSidedKernel(fuse, scale, numWaves)
    reads = kernel.reads("Address" + _oneSidedDead(scale))
    assert reads == [], reads[:4]


@pytest.mark.parametrize("fuse, scale, numWaves", _ONE_SIDED_CELLS, ids=_ONE_SIDED_IDS)
def test_onesided_mx_live_scale_descriptor_is_programmed(oneSidedKernel, fuse, scale, numWaves):
    """The set carrying the scale gets its Group1 writes; on TDMFuse 1-3, by the scale's own setup."""
    kernel = oneSidedKernel(fuse, scale, numWaves)
    if fuse == 0:
        group1 = kernel.setIndex(*("tdm%sGroup1" % name
                                   for name in _oneSidedCarriers(fuse, scale, numWaves, scale)))
    else:
        owner = _SCALE_SET_OWNER[fuse, scale]
        group1 = kernel.setIndex("tdm%sGroup1" % owner)
        own = kernel.setIndex("tdm%sGroup1" % scale)
        assert own in (None, group1), "%s does not ride %s's descriptor set" % (scale, owner)
        # The owner's own setup writes the same Group1 registers, so it alone proves nothing.
        assert re.search(r"\bmoves? %s\b" % scale, kernel.prologue), \
            "no guarded setup moves %s on %s's set" % (scale, owner)
    assert group1 is not None, "no Group1 registers carry %s" % scale
    assert kernel.tdmWrites(group1, _GROUP1_SGPRS), \
        "nothing programs the Group1 registers carrying %s" % scale


@pytest.mark.parametrize("fuse, scale, numWaves", _ONE_SIDED_CELLS, ids=_ONE_SIDED_IDS)
def test_onesided_mx_every_live_descriptor_set_issues_inside_the_main_loop(
        oneSidedKernel, fuse, scale, numWaves):
    kernel = oneSidedKernel(fuse, scale, numWaves)
    sets = _oneSidedLiveSets(kernel, fuse, scale, numWaves)
    assert None not in sets.values(), sets
    # A and B share one set (TDMFuse=0) or the scale shares its data tensor's set (TDMFuse 1-3).
    assert len(set(sets.values())) == 2, sets
    begin, end = kernel.mainLoop()
    issued = set(kernel.tdmIssues(begin, end))
    missing = {tc: index for tc, index in sets.items() if index not in issued}
    assert not missing, (missing, sorted(i for i in issued if i is not None))


@pytest.mark.parametrize("fuse, scale, numWaves", _ONE_SIDED_CELLS, ids=_ONE_SIDED_IDS)
def test_onesided_mx_no_wave_issues_a_descriptor_only_the_dead_scale_owns(
        oneSidedKernel, fuse, scale, numWaves):
    kernel = oneSidedKernel(fuse, scale, numWaves)
    deadSet = kernel.setIndex("tdm%sGroup0" % _oneSidedDead(scale))
    liveSets = set(_oneSidedLiveSets(kernel, fuse, scale, numWaves).values())
    # The dead scale either has no set of its own or aliases a live one.
    assert deadSet is None or deadSet in liveSets, (deadSet, sorted(liveSets))
    issued = set(kernel.tdmIssues()) - {None}
    assert issued <= liveSets, sorted(issued - liveSets)


@pytest.mark.parametrize("fuse, scale, numWaves", _ONE_SIDED_CELLS, ids=_ONE_SIDED_IDS)
def test_onesided_mx_every_live_set_advances_inside_the_main_loop(oneSidedKernel, fuse, scale, numWaves):
    """Each live set moves its global address (Group0 words 2-3) to the next K slice every iteration."""
    kernel = oneSidedKernel(fuse, scale, numWaves)
    begin, end = kernel.mainLoop()
    sets = _oneSidedLiveSets(kernel, fuse, scale, numWaves)
    still = {tc: index for tc, index in sets.items() if not kernel.tdmWrites(index + 2, 1, begin, end)}
    assert not still, still


@pytest.mark.parametrize("numWaves", [2, 4])
def test_onesided_mx_tdmfuse0_loads_a_lone_mxsb_through_the_scale_set(oneSidedKernel, numWaves):
    """MXSA leads the TDMFuse=0 scale set; with it dead the set still loads MXSB."""
    kernel = oneSidedKernel(0, "MXSB", numWaves)
    scaleSet = kernel.setIndex("tdmMXSBGroup0", "tdmMXSAGroup0")
    assert scaleSet is not None
    assert scaleSet in kernel.tdmIssues(), scaleSet


@pytest.mark.parametrize("fuse, scale, numWaves", _ONE_SIDED_CELLS, ids=_ONE_SIDED_IDS)
def test_onesided_mx_every_live_set_swaps_its_lds_buffer_inside_the_main_loop(
        oneSidedKernel, fuse, scale, numWaves):
    """Each live set moves its LDS write address (Group0 word 1) to the other buffer every iteration."""
    kernel = oneSidedKernel(fuse, scale, numWaves)
    begin, end = kernel.mainLoop()
    sets = _oneSidedLiveSets(kernel, fuse, scale, numWaves)
    unswapped = {tc: index for tc, index in sets.items() if not kernel.tdmWrites(index + 1, 1, begin, end)}
    assert not unswapped, unswapped


@pytest.mark.parametrize("fuse, scale", [(fuse, scale) for fuse in (2, 3) for scale in ("MXSA", "MXSB")])
def test_onesided_mx_lone_scale_on_a_block_set_subtracts_its_wave_offset(oneSidedKernel, fuse, scale):
    """On four waves the scale is the run on waves 2-3, so its setup and its tail reset use WaveIdx - 2."""
    kernel = oneSidedKernel(fuse, scale, 4)
    before = kernel.source.partition("label_LoopBeginL:")[0]
    after = kernel.source.partition("label_LoopEndL:")[2]
    assert len(re.findall(r"WaveIdx - 2\b", before)) >= 2, "component id and global offset"
    assert re.search(r"WaveIdx - 2\b", after), "tail reset"


@pytest.mark.parametrize("fuse", [1, 2, 3])
@pytest.mark.parametrize("waves", [_TWO_WAVES, {}], ids=["2w", "4w"])
def test_solution_accepts_rows_one_to_three_without_scales(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, waves):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, ProblemType=_NO_MX, **waves)
    assert sol.get("Valid") is True, out
    assert tdmGrouping(sol._state).name == TDM_FUSE_GROUPING[fuse]
    assert TF.tdmPairOnEveryWave(sol._state, "A", "B") is True
    assert tdmFusePaired(sol._state) is False


_MX_PROBLEMS = {"mx": {}, "no_mx": _NO_MX, "mx_a": _MXSA_ONLY, "mx_b": _MXSB_ONLY}


@pytest.mark.parametrize("fuse", [1, 2, 3])
@pytest.mark.parametrize("mx", sorted(_MX_PROBLEMS))
def test_solution_rejects_rows_one_to_three_on_one_wave(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, mx):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse,
                       ProblemType=_MX_PROBLEMS[mx], **_ONE_WAVE_64X64)
    assert sol.get("Valid") is False
    assert "TDMFuse=%d needs more than one wave (NumWaves=1)" % fuse in out, out


@pytest.mark.parametrize("mx", ["mx", "no_mx"])
def test_solution_takes_tdmfuse0_on_one_wave_without_a_shared_set(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, mx):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=0,
                       ProblemType=_MX_PROBLEMS[mx], **_ONE_WAVE_64X64)
    assert sol.get("Valid") is True, out
    assert sol._state["NumWaves"] == 1
    assert tdmGrouping(sol._state).name == TF.TDM_GROUPING_DEFAULT
    assert TF.tdmStashedMembers(sol._state) == ()
    assert TF.tdmSharedScaleSet(sol._state) is None


@pytest.mark.parametrize("fuse", [2, 3])
def test_solution_accepts_both_scales_on_two_waves(_gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, **_TWO_WAVES)
    assert sol.get("Valid") is True, out
    assert sol._state["NumWaves"] == 2
    assert tdmGrouping(sol._state).name == TDM_FUSE_GROUPING[fuse]
    assert TF.tdmStashedMembers(sol._state) == ("MXSB",)


def _tdmConfig(fuse, numWaves, mxA, mxB, halfPLR=0):
    """_oneSidedConfig's problem and fork with any MX state."""
    config = _oneSidedConfig(fuse, "MXSA", numWaves)
    problemType = config["BenchmarkProblems"][0][0]
    problemType["MXBlockA"], problemType["MXBlockB"] = mxA, mxB
    if halfPLR:
        config["BenchmarkProblems"][0][1]["ForkParameters"].append({"HalfPLR": [halfPLR]})
    return config


@pytest.fixture(scope="module")
def tdmKernel(gfx1250_iim, tmp_path_factory):
    """Derive and emit one kernel per (TDMFuse, NumWaves, MXBlockA, MXBlockB), on first use."""
    from config_harness import derive_states, emit_kernels_from_config

    root = tmp_path_factory.mktemp("tdm_fuse_cells")
    kernels = {}

    def emit(fuse, numWaves, mxA, mxB, halfPLR=0):
        cell = (fuse, numWaves, mxA, mxB, halfPLR)
        if cell not in kernels:
            path = root / ("tdmfuse%d_w%d_a%d_b%d_hplr%d.yaml" % cell)
            path.write_text(yaml.safe_dump(_tdmConfig(*cell)))
            states = derive_states(path, arch="gfx1250", limit_solutions=1)
            assert len(states) == 1, "%s derives no solution" % (cell,)
            assert (states[0]["TDMFuse"], states[0]["NumWaves"]) == (fuse, numWaves), cell
            results = emit_kernels_from_config(path, limit=1, arch="gfx1250")
            assert len(results) == 1, cell
            base, source, err = results[0]
            assert err == 0, (cell, base)
            kernels[cell] = (base, source)
        return kernels[cell]

    return emit


def test_one_wave_tdmfuse0_keeps_a_complete_descriptor_per_tensor(tdmKernel):
    """One wave keeps a Group0 and a Group1 for each of A and B, and each main-loop load names its
    own tensor's pair: no set is shared and no member is stashed."""
    _, source = tdmKernel(0, 1, 0, 0)
    k = _AsmKernel(source)
    group0 = [k.setIndex("tdm%sGroup0" % tc) for tc in ("A", "B")]
    group1 = [k.setIndex("tdm%sGroup1" % tc) for tc in ("A", "B")]
    assert None not in group0 + group1 and len(set(group0 + group1)) == 4, (group0, group1)
    begin, end = k.mainLoop()
    pairs = {(k.operandIndex(ops[0]), k.operandIndex(ops[1]))
             for mnemonic, ops in k.instructions[begin:end] if mnemonic.startswith("tensor_load_to_lds")}
    assert pairs == set(zip(group0, group1)), sorted(pairs)


@pytest.mark.parametrize("mxA, mxB", [(32, 0), (0, 32)], ids=["mx_a", "mx_b"])
def test_four_wave_lone_scale_splits_by_k_group(tdmKernel, mxA, mxB):
    """TDMFuse=0's lone scale rides all four waves with its tile split by K-group, two waves per
    K-group, as on two waves; the row split loses the pieces a K tail or a partial tile cuts."""
    _, source = tdmKernel(0, 4, mxA, mxB)
    assert "woffset = wCompId * mt//numComp(" not in source, "a scale piece is split by rows"
    assert "numMxKGroups(2) // numComp(2)" in source


@pytest.fixture(scope="module")
def sia3Kernel(gfx1250_iim, tmp_path_factory):
    """Emit the four-wave MX-both kernel with PrefetchGlobalRead 1, ScheduleIterAlg 3 and StaggerU 16."""
    from config_harness import derive_states, emit_kernels_from_config

    root = tmp_path_factory.mktemp("tdm_fuse_sia3")
    kernels = {}

    def emit(fuse):
        if fuse not in kernels:
            config = _tdmConfig(fuse, 4, 32, 32)
            for entry in config["BenchmarkProblems"][0][1]["ForkParameters"]:
                for key, value in (("PrefetchGlobalRead", 1), ("ScheduleIterAlg", 3), ("StaggerU", 16)):
                    if key in entry:
                        entry[key] = [value]
            path = root / ("tdmfuse%d_pgr1_sia3.yaml" % fuse)
            path.write_text(yaml.safe_dump(config))
            states = derive_states(path, arch="gfx1250", limit_solutions=1)
            assert len(states) == 1, "TDMFuse=%d derives no solution" % fuse
            assert (states[0]["PrefetchGlobalRead"], states[0]["ScheduleIterAlg"]) == (1, 3), fuse
            results = emit_kernels_from_config(path, limit=1, arch="gfx1250")
            assert len(results) == 1 and results[0][2] == 0, fuse
            kernels[fuse] = results[0][1]
        return kernels[fuse]

    return emit


# Scalar instructions that leave SCC alone; any other s_ instruction may write it.
_SCC_KEEPERS = ("s_mov", "s_cmov", "s_cselect", "s_wait", "s_nop", "s_barrier", "s_set_vgpr_msb",
                "s_delay", "s_sleep", "s_branch", "s_cbranch", "s_setprio")
_SGPR_PAIR = re.compile(r"s\[(\d+):(\d+)\]")


def _sccSource(k, at):
    """Index of the last instruction before `at` that may write SCC."""
    for i in range(at - 1, -1, -1):
        mnemonic = k.instructions[i][0]
        if mnemonic.startswith("s_") and not mnemonic.startswith(_SCC_KEEPERS):
            return i
    return None


def _lastWriter(k, operand, at):
    for i in range(at - 1, -1, -1):
        mnemonic, ops = k.instructions[i]
        if ops and ops[0] == operand and not mnemonic.startswith(_ASM_NO_DESTINATION):
            return i
    return None


def _assertHalvesShareScc(k, lo, hi, atLo, atHi, what):
    """The s_cselect pair writing (lo, hi), and the pairs feeding it, each read one compare's SCC."""
    wLo, wHi = _lastWriter(k, lo, atLo), _lastWriter(k, hi, atHi)
    assert wLo is not None and wHi is not None, what
    if not (k.instructions[wLo][0].startswith("s_cselect") and k.instructions[wHi][0].startswith("s_cselect")):
        return
    sLo, sHi = _sccSource(k, wLo), _sccSource(k, wHi)
    assert sLo == sHi and k.instructions[sLo][0].startswith(("s_cmp", "s_bitcmp")), \
        "%s: the halves read SCC from %s and %s" % (what, k.instructions[sLo], k.instructions[sHi])
    srcLo, srcHi = k.instructions[wLo][1][1], k.instructions[wHi][1][1]
    if re.fullmatch(r"s\d+", srcLo) and re.fullmatch(r"s\d+", srcHi):
        _assertHalvesShareScc(k, srcLo, srcHi, wLo, wHi, "%s <- (%s, %s)" % (what, srcLo, srcHi))


@pytest.mark.parametrize("fuse", [0, 1, 2, 3])
def test_pgr1_sia3_wrapped_tdm_increments_keep_their_compare(sia3Kernel, fuse):
    """PrefetchGlobalRead 1 with ScheduleIterAlg 3 spreads each TDM address increment over MFMA slots.
    Both halves of a 64-bit wrapped increment must read SCC from the same compare: an SCC write between
    them (the next sub-iteration's LDS bank swap) adds a WrapU high word on an ordinary iteration and
    sends the next load out of bounds. With no stashed member the global-read window stays develop's:
    on TDMFuse=0, A and MXSA give two items each (the RAP null-descriptor slot and the load)."""
    source = sia3Kernel(fuse)
    k = _AsmKernel(source)
    begin, end = k.mainLoop()
    checked = 0
    for at in range(begin, end):
        mnemonic, ops = k.instructions[at]
        if mnemonic != "s_add_u64" or len(ops) < 3 or "sgprtdm" not in ops[0]:
            continue
        pair = _SGPR_PAIR.fullmatch(ops[2])
        if pair:
            lo, hi = "s%s" % pair.group(1), "s%d" % (int(pair.group(1)) + 1)
            _assertHalvesShareScc(k, lo, hi, at, at, ops[0])
            checked += 1
    assert checked, "no wrapped TDM increment in the main loop"
    if fuse == 0:
        loop = source.partition("label_LoopBeginL:")[2].partition("label_LoopEndL:")[0]
        assert set(re.findall(r"grEndMfmaIndex:(\d+)", loop)) == {"3"}, re.findall(r"grEndMfmaIndex:\d+", loop)


def _tdmLoads(k, begin, end):
    """tensor_load_to_lds in instructions[begin:end] as (index, Group0 operand, Group1 operand)."""
    return [(i, ops[0], ops[1]) for i, (mnemonic, ops) in enumerate(k.instructions[begin:end], begin)
            if mnemonic == "tensor_load_to_lds"]


@pytest.mark.parametrize("fuse, owner, other", [(2, "A", "B"), (3, "B", "A")])
def test_two_waves_issue_mxsb_through_mxsas_set_after_it(tdmKernel, fuse, owner, other):
    """MXSA rides the owner's set on wave 1 and MXSB issues after it with its stash as Group0 and the
    set's Group1, whose MXSB fields are written before that load and MXSA's after it; no Group0 word
    moves between the set and the stash. The stash's enable word holds MXSB disabled on wave 0, which
    issues a NULL load through it."""
    _, source = tdmKernel(fuse, 2, 32, 32)
    k = _AsmKernel(source)
    ownerSet, otherSet, stash = (k.setIndex("tdm%sGroup0" % tc) for tc in (owner, other, "MXSB"))
    assert k.setIndex("tdmMXSAGroup0") == ownerSet
    assert stash is not None and stash not in (ownerSet, otherSet)
    assert "sgprtdmMXSBGroup1" not in k.symbols, "MXSB must not hold a Group1 of its own"
    # StinkyTofu may schedule other instructions between the wave compare and the s_cmov.
    assert re.search(r"s_cmp_lg_u32\s+s\[sgprWaveIdx\],\s*1\b", k.prologue) \
        and re.search(r"s_cmov_b32\s+s\[sgprtdmMXSBGroup0\+0\],\s*0\b", k.prologue), \
        "wave 0 must hold MXSB's stash disabled"
    ownerGroup0 = "s[sgprtdm%sGroup0+" % owner
    assert not [ops for _, ops in k.instructions
                if any(op.startswith(ownerGroup0) for op in ops) and any("sgprtdmMXSBGroup0" in op for op in ops)], \
        "no Group0 word may move between the set and MXSB's stash"
    begin, end = k.mainLoop()
    assert set(k.tdmIssues(begin, end)) == {ownerSet, otherSet, stash}
    loads = _tdmLoads(k, begin, end)
    stride = "s[sgprtdm%sGroup1+5]" % owner
    stashLoads = [n for n, load in enumerate(loads) if load[1].startswith("s[sgprtdmMXSBGroup0:")]
    assert stashLoads, "MXSB must issue through its stash"
    switched = []
    for n in stashLoads:
        at, _, group1 = loads[n]
        assert group1.startswith("s[sgprtdm%sGroup1:" % owner), group1
        prev = loads[n - 1][0] if n else begin
        nxt = loads[n + 1][0] if n + 1 < len(loads) else end
        before = [ops for _, ops in k.instructions[prev + 1:at] if ops and ops[0] == stride]
        after = [ops for _, ops in k.instructions[at + 1:nxt] if ops and ops[0] == stride]
        # MXSB's own load has MXSB's stride written last before it; the NULL load's window ends with MXSA's.
        if before and any("sgprSizeJ" in op for op in before[-1]):
            assert after and any("sgprSizeI" in op for op in after[0]), "MXSA's stride must come back"
            switched.append(n)
    assert switched, "MXSB's wave must switch the set's fields around its load"
    assert len(switched) < len(stashLoads), "the waves outside MXSB's share need their NULL load"
    # MXSA has no load of its own here: the shared set's load issues it on wave 1, before MXSB.
    firstOwner = min(n for n, load in enumerate(loads) if load[1].startswith("s[sgprtdm%sGroup0:" % owner))
    assert firstOwner < min(switched), "MXSA (the shared set's load) must issue before MXSB's"
    assert not [ops for _, ops in k.instructions
                if ops and ops[0] == "s[sgprtdmABIncs]" and any("GlobalReadIncsMXSB" in op for op in ops)]
@pytest.mark.parametrize("fuse, owner", [(2, "A"), (3, "B")])
def test_two_waves_advance_mxsas_set_and_mxsbs_stash(tdmKernel, fuse, owner):
    """Each main-loop iteration advances MXSA's global address in the set and MXSB's in its stash."""
    _, source = tdmKernel(fuse, 2, 32, 32)
    k = _AsmKernel(source)
    begin, end = k.mainLoop()

    def advanced(word):
        """Adds of the address pair onto itself; the advance is a 64-bit s_add_u64."""
        return [ops for mnemonic, ops in k.instructions[begin:end]
                if mnemonic in ("s_add_u32", "s_add_u64") and len(ops) > 2
                and ops[0].startswith(word) and ops[1] == ops[0]]

    assert advanced("s[sgprtdm%sGroup0+2" % owner), "the set's address must advance"
    assert advanced("s[sgprtdmMXSBGroup0+2"), "the stash's address must advance"


_TWO_SET_CELLS = [(fuse, waves, mxA, mxB) for fuse in (2, 3) for waves in (2, 4)
                  for mxA, mxB in ((32, 32), (32, 0), (0, 32), (0, 0))]


@pytest.mark.parametrize("fuse, numWaves, mxA, mxB", _TWO_SET_CELLS,
                         ids=["tdmfuse%d-w%d-a%d-b%d" % cell for cell in _TWO_SET_CELLS])
def test_tdmfuse_2_3_hold_two_complete_descriptor_sets(tdmKernel, fuse, numWaves, mxA, mxB):
    """TDMFuse 2/3 keep exactly two complete TDM descriptor sets, with MX scales on both tensors, on
    one or on neither; on two waves with both scales MXSB keeps a Group0 stash and no Group1."""
    _, source = tdmKernel(fuse, numWaves, mxA, mxB)
    k = _AsmKernel(source)
    stashed = numWaves == 2 and bool(mxA and mxB)
    live = ["A", "B"] + [tc for tc, mx in (("MXSA", mxA), ("MXSB", mxB))
                         if mx and not (tc == "MXSB" and stashed)]
    group1 = {k.setIndex("tdm%sGroup1" % tc) for tc in live}
    assert None not in group1 and len(group1) == 2, sorted(group1, key=str)
    assert ("sgprtdmMXSBGroup1" in k.symbols) == bool(mxB and not stashed)


@pytest.mark.parametrize("fuse", [2, 3])
def test_two_waves_halfplr_masks_mxsb_and_reenables_it_on_wave_1_only(tdmKernel, fuse):
    """The K tail must not wake wave 0's disabled MXSB stash, and MXSB's own increment stops."""
    _, source = tdmKernel(fuse, 2, 32, 32, halfPLR=1)
    k = _AsmKernel(source)
    begin, end = k.mainLoop()
    assert [ops for mnemonic, ops in k.instructions[begin:end]
            if mnemonic == "s_cmov_b32" and ops[0].startswith("s[sgprGlobalReadIncsMXSB")
            and ops[1] == "0"], "the last-iteration mask must zero MXSB's own increment"
    tail = source.partition("HalfPLR: re-enable TDM & align LDS buffer")[2]
    assert tail, "no HalfPLR K-tail adjust"
    tail = "\n".join(tail.splitlines()[:40])
    assert re.search(r"s_cmp_eq_u32\s+s\[sgprWaveIdx\],\s*1\b[^\n]*\n\s*"
                     r"s_cselect_b32\s+s\[sgprtdmMXSBGroup0\+0\],\s*1,\s*0\b", tail), tail
    assert not re.search(r"s_mov_b32\s+s\[sgprtdmMXSBGroup0\+0\],\s*1\b", tail), tail


@pytest.mark.parametrize("overrides, clause", [
    ({"UseSubtileImpl": True, "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2]},
     "is not implemented with UseSubtileImpl on more than one wave"),
])
@pytest.mark.parametrize("fuse", [1, 2, 3])
def test_solution_rejects_rows_one_to_three_without_scales_outside_their_limits(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, overrides, clause):
    overrides = dict(overrides)
    problemType = overrides.pop("ProblemType", _NO_MX)
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, ProblemType=problemType,
                       **overrides)
    assert sol.get("Valid") is False
    assert "TDMFuse=%d %s" % (fuse, clause) in out, out


@pytest.mark.parametrize("fuse", sorted(TDM_FUSE_GROUPING))
def test_only_rows_without_scales_put_both_data_tensors_on_every_wave(fuse):
    assert TF.tdmPairOnEveryWave(ks(fuse=fuse), "A", "B") is False
    assert TF.tdmPairOnEveryWave(ks(fuse=fuse), "MXSA", "MXSB") is False
    unscaled = ks(fuse=fuse, ProblemType={"MXBlockA": 0, "MXBlockB": 0})
    assert TF.tdmPairOnEveryWave(unscaled, "A", "B") is (fuse != 0)


def test_a_single_member_set_alone_rides_every_wave():
    """TDMFuse 2/3 leave one data tensor alone on every wave; tdmPairOnEveryWave needs both alone, and one wave separates nothing."""
    assert TF.tdmRidesEveryWave(ks(fuse=2), "B") is True
    assert TF.tdmRidesEveryWave(ks(fuse=3), "A") is True
    unscaledOneWave = ks(fuse=1, NumWaves=1, ProblemType={"MXBlockA": 0, "MXBlockB": 0})
    assert TF.tdmRidesEveryWave(unscaledOneWave, "A") is False
    # One wave declines rows 1-3 to TDMFuse=0's row; nothing is wave-separated.
    assert TF.tdmGrouping(unscaledOneWave).name == TF.TDM_GROUPING_DEFAULT


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
    {"TDMSplit": True}, {"UseSubtileImpl": True}, {"TDMInst": 1},
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


def test_live_members_split_their_set_without_overlap_or_idle_waves():
    """Sets split their waves from two waves on."""
    states = list(_acceptedMaskStates((2, 4, 8, 16)))
    assert {len(TF.tdmLiveScales(state)) for state in states} == {0, 1, 2}
    for state in states:
        every = (1 << state["NumWaves"]) - 1
        own = TF.tdmStashedMembers(state)
        for group in TF.liveGroups(state):
            # A member on a set of its own shares a wave; the others split the set's waves.
            masks = [TF.tdmWaveMask(state, tc) for tc in group if tc not in own]
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
    shift of WaveIdx, and its guard admits exactly those waves, with or without MX scales.
    One wave is TDMFuse=0's single-wave path, which issues each tensor from its own descriptor
    and reads no mask."""
    for state in _acceptedMaskStates():
        if state["NumWaves"] == 1:
            assert state["TDMFuse"] == 0 and not TF.tdmWaveSeparated(state), state
            assert TF.tdmStashedMembers(state) == (), state
            continue
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
    (8, 3, (((0, 1, 2, 3), 4), ((4, 5), 2), ((6, 7), 2))),
    (2, 3, (((0,), 1), ((1,), 1), ((1,), 1))),
    (4, 2, (((0, 1), 2), ((2, 3), 2))),
])
def test_block_layout_gives_two_members_halves_and_three_members_quarters(numWaves, numMembers,
                                                                          expected):
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
# Allocation matrix: TDMFuse 0-3 x MX state x NumWaves 2 and 4, on TDMInst=3 without
# TDMSplit or UseSubtileImpl.
# ---------------------------------------------------------------------------
# (TDMFuse, MX state, NumWaves): the resolved row, whether the requested row was declined,
# and the waves issuing A, MXSA, MXSB and B, () where no wave issues the tensor. Spelled
# out from the allocation rules so tdmWavePartition cannot redefine the expectation.
_ALLOCATION = {
    # TDMFuse=0 MX_AB: {A,B} + {MXSA,MXSB} by wave parity; a lone scale takes its set's waves.
    #                 row       declined  A             MXSA          MXSB          B
    (0, "no_mx", 2): ("MX_AB",  False,    (0,),         (),           (),           (1,)),
    (0, "mx_a", 2):  ("MX_AB",  False,    (0,),         (0, 1),       (),           (1,)),
    (0, "mx_b", 2):  ("MX_AB",  False,    (0,),         (),           (0, 1),       (1,)),
    (0, "mx", 2):    ("MX_AB",  False,    (0,),         (0,),         (1,),         (1,)),
    (0, "no_mx", 4): ("MX_AB",  False,    (0, 2),       (),           (),           (1, 3)),
    (0, "mx_a", 4):  ("MX_AB",  False,    (0, 2),       (0, 1, 2, 3), (),           (1, 3)),
    (0, "mx_b", 4):  ("MX_AB",  False,    (0, 2),       (),           (0, 1, 2, 3), (1, 3)),
    (0, "mx", 4):    ("MX_AB",  False,    (0, 2),       (0, 2),       (1, 3),       (1, 3)),
    # TDMFuse=1 paired: {A,MXSA} + {MXSB,B} by wave parity.
    (1, "no_mx", 2): ("paired", False,    (0, 1),       (),           (),           (0, 1)),
    (1, "mx_a", 2):  ("paired", False,    (0,),         (1,),         (),           (0, 1)),
    (1, "mx_b", 2):  ("paired", False,    (0, 1),       (),           (0,),         (1,)),
    (1, "mx", 2):    ("paired", False,    (0,),         (1,),         (0,),         (1,)),
    (1, "no_mx", 4): ("paired", False,    (0, 1, 2, 3), (),           (),           (0, 1, 2, 3)),
    (1, "mx_a", 4):  ("paired", False,    (0, 2),       (1, 3),       (),           (0, 1, 2, 3)),
    (1, "mx_b", 4):  ("paired", False,    (0, 1, 2, 3), (),           (0, 2),       (1, 3)),
    (1, "mx", 4):    ("paired", False,    (0, 2),       (1, 3),       (0, 2),       (1, 3)),
    # TDMFuse=2 A_MX: {A,MXSA,MXSB} + {B} in blocks; with both scales A takes half the
    # waves and each scale a quarter, so on two waves MXSA and MXSB share wave 1.
    (2, "no_mx", 2): ("A_MX",   False,    (0, 1),       (),           (),           (0, 1)),
    (2, "mx_a", 2):  ("A_MX",   False,    (0,),         (1,),         (),           (0, 1)),
    (2, "mx_b", 2):  ("A_MX",   False,    (0,),         (),           (1,),         (0, 1)),
    (2, "mx", 2):    ("A_MX",   False,    (0,),         (1,),         (1,),         (0, 1)),
    (2, "no_mx", 4): ("A_MX",   False,    (0, 1, 2, 3), (),           (),           (0, 1, 2, 3)),
    (2, "mx_a", 4):  ("A_MX",   False,    (0, 1),       (2, 3),       (),           (0, 1, 2, 3)),
    (2, "mx_b", 4):  ("A_MX",   False,    (0, 1),       (),           (2, 3),       (0, 1, 2, 3)),
    (2, "mx", 4):    ("A_MX",   False,    (0, 1),       (2,),         (3,),         (0, 1, 2, 3)),
    # TDMFuse=3 B_MX: the mirror of 2, B owning the scale set and A on every wave.
    (3, "no_mx", 2): ("B_MX",   False,    (0, 1),       (),           (),           (0, 1)),
    (3, "mx_a", 2):  ("B_MX",   False,    (0, 1),       (1,),         (),           (0,)),
    (3, "mx_b", 2):  ("B_MX",   False,    (0, 1),       (),           (1,),         (0,)),
    (3, "mx", 2):    ("B_MX",   False,    (0, 1),       (1,),         (1,),         (0,)),
    (3, "no_mx", 4): ("B_MX",   False,    (0, 1, 2, 3), (),           (),           (0, 1, 2, 3)),
    (3, "mx_a", 4):  ("B_MX",   False,    (0, 1, 2, 3), (2, 3),       (),           (0, 1)),
    (3, "mx_b", 4):  ("B_MX",   False,    (0, 1, 2, 3), (),           (2, 3),       (0, 1)),
    (3, "mx", 4):    ("B_MX",   False,    (0, 1, 2, 3), (2,),         (3,),         (0, 1)),
}

# What a wave share means at its NumWaves: (numComp, tdmWaveMask, tdmWaveComponents,
# tdmWaveGuard). A wave's component is (WaveIdx - offset) >> shift; a None shift is one component.
_SHARE = {
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
                                                     (2, 4)))
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
    every = (1 << numWaves) - 1
    own = TF.tdmStashedMembers(state)
    assert own == (("MXSB",) if fuse in (2, 3) and mx == "mx" and numWaves == 2 else ())
    for group in TF.liveGroups(state):
        # Members on the set's registers split its waves; one on its own set shares a wave.
        masks = [TF.tdmWaveMask(state, tc) for tc in group if tc not in own]
        assert all(a & b == 0 for a, b in itertools.combinations(masks, 2)), group
        assert sum(masks) == every, (group, masks)
        for tc in own:
            if tc in group:
                assert TF.tdmWaveMask(state, tc) & sum(masks), tc


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
    # A declined row resolves to MX_AB.
    kept = _ALLOCATION[fuse, mx, numWaves][0] == TDM_FUSE_GROUPING[fuse]
    group = _SHARED_SCALE_SET.get((fuse, mx)) if kept else None
    owner = None if group is None else ("A" if "A" in group else "B")
    assert TF.tdmSharedScaleSet(state) == group
    assert TF.tdmSharedScaleSetOwner(state) == owner
    assert TF.tdmSharedScaleSetActive(state) is (group is not None)
    assert TF.tdmSharedSetOrder(state, "A", "B") == (("B", "A") if owner == "B" else ("A", "B"))
    assert tdmFusePaired(state) is (fuse == 1 and mx == "mx" and kept)


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
        # One wave takes TDMFuse=0 only, declining rows 1-3 to its row: a Group0 per tensor,
        # and the mask zeroes each one's own increment.
        assert TF.tdmGroupingAccepted(state) is (fuse == 0)
        want = ("A", "B") + ("MXSA",) * mxA + ("MXSB",) * mxB
    elif fuse in (2, 3) and mxA and mxB and numWaves == 2:
        want += ("MXSB",)  # MXSB shares wave 1 with MXSA on a set of its own
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
        # One wave disables each live tensor's own Group0.
        disabled = {tc for tc in TDM_TENSORS if TF.tdmMemberIsLive(state, tc)}
    tPA = {"tensorChar": "A", "MX": {"tensorChar": "MXSA"}}
    tPB = {"tensorChar": "B", "MX": {"tensorChar": "MXSB"}}
    names = writer.tdmHalfPlrReenableSgprs(state, tPA, tPB)
    assert len(names) == len(set(names))
    assert disabled <= {writer.tdmDescriptorSetOwner(state, name[len("tdm"):-len("Group0+0")])
                        for name in names}


def _halfPlrMaskStates():
    return [dict(state, HalfPLR=1, PrefetchAcrossPersistent=0, enableTDMMetadata=False)
            for state in _acceptedMaskStates((1, 2, 4, 8))]


def _definedTdmWriter(staggerUCode=True):
    return Stub(states=types.SimpleNamespace(subtileIterateModeA=False, subtileIterateModeB=False,
                                             staggerUCode=staggerUCode, unrollIdx=0),
                defineSgpr=lambda reg, num, align=1: RegSet("s", "sgpr" + reg, 0),
                loopCounter=lambda kernel, idx: sgpr("LoopCounterL"))


def _definedTdmSgprs(writer, state):
    return {str(item).split()[1].rstrip(",")[len("sgpr"):] for item in writer.defineTdmSgprs(state).items()}


@pytest.mark.parametrize("state", _halfPlrMaskStates(), ids=_maskStateId)
def test_halfplr_mask_zeroes_exactly_the_increments_it_names(state):
    """graIncrementMask zeroes the own GlobalReadIncs of exactly tdmHalfPlrMaskedIncs's tensors."""
    writer = _definedTdmWriter()
    zeroed = re.findall(r"s_cmov_b32 s\[sgpr(\w+?)\], 0",
                        str(writer.graIncrementMask(state, {"tensorChar": "A"}, {"tensorChar": "B"})))
    own = {name[len("GlobalReadIncs"):] for name in zeroed if name.startswith("GlobalReadIncs")}
    assert own == set(TF.tdmHalfPlrMaskedIncs(state)), zeroed
    # Every shadow increment the mask zeroes is one defineTdmSgprs defines.
    assert {name for name in zeroed if not name.startswith("GlobalReadIncs")} \
        <= _definedTdmSgprs(writer, state), zeroed


@pytest.mark.parametrize("staggerUCode", [True, False])
@pytest.mark.parametrize("state", _halfPlrMaskStates(), ids=_maskStateId)
def test_remove_stagger_reads_the_copy_exactly_where_defined(state, staggerUCode):
    """removeStagger reads a copy exactly for the copies defineTdmSgprs defines; one wave keeps none."""
    writer = _definedTdmWriter(staggerUCode)
    prefix = "tdmRemoveStaggerIncs"
    copies = {name[len(prefix):] for name in _definedTdmSgprs(writer, state) if name.startswith(prefix)}
    reads = {tc for tc in TDM_TENSORS if writer.removeStaggerIncs(state, tc).startswith(prefix)}
    copied = staggerUCode and state["NumWaves"] > 1
    assert copies == reads == (set(TF.tdmHalfPlrMaskedIncs(state)) if copied else set())


@pytest.mark.parametrize("mx", ["mx_a", "mx_b", "mx"])
def test_a_lone_scale_keeps_its_increment_sgpr(mx):
    """A lone TDMFuse=0 scale advances by its own GlobalReadIncs in the loop, so it is not released early."""
    released = []
    counts = types.SimpleNamespace(numSgprGlobalReadIncs=1)
    writer = Stub(states=types.SimpleNamespace(a=counts, b=counts, mxsa=counts, mxsb=counts),
                  addSgprVarToPool=lambda name: released.append(name) or Module(name))
    state = _maskState(0, mx, 4)
    writer.releaseGlobalReadIncsSgprsAfterTdmWaveSep(state)
    scales = {"GlobalReadIncs" + tc for tc in ("MXSA", "MXSB") if TF.tdmMemberIsLive(state, tc)}
    if mx == "mx":
        assert scales <= set(released), released
    else:
        assert not scales & set(released), released


_WAVE_TEST = re.compile(
    r"^\s*(s_cmp_eq_u32|s_cmp_lg_u32|s_cmp_lt_u32|s_cmp_ge_u32|s_bitcmp1_b32)\s+s\[sgprWaveIdx\],\s*(\d+)", re.M)
_WAVE_TEST_SCC = {"s_cmp_eq_u32": lambda w, v: w == v, "s_cmp_lg_u32": lambda w, v: w != v,
                  "s_cmp_lt_u32": lambda w, v: w < v, "s_cmp_ge_u32": lambda w, v: w >= v,
                  "s_bitcmp1_b32": lambda w, v: (w >> v) & 1 == 1}


def _sccFor(module, wave):
    (op, value), = _WAVE_TEST.findall(str(module))
    return _WAVE_TEST_SCC[op](wave, int(value))


_GUARDED = [(state, tc) for state in _acceptedMaskStates() for tc in TDM_TENSORS
            if TF.tdmMemberIsLive(state, tc) and TF.tdmWaveGuard(state, tc)[0] not in ("all", "none")]


@pytest.mark.parametrize("state, tc", _GUARDED, ids=["%s-%s" % (_maskStateId(s), tc) for s, tc in _GUARDED])
def test_wave_select_admits_exactly_the_partition(state, tc):
    """The emitted WaveIdx tests carry exactly tdmWavePartition's waves and skip every other one."""
    waves = TF.tdmWavePartition(state, tc)[1]
    carries, skips = Module("carries"), Module("skips")
    scc1Carries = _tdmEmitWaveCarriesSCC(carries, state, tc)
    scc1Skips = Stub(isTdmWaveIdxLive=lambda kernel: True)._emitTdmWaveSetSkipSCC(skips, state, list(waves), tc)
    for wave in range(state["NumWaves"]):
        assert (_sccFor(carries, wave) == scc1Carries) == (wave in waves), (wave, str(carries))
        assert (_sccFor(skips, wave) == scc1Skips) == (wave not in waves), (wave, str(skips))


# Solution limits of TDMFuse and one-sided MX (messages and removed rules).
# Solution derivations: they need amdclang++ gfx1250 and are skipped otherwise.
_SUBTILE = {"UseSubtileImpl": True, "MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 2, 2]}
_EIGHT_WAVES = {"MatrixInstruction": [16, 16, 128, 1, 1, 2, 4, 2, 4], "WorkGroup": [32, 8, 1]}
_SIXTEEN_WAVES = {"MatrixInstruction": [16, 16, 128, 1, 1, 2, 2, 4, 4], "WorkGroup": [32, 16, 1]}
_SUBTILE_RULE = "is not implemented with UseSubtileImpl on more than one wave"
_WAVE_RULE = "with MX scales on both tensors supports 2 or 4 waves; got NumWaves=%d"
_HEAD = {"mx_a": "MX scales on one tensor only (MXBlockA=32, MXBlockB=0)",
         "mx_b": "MX scales on one tensor only (MXBlockA=0, MXBlockB=32)"}


@pytest.mark.parametrize("overrides", [{}, {"TDMInst": 0, "ProblemType": _NO_MX}],
                         ids=["tdm_on_a_and_b", "tdm_off"])
@pytest.mark.parametrize("fuse", [-1, 4])
def test_solution_rejects_a_tdmfuse_value_that_names_no_grouping(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, overrides):
    """Rejected ahead of every row rule; the layer itself raises for such a value."""
    assert fuse not in validParameters["TDMFuse"]
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, **overrides)
    assert sol.get("Valid") is False
    assert ("TDMFuse=%d names no descriptor grouping; supported values are 0, 1, 2 and 3"
            % fuse) in out, out


# Each state the layer declines, by cause: the TDMFuse values, the overrides and the rule
# that refuses the state before the TDMFuse block could resolve TDMFuse=0's grouping for it.
_LAYER_DECLINES = {
    "tdminst0": ((1, 2, 3), {"TDMInst": 0, "ProblemType": _NO_MX}, "is not implemented with TDMInst=0"),
    "tdminst1": ((1, 2, 3), {"TDMInst": 1}, "TDMA and TDMB must be enabled simultaneously"),
    "tdminst2": ((1, 2, 3), {"TDMInst": 2}, "TDMA and TDMB must be enabled simultaneously"),
    "tdmsplit": ((1, 2, 3), {"TDMSplit": True}, "TDMSplit is currently disabled"),
    "tdmsplit_one_wave": ((1, 2, 3), dict(_ONE_WAVE_64X64, TDMSplit=True),
                          "TDMSplit is currently disabled"),
    "one_wave": ((1, 2, 3), _ONE_WAVE_64X64, "needs more than one wave (NumWaves=1)"),
    "one_wave_no_mx": ((1, 2, 3), dict(_ONE_WAVE_64X64, ProblemType=_NO_MX),
                       "needs more than one wave (NumWaves=1)"),
    "subtile": ((1, 2, 3), _SUBTILE, _SUBTILE_RULE),
    "subtile_no_mx": ((1, 2, 3), dict(_SUBTILE, ProblemType=_NO_MX), _SUBTILE_RULE),
    "waves8": ((2, 3), _EIGHT_WAVES, _WAVE_RULE % 8),
    "waves16": ((2, 3), _SIXTEEN_WAVES, _WAVE_RULE % 16),
}
_DECLINE_CASES = [(name, fuse) for name, (fuses, _, _) in _LAYER_DECLINES.items() for fuse in fuses]


@pytest.mark.parametrize("name, fuse", _DECLINE_CASES, ids=["%s-%d" % case for case in _DECLINE_CASES])
def test_every_layer_decline_is_rejected_by_an_earlier_named_rule(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, name, fuse):
    _, overrides, rule = _LAYER_DECLINES[name]
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, **overrides)
    assert TF.tdmGroupingAccepted(sol._state) is False, name
    assert sol.get("Valid") is False
    assert rule in out, out


@pytest.mark.parametrize("mx", ["mx", "no_mx"])
@pytest.mark.parametrize("fuse", [1, 2, 3])
def test_one_wave_subtile_rows_are_rejected(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, mx):
    """Subtile does not lift the one-wave limit: rows 1-3 are rejected, by it wherever TDMFuse=0 derives."""
    overrides = dict(_ONE_WAVE_64X64, UseSubtileImpl=True, ProblemType=_MX_PROBLEMS[mx])
    sol0, out0 = _derive(gfx1250_iim, assembler, capsys, TDMFuse=0, **overrides)
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, **overrides)
    assert sol.get("Valid") is False, out
    if sol0.get("Valid"):
        assert "TDMFuse=%d needs more than one wave (NumWaves=1)" % fuse in out, (out0, out)


@pytest.mark.parametrize("fuse", [1, 2, 3])
def test_one_sided_subtile_gets_tdmfuse0s_verdict(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse):
    """MX on A with UseSubtileImpl: the one-sided UseSubtileImpl refusal comes before the wave count
    is derived, so rows 1-3 get TDMFuse=0's verdict, that refusal included. The layer declines
    rows 1-3 with UseSubtileImpl on every wave count."""
    overrides = dict(_SUBTILE, ProblemType=_MXSA_ONLY)
    sol0, out0 = _derive(gfx1250_iim, assembler, capsys, TDMFuse=0, **overrides)
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, **overrides)
    assert TF.tdmGroupingAccepted(sol._state) is False
    assert "needs more than one wave" not in out, out
    assert sol.get("Valid") == sol0.get("Valid"), (out0, out)
    refusal = "are not implemented with UseSubtileImpl:"
    assert refusal in out0 and refusal in out, (out0, out)


@pytest.mark.parametrize("mx", ["mx", "mx_a", "mx_b"])
def test_solution_refuses_reuse_across_persistent_with_one_mx_scale_by_name(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, mx):
    """ReuseAcrossPersistent needs a persistent kernel; there one scale is refused by name and
    two scales are not refused for being one-sided."""
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=0, ProblemType=_MX_PROBLEMS[mx],
                       ReuseAcrossPersistent=1, **_STREAMK)
    if mx == "mx":
        assert _ONE_SIDED_MX not in out, out
    else:
        assert sol.get("Valid") is False
        assert _HEAD[mx] in out and "ReuseAcrossPersistent" in out, out


# (TDMFuse, overrides, the whole rejection text) for each kept limit with a reworded message.
_LIMIT_MESSAGES = {
    "r01_tdminst0": (1, {"TDMInst": 0, "ProblemType": _NO_MX},
                     "TDMFuse=1 is not implemented with TDMInst=0: it chooses which TDM descriptor sets "
                     "A, B and their MX scales share, so it needs the TDM on both tensors (TDMInst=3). "
                     "Use TDMInst=3, or TDMFuse=0"),
    "r02_tdmsplit": (0, {"TDMSplit": True},
                     "TDMSplit is currently disabled: it has unresolved read-token and tensorcnt races "
                     "under the decoupled load-vs-compute wave layout. Use TDMSplit=0"),
    "r03_waves8": (2, _EIGHT_WAVES,
                   "TDMFuse=2 with MX scales on both tensors supports 2 or 4 waves; got NumWaves=8: "
                   "the A_MX grouping has a wave layout for its set {A,MXSA,MXSB} only on those counts. "
                   "Use TDMFuse=0 or 1 on other wave counts"),
    "r03_waves16": (3, _SIXTEEN_WAVES,
                    "TDMFuse=3 with MX scales on both tensors supports 2 or 4 waves; got NumWaves=16: "
                    "the B_MX grouping has a wave layout for its set {B,MXSA,MXSB} only on those counts. "
                    "Use TDMFuse=0 or 1 on other wave counts"),
    "r07_r09_subtile_mx": (2, _SUBTILE,
                           "TDMFuse=2 is not implemented with UseSubtileImpl on more than one wave "
                           "(NumWaves=4): the subtile path keeps one TDM descriptor per tensor, so no "
                           "descriptor set can be shared. Use UseSubtileImpl=0, or TDMFuse=0"),
    "r07_r09_subtile_no_mx": (1, dict(_SUBTILE, ProblemType=_NO_MX),
                              "TDMFuse=1 is not implemented with UseSubtileImpl on more than one wave "
                              "(NumWaves=4): the subtile path keeps one TDM descriptor per tensor, so "
                              "no descriptor set can be shared. Use UseSubtileImpl=0, or TDMFuse=0"),
    "r10_onesided_subtile": (0, dict(_SUBTILE, ProblemType=_MXSA_ONLY),
                             _HEAD["mx_a"] + " are not implemented with UseSubtileImpl: the subtile "
                             "scale loads always set up both MXSA and MXSB. Use UseSubtileImpl=0 (the "
                             "gfx1250 TDM path)"),
    "r15_divergent_pair": (2, {"PrefetchGlobalReadA": 1, "PrefetchGlobalReadB": 2},
                           "TDMFuse=2 with MX scales on both tensors is not implemented at a divergent "
                           "decoupled pair (PGRA=1, PGRB=2 -> 1 and 2 LDS blocks) on more than one "
                           "wave: MXSB is in A's descriptor group but follows B's LDS block count. Use "
                           "an equal pair, or TDMFuse=0 or 1"),
    "r18_onesided_divergent_pair": (0, {"ProblemType": _MXSB_ONLY, "PrefetchGlobalReadA": 1,
                                        "PrefetchGlobalReadB": 2},
                                    _HEAD["mx_b"] + " are not implemented at a divergent decoupled "
                                    "pair (PGRA=1, PGRB=2 -> 1 and 2 LDS blocks) on more than one wave "
                                    "(NumWaves=4): a set with one live member rides every wave, while "
                                    "the divergent-pair late fill re-slots a set by wave parity. Use an "
                                    "equal pair"),
    "r24_pap": (0, dict(_STREAMK, ProblemType=_MXSA_ONLY, PrefetchAcrossPersistent=1),
                _HEAD["mx_a"] + " are not implemented with PrefetchAcrossPersistent=1 on a persistent "
                "kernel: the persistent descriptor refresh, the PAP tail reset and the RAP null handle "
                "MX scales only as a pair. Put MX scales on both tensors or on neither, or use "
                "PrefetchAcrossPersistent=0"),
    "r24_rap": (2, dict(_STREAMK, ProblemType=_MXSB_ONLY, ReuseAcrossPersistent=1),
                _HEAD["mx_b"] + " are not implemented with ReuseAcrossPersistent=1 on a persistent "
                "kernel: the persistent descriptor refresh, the PAP tail reset and the RAP null handle "
                "MX scales only as a pair. Put MX scales on both tensors or on neither, or use "
                "ReuseAcrossPersistent=0"),
    "r25_one_wave": (2, _ONE_WAVE_64X64,
                     "TDMFuse=2 needs more than one wave (NumWaves=1): it splits the waves among the "
                     "TDM descriptor sets of A, B and their MX scales, while a one-wave kernel keeps "
                     "one descriptor per tensor. Use TDMFuse=0 on one wave"),
}


@pytest.mark.parametrize("name", sorted(_LIMIT_MESSAGES))
def test_solution_names_each_limit_it_keeps(_gp_gfx1250, gfx1250_iim, assembler, capsys, name):
    """Each kept limit says what is unsupported, why, and what to use instead."""
    fuse, overrides, message = _LIMIT_MESSAGES[name]
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, **overrides)
    assert sol.get("Valid") is False
    assert message in out, out


# Sparse metadata on TDMFuse 1-3.
# tdmMetadataGroup0 belongs to no TDMFuse set: the A/B increment call advances it once per
# iteration on every arm, and the scale call never does.
_METADATA_ADVANCE = ("tdmMetadataGroup0", "GlobalReadIncsMetadata")
_SPARSE_CELLS = [(fuse, mx, numWaves, sparse) for fuse in sorted(TDM_FUSE_GROUPING)
                 for mx in sorted(_MX_STATES) for numWaves in (2, 4) for sparse in (1, 2)]
_SPARSE_IDS = ["tdmfuse%d-%s-w%d-sparse%d" % cell for cell in _SPARSE_CELLS]


def _sparseState(fuse, mx, numWaves, sparse, enableTDMMetadata=True):
    state = _maskState(fuse, mx, numWaves)
    state["ProblemType"]["Sparse"] = sparse
    state["enableTDMMetadata"] = enableTDMMetadata
    return state


def _sparseTensorParams(state):
    """getTensorParameters' isA, isB and is_sparse for A, B and their live scales."""
    sparse = state["ProblemType"]["Sparse"]
    tPA = {"tensorChar": "A", "isA": True, "isB": False, "is_sparse": sparse == 1}
    tPB = {"tensorChar": "B", "isA": False, "isB": True, "is_sparse": sparse == 2}
    for tP, mx in ((tPA, "MXSA"), (tPB, "MXSB")):
        if TF.tdmMemberIsLive(state, mx):
            tP["MX"] = {"tensorChar": mx, "isA": False, "isB": False, "is_sparse": False}
    return tPA, tPB


def _recordIncrements(writer, monkeypatch):
    """Bind the wave-separated increment onto `writer`; return the (set, increment) list it advances."""
    calls = []
    comp = types.SimpleNamespace(
        incrementGlobalAddr=lambda w, group, inc: calls.append((group, inc)) or Module(group))
    monkeypatch.setitem(KernelWriterAssembly.tdmIncrementABWaveSperated.__globals__,
                        "TensorDataMoverLoad", types.SimpleNamespace(find=lambda w: comp))
    for name in ("tdmIncrementABWaveSperated", "_tdmIncrementMetadataWaveSeparated",
                 "_tdmPairedParityOrder"):
        fn = getattr(KernelWriterAssembly, name, None)
        if fn is not None:
            setattr(writer, name, fn.__get__(writer, type(writer)))
    return calls


def _abAndScaleAdvances(writer, calls, state):
    """globalReadIncrementAB's two calls: the A/B pair, then the scale pair when a scale is live."""
    tPA, tPB = _sparseTensorParams(state)
    writer.tdmIncrementABWaveSperated(state, tPA, tPB, 0, 0)
    abCall = list(calls)
    del calls[:]
    if "MX" in tPA or "MX" in tPB:
        writer.tdmIncrementABWaveSperated(state, tPA.get("MX"), tPB.get("MX"), 0, 0)
    return abCall, list(calls)


@pytest.mark.parametrize("fuse, mx, numWaves, sparse", _SPARSE_CELLS, ids=_SPARSE_IDS)
def test_sparse_metadata_advances_once_on_the_ab_call(monkeypatch, fuse, mx, numWaves, sparse):
    """Every arm advances tdmMetadataGroup0 exactly once per iteration, on the A/B call."""
    writer = Stub(states=types.SimpleNamespace(unrollIdx=0, staggerUCode=False))
    calls = _recordIncrements(writer, monkeypatch)
    abCall, scaleCall = _abAndScaleAdvances(writer, calls, _sparseState(fuse, mx, numWaves, sparse))
    assert abCall.count(_METADATA_ADVANCE) == 1, abCall
    assert _METADATA_ADVANCE not in scaleCall, scaleCall


# Shared-scale and every-wave arms advance the metadata only where enableTDMMetadata defines its set.
_GUARDED_METADATA_CELLS = [cell for cell in sorted({c[:3] for c in _SPARSE_CELLS})
                           if TF.tdmSharedScaleSetActive(_sparseState(*cell, 1))
                           or TF.tdmPairOnEveryWave(_sparseState(*cell, 1), "A", "B")]


@pytest.mark.parametrize("fuse, mx, numWaves", _GUARDED_METADATA_CELLS,
                         ids=["tdmfuse%d-%s-w%d" % cell for cell in _GUARDED_METADATA_CELLS])
def test_metadata_in_vgprs_names_no_tdm_metadata_set(monkeypatch, fuse, mx, numWaves):
    """DirectToVgprSparseMetadata defines no tdmMetadataGroup0, so these arms must not advance it."""
    writer = Stub(states=types.SimpleNamespace(unrollIdx=0, staggerUCode=False))
    calls = _recordIncrements(writer, monkeypatch)
    state = _sparseState(fuse, mx, numWaves, 1, enableTDMMetadata=False)
    abCall, scaleCall = _abAndScaleAdvances(writer, calls, state)
    assert not [call for call in abCall + scaleCall if call[0] == "tdmMetadataGroup0"]


_SPARSE_PROBLEM = {"mx": {}, "mx_a": _MXSA_ONLY, "mx_b": _MXSB_ONLY, "no_mx": _NO_MX}


# Sparse B with F4 B (mx) stays out: its global-read width is 2, not the multiple of 4 Sparse B needs,
# and TDMFuse=0 refuses it the same way.
_SPARSE_ACCEPT_CELLS = [(mx, sparse) for mx in sorted(_SPARSE_PROBLEM) for sparse in (1, 2)
                        if (mx, sparse) != ("mx", 2)]


@pytest.mark.parametrize("mx, sparse", _SPARSE_ACCEPT_CELLS)
@pytest.mark.parametrize("fuse", [1, 2, 3])
def test_solution_accepts_sparse_metadata_on_rows_one_to_three(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, mx, sparse):
    """Rows 1-3 take TDM sparse metadata with MX scales on both tensors, on one, or on neither."""
    problemType = dict(_SPARSE_PROBLEM[mx], Sparse=sparse)
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, ProblemType=problemType)
    assert "sparse metadata" not in out, out
    assert sol.get("Valid") is True, out


_SPARSE_FP16 = {"DataType": "h", "DestDataType": "s", "ComputeDataType": "s", "HighPrecisionAccumulate": True}
_SPARSE_FP16_MI = {2: [16, 16, 64, 1, 1, 2, 4, 2, 1], 4: [16, 16, 64, 1, 1, 2, 2, 2, 2]}


def _sparseConfig(fuse, numWaves, mxA, mxB, sparse):
    """_tdmConfig's problem and fork with sparse metadata; without MX scales, FP16 in the shape
    of develop's spmm_tdm_all (MI 16x16x64, DepthU 128)."""
    config = _tdmConfig(fuse, numWaves, mxA, mxB)
    problemType = config["BenchmarkProblems"][0][0]
    problemType["Sparse"] = sparse
    if not mxA and not mxB:
        for key in ("MacDataTypeA", "MacDataTypeB", "DataTypeMXSA", "DataTypeMXSB"):
            problemType.pop(key, None)
        problemType.update(_SPARSE_FP16)
        for entry in config["BenchmarkProblems"][0][1]["ForkParameters"]:
            if "MatrixInstruction" in entry:
                entry["MatrixInstruction"] = [_SPARSE_FP16_MI[numWaves]]
            if "DepthU" in entry:
                entry["DepthU"] = [128]
    return config


@pytest.fixture(scope="module")
def sparseKernel(gfx1250_iim, tmp_path_factory):
    """Derive and emit one kernel per (TDMFuse, NumWaves, MXBlockA, MXBlockB, Sparse), on first use."""
    from config_harness import derive_states, emit_kernels_from_config

    root = tmp_path_factory.mktemp("tdm_fuse_sparse_cells")
    kernels = {}

    def emit(*cell):
        if cell not in kernels:
            path = root / ("tdmfuse%d_w%d_a%d_b%d_sparse%d.yaml" % cell)
            path.write_text(yaml.safe_dump(_sparseConfig(*cell)))
            states = derive_states(path, arch="gfx1250", limit_solutions=1)
            assert len(states) == 1, "%s derives no solution" % (cell,)
            assert (states[0]["TDMFuse"], states[0]["NumWaves"]) == cell[:2], cell
            assert states[0]["enableTDMMetadata"], cell
            results = emit_kernels_from_config(path, limit=1, arch="gfx1250")
            assert len(results) == 1, cell
            base, source, err = results[0]
            assert err == 0, (cell, base)
            kernels[cell] = source
        return kernels[cell]

    return emit


def _mainLoopMetadata(source):
    """Main-loop (metadata address advances, metadata issues, odd-wave skips)."""
    k = _AsmKernel(source)
    begin, end = k.mainLoop()
    # The increment reaches the add through the StaggerU wrap select, so match the self-add.
    advances = [ops for _, ops in k.instructions[begin:end]
                if tuple(ops[:2]) == ("s[sgprtdmMetadataGroup0+2]", "s[sgprtdmMetadataGroup0+2]")]
    issues = k.tdmIssues(begin, end).count(k.setIndex("tdmMetadataGroup0"))
    skips = [name for name, pos in k.labels.items()
             if "TdmMetadataSkipOddWave" in name and begin <= pos < end]
    return len(advances), issues, len(skips)


# With MX scales the TDMFuse=0 reference itself overflows SGPRs (err -2, develop included), so the
# kernel comparison runs without scales; the MX states are covered at the writer level above.
@pytest.mark.parametrize("sparse", [1, 2])
@pytest.mark.parametrize("numWaves", [2, 4])
@pytest.mark.parametrize("mx", ["no_mx"])
@pytest.mark.parametrize("fuse", [1, 2, 3])
def test_sparse_rows_load_and_advance_the_metadata_as_tdmfuse0_does(
        gfx1250_iim, sparseKernel, fuse, mx, numWaves, sparse):
    """Each main-loop iteration advances tdmMetadataGroup0 once and loads it on even waves only."""
    mxA, mxB = _MX_STATES[mx]
    reference = _mainLoopMetadata(sparseKernel(0, numWaves, mxA, mxB, sparse))
    assert min(reference) >= 1, reference
    assert _mainLoopMetadata(sparseKernel(fuse, numWaves, mxA, mxB, sparse)) == reference


# Persistent and StreamK on TDMFuse 1-3, without MX scales or with MX on one tensor.
# ---------------------------------------------------------------------------
# A persistent kernel without stagger code (on gfx1250 TDM: an enabled ClusterDim) makes a
# TLU=False tensor's GlobalReadIncs an assembler constant, except where A and B ride separate
# TDM sets, which read it as an SGPR. StreamK starts every live set at StreamKLocalStart times
# the increment it advances by; DataParallel starts each tile at K=0. The writer-stub tests
# need rocisa; the kernel cells also need amdclang++ gfx1250 and are skipped otherwise.
# ---------------------------------------------------------------------------
_STREAMK_SPLIT = {"StreamK": 3, "StreamKForceDPOnly": 0}
_POLICY_FORKS = {
    "plain": {},
    "dp": {"TileProcessingStrategy": "DataParallel", "WorkAssignment": "StaticGrid"},
    "streamk": {"TileProcessingStrategy": "StreamK", "WorkAssignment": "StaticGrid"},
}
_K_OFFSET_MUL = re.compile(r"s_mul_i32\s+(s\[?\d+\]?),\s*s\[sgprStreamKLocalStart\],\s*s\[sgpr(\w+)\]")
_K_OFFSET_ADD = re.compile(r"s_add_u32\s+s\[sgpr(tdm\w+Group0)\+2\],\s*s\[sgpr\1\+2\],\s*(s\[?\d+\]?)")
_ASM_NUMERIC_SGPRS = re.compile(r"^s(?:\[(\d+)(?::(\d+))?\]|(\d+))$")


def _policyState(fuse, mx, numWaves, policy):
    return dict(_maskState(fuse, mx, numWaves), enableTDMA=True, enableTDMB=True,
                **_POLICY_FORKS[policy])


def _sgprNumber(operand):
    return int(operand.strip("s[]"))


def _kOffsetProducts(text):
    """{(Group0 name, increment)}: each StreamKLocalStart product added to a TDM set's address.

    A product counts only while its register still holds it. Any other instruction that writes
    the register ends that, so a later add through the same temporary (the next tile's global
    offset, for one) is not taken for a K-offset.
    """
    latest, pairs = {}, set()
    for line in _ASM_BLOCK_COMMENT.sub("", text).splitlines():
        code = line.split("//", 1)[0].strip()
        mul = _K_OFFSET_MUL.search(code)
        if mul:
            latest[_sgprNumber(mul.group(1))] = mul.group(2)
            continue
        add = _K_OFFSET_ADD.search(code)
        if add and _sgprNumber(add.group(2)) in latest:
            pairs.add((add.group(1), latest[_sgprNumber(add.group(2))]))
        mnemonic, _, rest = code.partition(" ")
        written = _ASM_NUMERIC_SGPRS.match(rest.split(",", 1)[0].strip())
        if written and not mnemonic.startswith(_ASM_NO_DESTINATION):
            first = int(written.group(1) or written.group(3))
            for index in range(first, int(written.group(2) or first) + 1):
                latest.pop(index, None)
    return pairs


class _TmpSgprs:
    """allocTmpSgpr's context manager on a writer stub."""

    def __init__(self, num, tag=None, **kwargs):
        self.idx = 100

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _kOffsetText(writerClass, state, tcA, tcB):
    """tdmApplyTileKOffsetWaveSeparated's emission on a writer stub; None is an absent scale."""
    writer = Stub(allocTmpSgpr=_TmpSgprs)
    for name in ("tdmApplyTileKOffsetWaveSeparated", "_tdmPairedParityOrder"):
        setattr(writer, name, getattr(writerClass, name).__get__(writer, Stub))
    tP = lambda tc: None if tc is None else {"tensorChar": tc}
    return str(writer.tdmApplyTileKOffsetWaveSeparated(state, tP(tcA), tP(tcB)))


def _scalePair(scale):
    return (scale, None) if scale == "MXSA" else (None, scale)


@pytest.mark.parametrize("state", [dict(s, enableTDMA=True, enableTDMB=True)
                                   for s in _acceptedMaskStates((1, 2, 4, 8))], ids=_maskStateId)
def test_separate_ab_sets_are_exactly_the_raw_increment_readers(state):
    """The writer keeps GlobalReadIncsA/B in SGPRs for a persistent kernel exactly where
    tdmSeparateABDescriptors holds; those are the arrangements whose readers take no constant."""
    readsRaw = TF.tdmWaveSeparated(state) and (TF.tdmPairOnEveryWave(state, "A", "B")
                                               or TF.tdmSharedScaleSetActive(state)
                                               or tdmFusePaired(state))
    assert TF.tdmSeparateABDescriptors(state) == bool(readsRaw)


@pytest.mark.parametrize("numWaves", [2, 4])
@pytest.mark.parametrize("fuse", [1, 2, 3])
def test_streamk_moves_each_set_of_an_every_wave_pair_by_its_own_increment(fuse, numWaves):
    """Without MX, TDMFuse 1-3 give A and B a set each on every wave and define no tdmABIncs."""
    text = _kOffsetText(KernelWriterAssembly, _policyState(fuse, "no_mx", numWaves, "streamk"), "A", "B")
    assert _kOffsetProducts(text) == {("tdmAGroup0", "GlobalReadIncsA"),
                                      ("tdmBGroup0", "GlobalReadIncsB")}, text
    assert "tdmABIncs" not in text, text


@pytest.mark.parametrize("numWaves", [2, 4])
@pytest.mark.parametrize("scale", ["MXSA", "MXSB"])
def test_streamk_moves_a_lone_tdmfuse0_scale_by_its_own_increment(scale, numWaves):
    """TDMFuse=0's lone scale advances by its own increment on a set of its own, so it must start
    at StreamKLocalStart like A and B; the scale call passes None for the absent scale."""
    mx = "mx_a" if scale == "MXSA" else "mx_b"
    text = _kOffsetText(KernelWriterAssembly, _policyState(0, mx, numWaves, "streamk"), *_scalePair(scale))
    assert _kOffsetProducts(text) == {("tdm%sGroup0" % scale, "GlobalReadIncs" + scale)}, text


@pytest.mark.parametrize("numWaves", [2, 4])
@pytest.mark.parametrize("scale", ["MXSA", "MXSB"])
@pytest.mark.parametrize("fuse", [1, 2, 3])
def test_streamk_moves_a_lone_scale_on_a_shared_set_only_on_the_ab_call(fuse, scale, numWaves):
    """On TDMFuse 1-3 a lone scale rides its data tensor's set: the A/B call moves that set by
    tdmABIncs, which holds the scale's increment on the scale's waves, and the other data set by
    its own increment; the scale call adds nothing."""
    state = _policyState(fuse, "mx_a" if scale == "MXSA" else "mx_b", numWaves, "streamk")
    owner = _SCALE_SET_OWNER[fuse, scale]
    other = "B" if owner == "A" else "A"
    ab = _kOffsetText(KernelWriterAssembly, state, "A", "B")
    assert _kOffsetProducts(ab) == {("tdm%sGroup0" % owner, "tdmABIncs"),
                                    ("tdm%sGroup0" % other, "GlobalReadIncs" + other)}, ab
    scaleCall = _kOffsetText(KernelWriterAssembly, state, *_scalePair(scale))
    assert "StreamKLocalStart" not in scaleCall, scaleCall


@pytest.mark.parametrize("fuse, owner, other", [(2, "A", "B"), (3, "B", "A")])
def test_streamk_moves_every_set_of_the_two_wave_both_scale_rows(fuse, owner, other):
    """Wave 1 carries MXSA on the owner's set, where tdmABIncs holds MXSA's increment, and MXSB on
    a set of its own, which the scale call moves by MXSB's increment."""
    state = _policyState(fuse, "mx", 2, "streamk")
    ab = _kOffsetText(KernelWriterAssembly, state, "A", "B")
    scales = _kOffsetText(KernelWriterAssembly, state, "MXSA", "MXSB")
    assert _kOffsetProducts(ab) == {("tdm%sGroup0" % owner, "tdmABIncs"),
                                    ("tdm%sGroup0" % other, "GlobalReadIncs" + other)}, ab
    assert _kOffsetProducts(scales) == {("tdmMXSBGroup0", "GlobalReadIncsMXSB")}, scales


@pytest.mark.parametrize("fuse, mx, tcA, tcB", [
    (1, "no_mx", "A", "B"), (2, "mx", "MXSA", "MXSB"), (0, "mx_a", "MXSA", None), (0, "mx_b", None, "MXSB"),
    (2, "mx_a", "MXSA", None),
], ids=["every_wave_pair", "own_scale_set", "lone_mxsa", "lone_mxsb", "shared_lone_scale"])
def test_data_parallel_tiles_take_no_k_offset(fuse, mx, tcA, tcB):
    """DataParallel allocates no StreamKLocalStart; every tile starts at K=0."""
    text = _kOffsetText(KernelWriterAssembly, _policyState(fuse, mx, 2, "dp"), tcA, tcB)
    assert "StreamKLocalStart" not in text, text


_POLICY_KERNELS = {}


def _policyEmit(gfx1250_iim, tmp_path_factory, fuse, numWaves, mxA, mxB, policy, cluster=(1, 1)):
    """Derive and emit _tdmConfig's cell under an execution policy and a ClusterDim: (base, source, err)."""
    from config_harness import derive_states, emit_kernels_from_config

    cell = (fuse, numWaves, mxA, mxB, policy, tuple(cluster))
    config = _tdmConfig(fuse, numWaves, mxA, mxB)
    fork = config["BenchmarkProblems"][0][1]["ForkParameters"]
    fork += [{key: [value]} for key, value in _POLICY_FORKS[policy].items()]
    fork.append({"ClusterDim": [list(cluster)]})
    path = tmp_path_factory.mktemp("tdm_fuse_policy") / "cell.yaml"
    path.write_text(yaml.safe_dump(config))
    states = derive_states(path, arch="gfx1250", limit_solutions=1)
    assert len(states) == 1, "%s derives no solution" % (cell,)
    assert (states[0]["TDMFuse"], states[0]["NumWaves"]) == (fuse, numWaves), cell
    results = emit_kernels_from_config(path, limit=1, arch="gfx1250")
    assert len(results) == 1, cell
    return results[0]


def _policyKernel(gfx1250_iim, tmp_path_factory, fuse, numWaves, mxA, mxB, policy, cluster=(1, 1)):
    """_policyEmit's kernel, emitted once per cell; the emitter must succeed."""
    cell = (fuse, numWaves, mxA, mxB, policy, tuple(cluster))
    if cell not in _POLICY_KERNELS:
        base, source, err = _policyEmit(gfx1250_iim, tmp_path_factory, fuse, numWaves, mxA, mxB,
                                        policy, cluster)
        assert err == 0, (cell, base)
        _POLICY_KERNELS[cell] = _AsmKernel(source)
    return _POLICY_KERNELS[cell]


# Both scales on two waves overflow the SGPR budget under StreamK on TDMFuse 2/3. Codegen rejects the kernel
# (error 2, nothing emitted); no Solution rule refuses it.
_STREAMK_SGPR_OVERFLOW = {(2, 2, 32, 32), (3, 2, 32, 32)}


def _assertSgprOverflow(capsys, emitted):
    """Codegen's SGPR-overflow rejection: error 2 and no kernel source."""
    _, source, err = emitted
    out = capsys.readouterr().out
    assert err == -2 and not source, (err, len(source or ""))
    assert "Generating kernel source resulted in error 2" in out, out[-2000:]
    assert "overflowed max SGPRS" in out, out[-2000:]


def _sgprNamesUsed(kernel, *prefixes):
    pattern = re.compile(r"\bsgpr((?:%s)\w*)" % "|".join(prefixes))
    return {name for _, operands in kernel.instructions for op in operands for name in pattern.findall(op)}


# Every arrangement with separate A and B sets under a cluster, which drops stagger code:
# DataParallel with [2, 4]; StreamK takes a cluster only without a Y-extent, so [2, 1].
_CLUSTER_CELLS = [(fuse, 2, 0, 0, "dp", (2, 4)) for fuse in (1, 2, 3)] \
    + [(1, 2, 0, 32, "dp", (2, 4)), (2, 2, 32, 0, "dp", (2, 4)), (3, 2, 0, 32, "dp", (2, 4))] \
    + [(fuse, 2, 32, 32, "dp", (2, 4)) for fuse in (1, 2, 3)] + [(2, 4, 32, 32, "dp", (2, 4))] \
    + [(fuse, 2, 0, 0, "streamk", (2, 1)) for fuse in (1, 2, 3)] \
    + [(2, 2, 32, 0, "streamk", (2, 1)), (2, 2, 32, 32, "streamk", (2, 1)), (3, 2, 32, 32, "streamk", (2, 1))]


@pytest.mark.parametrize("fuse, numWaves, mxA, mxB, policy, cluster", _CLUSTER_CELLS,
                         ids=["tdmfuse%d-w%d-a%d-b%d-%s-c%dx%d" % (c[:5] + c[5]) for c in _CLUSTER_CELLS])
def test_persistent_cluster_kernels_read_only_defined_increment_sgprs(
        gfx1250_iim, tmp_path_factory, fuse, numWaves, mxA, mxB, policy, cluster):
    """Separate A and B sets keep GlobalReadIncsA/B as SGPRs on a persistent cluster kernel,
    without MX scales, with one scale and with MX scales on both tensors."""
    k = _policyKernel(gfx1250_iim, tmp_path_factory, fuse, numWaves, mxA, mxB, policy, cluster)
    used = _sgprNamesUsed(k, "GlobalReadIncs", "tdm", "StreamKLocal")
    assert {"GlobalReadIncsA", "GlobalReadIncsB"} <= used, sorted(used)
    assert not {name for name in used if "sgpr" + name not in k.symbols}, sorted(used)


def _expectedKOffsets(fuse, mxA, mxB):
    """{(tensor whose Group0 names the set, increment)}: one StreamK K-offset per live set."""
    scales = [tc for tc, mx in (("MXSA", mxA), ("MXSB", mxB)) if mx]
    if fuse == 0:
        if not scales:
            return {("A", "tdmABIncs")}
        inc = "tdmMXSAMXSBIncs" if len(scales) == 2 else "GlobalReadIncs" + scales[0]
        return {("A", "tdmABIncs"), (scales[0], inc)}
    if not scales:
        return {("A", "GlobalReadIncsA"), ("B", "GlobalReadIncsB")}
    owner = _SCALE_SET_OWNER[fuse, scales[0]]
    other = "B" if owner == "A" else "A"
    expected = {(owner, "tdmABIncs"), (other, "GlobalReadIncs" + other)}
    if len(scales) == 2:
        expected.add(("MXSB", "GlobalReadIncsMXSB"))
    return expected


_STREAMK_CELLS = [(fuse, numWaves, 0, 0) for fuse in (1, 2, 3) for numWaves in (2, 4)] \
    + [(0, numWaves, mxA, mxB) for numWaves in (2, 4) for mxA, mxB in ((32, 0), (0, 32))] \
    + [(fuse, 2, mxA, mxB) for fuse in (1, 2, 3) for mxA, mxB in ((32, 0), (0, 32))] \
    + [(2, 2, 32, 32), (3, 2, 32, 32), (0, 2, 32, 32)]


@pytest.mark.parametrize("fuse, numWaves, mxA, mxB", _STREAMK_CELLS,
                         ids=["tdmfuse%d-w%d-a%d-b%d" % cell for cell in _STREAMK_CELLS])
def test_streamk_kernels_move_every_live_set_to_the_tile_start(
        gfx1250_iim, tmp_path_factory, capsys, fuse, numWaves, mxA, mxB):
    """Each live descriptor set takes StreamKLocalStart times the increment it advances by."""
    if (fuse, numWaves, mxA, mxB) in _STREAMK_SGPR_OVERFLOW:
        _assertSgprOverflow(capsys, _policyEmit(gfx1250_iim, tmp_path_factory, fuse, numWaves, mxA, mxB,
                                                "streamk"))
        return
    k = _policyKernel(gfx1250_iim, tmp_path_factory, fuse, numWaves, mxA, mxB, "streamk")
    found = {(k.setIndex(group), inc) for group, inc in _kOffsetProducts(k.source)}
    expected = {(k.setIndex("tdm%sGroup0" % tc), inc) for tc, inc in _expectedKOffsets(fuse, mxA, mxB)}
    assert None not in {index for index, _ in expected}, expected
    assert found == expected, sorted(_kOffsetProducts(k.source))


@pytest.mark.parametrize("policy", ["dp", "streamk"])
@pytest.mark.parametrize("fuse", [2, 3])
def test_each_persistent_tile_rebuilds_mxsb_disabled_on_wave_0(gfx1250_iim, tmp_path_factory, capsys,
                                                               fuse, policy):
    """The per-tile descriptor rebuild runs inside the persistent loop, so wave 0 holds MXSB's stash
    disabled on every tile, not only the first."""
    if policy == "streamk" and (fuse, 2, 32, 32) in _STREAMK_SGPR_OVERFLOW:
        _assertSgprOverflow(capsys, _policyEmit(gfx1250_iim, tmp_path_factory, fuse, 2, 32, 32, policy))
        return
    k = _policyKernel(gfx1250_iim, tmp_path_factory, fuse, 2, 32, 32, policy)
    starts = [index for name, index in k.labels.items() if "PersistentLoopStart" in name]
    begin, _ = k.mainLoop()
    disables = [index for index, (mnemonic, operands) in enumerate(k.instructions)
                if mnemonic == "s_cmov_b32" and operands[:2] == ["s[sgprtdmMXSBGroup0+0]", "0"]]
    assert starts and disables, (starts, disables)
    assert any(min(starts) <= index < begin for index in disables), (starts, disables, begin)


@pytest.mark.parametrize("policy", [_STREAMK, _STREAMK_SPLIT], ids=["persistent_dp", "streamk"])
@pytest.mark.parametrize("mx", sorted(_ONE_SCALE))
@pytest.mark.parametrize("fuse", [0, 1, 2, 3])
def test_solution_accepts_one_mx_scale_with_streamk_and_data_parallel(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, mx, policy):
    """One-sided MX takes StreamK and data-parallel persistent kernels."""
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse,
                       ProblemType=_ONE_SCALE[mx], **policy)
    assert sol.get("Valid") is True, out
    assert _ONE_SIDED_MX not in out, out


@pytest.mark.parametrize("policy", [_STREAMK, _STREAMK_SPLIT], ids=["persistent_dp", "streamk"])
@pytest.mark.parametrize("fuse", [1, 2, 3])
def test_solution_accepts_rows_one_to_three_without_scales_with_streamk_and_data_parallel(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, policy):
    """TDMFuse 1-3 without MX take StreamK and data-parallel persistent kernels."""
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, ProblemType=_NO_MX, **policy)
    assert sol.get("Valid") is True, out
    assert sol._state["TDMFuse"] == fuse


@pytest.mark.parametrize("policy", [_STREAMK, _STREAMK_SPLIT], ids=["persistent_dp", "streamk"])
@pytest.mark.parametrize("fuse", [2, 3])
def test_solution_accepts_both_scales_on_two_waves_with_streamk_and_data_parallel(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, policy):
    """The two-wave both-scale rows stash MXSB under StreamK and data-parallel kernels."""
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse, **_TWO_WAVES, **policy)
    assert sol.get("Valid") is True, out
    assert TF.tdmStashedMembers(sol._state) == ("MXSB",)


@pytest.mark.parametrize("mx, missing", [("mx_a", "DirectToLdsMXSB"), ("mx_b", "DirectToLdsMXSA")])
def test_one_wave_one_sided_mx_raises_in_the_directtolds_check(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, mx, missing):
    """One-sided MX is handled on more than one wave only. On one wave the DirectToLds check
    compares both pairs, so it reads the key of the scale the problem does not carry, which
    is never set: B's pair after A's equal one with MX on A, A's pair first with MX on B."""
    with pytest.raises(KeyError, match=missing):
        _derive(gfx1250_iim, assembler, capsys, TDMFuse=0, ProblemType=_ONE_SCALE[mx], **_WAVES[1])


@pytest.mark.parametrize("policy", [{}, _STREAMK_SPLIT], ids=["default", "streamk"])
@pytest.mark.parametrize("mx", sorted(_ONE_SCALE))
@pytest.mark.parametrize("fuse", [1, 2, 3])
def test_one_wave_rejects_rows_one_to_three_with_one_mx_scale(
        _gp_gfx1250, gfx1250_iim, assembler, capsys, fuse, mx, policy):
    """The one-wave rule rejects TDMFuse 1-3 before any one-sided MX limit or the DirectToLds check."""
    sol, out = _derive(gfx1250_iim, assembler, capsys, TDMFuse=fuse,
                       ProblemType=_ONE_SCALE[mx], **_WAVES[1], **policy)
    assert sol.get("Valid") is False
    assert _ONE_SIDED_MX not in out, out
    assert "TDMFuse=%d needs more than one wave (NumWaves=1)" % fuse in out, out
