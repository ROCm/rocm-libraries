################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for the `Solution` derivation statics (slice 3): the
GRVW / tile-dim setters, the WaveSeparateGlobalRead check, the
VgprForLocalReadPacking / DirectToVgpr / DirectToLds doability predicates, the
MI-output-info helper, and the two big `assign*DerivedParameters` entry points.

Small statics are driven with crafted minimal states (one mutation per reject
branch); the larger predicates and the `assign*` methods are seeded from a real
fully-derived solution state. The exhaustive reject matrix of the two giant
`assign*` methods (a full config matrix across dtype/ISA/MI) is a further
increment — see resistance.md.
"""

import copy
import importlib

import pytest

from Tensile.Common.DataType import DataType

S = importlib.import_module("Tensile.SolutionStructs.Solution")
Solution = S.Solution

pytestmark = pytest.mark.unit


# ===========================================================================
# setGlobalReadVectorWidth
# ===========================================================================

def _grvw_state():
    return {"UseSubtileImpl": False, "NumThreads": 256}


def test_set_grvw_valid(snapshot):
    state = _grvw_state()
    rv = Solution.setGlobalReadVectorWidth(state, "A", totalVectors=512, grvw=4, printRejectionReason=False)
    assert {"rv": rv, "grvw": state["GlobalReadVectorWidthA"], "numloads": state["NumLoadsA"]} == snapshot


def test_set_grvw_invalid_grvw_non_subtile(snapshot):
    state = _grvw_state()
    rv = Solution.setGlobalReadVectorWidth(state, "A", totalVectors=512, grvw=3, printRejectionReason=False)
    assert {"rv": rv, "grvw": state["GlobalReadVectorWidthA"], "numloads": state["NumLoadsA"]} == snapshot


def test_set_grvw_invalid_grvw_subtile_bypass(snapshot):
    state = {"UseSubtileImpl": True, "NumThreads": 256}
    rv = Solution.setGlobalReadVectorWidth(state, "B", totalVectors=512, grvw=3, printRejectionReason=False)
    assert {"rv": rv, "grvw": state["GlobalReadVectorWidthB"]} == snapshot


def test_set_grvw_totalvectors_not_multiple(snapshot):
    state = _grvw_state()
    rv = Solution.setGlobalReadVectorWidth(state, "A", totalVectors=500, grvw=4, printRejectionReason=False)
    assert {"rv": rv} == snapshot


# ===========================================================================
# checkAndAssignWaveSeparateGlobalRead
# ===========================================================================

def _wsgr_state(**over):
    state = {
        "NumThreads": 256, "WavefrontSize": 64, "Valid": True,
        "WaveSeparateGlobalReadA": 1,
        "_DepthUA": 8, "MacroTileA": 128,
        "ProblemType": {"TLUA": True},
    }
    state.update(over)
    return state


def test_wsgr_off_is_noop():
    state = _wsgr_state(WaveSeparateGlobalReadA=0)
    Solution.checkAndAssignWaveSeparateGlobalRead(state, "A", False)
    assert state["Valid"] is True


def test_wsgr_tlu_valid():
    state = _wsgr_state()  # TLU, _DepthUA=8, numWaves=4 -> 8%4==0 -> ok
    Solution.checkAndAssignWaveSeparateGlobalRead(state, "A", False)
    assert state["Valid"] is True


def test_wsgr_tlu_depthu_not_multiple_rejects():
    state = _wsgr_state(_DepthUA=6)  # 6 % 4 != 0
    Solution.checkAndAssignWaveSeparateGlobalRead(state, "A", False)
    assert state["Valid"] is False


def test_wsgr_not_tlu_macrotile_not_multiple_rejects():
    state = _wsgr_state(ProblemType={"TLUA": False}, MacroTileA=130)  # 130 % 4 != 0
    Solution.checkAndAssignWaveSeparateGlobalRead(state, "A", False)
    assert state["Valid"] is False


def test_wsgr_not_tlu_macrotile_multiple_ok():
    state = _wsgr_state(ProblemType={"TLUA": False}, MacroTileA=128)  # 128 % 4 == 0
    Solution.checkAndAssignWaveSeparateGlobalRead(state, "A", False)
    assert state["Valid"] is True


# ===========================================================================
# getMIOutputInfo — dtype + cap branches (real isaInfoMap)
# ===========================================================================

def test_mi_output_info_real_isa(real_state, isa_info_map, snapshot):
    assert Solution.getMIOutputInfo(real_state, isa_info_map) == snapshot


def test_mi_output_info_f64(real_state, isa_info_map, snapshot):
    # f64 MIOutputTypeNameAbbrev -> (1, 2) on an MFMA isa.
    real_state["ProblemType"]["DataType"] = DataType("d")
    assert Solution.getMIOutputInfo(real_state, isa_info_map) == snapshot


# ===========================================================================
# isVgprForLocalReadPackingDoable
# ===========================================================================

def test_vgpr_lrpacking_disabled_not_mi(real_state, isa_info_map):
    real_state["EnableMatrixInstruction"] = False
    assert Solution.isVgprForLocalReadPackingDoable(real_state, isa_info_map) is False


def test_vgpr_lrpacking_disabled_low_plr(real_state, isa_info_map):
    real_state["PrefetchLocalRead"] = 0
    real_state["DirectToVgprA"] = False
    real_state["DirectToVgprB"] = False
    assert Solution.isVgprForLocalReadPackingDoable(real_state, isa_info_map) is False


def test_vgpr_lrpacking_disabled_wide_dtype(real_state, isa_info_map):
    # numRegisters >= 1 (single) -> not doable.
    real_state["ProblemType"]["DataType"] = DataType("s")
    assert Solution.isVgprForLocalReadPackingDoable(real_state, isa_info_map) is False


def test_vgpr_lrpacking_result_for_real_state(real_state, isa_info_map, snapshot):
    # The unmutated real state (gfx942, half) -> pin whatever it yields.
    assert Solution.isVgprForLocalReadPackingDoable(real_state, isa_info_map) == snapshot


# ===========================================================================
# isDirectToVgprDoable / isDirectToLdsDoable — early/reachable branches from
# the real state. (The full reject matrix needs a DTV/DTL-passing base config
# across many keys; that config-matrix sweep is a further increment — see
# resistance.md.)
# ===========================================================================

@pytest.mark.parametrize("tc", ["A", "B"], ids=["A", "B"])
def test_dtv_doable_real_state(real_state, isa_info_map, tc, snapshot):
    # Pin the outcome for the real state (object-free: just the bool).
    result = Solution.isDirectToVgprDoable(real_state, tc, False, isa_info_map)
    assert {"result": result} == snapshot


def test_dtv_not_matrix_instruction_rejects(real_state, isa_info_map):
    real_state["EnableMatrixInstruction"] = False
    assert Solution.isDirectToVgprDoable(real_state, "A", False, isa_info_map) is False


@pytest.mark.parametrize("tc", ["A", "B"], ids=["A", "B"])
def test_dtl_doable_real_state(real_state, isa_info_map, tc, snapshot):
    result = Solution.isDirectToLdsDoable(real_state, tc, isa_info_map, False)
    assert {"result": result} == snapshot


def test_dtl_subtile_impl_short_circuits(real_state, isa_info_map):
    real_state["UseSubtileImpl"] = True
    assert Solution.isDirectToLdsDoable(real_state, "A", isa_info_map, False) is True


def test_dtl_not_matrix_instruction_rejects(real_state, isa_info_map):
    # Reach the "MatrixInstruction only" reject: keep numBytesPerLoad >= 4,
    # MT power-of-2, then disable EnableMatrixInstruction.
    real_state["EnableMatrixInstruction"] = False
    out = Solution.isDirectToLdsDoable(real_state, "A", isa_info_map, False)
    assert out is False


# ===========================================================================
# assignProblemIndependentDerivedParameters / assignDerivedParameters
# (happy re-run on the real state + a reachable reject variant)
# ===========================================================================

def test_assign_problem_independent_early_return(real_state, isa_info_map):
    # AssignedProblemIndependentDerivedParameters is True on a derived state ->
    # the guard returns immediately, leaving the state unchanged.
    real_state["AssignedProblemIndependentDerivedParameters"] = True
    before = dict(real_state)
    Solution.assignProblemIndependentDerivedParameters(real_state, False, isa_info_map)
    assert real_state == before


def test_assign_problem_independent_rerun(real_state, isa_info_map, snapshot):
    # Reset the flag and re-run the full problem-independent derivation on a
    # self-consistent state; pin a few derived scalars + Valid.
    real_state["AssignedProblemIndependentDerivedParameters"] = False
    Solution.assignProblemIndependentDerivedParameters(real_state, False, isa_info_map)
    keys = ["Valid", "NumThreads", "MacroTile0", "MacroTile1", "ThreadTile0",
            "ThreadTile1", "SubGroup0", "SubGroup1", "_ScheduleIterAlg"]
    assert {k: real_state.get(k) for k in keys} == snapshot


def test_assign_problem_independent_sia4_no_backend_rejects(real_state, isa_info_map):
    # ScheduleIterAlg=4 requires the StinkyTofu backend; on a build without it
    # (or an unsupported ISA) the derivation rejects. Pin that it does not stay
    # Valid (covers the SIA=4 guard block).
    import rocisa
    real_state["AssignedProblemIndependentDerivedParameters"] = False
    real_state["ScheduleIterAlg"] = 4
    Solution.assignProblemIndependentDerivedParameters(real_state, False, isa_info_map)
    if not (rocisa.hasStinkyTofuBackend() and rocisa.isSupportedByStinkyTofu(real_state["ISA"])):
        assert real_state["Valid"] is False
    else:
        assert "_StinkyTofuOptLevel" in real_state


def test_assign_derived_parameters_rerun(real_state, isa_info_map, assembler, snapshot):
    # Reset both derivation flags and re-run the full derivation on a
    # self-consistent state; pin a few derived scalars + Valid. Signature:
    # (state, splitGSU, printRejectionReason, printIndexAssignmentInfo,
    #  isaInfoMap, rocmVersion).
    real_state["AssignedDerivedParameters"] = False
    real_state["AssignedProblemIndependentDerivedParameters"] = False
    Solution.assignDerivedParameters(
        real_state, False, False, False, isa_info_map, assembler.rocm_version
    )
    keys = ["Valid", "AssignedDerivedParameters", "WavefrontSize", "MaxLDS",
            "EnableF32XdlMathOp", "UseF32XEmulation", "NumThreads",
            "MacroTile0", "MacroTile1", "DepthU"]
    assert {k: real_state.get(k) for k in keys} == snapshot


@pytest.mark.parametrize("grvw", [16, 32])
def test_set_grvw_accepts_wide_members(grvw):
    state = _grvw_state()
    assert Solution.setGlobalReadVectorWidth(
        state, "A", totalVectors=512, grvw=grvw, printRejectionReason=False
    ) is True
    assert state["GlobalReadVectorWidthA"] == grvw


@pytest.mark.parametrize(
    "tc,dtype,sparse,expected",
    [
        ("A", "Float4", 0, 2),
        ("A", "Float6", 0, 4),
        ("A", "Half", 1, 4),
        ("A", "Half", 0, 1),
        ("B", "Half", 2, 4),
        ("B", "Half", 1, 1),
    ],
)
def test_set_grvw_tdm_dtype_and_sparse_rules(tc, dtype, sparse, expected):
    state = {
        "UseSubtileImpl": False,
        "TDMInst": 3,
        "NumThreads": 256,
        "ProblemType": {"DataType" + tc: DataType(dtype), "Sparse": sparse},
    }
    assert Solution.setGlobalReadVectorWidth(
        state, tc, totalVectors=512, grvw=4, printRejectionReason=False
    ) is True
    assert state["GlobalReadVectorWidth" + tc] == expected


def _classic_tile_state(tlu, coalesced=2, macro_tile=127):
    return {
        "WaveSeparateGlobalReadA": 0,
        "NumThreads": 256,
        "WavefrontSize": 64,
        "enableGLTrA": 0,
        "NumLoadsCoalescedA": coalesced,
        "DirectToVgprA": 0,
        "ProblemType": {"TLUA": tlu},
        "MIWaveTileA": 1,
        "GlobalReadVectorWidthA": 4,
        "MatrixInstK": 1,
        "LocalSplitU": 1,
        "MIInputPerThread": 1,
        "NumLoadsA": 4,
        "UseGeneralizedNLCOneA": False,
        "MacroTileA": macro_tile,
    }


@pytest.mark.parametrize(
    "tlu,expected",
    [
        (True, (63, 9)),
        (False, (9, 63)),
    ],
)
def test_set_global_load_tile_dim_classic_orients_axes(tlu, expected):
    state = _classic_tile_state(tlu)
    assert Solution.setGlobalLoadTileDimClassic(
        state, "A", 0, 8, 8, 17, False
    ) is True
    assert (state["LSCA"], state["LSPA"]) == expected
    assert state["NumLoadsPerpendicularA"] == 2


def test_set_global_load_tile_dim_classic_rejects_excess_coalescing():
    state = _classic_tile_state(True, coalesced=8, macro_tile=128)
    assert Solution.setGlobalLoadTileDimClassic(
        state, "A", 0, 8, 8, 16, False
    ) is False
    assert state["Valid"] is False


@pytest.mark.parametrize(
    "cluster_dim,expected",
    [([1, 1], False), ([2, 1], True)],
)
def test_problem_independent_derivation_sets_multicast(
    real_state, isa_info_map, cluster_dim, expected
):
    real_state["AssignedProblemIndependentDerivedParameters"] = False
    real_state["ClusterDim"] = cluster_dim
    Solution.assignProblemIndependentDerivedParameters(real_state, False, isa_info_map)
    assert real_state["Multicast"] is expected


def test_problem_independent_derivation_enables_f32x_emulation(real_state, isa_info_map):
    real_state["AssignedProblemIndependentDerivedParameters"] = False
    real_state["UseF32XEmulation"] = True
    Solution.assignProblemIndependentDerivedParameters(real_state, False, isa_info_map)
    assert real_state["UseDirect32XEmulation"] is True
    assert real_state["UseMFMAF32XEmulation"] is True
    assert real_state["UseDot2F32XEmulation"] is False


def test_disable_runtime_stagger_u_resets_fields_without_pollution():
    state = {
        "StaggerU": 9,
        "StaggerUMapping": 8,
        "StaggerUStride": 7,
        "InternalSupportParams": {
            "SupportCustomStaggerU": True,
            "Other": 42,
        },
    }

    S._disableRuntimeStaggerU(state)

    assert (state["StaggerU"], state["StaggerUMapping"], state["StaggerUStride"]) == (
        0, 0, 0
    )
    assert state["InternalSupportParams"] == {
        "SupportCustomStaggerU": False,
        "Other": 42,
    }


@pytest.mark.parametrize(
    "prefetch,tdm,disabled",
    [
        (True, 3, True),
        (True, 0, False),
        (False, 3, False),
        (False, 0, False),
    ],
)
def test_disable_unsupported_runtime_stagger_u_requires_pap_and_tdm3(
    prefetch, tdm, disabled
):
    state = {
        "PrefetchAcrossPersistent": prefetch,
        "TDMInst": tdm,
        "StaggerU": 9,
        "StaggerUMapping": 8,
        "StaggerUStride": 7,
        "InternalSupportParams": {"SupportCustomStaggerU": True},
    }

    S._disableUnsupportedRuntimeStaggerU(state)

    expected = 0 if disabled else 9
    assert state["StaggerU"] == expected
    assert state["InternalSupportParams"]["SupportCustomStaggerU"] is (not disabled)


@pytest.mark.parametrize(
    "streamk,atomic,expected",
    [
        (3, 0, True),
        (0, 0, False),
        (4, 0, False),
        (3, 1, False),
    ],
)
def test_validate_streamk_force_dp_only(streamk, atomic, expected):
    state = {
        "StreamKForceDPOnly": True,
        "StreamK": streamk,
        "StreamKAtomic": atomic,
    }
    assert S._validateStreamKForceDPOnly(state, False) is expected
    assert state.get("Valid", True) is expected


def test_validate_streamk_force_dp_only_disabled_short_circuits():
    state = {"StreamKForceDPOnly": False}
    assert S._validateStreamKForceDPOnly(state, False) is True
    assert "Valid" not in state


@pytest.mark.parametrize(
    "tc,index,expected",
    [
        ("A", 11, True),
        ("A", 12, False),
        ("B", 21, True),
        ("B", 22, False),
        (None, 11, True),
        (None, 21, True),
    ],
)
def test_extractable_index_excludes_final_packed_index(tc, index, expected):
    state = {
        "PackedC0IndicesX": [10, 11, 12],
        "PackedC1IndicesX": [20, 21, 22],
    }
    if tc is None:
        result = S.isExtractableIndex(state, index)
    else:
        result = S.isExtractableIndex(state, index, tc)
    assert result is expected


def test_activation_setting_defaults_to_empty_enum():
    setting = S.activationSetting()
    assert setting.activationEnum == ""
    assert not setting.activationEnum


def _mx_state(load="Auto", scale="Auto", tdm=0, isa=(9, 4, 2), mx_block=False):
    return {
        "ProblemType": {"MXBlockA": mx_block, "MXBlockB": False},
        "MXLoadInst": load,
        "MXScaleFormat": scale,
        "TDMInst": tdm,
        "ISA": isa,
        "StreamK": 0,
        "Valid": True,
    }


@pytest.mark.parametrize(
    "state,asm_caps,arch_caps,expected",
    [
        (_mx_state(tdm=2), {"HasTDM": True}, {"HasMXScaleSwizzle": True},
         (True, "TDM", "InMemorySwizzle")),
        (_mx_state(), {"HasTDM": True}, {"HasMXScaleSwizzle": False},
         (True, "BufferLoad", "NoSwizzle")),
        (_mx_state(isa=(9, 5, 0), mx_block=True), {"HasTDM": True},
         {"HasMXScaleSwizzle": True}, (True, "BufferLoad", "HostPreSwizzle")),
        (_mx_state(load="GlobalLoad", scale="NoSwizzle"), {"HasTDM": True},
         {"HasMXScaleSwizzle": True}, (False, "GlobalLoad", "NoSwizzle")),
    ],
    ids=["tdm", "buffer-load", "gfx950-host-preswizzle", "unsupported-global-load"],
)
def test_mx_scale_layout_and_transport(state, asm_caps, arch_caps, expected):
    result = S._deriveAndValidateMXScaleLayoutAndTransport(
        state, asm_caps, arch_caps, False
    )
    assert (result, state["MXLoadInst"], state["MXScaleFormat"]) == expected
    assert state["Valid"] is result


@pytest.mark.parametrize(
    "overrides",
    [
        {"GlobalReadVectorWidthA": 4},
        {"GlobalReadVectorWidthA": 8},
        {"GlobalReadVectorWidthA": 1},
        {"EnableMatrixInstruction": False},
        {"LocalReadVectorWidthA": 8},
        {"NumThreads": 250},
    ],
    ids=["b64", "b128", "sub-dword", "not-mi", "wide-lrvw", "bad-wave-count"],
)
def test_direct_to_lds_rejects_unsupported_shapes(real_state, isa_info_map, overrides):
    state = copy.deepcopy(real_state)
    state.pop("SolutionIndex", None)
    state.pop("SolutionNameMin", None)
    state.update({"GlobalReadVectorWidthA": 2, "UseGeneralizedNLCOneA": True})
    state.update(overrides)
    state["Valid"] = True

    assert Solution.isDirectToLdsDoable(state, "A", isa_info_map, False) is False
