# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-only validator tests for PartialRMS and TileQuant epilogue constraints."""

import pytest

pytestmark = pytest.mark.unit

from Tensile.Common.DataType import DataType
from Tensile.SolutionStructs.Solution import _validatePartialRMS, _validateTileQuant


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _makePartialRMSState(mt0, mt1, quant=False, residual=False):
    return {
        "PartialRMS": True,
        "PartialRMSQuant": quant,
        "UseSubtileImpl": True,
        "ISA": (9, 5, 0),
        "ProblemType": {"DataType": DataType("B"), "OutputAmaxD": False, "GroupedGemm": False},
        "StreamK": 0,
        "StreamKForceDPOnly": True,
        "MIArchVgpr": False,
        "PrefetchAcrossPersistent": 0,
        "MacroTile0": mt0,
        "MacroTile1": mt1,
        "MIWaveGroup": [1, 1],
        "MatrixInstN": 16,
        "MatrixInstM": 16,
        "MaxLDS": 65536,
        "WavefrontSize": 64,
        "Valid": True,
    }


def _partialRMSAccepted(mt0, mt1):
    state = _makePartialRMSState(mt0, mt1)
    _validatePartialRMS(state, False)
    return state.get("Valid")


def _makeTileQuantState(q0, q1, mt0=128, mt1=128, wg_m=1, wg_n=1):
    return {
        "DQuantType": "Tile",
        "DQuantSize0": q0,
        "DQuantSize1": q1,
        "UseSubtileImpl": True,
        "ISA": (9, 5, 0),
        "ProblemType": {
            "DataType": DataType("B"),
            "DestDataType": DataType("F8"),
            "ComputeDataType": DataType("S"),
            "HighPrecisionAccumulate": True,
            "UseBeta": False,
            "UseScaleCD": False,
            "OutputAmaxD": False,
            "GroupedGemm": False,
            "Batched": False,
            "NumIndicesC": 2,
        },
        "PartialRMS": False,
        "StreamK": 3,
        "StreamKForceDPOnly": True,
        "MIArchVgpr": False,
        "MacroTile0": mt0,
        "MacroTile1": mt1,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MIWaveGroup": [wg_m, wg_n],
        "WavefrontSize": 64,
        "MaxLDS": 65536,
        "PrefetchAcrossPersistent": 0,
        "Valid": True,
    }


# ---------------------------------------------------------------------------
# PartialRMS macro-tile multiple-of-64 constraint
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mt0,mt1", [
    (320, 320),  # large square tile must be accepted
    (64, 64),
    (128, 256),
])
def test_partialrms_multipleOf64Accepted(mt0, mt1):
    assert _partialRMSAccepted(mt0, mt1) is True


@pytest.mark.parametrize("mt0,mt1", [
    (100, 320),  # MT0 not a multiple of 64
    (320, 100),  # MT1 not a multiple of 64
    (32, 64),    # power of two but not a multiple of 64
])
def test_partialrms_nonMultipleOf64Rejected(mt0, mt1):
    assert _partialRMSAccepted(mt0, mt1) is False


# ---------------------------------------------------------------------------
# PartialRMSQuant implication constraint
# ---------------------------------------------------------------------------

def test_partialrmsquant_requires_partialrms():
    state = _makePartialRMSState(128, 128, quant=True)
    state["PartialRMS"] = False
    _validatePartialRMS(state, False)
    assert state.get("Valid") is False


def test_partialrmsquant_with_partialrms_accepted():
    state = _makePartialRMSState(128, 128, quant=True)
    _validatePartialRMS(state, False)
    assert state.get("Valid") is True


# ---------------------------------------------------------------------------
# TileQuant constraint logic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q0,q1", [(16, 16), (32, 32), (64, 64), (128, 128)])
def test_tilequant_valid(q0, q1):
    state = _makeTileQuantState(q0, q1)
    _validateTileQuant(state, True)
    assert state["Valid"]


@pytest.mark.parametrize("q0,q1", [(48, 64), (64, 48), (3, 16)])
def test_tilequant_reject_nonpow2(q0, q1):
    state = _makeTileQuantState(q0, q1)
    _validateTileQuant(state, False)
    assert not state["Valid"]


def test_tilequant_reject_nondividing():
    # 96 does not divide 128 evenly.
    state = _makeTileQuantState(96, 128)
    _validateTileQuant(state, False)
    assert not state["Valid"]


def test_tilequant_reject_exceeds_wave():
    # q0=64 divides mt0=128, but wave span = 128//4 = 32, so 64 > 32 triggers the guard.
    state = _makeTileQuantState(64, 64, mt0=128, mt1=128, wg_m=4, wg_n=4)
    _validateTileQuant(state, False)
    assert not state["Valid"]


def test_tilequant_reject_non_fp8_dest():
    state = _makeTileQuantState(16, 16)
    state["ProblemType"]["DestDataType"] = DataType("B")
    _validateTileQuant(state, False)
    assert not state["Valid"]


def test_tilequant_reject_use_scale_cd():
    state = _makeTileQuantState(16, 16)
    state["ProblemType"]["UseScaleCD"] = True
    _validateTileQuant(state, False)
    assert not state["Valid"]


def test_tilequant_reject_with_partialrms():
    state = _makeTileQuantState(16, 16)
    state["PartialRMS"] = True
    _validateTileQuant(state, False)
    assert not state["Valid"]


def test_tilequant_reject_use_beta():
    # beta != 0 corrupts per-tile scaling semantics.
    state = _makeTileQuantState(16, 16)
    state["ProblemType"]["UseBeta"] = True
    _validateTileQuant(state, False)
    assert not state["Valid"]


def test_tilequant_arbitrary_alpha_not_validated():
    # Alpha is a runtime scalar; the validator does not restrict it.
    state = _makeTileQuantState(16, 16)
    _validateTileQuant(state, True)
    assert state["Valid"]


def test_tilequant_reject_use_bias():
    state = _makeTileQuantState(16, 16)
    state["ProblemType"]["UseBias"] = 1
    _validateTileQuant(state, False)
    assert not state["Valid"]


def test_tilequant_reject_use_e():
    state = _makeTileQuantState(16, 16)
    state["ProblemType"]["UseE"] = True
    _validateTileQuant(state, False)
    assert not state["Valid"]


def test_tilequant_reject_use_gate_residual():
    state = _makeTileQuantState(16, 16)
    state["ProblemType"]["UseGateResidual"] = True
    _validateTileQuant(state, False)
    assert not state["Valid"]


def test_tilequant_reject_use_scale_alpha_vec():
    state = _makeTileQuantState(16, 16)
    state["ProblemType"]["UseScaleAlphaVec"] = 1
    _validateTileQuant(state, False)
    assert not state["Valid"]


def test_tilequant_reject_q0_between_rowsperlane_and_matrix_inst_m():
    # q0=8: rowsPerLane(4) < q0 < MatrixInstM(16) -> unsupported gap.
    state = _makeTileQuantState(8, 32)
    _validateTileQuant(state, False)
    assert not state["Valid"]


@pytest.mark.parametrize("q0,q1", [(1, 32), (2, 32), (4, 32), (1, 16), (4, 128)])
def test_tilequant_valid_subrow(q0, q1):
    state = _makeTileQuantState(q0, q1)
    _validateTileQuant(state, True)
    assert state["Valid"]


def test_tilequant_reject_subrow_q1_below_mfma_n():
    # q0=1 is sub-row OK, but q1=8 < mfmaN=16 -> still rejected.
    state = _makeTileQuantState(1, 8)
    _validateTileQuant(state, False)
    assert not state["Valid"]
