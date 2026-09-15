# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Host-only validator tests for RMSEpilogue epilogue constraints."""

import pytest

pytestmark = pytest.mark.unit

from Tensile.Common.DataType import DataType
from Tensile.SolutionStructs.Solution import (
    _validateRMSEpilogue,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _makeRMSEpilogueState(mt0=128, mt1=128, dest="B", mi=16):
    """Build a minimal state that passes all _validateRMSEpilogue structural checks."""
    return {
        "RMSEpilogue": True,
        "UseSubtileImpl": True,
        "ISA": (9, 5, 0),
        "ProblemType": {
            "DataType": DataType("B"),
            "DestDataType": DataType(dest),
            "HighPrecisionAccumulate": True,
            "UseBeta": False,
            "OutputAmaxD": False,
            "GroupedGemm": False,
        },
        "StreamK": 0,
        "StreamKForceDPOnly": True,
        "MIArchVgpr": False,
        "PrefetchAcrossPersistent": 0,
        "MacroTile0": mt0,
        "MacroTile1": mt1,
        "MIWaveGroup": [1, 1],
        "MatrixInstN": mi,
        "MatrixInstM": mi,
        "MaxLDS": 65536,
        "WavefrontSize": 64,
        "Valid": True,
    }


# ---------------------------------------------------------------------------
# RMSEpilogue macro-tile multiple-of-64 constraint
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mt0,mt1", [
    (320, 320),  # large square tile must be accepted
    (64, 64),
    (128, 256),
])
def test_rmsepilogue_multipleOf64Accepted(mt0, mt1):
    state = _makeRMSEpilogueState(mt0=mt0, mt1=mt1)
    _validateRMSEpilogue(state, False)
    assert state.get("Valid") is True


@pytest.mark.parametrize("mt0,mt1", [
    (100, 320),  # MT0 not a multiple of 64
    (320, 100),  # MT1 not a multiple of 64
    (32, 64),    # power of two but not a multiple of 64
])
def test_rmsepilogue_nonMultipleOf64Rejected(mt0, mt1):
    state = _makeRMSEpilogueState(mt0=mt0, mt1=mt1)
    _validateRMSEpilogue(state, False)
    assert state.get("Valid") is False


# ---------------------------------------------------------------------------
# RMSEpilogue MI 16x16 constraint
# ---------------------------------------------------------------------------

def test_rmsepilogue_mi16x16_accepted():
    state = _makeRMSEpilogueState(dest="B", mi=16)
    _validateRMSEpilogue(state, False)
    assert state.get("Valid") is True


def test_rmsepilogue_wrong_mi_rejected():
    state = _makeRMSEpilogueState(dest="B", mi=32)
    _validateRMSEpilogue(state, False)
    assert state.get("Valid") is False


# ---------------------------------------------------------------------------
# RMSEpilogue MXFP8 derived-condition tests
# ---------------------------------------------------------------------------

def test_rmsepilogue_mxfp8_f8_dest_accepted():
    # F8 dest + RMSEpilogue + HPA + UseBeta=False: MXFP8 path accepted.
    state = _makeRMSEpilogueState(dest="F8", mi=16)
    _validateRMSEpilogue(state, False)
    assert state.get("Valid") is True


def test_rmsepilogue_non_f8_dest_no_quant():
    # Non-F8 dest: useMxfp8=False, no MXFP8 checks run.
    state = _makeRMSEpilogueState(dest="B", mi=16)
    _validateRMSEpilogue(state, False)
    assert state.get("Valid") is True


def test_rmsepilogue_mxfp8_no_hpa_rejected():
    # F8 dest without HighPrecisionAccumulate: MXFP8 path requires HPA=True.
    state = _makeRMSEpilogueState(dest="F8", mi=16)
    state["ProblemType"]["HighPrecisionAccumulate"] = False
    _validateRMSEpilogue(state, False)
    assert state.get("Valid") is False


def test_rmsepilogue_mxfp8_use_beta_rejected():
    # F8 dest with UseBeta=True: MXFP8 path requires UseBeta=False.
    state = _makeRMSEpilogueState(dest="F8", mi=16)
    state["ProblemType"]["UseBeta"] = True
    _validateRMSEpilogue(state, False)
    assert state.get("Valid") is False


def test_rmsepilogue_skipped_when_false():
    # RMSEpilogue=False: validator returns early, nothing rejected.
    state = _makeRMSEpilogueState(dest="B", mi=16)
    state["RMSEpilogue"] = False
    _validateRMSEpilogue(state, False)
    assert state.get("Valid") is True
