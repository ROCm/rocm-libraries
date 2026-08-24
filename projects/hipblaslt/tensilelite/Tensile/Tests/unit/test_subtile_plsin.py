#!/usr/bin/env python3
################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
################################################################################

import os
import importlib.util
from copy import deepcopy

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
PLSIN_PATH = os.path.join(TENSILE_ROOT, "Tensile", "Components", "Subtile", "Plsin.py")

_SPEC = importlib.util.spec_from_file_location("subtile_plsin", PLSIN_PATH)
_PLSIN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PLSIN)
computeSubtilePlsin = _PLSIN.computeSubtilePlsin
plsinLargeTile = _PLSIN.plsinLargeTile
PLSIN_WEAVE_LOOKAHEAD = _PLSIN.PLSIN_WEAVE_LOOKAHEAD


class _DType:
    def __init__(self, kind):
        self.kind = kind

    def isFloat4(self):
        return self.kind == "f4"

    def isBFloat16(self):
        return self.kind == "bf16"

    def isHalf(self):
        return self.kind == "half"

    def isSingle(self):
        return self.kind == "single"


def _eligible_kernel():
    return {
        "ISA": (9, 5, 0),
        "UseSubtileImpl": True,
        "EnableMatrixInstruction": True,
        "PrefetchGlobalRead": 2,
        "BufferStore": True,
        "WavefrontSize": 64,
        "StoreRemapVectorWidth": 0,
        "StreamK": 3,
        "StreamKAtomic": False,
        "DepthU": 512,
        "MIWaveTile": [8, 8],
        "MacroTile0": 256,
        "MacroTile1": 256,
        "_GlobalAccumulation": None,
        "TailloopInNll": False,
        "ProblemType": {
            "DataTypeA": _DType("f4"),
            "DataTypeB": _DType("f4"),
            "DestDataType": _DType("bf16"),
            "ComputeDataType": _DType("single"),
            "HighPrecisionAccumulate": True,
            "MXBlockA": 32,
            "MXBlockB": 32,
            "UseE": False,
            "UseScaleCD": False,
            "UseGateResidual": False,
            "ActivationType": "none",
        },
    }


@pytest.mark.parametrize("stream_k", [3, 4, 5])
def test_plsin_accepts_supported_streamk_modes(stream_k):
    kernel = _eligible_kernel()
    kernel["StreamK"] = stream_k
    assert computeSubtilePlsin(kernel)


@pytest.mark.parametrize("stream_k", [0, 1, 2])
def test_plsin_rejects_unvalidated_streamk_modes(stream_k):
    kernel = _eligible_kernel()
    kernel["StreamK"] = stream_k
    assert not computeSubtilePlsin(kernel)


def test_plsin_requires_mx_block_scaled_float4():
    kernel = _eligible_kernel()
    kernel["ProblemType"]["MXBlockA"] = 0
    kernel["ProblemType"]["MXBlockB"] = 0
    assert not computeSubtilePlsin(kernel)

    kernel = _eligible_kernel()
    kernel["ProblemType"]["DataTypeA"] = _DType("bf16")
    kernel["ProblemType"]["DataTypeB"] = _DType("bf16")
    assert not computeSubtilePlsin(kernel)


@pytest.mark.parametrize("depth_u, expected", [(256, True), (512, True), (768, False)])
def test_plsin_requires_power_of_two_depth_u(depth_u, expected):
    kernel = _eligible_kernel()
    kernel["DepthU"] = depth_u
    assert computeSubtilePlsin(kernel) is expected


def test_plsin_requires_fp32_compute():
    kernel = _eligible_kernel()
    kernel["ProblemType"]["ComputeDataType"] = _DType("half")
    assert not computeSubtilePlsin(kernel)


@pytest.mark.parametrize("feature", ["UseE", "UseGateResidual"])
def test_plsin_falls_back_for_unsupported_epilogues(feature):
    kernel = _eligible_kernel()
    kernel["ProblemType"][feature] = True
    assert not computeSubtilePlsin(kernel)


def test_plsin_supports_scale_cd_epilogue():
    kernel = _eligible_kernel()
    kernel["ProblemType"]["UseScaleCD"] = True
    assert computeSubtilePlsin(kernel)


def test_plsin_falls_back_for_tailloop_in_nll():
    kernel = _eligible_kernel()
    kernel["TailloopInNll"] = True
    assert not computeSubtilePlsin(kernel)


def test_plsin_keeps_supported_activation_eligible():
    kernel = _eligible_kernel()
    kernel["ActivationFused"] = True
    kernel["ProblemType"]["ActivationType"] = "relu"
    assert computeSubtilePlsin(kernel)


@pytest.mark.parametrize(
    "mi_wave_tile, expected",
    [
        ([8, 8], True),
        ([4, 13], True),
        ([3, 16], True),
        ([3, 4], True),
        ([4, 14], False),
        ([2, 4], False),
        ([2, 16], False),
        ([8, 9], False),
    ],
)
def test_plsin_register_budget_boundaries(mi_wave_tile, expected):
    kernel = _eligible_kernel()
    kernel["MIWaveTile"] = deepcopy(mi_wave_tile)
    assert computeSubtilePlsin(kernel) is expected


# ── plsinLargeTile: coord-weave gate, independent of PLSIN eligibility ──


@pytest.mark.parametrize(
    "mt0, mt1, expected",
    [
        (256, 256, False),
        (128, 256, False),
        (256, 512, True),
        (512, 256, True),
        (512, 512, True),
    ],
)
def test_plsin_large_tile_boundary(mt0, mt1, expected):
    kernel = _eligible_kernel()
    kernel["MacroTile0"] = mt0
    kernel["MacroTile1"] = mt1
    assert plsinLargeTile(kernel) is expected


def test_plsin_large_tile_does_not_gate_eligibility():
    # A MacroTile > 256 is a coord-WEAVE decision only; it must not by itself
    # disable PLSIN when the register/spill/profit budgets still pass.
    kernel = _eligible_kernel()
    kernel["MacroTile0"] = 512
    kernel["MacroTile1"] = 512
    assert plsinLargeTile(kernel) is True
    assert computeSubtilePlsin(kernel) is True


# ── PLSIN_WEAVE_LOOKAHEAD: shared profitability threshold ──


def test_weave_lookahead_is_shared_positive_constant():
    # The scheduler weave (LogicalScheduler) and the eligibility gate must read the
    # SAME lookahead. Lock its value here so a change in one place without the other
    # is caught (the two are documented to move together in Plsin.py).
    assert PLSIN_WEAVE_LOOKAHEAD == 2
    assert PLSIN_WEAVE_LOOKAHEAD > 0


def test_eligible_tile_has_more_store_pairs_than_lookahead():
    # Gate/weave consistency: any tile the gate accepts must leave at least one
    # store-pair in the loop to hide the woven pairs under, i.e.
    # numStorePairs (= MIWT0*MIWT1//2) > PLSIN_WEAVE_LOOKAHEAD.
    kernel = _eligible_kernel()
    miwt = kernel["MIWaveTile"]
    assert computeSubtilePlsin(kernel) is True
    numStorePairs = miwt[0] * miwt[1] // 2
    assert numStorePairs > PLSIN_WEAVE_LOOKAHEAD
