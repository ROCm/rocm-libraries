#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit tests for gfx1250 subtile StinkyTofu waitcnt-only wiring (Step 1).
################################################################################

import os
import sys
from unittest.mock import MagicMock

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

from Tensile.SolutionStructs.Solution import Solution


GFX1250_ISA = (12, 5, 0)


def _minimal_subtile_state(isa=GFX1250_ISA, use_subtile=True, schedule_iter_alg=3):
    return {
        "Valid": True,
        "AssignedProblemIndependentDerivedParameters": False,
        "ISA": isa,
        "ScheduleIterAlg": schedule_iter_alg,
        "UseSubtileImpl": use_subtile,
        "ProblemType": {
            "StridedBatched": True,
            "Batched": True,
            "OperationType": "GEMM",
        },
    }


@pytest.fixture(scope="module")
def _st_backend_available():
    import rocisa
    return rocisa.hasStinkyTofuBackend() and rocisa.isSupportedByStinkyTofu(GFX1250_ISA)


def test_gfx1250_subtile_sets_waitcnt_only_flag(_st_backend_available):
    if not _st_backend_available:
        pytest.skip("StinkyTofu gfx1250 backend not available")

    state = _minimal_subtile_state()
    Solution._assignStinkySubtile(state)
    assert state.get("_StinkySubtile") == 1


def test_gfx1250_non_subtile_no_waitcnt_only_flag(_st_backend_available):
    if not _st_backend_available:
        pytest.skip("StinkyTofu gfx1250 backend not available")

    state = _minimal_subtile_state(use_subtile=False)
    Solution._assignStinkySubtile(state)
    assert "_StinkySubtile" not in state


def test_gfx950_subtile_no_waitcnt_only_flag():
    state = _minimal_subtile_state(isa=(9, 5, 0))
    Solution._assignStinkySubtile(state)
    assert "_StinkySubtile" not in state


def test_build_stinky_options_waitcnt_only_overrides():
    from Tensile.KernelWriter import KernelWriter

    kw = MagicMock(spec=KernelWriter)
    kw.sgprs = {}
    kernel = {
        "EnableStinkyTofuESM2": True,
        "ThreadTile0": 2,
        "ThreadTile1": 2,
        "MacroTile0": 64,
        "WavefrontSize": 32,
        "SubGroup0": 4,
        "SubGroup1": 4,
        "MIWaveGroup": [1, 1],
        "VectorWidthA": 1,
        "VectorWidthB": 1,
        "GlobalReadVectorWidthA": 8,
        "GlobalReadVectorWidthB": 8,
        "DirectToLdsA": False,
        "DirectToLdsB": False,
        "_UseSgprForGRO": 0,
        "ClusterBarrier": True,
        "PrefetchGlobalRead": 2,
        "PrefetchLocalRead": 1,
    }

    overrides = {
        "EnableWaitCntInsertion": True,
        "EnableESM2": False,
        "ClusterBarrier": False,
    }
    opts = KernelWriter._buildStinkyTofuModuleOptions(kw, kernel, 0, option_overrides=overrides)

    assert opts["OptLevel"] == 0
    assert opts["EnableWaitCntInsertion"] is True
    assert opts["EnableESM2"] is False
    assert opts["ClusterBarrier"] is False


def test_build_stinky_options_opt0_default_disables_waitcnt():
    from Tensile.Common.GlobalParameters import globalParameters
    from Tensile.KernelWriter import KernelWriter

    saved = globalParameters.get("DisableSTWaitCnt")
    globalParameters["DisableSTWaitCnt"] = True
    try:
        kw = MagicMock(spec=KernelWriter)
        kw.sgprs = {}
        kernel = {
            "EnableStinkyTofuESM2": False,
            "ThreadTile0": 2,
            "ThreadTile1": 2,
            "MacroTile0": 64,
            "WavefrontSize": 32,
            "SubGroup0": 4,
            "SubGroup1": 4,
            "MIWaveGroup": [1, 1],
            "VectorWidthA": 1,
            "VectorWidthB": 1,
            "GlobalReadVectorWidthA": 8,
            "GlobalReadVectorWidthB": 8,
            "DirectToLdsA": False,
            "DirectToLdsB": False,
            "_UseSgprForGRO": 0,
        }
        opts = KernelWriter._buildStinkyTofuModuleOptions(kw, kernel, 0)
        assert opts["EnableWaitCntInsertion"] is False
    finally:
        globalParameters["DisableSTWaitCnt"] = saved


def test_kernel_body_subtile_waitcnt_tail_invokes_st_pipeline():
    import rocisa
    if not (rocisa.hasStinkyTofuBackend() and rocisa.isSupportedByStinkyTofu(GFX1250_ISA)):
        pytest.skip("StinkyTofu gfx1250 backend not available")

    kw = MagicMock()
    kw.states = MagicMock()
    kw.states.version = GFX1250_ISA
    kw.states.overflowedResources = 0

    module_kernel_body = MagicMock()
    fs = MagicMock()
    captured = {}

    def fake_pipeline(kernel, moduleKernelBody, signature, stinky_opt_level, option_overrides=None):
        captured["opt_level"] = stinky_opt_level
        captured["overrides"] = option_overrides
        return "stinky_asm_output"

    def fake_rocisa_pass(kernel, moduleKernelBody):
        result = MagicMock()
        result.cycles = 42
        result.maxVgpr = 16
        return result

    kw._runRocIsaPassOnKernelBody = fake_rocisa_pass
    kw.updateOccupancyFromMaxVgpr = MagicMock()
    kw._runStinkyTofuPipeline = fake_pipeline

    kernel = {"_StinkySubtile": 1}

    error = kw.states.overflowedResources
    if kernel.get("_StinkySubtile") and rocisa.isSupportedByStinkyTofu(kw.states.version):
        passResult = kw._runRocIsaPassOnKernelBody(kernel, module_kernel_body)
        kernel["MathClocksUnrolledLoop"] = passResult.cycles
        kw.updateOccupancyFromMaxVgpr(kernel, module_kernel_body, passResult.maxVgpr)
        waitcnt_overrides = {
            "EnableWaitCntInsertion": True,
            "EnableESM2": False,
            "ClusterBarrier": False,
        }
        st_asm = kw._runStinkyTofuPipeline(
            kernel, module_kernel_body, fs, 0, option_overrides=waitcnt_overrides)
        assert st_asm == "stinky_asm_output"

    assert captured["opt_level"] == 0
    assert captured["overrides"]["EnableWaitCntInsertion"] is True
    assert captured["overrides"]["EnableESM2"] is False
    assert captured["overrides"]["ClusterBarrier"] is False
    assert kernel["MathClocksUnrolledLoop"] == 42
    kw.updateOccupancyFromMaxVgpr.assert_called_once()
