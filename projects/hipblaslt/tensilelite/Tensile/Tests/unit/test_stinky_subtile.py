#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
# Unit tests for subtile StinkyTofu waitcnt-only wiring.
################################################################################

import os
import sys
from unittest.mock import MagicMock

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)


GFX1250_ISA = (12, 5, 0)


def test_subtile_stinky_waitcnt_overrides():
    """Subtile ST overrides: matrix reuse + mode2 wait-alu, no GR/LR waitcnt,
    plus force-enabled VGPR-MSB and software (instruction) prefetch."""
    from Tensile.KernelWriter import KernelWriter

    kw = MagicMock()
    kw.states.asmCaps = {"HasVgprMSB": True, "HasVgprMSB16": True}
    kw.sgprs = {"SwPrefetchScratch": 5}

    overrides = KernelWriter._subtileStinkyTofuOverrides(kw)
    assert overrides["EnableESM2"] is True
    assert overrides["EnableWaitCntInsertion"] is False
    assert overrides["ClusterBarrier"] is False
    assert overrides["VgprMsbMode"] == 2
    assert overrides["EnableSwPrefetchInsertion"] is True


def test_subtile_stinky_overrides_msb8_and_no_prefetch_scratch():
    """MSB falls back to the 8-bit form; prefetch is not forced without scratch."""
    from Tensile.KernelWriter import KernelWriter

    kw = MagicMock()
    kw.states.asmCaps = {"HasVgprMSB": True, "HasVgprMSB16": False}
    kw.sgprs = {}

    overrides = KernelWriter._subtileStinkyTofuOverrides(kw)
    assert overrides["VgprMsbMode"] == 1
    assert "EnableSwPrefetchInsertion" not in overrides


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
    """Subtile path unconditionally runs the ST pipeline with waitcnt overrides."""
    from Tensile.KernelWriter import KernelWriter

    kw = MagicMock()
    kw.states = MagicMock()
    kw.states.version = GFX1250_ISA
    kw.states.overflowedResources = 0
    kw.states.asmCaps = {"HasVgprMSB": True, "HasVgprMSB16": True}
    kw.sgprs = {"SwPrefetchScratch": 5}

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

    kernel = {}

    error = kw.states.overflowedResources
    passResult = kw._runRocIsaPassOnKernelBody(kernel, module_kernel_body)
    kernel["MathClocksUnrolledLoop"] = passResult.cycles
    kw.updateOccupancyFromMaxVgpr(kernel, module_kernel_body, passResult.maxVgpr)
    overrides = KernelWriter._subtileStinkyTofuOverrides(kw)
    st_asm = kw._runStinkyTofuPipeline(
        kernel, module_kernel_body, fs, 0, option_overrides=overrides)
    assert st_asm == "stinky_asm_output"

    assert captured["opt_level"] == 0
    assert captured["overrides"]["EnableESM2"] is True
    assert captured["overrides"]["EnableWaitCntInsertion"] is False
    assert captured["overrides"]["ClusterBarrier"] is False
    assert kernel["MathClocksUnrolledLoop"] == 42
    kw.updateOccupancyFromMaxVgpr.assert_called_once()


def test_stinky_region_module_name_mapping():
    from Tensile.Components.Subtile.LogicalScheduler import LogicalScheduler

    assert LogicalScheduler._stinkyRegionModuleName("PRELOOP") == "loopBody"
    assert LogicalScheduler._stinkyRegionModuleName("MAINLOOP_C0") == "loopBody"
    assert LogicalScheduler._stinkyRegionModuleName("NGLL_C1") == "loopBody"
    assert LogicalScheduler._stinkyRegionModuleName("NLL_C0") == "noLoadLoopBody"
    assert LogicalScheduler._stinkyRegionModuleName("TAILLOOP") == "noLoadLoopBody"


def test_mainloop_exposes_top_level_loop_body_module():
    """loopBody must be a top-level kernel-body module for ST region detection."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from rocisa.code import Module
    from Tensile.Components.Subtile.Kernel import mainLoop

    loop_body = Module("loopBody")
    tail_loop = Module("TAILLOOP")
    ti = MagicMock()
    ti.subtileShape = [1, 1]
    ti.localMMATileGrid = [4, 4]
    ti.loadRatioGR = 1.0
    writer = SimpleNamespace(
        tPA=MagicMock(), tPB=MagicMock(),
        states=SimpleNamespace(
            a=SimpleNamespace(tileInfo=ti),
            b=SimpleNamespace(tileInfo=ti),
            mxsa=SimpleNamespace(tileInfo=None),
            mxsb=SimpleNamespace(tileInfo=None),
            d=SimpleNamespace(tileInfo=MagicMock()),
            regCaps={"MaxVgpr": 256},
            archCaps={},
        ),
        vgprPool=MagicMock(size=MagicMock(return_value=256),
                           available=MagicMock(return_value=0)),
        allocTmpSgpr=MagicMock(),
        calculateLoopNumIter=MagicMock(return_value=Module()),
        computeTailLoopSrdLimit=MagicMock(return_value=Module()),
        closeLoop=MagicMock(return_value=Module()),
    )
    writer.vgprPool.checkOut = MagicMock(return_value=0)
    writer.vgprPool.checkIn = MagicMock()
    kernel = {
        "PrefetchGlobalRead": 0,
        "NoTailLoop": True,
        "ProblemType": {},
        "MatrixInstK": 32,
        "enableTDMA": False,
        "enableTDMB": False,
    }

    scheduler = MagicMock()
    scheduler.build = MagicMock()
    scheduler.getNumVgpr = MagicMock(return_value=0)
    scheduler.allocVgprTiles = MagicMock()
    scheduler.populate_instructions = MagicMock()
    scheduler.emitMainAndExitLoops = MagicMock(return_value=loop_body)
    scheduler.deallocVgprTiles = MagicMock()

    with patch("Tensile.Components.Subtile.Kernel.LogicalScheduler", return_value=scheduler), \
         patch("Tensile.Components.Subtile.Kernel.MFMASchedulerConfig") as mock_cfg, \
         patch("Tensile.Components.Subtile.Kernel.MFMASchedulerConfig.get_partition_candidates",
               return_value=[(1, 1)]):
        mock_cfg.return_value = MagicMock()
        parts = mainLoop(writer, kernel)

    assert any(getattr(p, "name", None) == "loopBody" for p in parts)


def test_mainloop_exposes_top_level_tail_no_load_loop_body_module():
    """Tail loop body must be a top-level noLoadLoopBody module for ST waitcnt."""
    from types import SimpleNamespace
    from unittest.mock import MagicMock, patch

    from rocisa.code import Module
    from Tensile.Components.Subtile.Kernel import mainLoop

    loop_body = Module("loopBody")
    tail_loop_body = Module("noLoadLoopBody")
    ti = MagicMock()
    ti.subtileShape = [1, 1]
    ti.localMMATileGrid = [4, 4]
    ti.loadRatioGR = 1.0
    writer = SimpleNamespace(
        tPA=MagicMock(), tPB=MagicMock(),
        states=SimpleNamespace(
            a=SimpleNamespace(tileInfo=ti),
            b=SimpleNamespace(tileInfo=ti),
            mxsa=SimpleNamespace(tileInfo=None),
            mxsb=SimpleNamespace(tileInfo=None),
            d=SimpleNamespace(tileInfo=MagicMock()),
            regCaps={"MaxVgpr": 256},
            archCaps={},
        ),
        vgprPool=MagicMock(size=MagicMock(return_value=256),
                           available=MagicMock(return_value=0)),
        allocTmpSgpr=MagicMock(),
        calculateLoopNumIter=MagicMock(return_value=Module()),
        computeTailLoopSrdLimit=MagicMock(return_value=Module()),
        closeLoop=MagicMock(return_value=Module()),
    )
    writer.vgprPool.checkOut = MagicMock(return_value=0)
    writer.vgprPool.checkIn = MagicMock()
    kernel = {
        "PrefetchGlobalRead": 0,
        "NoTailLoop": False,
        "ProblemType": {},
        "MatrixInstK": 32,
        "enableTDMA": False,
        "enableTDMB": False,
    }

    scheduler = MagicMock()
    scheduler.build = MagicMock()
    scheduler.getNumVgpr = MagicMock(return_value=0)
    scheduler.allocVgprTiles = MagicMock()
    scheduler.populate_instructions = MagicMock()
    scheduler.emitMainAndExitLoops = MagicMock(return_value=loop_body)
    scheduler.emitTailLoop = MagicMock(
        return_value=[Module("TailLoopSetup"), tail_loop_body, Module("TailLoopCleanup")])
    scheduler.deallocVgprTiles = MagicMock()
    scheduler._is_multi_du = MagicMock(return_value=False)

    with patch("Tensile.Components.Subtile.Kernel.LogicalScheduler", return_value=scheduler), \
         patch("Tensile.Components.Subtile.Kernel.MFMASchedulerConfig") as mock_cfg, \
         patch("Tensile.Components.Subtile.Kernel.MFMASchedulerConfig.get_partition_candidates",
               return_value=[(1, 1)]):
        mock_cfg.return_value = MagicMock()
        parts = mainLoop(writer, kernel)

    assert any(getattr(p, "name", None) == "noLoadLoopBody" for p in parts)
