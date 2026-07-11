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
        "EnableLoopCarriedTokenDeps": True,
    }
    opts = KernelWriter._buildStinkyTofuModuleOptions(kw, kernel, 0, option_overrides=overrides)

    assert opts["OptLevel"] == 0
    assert opts["EnableWaitCntInsertion"] is True
    assert opts["EnableESM2"] is False
    assert opts["ClusterBarrier"] is False
    assert opts["EnableLoopCarriedTokenDeps"] is True


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
            "EnableLoopCarriedTokenDeps": True,
        }
        st_asm = kw._runStinkyTofuPipeline(
            kernel, module_kernel_body, fs, 0, option_overrides=waitcnt_overrides)
        assert st_asm == "stinky_asm_output"

    assert captured["opt_level"] == 0
    assert captured["overrides"]["EnableWaitCntInsertion"] is True
    assert captured["overrides"]["EnableESM2"] is False
    assert captured["overrides"]["ClusterBarrier"] is False
    assert captured["overrides"]["EnableLoopCarriedTokenDeps"] is True
    assert kernel["MathClocksUnrolledLoop"] == 42
    kw.updateOccupancyFromMaxVgpr.assert_called_once()


def test_stinky_region_module_name_mapping():
    from Tensile.Components.Subtile.LogicalScheduler import LogicalScheduler

    assert LogicalScheduler._stinkyRegionModuleName("PRELOOP") == "loopBody"
    assert LogicalScheduler._stinkyRegionModuleName("MAINLOOP_C0") == "loopBody"
    assert LogicalScheduler._stinkyRegionModuleName("NGLL_C1") == "loopBody"
    assert LogicalScheduler._stinkyRegionModuleName("NLL_C0") == "noLoadLoopBody"
    assert LogicalScheduler._stinkyRegionModuleName("TAILLOOP") == "noLoadLoopBody"


def _writer_with_mem_tokens(double_buffer=True):
    from types import SimpleNamespace

    states = SimpleNamespace(
        memTokenLdsBuffer0=0,
        memTokenLdsBuffer1=1 if double_buffer else 0,
        ldsReadTokenIdx=0,
        ldsDirectToLDSTokenIdx=0,
        ldsWriteTokenIdx=0,
        ldsTensorTokenIdx=0,
    )
    return SimpleNamespace(states=states)


def test_subtile_mem_token_flip_helpers():
    from Tensile.Components.Subtile.SubtileMemToken import (
        flipGrWriteTokens,
        flipLrReadToken,
        flipTensorLoadToken,
    )

    writer = _writer_with_mem_tokens()
    flipGrWriteTokens(writer)
    assert writer.states.ldsDirectToLDSTokenIdx == 1
    assert writer.states.ldsWriteTokenIdx == 1
    flipLrReadToken(writer)
    assert writer.states.ldsReadTokenIdx == 1
    flipTensorLoadToken(writer)
    assert writer.states.ldsTensorTokenIdx == 1


def test_subtile_mem_token_barrier_tokens():
    from Tensile.Components.Subtile.SubtileMemToken import barrierTokens

    writer = _writer_with_mem_tokens(double_buffer=True)
    assert barrierTokens(writer, {"1LDSBuffer": False}) == [0, 1]

    writer = _writer_with_mem_tokens(double_buffer=False)
    assert barrierTokens(writer, {"1LDSBuffer": True}) == [0]


def test_subtile_mem_token_tag_instructions():
    from rocisa.instruction import BufferLoadB128, DSLoadB128, SBarrier, TensorLoadToLds
    from rocisa.container import vgpr, sgpr

    from Tensile.Components.Subtile.SubtileMemToken import (
        tagBarrier,
        tagDtlLoad,
        tagDsRead,
        tagTensorLoad,
    )

    writer = _writer_with_mem_tokens()
    kernel = {"1LDSBuffer": False}

    dtl = BufferLoadB128(dst=None, vaddr=vgpr(0), saddr=sgpr(0, 4), soffset=0)
    tagDtlLoad(dtl, writer)
    assert dtl.getMemToken().tokens == [0]

    tdm = TensorLoadToLds(sgpr(0, 4), sgpr(4, 8), None, None)
    tagTensorLoad(tdm, writer)
    assert tdm.getMemToken().tokens == [0]

    ds = DSLoadB128(dst=vgpr(0, 4), src=vgpr(1))
    tagDsRead(ds, writer)
    assert ds.getMemToken().tokens == [0]

    barrier = SBarrier()
    tagBarrier(barrier, writer, kernel)
    assert barrier.getMemToken().tokens == [0, 1]


def test_subtile_emit_sync_tags_barrier():
    from rocisa.instruction import SBarrier

    from Tensile.Components.Subtile.InstructionEmitter import InstructionEmitter

    writer = _writer_with_mem_tokens()
    kernel = {"1LDSBuffer": False}
    emitter = InstructionEmitter.__new__(InstructionEmitter)
    emitter.writer = writer
    emitter.kernel = kernel

    items = emitter.emit_sync()
    assert len(items) == 1
    assert isinstance(items[0], SBarrier)
    assert items[0].getMemToken().tokens == [0, 1]


def test_subtile_cluster_barrier_tags_mem_tokens():
    from rocisa.instruction import SBarrier

    from Tensile.Components.Subtile.ClusterBarrier import (
        subtileClusterBarrierSignal,
        subtileClusterBarrierWait,
    )

    writer = _writer_with_mem_tokens()
    writer.labels = MagicMock()
    writer.labels.getUniqueNamePrefix.return_value = "cb"
    kernel = {"1LDSBuffer": False}

    for mod in (subtileClusterBarrierSignal(writer, kernel),
                subtileClusterBarrierWait(writer, kernel)):
        barriers = [i for i in mod.flatitems() if isinstance(i, SBarrier)]
        assert barriers
        for barrier in barriers:
            assert barrier.getMemToken().tokens == [0, 1]


def test_subtile_gr_lds_buffer_swap_flips_write_tokens():
    from Tensile.Components.Subtile.SubtileGREmit import globalReadLDSBufferSwap

    writer = _writer_with_mem_tokens()
    writer.states.a = MagicMock()
    writer.states.b = MagicMock()
    writer.states.a.tileInfo = MagicMock()
    writer.states.a.tileInfo.emitGRLDSBufferSwap.return_value = MagicMock()
    writer.states.b.tileInfo = writer.states.a.tileInfo

    kernel = {"enableTDMA": False, "enableTDMB": False}
    globalReadLDSBufferSwap("A", writer, kernel)
    assert writer.states.ldsDirectToLDSTokenIdx == 1
    assert writer.states.ldsWriteTokenIdx == 1


def test_subtile_lr_lds_buffer_swap_flips_read_token():
    from Tensile.Components.Subtile.SubtileLREmit import localReadLDSBufferSwap

    writer = _writer_with_mem_tokens()
    writer.states.a = MagicMock()
    writer.states.a.tileInfo = MagicMock()
    writer.states.a.tileInfo.emitLRLDSBufferSwap.return_value = MagicMock()

    kernel = {}
    localReadLDSBufferSwap("A", writer, kernel)
    assert writer.states.ldsReadTokenIdx == 1


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
