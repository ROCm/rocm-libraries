################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Solution derivation of the deep TDM LDS ring (PrefetchGlobalRead 3-4 on TDM A+B)."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from Tensile.Common.GlobalParameters import defaultSolution, globalParameters
from Tensile.Common.Types import IsaInfo, IsaVersion, SemanticVersion
from Tensile.Components.TDMRing import _TDM_RING_UNSUPPORTED_FLAGS, tdmDeepRing
from Tensile.SolutionStructs.Solution import Solution

pytestmark = pytest.mark.unit

# Solution.__init__ reads these module-level defaults, which sibling tests mutate in place.
_PRISTINE_DEFAULT_SOLUTION = deepcopy(defaultSolution)
_PRISTINE_GLOBAL_PARAMETERS = deepcopy(globalParameters)


@pytest.fixture(autouse=True)
def _isolate_global_solution_state():
    def _restore(target, pristine):
        target.clear()
        target.update(deepcopy(pristine))

    _restore(defaultSolution, _PRISTINE_DEFAULT_SOLUTION)
    _restore(globalParameters, _PRISTINE_GLOBAL_PARAMETERS)
    yield
    _restore(defaultSolution, _PRISTINE_DEFAULT_SOLUTION)
    _restore(globalParameters, _PRISTINE_GLOBAL_PARAMETERS)


class _DefaultFalseDict(dict):
    def __missing__(self, key):
        return False


def _isaInfoMap():
    asmCaps = _DefaultFalseDict({
        "SupportedISA": True, "HasMFMA": True, "HasTDM": True, "HasDirectToLds": True,
        "HasDirectToLdsx4": True, "HasNTModifier": True, "HasMFMA_f64": True,
        "HasGLTr8B64": True, "HasGLTr16B128": True, "HasLDSTr": True,
    })
    # Four slots of the 64x32 f32 tile at DepthU 128 take 192 KiB of LDS.
    archCaps = _DefaultFalseDict({
        "DeviceLDS": 327680, "HasEccHalf": True, "HasSchedMode": True, "HasAccCD": True,
        "HasMXScaleSwizzle": True,
    })
    return {IsaVersion(9, 5, 0): IsaInfo(asmCaps, archCaps, _DefaultFalseDict(),
                                         _DefaultFalseDict())}


def _solution(**overrides):
    """Two-wave TDM A+B state under ScheduleIterAlg 0, from the TDM contract that
    test_PrefetchAcrossPersistent validates, without StreamK and PAP."""
    config = {
        "ISA": [9, 5, 0],
        "EnableMatrixInstruction": True,
        "ProblemType": {
            "OperationType": "GEMM",
            "DataType": "s",
            "DestDataType": "s",
            "ComputeDataType": "s",
            "TransposeA": True,
            "TransposeB": False,
            "UseBeta": True,
            "Batched": True,
            "StridedBatched": True,
        },
        "MatrixInstruction": [32, 32, 8, 1],
        "MIBlock": [32, 32, 8, 1, 1, 1],
        "MIWaveGroup": [2, 1],
        "MIWaveTile": [1, 1],
        "MIInputPerThread": 1,
        "WorkGroup": [128, 1, 1],
        "WavefrontSize": 64,
        "PrefetchGlobalRead": 2,
        "ScheduleIterAlg": 0,
        "TDMInst": 3,
        "StaggerU": 0,
        "1LDSBuffer": 0,
        "BufferLoad": True,
        "BufferStore": True,
        "StoreRemapVectorWidth": 0,
        "SuppressNoLoadLoop": False,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "UseSubtileImpl": False,
        "DepthU": 128,
        "LocalSplitU": 1,
    }
    config.update(overrides)
    assembler = SimpleNamespace(code_object_version="default",
                                rocm_version=SemanticVersion(6, 4, 0))
    return Solution(config, False, True, False, assembler, _isaInfoMap())


def test_two_wave_tdm_pgr2_base_is_valid(capsys):
    """The base the ring cases change only PrefetchGlobalRead of."""
    solution = _solution()
    out = capsys.readouterr().out
    assert solution["Valid"] is True, out
    assert solution["enableTDMA"] and solution["enableTDMB"]
    assert solution["NumWaves"] >= 2
    assert not tdmDeepRing(solution)


@pytest.mark.parametrize("pgr", [3, 4])
def test_ring_takes_one_lds_block_per_stage(capsys, pgr):
    solution = _solution(PrefetchGlobalRead=pgr)
    out = capsys.readouterr().out
    assert solution["Valid"] is True, out
    assert tdmDeepRing(solution)
    assert solution["NumLdsBlk"] == pgr
    assert solution["ScheduleGROverBarrier"] == 0
    # Checked while still auto (-1), the custom main-loop schedule must then resolve off;
    # every other flag the ring rejects when on ends up off as well.
    assert solution["UseCustomMainLoopSchedule"] == 0
    for flag in _TDM_RING_UNSUPPORTED_FLAGS:
        assert solution.get(flag, 0) in (0, False), (flag, solution.get(flag))
    assert solution["LDSSegmentInterleave"] == 0
    assert "TDM LDS ring" not in out


@pytest.mark.parametrize("cms, valid", [pytest.param(0, True, id="cms-off"),
                                        pytest.param(1, False, id="cms-on")])
def test_ring_custom_main_loop_schedule_set_explicitly(capsys, cms, valid):
    solution = _solution(PrefetchGlobalRead=4, UseCustomMainLoopSchedule=cms)
    out = capsys.readouterr().out
    assert solution["Valid"] is valid, out
    if valid:
        assert solution["UseCustomMainLoopSchedule"] == 0
    else:
        assert ("TDM LDS ring (PrefetchGlobalRead=4): UseCustomMainLoopSchedule is not "
                "supported yet") in out


@pytest.mark.parametrize("seg, valid", [pytest.param(-1, True, id="seg-auto"),
                                        pytest.param(0, True, id="seg-off"),
                                        pytest.param(1, False, id="seg-on")])
def test_ring_lds_segment_interleave(capsys, seg, valid):
    """Auto resolves off on a ring, as the custom main-loop schedule does; 1 is rejected."""
    solution = _solution(PrefetchGlobalRead=4, LDSSegmentInterleave=seg)
    out = capsys.readouterr().out
    assert solution["Valid"] is valid, out
    if valid:
        assert solution["LDSSegmentInterleave"] == 0
    else:
        assert "LDSSegmentInterleave=1" in out


def test_ring_rejects_plr_pack_set_explicitly(capsys):
    """ScheduleIterAlg 0 resolves UsePLRPack off only after the ring check, which rejects 1."""
    solution = _solution(PrefetchGlobalRead=4, UsePLRPack=1)
    out = capsys.readouterr().out
    assert solution["Valid"] is False
    assert "TDM LDS ring (PrefetchGlobalRead=4): UsePLRPack is not supported yet" in out


def test_pgr3_without_tdm_keeps_the_direct_to_lds_rule(capsys):
    solution = _solution(PrefetchGlobalRead=3, TDMInst=0)
    out = capsys.readouterr().out
    assert solution["Valid"] is False
    assert "PrefetchGlobalRead>=3 Supports only DirectToLdsA and DirectToLdsB" in out
    assert "TDM LDS ring" not in out
