#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for DirectToVgprDot2A: dot2 WaveSplitK kernels whose v_dot2 read
A straight from the global-read registers (dequantized in place), while only B
goes through LDS."""

import re

import pytest

from test_w4a16_block_dequant import _emit_solution, toolchain  # noqa: F401  (fixture)

pytestmark = pytest.mark.unit


def _dot2_params(isa, *, pgr=1, depthU=1024, nwsk=32, glvwA=None, zeroPoint=False, **overrides):
    problemType = {
        "OperationType": "GEMM",
        "DataType": "H",
        "DataTypeA": "I4",
        "MacDataTypeA": "H",
        "DestDataType": "H",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "StridedBatched": True,
        "UseScaleAB": "Block",
        "ScaleBlockSizesA": [128],
        "ScaleZeroPointA": zeroPoint,
        "Int4EncodingA": "UnsignedBias8",
    }
    problemType.update(overrides.pop("problemType", {}))
    params = {
        "ProblemType": problemType,
        "ISA": isa,
        "MatrixInstruction": [],
        "EnableMatrixInstruction": False,
        "WorkGroup": [8 if zeroPoint else 1, 1, nwsk],
        "ThreadTile": [4, 1],
        "WaveSplitK": True,
        "WavefrontSize": 32,
        "DepthU": depthU,
        "KernelLanguage": "Assembly",
        "PrefetchGlobalRead": pgr,
        "PrefetchLocalRead": 1,
        "ScheduleIterAlg": 1,
        "ScheduleLocalWrite": 1,
        "StaggerU": 0,
        "GlobalSplitU": 1,
        "GlobalSplitUAlgorithm": "MultipleBuffer",
        "InnerUnroll": 1,
        "TransposeLDS": 2,
        "VectorWidthA": 1,
        "VectorWidthB": 1,
        "StoreVectorWidth": 1,
        "GlobalReadVectorWidthA": glvwA if glvwA is not None else depthU // nwsk,
        "GlobalReadVectorWidthB": -1,
        "LocalReadVectorWidth": -1,
        "DirectToVgprA": False,
        "DirectToVgprB": False,
        "DirectToVgprDot2A": True,
        "AssertSummationElementMultiple": depthU,
    }
    params.update(overrides)
    return params


def _solution(toolchain, **kw):  # noqa: F811
    from Tensile.SolutionStructs.Solution import Solution

    isa, iim, assembler, _cxx = toolchain
    return Solution(_dot2_params(isa, **kw), False, True, False, assembler, iim)


@pytest.mark.parametrize("pgr", [0, 1])
@pytest.mark.parametrize("zeroPoint", [False, True])
def test_valid(toolchain, pgr, zeroPoint):
    sol = _solution(toolchain, pgr=pgr, zeroPoint=zeroPoint)
    assert sol.get("Valid") is True


@pytest.mark.parametrize(
    "kw",
    [
        {"glvwA": 16},                         # one load no longer covers the lane's K block
        {"WorkGroup": [1, 2, 32]},             # A rows would be split across N-waves
        {"pgr": 2},                            # would need a third A register set
        {"InnerUnroll": 2},
        {"AssertSummationElementMultiple": 256},  # tail loop reads A through LDS
    ],
)
def test_rejects_unsupported(toolchain, kw):
    assert _solution(toolchain, **kw).get("Valid") is not True


def test_name_unchanged_when_off(toolchain):
    """The new parameter is only named when set, so existing kernel names stay."""
    from Tensile.SolutionStructs.Naming import getKernelNameMin

    sol = _solution(toolchain, DirectToVgprDot2A=False, PrefetchGlobalRead=1)
    assert "DTVDA" not in getKernelNameMin(sol, False)
    sol = _solution(toolchain)
    assert "DTVDA1" in getKernelNameMin(sol, False)


@pytest.mark.parametrize("pgr", [0, 1])
def test_a_bypasses_lds(toolchain, pgr):
    sol = _solution(toolchain, pgr=pgr)
    _kernel, src = _emit_solution(toolchain, sol)
    # A is neither written to nor read from LDS ...
    assert not re.search(r"ds_(store|write)\w* \S+, v\[vgprG2LA", src)
    assert not re.search(r"ds_(load|read)\w* v\[vgprValuA", src)
    # ... the v_dot2 read it from the global-read registers,
    dots = re.findall(r"v_dot2_f32_f16 \S+, v\[(vgprG2LA2?)\+\d+\]", src)
    assert dots
    # and with one load in flight both register sets are used.
    assert set(dots) == ({"vgprG2LA", "vgprG2LA2"} if pgr else {"vgprG2LA"})


def test_dequant_waits_for_its_load(toolchain):
    """A's dequant items carry no LDS store, yet each needs the vmcnt wait that the
    scheduler otherwise attaches to a local write."""
    _kernel, src = _emit_solution(toolchain, _solution(toolchain, pgr=1))
    begin = src.index("label_LoopBeginL:")
    body = src[begin:src.index("label_LoopEndL_evenexit:", begin)]
    dequants = list(re.finditer(r"w4a16: int4 #0 and #4", body))
    assert dequants, "no dequant in the main loop"
    for m in dequants:
        before = body[:m.start()]
        lastLoad = before.rfind("buffer_load")
        lastWait = before.rfind("wait for global read before writing to local")
        assert lastWait > lastLoad, "dequant reads A before its load is waited for"
