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


def _loopBody(src):
    begin = src.index("label_LoopBeginL:")
    return src[begin:src.index("label_LoopEndL", begin)]


def test_slots_valid(toolchain):
    """Several A loads per lane ("slots") along K: 4 x 8 int4 at DepthU 1024."""
    sol = _solution(toolchain, pgr=0, glvwA=8)
    assert sol.get("Valid") is True
    assert sol["NumLoadsCoalescedA"] == 4


def test_slots_dequant_behind_own_wait(toolchain):
    """PGR0 consumes A slots in issue order: each slot waits only for itself, with
    the later slots still in flight, and B is not held back by A."""
    sol = _solution(toolchain, pgr=0, glvwA=8, ThreadTile=[1, 1])
    body = _loopBody(_emit_solution(toolchain, sol)[1])
    waits = [int(n) for n in re.findall(r"s_waitcnt vmcnt\((\d+)\)\s+// A slot \d+ loaded", body)]
    assert waits == [3, 2, 1, 0]
    # B's LDS store leaves the four A loads outstanding.
    assert re.search(r"s_waitcnt vmcnt\(4\)\s+// 5wait for global read B", body)
    # Scales are issued before A, so an A slot's wait does not cover later A loads.
    kinds = re.findall(r"buffer_load_\w+ v\[vgpr(G2LScaleA|G2LA)\b", body)
    assert kinds.index("G2LA") > max(i for i, k in enumerate(kinds) if k == "G2LScaleA")


def _dtvb(**kw):
    kw.setdefault("pgr", 0)
    kw.setdefault("glvwA", 8)
    kw.setdefault("ThreadTile", [1, 1])
    kw.setdefault("GlobalReadVectorWidthB", kw["glvwA"])
    return dict(kw, DirectToVgprDot2B=True)


def test_dtvb_valid(toolchain):
    assert _solution(toolchain, **_dtvb()).get("Valid") is True


@pytest.mark.parametrize(
    "kw",
    [
        {"WorkGroup": [2, 1, 32]},   # B would be spread over two row groups
        {"pgr": 1},
        {"GlobalReadVectorWidthB": 4},
    ],
)
def test_dtvb_rejects_unsupported(toolchain, kw):
    assert _solution(toolchain, **_dtvb(**kw)).get("Valid") is not True


def test_dtvb_no_lds(toolchain):
    body = _loopBody(_emit_solution(toolchain, _solution(toolchain, **_dtvb()))[1])
    assert not re.search(r"\bds_(store|load|write|read)", body)
    dots = re.findall(r"v_dot2_f32_f16 \S+, v\[vgprG2LA\+\d+\], v\[vgprG2LB\+\d+\]", body)
    assert len(dots) == 16
