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


def test_dtvb_wide_b_shares_one_offset(toolchain):
    """A B vector wider than one load (32 halves) is four b128 loads stepping
    through the vector with the instruction offset."""
    sol = _solution(toolchain, **_dtvb(glvwA=32, depthU=1024))
    assert sol.get("Valid") is True
    body = _loopBody(_emit_solution(toolchain, sol)[1])
    loads = re.findall(r"buffer_load_b128 v\[vgprG2LB\+(\d+):\S+, v\[vgprGlobalReadOffsetB\+(\d+)\], "
                       r"\S+, 0 offen offset:(\d+)", body)
    assert loads == [(str(4 * i), "0", str(16 * i)) for i in range(4)]


def test_dtvb_wide_b_limit(toolchain):
    assert _solution(toolchain, **_dtvb(glvwA=32, depthU=1024)).get("Valid") is True
    # 64 halves (128 bytes) per vector is still too wide.
    assert _solution(toolchain, **_dtvb(glvwA=64, depthU=2048)).get("Valid") is not True
    # Without DirectToVgprDot2B the generic 24-byte limit stays.
    assert _solution(toolchain, pgr=0, glvwA=32, depthU=1024,
                     GlobalReadVectorWidthB=32).get("Valid") is not True


@pytest.mark.parametrize("nt, mods", [(0, ""), (2, " slc"), (6, " slc dlc")])
def test_non_temporal_a_streams_with_dlc(toolchain, nt, mods):
    """gfx11 has no nt modifier; the nt bit of NonTemporalA becomes dlc."""
    sol = _solution(toolchain, **_dtvb(glvwA=32, depthU=1024, NonTemporalA=nt))
    assert sol["NonTemporalA"] == nt
    body = _loopBody(_emit_solution(toolchain, sol)[1])
    loads = re.findall(r"buffer_load_b128 v\[vgprG2LA\+\d+:\S+, \S+, \S+, 0 offen offset:0([a-z ]*?) //", body)
    assert loads and set(loads) == {mods}


def test_wave_split_k_skips_zero_quotient_division(toolchain):
    """The 1D workgroup index is split without dividing when the quotient is 0."""
    src = _emit_solution(toolchain, _solution(toolchain, **_dtvb()))[1]
    for name in ("RemapWG2Done", "RemapWG1Done"):
        assert re.search(r"s_cbranch_scc1 label_%s" % name, src)


@pytest.mark.parametrize("tile0", [1, 3])
def test_zero_point_tile_below_eight_rows(toolchain, tile0):
    """A zero-point word holds eight rows; a smaller tile starts its SRD at the word
    of its first row and adds the row's position within that word."""
    sol = _solution(toolchain, zeroPoint=True, problemType={"ScaleBlockSizesA": [32]},
                    **_dtvb(glvwA=32, depthU=1024, WorkGroup=[1, 1, 32], ThreadTile=[tile0, 1]))
    assert sol.get("Valid") is True
    src = _emit_solution(toolchain, sol)[1]
    assert "scaleZeroA: row origin to words" in src
    assert "scaleZeroA: origin row within its word" in src


def test_block_scale_has_no_runtime_stagger_or_user_gsu(toolchain):
    """The scale pointer follows neither a staggered K start nor a split K, so the
    host must not request them and the kernel carries no stagger code."""
    sol = _solution(toolchain, **_dtvb())
    assert sol["InternalSupportParams"]["SupportCustomStaggerU"] is False
    assert sol["InternalSupportParams"]["SupportUserGSU"] is False
    src = _emit_solution(toolchain, sol)[1]
    assert "StaggerUIter" not in src


def test_dtvb_sets_up_no_lds_addresses(toolchain):
    src = _emit_solution(toolchain, _solution(toolchain, **_dtvb()))[1]
    assert "local read addresses: final offsets" not in src
    assert "local write addresses: first offset" not in src
    # A alone bypassing LDS still needs B's addresses.
    src = _emit_solution(toolchain, _solution(toolchain, pgr=0, glvwA=8, ThreadTile=[1, 1]))[1]
    assert "local read addresses: final offsets b" in src


def _prologue(src):
    return src[src.index("label_ASM_Start:"):src.index("label_openLoopL:")]


def _epilogue(src):
    return src[src.index("label_LoopEndL"):src.index("label_KernelEnd:")]


def test_short_wave_rare_paths_out_of_line(toolchain, tmp_path):
    """gfx11 WaveSplitK waves run the common case (ArgType 0, WGM 1, first tile row,
    alpha != 0) as straight-line code; the rare paths sit after s_endpgm and still
    assemble there."""
    from Tensile.Toolchain.Component import Assembler

    _isa, _iim, _assembler, cxx = toolchain
    src = _emit_solution(toolchain, _solution(toolchain, **_dtvb(WorkGroupMapping=1)))[1]
    assert not re.search(r"^s_branch |s_cbranch_vccz", _prologue(src), re.M)
    # Only the stores jump, to their common end.
    assert set(re.findall(r"^s_branch (\w+)", _epilogue(src), re.M)) == {"label_GW_End"}
    cold = src[src.index("label_KernelEnd:"):]
    for label in ("HBMArgs", "WGMXCC", "MultiGemm", "RemapWG", "WGMRemap",
                  "GeneralBatchedGemmLoadA", "GeneralBatchedGemmLoadB",
                  "GeneralBatchedGemmLoadC", "GeneralBatchedGemmLoadD"):
        assert re.search(r"^label_%s:" % label, cold, re.M), label
    # Cold blocks restate the SGPR names that the loop's UNDEFs dropped.
    assert re.search(r"^\.set sgprAddressA, \d+", cold, re.M)
    srcPath = tmp_path / "k.s"
    srcPath.write_text(src)
    Assembler(cxx, "5")("gfx1151", 32, str(srcPath), str(tmp_path / "k.o"))
    assert (tmp_path / "k.o").exists()


def test_short_wave_static_gsu(toolchain):
    """Block scales disable user GSU, so GSU is 1 and its runtime checks go."""
    src = _emit_solution(toolchain, _solution(toolchain, **_dtvb()))[1]
    body = src[:src.index("label_KernelEnd:")]
    assert not re.search(r",\s*s\[sgprGSU\]", body)  # never read
    assert "label_GW_B0_MB" not in src  # no GSU > 1 store


@pytest.mark.parametrize("tile0", [1, 2, 4])
def test_single_wave_rows_are_constant(toolchain, tile0):
    """One wave covers K, so its tile rows are compile-time constants: no per-row
    v_mul_lo_u32 in the global-read or store offsets."""
    sol = _solution(toolchain, **_dtvb(glvwA=16, depthU=512, ThreadTile=[tile0, 1], WorkGroupMapping=1))
    src = _emit_solution(toolchain, sol)[1]
    assert "v_mul_lo_u32" not in _prologue(src)
    assert "v_mul_lo_u32" not in _epilogue(src)
    assert re.search(r"v_add_nc_u32 v\[vgprGlobalReadOffsetA\+0\], 0x10, v\d+\s+// gROA_0: unroll \+ prepad", src)
    for row in range(1, tile0):
        assert "gROA_%u: + unroll" % row in src
