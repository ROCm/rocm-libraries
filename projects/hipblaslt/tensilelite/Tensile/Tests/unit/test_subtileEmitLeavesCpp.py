#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression tests for the C++ (nanobind) subtile *emit-leaf* path.

These cover the smallest rocisa-adjacent emit leaves now backed unconditionally
by the ``tensile_writer.subtile`` extension for the supported AB cases:

  * ``Kernel._selectF8F6F4InstType`` / ``Kernel.emitMfmaInstruction`` — the MFMA
    F8F6F4 instType selection (``tensile_writer.subtile.emit.mfma_f8f6f4_inst_type``).
  * ``SubtileGREmit.emitSingleBufferLoad`` — the buffer-load instruction-shape
    plan (``ABTileInfoQuery.singleBufferLoadPlan``).
  * ``SubtileLREmit.emitSingleDsRead`` — the ds_read instruction-shape plan
    (``ABTileInfoQuery.singleDsReadPlan``).

There is no longer a parallel Python formula for these ported AB cases, so the
tests lock the produced *values* against the documented reference math (golden
oracle) and assert the MFMA emission stays well-formed. They run only when the
compiled extension is importable (it is a hard dependency of Kernel.py);
otherwise they skip.

PR creation for this slice is human-only: a ``human:pr`` task is filed for
Bryant Nelson only after review says merge-ready. Agents never open PRs.
"""

import math
import os
import sys
from types import SimpleNamespace

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

# Both the ISA layer (rocisa) and the compiled emit/query layers must exist.
pytest.importorskip("rocisa")
cppemit = pytest.importorskip("tensile_writer.subtile.emit")
cppti = pytest.importorskip("tensile_writer.subtile.tile_info")
cppgeo = pytest.importorskip("tensile_writer.subtile.geometry")

from rocisa.enum import InstType

from Tensile.Common.DataType import DataType
from Tensile.Components.Subtile.Kernel import (
    TileInfo,
    emitMfmaInstruction,
    _selectF8F6F4InstType,
    AB_B16,
    AB_B8,
    AB_B4,
    AB_B16_2x2,
    AB_B4_2x2,
)


def _init_rocisa_gfx950():
    """Pin rocisa to gfx950 (wave64) for deterministic MFMA string emission.

    Inlined rather than imported from gpu_test_helpers so this pure-string test
    has no GPU-runtime (hip) import dependency.
    """
    import shutil
    from rocisa import rocIsa
    from Tensile.Common.Architectures import gfxToIsa
    ri = rocIsa.getInstance()
    isa = gfxToIsa("gfx950")
    asmpath = shutil.which("amdclang++") or "/usr/bin/amdclang++"
    ri.init(isa, asmpath)
    ri.setKernel(isa, 64)


# Plan geometries with a (MatrixInstK, DepthU) that builds a valid TileInfo for
# the 128x128 / MIWaveGroup=[2,2] kernel below. DepthU is chosen per layout so
# depthU == subtileShape[1] * mmaK * globalSubtileGrid[1] (the
# TileInfo._check_dim coverage constraint): bf16 (instK=32) -> 64, fp8
# (instK=128) -> 128, fp4 (instK=128, bpe=0.5) -> 256.
PLAN_GEOMS = {
    "AB_B16": (AB_B16, 32, 64),
    "AB_B8": (AB_B8, 128, 128),
    "AB_B4": (AB_B4, 128, 256),
    "AB_B16_2x2": (AB_B16_2x2, 32, 64),
    "AB_B4_2x2": (AB_B4_2x2, 128, 256),
}


def _mk_plan_kernel(inst_k, depth_u):
    """Minimal kernel dict that builds a valid ABTilePair TileInfo.

    Mirrors gpu_test_helpers._create_kernel for a 128x128 macro tile with
    MIWaveGroup=[2,2] (the wave grouping that divides both M tiles evenly),
    but defined inline so this test keeps no hip / GPU-runtime dependency.
    """
    return {
        "DepthU": depth_u,
        "_DepthU": depth_u,
        "_DepthUA": depth_u,
        "_DepthUB": depth_u,
        "MacroTileA": 128,
        "MacroTileB": 128,
        "MacroTile0": 128,
        "MacroTile1": 128,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": inst_k,
        "MIWaveGroup": [2, 2],
        "WavefrontSize": 64,
        "UseSubtileImpl": True,
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "ProblemType": {},
    }


@pytest.fixture(scope="module", autouse=True)
def _rocisa_once():
    # Pure string tests: pin rocisa to gfx950 so the MFMA emission path matches
    # the assertions regardless of the host GPU.
    _init_rocisa_gfx950()


# ---------------------------------------------------------------------------
# emitMfmaInstruction — instType selection (now C++-only).
# ---------------------------------------------------------------------------
class _StubPool:
    def __init__(self):
        self._next = 200

    def checkOut(self, n=1, *a, **kw):
        v = self._next
        self._next += n
        return v

    def checkIn(self, *a, **kw):
        pass


def _mkTile(start, count, pool):
    return SimpleNamespace(
        regList=SimpleNamespace(indices=list(range(start, start + count)), pool=pool)
    )


def _mkKernel(dA, dB, miK=128, sourceSwap=False, miArchVgpr=True):
    return {
        "MatrixInstK": miK,
        "MIArchVgpr": miArchVgpr,
        "SourceSwap": sourceSwap,
        "_subtileUnitScaleVgpr": 250,
        "ProblemType": {
            "DataTypeA": DataType(dA) if dA else None,
            "DataTypeB": DataType(dB) if dB else None,
            "MXBlockA": 0,
            "MXBlockB": 0,
        },
    }


# (DataTypeA, DataTypeB, sourceSwap) covering pure + mixed F8/BF8/F4 cases.
MFMA_CASES = [
    ("F8", "F8", False),
    ("F8", "F8", True),
    ("B8", "B8", False),
    ("F8", "B8", False),
    ("F8", "B8", True),
    ("B8", "F8", False),
    ("F4", "F4", False),
    ("F8", "F4", False),
    ("F8", "F4", True),
    ("F4", "F8", False),
    ("B8", "F4", False),
    ("F4", "B8", True),
]

# Golden reference for the F8F6F4 instType mapping (matches the C++ port and the
# prior pure-Python branches). SourceSwap swaps the operand formats first.
_FMT = {"F8": "f8", "B8": "bf8", "F4": "f4"}
_INST_TABLE = {
    ("f8", "f8"): "INST_F8",
    ("bf8", "bf8"): "INST_BF8",
    ("f4", "f4"): "INST_F4",
    ("f8", "bf8"): "INST_F8_BF8",
    ("bf8", "f8"): "INST_BF8_F8",
    ("f8", "f4"): "INST_F8_F4",
    ("f4", "f8"): "INST_F4_F8",
    ("bf8", "f4"): "INST_B8_F4",
    ("f4", "bf8"): "INST_F4_B8",
}


def _expected_inst_type(dA, dB, swap):
    a, b = _FMT[dA], _FMT[dB]
    if swap:
        a, b = b, a
    return getattr(InstType, _INST_TABLE[(a, b)])


@pytest.mark.parametrize("dA,dB,swap", MFMA_CASES)
def test_select_f8f6f4_inst_type(dA, dB, swap):
    """The C++-backed instType selection must return the golden InstType."""
    kernel = _mkKernel(dA, dB, miK=128, sourceSwap=swap)
    assert _selectF8F6F4InstType(kernel) == _expected_inst_type(dA, dB, swap)


@pytest.mark.parametrize("dA,dB,swap", MFMA_CASES)
def test_emitMfmaInstruction_emits_asm(dA, dB, swap):
    """emitMfmaInstruction renders a non-empty MFMA module for each covered
    F8/F6/F4 case using the C++-backed instType selection."""
    writer = SimpleNamespace(vgprPool=_StubPool(), agprPool=_StubPool())
    aWidth = 8 if dA in ("F8", "B8") else 4
    bWidth = 8 if dB in ("F8", "B8") else 4
    kernel = _mkKernel(dA, dB, miK=128, sourceSwap=swap)
    tA = _mkTile(0, aWidth, writer.vgprPool)
    tB = _mkTile(16, bWidth, writer.vgprPool)
    tC = _mkTile(32, 4, writer.vgprPool)
    tD = _mkTile(64, 4, writer.vgprPool)

    asm = str(emitMfmaInstruction(
        writer, kernel, tA, tB, tC, tD,
        scaleAVgpr=100, scaleBVgpr=101, scaleAsel=2, scaleBsel=1))
    assert asm.strip(), f"empty MFMA asm for A={dA} B={dB} swap={swap}"
    assert "mfma" in asm.lower(), f"no mfma mnemonic for A={dA} B={dB} swap={swap}"


def test_mfma_inst_type_unsupported_raises():
    """Unsupported predicate combinations raise, e.g. all-false predicates."""
    with pytest.raises(Exception):
        cppemit.mfma_f8f6f4_inst_type(False, False, False, False, False, False, False)


# ---------------------------------------------------------------------------
# emitSingleBufferLoad / emitSingleDsRead — instruction-shape plan values.
#
# The plans are computed by ABTileInfoQuery (no Python twin); lock them against
# the documented reference math derived from the Python TileInfo state.
# ---------------------------------------------------------------------------
def _iter_subtiles(ti):
    for s0 in range(int(ti.localSubtileGrid[0])):
        for s1 in range(int(ti.localSubtileGrid[1])):
            yield s0, s1


def _ref_single_buffer_load_plan(ti, s0, s1):
    """Reference buffer-load plan (skip flag, offsetK, m0 offsets)."""
    linearId = s1 * ti.localSubtileGrid[0] + s0
    grBaseId = int(math.floor(linearId / ti.loadRatioGR)) if ti.loadRatioGR else 0
    if ti.loadRatioGR > 1:
        firstInGroup = int(grBaseId * ti.loadRatioGR)
        if linearId != firstInGroup:
            return SimpleNamespace(skip=True, grBaseId=grBaseId, offsetK=0, m0Offsets=[])
    offsetK = s1 * int(ti.mmaTileShape[1] * ti.subtileShape[1] * ti.bpe)
    subtileOffset = int(math.ceil(ti.loadRatioGR * ti.subtileSize))
    m0Offsets = [
        int(i * subtileOffset
            + (s0 + s1 * ti.globalSubtileGrid[0]) * ti.subtileSize)
        for i in range(ti.numGRPerSubtile)
    ]
    return SimpleNamespace(skip=False, grBaseId=grBaseId, offsetK=offsetK, m0Offsets=m0Offsets)


def _ref_single_ds_read_plan(ti, s0, s1, subIterK, numRegs):
    """Reference ds_read plan (DS offset, register stride, per-read map)."""
    regsPerDsRead = ti.loadWidthLR // 4
    mfmaId = ti.getSubtileShapeLinearId(subIterK, 0)
    offsetStride = int(ti.subtileSize)
    offset = s0 * offsetStride + s1 * int(ti.globalSubtileGrid[0]) * offsetStride
    numReadsForTile = numRegs // regsPerDsRead
    reads = [(r * regsPerDsRead, mfmaId * numReadsForTile + r)
             for r in range(numReadsForTile)]
    return SimpleNamespace(regsPerDsRead=regsPerDsRead, mfmaId=mfmaId,
                           offset=offset, numReadsForTile=numReadsForTile, reads=reads)


@pytest.mark.parametrize("name", list(PLAN_GEOMS))
def test_single_buffer_load_plan_values(name):
    """ABTileInfoQuery.singleBufferLoadPlan must match the reference plan
    (skip flag, offsetK, m0 offsets) for every local subtile."""
    geom, inst_k, depth_u = PLAN_GEOMS[name]
    kernel = _mk_plan_kernel(inst_k, depth_u)
    for tc in ("A", "B"):
        ti = TileInfo(geom, tc, None, kernel)
        for s0, s1 in _iter_subtiles(ti):
            got = ti.singleBufferLoadPlan(s0, s1)
            ref = _ref_single_buffer_load_plan(ti, s0, s1)
            ctx = f"{name}/{tc} subtile ({s0},{s1})"
            assert got.skip == ref.skip, f"{ctx}: skip {got.skip} vs {ref.skip}"
            if ref.skip:
                continue
            assert got.grBaseId == ref.grBaseId, f"{ctx}: grBaseId"
            assert got.offsetK == ref.offsetK, f"{ctx}: offsetK"
            assert list(got.m0Offsets) == list(ref.m0Offsets), f"{ctx}: m0Offsets"


@pytest.mark.parametrize("name", list(PLAN_GEOMS))
@pytest.mark.parametrize("numRegs", [4, 8])
def test_single_ds_read_plan_values(name, numRegs):
    """ABTileInfoQuery.singleDsReadPlan must match the reference plan
    (DS offset, register stride, per-read map) for every subtile / subIterK."""
    geom, inst_k, depth_u = PLAN_GEOMS[name]
    kernel = _mk_plan_kernel(inst_k, depth_u)
    for tc in ("A", "B"):
        ti = TileInfo(geom, tc, None, kernel)
        for s0, s1 in _iter_subtiles(ti):
            for subIterK in range(int(ti.lrSubtileShape[1])):
                got = ti.singleDsReadPlan(s0, s1, subIterK, numRegs)
                ref = _ref_single_ds_read_plan(ti, s0, s1, subIterK, numRegs)
                ctx = f"{name}/{tc} ({s0},{s1}) k={subIterK} nr={numRegs}"
                assert got.regsPerDsRead == ref.regsPerDsRead, f"{ctx}: regsPerDsRead"
                assert got.mfmaId == ref.mfmaId, f"{ctx}: mfmaId"
                assert got.offset == ref.offset, f"{ctx}: offset"
                assert got.numReadsForTile == ref.numReadsForTile, f"{ctx}: numReads"
                got_reads = [(r.dstRegOffset, r.addrIdx) for r in got.reads]
                assert got_reads == ref.reads, f"{ctx}: reads {got_reads} vs {ref.reads}"
