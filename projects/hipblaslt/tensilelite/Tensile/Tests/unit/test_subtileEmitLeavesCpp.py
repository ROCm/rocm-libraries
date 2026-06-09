#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Parity tests for the optional C++ (nanobind) subtile *emit-leaf* path.

These tests cover the smallest rocisa-adjacent emit leaves ported to the
``tensile_writer.subtile`` extension:

  * ``Kernel.emitMfmaInstruction`` — the MFMA F8F6F4 instType selection
    (``tensile_writer.subtile.emit.mfma_f8f6f4_inst_type``).
  * ``SubtileGREmit.emitSingleBufferLoad`` — the buffer-load instruction-shape
    plan (``ABTileInfoQuery.singleBufferLoadPlan``).
  * ``SubtileLREmit.emitSingleDsRead`` — the ds_read instruction-shape plan
    (``ABTileInfoQuery.singleDsReadPlan``).

They run only when the compiled extension is importable; otherwise they skip,
so the default (Python-only) TensileLite build is unaffected. The C++ path is
exercised by flipping the SubtileGeometry delegation switch at call time, so the
*same* inputs are run through both code paths and asserted equivalent.

PR creation for this slice is human-only: a ``human:pr`` task is filed for
Bryant Nelson only after review says merge-ready. Agents never open PRs.
"""

import contextlib
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

from Tensile.Common.DataType import DataType
from Tensile.Components.Subtile import SubtileGeometry as sg
from Tensile.Components.Subtile import Kernel as krn
from Tensile.Components.Subtile.Kernel import (
    TileInfo,
    emitMfmaInstruction,
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


# Plan-parity geometries with a (MatrixInstK, DepthU) that builds a valid
# TileInfo for the 128x128 / MIWaveGroup=[2,2] kernel below. DepthU is chosen
# per layout so depthU == subtileShape[1] * mmaK * globalSubtileGrid[1] (the
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


@contextlib.contextmanager
def cpp_delegation():
    """Temporarily enable C++ delegation for geometry + emit leaves.

    Sets all three switches the delegated paths read: ``sg._USE_CPP`` /
    ``sg._CPP`` gate the geometry ``_cpp_twin`` (reused by the TileInfo query
    layer for the buffer-load / ds-read plans) and ``krn._CPP_EMIT`` gates the
    MFMA instType selection.
    """
    saved_use, saved_cpp = sg._USE_CPP, sg._CPP
    saved_emit = krn._CPP_EMIT
    sg._USE_CPP = True
    sg._CPP = cppgeo
    krn._CPP_EMIT = cppemit
    try:
        yield
    finally:
        sg._USE_CPP = saved_use
        sg._CPP = saved_cpp
        krn._CPP_EMIT = saved_emit


# ---------------------------------------------------------------------------
# emitMfmaInstruction — instType selection delegated to C++.
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


@pytest.mark.parametrize("dA,dB,swap", MFMA_CASES)
def test_emitMfmaInstruction_cpp_matches_python(dA, dB, swap):
    """The C++-delegated instType selection must produce byte-identical MFMA
    assembly to the native Python path for all covered F8/F6/F4 cases."""
    writer = SimpleNamespace(vgprPool=_StubPool(), agprPool=_StubPool())
    aWidth = 8 if dA in ("F8", "B8") else 4
    bWidth = 8 if dB in ("F8", "B8") else 4
    kernel = _mkKernel(dA, dB, miK=128, sourceSwap=swap)
    tA = _mkTile(0, aWidth, writer.vgprPool)
    tB = _mkTile(16, bWidth, writer.vgprPool)
    tC = _mkTile(32, 4, writer.vgprPool)
    tD = _mkTile(64, 4, writer.vgprPool)

    args = (writer, kernel, tA, tB, tC, tD)
    kwargs = dict(scaleAVgpr=100, scaleBVgpr=101, scaleAsel=2, scaleBsel=1)

    asm_py = str(emitMfmaInstruction(*args, **kwargs))
    with cpp_delegation():
        asm_cpp = str(emitMfmaInstruction(*args, **kwargs))
    assert asm_py == asm_cpp, (
        f"MFMA asm mismatch for A={dA} B={dB} swap={swap}:\n"
        f"PY:\n{asm_py}\nCPP:\n{asm_cpp}"
    )


def test_mfma_inst_type_unsupported_raises():
    """Unsupported predicate combinations raise (so the Python caller can fall
    back), e.g. all-false predicates."""
    with pytest.raises(Exception):
        cppemit.mfma_f8f6f4_inst_type(False, False, False, False, False, False, False)


# ---------------------------------------------------------------------------
# emitSingleBufferLoad / emitSingleDsRead — instruction-shape plan parity.
# ---------------------------------------------------------------------------
def _iter_subtiles(ti):
    for s0 in range(int(ti.localSubtileGrid[0])):
        for s1 in range(int(ti.localSubtileGrid[1])):
            yield s0, s1


@pytest.mark.parametrize("name", list(PLAN_GEOMS))
def test_single_buffer_load_plan_parity(name):
    """ABTileInfoQuery.singleBufferLoadPlan must match the Python TileInfo plan
    (skip flag, offsetK, m0 offsets) for every local subtile."""
    geom, inst_k, depth_u = PLAN_GEOMS[name]
    kernel = _mk_plan_kernel(inst_k, depth_u)
    for tc in ("A", "B"):
        ti = TileInfo(geom, tc, None, kernel)
        for s0, s1 in _iter_subtiles(ti):
            py = ti.singleBufferLoadPlan(s0, s1)
            with cpp_delegation():
                cpp = ti.singleBufferLoadPlan(s0, s1)
            ctx = f"{name}/{tc} subtile ({s0},{s1})"
            assert py.skip == cpp.skip, f"{ctx}: skip {py.skip} vs {cpp.skip}"
            if py.skip:
                continue
            assert py.grBaseId == cpp.grBaseId, f"{ctx}: grBaseId"
            assert py.offsetK == cpp.offsetK, f"{ctx}: offsetK"
            assert list(py.m0Offsets) == list(cpp.m0Offsets), f"{ctx}: m0Offsets"


@pytest.mark.parametrize("name", list(PLAN_GEOMS))
@pytest.mark.parametrize("numRegs", [4, 8])
def test_single_ds_read_plan_parity(name, numRegs):
    """ABTileInfoQuery.singleDsReadPlan must match the Python TileInfo plan
    (DS offset, register stride, per-read map) for every subtile / subIterK."""
    geom, inst_k, depth_u = PLAN_GEOMS[name]
    kernel = _mk_plan_kernel(inst_k, depth_u)
    for tc in ("A", "B"):
        ti = TileInfo(geom, tc, None, kernel)
        for s0, s1 in _iter_subtiles(ti):
            for subIterK in range(int(ti.lrSubtileShape[1])):
                py = ti.singleDsReadPlan(s0, s1, subIterK, numRegs)
                with cpp_delegation():
                    cpp = ti.singleDsReadPlan(s0, s1, subIterK, numRegs)
                ctx = f"{name}/{tc} ({s0},{s1}) k={subIterK} nr={numRegs}"
                assert py.regsPerDsRead == cpp.regsPerDsRead, f"{ctx}: regsPerDsRead"
                assert py.mfmaId == cpp.mfmaId, f"{ctx}: mfmaId"
                assert py.offset == cpp.offset, f"{ctx}: offset"
                assert py.numReadsForTile == cpp.numReadsForTile, f"{ctx}: numReads"
                py_reads = [(r.dstRegOffset, r.addrIdx) for r in py.reads]
                cpp_reads = [(r.dstRegOffset, r.addrIdx) for r in cpp.reads]
                assert py_reads == cpp_reads, f"{ctx}: reads {py_reads} vs {cpp_reads}"
