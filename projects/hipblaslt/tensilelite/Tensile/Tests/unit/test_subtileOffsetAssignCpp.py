#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Parity tests for the optional C++ (nanobind) subtile GR/LR *offset
assignment* path (B16 / TLU0).

``SubtileGREmit.graTileAssignment`` and ``SubtileLREmit.lraTileAssignment``
derive the offset-assignment scalar math (block/row/partition sizes, advance
and rotation strides, the subtile soffset stride) before emitting the rocisa
offset-calculation instructions. That math is ported to the C++
``ABTileInfoQuery.grOffsetAssignPlan`` / ``lrOffsetAssignPlan``; the rocisa
emission stays in Python.

These tests assert that, for the BF16 row-major (TLU0) path, the C++-delegated
emission is **byte-identical** to the native Python (legacy) emission. They are
pure-string (no GPU runtime / hip dependency): rocisa is pinned to gfx950 so the
emitted assembly is deterministic regardless of the host GPU.

GPU functional validation (Tensile/Tests/unit/test_graTileAssignment.py and
test_lraTileAssignment.py) requires gfx950 hardware and is gated separately.

PR creation for this slice is human-only: a ``human:pr`` task is filed for
Bryant Nelson only after review says merge-ready. Agents never open PRs.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

# rocisa (ISA emission) and the compiled geometry + tile_info layers must exist.
pytest.importorskip("rocisa")
cppgeo = pytest.importorskip("tensile_writer.subtile.geometry")
cppti = pytest.importorskip("tensile_writer.subtile.tile_info")

from rocisa.register import RegisterPool
from rocisa.enum import RegisterType

from Tensile.Components.Subtile import Kernel as krn
from Tensile.Components.Subtile.Kernel import TileInfo, AB_B16, AB_B8
from Tensile.Components.Subtile.SubtileGREmit import graTileAssignment
from Tensile.Components.Subtile.SubtileLREmit import lraTileAssignment

import contextlib

WAVESIZE = 64


def _init_rocisa_gfx950():
    """Pin rocisa to gfx950 (wave64) for deterministic string emission."""
    import shutil
    from rocisa import rocIsa
    from Tensile.Common.Architectures import gfxToIsa
    ri = rocIsa.getInstance()
    isa = gfxToIsa("gfx950")
    asmpath = shutil.which("amdclang++") or "/usr/bin/amdclang++"
    ri.init(isa, asmpath)
    ri.setKernel(isa, WAVESIZE)


@pytest.fixture(scope="module", autouse=True)
def _rocisa_once():
    _init_rocisa_gfx950()


@contextlib.contextmanager
def cpp_delegation():
    """Temporarily enable the C++ tile_info delegation.

    The geometry layer is always C++; this flips ``Kernel._USE_CPP`` so the
    TileInfo query layer (which the offset-assignment plans reuse) routes
    through ``ABTileInfoQuery``.
    """
    saved_use = krn._USE_CPP
    krn._USE_CPP = True
    try:
        yield
    finally:
        krn._USE_CPP = saved_use


def _mock_dtype(num_bytes=2):
    mock = MagicMock()
    mock.numBytes.return_value = num_bytes
    return mock


def _wave_group(mt_a, mt_b):
    if ((mt_a // 16) % 2 == 0) and ((mt_b // 16) % 2 == 0):
        return [2, 2]
    if ((mt_a // 16) % 2 != 0) and ((mt_b // 16) % 4 == 0):
        return [1, 4]
    if ((mt_a // 16) % 4 == 0) and ((mt_b // 16) % 2 != 0):
        return [4, 1]
    raise ValueError(f"Unsupported wave grouping for {mt_a}x{mt_b}")


def _make_kernel(mt_a, mt_b, depth_u, inst_k=32, bpe=2):
    dtype = _mock_dtype(bpe)
    return {
        "DepthU": depth_u,
        "_DepthU": depth_u,
        "_DepthUA": depth_u,
        "_DepthUB": depth_u,
        "MacroTileA": mt_a,
        "MacroTileB": mt_b,
        "MacroTile0": mt_a,
        "MacroTile1": mt_b,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": inst_k,
        "MIWaveGroup": _wave_group(mt_a, mt_b),
        "WavefrontSize": WAVESIZE,
        "UseSubtileImpl": True,
        "NonTemporalA": 0,
        "NonTemporalB": 0,
        "ProblemType": {
            "DataTypeA": dtype,
            "DataTypeB": dtype,
            "ComputeDataType": _mock_dtype(4),
        },
    }


def _build_writer(mt_a, mt_b, depth_u, geometry=None, inst_k=32, bpe=2):
    """Minimal writer + TileInfo setup (no hip dependency).

    Mirrors gpu_test_helpers.create_writer / _generate_tile_asm but inlined so
    this pure-string test keeps no GPU-runtime import.
    """
    if geometry is None:
        geometry = AB_B16
    writer = SimpleNamespace()
    writer.vgprPool = RegisterPool(0, RegisterType.Vgpr,
                                   defaultPreventOverflow=False, printRP=False)
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr,
                                   defaultPreventOverflow=False, printRP=False)
    writer.sgprs = {}
    writer.vgprPool.checkOut(1)  # v0 = Serial

    kernel = _make_kernel(mt_a, mt_b, depth_u, inst_k=inst_k, bpe=bpe)
    tileInfoA = TileInfo(geometry, 'A', writer, kernel)
    tileInfoB = TileInfo(geometry, 'B', writer, kernel)
    writer.agprPool = RegisterPool(0, RegisterType.Accvgpr,
                                   defaultPreventOverflow=False, printRP=False)
    writer.states = SimpleNamespace(
        a=SimpleNamespace(tileInfo=tileInfoA),
        b=SimpleNamespace(tileInfo=tileInfoB),
        regCaps={"MaxSgpr": 106, "MaxVgpr": 256, "PhysicalMaxVgpr": 512},
        archCaps={"LDSBankCount": 64, "LDSBankWidth": 4},
    )
    readSize = 2 * tileInfoA.subtileSize
    numASubtiles = tileInfoA.globalSubtileGrid[0] * tileInfoA.globalSubtileGrid[1]
    writer.ldsStartOffsetA = 0
    writer.ldsStartOffsetB = int(((numASubtiles * tileInfoA.subtileSize + readSize - 1) // readSize) * readSize)

    # Reserve s0-s11 (HW + strides), as the GPU helpers do, then offset regs.
    writer.sgprPool.checkOut(12)
    writer.sgprs["StrideA0I"] = 10
    writer.sgprs["StrideB1J"] = 11
    tileInfoA.allocOffsetRegisters(writer, kernel)
    tileInfoB.allocOffsetRegisters(writer, kernel)
    return writer, kernel, tileInfoA, tileInfoB


# Representative BF16 TLU0 configs covering the three loadRatioGR wave-partition
# modes (1.0: 2x2, 0.5: 1x4, 2.0: 4x1) plus a small-MT case.
B16_CONFIGS = [
    (256, 256, 64),  # MIWaveGroup [2,2]
    (80, 64, 64),    # MIWaveGroup [1,4]
    (64, 48, 64),    # MIWaveGroup [4,1]
    (16, 64, 64),    # MIWaveGroup [1,4], small MT
]


@pytest.mark.parametrize("mt_a,mt_b,depth_u", B16_CONFIGS,
                         ids=lambda c: c if isinstance(c, str) else None)
def test_gra_tile_assignment_cpp_matches_python(mt_a, mt_b, depth_u):
    writer, kernel, _, _ = _build_writer(mt_a, mt_b, depth_u)
    asm_py = str(graTileAssignment(writer, kernel, useSwizzling=True))
    with cpp_delegation():
        asm_cpp = str(graTileAssignment(writer, kernel, useSwizzling=True))
    assert asm_py == asm_cpp, (
        f"GR offset-assignment asm mismatch for {mt_a}x{mt_b}x{depth_u}:\n"
        f"PY:\n{asm_py}\nCPP:\n{asm_cpp}"
    )


@pytest.mark.parametrize("mt_a,mt_b,depth_u", B16_CONFIGS,
                         ids=lambda c: c if isinstance(c, str) else None)
def test_lra_tile_assignment_cpp_matches_python(mt_a, mt_b, depth_u):
    writer, kernel, _, _ = _build_writer(mt_a, mt_b, depth_u)
    asm_py = str(lraTileAssignment(writer, kernel))
    with cpp_delegation():
        asm_cpp = str(lraTileAssignment(writer, kernel))
    assert asm_py == asm_cpp, (
        f"LR offset-assignment asm mismatch for {mt_a}x{mt_b}x{depth_u}:\n"
        f"PY:\n{asm_py}\nCPP:\n{asm_cpp}"
    )


def test_fp8_stays_on_python_fallback():
    """FP8 (bpe == 1) must NOT take the C++ offset-assignment path even with
    delegation enabled — its swizzle differs and is out of scope for this
    slice. The gate predicate is what guarantees the documented fallback."""
    writer, kernel, tiA, tiB = _build_writer(128, 128, 128,
                                             geometry=AB_B8, inst_k=128, bpe=1)
    with cpp_delegation():
        assert tiA._useCppOffsetAssign() is False
        assert tiB._useCppOffsetAssign() is False
