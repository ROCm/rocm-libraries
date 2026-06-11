#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Integration smoke tests for the C++ (nanobind) subtile GR/LR *offset
assignment* path, covering each AB dtype (BF16/B16, FP4/B4, FP8/B8) including
the 2x2 tile-shape variants.

``Kernel.graTileAssignment`` and ``Kernel.lraTileAssignment`` derive the
offset-assignment scalar math via the C++ ``ABTileInfoQuery.grOffsetAssignPlan``
/ ``lrOffsetAssignPlan``, then emit rocisa offset-calculation instructions in
Python.

These tests are pure-string (no GPU runtime / hip dependency): rocisa is pinned
to gfx950 for deterministic emission.  They verify:

1. Non-empty emission for each dtype family (BF16, FP4, FP8).
2. Deterministic emission — two independent calls produce identical output.
3. Key label/mnemonic presence confirming wave-ID, column-masking, and stride
   code paths are taken.

The C++ offset-assignment *scalar plan* values are locked by the native C++
gtest suite (cpp/tests/tile_info_test.cpp).  GPU functional validation requires
gfx950 hardware and is gated separately.

Scope note: TLU1 geometry is intentionally excluded — GR/LR register allocation
is still a singledispatch stub there and is tracked as separate follow-up work.
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
# geometry is a C++ nanobind attribute on tensile_writer.subtile, not a separate
# Python module file, so guard on the parent package rather than the attribute.
pytest.importorskip("rocisa")
_cppsubtile = pytest.importorskip("tensile_writer.subtile")
cppgeo = _cppsubtile.geometry
cppti = pytest.importorskip("tensile_writer.subtile.tile_info")

from rocisa.register import RegisterPool
from rocisa.enum import RegisterType

from Tensile.Components.Subtile.Kernel import (
    TileInfo, AB_B16, AB_B8, AB_B4, AB_B16_2x2, AB_B4_2x2)
from Tensile.Components.Subtile.Kernel import graTileAssignment
from Tensile.Components.Subtile.Kernel import lraTileAssignment

WAVESIZE = 64

# (label, geometry, mt_a, mt_b, depth_u, inst_k, bpe). Covers the three
# loadRatioGR wave-partition modes for BF16 (2x2 / 1x4 / 4x1), the 2x2
# tile-shape variants, FP4 (bpe 0.5), and FP8 (distinct block-swap swizzle).
CONFIGS = [
    ("AB_B16",     AB_B16,     256, 256, 64,  32,  2),    # MIWaveGroup [2,2]
    ("AB_B16",     AB_B16,     80,  64,  64,  32,  2),    # [1,4]
    ("AB_B16",     AB_B16,     64,  48,  64,  32,  2),    # [4,1]
    ("AB_B16",     AB_B16,     16,  64,  64,  32,  2),    # [1,4], small MT
    ("AB_B16_2x2", AB_B16_2x2, 256, 256, 64,  32,  2),
    ("AB_B4",      AB_B4,      128, 128, 256, 128, 1),    # fp4 (bpe 0.5)
    ("AB_B4_2x2",  AB_B4_2x2,  256, 256, 256, 128, 1),
    ("AB_B8",      AB_B8,      128, 128, 128, 128, 1),    # fp8
    ("AB_B8",      AB_B8,      256, 256, 128, 128, 1),
]


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


def _emit_all():
    """Emit GR + LR offset-assignment asm for every config, keyed by section."""
    sections = {}
    for name, geom, mt_a, mt_b, du, ik, bpe in CONFIGS:
        label = f"{name}:{mt_a}x{mt_b}x{du}"
        writer, kernel, _, _ = _build_writer(mt_a, mt_b, du, geometry=geom,
                                             inst_k=ik, bpe=bpe)
        sections[("GR", label)] = str(graTileAssignment(writer, kernel,
                                                        useSwizzling=True))
        writer, kernel, _, _ = _build_writer(mt_a, mt_b, du, geometry=geom,
                                             inst_k=ik, bpe=bpe)
        sections[("LR", label)] = str(lraTileAssignment(writer, kernel))
    return sections


def test_offset_assignment_deterministic():
    """C++-driven GR/LR offset assignment produces identical output on two
    independent calls — verifies emission is side-effect-free."""
    first = _emit_all()
    second = _emit_all()
    for name, geom, mt_a, mt_b, du, ik, bpe in CONFIGS:
        label = f"{name}:{mt_a}x{mt_b}x{du}"
        for kind in ("GR", "LR"):
            assert first[(kind, label)] == second[(kind, label)], (
                f"{kind} offset-assignment not deterministic for {label}")


def test_offset_assignment_mnemonic_sanity():
    """Key rocisa mnemonics appear in GR/LR offset-assignment output for BF16.

    Pins wave-ID extraction (v_lshrrev_b32), column masking (v_and_b32), and
    stride multiply (s_mul_i32) without hard-coding full instruction sequences.
    """
    writer, kernel, _, _ = _build_writer(256, 256, 64, geometry=AB_B16)
    gr = str(graTileAssignment(writer, kernel, useSwizzling=True))
    writer2, kernel2, _, _ = _build_writer(256, 256, 64, geometry=AB_B16)
    lr = str(lraTileAssignment(writer2, kernel2))
    for asm, tag in [(gr, "GR"), (lr, "LR")]:
        assert "v_lshrrev_b32" in asm, f"{tag}: missing v_lshrrev_b32"
        assert "v_and_b32" in asm, f"{tag}: missing v_and_b32"
    assert "s_mul_i32" in gr, "GR: missing s_mul_i32 stride multiply"


@pytest.mark.parametrize("name,geom,bpe", [
    ("AB_B16", AB_B16, 2),
    ("AB_B4", AB_B4, 1),
    ("AB_B8", AB_B8, 1),
])
def test_offset_assignment_emits_nonempty(name, geom, bpe):
    """Smoke test: the C++-driven emit produces non-empty asm for each dtype
    family (BF16, FP4, FP8) without relying on the golden snapshot."""
    inst_k = 32 if bpe == 2 else 128
    mt_a, mt_b, du = (256, 256, 64) if bpe == 2 else (128, 128, 128 if name == "AB_B8" else 256)
    writer, kernel, tiA, _ = _build_writer(mt_a, mt_b, du, geometry=geom,
                                           inst_k=inst_k, bpe=bpe)
    gr = str(graTileAssignment(writer, kernel, useSwizzling=True))
    writer, kernel, _, _ = _build_writer(mt_a, mt_b, du, geometry=geom,
                                         inst_k=inst_k, bpe=bpe)
    lr = str(lraTileAssignment(writer, kernel))
    assert "GR Offset Calculation" in gr
    assert "LR Offset Calculation" in lr
    # FP8 uses the distinct block-swap swizzle; BF16/FP4 use the DPP pair-swap.
    if bpe == 1 and name == "AB_B8":
        assert tiA.lrOffsetAssignPlan(writer, kernel).isFp8 is True
