#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Integration smoke tests for the C++ (nanobind) MX *scale* GR/LR
offset-assignment path (swizzled scale) for MXFP4 and MXFP8 gfx950 geometries.

``Kernel.graTileAssignmentScaleSwizzled`` and ``lraTileAssignmentScaleSwizzled``
derive scale offset-assignment scalar math via the C++
``MXScaleTileInfoQuery.scaleGrOffsetAssignPlan`` / ``scaleLrOffsetAssignPlan``,
then emit rocisa instructions in Python.

These tests are pure-string (no GPU runtime / hip dependency): rocisa is pinned
to gfx950 for deterministic emission.  They verify:

1. Non-empty emission with correct section labels for each dtype family.
2. Deterministic emission — two independent calls produce identical output.
3. Key mnemonic presence confirming thread-group shift, group masking, and
   stride-scale code paths are taken.

The C++ scalar plan values (including the integer-type fix that replaced the
legacy float crash in ``hex(numThreadsPerGroup - 1)``) are locked by the native
C++ gtest suite (cpp/tests/tile_info_test.cpp).
"""

import os
import sys
from types import SimpleNamespace

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
    TileInfo, MXSA_B4, MXSB_B4, MXSA_B8, MXSB_B8)
from Tensile.Components.Subtile.Kernel import (
    graTileAssignmentScaleSwizzled, lraTileAssignmentScaleSwizzled)

WAVESIZE = 64
MXBLOCK = 32

# (label, geomA, geomB, mt_a, mt_b, depth_u, wave_group). MacroTile divisible
# by 32 and data depthU divisible by 256 so the (2,2) scale LR subtile tiles
# cleanly for both wave groupings. Covers MXFP4 / MXFP8 and the [2,2] / [4,1]
# wave partitions.
CONFIGS = [
    ("MXFP4", MXSA_B4, MXSB_B4, 256, 256, 256, [2, 2]),
    ("MXFP8", MXSA_B8, MXSB_B8, 256, 256, 256, [2, 2]),
    ("MXFP4", MXSA_B4, MXSB_B4, 512, 256, 256, [4, 1]),
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


def _make_kernel(mt_a, mt_b, depth_u, wave_group):
    return {
        "DepthU": depth_u,
        "_DepthU": depth_u,
        "_DepthUA": depth_u,
        "_DepthUB": depth_u,
        "_DepthUMXSA": depth_u // MXBLOCK,
        "_DepthUMXSB": depth_u // MXBLOCK,
        "MacroTileA": mt_a,
        "MacroTileB": mt_b,
        "MacroTile0": mt_a,
        "MacroTile1": mt_b,
        "MatrixInstM": 16,
        "MatrixInstN": 16,
        "MatrixInstK": 128,
        "MIWaveGroup": wave_group,
        "WavefrontSize": WAVESIZE,
        "UseSubtileImpl": True,
        "NonTemporalMXSA": 0,
        "NonTemporalMXSB": 0,
        "ProblemType": {"MXBlockA": MXBLOCK, "MXBlockB": MXBLOCK},
    }


def _build_writer(geomA, geomB, mt_a, mt_b, depth_u, wave_group):
    """Minimal writer + scale TileInfo setup (no hip dependency)."""
    writer = SimpleNamespace()
    writer.vgprPool = RegisterPool(0, RegisterType.Vgpr,
                                   defaultPreventOverflow=False, printRP=False)
    writer.sgprPool = RegisterPool(0, RegisterType.Sgpr,
                                   defaultPreventOverflow=False, printRP=False)
    writer.sgprs = {}
    writer.vgprPool.checkOut(1)  # v0 = Serial

    kernel = _make_kernel(mt_a, mt_b, depth_u, wave_group)
    tiMXSA = TileInfo(geomA, 'MXSA', writer, kernel)
    tiMXSB = TileInfo(geomB, 'MXSB', writer, kernel)
    writer.agprPool = RegisterPool(0, RegisterType.Accvgpr,
                                   defaultPreventOverflow=False, printRP=False)
    writer.states = SimpleNamespace(
        mxsa=SimpleNamespace(tileInfo=tiMXSA),
        mxsb=SimpleNamespace(tileInfo=tiMXSB),
        regCaps={"MaxSgpr": 106, "MaxVgpr": 256, "PhysicalMaxVgpr": 512},
        archCaps={"LDSBankCount": 64, "LDSBankWidth": 4},
    )
    # Writer-owned LDS layout (stays Python-side; the C++ plan never sees it).
    writer.ldsStartOffsetMXSA = 0x1000
    writer.ldsStartOffsetMXSB = 0x1400
    writer.ldsTotalSize = 0x2000

    # Reserve s0-s11 (HW + strides), then offset regs.
    writer.sgprPool.checkOut(12)
    writer.sgprs["StridesMXSA"] = 10
    writer.sgprs["StridesMXSB"] = 11
    tiMXSA.allocOffsetRegisters(writer, kernel)
    tiMXSB.allocOffsetRegisters(writer, kernel)
    return writer, kernel


def _emit_all():
    """Emit scale GR + LR offset-assignment asm for every config."""
    sections = {}
    for name, ga, gb, mt_a, mt_b, du, wg in CONFIGS:
        label = f"{name}:{mt_a}x{mt_b}x{du}:wg{wg[0]}x{wg[1]}"
        writer, kernel = _build_writer(ga, gb, mt_a, mt_b, du, wg)
        sections[("GR", label)] = str(graTileAssignmentScaleSwizzled(writer, kernel))
        writer, kernel = _build_writer(ga, gb, mt_a, mt_b, du, wg)
        sections[("LR", label)] = str(lraTileAssignmentScaleSwizzled(writer, kernel))
    return sections


def test_scale_offset_assignment_deterministic():
    """C++-driven MX scale GR/LR offset assignment produces identical output on
    two independent calls — verifies emission is side-effect-free."""
    first = _emit_all()
    second = _emit_all()
    for name, ga, gb, mt_a, mt_b, du, wg in CONFIGS:
        label = f"{name}:{mt_a}x{mt_b}x{du}:wg{wg[0]}x{wg[1]}"
        for kind in ("GR", "LR"):
            assert first[(kind, label)] == second[(kind, label)], (
                f"{kind} scale offset-assignment not deterministic for {label}")


def test_scale_offset_assignment_mnemonic_sanity():
    """Key rocisa mnemonics appear in scale GR/LR offset-assignment output.

    Pins thread-group shift (v_lshrrev_b32 by 0x4), group-relative masking
    (v_and_b32), bpe stride scaling (s_lshl_b32), and stride multiply
    (v_mul_lo_u32) without hard-coding the full instruction sequence.
    """
    writer, kernel = _build_writer(MXSA_B4, MXSB_B4, 256, 256, 256, [2, 2])
    gr = str(graTileAssignmentScaleSwizzled(writer, kernel))
    writer2, kernel2 = _build_writer(MXSA_B4, MXSB_B4, 256, 256, 256, [2, 2])
    lr = str(lraTileAssignmentScaleSwizzled(writer2, kernel2))
    # GR: serial / numThreadsPerGroup (right-shift), group masking, stride ops.
    assert "v_lshrrev_b32" in gr, "GR: missing v_lshrrev_b32"
    assert "v_and_b32" in gr, "GR: missing v_and_b32"
    assert "s_lshl_b32" in gr, "GR: missing s_lshl_b32 bpe stride scale"
    assert "v_mul_lo_u32" in gr, "GR: missing v_mul_lo_u32 stride multiply"
    # LR: waveId extraction and lane-offset calculation.
    assert "v_lshrrev_b32" in lr, "LR: missing v_lshrrev_b32"
    assert "v_and_b32" in lr, "LR: missing v_and_b32"
    assert "v_lshlrev_b32" in lr, "LR: missing v_lshlrev_b32"


@pytest.mark.parametrize("name,ga,gb", [
    ("MXFP4", MXSA_B4, MXSB_B4),
    ("MXFP8", MXSA_B8, MXSB_B8),
])
def test_scale_offset_assignment_emits_nonempty(name, ga, gb):
    """Smoke test: the C++-driven scale emit produces non-empty asm without
    relying on the golden snapshot, and the scale plans are integer-typed
    (the legacy float-immediate crash is fixed)."""
    writer, kernel = _build_writer(ga, gb, 256, 256, 256, [2, 2])
    gr = str(graTileAssignmentScaleSwizzled(writer, kernel))
    writer, kernel = _build_writer(ga, gb, 256, 256, 256, [2, 2])
    lr = str(lraTileAssignmentScaleSwizzled(writer, kernel))
    assert "Computing GR Offset for MXSA" in gr
    assert "Computing GR Offset for MXSB" in gr
    assert "LR Offset Calculation for Scale Tensors" in lr

    tiMXSA = writer.states.mxsa.tileInfo
    gplan = tiMXSA.scaleGrOffsetAssignPlan()
    assert isinstance(gplan.numThreadsPerGroup, int)
    assert gplan.numThreadsPerGroup > 0
    lplan = tiMXSA.scaleLrOffsetAssignPlan(kernel)
    assert isinstance(lplan.totalScaleBytes, int)
    assert lplan.isA is True
