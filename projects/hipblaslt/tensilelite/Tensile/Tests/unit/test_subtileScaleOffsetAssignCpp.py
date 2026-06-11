#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Regression tests for the C++ (nanobind) MX *scale* GR/LR offset-assignment
path (swizzled scale), now C++-only for the gfx950 scale geometries.

``Kernel.graTileAssignmentScaleSwizzled`` and
``lraTileAssignmentScaleSwizzled`` derive the scale offset-assignment scalar
math (threads-per-scale-group, per-wave-partition byte stride, M-wave count,
and the MXSA-vs-MXSB partition-axis selector) before emitting the rocisa scale
offset-calculation instructions. That math is computed by the C++
``MXScaleTileInfoQuery.scaleGrOffsetAssignPlan`` / ``scaleLrOffsetAssignPlan``
(via ``TileInfo.scaleGrOffsetAssignPlan`` / ``scaleLrOffsetAssignPlan``) for
MXFP4 and MXFP8. The rocisa emission stays in Python.

These tests are pure-string (no GPU runtime / hip dependency): rocisa is pinned
to gfx950 so the emitted assembly is deterministic. They lock the C++-driven
emission against a committed golden snapshot
(``subtileScaleOffsetAssign_golden.txt``). The golden is the *new* C++-driven
emit: the deleted Python ``_legacy`` swizzled-scale GR offset path derived
``numThreadsPerGroup`` from the float ``lrSubtileSize`` and crashed on
``hex(... - 1)`` (``TypeError: 'float' object cannot be interpreted as an
integer``) for every gfx950 scale geometry, so it never produced a reference.
Integer-typing the C++ plan is the fix. Regenerate the golden with
``UPDATE_SCALE_OFFSET_GOLDEN=1`` after an intentional emit change.

The C++ scale offset-assignment *scalar plan* values are additionally locked by
the native C++ gtest suite (cpp/tests/tile_info_test.cpp). GPU functional
validation requires gfx950 hardware and is gated separately.
"""

import os
import sys
from types import SimpleNamespace

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

from Tensile.Components.Subtile.Kernel import (
    TileInfo, MXSA_B4, MXSB_B4, MXSA_B8, MXSB_B8)
from Tensile.Components.Subtile.Kernel import (
    graTileAssignmentScaleSwizzled, lraTileAssignmentScaleSwizzled)

WAVESIZE = 64
MXBLOCK = 32
GOLDEN_PATH = os.path.join(SCRIPT_DIR, "subtileScaleOffsetAssign_golden.txt")

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


def _labels():
    return [f"{name}:{mt_a}x{mt_b}x{du}:wg{wg[0]}x{wg[1]}"
            for name, ga, gb, mt_a, mt_b, du, wg in CONFIGS]


def _serialize(sections):
    parts = []
    for label in _labels():
        parts.append(f"===GR {label}===\n{sections[('GR', label)]}")
        parts.append(f"===LR {label}===\n{sections[('LR', label)]}")
    return "\n".join(parts) + "\n"


def _parse_golden(text):
    sections = {}
    cur_key = None
    cur_lines = []
    for line in text.splitlines():
        if line.startswith("===GR ") and line.endswith("==="):
            if cur_key is not None:
                sections[cur_key] = "\n".join(cur_lines)
            cur_key = ("GR", line[len("===GR "):-3])
            cur_lines = []
        elif line.startswith("===LR ") and line.endswith("==="):
            if cur_key is not None:
                sections[cur_key] = "\n".join(cur_lines)
            cur_key = ("LR", line[len("===LR "):-3])
            cur_lines = []
        else:
            cur_lines.append(line)
    if cur_key is not None:
        sections[cur_key] = "\n".join(cur_lines)
    return sections


def test_scale_offset_assignment_matches_golden():
    """C++-driven MX scale GR/LR offset assignment is deterministic.

    Emits the MXFP4 / MXFP8 configs and locks them against the committed golden
    snapshot."""
    emitted = _emit_all()

    if os.environ.get("UPDATE_SCALE_OFFSET_GOLDEN") == "1":
        with open(GOLDEN_PATH, "w") as f:
            f.write(_serialize(emitted))
        pytest.skip(f"Regenerated golden at {GOLDEN_PATH}")

    assert os.path.exists(GOLDEN_PATH), (
        f"Missing golden {GOLDEN_PATH}; regenerate with UPDATE_SCALE_OFFSET_GOLDEN=1")
    with open(GOLDEN_PATH) as f:
        golden = _parse_golden(f.read())

    for label in _labels():
        for kind in ("GR", "LR"):
            key = (kind, label)
            assert key in golden, f"golden missing section {key}"
            assert emitted[key] == golden[key], (
                f"{kind} scale offset-assignment asm mismatch for {label}:\n"
                f"EMITTED:\n{emitted[key]}\nGOLDEN:\n{golden[key]}")


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
