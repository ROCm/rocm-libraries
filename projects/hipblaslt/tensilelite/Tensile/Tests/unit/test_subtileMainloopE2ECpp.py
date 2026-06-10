#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""End-to-end smoke test for the C++ (nanobind) subtile path.

This wires the ported subtile C++ slices through the *complete* Subtile
mainloop emission for one BF16 case and asserts the generated assembly is
deterministic and complete. All slices below are now C++-only (no env switch):

Target case (operator priority for the first end-to-end slice):
  gfx950, BF16 (AB_B16, row-major / TLU0), UseSubtileImpl, PGR=1,
  NoTailLoop=1, no MX scale, no TLU1, no tail mask.

The full pipeline exercised here is:

    LogicalScheduler.build()
    LogicalScheduler.allocVgprTiles()
    LogicalScheduler.populate_instructions()   # -> InstructionEmitter leaves
    LogicalScheduler.emitMainAndExitLoops()     # -> instructionSchedule per group

Every ported slice participates in producing the mainloop, unconditionally
through the compiled extension:

  * SubtileGeometry query math          (tensile_writer.subtile.geometry)
  * ABTileInfoQuery read-only queries   (tensile_writer.subtile.tile_info)
  * GR/LR offset-assignment plans       (ABTileInfoQuery.{gr,lr}OffsetAssignPlan)
  * single buffer-load / ds-read plans  (singleBufferLoadPlan / singleDsReadPlan)
  * MFMA instType selection             (emit leaves)
  * instruction scheduler               (tensile_writer.subtile.instruction_scheduler)

This is a pure-string test (rocisa pinned to gfx950); no GPU runtime / hip
dependency. GPU functional validation is gated separately (gfx950 hardware).
"""

import os
import sys

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)
# Sibling unit-test modules (same directory) supply the mock-writer harness.
sys.path.insert(0, SCRIPT_DIR)

# rocisa (ISA emission) and the compiled geometry / tile_info / scheduler layers
# must all be importable for the C++ path to mean anything.
pytest.importorskip("rocisa")
pytest.importorskip("tensile_writer.subtile.geometry")
pytest.importorskip("tensile_writer.subtile.tile_info")
pytest.importorskip("tensile_writer.subtile.instruction_scheduler")

from Tensile.Components.Subtile.LogicalScheduler import LogicalScheduler

# Reuse the existing pure-string mock-writer harness rather than duplicating it.
from test_SubtileBasedLogicalScheduler import (
    create_kernel,
    make_cfg_bf16_pgr1,
    make_writer_and_tileinfos,
)


def _emit_bf16_mainloop(MT0, MT1, depthU=64):
    """Generate the full BF16 PGR=1 / NoTailLoop mainloop assembly as a string.

    Drives the production Subtile mainloop emission path with a fresh mock
    writer so register-pool state is deterministic across runs.
    """
    kernel = create_kernel(MT0, MT1, fp4=False, depthU=depthU)
    kernel["NoTailLoop"] = True
    writer, tiA, tiB, _scaleA, _scaleB, dTileInfo = \
        make_writer_and_tileinfos(kernel, fp4=False)

    cfg = make_cfg_bf16_pgr1(MT0, MT1, depthU=depthU)
    assert cfg.pgr == 1

    sched = LogicalScheduler(cfg)
    sched.build()
    sched.allocVgprTiles(writer, tiA, tiB)
    try:
        sched.populate_instructions(
            writer, kernel,
            tileInfoA=tiA, tileInfoB=tiB,
            dtileInfo=dTileInfo,
        )
        asm = str(sched.emitMainAndExitLoops(writer, kernel))
    finally:
        sched.deallocVgprTiles(writer)
    return asm


# BF16 TLU0 configs spanning the loadRatioGR wave-partition modes
# (256x256 -> 2x2, 128x128 -> 2x2 different DU subIterK count). Both are
# PGR=1, no scale, no tail loop — the first end-to-end target.
BF16_CONFIGS = [
    (256, 256, 64),
    (128, 128, 64),
]


@pytest.mark.parametrize("MT0,MT1,depthU", BF16_CONFIGS,
                         ids=lambda c: c if isinstance(c, str) else None)
def test_bf16_mainloop_cpp_is_deterministic(MT0, MT1, depthU):
    """The complete BF16 PGR=1 / NoTailLoop mainloop, produced unconditionally
    through the C++-backed ported slices, is deterministic across fresh runs."""
    asm_first = _emit_bf16_mainloop(MT0, MT1, depthU)
    asm_second = _emit_bf16_mainloop(MT0, MT1, depthU)

    assert asm_first == asm_second, (
        f"BF16 mainloop asm not deterministic for {MT0}x{MT1}x{depthU}:\n"
        f"--- FIRST ---\n{asm_first}\n--- SECOND ---\n{asm_second}"
    )


def test_bf16_mainloop_is_nonempty_and_complete():
    """Sanity: the generated mainloop is non-trivial and contains the expected
    control-flow + the C++-driven leaf instructions (GR/LR/MFMA)."""
    asm = _emit_bf16_mainloop(256, 256, 64)
    assert "LoopBeginL:" in asm, "missing mainloop label"
    assert "v_mfma" in asm, "missing MFMA leaves"
    assert "buffer_load" in asm, "missing global-read (GR) leaves"
    assert "ds_read" in asm or "ds_load" in asm, "missing local-read (LR) leaves"
