#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""End-to-end parity smoke test for the optional C++ (nanobind) subtile path.

This wires the already-ported subtile C++ slices through the *complete* Subtile
mainloop emission for one BF16 case and asserts the generated assembly is
**byte-identical** to the pure-Python path.

Target case (operator priority for the first end-to-end slice):
  gfx950, BF16 (AB_B16, row-major / TLU0), UseSubtileImpl, PGR=1,
  NoTailLoop=1, no MX scale, no TLU1, no tail mask.

The full pipeline exercised here is:

    LogicalScheduler.build()
    LogicalScheduler.allocVgprTiles()
    LogicalScheduler.populate_instructions()   # -> InstructionEmitter leaves
    LogicalScheduler.emitMainAndExitLoops()     # -> instructionSchedule per group

With C++ delegation enabled every ported slice participates in producing the
mainloop:

  * SubtileGeometry query math          (tensile_writer.subtile.geometry)
  * ABTileInfoQuery read-only queries   (tensile_writer.subtile.tile_info)
  * GR/LR offset-assignment plans       (ABTileInfoQuery.{gr,lr}OffsetAssignPlan)
  * single buffer-load / ds-read plans  (singleBufferLoadPlan / singleDsReadPlan)
  * MFMA instType selection             (emit leaves)
  * instruction scheduler               (tensile_writer.subtile.instruction_scheduler)

Byte-identical output is the contract: a default (delegation-off) build is
unchanged, and the opt-in C++ build emits the same kernel for this case. Cases
the C++ slices do not cover fall back to Python transparently (the gate
predicates / try-except in each Subtile module), so this test also pins that
the default path stays pure-Python.

This is a pure-string test (rocisa pinned to gfx950); no GPU runtime / hip
dependency. GPU functional validation is gated separately (gfx950 hardware).
"""

import contextlib
import os
import sys

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)
# Sibling unit-test modules (same directory) supply the mock-writer harness.
sys.path.insert(0, SCRIPT_DIR)

# rocisa (ISA emission) and the compiled geometry / tile_info / scheduler layers
# must all be importable for the delegated path to mean anything.
pytest.importorskip("rocisa")
pytest.importorskip("tensile_writer.subtile.geometry")
pytest.importorskip("tensile_writer.subtile.tile_info")
pytest.importorskip("tensile_writer.subtile.instruction_scheduler")

from Tensile.Components.Subtile import SubtileGeometry as _sg
from Tensile.Components.Subtile import LogicalScheduler as _ls
from Tensile.Components.Subtile import InstructionScheduler as _isched
from Tensile.Components.Subtile.LogicalScheduler import LogicalScheduler

# Reuse the existing pure-string mock-writer harness rather than duplicating it.
from test_SubtileBasedLogicalScheduler import (
    create_kernel,
    make_cfg_bf16_pgr1,
    make_writer_and_tileinfos,
)


@contextlib.contextmanager
def cpp_delegation():
    """Enable C++ delegation across every Subtile slice for the duration.

    Mirrors the real opt-in (``TENSILE_WRITER_CPP=1`` + installed extension) by
    re-resolving each module's ``_CPP`` handle, but scoped to the test so the
    process default stays pure-Python. Restores all switches on exit.
    """
    os.environ["TENSILE_WRITER_CPP"] = "1"
    saved = {
        _sg: (_sg._USE_CPP, _sg._CPP),
        _ls: (_ls._USE_CPP, _ls._CPP),
        _isched: (_isched._USE_CPP, _isched._CPP),
    }
    try:
        _sg._CPP = _sg._resolve_cpp_geometry()
        _sg._USE_CPP = _sg._CPP is not None
        _ls._CPP = _ls._resolve_cpp_logical_scheduler()
        _ls._USE_CPP = _ls._CPP is not None
        _isched._CPP = _isched._resolve_cpp_scheduler()
        _isched._USE_CPP = _isched._CPP is not None
        assert _sg._USE_CPP and _ls._USE_CPP and _isched._USE_CPP, (
            "C++ delegation requested but one or more extensions did not resolve"
        )
        yield
    finally:
        for mod, (use, cpp) in saved.items():
            mod._USE_CPP = use
            mod._CPP = cpp
        os.environ.pop("TENSILE_WRITER_CPP", None)


def _emit_bf16_mainloop(MT0, MT1, depthU=64):
    """Generate the full BF16 PGR=1 / NoTailLoop mainloop assembly as a string.

    Drives the production Subtile mainloop emission path with a fresh mock
    writer so register-pool state is deterministic across Python / C++ runs.
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
def test_bf16_mainloop_cpp_matches_python(MT0, MT1, depthU):
    """The complete BF16 PGR=1 / NoTailLoop mainloop is byte-identical whether
    the ported slices run in Python or delegate to C++."""
    asm_py = _emit_bf16_mainloop(MT0, MT1, depthU)
    with cpp_delegation():
        asm_cpp = _emit_bf16_mainloop(MT0, MT1, depthU)

    assert asm_py == asm_cpp, (
        f"BF16 mainloop asm mismatch for {MT0}x{MT1}x{depthU}:\n"
        f"--- PYTHON ---\n{asm_py}\n--- C++ ---\n{asm_cpp}"
    )


def test_bf16_mainloop_is_nonempty_and_complete():
    """Sanity: the generated mainloop is non-trivial and contains the expected
    control-flow + the delegated leaf instructions (GR/LR/MFMA)."""
    asm = _emit_bf16_mainloop(256, 256, 64)
    assert "LoopBeginL:" in asm, "missing mainloop label"
    assert "v_mfma" in asm, "missing MFMA leaves"
    assert "buffer_load" in asm, "missing global-read (GR) leaves"
    assert "ds_read" in asm or "ds_load" in asm, "missing local-read (LR) leaves"


def test_default_path_is_python_only():
    """Without the opt-in, the Subtile slices must stay pure-Python so the
    default build is byte-identical to the pre-C++ behavior."""
    assert _sg._USE_CPP is False
    assert _ls._USE_CPP is False
    assert _isched._USE_CPP is False
