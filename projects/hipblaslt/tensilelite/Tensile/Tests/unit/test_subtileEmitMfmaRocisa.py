#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""KernelWriter/rocisa integration test for the subtile MFMA emit leaf.

This is NOT a C++-parity test. The writer-free value layer it used to sit
alongside — the MFMA F8F6F4 instType *selection* mapping and the
single-buffer-load / single-ds-read instruction-shape *plans* — has been ported
to native C++ gtest under ``cpp_migration/cpp/tests`` (see
``emit_leaves_test.cpp``). What remains here genuinely exercises
``Kernel.emitMfmaInstruction`` building a real ``rocisa`` Module: it renders the
MFMA assembly string for each supported F8/F6/F4 case and asserts the emission
stays well-formed. That path depends on the rocisa ISA layer and the writer's
register pools, so it has no C++-only equivalent and stays in Python.

It runs only when the compiled extension and rocisa are importable (both are
hard dependencies of Kernel.py); otherwise it skips.

PR creation for this slice is human-only: a ``human:pr`` task is filed for
Bryant Nelson only after review says merge-ready. Agents never open PRs.
"""

import os
import sys
from types import SimpleNamespace

import pytest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TENSILE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, TENSILE_ROOT)

# Both the ISA layer (rocisa) and the compiled emit/query layers must exist.
pytest.importorskip("rocisa")
pytest.importorskip("tensile_writer.subtile.emit")

from Tensile.Common.DataType import DataType
from Tensile.Components.Subtile.Kernel import emitMfmaInstruction


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


@pytest.fixture(scope="module", autouse=True)
def _rocisa_once():
    # Pure string test: pin rocisa to gfx950 so the MFMA emission path matches
    # the assertions regardless of the host GPU.
    _init_rocisa_gfx950()


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
def test_emitMfmaInstruction_emits_asm(dA, dB, swap):
    """emitMfmaInstruction renders a non-empty MFMA module for each covered
    F8/F6/F4 case using the C++-backed instType selection.

    This is the rocisa integration check: the C++ instType *mapping* itself is
    covered by emit_leaves_test.cpp; here we confirm the writer wires that
    decision into a well-formed rocisa MFMA emission."""
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
