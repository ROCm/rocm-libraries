################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P6 — gfx942 GSU MultipleBufferSingleKernel (MBSK) arm characterization.

Exercises the MBSK-gated arms of Tensile/Components/GSU.py that are NOT
reached by the existing test_r3_gsu_on_char.py (which uses MultipleBuffer and
SingleBuffer algorithms but not MultipleBufferSingleKernel).

Target lines in Tensile/Components/GSU.py:
  GSUOn.graWorkGroup             line 326-337  (_GlobalAccumulation==MBSK)
  GSUOn.globalWriteBatchProlog   line 725-728  (_GlobalAccumulation==MBSK)
  GSUOn.defineAndResources       line 734-745  (_GlobalAccumulation==MBSK)
  GSUOn.reductionBranches        line 789-803  (entry to reductionProcedure)
  GSUOn.reductionProcedure       line 805+     (workspace/reduction machinery)
  GSUOn.syncOffsetPreparation    line 1054+    (synchronizer setup)
  GSUOn.partialWriteBatch        line 1111+    (workspace write path)

Config: gsu_mbsk.yaml
  DataType h/h/s (half/half/float) — MBSK allows half, rejects double
  GlobalSplitU: [2, 4] — GSU>1 to activate MBSK workspace arms
  GlobalSplitUAlgorithm: MultipleBufferSingleKernel — sets _GlobalAccumulation=MBSK

Assertions: >=1 kernel emitted with err==0, MBSK-specific assembly markers
  (GSUAMBSK in basename, GSUSync register, AddressTD, Synchronizer, MTOffset).
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx942"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
    "gsu_mbsk.yaml",
)


def test_r6_gsu3_mbsk_emits_assembly():
    """MBSK sweep (GSU=[2,4], MBSK) emits >=1 gfx942 kernel with err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel from MBSK sweep, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, f"Kernel {base!r} source too short"
        assert base.startswith("Cijk_"), f"Unexpected basename: {base!r}"


def test_r6_gsu3_mbsk_asm_markers():
    """MBSK kernels contain MBSK-specific assembly: GSUAMBSK, GSUSync, AddressTD."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1
    for base, src, err in results:
        assert err == 0, f"Kernel {base!r} failed: err={err}"
        # GSUAMBSK tag in the assembly source confirms MBSK algorithm was selected.
        # Note: the returned basename may be a hash; the full name lives in the src.
        assert "GSUAMBSK" in src, (
            f"Kernel {base!r} missing GSUAMBSK in src — not using MBSK algorithm"
        )
        # GSUSync register is emitted by MBSK-gated arms (syncOffsetPreparation)
        assert "GSUSync" in src, (
            f"Kernel {base!r} missing GSUSync — MBSK synchronizer arm not reached"
        )
        # AddressTD (workspace address) is set up in graWorkGroup for MBSK
        assert "AddressTD" in src, (
            f"Kernel {base!r} missing AddressTD — MBSK workspace setup not reached"
        )
        # Synchronizer register is used in defineAndResources for MBSK
        assert "Synchronizer" in src, (
            f"Kernel {base!r} missing Synchronizer — MBSK defineAndResources not reached"
        )
        # MTOffset is emitted by reductionProcedure / partialWriteBatch for MBSK
        assert "MTOffset" in src, (
            f"Kernel {base!r} missing MTOffset — MBSK reduction procedure not reached"
        )
