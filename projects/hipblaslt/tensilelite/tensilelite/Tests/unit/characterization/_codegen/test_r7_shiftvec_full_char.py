################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R7 — ShiftVectorComponents full characterization test.

Targets: Tensile/Components/ShiftVectorComponents.py
  Primary ranges:   47-200  (ShiftVectorComponentsVALU.__call__)
                    552-784 (ShiftVectorComponentsMFMAAllThread)
  Secondary ranges: 208-546 (ShiftVectorComponentsMFMA dispatch + PartialThread)

Strategy
--------
Two config sweeps are combined in one isolated test run:

  A) VALU sweep (existing shiftvector2 config, lines 47-200):
     WorkGroup+ThreadTile kernels with VectorWidthA=4 and
     AssertFree0ElementMultiple=1 so the VALU edge-shift path (lines 47-200)
     runs for all three ThreadTile shapes.  Covers ~105 lines in the
     47-200 target range.

  B) MFMA sweep (new shiftvec_full config, lines 208-546):
     Two gfx942 MFMA shapes ([16,16,16] and [32,32,8]) with VectorWidthA=2
     and AssertFree0ElementMultiple=1 so the MFMA edge-shift path routes to
     ShiftVectorComponentsMFMAPartialThread.  SourceSwap=[0,1] varies the
     thread-coal orientation to cover both conThInProcDim branches at lines
     290-297.

Dead-code documentation (P5-ceiling evidence)
---------------------------------------------
Lines 552-784 (ShiftVectorComponentsMFMAAllThread):
  Dispatch condition at line 237: ``glvw > allContOutCoal * numThreadInCoal``.
  Solution.py clips GlobalReadVectorWidthA down to glvwAlimit, which is
  computed as ``MIOutputVectorWidth * (WavefrontSize // matrixInstN)`` for
  SourceSwap=0 / isA=True -- exactly equal to the AllThread threshold.
  Because the clip sets glvw == threshold, ``glvw > threshold`` is never True.
  P5-ceiling: dead code at ShiftVectorComponents.py:237-238 + 552-784.

Line 122 (ShiftVectorComponentsVALU, glvw < vectorWidth branch):
  Requires GlobalReadVectorWidthA < VectorWidthA.  Solution.py (lines
  5029-5032) rejects GRVWA > 1 unless GRVWA == VWA; GRVWA=1 gives
  GuaranteeNoPartialA = (1 % 1 == 0) = True, suppressing shiftVectorComponents
  (KernelWriter.py:5894). P5-ceiling: dead code.

pytestmark = pytest.mark.unit per project convention.
CPU-only; no GPU device required.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx942"

# VALU config (covers lines 47-200 of ShiftVectorComponentsVALU)
_CFG_VALU = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
    "shiftvector2.yaml",
)

# MFMA config (covers lines 208-546 of ShiftVectorComponentsMFMA + PartialThread)
_CFG_MFMA = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
    "shiftvec_full.yaml",
)


def test_r7_shiftvec_full_valu_emits():
    """ShiftVectorComponentsVALU path (lines 47-200) emits real gfx942 assembly.

    Three VALU kernels (different ThreadTile shapes) with VectorWidthA=4 and
    AssertFree0ElementMultiple=1 so the edge-shift path runs for all shapes.
    Covers ~105 executable lines in the 47-200 target range.
    """
    results = emit_kernels_from_config(_CFG_VALU, limit=8, arch=_ARCH)
    assert len(results) >= 1, (
        f"Expected >=1 VALU kernel from shiftvector2 sweep, got {len(results)}"
    )
    assert all(err == 0 for (_b, _s, err) in results), (
        "All VALU kernels must emit err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 50, (
            f"Expected non-trivial asm for {base!r}"
        )
        assert ".amdgcn_target" in src, f"Expected AMDGCN target in {base!r}"
        assert "gfx942" in src, f"Expected gfx942 arch in {base!r}"
        # VALU kernels emit ShiftVectorComponents label sequences
        assert "ShiftVectorComponents" in src, (
            f"Expected ShiftVectorComponents labels in VALU asm {base!r}"
        )


def test_r7_shiftvec_full_mfma_emits():
    """ShiftVectorComponentsMFMAPartialThread (lines 208-546) emits gfx942 asm.

    Two MFMA shapes x SourceSwap=[0,1] with VectorWidthA=2 and
    AssertFree0ElementMultiple=1 drive the MFMA edge-shift partial-thread path.
    At least one kernel must emit successfully; all valid permutations must be
    err==0.

    Lines 552-784 (AllThread) are documented dead code: glvwAlimit in
    Solution.py equals the AllThread threshold, so the dispatch at line 237
    never fires. P5-ceiling evidence at ShiftVectorComponents.py:237-238.
    """
    results = emit_kernels_from_config(_CFG_MFMA, limit=8, arch=_ARCH)
    assert len(results) >= 1, (
        f"Expected >=1 MFMA kernel from shiftvec_full sweep, got {len(results)}"
    )
    assert all(err == 0 for (_b, _s, err) in results), (
        "All MFMA kernels must emit err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 50, (
            f"Expected non-trivial asm for {base!r}"
        )
        assert ".amdgcn_target" in src, f"Expected AMDGCN target in {base!r}"
        assert "gfx942" in src, f"Expected gfx942 arch in {base!r}"
        # MFMA kernels also emit ShiftVectorComponents label sequences
        assert "ShiftVectorComponents" in src, (
            f"Expected ShiftVectorComponents labels in MFMA asm {base!r}"
        )
