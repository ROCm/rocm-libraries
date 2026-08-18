################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R4 — gfx942 ShiftVectorComponents advanced-feature characterization test.

Targets Tensile/Components/ShiftVectorComponents.py ranges:
  - 47-200  : ShiftVectorComponentsVALU.__call__ (VALU edge-shift sweep)
  - 552-784 : ShiftVectorComponentsMFMAAllThread (dead-code analysis below)

VectorWidth sweep [1,4] over the VALU path (EnableMatrixInstruction=False,
AssertFree0ElementMultiple=1) drives all branches in the VALU arm:
  - VW=1 : baseline path, glvw==vectorWidth so the log2-shift branch at
            line 122 is skipped (glvw < vectorWidth is False).
  - VW=4 : wider per-element loops at lines 162-180; thread-tile0 stride
            differs; exercises the ``dst``/``src`` offset arithmetic with
            a 4-wide local-read vector.

DEAD-CODE ANALYSIS — lines 552-784 (ShiftVectorComponentsMFMAAllThread):
  The dispatch at line 237 routes to AllThread only when
      glvw > allContOutCoal * numThreadInCoal.
  For every valid MFMA config, ``glvwAlimit`` (computed at Solution time)
  equals exactly (allContOutCoal * numThreadInCoal / VW) * VW ==
  the AllThread threshold. Because Solution.py clips glvw DOWN to glvwAlimit
  whenever partialA/partialB is True (the only case where ShiftVector runs),
  the condition ``glvw > threshold`` is never satisfied:
    - SourceSwap=0, A: glvwAlimit = MIOutputVW*(WS//matN) == threshold/VW
    - SourceSwap=0, B: glvwBlimit = matN*VW == threshold (VW*matN)
    - SourceSwap=1, A: glvwAlimit = matM*VW == threshold (VW*matM)
    - SourceSwap=1, B: glvwBlimit = MIOutputVW*(WS//matM) == threshold/VW
  All four cases: max-reachable glvw <= AllThread threshold.  AllThread is
  dead code in the standard kernel-generation pipeline.  This is recorded as
  P5 ceiling evidence at ShiftVectorComponents.py:237-238 (dispatch) and
  552-784 (body).

pytestmark = pytest.mark.unit per project convention.
CPU-only; no GPU device required.
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
    "shiftvector2.yaml",
)


def test_r4_shiftvector2_emits_assembly():
    """ShiftVectorComponents VALU VW-sweep emits real gfx942 assembly, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel from shiftvector2 sweep, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, (
            f"Expected non-trivial asm for {base!r}"
        )
        assert ".amdgcn_target" in src, f"Expected AMDGCN target in {base!r}"
        assert "gfx942" in src, f"Expected gfx942 arch marker in {base!r}"
        assert base.startswith("Cijk_"), f"Expected Cijk_ prefix in {base!r}"


def test_r4_shiftvector2_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the shiftvector2 emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
