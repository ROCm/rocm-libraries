################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R2 — gfx942 ShiftVectorComponents VALU path characterization test.

Drives the ShiftVectorComponents VALU edge-shift sweep config
(``data/_designed/gfx942/shiftvector.yaml``) through the config-driven emit
harness and asserts all emitted kernels have err==0.

Target file: Tensile/Components/ShiftVectorComponents.py
Missing ranges targeted: 47-200 (ShiftVectorComponentsVALU.__call__).
The VALU path is entered when EnableMatrixInstruction=False, which the harness
detects from WorkGroup+ThreadTile parameters (no MatrixInstruction block).

Three kernels are emitted (one per ThreadTile shape) sweeping VectorWidthA=2,
all with AssertFree0ElementMultiple=1 so edge shifting is unconditionally
active during code emission.
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
    "shiftvector.yaml",
)


def test_r2_shiftvector_emits_assembly():
    """ShiftVectorComponents VALU sweep emits real gfx942 assembly, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel from shiftvector sweep, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"Expected all err==0, got {[err for (_b,_s,err) in results]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, (
            f"Expected non-trivial asm for {base}"
        )
        assert ".amdgcn_target" in src, f"Expected AMDGCN target in {base}"
        assert "gfx942" in src, f"Expected gfx942 arch in {base}"
        assert base.startswith("Cijk_"), f"Expected Cijk_ prefix in {base}"


def test_r2_shiftvector_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the shiftvector emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
