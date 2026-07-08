################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R3 -- gfx942 StreamK XCCMapping non-power-of-2 characterization (CPU-only).

Exercises the non-power-of-2 branch in ``XCCMappingOn.__call__``
(Tensile/Components/StreamK.py lines 78-79) by using
``StreamKXCCMapping=3`` (non-power-of-2 divisor):

Target missing range (methodology-A):
  78-79  XCCMappingOn.__call__ non-power-of-2 branch
         Condition: (divisor & (divisor - 1)) != 0
         StreamKXCCMapping=3 => 3 & 2 = 2 != 0 => triggers allocation of
         extra temp SGPRs via checkOutAligned(2, 2, "sTmp") (line 78)
         and ContinuousRegister(idx=sTmp, size=2) (line 79).

The existing R2 StreamK test uses StreamKXCCMapping=0 (power-of-2),
which skips the ``if ((divisor & (divisor - 1)) != 0)`` branch.
StreamKXCCMapping=3 is valid (valid values: {0, 2, 3, 4, 5, 6, 7, 8}).

CPU-only: no GPU required. The emit harness instantiates rocisa and runs
Python+rocisa codegen without compiling or launching any GPU kernels.
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
    "streamk_xccm.yaml",
)


def test_r3_streamk_xccm_gfx942_emits_assembly():
    """gfx942 StreamK XCCMapping=3 (non-power-of-2) emits real assembly with err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=4, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got 0"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"Expected all err==0, got: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, (
            f"Kernel {base!r} emitted suspiciously short source"
        )
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx942" in src, f"Kernel {base!r} missing gfx942 target"
        assert base.startswith("Cijk_"), f"Kernel {base!r} has unexpected prefix"


def test_r3_streamk_xccm_gfx942_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the gfx942 XCCMap emit."""
    results = emit_kernels_from_config(_CONFIG, limit=4, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
