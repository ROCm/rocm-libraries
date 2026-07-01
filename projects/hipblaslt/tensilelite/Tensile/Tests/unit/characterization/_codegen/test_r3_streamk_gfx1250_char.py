################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R3 -- gfx1250 StreamK deeper-coverage characterization (CPU-only).

Exercises arch-specific and VGPR-constant paths in
``Tensile/Components/StreamK.py`` that the earlier gfx942 R2 sweep
(test_r2_streamk_char.py) could not reach because those branches are gated
on ISA=(12,5,0) or arch-caps that only exist on gfx1250:

Target missing ranges (methodology-A):
  78-79   XCCMappingOn non-power-of-2 branch (StreamKXCCMapping=3, divisor&(divisor-1)!=0)
  141-153 StreamKMemoryOrdering.preVolatileVmem (RequiresXCntForVolatileVMEM=1, gfx1250)
  202-203 StreamKMemoryOrderingDefault.flagBufferMubuf
  206-246 StreamKMemoryOrderingDevScopeFences (HasInvWbDevFences=True)
  276-287 StreamK.shiftSrd for gfx125x (version[:2]==(12,5))
  315-318 computeTotalIters VReadfirstlane path (isStreamKConstantsToVgprEnabled=True)
  342-344 skTileIndex VReadfirstlane paths
  392-409 skExtraIters with skConstsInVgprs=True

gfx1250 triggers ``isStreamKConstantsToVgprEnabled=True`` for all StreamK != 4,
which inserts VReadfirstlane to move StreamK constants from SGPRs to VGPRs,
covering the branches guarded by that condition throughout StreamK.py.

``StreamKXCCMapping=3`` (non-power-of-2) exercises the ``XCCMappingOn.__call__``
branch that allocates extra temp SGPRs for non-power-of-2 divisors (lines 78-79).

CPU-only: no GPU required. The emit harness instantiates rocisa and runs
Python+rocisa codegen without compiling or launching any GPU kernels.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "streamk.yaml",
)


def test_r3_streamk_gfx1250_emits_assembly():
    """gfx1250 StreamK family sweep emits real assembly with err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got 0"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"Expected all err==0, got: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 50, (
            f"Kernel {base!r} emitted suspiciously short source"
        )
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx1250" in src, f"Kernel {base!r} missing gfx1250 target"
        assert base.startswith("Cijk_"), f"Kernel {base!r} has unexpected prefix"


def test_r3_streamk_gfx1250_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the gfx1250 StreamK emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
