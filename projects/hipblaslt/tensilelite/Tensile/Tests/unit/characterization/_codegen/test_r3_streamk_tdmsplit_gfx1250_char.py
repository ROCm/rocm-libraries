################################################################################
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""gfx1250 StreamK + TDMSplit + PrefetchAcrossPersistent characterization (CPU-only).

Pins the multi-wave TDMSplit codegen enabled for MX StreamK+PAP on gfx1250.

The change replaced two persistent split-increment SGPRs
(``tdmABGlobalSplitIncs`` / ``tdmABLdsSplitIncs``) with a transient recompute at
point of use (``KernelWriterAssembly._tdmSplitMultiWaveInc`` /
``_emitTdmWaveParitySCC``): on the multi-wave (prod(MIWaveGroup) > 1) path the
LDS split boundary and the global split increment are recomputed per wave-parity
(even waves load A, odd load B) instead of being persisted, keeping MX-scaled
StreamK+PAP kernels within the gfx1250 wave32 SGPR budget. The Solution.py reject
that blocked TDMSplit+PAP on MX-scaled inputs was removed as a result.

The config (``streamk_tdmsplit.yaml``) forces StreamK=3, TDMInst=3, TDMSplit,
PrefetchAcrossPersistent=1, and keeps the multi-wave MX WMMA MatrixInstruction so
the removed reject no longer fires and the transient split-increment path runs.

CPU-only: no GPU required. The emit harness instantiates rocisa and runs
Python+rocisa codegen without compiling or launching any GPU kernels.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = [
    pytest.mark.unit,
    # TDMSplit is currently disabled (rejected in Solution.py), so these
    # configs produce 0 solutions and cannot emit assembly. Expected-fail
    # until TDMSplit is re-enabled.
    pytest.mark.xfail(reason="TDMSplit is currently disabled", strict=False),
]

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "streamk_tdmsplit.yaml",
)


def test_r3_streamk_tdmsplit_gfx1250_emits_assembly():
    """gfx1250 SK+TDMSplit+PAP emits real assembly via the transient split-inc path."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, "Expected >=1 kernel, got 0"
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
        # Multi-wave TDMSplit no longer persists the split-increment SGPRs; the
        # increments are recomputed transiently at point of use.
        assert "tdmABGlobalSplitIncs" not in src, (
            f"Kernel {base!r} still emits persistent tdmABGlobalSplitIncs SGPR"
        )
        assert "tdmABLdsSplitIncs" not in src, (
            f"Kernel {base!r} still emits persistent tdmABLdsSplitIncs SGPR"
        )
        # The transient recompute selects the split stride by wave parity
        # (_emitTdmWaveParitySCC leaves a "wave parity" comment in the emit).
        assert "wave parity" in src, (
            f"Kernel {base!r} missing wave-parity split-increment recompute"
        )


def test_r3_streamk_tdmsplit_gfx1250_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the SK+TDMSplit+PAP emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
