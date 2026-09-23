################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R3 -- gfx942 AsmAddressCalculation addressing-arms characterization.

Drives the designed addrstore.yaml config (f8n + UseScaleAB=Vector +
UseScaleAlphaVec=1, two MFMA shapes) through the config-driven emit harness
and asserts every emitted kernel is real gfx942 AMDGCN assembly with err==0.

Target missing ranges in Tensile/AsmAddressCalculation.py:
  - 320-352  : ScaleAlphaVec arm (optSingleColVgpr branch)
  - 382-396  : ScaleAlphaVec arm (optSharedColVgpr branch)
  - 397-410  : ScaleAVec arm     (optSharedColVgpr branch)
  - 411-424  : ScaleBVec arm     (optSharedColVgpr branch)
  - 452-466  : ScaleAlphaVec arm (else/non-opt branch)
  - 467-480  : ScaleAVec arm     (else/non-opt branch)
  - 481-494  : ScaleBVec arm     (else/non-opt branch)

Gating knobs:
  UseScaleAB: "Vector"   -> ScaleAVec + ScaleBVec emitScaleToBpe arms
  UseScaleAlphaVec: 1    -> ScaleAlphaVec emitScaleToBpe arms
  DataType: f8n          -> required for UseScaleAB=Vector on gfx942

Golden: order-invariant {basename, err} digest snapshot (P4 R3).
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
    "addrstore.yaml",
)


def test_r3_addrstore_gfx942_emits_assembly():
    """Addrstore config emits real gfx942 assembly for both MFMA shapes, err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=4, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, f"Kernel {base!r} source too short"
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx942" in src, f"Kernel {base!r} missing gfx942 arch marker"
        assert base.startswith("Cijk_"), f"Unexpected basename: {base!r}"


def test_r3_addrstore_gfx942_golden(snapshot):
    """P4 golden: order-invariant {basename, err} digest of the addrstore emit."""
    results = emit_kernels_from_config(_CONFIG, limit=4, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
