################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R3 — gfx942 GlobalWriteBatch characterization (no GPU required).

Drives the designed globalwrite.yaml config (fp8n-bf16 + UseScaleAB=Vector +
UseScaleAlphaVec + StorePriorityOpt fork) through the config-driven emit harness
and asserts every emitted kernel is real gfx942 AMDGCN assembly with err==0.

Target missing ranges in Tensile/Components/GlobalWriteBatch.py:
  - 224-231: UseScaleAB=Vector in globalStoreWait interleaved path
  - 229-231: UseScaleAlphaVec in globalStoreWait interleaved path
  - 286-291: UseScaleAB=Vector + UseScaleAlphaVec in non-interleaved wait path
  - 374-405: alphaBeforeLoadC path (StorePriorityOpt gate, MIArchVgpr=1 required)
  - 459-498: edge store / beta load setup paths
  - 536-542: addEpilogueLoad gwvw != factor_gwvw remain_load inner loop

Golden: order-invariant {basename, err} digest snapshot (seeded).
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
    "globalwrite.yaml",
)


def test_r3_globalwrite_gfx942_emits_assembly():
    """GlobalWrite seed config emits real gfx942 assembly, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
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


def test_r3_globalwrite_gfx942_golden(snapshot):
    """P4 golden: order-invariant {basename, err} digest of the globalwrite emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
