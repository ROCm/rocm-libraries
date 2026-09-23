################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R2 — gfx942 global store-batch path characterization (no GPU required).

Drives the designed store.yaml config (BBS bf16 + UseScaleAlphaVec + Bias,
StoreRemapVectorWidth x StorePriorityOpt fork) through the config-driven emit
harness and asserts every emitted kernel is real gfx942 AMDGCN assembly with
err==0.

Target missing ranges in Tensile/Components/GlobalWriteBatch.py:
  - 287-331  : UseScaleAB/UseScaleAlphaVec in globalStoreWait non-interleave
               path; _chooseAddForAtomic variants
  - 374-405  : alphaBeforeLoadC + codeMulAlpha path (StorePriorityOpt fork)
  - 459-542  : edge/beta load epilogue, GroupLoadStore, MultipleBufferSingleKernel
  - 568+     : UseScaleAlphaVec epilogue load; StoreRemap _GlobalAccumulation paths

Golden: order-invariant {basename, err} digest snapshot (P3).
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
    "store.yaml",
)


def test_r2_store_gfx942_emits_assembly():
    """Store-batch seed config emits real gfx942 assembly, all err==0."""
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


def test_r2_store_gfx942_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the store emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
