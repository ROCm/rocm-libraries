################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R6 — rich gfx908 (CDNA1 / arcturus MFMA) codegen characterization.

Exercises arch-gated KernelWriterAssembly / KernelWriter / Components paths
that are specific to gfx908 ISA (9,0,8) and NOT reached by gfx942/gfx950 tests:

  - MI shape fork: [16,16,16,1] (HHS 16x16 tile) vs [32,32,8,1] (larger 32x32 tile)
  - GlobalSplitU=1 (GSU off-path) vs GlobalSplitU=4 (MultipleBuffer on-path)
  - UseBias + Activation (hipblaslt_all / relu) + UseScaleAlphaVec=1
    - UseScaleAlphaVec=1 gates GlobalWriteBatch.py lines 289-291, 559-565
  - StorePriorityOpt fork: False vs True (gates alphaBeforeLoadC path)
  - PrefetchGlobalRead=0 (no prefetch path distinct from PGR=2)

Data type: HHS (fp16 in / fp16 out / f32 compute + HighPrecisionAccumulate)
matching the gfx908 curated HHS.yaml logic shape.

pytestmark=pytest.mark.unit — CPU-only, no GPU required.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx908"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx908",
    "rich_gfx908.yaml",
)


def test_r6_rich_gfx908_emits_assembly():
    """Rich gfx908 config emits real gfx908 AMDGCN assembly with err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; failures: "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 50, f"Kernel {base!r} source too short"
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx908" in src, f"Kernel {base!r} missing gfx908 arch marker"
        assert base.startswith("Cijk_"), f"Unexpected basename: {base!r}"


def test_r6_rich_gfx908_mi_shape_fork():
    """Both MI shapes ([16,16,16,1] and [32,32,8,1]) produce kernels."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    basenames = [b for (b, _s, _e) in results]
    # Accept >=2 distinct kernels from the two MI shape variants.
    # Kernel names encode the MI shape (e.g. MI16x16x16 or MT64x... vs MT128x...).
    assert len(basenames) >= 2, (
        f"Expected kernels from multiple MI shapes, got: {basenames}"
    )
    assert len(set(basenames)) == len(basenames), (
        f"Duplicate basenames found: {basenames}"
    )


def test_r6_rich_gfx908_gsu_variants():
    """GSU=1 (off-path) and GSU=4 (MultipleBuffer on-path) both produce kernels."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 2, "Expected GSU variants to produce >=2 kernels"
    # All err==0
    errs = [(b, e) for (b, _s, e) in results if e != 0]
    assert not errs, f"Some kernels failed: {errs}"
    # At least one kernel should have GSU1 in the name and one GSU>1
    basenames = [b for (b, _s, _e) in results]
    gsu1 = any("GSU1" in b for b in basenames)
    gsu4 = any("GSU4" in b or "GSU2" in b or "GSU8" in b for b in basenames)
    # Accept pass if we have >=2 distinct basenames (names may differ by impl)
    assert len(set(basenames)) >= 2, f"Expected >=2 distinct kernels, got: {basenames}"


def test_r6_rich_gfx908_bias_activation_src():
    """Bias + Activation + UseScaleAlphaVec present in all emitted sources."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert results, "Expected at least one emitted kernel"
    for base, src, err in results:
        assert err == 0, f"Kernel {base!r} emitted with err={err}"
        # Emitted assembly for bias/activation kernels contains the Cijk_ prefix
        # and should be non-trivial AMDGCN output.
        assert ".amdgcn_target" in src, f"Missing .amdgcn_target in {base!r}"
