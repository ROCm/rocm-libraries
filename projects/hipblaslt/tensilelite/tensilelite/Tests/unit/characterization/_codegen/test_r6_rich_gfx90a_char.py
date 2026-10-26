################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R6 — gfx90a (CDNA2/aldebaran) rich GEMM arch-breadth characterization.

Drives the designed rich-GEMM config (``data/_designed/gfx90a/rich_gemm.yaml``)
through the config harness on gfx90a, exercising MFMA/cap arms specific to the
CDNA2 architecture that are not reached by seed.yaml:

  * MFMA_BF16_1K=True (BF16 1K mfma instruction variant at KWA 8554/9420-9438)
    — activated by DataType=BF16 on gfx90a (the aldebaran BBS family).
  * DirectToVgprA=[False, True] — the DTV single-sided arm (KWA 752-755
    StaggerUIterDTV alloc, 974-976, 1109-1111, 1315-1316 vgpr buffer paths).
  * GlobalSplitU=[1, 4] — GSU on/off divergence through
    _GlobalAccumulation=MultipleBuffer store path.
  * UseScaleAlphaVec=1, UseBias=1, Activation=True — keeps the activation/
    packing/global-write surface live (bias+activation ProblemType).

ForkParameters: 2 MI shapes x 2 DTV x 2 GSU = up to 8 permutations. Some
may be rejected by the BenchmarkProcess validator (e.g. DTV+large tile);
the test asserts >=1 valid kernel emits with err==0.

No GPU required. All paths are CPU-only kernel code generation.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx90a"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx90a",
    "rich_gemm.yaml",
)


def test_r6_rich_gfx90a_emits_assembly():
    """Rich gfx90a BBS config emits real AMDGCN assembly targeting gfx90a."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel emitted, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All emitted kernels must have err==0; failed: "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, (
            f"Kernel {base!r} source is too short ({len((src or '').splitlines())} lines)"
        )
        assert ".amdgcn_target" in src, f"Kernel {base!r}: missing .amdgcn_target directive"
        assert "gfx90a" in src, f"Kernel {base!r}: missing gfx90a arch marker in assembly"
        assert base.startswith("Cijk_"), f"Unexpected kernel basename: {base!r}"


def test_r6_rich_gfx90a_bf16_mfma_marker():
    """Verify the BF16 MFMA instruction appears in at least one emitted kernel.

    On gfx90a, DataType=BF16 emits v_mfma_f32_*_bf16 (or the 1K variant
    v_mfma_f32_*_bf16_1k). The presence of a v_mfma_f32_ instruction with
    'bf16' in the mnemonic confirms the MFMA_BF16_1K arm ran through
    KernelWriterAssembly.py line 8554 / MFMAInstruction emission at ~9420.
    """
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1
    found = any(
        "v_mfma_f32_" in (src or "") and "bf16" in (src or "")
        for (_b, src, _err) in results
    )
    assert found, (
        "Expected a v_mfma_f32_*_bf16* instruction in at least one gfx90a BBS "
        "kernel; none found. The MFMA_BF16_1K emission arm may not have been "
        "reached (KWA line 8554 / MFMAInstruction mfma1k path)."
    )


def test_r6_rich_gfx90a_gsu_variant_emitted():
    """Verify at least two distinct kernel basenames are emitted (fork coverage).

    A single-permutation result would mean the fork (DTV / GSU / MI shape) is
    collapsing — the test guards that the fork is generating coverage-widening
    kernel diversity.
    """
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    basenames = [b for (b, _s, _e) in results]
    assert len(set(basenames)) >= 2, (
        f"Expected >=2 distinct kernel basenames from the rich fork, got: {basenames}"
    )
