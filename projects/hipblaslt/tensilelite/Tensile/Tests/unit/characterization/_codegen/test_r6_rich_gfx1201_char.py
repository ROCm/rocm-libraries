################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R6 — gfx1201 (RDNA4 / WMMA V2) rich GEMM codegen characterization.

Drives a designed multi-fork gfx1201 config (HHS fp16, bias, activation,
UseScaleAlphaVec, GSU fork, two MIWaveTile shapes) through the config-harness
emit path, exercising RDNA4-specific WMMA V2 arms in KernelWriter.py /
Components/* that the baseline test_emit_gfx1201_char.py does not reach via
the narrow single-permutation curated logic files.

Target paths in Tensile/KernelWriter.py (gfx1201: HasWMMA_V2=True,
HasWMMA_V1=False, HasEccHalf=False):
  - else branch of "if HasEccHalf or not HasWMMA_V1" (lines ~1012-1018):
    WMMA V2 instPerRegPack = 0 (not int8 pack) => pack scheduling
    produces instPerPackA/instPerPackB = 0 paths
  - bpeCinternal override for WMMA half compute (line ~6731):
    HasWMMA_V1=False means the "special case for wmma h and b" block is
    skipped; bpeCinternal stays at default
  - HHH_WMMA evaluation (line 6737): HPA=True forces HHH_WMMA=False even
    with HasWMMA + DestDataType.isHalf; exercises the else branch
  - GSU>1 + MultipleBuffer path on WMMA arch (GlobalSplitU=4 kernel):
    exercises _GlobalAccumulation=MultipleBuffer reduction path
  - UseScaleAlphaVec=1 bias/activation epilogue in GlobalWriteBatch
  - Two MIWaveTile shapes ([4,1] and [2,2]) exercise different
    SubGroup/WorkGroup geometry derivations in Solution.py

No GPU required; all emit paths are CPU-only code generation.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx1201"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1201",
    "rich_wmma.yaml",
)


def test_r6_rich_gfx1201_emits_assembly():
    """Rich gfx1201 WMMA config emits real assembly with err==0 for all kernels."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"Expected >=1 kernel, got {len(results)}"
    assert all(err == 0 for (_b, _s, err) in results), (
        "All kernels must emit with err==0; "
        + str([(b, e) for (b, _s, e) in results if e != 0])
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 50, f"Kernel {base!r} source too short"
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx1201" in src, f"Kernel {base!r} missing gfx1201 arch marker"
        assert base.startswith("Cijk_"), f"Unexpected basename: {base!r}"


def test_r6_rich_gfx1201_wmma_arch_marker():
    """Emitted kernels confirm gfx12 ISA target string (WMMA V2 path)."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1
    # All emitted kernels should carry the gfx1201 target directive
    for base, src, err in results:
        assert err == 0, f"Kernel {base!r} emitted with err={err}"
        assert "amdgcn-amd-amdhsa" in src, (
            f"Kernel {base!r}: missing amdgcn-amd-amdhsa in target; "
            "WMMA V2 path may not have been taken"
        )


def test_r6_rich_gfx1201_gsu_variants():
    """At least two distinct kernels emitted — confirms GSU fork produced variants."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    basenames = [b for (b, _s, _e) in results]
    assert len(basenames) >= 2, (
        f"Expected >=2 kernels from GSU/MI fork, got {len(basenames)}: {basenames}"
    )
    # Basenames must be distinct
    assert len(set(basenames)) == len(basenames), (
        f"Duplicate basenames in emit results: {basenames}"
    )
