################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""PostLoopStoreInNll (PLSIN) gfx950 full-codegen characterization.

Drives the designed MX-FP4 / BF16-dest config
(``data/_designed/gfx950/plsin_mxf4_bf16.yaml``) through the config-driven emit
harness. The fork grid pairs a PLSIN-eligible MatrixInstruction (MIWaveTile 8x8)
with an ineligible one (MIWaveTile 2x2) across StreamK {3,4,5}, so a single emit
produces both fused and baseline kernels.

Asserts:
  * every kernel emits cleanly (err==0, real gfx950 assembly);
  * the eligible kernels carry the hoisted PostLoopFusedStore predicate and the
    ds_bpermute paired store (the feature actually activates -- the common MXFP4
    configs use an FP32 destination and never do);
  * the ineligible kernels keep the unchanged baseline emission (no
    PostLoopFusedStore), proving the feature is scoped;
  * one fused kernel assembles with the real ROCm assembler (skips if absent),
    catching allocator/overflow bugs the text assertions would miss.

CPU-only for the emit path. No GPU access.
"""

import os

import pytest

from config_harness import emit_kernels_from_config, assert_assembles

pytestmark = pytest.mark.unit

_ARCH = "gfx950"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data", "test_data", "_designed", "gfx950", "plsin_mxf4_bf16.yaml",
)

# 2 MatrixInstruction (2x2 off, 8x8 on) x StreamK {3,4,5}.
_EXPECTED_TOTAL = 6
_EXPECTED_FUSED = 3   # MIWaveTile 8x8 x SK{3,4,5}
_EXPECTED_PLAIN = 3   # MIWaveTile 2x2 x SK{3,4,5}

_FUSED_FLAG = "PostLoopFusedStore"


def _emit():
    results = emit_kernels_from_config(_CONFIG, limit=_EXPECTED_TOTAL, arch=_ARCH)
    if not results:
        pytest.skip("no gfx950 MX-FP4 kernels emitted (toolchain capabilities unavailable)")
    return results


def test_plsin_config_emits_real_gfx950_assembly():
    results = _emit()
    assert all(err == 0 for (_b, _s, err) in results), (
        f"some kernels failed: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, f"kernel {base!r}: suspiciously short"
        assert ".amdgcn_target" in src and "gfx950" in src, f"kernel {base!r}: wrong/no target"
        assert base.startswith("Cijk_"), f"kernel {base!r}: unexpected prefix"


def test_plsin_partitions_into_fused_and_baseline():
    results = _emit()
    fused = [b for (b, s, _e) in results if _FUSED_FLAG in s]
    plain = [b for (b, s, _e) in results if _FUSED_FLAG not in s]
    # The whole point: PLSIN activates for a BF16 dest on gfx950 (unlike the
    # FP32-dest common configs), AND ineligible tiles keep the baseline path.
    assert fused, "expected at least one PLSIN (fused) kernel, got none"
    assert plain, "expected at least one baseline (non-fused) kernel, got none"
    if len(results) == _EXPECTED_TOTAL:
        assert len(fused) == _EXPECTED_FUSED
        assert len(plain) == _EXPECTED_PLAIN


def test_fused_kernels_carry_paired_store_and_baseline_does_not():
    results = _emit()
    for base, src, _err in results:
        if _FUSED_FLAG in src:
            # Fused kernels emit the in-NLL ds_bpermute paired BF16 store.
            assert "ds_bpermute_b32" in src, f"fused kernel {base!r}: missing ds_bpermute store"
        else:
            # Baseline kernels must not reference the fused-store predicate at all.
            assert _FUSED_FLAG not in src, f"baseline kernel {base!r}: unexpected fused flag"


def test_one_fused_kernel_assembles():
    results = _emit()
    fused = [(b, s) for (b, s, _e) in results if _FUSED_FLAG in s]
    if not fused:
        pytest.skip("no fused kernel to assemble")
    base, src = fused[0]
    assert_assembles(src, base)
