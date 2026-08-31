################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P4 — gfx950 SubtileLREmit FP8 + localReadDoSubtile characterization.

Drives the P4 designed BenchmarkProblems config
(``data/_designed/gfx950/subtile_lr_fp8.yaml``) through the config-driven
emit harness and asserts every emitted kernel is real gfx950 AMDGCN assembly
with err==0.

Target: Tensile/Components/Subtile/SubtileLREmit.py, uncovered ranges:
  490-535  (_lraTileAssignment_fp8_legacy) — FP8 block-swap swizzle path,
            activated when tileInfoA.bpe == 1 (DataType: F8, UseSubtileImpl=1)
  613-655  (emitSubtileDsRead, localReadDoSubtile) — called from preLoop when
            PrefetchLocalRead >= 1 with UseSubtileImpl=1

The DataType: F8 (pure FP8 A+B, bpe=1) + UseSubtileImpl=1 + DirectToLds=1
combination routes _lraTileAssignment_legacy through the FP8 block-swap arm.
PrefetchLocalRead=1 triggers localReadDoSubtile() in the preloop emit.

CPU-only. No GPU, no compile, no hardware access.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx950",
    "subtile_lr_fp8.yaml",
)


def test_r3_subtile_lr_fp8_gfx950_emits_assembly():
    """UseSubtileImpl=1 + DataType:F8 config emits real gfx950 assembly, all err==0.

    Exercises SubtileLREmit FP8 path (lines 490-535) and
    localReadDoSubtile/emitSubtileDsRead (lines 613-655).
    """
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got 0 (config: {_CONFIG})"
    assert all(err == 0 for (_b, _s, err) in results), (
        f"some kernels failed: {[(b, e) for b, _s, e in results if e != 0]}"
    )
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 100, f"kernel {base!r}: suspiciously short assembly"
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx950" in src, f"kernel {base!r}: wrong arch in assembly"
        assert base.startswith("Cijk_"), f"kernel {base!r}: unexpected basename prefix"


def test_r3_subtile_lr_fp8_gfx950_golden(snapshot):
    """Order-invariant golden: pin {basename, err} for every emitted FP8 subtile kernel."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
