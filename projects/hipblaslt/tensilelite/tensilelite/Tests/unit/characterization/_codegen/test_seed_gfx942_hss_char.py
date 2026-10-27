################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P2-widen — gfx942 HSS (fp16-in / fp32-out) family seed emit characterization.

The gfx942 attribution entry ``gfx942__HSS_BH_Bias_yaml`` (+32 marginal
whole-project lines) is an fp16-in / fp32-out GEMM with Bias — a distinct
DataType -> DestDataType pairing (h -> s) the dominant bf16 seed and the
h-in/h-out family seeds do not reach. No common config uses HSS as
BenchmarkProblems[0], so this drives an authored single-permutation seed
(``data/_designed/gfx942/hss.yaml``) and asserts the emitted kernel is real
gfx942 assembly with err==0.

P3: snapshot golden added — order-invariant {basename, err} digest per kernel.
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
    "hss.yaml",
)


def test_seed_gfx942_hss_emits_assembly():
    """The HSS family seed config emits real gfx942 assembly, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1
    assert all(err == 0 for (_b, _s, err) in results)


def test_gfx942_hss_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of every emitted kernel."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
