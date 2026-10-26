################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P2-widen — gfx942 GroupedGemm (GG) family seed emit characterization.

The gfx942 attribution ranks ``gfx942__GG_yaml`` third (+448 marginal
whole-project lines). GroupedGemm (``GroupedGemm: True``) is a distinct
ProblemType the dominant single-problem bf16 seed cannot reach. This drives the
existing validated common GroupedGemm config
(``Tests/common/groupedgemm/grouped_gemm.yaml``, BenchmarkProblems[0] is an
fp16 GroupedGemm + Bias + Activation) and asserts every emitted kernel is real
gfx942 assembly with err==0.

P3 adds an order-invariant {basename, err} snapshot golden for this seed.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx942"

# Tests/common dir: _codegen -> characterization -> unit -> Tests, then common.
_COMMON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "common",
)

# Existing validated GroupedGemm config; BenchmarkProblems[0] carries the
# GroupedGemm flag the dominant seed lacks.
_CONFIG = os.path.join(_COMMON, "groupedgemm", "grouped_gemm.yaml")

_LIMIT = 8


def test_seed_gfx942_gg_emits_assembly():
    """The GroupedGemm family seed config emits real gfx942 assembly, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=_LIMIT, arch=_ARCH)
    assert len(results) >= 1
    assert all(err == 0 for (_b, _s, err) in results)


def test_gfx942_gg_golden(snapshot):
    """Order-invariant {basename, err} digest golden for the GG family seed (P3)."""
    results = emit_kernels_from_config(_CONFIG, limit=_LIMIT, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
