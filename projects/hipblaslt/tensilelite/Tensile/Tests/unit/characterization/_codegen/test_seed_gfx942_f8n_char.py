################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P2-widen — gfx942 F8N (fp8) family seed emit characterization.

The gfx942 attribution (``attribution-gfx942.json``) ranks ``gfx942__F8N_multi_yaml``
second (+1133 marginal whole-project lines) — a distinct fp8 (``DataType: F8N``)
ProblemType the dominant bf16 seed cannot reach. This drives an existing,
validated common F8N config (``Tests/common/gemm/fp8n.yaml``, whose first
BenchmarkProblems entry is an F8N GEMM with ScaleAB/ScaleCD/ScaleAlphaVec +
Bias + Activation) through the config-driven emit harness and asserts every
emitted kernel is real gfx942 assembly with err==0.

No snapshot (that is P3); this proves the family seed emits cleanly.
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

# Existing validated F8N config; harness reads BenchmarkProblems[0] (the F8N
# GEMM ProblemType), so this seeds the fp8 family the dominant bf16 seed misses.
_CONFIG = os.path.join(_COMMON, "gemm", "fp8n.yaml")


def test_seed_gfx942_f8n_emits_assembly():
    """The F8N family seed config emits real gfx942 assembly, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1
    assert all(err == 0 for (_b, _s, err) in results)


def test_gfx942_f8n_golden(snapshot):
    """Order-invariant {basename, err} digest golden for the F8N family seed."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
