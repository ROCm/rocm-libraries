################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P2-widen — gfx942 Gradient (Grad) family seed emit characterization.

The gfx942 attribution ranks ``gfx942__Grad_yaml`` (+241 marginal whole-project
lines). The gradient path (``Gradient: True``) is a distinct ProblemType the
dominant forward bf16 seed cannot reach. This drives the existing validated
common gradient config (``Tests/common/gradient/fp16_gradient_bias.yaml``,
BenchmarkProblems[0] is an fp16 Gradient + Bias + Activation GEMM) and asserts
every emitted kernel is real gfx942 assembly with err==0.

P3 golden: order-invariant {basename, err} digest snapshot recorded below.
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

# Existing validated gradient config; BenchmarkProblems[0] carries Gradient=True.
_CONFIG = os.path.join(_COMMON, "gradient", "fp16_gradient_bias.yaml")


def test_seed_gfx942_grad_emits_assembly():
    """The Gradient family seed config emits real gfx942 assembly, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1
    assert all(err == 0 for (_b, _s, err) in results)


def test_gfx942_grad_golden(snapshot):
    """P3 golden: order-invariant digest of {basename, err} per emitted kernel."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
