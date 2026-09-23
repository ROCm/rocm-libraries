################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R4 — gfx942 Gradient+BiasSrc-A+UseScaleAB characterization test.

Drives the grad_biassrc_a.yaml config (HHS TN, Gradient=True, BiasSrc=A,
UseScaleAB=Scalar) through the config-driven emit harness and asserts
every emitted kernel is real gfx942 AMDGCN assembly with err==0.

Target: KernelWriter.py gradient/multi-bias arms:
  - line 2895: initSumUnroll call (Gradient+UseBias+BiasSrc A)
  - line 3517: loopSum call in noLoadLoop NLL path
  - line 4361: loopSum call in main loop body
  - line 4873: initSumUnroll in shadow init path (doShadowInit==2)
  - line 5813: loopSum call in tail loop
  - line 6101: biasSumUnroll assignment + assert (Gradient+BiasSrc A/B)
  - line 7941: numVgprValu = MIWaveTile[0] (BiasSrc==A branch)
  - lines 8673-8677: preloadScaleA/B logic (UseScaleAB==Scalar)

Strategy: HHS (fp16/fp16/f32) TN layout with Gradient=True, UseBias=1,
BiasSrc=A, UseScaleAB=Scalar drives the gradient bias-sum-unroll arms plus
the scalar scale preload path. No Activation avoids extra complexity.
Single MI shape + GSU=1 produces 1 kernel.

Measured marginal KernelWriter.py lines in target ranges: >= 15.
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
    "grad_biassrc_a.yaml",
)


def test_r4_gradbias_gfx942_emits_assembly():
    """Gradient+BiasSrc-A+UseScaleAB=Scalar config emits real gfx942 assembly, err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, (
        f"Expected >=1 kernel from grad_biassrc_a.yaml, got {len(results)}. "
        "Check that HHS TN + Gradient=True + UseBias=1 + BiasSrc=A + UseScaleAB=Scalar is valid."
    )
    for base, src, err in results:
        assert err == 0, f"Kernel {base!r} emitted with err={err}, expected 0"
        assert src and len(src.splitlines()) > 100, (
            f"Kernel {base!r} source too short ({len(src.splitlines())} lines)"
        )
        assert ".amdgcn_target" in src, f"Kernel {base!r} missing .amdgcn_target"
        assert "gfx942" in src, f"Kernel {base!r} missing gfx942 arch marker"
        assert base.startswith("Cijk_"), f"Unexpected basename: {base!r}"
