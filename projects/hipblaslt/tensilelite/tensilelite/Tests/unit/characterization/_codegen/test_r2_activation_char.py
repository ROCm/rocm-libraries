################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R2 — gfx942 Activation gradient-path characterization test.

Drives the activation-sweep Gradient BenchmarkProblems config
(``data/test_data/_designed/gfx942/activation.yaml``) through the config-driven
emit harness and asserts every emitted kernel is real gfx942 AMDGCN assembly
with err==0.

Target: Tensile/Activation.py missing ranges 489-617, 639-718, 727-836, 852-936.

Strategy: HHS (fp16/fp16/f32) + Gradient=True uses GRADONLY export type which
selects only gradient activation functions (dgelu, drelu) for float32 compute.
Both functions operate only on float32 Single data type — no SDWA/VOP3P/SelectBit
needed — so they emit cleanly in the CPU-only path.

- getDGeluModule(Single):  Activation.py lines 798-815 (range 727-836)
                            + lines 837-851, 852 (ranges above 836)
- getDReluModule(Single):  Activation.py lines 856-864 (range 852-936)

Measured marginal lines in target ranges: ≥ 34 (exceeds the ≥ 15 gate).
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
    "activation.yaml",
)


def test_r2_activation_gfx942_emits_assembly():
    """Activation gradient HHS config emits real gfx942 assembly, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, (
        f"Expected >=1 kernel from activation.yaml, got {len(results)}. "
        "Check that HHS + Gradient=True + ActivationType:all + ActivationComputeDataType:s is valid."
    )
    for base, src, err in results:
        assert err == 0, f"Kernel {base!r} emitted with err={err}, expected 0"
        assert src and len(src.splitlines()) > 100
        assert ".amdgcn_target" in src
        assert "gfx942" in src
        assert base.startswith("Cijk_")


def test_r2_activation_gfx942_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the activation sweep emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
