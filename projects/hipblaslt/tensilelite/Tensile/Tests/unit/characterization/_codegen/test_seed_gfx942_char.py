################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P2 — gfx942 designed-seed config emit characterization (no golden yet).

Drives the P2 designed BenchmarkProblems config
(``data/_designed/gfx942/seed.yaml``) through the config-driven emit harness
(``emit_kernels_from_config``) and asserts every emitted kernel is real
gfx942 AMDGCN assembly with err==0. The dominant attribution entry for gfx942
is a bf16 BH + Bias + Activation GEMM; this seed reproduces it and adds a tiny
MatrixInstruction x DirectToVgprA fork (4 kernels) for marginal coverage.

P3 records goldens; this phase only proves the design emits cleanly.
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
    "seed.yaml",
)


def test_seed_gfx942_emits_assembly():
    """The designed seed config emits real gfx942 assembly, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1
    assert all(err == 0 for (_b, _s, err) in results)
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 1000
        # Sanity: real AMDGCN assembly for the expected target.
        assert ".amdgcn_target" in src
        assert "gfx942" in src
        assert base.startswith("Cijk_")


def test_gfx942_dominant_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the gfx942 seed emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
