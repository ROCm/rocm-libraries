################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P2 — gfx950 designed-seed codegen-emit smoke (no golden yet; P3 records it).

Drives the minimal P2 ``BenchmarkProblems`` seed config
(``data/_designed/gfx950/seed.yaml``) through the config-driven emit harness
(:func:`config_harness.emit_kernels_from_config`) and asserts every emitted
kernel is real AMDGCN assembly for gfx950 with ``err == 0``.

The seed encodes the highest-yield gfx950 attribution axes (StreamK + F8 +
CustomSchedule from the dominant entry, plus GlobalSplitU and a second
MatrixInstruction/LocalRead shape) in a single ForkParameters sweep that emits
<= 8 kernels.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"
_LIMIT = 8
_SEED = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    _ARCH,
    "seed.yaml",
)


def test_seed_gfx950_emits_assembly():
    """The designed seed emits real gfx950 AMDGCN assembly, all err == 0."""
    results = emit_kernels_from_config(_SEED, limit=_LIMIT, arch=_ARCH)
    assert len(results) >= 1
    assert all(err == 0 for (_b, _s, err) in results)
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 1000
        assert ".amdgcn_target" in src
        assert "gfx950" in src
        assert base.startswith("Cijk_")


def test_gfx950_dominant_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest for the gfx950 dominant seed."""
    results = emit_kernels_from_config(_SEED, limit=_LIMIT, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
