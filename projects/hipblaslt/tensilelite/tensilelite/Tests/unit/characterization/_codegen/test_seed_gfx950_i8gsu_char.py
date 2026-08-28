################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P2-WIDEN — gfx950 I8_GSU (int8 + GlobalSplitU) family seed (emit smoke).

The dominant F8/StreamK seed does not reach the Int8 ProblemType the I8_GSU
attribution entry (marginal 192 lines) covers. This widens the seed set with
the shipped ``Tests/common/gemm/gfx950/i8_gsu_gfx950.yaml`` (i8/i8/s,
GlobalSplitU) and asserts it emits real gfx950 assembly with err==0.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"
_SEED = "Tensile/Tests/common/gemm/gfx950/i8_gsu_gfx950.yaml"


def test_seed_gfx950_i8gsu_emits_assembly():
    """int8 + GSU (I8_GSU) config emits real gfx950 assembly, all err==0."""
    results = emit_kernels_from_config(_SEED, limit=8, arch=_ARCH)
    assert len(results) >= 1
    assert all(err == 0 for (_b, _s, err) in results)


def test_gfx950_i8gsu_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest snapshot."""
    results = emit_kernels_from_config(_SEED, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
