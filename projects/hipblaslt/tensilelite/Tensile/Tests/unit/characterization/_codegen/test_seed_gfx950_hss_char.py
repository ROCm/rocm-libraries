################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P2-WIDEN — gfx950 HSS (half-in / mixed-precision) family seed (emit smoke).

The dominant F8/StreamK seed does not reach the mixed half-in ProblemType the
HSS attribution entry (marginal 146 lines) covers. This widens the seed set
with the shipped ``Tests/common/gemm/fp16fp32mix.yaml`` (DataTypeA=s, DataType=h
mixed-precision) and asserts it emits real gfx950 assembly with err==0.
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"
_SEED = "Tensile/Tests/common/gemm/fp16fp32mix.yaml"


def test_seed_gfx950_hss_emits_assembly():
    """half-in mixed (HSS) config emits real gfx950 assembly, all err==0."""
    results = emit_kernels_from_config(_SEED, limit=8, arch=_ARCH)
    assert len(results) >= 1
    assert all(err == 0 for (_b, _s, err) in results)


def test_gfx950_hss_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest for the HSS seed."""
    results = emit_kernels_from_config(_SEED, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
