################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P2-WIDEN — gfx950 BBS (bf16) family seed (codegen-emit smoke + P3 golden).

The dominant gfx950 seed is an F8/StreamK config; because the config harness
reads only ``BenchmarkProblems[0]`` it cannot reach the distinct BF16
ProblemType/DataType that the BBS attribution entry (marginal 1277 lines)
covers. This widens the gfx950 seed set with one BF16 (b/b/s) GEMM config from
``Tests/common/gemm/bf16_tn.yaml`` and asserts it emits real gfx950 assembly
with err==0.
"""

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"
_SEED = "Tensile/Tests/common/gemm/bf16_tn.yaml"
_LIMIT = 8


def test_seed_gfx950_bbs_emits_assembly():
    """BF16 (BBS) config emits real gfx950 assembly, all err==0."""
    results = emit_kernels_from_config(_SEED, limit=_LIMIT, arch=_ARCH)
    assert len(results) >= 1
    assert all(err == 0 for (_b, _s, err) in results)


def test_gfx950_bbs_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the BBS emit."""
    results = emit_kernels_from_config(_SEED, limit=_LIMIT, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
