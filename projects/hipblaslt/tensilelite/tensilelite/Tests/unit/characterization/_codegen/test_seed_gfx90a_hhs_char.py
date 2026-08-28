################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P2-WIDEN — gfx90a HHS (Half in / Half out, Single compute) family seed test.

The dominant gfx90a seed is a BF16 (BBS) ProblemType, so the harness (which
reads only ``BenchmarkProblems[0]``) cannot reach the Half-input /
DestDataType=Half codegen surface. Attribution entry ``gfx90a__HHS_yaml`` flagged
15 marginal whole-project lines for the fp16 GlobalWriteBatch packing path.

This drives the shipped fp16 config ``Tensile/Tests/common/gemm/fp16_tn.yaml``
(BenchmarkProblems[0] is DataType=h / DestDataType=h / ComputeDataType=s; not
skipped on gfx90a) through the config harness and asserts the emitted kernels are
real gfx90a AMDGCN assembly, every kernel err==0. No snapshot (P3 records
goldens).
"""

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx90a"
_SEED = "Tensile/Tests/common/gemm/fp16_tn.yaml"
_LIMIT = 6


def test_seed_gfx90a_hhs_emits_assembly():
    """The gfx90a fp16 (HHS) seed emits real AMDGCN assembly, every kernel err==0."""
    results = emit_kernels_from_config(_SEED, limit=_LIMIT, arch=_ARCH)
    assert len(results) >= 1
    assert all(err == 0 for (_b, _s, err) in results)
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 1
        assert base.startswith("Cijk_")


def test_gfx90a_hhs_golden(snapshot):
    """Order-invariant {basename, err} digest golden for the gfx90a HHS seed."""
    results = emit_kernels_from_config(_SEED, limit=_LIMIT, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
