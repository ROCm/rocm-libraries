################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P2-WIDEN — gfx90a DB (FP64/Double) family seed emit test.

The dominant gfx90a seed is a BF16 (BBS) single-ProblemType config, so the
harness (which reads only ``BenchmarkProblems[0]``) cannot reach the
DataType=Double codegen surface. Attribution entry ``gfx90a__DB_yaml`` flagged
660 marginal whole-project lines for the FP64 path (the [4,4,4] FP64 MFMA,
ShiftVectorComponents / edge-store handling).

This drives the shipped, gfx90a-dedicated FP64 config
``Tensile/Tests/common/gemm/f64_gfx90a.yaml`` (BenchmarkProblems[0] is
DataType=d) through the config harness and asserts the emitted kernels are real
gfx90a AMDGCN assembly, every kernel err==0. No snapshot (P3 records goldens).
"""

import os

import pytest

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx90a"
_SEED = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx90a",
    "db.yaml",
)


def test_seed_gfx90a_db_emits_assembly():
    """The gfx90a FP64 (DB) seed emits real AMDGCN assembly, every kernel err==0."""
    results = emit_kernels_from_config(_SEED, limit=6, arch=_ARCH)
    assert len(results) >= 1
    assert all(err == 0 for (_b, _s, err) in results)
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 1
        assert base.startswith("Cijk_")


def test_gfx90a_db_golden(snapshot):
    """Order-invariant {basename, err} digest for the gfx90a DB seed (P3 golden)."""
    results = emit_kernels_from_config(_SEED, limit=6, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
