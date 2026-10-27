################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P2-widen — gfx942 fp64 (DB) family seed emit characterization.

The gfx942 attribution ranks ``gfx942__DB_yaml`` (+53 marginal whole-project
lines). fp64 (``DataType: d``) is a distinct data-type family the dominant bf16
seed cannot reach (different MAC/MatrixInstruction component paths). This drives
the existing validated common fp64 config (``Tests/common/gemm/f64.yaml``,
BenchmarkProblems[0] is an NN fp64 GEMM) and asserts every emitted kernel is
real gfx942 assembly with err==0.

No snapshot (that is P3); this proves the family seed emits cleanly.
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

# Existing validated fp64 config; BenchmarkProblems[0] is DataType d (fp64).
_CONFIG = os.path.join(_COMMON, "gemm", "f64.yaml")


def test_seed_gfx942_db_emits_assembly():
    """The fp64 family seed config emits real gfx942 assembly, all err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1
    assert all(err == 0 for (_b, _s, err) in results)


def test_gfx942_db_golden(snapshot):
    """Order-invariant {basename, err} digest golden for the gfx942 fp64 seed."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
