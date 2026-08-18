################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""R2 — gfx942 solution-derivation breadth characterization.

Drives the R2 designed BenchmarkProblems config
(``data/_designed/gfx942/solution.yaml``) through the config-driven emit
harness (``emit_kernels_from_config``) to exercise the wide ForkParameters
cartesian sweeping DepthU, VectorWidthA, GlobalSplitU, and PrefetchGlobalRead.

Target lines in Tensile/SolutionStructs/Solution.py:
  - 500-680 : assignProblemIndependentDerivedParameters (MIBlock tile assignment,
    MacroTile, UseDotInstruction, EnableMatrixInstruction branch arms)
  - 702-900 : tailLoopOpt paths, MX/NonDTL guard gates, TransposeLDS/UnrollMajorLDS
    derivation, VectorWidth auto-derive from MIWaveTile ratio
  - 1024-1300: setGlobalLoadTileDimClassic NLC search, isDirectToVgprDoable checks
    (PrefetchGlobalRead==0 reject, TLU+VW!=GRVW reject, ASEM-set arm)

Many fork permutations will be rejected by Solution validation — that rejection
code (reject() calls in the ranges above) is part of the target surface.
The test accepts any mix of err==0 valid kernels and err!=0 rejects (which map
to no-src emit), but requires >=1 valid kernel overall.

pytestmark = pytest.mark.unit.  CPU-only; no GPU, no compile.
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
    "solution.yaml",
)


def test_r2_solution_gfx942_emits_assembly():
    """The R2 solution-breadth config emits >=1 valid gfx942 kernel (err==0)."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1
    # At least one kernel must succeed (err==0).
    assert any(err == 0 for (_b, _s, err) in results), (
        "Expected >=1 err==0 kernel; got " + str([(b, e) for b, _, e in results])
    )
    for base, src, err in results:
        if err == 0:
            # Valid kernels must be real gfx942 AMDGCN assembly.
            assert src and len(src.splitlines()) > 100
            assert ".amdgcn_target" in src
            assert "gfx942" in src
            assert base.startswith("Cijk_")


def test_r2_solution_gfx942_golden(snapshot):
    """R2 golden: order-invariant {basename, err} digest of the solution emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
