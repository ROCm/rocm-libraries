################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""P2 — gfx90a (aldebaran, CDNA2/MFMA) designed-seed emit test.

Drives the minimal designed BenchmarkProblems config
(``data/_designed/gfx90a/seed.yaml``) through the config harness
(``BenchmarkProcess -> constructForkPermutations -> _generateForkedSolutions ->
emit``) and asserts the emitted kernels are real AMDGCN assembly for gfx90a.

The config is a single BBS (BF16/BF16 -> Single) ProblemType — the dominant
attribution entry for gfx90a — forked over two MatrixInstruction tile shapes and
GlobalSplitU 1/2 (4 permutations -> 4 kernels). No snapshot yet; P3 records the
goldens.
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
    "seed.yaml",
)


def test_seed_gfx90a_emits_assembly():
    """The designed gfx90a seed emits real AMDGCN assembly, every kernel err==0."""
    results = emit_kernels_from_config(_SEED, limit=4, arch=_ARCH)
    assert len(results) >= 1
    assert all(err == 0 for (_b, _s, err) in results)
    for base, src, _err in results:
        assert src and len(src.splitlines()) > 1000
        # Sanity: real AMDGCN assembly targeting gfx90a.
        assert ".amdgcn_target" in src
        assert "gfx90a" in src
        assert base.startswith("Cijk_")


def test_gfx90a_dominant_golden(snapshot):
    """Order-invariant {basename, err} digest snapshot for gfx90a dominant seed (P3 golden)."""
    results = emit_kernels_from_config(_SEED, limit=4, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
