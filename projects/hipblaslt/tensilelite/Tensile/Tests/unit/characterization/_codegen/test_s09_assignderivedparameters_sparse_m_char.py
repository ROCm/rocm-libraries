################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S09 - Solution.assignDerivedParameters Sparse metadata characterization.

Drives the designed Sparse==2 (SparseB) config
(``data/test_data/_designed/gfx942/s09_assignderivedparameters_sparse_m.yaml``)
through the config-driven emit harness. The coverage target is
``Tensile/SolutionStructs/Solution.py`` ``assignDerivedParameters``: the
Sparse==2 GRVW/GLT derivation, the partialM branch, the ``<glvwMlimit`` GRVW
fallback sub-branch, and the DirectToLdsMetadata block for ``sparseTc='B'``.

``DirectToVgprSparseMetadata=0`` keeps the LDS metadata path, and the
``DirectToLdsMetadata: [0, 1]`` fork exercises both metadata routes. Emission
runs ``assignDerivedParameters`` followed by kernel emission, so the target
lines fire during the ``emit_kernels_from_config`` call.

CPU-only; no GPU, no compile, no hardware. pytestmark = pytest.mark.unit.
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
    "s09_assignderivedparameters_sparse_m.yaml",
)


def test_s09_assignderivedparameters_sparse_m_emits():
    """Sparse==2 (SparseB) config emits kernels and all have err==0."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    assert len(results) >= 1, f"expected >=1 kernel, got {len(results)}"
    for base, src, err in results:
        assert err == 0, f"kernel {base!r} emitted with err={err}"
        assert base.startswith("Cijk_")
        assert ".amdgcn_target" in src, f"kernel {base!r}: missing .amdgcn_target"
        assert "gfx942" in src, f"kernel {base!r}: wrong arch in assembly"


def test_s09_assignderivedparameters_sparse_m_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the sparse emit."""
    results = emit_kernels_from_config(_CONFIG, limit=8, arch=_ARCH)
    digest = sorted(
        ({"basename": b, "err": e} for (b, _s, e) in results),
        key=lambda d: d["basename"],
    )
    assert digest == snapshot
