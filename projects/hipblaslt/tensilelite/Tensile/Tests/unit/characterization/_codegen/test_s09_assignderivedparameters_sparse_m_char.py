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

from config_harness import assert_config_emits

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
    """The selected configuration emits valid assembly."""
    assert_config_emits(_CONFIG, _ARCH, limit=8, validate_source=True)
