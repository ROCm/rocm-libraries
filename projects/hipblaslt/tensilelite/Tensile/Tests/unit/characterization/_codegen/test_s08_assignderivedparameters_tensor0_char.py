################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S08 - Solution.assignDerivedParameters sparse-B metadata characterization.

Drives the designed Sparse=2 (2:4 sparse B) config
(``data/test_data/_designed/gfx942/s08_assignderivedparameters_tensor0.yaml``)
through the config-driven emit harness. Targets the sparse-B metadata cluster in
``Tensile/SolutionStructs/Solution.py`` ``assignDerivedParameters``:

  - line 2069: ``Sparse==2 and DirectToVgprSparseMetadata`` reject arm (Sparse B
    does not support DTVSM), hit by the ``DirectToVgprSparseMetadata:True`` fork,
  - lines 2076-2085: ``Sparse==2`` non-DTVSM arm that copies the metadata thread
    tile from the B side, hit by the ``DirectToVgprSparseMetadata:False`` fork.

``assignDerivedParameters`` runs during emission, so the target lines fire during
the ``emit_kernels_from_config`` call.

CPU-only; no GPU, no compile, no hardware. pytestmark = pytest.mark.unit.
"""

import os

import pytest

from config_harness import assert_config_emits_golden

pytestmark = pytest.mark.unit

_ARCH = "gfx942"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
    "s08_assignderivedparameters_tensor0.yaml",
)


def test_s08_assignderivedparameters_tensor0_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the sparse emit."""
    assert_config_emits_golden(_CONFIG, _ARCH, snapshot, limit=8, validate_source=True)
