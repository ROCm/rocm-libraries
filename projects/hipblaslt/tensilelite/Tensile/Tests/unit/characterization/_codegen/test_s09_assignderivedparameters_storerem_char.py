# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
################################################################################
"""S09 - Solution.assignDerivedParameters StoreRemap characterization.

Drives the designed StoreRemap config
(``data/test_data/_designed/gfx942/s09_assignderivedparameters_storerem.yaml``)
through the config-driven emit harness. Targets the StoreRemapVectorWidth
derivation and reject sites in ``Tensile/SolutionStructs/Solution.py``
``assignDerivedParameters``:

  - the ``storeRemap: Per wave single global write ... one M column`` reject
    where ``SRVW*WavefrontSize < MacroTile0`` (small SRVW, large MT0).

The bf16 BBS TN config forks two MatrixInstruction solutions: one passes the
guard and derives a valid kernel, the other trips the early-return reject.
``emit_kernels_from_config`` runs ``assignDerivedParameters`` plus emission, so
the target lines fire during the emit call.

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
    "s09_assignderivedparameters_storerem.yaml",
)


def test_s09_assignderivedparameters_storerem_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the StoreRemap emit."""
    assert_config_emits_golden(_CONFIG, _ARCH, snapshot, limit=8, validate_source=True)
