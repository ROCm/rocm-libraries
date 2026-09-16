################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S06 - KernelWriter AdaptiveGemmNTAB 3-way NT dispatch characterization.

Drives the designed AdaptiveGemmNTAB config
(``data/test_data/_designed/gfx942/s06_adaptivegemmntab_3_way_nt_dispat.yaml``)
through the config-driven emit harness. Targets the ``kernelBody``
AdaptiveGemmNTAB!=0 branch in ``Tensile/KernelWriter.py``:

  - builds the 3-way NT combo dispatch ``[[0,0],[0,4],[4,0]]``,
  - emits the bit-extract SAndB32,
  - snapshots states/pack/tPA/tPB, restores per-combo, and
  - loops emitting each ``_kernelBody``.

The default AdaptiveGemmNTAB==0 path calls ``_kernelBody`` once; setting
AdaptiveGemmNTAB=1 forces the else branch, firing miss lines 5894, 5911, 5930,
5943, 5951, 5957, 5960. ``emit`` runs assignDerivedParameters + emission so the
target lines fire during the emit call.

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
    "s06_adaptivegemmntab_3_way_nt_dispat.yaml",
)


def test_s06_adaptivegemmntab_3_way_nt_dispat_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    assert_config_emits_golden(_CONFIG, _ARCH, snapshot, limit=8, validate_source=True)
