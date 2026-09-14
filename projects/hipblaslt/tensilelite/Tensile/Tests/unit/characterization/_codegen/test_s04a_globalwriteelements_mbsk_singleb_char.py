################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S04a - KernelWriterAssembly globalWriteElements MBSK/SingleB characterization.

Drives the designed AdaptiveGemmGSUA=1 config
(``data/test_data/_designed/gfx942/s04a_globalwriteelements_mbsk_singleb.yaml``)
through the config-driven emit harness. Targets the AdaptiveGemmGSUA==1
else-branch label-wiring arms in ``Tensile/KernelWriterAssembly.py``
(``globalWriteElements``):

  - the per-algorithm mode selection (MultipleBuffer / MultipleBufferSingleKernel
    / SingleBuffer), including the ``DataType.isDouble()`` guard, and
  - the MBSK / MB label emission loop plus its AdaptiveGemmGSUA==1 tail restore.

The config forks all three GSU algorithms with AdaptiveGemmGSUA=1 and
GlobalSplitU>0 (=> noGSUBranch False) using a half datatype (MBSK rejects
double). ``emit`` runs assignDerivedParameters + emission, so these lines fire
during the emit call.

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
    "s04a_globalwriteelements_mbsk_singleb.yaml",
)


def test_s04a_globalwriteelements_mbsk_singleb_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    assert_config_emits_golden(_CONFIG, _ARCH, snapshot, limit=8, validate_source=True)
