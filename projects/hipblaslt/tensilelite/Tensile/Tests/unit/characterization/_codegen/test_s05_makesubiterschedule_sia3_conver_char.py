################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S05 - KernelWriter._makeSubIterSchedule SIA3 ConvertAfterDS characterization.

Drives the designed FP8 ConvertAfterDS config
(``data/test_data/_designed/gfx950/s05_makesubiterschedule_sia3_conver.yaml``)
through the config-driven emit harness. Targets the ``_makeSubIterSchedule``
instPerPack arms in ``Tensile/KernelWriter.py`` (lines 1297,1298): the
ConvertAfterDS FP8 ``lrvwTile`` pack path that fires under
ScheduleIterAlg=3 with an FP8 A-read converted after the local-read (UMLDS
Hascvt convert-after-DS).

``assignDerivedParameters`` + emission both run during ``emit_kernels_from_config``,
so the target lines fire during the emit call.

CPU-only; no GPU, no compile, no hardware. pytestmark = pytest.mark.unit.
"""

import os

import pytest

from config_harness import assert_config_emits_golden

pytestmark = pytest.mark.unit

_ARCH = "gfx950"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx950",
    "s05_makesubiterschedule_sia3_conver.yaml",
)


def test_s05_makesubiterschedule_sia3_conver_golden(snapshot):
    """Golden: order-invariant {basename, err} digest of the emit."""
    assert_config_emits_golden(_CONFIG, _ARCH, snapshot, limit=8, validate_source=True)
