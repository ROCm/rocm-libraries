################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S02b - KernelWriterAssembly globalReadBody non-guardK prefetch characterization.

Drives the designed non-BufferLoad flat global-read config
(``data/test_data/_designed/gfx942/s02b_globalreadbody_non_guardk_prefet.yaml``)
through the config-driven emit harness. Targets the ``globalReadDo`` (mode=1)
prefetch load arms in ``Tensile/KernelWriterAssembly.py``, specifically the
non-BufferLoad flat global-read address path (line 11810).

The ``config_harness`` derives only ``BenchmarkProblems[0]``; the leading entry
of the config isolates the flat (BufferLoad=False) path so the target lines fire
during ``assignDerivedParameters`` + emission.

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
    "s02b_globalreadbody_non_guardk_prefet.yaml",
)


def test_s02b_globalreadbody_non_guardk_prefet_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the flat emit."""
    assert_config_emits_golden(_CONFIG, _ARCH, snapshot, limit=8, validate_source=True)
