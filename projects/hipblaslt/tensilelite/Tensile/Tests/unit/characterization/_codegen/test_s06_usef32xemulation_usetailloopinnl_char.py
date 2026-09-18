################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S06 - KernelWriter UseF32XEmulation + useTailloopInNll characterization.

Drives the designed TF32-emulation config
(``data/test_data/_designed/gfx950/s06_usef32xemulation_usetailloopinnl.yaml``)
through the config-driven emit harness. Targets the noLoadLoopBody
TF32-pack-after-ShiftK block in ``Tensile/KernelWriter.py``
(``if kernel["UseF32XEmulation"] and useTailloopInNll:``) that interleaves the
packA/packB Pre + pack items with searchStrings ``__TF32_1``/``__TF32_2``.

gfx950 is the sole arch that reaches this block: it has HasF32XEmulation=1,
MFMA=1, WMMA=0, and HasMFMA_xf32=0, so both ``UseF32XEmulation`` and
``useTailloopInNll`` survive Solution derivation. gfx1250 (WMMA) force-disables
TailloopInNll; gfx942 has native xf32 and never enables emulation.

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
    "s06_usef32xemulation_usetailloopinnl.yaml",
)


def test_s06_usef32xemulation_usetailloopinnl_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    assert_config_emits_golden(_CONFIG, _ARCH, snapshot, limit=8, validate_source=True)
