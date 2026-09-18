################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S00 - KernelWriterAssembly macroAndSet flat (non-buffer) characterization.

Drives the designed flat-addressing config
(``data/test_data/_designed/gfx942/kwa_macroandset_flat.yaml``) through the
config-driven emit harness. Targets the ``macroAndSet`` RegSet emission in
``Tensile/KernelWriterAssembly.py``:

  - the non-BufferLoad else-arm that assigns ``vgprGlobalReadAddrA/B``, and
  - the ``globalReadIncsUseVgpr`` RegSets assigning ``vgprGlobalReadIncsA/B``,

plus the flat-addressing emit arms these RegSets feed downstream
(graFinalOffsets / globalReadInc). ``KernelWriter.py`` sets
``globalReadIncsUseVgpr = False if kernel["BufferLoad"] else True``, so
``BufferLoad=False`` forces both arms. The MXSA/MXSB variants of the same
RegSets are unreachable here (Solution derivation forces BufferLoad for
MXBlock), so a MX config can never take the flat else-branch.

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
    "kwa_macroandset_flat.yaml",
)


def test_s00_macroandset_flat_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the flat emit."""
    assert_config_emits_golden(_CONFIG, _ARCH, snapshot, limit=8, validate_source=True)
