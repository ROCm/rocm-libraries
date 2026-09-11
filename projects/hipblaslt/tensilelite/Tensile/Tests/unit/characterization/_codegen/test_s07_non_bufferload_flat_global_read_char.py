################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S07 - KernelWriter flat (non-buffer) global-read address VGPR characterization.

Drives the designed flat-addressing config
(``data/test_data/_designed/gfx942/s07_non_bufferload_flat_global_read.yaml``)
through the config-driven emit harness. Targets the flat (non-BufferLoad)
global-read address VGPR arms in ``Tensile/KernelWriter.py``:

  - the else-of-BufferLoad ``numVgprGlobalReadAddressesB`` assignment,
  - the ``globalReadIncsUseVgpr`` (flat) ``numVgprGlobalReadIncsB`` assignment,
  - the ``startVgprGlobalReadAddressesA/B`` flat else-branch.

``KernelWriter.py`` derives the flat global-read address layout when
``BufferLoad=False``, so a simple fp32 NN GEMM with ``BufferLoad=False`` forces
those arms during ``assignDerivedParameters`` + emission. The MXSA/MXSB flat
variants are unreachable here (MXBlock forces BufferLoad on gfx942).

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
    "s07_non_bufferload_flat_global_read.yaml",
)


def test_s07_non_bufferload_flat_global_read_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the flat emit."""
    assert_config_emits_golden(_CONFIG, _ARCH, snapshot, limit=8, validate_source=True)
