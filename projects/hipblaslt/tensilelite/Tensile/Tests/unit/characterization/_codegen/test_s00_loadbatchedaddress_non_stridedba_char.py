################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S00 - KernelWriterAssembly loadBatchedAddress non-StridedBatched characterization.

Drives the designed non-StridedBatched config
(``data/test_data/_designed/gfx942/s00_loadbatchedaddress_non_stridedba.yaml``)
through the config-driven emit harness. Targets ``loadBatchedAddress`` in
``Tensile/KernelWriterAssembly.py``, which is emitted only when
``not kernel["ProblemType"]["StridedBatched"]`` and dereferences an array of
buffer pointers for C/D, Beta-C (UseBeta), and A/B.

On gfx942 (MFMA, no ``RequiresXCntForVolatileVMEM``) the emit takes the else
arms (SLoadB64) for the D, C, and A/B pointer loads; the XCnt arms need a
gfx1250 twin. ``assignDerivedParameters`` + emission run during the emit call,
so the target lines fire without GPU, compile, or hardware.

CPU-only; pytestmark = pytest.mark.unit.
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
    "s00_loadbatchedaddress_non_stridedba.yaml",
)


def test_s00_loadbatchedaddress_non_stridedba_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    assert_config_emits_golden(_CONFIG, _ARCH, snapshot, limit=8, validate_source=True)
