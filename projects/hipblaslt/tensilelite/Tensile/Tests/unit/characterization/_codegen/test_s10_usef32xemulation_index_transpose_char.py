################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S10 - LocalRead UseF32XEmulation index-transpose characterization.

Drives the designed gfx1250 UseF32XEmulation config
(``data/test_data/_designed/gfx1250/s10_usef32xemulation_index_transpose.yaml``)
through the config-driven emit harness. Targets the F32X pack / index-transpose
paths in ``Tensile/Components/LocalRead.py``.

``UseF32XEmulation`` (DataType=S + F32XdlMathOp=X + HPA on gfx1250 WMMA_V3)
forces ``needPack`` and the F32X pack loop; ``VectorWidthA/B=2`` gives
``lrvwTile>1`` (index-transpose candidate); the widened tile / DepthU grow
``numReadsPerUnroll`` / ``numVgpr`` so ``multiGroupXF32`` engages. ``emit`` runs
``assignDerivedParameters`` + emission, so the target LocalRead lines fire during
the emit call.

CPU-only; no GPU, no compile, no hardware. pytestmark = pytest.mark.unit.
"""

import os

import pytest

from config_harness import assert_config_emits_golden

pytestmark = pytest.mark.unit

_ARCH = "gfx1250"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx1250",
    "s10_usef32xemulation_index_transpose.yaml",
)


def test_s10_usef32xemulation_index_transpose_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    assert_config_emits_golden(_CONFIG, _ARCH, snapshot, limit=8, validate_source=True)
