################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S10 - LocalRead MFMA main-path metadata VPerm K-packing characterization.

Drives the designed sparse metadata config
(``data/test_data/_designed/gfx942/s10_mfma_main_path_metadata_vperm_k.yaml``)
through the config-driven emit harness. Targets the ``numSplitMetadata`` block
in ``Tensile/Components/LocalRead.py`` (1428-1534) plus the surrounding
metadata VPerm K-packing arms:

  - reach ``elif lrvwTile > 1 and not UseF32XEmulation`` (LR:1412) with sparse
    metadata (lrvwTileMetadata = VectorWidthA = 2 > 1), and
  - branch on ``MIInputPerThUnroll`` (==2 -> LR:1518, ==1 -> LR:1530) via the
    16x16x32 / 16x16x64 MatrixInstruction forks.

Sparse==1 with DirectToVgprSparseMetadata=False and TransposeLDS=0 forces the
metadata local-read pack path; ``emit`` runs assignDerivedParameters +
emission so these lines fire during the emit call.

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
    "s10_mfma_main_path_metadata_vperm_k.yaml",
)


def test_s10_mfma_main_path_metadata_vperm_k_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    assert_config_emits_golden(_CONFIG, _ARCH, snapshot, limit=8, validate_source=True)
