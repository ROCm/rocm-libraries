################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S06 - KernelWriter _loopBody DTV/pack bufferIdx off characterization.

Drives the designed F32X UsePLRPack config
(``data/test_data/_designed/gfx950/s06_loopbody_dtv_pack_bufferidx_off.yaml``)
through the config-driven emit harness. Targets the main-loop body block in
``Tensile/KernelWriter.py`` (~4515-4711) plus the ``doFullPackCodePrefetch`` /
``usePLRPack`` ripple:

  - the SubTileIdx / DTV pack bufferIdx offset arms (~4516-4520, 4629, 4709,
    4711), and
  - the full-pack-code-prefetch derivation these depend on (~9006-9042,
    9887-9925).

``doFullPackCodePrefetch = UsePLRPack and not UseCustomMainLoopSchedule`` is
only kept when ``UseF32XEmulation and ForceUnrollSubIter and SIA3 and MI``, so
this F32X NT config on gfx950 (HasMFMA -> MFMA F32X emulation, ExpandPointerSwap
off, DepthU==MatrixInstK==32, even MIWaveTile) reaches the arms. ``emit`` runs
``assignDerivedParameters`` + emission so the target lines fire during emit.

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
    "s06_loopbody_dtv_pack_bufferidx_off.yaml",
)


def test_s06_loopbody_dtv_pack_bufferidx_off_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    assert_config_emits_golden(_CONFIG, _ARCH, snapshot, limit=8, validate_source=True)
