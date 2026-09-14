################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S02b - KernelWriterAssembly globalReadGuardK Body tail-loop characterization.

Drives the designed non-buffer flat-addressing config
(``data/test_data/_designed/gfx950/s02b_globalreadguardk_body_tail_loop.yaml``)
through the config-driven emit harness. Targets the ``globalReadGuardK`` Body
guarded-read emission in ``Tensile/KernelWriterAssembly.py``:

  - 11029 : the non-BufferLoad else-arm VCmpXLtU64 addr<maxAddr masking, and
  - 11227 : the BufferLoad=0 checkIn of the maxAddr/bpe/zero VGPRs.

``BufferLoad=False`` forces the flat else-arm, and choosing K (=130) not a
multiple of DepthU (=16) makes the tail loop fire ``globalReadDo(mode=2)`` ->
``globalReadGuardK`` Body, arming both target lines during emission.

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
    "s02b_globalreadguardk_body_tail_loop.yaml",
)


def test_s02b_globalreadguardk_body_tail_loop_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the tail-loop emit."""
    assert_config_emits_golden(_CONFIG, _ARCH, snapshot, limit=8, validate_source=True)
