################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S01 - KernelWriterAssembly lraOffsetConversionForDTLandNLC characterization.

Drives the designed DirectToLds + NumLoadsCoalesced>1 config
(``data/test_data/_designed/gfx950/s01_lraoffsetconversionfordtlandnlc.yaml``)
through the config-driven emit harness. Targets
``lraOffsetConversionForDTLandNLC`` in ``Tensile/KernelWriterAssembly.py`` -
the 6060-6078 bit-rotation compute block (lines 6061,6071 and the surrounding
DTL/NLC offset conversion), reached when a bf16 TN kernel with DirectToLdsA,
GlobalReadVectorWidthA*bpeDS>4 (b128 DTL, gfx950-only), WaveSeparateGlobalReadA
(forcing UseGeneralizedNLCOneA=False), and NumLoadsCoalescedA>1 is emitted.

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
    "s01_lraoffsetconversionfordtlandnlc.yaml",
)


def test_s01_lraoffsetconversionfordtlandnlc_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    assert_config_emits_golden(_CONFIG, _ARCH, snapshot, limit=8, validate_source=True)
