################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S08 - Solution.py depthUIteration cluster characterization.

Drives the designed config
(``data/test_data/_designed/gfx1250/s08_depthuiteration_multi_du_fp6_lds.yaml``)
through the config-driven emit harness. The emit path runs
``assignDerivedParameters`` before emission, so the ``depthUIteration`` cluster
in ``Tensile/SolutionStructs/Solution.py`` (~3081-3093) fires during the emit
call. The config combines a TDM auto (-1) iterate-mode resolution group, a
TDM explicit-mask reject group, and an fp6 LdsPad clamp/reject group so the
derivation exercises the currently-missing arms of that block.

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
    "s08_depthuiteration_multi_du_fp6_lds.yaml",
)


def test_s08_depthuiteration_multi_du_fp6_lds_golden(snapshot):
    """P3 golden: order-invariant {basename, err} digest of the emit."""
    assert_config_emits_golden(_CONFIG, _ARCH, snapshot, limit=8, validate_source=True)
