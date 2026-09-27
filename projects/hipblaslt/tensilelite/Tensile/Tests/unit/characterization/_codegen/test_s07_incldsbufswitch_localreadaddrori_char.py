################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################
"""S07 - KernelWriter IncLdsBufSwitch LocalReadAddrOrig backup characterization.

Drives the designed IncLdsBufSwitch config
(``data/test_data/_designed/gfx942/s07_incldsbufswitch_localreadaddrori.yaml``)
through the config-driven emit harness. Targets the
``self.states.IncLdsBufSwitch`` backup of ``startVgprLocalReadAddrOrig`` for
the A/B arms in ``Tensile/KernelWriter.py`` (lines 8690-8697).

``IncLdsBufSwitch`` is derived True when ``NumLdsBlk >= 3`` (KernelWriter.py),
which follows from ``PrefetchGlobalRead >= 3`` with the DirectToLds recipe.
The MXSA/MXSB arms of the same backup are unreachable on gfx942 (no MX), so
only the A/B arms fire here. ``assignDerivedParameters`` + emission run during
the emit call, so the target lines fire.

CPU-only; no GPU, no compile, no hardware. pytestmark = pytest.mark.unit.
"""

import os

import pytest

from config_harness import assert_config_emits

pytestmark = pytest.mark.unit

_ARCH = "gfx942"

_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "data",
    "test_data",
    "_designed",
    "gfx942",
    "s07_incldsbufswitch_localreadaddrori.yaml",
)


def test_s07_incldsbufswitch_localreadaddrori_emits():
    """The selected configuration emits valid assembly."""
    assert_config_emits(_CONFIG, _ARCH, limit=8, validate_source=True)
