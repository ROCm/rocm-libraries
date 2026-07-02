################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for Tensile.Common.Architectures registration entries.

Verifies that gfx1032 (RDNA2, Navi 23, e.g. RX 6650 XT) is present in the
architecture map and the supported-ISA list. See issue #1202.
"""

import pytest

from Tensile.Common.Architectures import SUPPORTED_ISA, architectureMap
from Tensile.Common.Types import IsaVersion

pytestmark = pytest.mark.unit


class TestGfx1032Registration:
    def test_gfx1032_in_architecture_map(self):
        assert "gfx1032" in architectureMap

    def test_gfx1032_maps_to_navi23(self):
        assert architectureMap["gfx1032"] == "navi23"

    def test_isa_version_10_3_2_in_supported_isa(self):
        assert IsaVersion(10, 3, 2) in SUPPORTED_ISA

    def test_gfx1030_still_registered(self):
        # Regression guard: gfx1030 (Navi 21) was already registered and
        # must not be lost when gfx1032 is added.
        assert "gfx1030" in architectureMap
        assert architectureMap["gfx1030"] == "navi21"
        assert IsaVersion(10, 3, 0) in SUPPORTED_ISA
