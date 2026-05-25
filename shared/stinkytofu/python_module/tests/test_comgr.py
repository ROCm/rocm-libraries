# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for comgr-based toolchain capability probing."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../build/lib"))

import stinkytofu


class TestComgrSupport:
    def test_has_comgr_support(self):
        assert stinkytofu.hasComgrSupport() is True


class TestProbeToolchainCaps:
    def test_gfx1250_returns_dict(self):
        caps = stinkytofu.probeToolchainCaps([12, 5, 0])
        assert isinstance(caps, dict)
        assert "VgprMsbMode" in caps

    def test_gfx1250_vgpr_msb_mode_non_zero(self):
        caps = stinkytofu.probeToolchainCaps([12, 5, 0])
        assert caps["VgprMsbMode"] != stinkytofu.VgprMsbMode.NONE.value

    def test_gfx1250_vgpr_msb_mode_is_msb8_or_msb16(self):
        caps = stinkytofu.probeToolchainCaps([12, 5, 0])
        assert caps["VgprMsbMode"] in (
            stinkytofu.VgprMsbMode.MSB8.value,
            stinkytofu.VgprMsbMode.MSB16.value,
        )


class TestVgprMsbModeEnum:
    def test_enum_values(self):
        assert stinkytofu.VgprMsbMode.NONE.value == 0
        assert stinkytofu.VgprMsbMode.MSB8.value == 1
        assert stinkytofu.VgprMsbMode.MSB16.value == 2
