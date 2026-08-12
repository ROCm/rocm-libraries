# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for caps and StinkyTofu arch-probe forwarding."""

import os
import sys
import unittest

_PKG_PARENT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import rocisa_stinkytofu_adaptor as rocisa  # noqa: E402
from rocisa_stinkytofu_adaptor import caps  # noqa: E402

_GFX1250 = (12, 5, 0)


def _stinkytofu_available() -> bool:
    try:
        import stinkytofu  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_stinkytofu_available(), "stinkytofu binding not on PYTHONPATH")
class TestGetCapsDynamic(unittest.TestCase):
    def test_gfx1250_shape(self):
        asm, arch, reg, bugs = caps.getCaps(_GFX1250)
        self.assertIsInstance(asm, dict)
        self.assertIsInstance(arch, dict)
        self.assertIsInstance(reg, dict)
        self.assertIsInstance(bugs, dict)
        self.assertIn("SupportedISA", asm)
        self.assertIn("HasWave32", arch)
        self.assertIn("MaxVgpr", reg)

    def test_gfx1250_modifier_caps_present(self):
        asm, _, _, _ = caps.getCaps(_GFX1250)
        for key in ("HasTHModifier", "HasNVModifier", "HasGlobalPrefetch", "HasXcnt"):
            self.assertIn(key, asm)
            self.assertIn(asm[key], (0, 1))

    def test_matches_stinkytofu_api(self):
        import stinkytofu

        direct = stinkytofu.getHardwareCaps(list(_GFX1250))
        via_caps = caps.getCaps(_GFX1250)
        # getCaps() may inject supplemental caps (e.g. HasWMMA_AccImmZero)
        # that the binding does not yet probe; verify the binding's own caps
        # are a subset of the returned dict.
        self.assertTrue(
            dict(direct["asmCaps"]).items() <= via_caps[0].items(),
            "getCaps asmCaps must be a superset of stinkytofu raw caps")
        self.assertEqual(via_caps[1], dict(direct["archCaps"]))
        self.assertTrue(
            dict(direct["regCaps"]).items() <= via_caps[2].items(),
            "getCaps regCaps must be a superset of stinkytofu raw caps")
        self.assertEqual(via_caps[3], dict(direct["asmBugs"]))

    def test_returns_fresh_copies(self):
        asm1, _, _, _ = caps.getCaps(_GFX1250)
        asm2, _, _, _ = caps.getCaps(_GFX1250)
        asm1["HasSCOPEModifier"] = 0
        self.assertNotEqual(asm1["HasSCOPEModifier"], asm2["HasSCOPEModifier"])


@unittest.skipUnless(_stinkytofu_available(), "stinkytofu binding not on PYTHONPATH")
class TestStinkyTofuArchProbes(unittest.TestCase):
    def test_has_backend_when_binding_present(self):
        self.assertTrue(rocisa.hasStinkyTofuBackend())

    def test_gfx1250_supported(self):
        self.assertTrue(rocisa.isSupportedByStinkyTofu(_GFX1250))
        self.assertTrue(rocisa.isSupportedByStinkyTofu("gfx1250"))

    def test_matches_stinkytofu_api(self):
        import stinkytofu

        self.assertEqual(
            rocisa.isSupportedByStinkyTofu(_GFX1250),
            stinkytofu.isSupportedByStinkyTofu(list(_GFX1250)),
        )
        self.assertEqual(
            rocisa.getRegisteredArchKeys(),
            stinkytofu.getRegisteredArchKeys(),
        )
        self.assertIn("gfx1250", rocisa.getRegisteredArchKeys())


if __name__ == "__main__":
    unittest.main()
