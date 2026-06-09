################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################
"""Tests for ``caps.getCaps`` delegating to ``stinkytofu.getHardwareCaps``."""

import os
import sys
import unittest

_PKG_PARENT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

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
        self.assertEqual(via_caps[0], dict(direct["asmCaps"]))
        self.assertEqual(via_caps[1], dict(direct["archCaps"]))
        self.assertEqual(via_caps[2], dict(direct["regCaps"]))
        self.assertEqual(via_caps[3], dict(direct["asmBugs"]))

    def test_returns_fresh_copies(self):
        asm1, _, _, _ = caps.getCaps(_GFX1250)
        asm2, _, _, _ = caps.getCaps(_GFX1250)
        asm1["HasSCOPEModifier"] = 0
        self.assertNotEqual(asm1["HasSCOPEModifier"], asm2["HasSCOPEModifier"])


if __name__ == "__main__":
    unittest.main()
