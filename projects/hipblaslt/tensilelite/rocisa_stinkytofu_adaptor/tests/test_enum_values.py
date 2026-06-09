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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
################################################################################
"""Verify develop-gap enum members expose correct C++ integral values."""

import os
import sys
import unittest

_PKG_PARENT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir)
)
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

from rocisa_stinkytofu_adaptor.enum import (  # noqa: E402
    InstType,
    NonVolatile,
    TemporalHint,
)


class TestDevelopGapEnumValues(unittest.TestCase):
    def test_temporal_hint_values(self):
        self.assertEqual(TemporalHint.TH_NONE, -1)
        self.assertEqual(TemporalHint.TH_RT, 0)
        self.assertEqual(TemporalHint.TH_NT, 1)
        self.assertEqual(TemporalHint.TH_LU, 3)
        self.assertEqual(TemporalHint.TH_WB, 3)
        self.assertEqual(TemporalHint.TH_RESERVED, 7)
        self.assertEqual(TemporalHint.TH_NT_WB, 7)

    def test_non_volatile_values(self):
        self.assertEqual(NonVolatile.NV_NONE, 0)
        self.assertEqual(NonVolatile.NV, 1)

    def test_inst_type_b192_value(self):
        self.assertEqual(InstType.INST_B192, 22)
        self.assertEqual(InstType.INST_NOTYPE, 68)


if __name__ == "__main__":
    unittest.main()
