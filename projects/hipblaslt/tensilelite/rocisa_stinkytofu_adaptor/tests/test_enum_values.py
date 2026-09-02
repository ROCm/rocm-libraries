# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
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
