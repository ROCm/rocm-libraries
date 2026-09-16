# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import unittest

import roc_host_numerics


class ImportTests(unittest.TestCase):
    def test_version(self):
        self.assertEqual(roc_host_numerics.__version__, "0.1.0")
