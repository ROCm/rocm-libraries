# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import unittest

import hostnumerics


class ImportTests(unittest.TestCase):
    def test_version(self):
        self.assertEqual(hostnumerics.__version__, "0.1.0")
