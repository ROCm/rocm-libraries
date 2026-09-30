# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the gfx1151 WMMA GEMM verification compatibility wrapper."""

from __future__ import annotations

import unittest
from unittest import mock

from rocke.examples.gfx1151 import wmma_gemm_verify


class TestWmmaGemmVerifyWrapper(unittest.TestCase):
    def test_main_delegates_to_f16_verifier(self):
        verifier = mock.Mock()
        verifier.main.return_value = 17

        with mock.patch.object(
            wmma_gemm_verify.importlib,
            "import_module",
            return_value=verifier,
        ) as import_module:
            self.assertEqual(wmma_gemm_verify.main(), 17)

        import_module.assert_called_once_with(
            "rocke.examples.gfx1151.gemm.scripts.01_f16_verify"
        )
        verifier.main.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
