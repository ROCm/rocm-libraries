# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""No-GPU tests for the ``arch_specific`` marker helper."""

from __future__ import annotations

import subprocess
import sys
import unittest

from rocke.tests.arch_marks import ARCH_SPECIFIC_ATTR, arch_specific


class TestArchSpecific(unittest.TestCase):
    def test_function_carries_archs_and_reason(self):
        @arch_specific(("gfx950",), reason="feature exists on gfx950 only")
        def case():
            pass

        self.assertEqual(
            getattr(case, ARCH_SPECIFIC_ATTR),
            (("gfx950",), "feature exists on gfx950 only"),
        )

    def test_class_carries_the_mark(self):
        @arch_specific(["gfx942", "gfx950"], reason="CDNA only")
        class Case:
            pass

        self.assertEqual(getattr(Case, ARCH_SPECIFIC_ATTR)[0], ("gfx942", "gfx950"))

    def test_reason_is_stripped(self):
        case = arch_specific(("gfx942",), reason="  why  ")(lambda: None)
        self.assertEqual(getattr(case, ARCH_SPECIFIC_ATTR)[1], "why")

    def test_empty_reason_is_rejected(self):
        for reason in ("", "   ", None):
            with self.subTest(reason=reason), self.assertRaises(ValueError):
                arch_specific(("gfx950",), reason=reason)

    def test_empty_arch_list_is_rejected(self):
        with self.assertRaises(ValueError):
            arch_specific((), reason="why")

    def test_string_instead_of_list_is_rejected(self):
        with self.assertRaises(TypeError):
            arch_specific("gfx950", reason="why")

    def test_unknown_arch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "gfx999"):
            arch_specific(("gfx999",), reason="why")

    def test_marked_unittest_method_runs(self):
        class Case(unittest.TestCase):
            @arch_specific(("gfx950",), reason="runs, not skipped")
            def test_it(self):
                pass

        result = unittest.TestResult()
        Case("test_it").run(result)
        self.assertEqual(
            (result.testsRun, result.skipped, result.failures), (1, [], [])
        )

    def test_import_does_not_load_torch(self):
        code = "import sys, rocke.tests.arch_marks; sys.exit('torch' in sys.modules)"
        self.assertEqual(
            subprocess.run([sys.executable, "-c", code], check=False).returncode, 0
        )


@arch_specific(("gfx950",), reason="pytest-style test keeps running")
def test_marked_pytest_function_runs():
    assert True
