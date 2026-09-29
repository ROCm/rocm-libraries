# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for :mod:`rocke.core.ir_golden`, the shared all-flavor golden comparator.

The property that matters: a new entry in ``LLVM_FLAVORS`` needs no golden-test
edit, and every golden checked through ``check_golden`` fails until re-blessed.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rocke.core import ir_golden
from rocke.core.ir_golden import GOLDEN_FLAVORS, check_golden, compare
from rocke.core.lower_llvm import LLVM_FLAVORS


def _sub(sha="a"):
    return {"cases": {"k": {"sha256": sha}}}


class TestIrGolden(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "golden.json"

    def _write(self, flavors):
        self.path.write_text(json.dumps({"flavors": {fl: _sub() for fl in flavors}}))

    def test_golden_flavors_is_llvm_flavors(self):
        self.assertIs(GOLDEN_FLAVORS, LLVM_FLAVORS)

    def test_every_flavor_checked(self):
        self._write(GOLDEN_FLAVORS)
        seen = []
        self.assertEqual(
            check_golden(self.path, lambda fl: seen.append(fl) or _sub()), []
        )
        self.assertEqual(tuple(seen), GOLDEN_FLAVORS)

    def test_new_flavor_fails_until_reblessed(self):
        self._write(GOLDEN_FLAVORS)
        grown = GOLDEN_FLAVORS + ("llvm99",)
        with patch.object(ir_golden, "GOLDEN_FLAVORS", grown):
            errors = check_golden(self.path, lambda fl: _sub())
            self.assertEqual(len(errors), 1)
            self.assertIn("'llvm99'", errors[0])
            self._write(grown)
            self.assertEqual(check_golden(self.path, lambda fl: _sub()), [])

    def test_stale_flavor_reported(self):
        self._write(GOLDEN_FLAVORS + ("llvm19",))
        errors = check_golden(self.path, lambda fl: _sub())
        self.assertEqual(len(errors), 1)
        self.assertIn("stale flavor 'llvm19'", errors[0])

    def test_single_flavor_ignores_others(self):
        self._write(GOLDEN_FLAVORS[:1])
        self.assertEqual(
            check_golden(self.path, lambda fl: _sub(), GOLDEN_FLAVORS[0]), []
        )

    def test_drift_prefixed_by_flavor(self):
        self._write(GOLDEN_FLAVORS)
        bad = GOLDEN_FLAVORS[-1]
        errors = check_golden(self.path, lambda fl: _sub("b" if fl == bad else "a"))
        self.assertEqual(errors, [f"[{bad}] k: a -> b"])

    def test_compare_case_sets_and_failures(self):
        base = {
            "cases": {"k": {"sha256": "a"}, "gone": {"sha256": "x"}},
            "expected_failures": {"f": {"type": "E", "message": "m"}},
        }
        cur = {
            "cases": {"k": {"sha256": "a"}, "added": {"sha256": "y"}},
            "expected_failures": {"f": {"type": "E", "message": "m2"}},
        }
        errors = compare(base, cur)
        self.assertIn("cases: missing current gone", errors)
        self.assertIn("cases: new current added", errors)
        self.assertTrue(any(e.startswith("f: failure changed") for e in errors))
        self.assertEqual(compare(base, base), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
