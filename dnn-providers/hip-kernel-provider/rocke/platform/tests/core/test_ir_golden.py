# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for :mod:`rocke.core.ir_golden`, the shared all-flavor golden comparator.

A new compiler lane needs no golden-test edit: each golden must add either a
new profile baseline or an explicit alias to a structurally identical profile.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rocke.core import ir_golden
from rocke.core.ir_golden import (
    GOLDEN_ALIASES,
    GOLDEN_FLAVORS,
    GOLDEN_PROFILE_FLAVORS,
    check_golden,
    compare,
)
from rocke.core.lower_llvm import LLVM_FLAVORS


def _sub(sha="a"):
    return {"cases": {"k": {"sha256": sha}}}


class TestIrGolden(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "golden.json"

    def _write(self, flavors, aliases=None):
        self.path.write_text(
            json.dumps(
                {
                    "aliases": aliases or {},
                    "flavors": {fl: _sub() for fl in flavors},
                }
            )
        )

    def test_golden_flavors_is_llvm_flavors(self):
        self.assertIs(GOLDEN_FLAVORS, LLVM_FLAVORS)

    def test_every_flavor_checked(self):
        self._write(GOLDEN_PROFILE_FLAVORS, GOLDEN_ALIASES)
        seen = []

        def run(fl):
            seen.append(fl)
            return _sub()

        self.assertEqual(check_golden(self.path, run), [])
        self.assertEqual(tuple(seen), GOLDEN_FLAVORS)

    def test_new_flavor_fails_until_profile_or_alias_is_declared(self):
        self._write(GOLDEN_PROFILE_FLAVORS, GOLDEN_ALIASES)
        grown = GOLDEN_FLAVORS + ("llvm99",)
        with patch.object(ir_golden, "GOLDEN_FLAVORS", grown):
            errors = check_golden(self.path, lambda fl: _sub())
            self.assertEqual(len(errors), 1)
            self.assertIn("'llvm99'", errors[0])

    def test_shared_profile_alias_checks_the_requested_lane(self):
        self._write(GOLDEN_PROFILE_FLAVORS, GOLDEN_ALIASES)
        seen = []
        self.assertEqual(
            check_golden(self.path, lambda fl: seen.append(fl) or _sub()), []
        )
        self.assertEqual(tuple(seen), GOLDEN_FLAVORS)

    def test_stale_profile_document_reported(self):
        self._write(GOLDEN_PROFILE_FLAVORS + ("llvm19",), GOLDEN_ALIASES)
        errors = check_golden(self.path, lambda fl: _sub())
        self.assertEqual(len(errors), 1)
        self.assertIn("profile documents", errors[0])

    def test_single_flavor_ignores_others(self):
        self._write(GOLDEN_FLAVORS[:1])
        self.assertEqual(
            check_golden(self.path, lambda fl: _sub(), GOLDEN_FLAVORS[0]), []
        )

    def test_drift_prefixed_by_flavor(self):
        self._write(GOLDEN_PROFILE_FLAVORS, GOLDEN_ALIASES)
        bad = GOLDEN_FLAVORS[-1]
        errors = check_golden(self.path, lambda fl: _sub("b" if fl == bad else "a"))
        self.assertEqual(errors, [f"[{bad}] k: a -> b"])

    def test_wrong_profile_alias_is_rejected(self):
        self._write(GOLDEN_PROFILE_FLAVORS, {"llvm24": "llvm22"})
        errors = check_golden(self.path, lambda fl: _sub())
        self.assertEqual(len(errors), 1)
        self.assertIn("do not match emission-profile aliases", errors[0])

    def test_direct_and_aliased_lane_is_rejected(self):
        self._write(GOLDEN_PROFILE_FLAVORS + ("llvm24",), GOLDEN_ALIASES)
        errors = check_golden(self.path, lambda fl: _sub())
        self.assertTrue(any("both direct and aliased" in error for error in errors))

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
