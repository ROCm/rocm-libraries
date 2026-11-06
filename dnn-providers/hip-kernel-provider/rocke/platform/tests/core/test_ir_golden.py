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
    GOLDEN_SCHEMA,
    check_golden,
    check_golden_flavors,
    compare,
    golden_document,
    golden_shard,
    load_golden,
    run_golden_cases,
    write_golden_shards,
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
        aliases = {"llvm24": "llvm23"}
        profiles = tuple(fl for fl in GOLDEN_FLAVORS if fl not in aliases)
        self._write(profiles, {"llvm24": "llvm22"})
        with patch.object(ir_golden, "GOLDEN_ALIASES", aliases), patch.object(
            ir_golden, "GOLDEN_PROFILE_FLAVORS", profiles
        ):
            errors = check_golden(self.path, lambda fl: _sub())
        self.assertEqual(len(errors), 1)
        self.assertIn("do not match emission-profile aliases", errors[0])

    def test_direct_and_aliased_lane_is_rejected(self):
        aliases = {"llvm24": "llvm23"}
        profiles = tuple(fl for fl in GOLDEN_FLAVORS if fl not in aliases)
        self._write(profiles + ("llvm24",), aliases)
        with patch.object(ir_golden, "GOLDEN_ALIASES", aliases), patch.object(
            ir_golden, "GOLDEN_PROFILE_FLAVORS", profiles
        ):
            errors = check_golden(self.path, lambda fl: _sub())
        self.assertTrue(any("both direct and aliased" in error for error in errors))

    def test_flavor_set_checked_without_lowering(self):
        self._write(GOLDEN_PROFILE_FLAVORS, GOLDEN_ALIASES)
        self.assertEqual(check_golden_flavors(self.path), [])
        self._write(GOLDEN_PROFILE_FLAVORS + ("llvm19",), GOLDEN_ALIASES)
        errors = check_golden_flavors(self.path)
        self.assertEqual(len(errors), 1)
        self.assertIn("profile documents", errors[0])

    def test_per_flavor_checks_match_one_all_flavor_check(self):
        # A caller that checks the flavors one at a time gets the same errors
        # as one all-flavor call, as long as it also checks the flavor set.
        self._write(GOLDEN_PROFILE_FLAVORS, GOLDEN_ALIASES)
        bad = GOLDEN_FLAVORS[-1]

        def run(fl):
            return _sub("b" if fl == bad else "a")

        split = check_golden_flavors(self.path)
        for fl in GOLDEN_FLAVORS:
            split.extend(f"[{fl}] {e}" for e in check_golden(self.path, run, fl))
        self.assertEqual(split, check_golden(self.path, run))
        self.assertEqual(split, [f"[{bad}] k: a -> b"])

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


class TestGoldenShards(unittest.TestCase):
    """A golden directory of per-family shards checks like the merged file."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)

    def _shard(self, name, cases, **top):
        doc = {"schema": "s", "aliases": dict(GOLDEN_ALIASES), **top}
        doc["flavors"] = {
            fl: {"cases": {c: {"sha256": "a"} for c in cases}}
            for fl in GOLDEN_PROFILE_FLAVORS
        }
        (self.dir / f"{name}.json").write_text(json.dumps(doc))

    def _run(self, fl):
        return {"cases": {c: {"sha256": "a"} for c in ("x/1", "x/2", "y/1")}}

    def test_shards_merge(self):
        self._shard("x", ["x/1", "x/2"])
        self._shard("y", ["y/1"])
        doc = load_golden(self.dir)
        self.assertEqual(doc["schema"], "s")
        self.assertEqual(doc["aliases"], GOLDEN_ALIASES)
        for sub in doc["flavors"].values():
            self.assertEqual(set(sub["cases"]), {"x/1", "x/2", "y/1"})
        self.assertEqual(check_golden(self.dir, self._run), [])

    def test_missing_shard_reported_as_new_cases(self):
        self._shard("x", ["x/1", "x/2"])
        errors = check_golden(self.dir, self._run)
        self.assertTrue(errors)
        self.assertTrue(all("new current y/1" in e for e in errors))

    def test_misfiled_case_rejected(self):
        self._shard("x", ["x/1", "x/2", "y/1"])
        errors = check_golden(self.dir, self._run)
        self.assertEqual(len(errors), 1)
        self.assertIn("belongs in y.json", errors[0])

    def test_inconsistent_aliases_rejected(self):
        self._shard("x", ["x/1", "x/2"])
        self._shard("y", ["y/1"], aliases={"llvm24": "llvm23"})
        errors = check_golden(self.dir, self._run)
        self.assertEqual(len(errors), 1)
        self.assertIn("re-bless every shard", errors[0])

    def test_unparsable_shard_named(self):
        self._shard("x", ["x/1", "x/2"])
        (self.dir / "y.json").write_text("{\n<<<<<<< HEAD\n}")
        errors = check_golden(self.dir, self._run)
        self.assertEqual(len(errors), 1)
        self.assertTrue(errors[0].startswith("golden shard y.json:"), errors[0])

    def test_empty_directory_rejected(self):
        errors = check_golden(self.dir, self._run)
        self.assertEqual(len(errors), 1)
        self.assertIn("no *.json shards", errors[0])

    def test_golden_shard_is_first_segment(self):
        self.assertEqual(golden_shard("attention/gfx942/2d_bf16"), "attention")


class TestWriteGoldenShards(unittest.TestCase):
    """The shared writer: one shard per family, partial rewrites, stale shards."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name) / "golden"

    @staticmethod
    def _doc(cases, failures=()):
        def run(fl):
            return {
                "cases": {c: {"sha256": f"{fl}:{c}"} for c in cases},
                "expected_failures": {
                    c: {"type": "ValueError", "message": "no"} for c in failures
                },
            }

        return golden_document(run)

    def _stems(self):
        return sorted(p.stem for p in self.dir.glob("*.json"))

    def test_one_shard_per_family(self):
        doc = self._doc(["x/1", "x/2", "y/1"], failures=["z/1"])
        write_golden_shards(self.dir, doc)
        self.assertEqual(self._stems(), ["x", "y", "z"])
        x = json.loads((self.dir / "x.json").read_text())
        self.assertEqual(x["schema"], GOLDEN_SCHEMA)
        self.assertEqual(x["aliases"], GOLDEN_ALIASES)
        self.assertEqual(set(x["flavors"]), set(GOLDEN_PROFILE_FLAVORS))
        for fl, sub in x["flavors"].items():
            self.assertEqual(set(sub["cases"]), {"x/1", "x/2"})
            self.assertEqual(sub["expected_failures"], {})
        z = json.loads((self.dir / "z.json").read_text())
        for sub in z["flavors"].values():
            self.assertEqual(sub["cases"], {})
            self.assertEqual(set(sub["expected_failures"]), {"z/1"})

    def test_format_is_sorted_indented_with_newline(self):
        doc = self._doc(["x/1"])
        write_golden_shards(self.dir, doc)
        text = (self.dir / "x.json").read_text()
        self.assertTrue(text.endswith("}\n"))
        self.assertEqual(
            text, json.dumps(json.loads(text), indent=2, sort_keys=True) + "\n"
        )

    def test_round_trips_through_load_golden(self):
        doc = self._doc(["x/1", "x/2", "y/1"], failures=["y/2"])
        write_golden_shards(self.dir, doc)
        loaded = load_golden(self.dir)
        self.assertEqual(loaded["schema"], GOLDEN_SCHEMA)
        self.assertEqual(loaded["aliases"], GOLDEN_ALIASES)
        for fl, sub in doc["flavors"].items():
            self.assertEqual(loaded["flavors"][fl], sub)

    def test_shards_leaves_other_shards_untouched(self):
        write_golden_shards(self.dir, self._doc(["x/1", "y/1"]))
        before = (self.dir / "y.json").read_text()
        partial = self._doc(["x/1", "x/3"])
        write_golden_shards(self.dir, partial, shards={"x"})
        self.assertEqual(self._stems(), ["x", "y"])
        self.assertEqual((self.dir / "y.json").read_text(), before)
        x = load_golden(self.dir / "x.json")
        for sub in x["flavors"].values():
            self.assertEqual(set(sub["cases"]), {"x/1", "x/3"})

    def test_removed_family_shard_deleted(self):
        write_golden_shards(self.dir, self._doc(["x/1", "y/1"]))
        write_golden_shards(self.dir, self._doc(["x/1"]))
        self.assertEqual(self._stems(), ["x"])

    def test_shards_deletes_only_named_empty_family(self):
        write_golden_shards(self.dir, self._doc(["x/1", "y/1", "z/1"]))
        write_golden_shards(self.dir, self._doc(["x/1"]), shards={"x", "y"})
        self.assertEqual(self._stems(), ["x", "z"])


class TestRunGoldenCases(unittest.TestCase):
    """The shared corpus runner records failures and filters by shard."""

    @staticmethod
    def _lower(case, flavor):
        if case["case_id"].endswith("bad"):
            raise ValueError(f"cannot lower at {flavor}")
        return {"sha256": case["case_id"]}, f"; {case['case_id']}"

    def test_failures_and_shards(self):
        cases = [{"case_id": c} for c in ("x/1", "x/bad", "y/1")]
        sub = run_golden_cases(cases, "llvm22", lower=self._lower, shards={"x"})
        self.assertEqual(sub["cases"], {"x/1": {"sha256": "x/1"}})
        self.assertEqual(
            sub["expected_failures"],
            {"x/bad": {"type": "ValueError", "message": "cannot lower at llvm22"}},
        )

    def test_ir_dir_writes_modules(self):
        with tempfile.TemporaryDirectory() as tmp:
            ir_dir = Path(tmp) / "ir"
            run_golden_cases(
                [{"case_id": "x/1"}],
                "llvm22",
                lower=self._lower,
                ir_dir=ir_dir,
                ir_name=lambda cid: cid.replace("/", "__") + ".ll",
            )
            self.assertEqual((ir_dir / "x__1.ll").read_text(), "; x/1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
