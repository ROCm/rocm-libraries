# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Golden routing of the differential harness (run_diff.py).

Each parity family is anchored in the golden file beside its emitters: a
platform family in tests/instances/differential/golden, a library family in
library/tests/parity/golden. These tests pin that routing and the checks that
refuse a family blessed in the wrong file (GOLDEN MISROUTED) or in both
(GOLDEN CONFLICT).

They live here, not beside run_diff.py in differential/, because CMake leaves
that directory out of the installed test tree, so its tests run in no installed
CTest entry. run_diff.py and the golden files it reads are installed beside
this file (tests/instances/differential, tests/library/tests/parity), so these
tests run in the installed rocke_pytest_* part for tests/instances as well as
in run_all.py's pytest pass over the source tree.

Host-only. The gate tests run main()'s golden step over a temp tree with the
reference emitters stubbed; the real-tree tests read the committed goldens and
skip in a platform-only tree. Importing run_diff creates its /tmp working dir.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

# The differential harness is runner tooling, not a package; import it by path,
# the way its other callers do. The same relative location holds in the source
# and the installed tree.
sys.path.insert(0, str(Path(__file__).resolve().parent / "differential"))

import run_diff  # noqa: E402 -- after sys.path shim
from rocke.tests import ci_shard


class TestGoldenRouting(unittest.TestCase):
    """Each family is anchored in the golden file beside its emitters.

    A family checked against the wrong file finds none of its configs blessed
    there, so they read as NEW (informational) and the check would print
    GOLDEN OK. These tests pin the routing and the checks that refuse it.
    """

    def test_the_tree_routes_library_families_to_the_library_golden(self):
        fams = dict(run_diff.find_families(python_only=True))
        if not run_diff.LIB_PARITY.is_dir():
            self.skipTest("platform-only tree: no library parity families")
        # One real family from each home, so a rename shows up here.
        self.assertEqual(fams["batched_gemm"], run_diff.PARITY)
        self.assertEqual(fams["conv_abi"], run_diff.LIB_PARITY)
        for name, src_dir in fams.items():
            with self.subTest(family=name):
                expected = (
                    run_diff.LIB_GOLDEN_FILE
                    if src_dir == run_diff.LIB_PARITY
                    else run_diff.GOLDEN_FILE
                )
                self.assertEqual(run_diff.golden_file_for(src_dir), expected)
        self.assertNotEqual(run_diff.LIB_GOLDEN_FILE, run_diff.GOLDEN_FILE)

    def test_the_committed_goldens_bless_each_family_in_its_home_only(self):
        if not run_diff.LIB_GOLDEN_FILE.exists():
            self.skipTest("platform-only tree: no library golden")
        fams = run_diff.find_families(python_only=True)
        stores = {
            path: run_diff.GoldenStore.load(path)
            for path in (run_diff.GOLDEN_FILE, run_diff.LIB_GOLDEN_FILE)
        }
        for name, src_dir in fams:
            home = run_diff.golden_file_for(src_dir)
            for path, store in stores.items():
                if path == home:
                    continue
                blessed = set(store.section(run_diff.IR_CANONICAL, None))
                for flavor in store.ll:
                    blessed |= set(store.section(run_diff.LL, flavor))
                with self.subTest(family=name, file=str(path)):
                    self.assertNotIn(name, blessed)


class GoldenGateTree(unittest.TestCase):
    """main()'s golden step over a temp tree: a platform family ``plat`` and a
    library family ``libfam``, each with its own golden file. Holds no tests."""

    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        self.parity = root / "platform" / "parity"
        self.lib_parity = root / "library" / "parity"
        for d, name in ((self.parity, "plat"), (self.lib_parity, "libfam")):
            d.mkdir(parents=True)
            (d / f"{name}_emit.py").write_text("")
        self.golden = root / "platform" / "golden" / "llvm_gfx_all.json"
        self.lib_golden = self.lib_parity / "golden" / "llvm_gfx_all.json"
        for name, value in (
            ("PARITY", self.parity),
            ("LIB_PARITY", self.lib_parity),
            ("GOLDEN_FILE", self.golden),
            ("LIB_GOLDEN_FILE", self.lib_golden),
        ):
            patcher = mock.patch.object(run_diff, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        env = mock.patch.dict(os.environ, {"ROCKE_LLVM_FLAVOR": "llvm23"})
        env.start()
        self.addCleanup(env.stop)
        self.current = {"plat": {"0": "p0"}, "libfam": {"0": "l0"}}

    def bless(self, path, shas):
        store = run_diff.GoldenStore()
        store.record(run_diff.LL, "llvm23", shas)
        store.save(path)

    def main(self, step, *extra):
        argv = ["run_diff.py", "--python-only", step, "--flavor", "llvm23", *extra]
        out = io.StringIO()
        reference = mock.patch.object(
            run_diff, "python_reference_families", side_effect=self.reference_families
        )
        argv_patch = mock.patch.object(sys, "argv", argv)
        with reference, argv_patch, contextlib.redirect_stdout(out):
            code = run_diff.main()
        return code, out.getvalue()

    def reference_families(self, fams, flavors, *_):
        """The stand-in for the Python emitters: ``self.current``'s hashes."""
        shas = {name: self.current[name] for name, _ in fams}
        return {flavor: (shas, []) for flavor in flavors}


class TestGoldenRoutingGates(GoldenGateTree):
    """The golden step routes each family to the file beside its emitters."""

    def test_a_fixture_family_routes_by_its_emitter_directory(self):
        self.assertEqual(
            run_diff.find_families(python_only=True),
            [("plat", self.parity), ("libfam", self.lib_parity)],
        )
        self.assertEqual(run_diff.golden_file_for(self.parity), self.golden)
        self.assertEqual(run_diff.golden_file_for(self.lib_parity), self.lib_golden)

    def test_each_family_checked_in_its_home_is_golden_ok(self):
        self.bless(self.golden, {"plat": {"0": "p0"}})
        self.bless(self.lib_golden, {"libfam": {"0": "l0"}})
        code, out = self.main("--check-golden")
        self.assertEqual(code, 0, out)
        self.assertEqual(out.count("GOLDEN OK"), 2)
        self.assertIn(str(self.lib_golden), out)

    def test_a_library_change_fails_against_the_library_golden(self):
        self.bless(self.golden, {"plat": {"0": "p0"}})
        self.bless(self.lib_golden, {"libfam": {"0": "old"}})
        code, out = self.main("--check-golden")
        self.assertEqual(code, 1)
        self.assertIn("GOLDEN MISMATCH", out)

    def test_a_mismatch_prints_the_rerecord_command_for_its_families(self):
        self.bless(self.golden, {"plat": {"0": "p0"}})
        self.bless(self.lib_golden, {"libfam": {"0": "old"}})
        code, out = self.main("--check-golden")
        self.assertEqual(code, 1)
        self.assertIn(
            "\n    python tools/golden.py rerecord differential --only libfam\n", out
        )

    def test_a_hash_edited_without_a_rerecord_is_a_stale_fingerprint(self):
        self.bless(self.golden, {"plat": {"0": "p0"}})
        self.bless(self.lib_golden, {"libfam": {"0": "l0"}})
        # A hand edit, or a merge that took one side's hashes: the hashes match
        # the emitters, but not the fingerprint the record wrote.
        document = json.loads(self.golden.read_text())
        document["ll"]["llvm23"]["plat"]["1"] = "merged"
        self.golden.write_text(json.dumps(document))
        code, out = self.main("--check-golden")
        self.assertEqual(code, 1)
        self.assertIn(f"GOLDEN FINGERPRINT STALE ({self.golden}): plat\n", out)
        self.assertIn("python tools/golden.py rerecord differential --only plat", out)

    def test_a_family_blessed_in_both_files_is_a_conflict(self):
        # Reported even when the run leaves that family out: a second library
        # family keeps both files in the check.
        (self.lib_parity / "libfam2_emit.py").write_text("")
        self.current["libfam2"] = {"0": "m0"}
        self.bless(self.golden, {"plat": {"0": "p0"}, "libfam": {"0": "l0"}})
        self.bless(self.lib_golden, {"libfam": {"0": "l0"}, "libfam2": {"0": "m0"}})
        for extra in ((), ("--exclude", "libfam")):
            with self.subTest(extra=extra):
                code, out = self.main("--check-golden", *extra)
                self.assertEqual(code, 1)
                self.assertIn(
                    "GOLDEN CONFLICT: blessed in two golden files: libfam\n", out
                )
                self.assertNotIn("GOLDEN OK", out)

    def test_a_library_family_blessed_only_in_the_platform_file_fails(self):
        self.bless(self.golden, {"plat": {"0": "p0"}, "libfam": {"0": "l0"}})
        for extra in ((), ("--only", "libfam")):
            with self.subTest(extra=extra):
                code, out = self.main("--check-golden", *extra)
                self.assertEqual(code, 1)
                self.assertIn("GOLDEN MISROUTED", out)
                self.assertIn("libfam", out)
                self.assertNotIn("GOLDEN OK", out)

    def test_routing_every_family_to_the_platform_golden_fails(self):
        self.bless(self.golden, {"plat": {"0": "p0"}})
        self.bless(self.lib_golden, {"libfam": {"0": "l0"}})
        with mock.patch.object(run_diff, "golden_file_for", return_value=self.golden):
            code, out = self.main("--check-golden")
        self.assertEqual(code, 1)
        self.assertIn("GOLDEN MISROUTED", out)
        self.assertNotIn("GOLDEN OK", out)

    def test_an_explicit_golden_file_takes_every_family(self):
        explicit = self.golden.parent.parent / "explicit.json"
        self.bless(explicit, self.current)
        self.bless(self.golden, {"libfam": {"0": "l0"}})  # not consulted
        code, out = self.main("--check-golden", "--golden-file", str(explicit))
        self.assertEqual(code, 0, out)

    def test_a_record_writes_each_family_into_its_home(self):
        code, out = self.main("--record-golden")
        self.assertEqual(code, 0, out)
        for path, expected in (
            (self.golden, {"plat": {"0": "p0"}}),
            (self.lib_golden, {"libfam": {"0": "l0"}}),
        ):
            section = run_diff.GoldenStore.load(path).section(run_diff.LL, "llvm23")
            self.assertEqual(section, expected)


class TestShardedGoldenCheck(GoldenGateTree):
    """A golden check with the gtest shard flag checks only its CI shard's
    families: the shards of one run check every family exactly once."""

    SHARDS = 4

    def setUp(self):
        super().setUp()
        # Enough families that every one of the four shards owns some.
        for index in range(12):
            name = f"plat{index}"
            (self.parity / f"{name}_emit.py").write_text("")
            self.current[name] = {"0": f"{name}-0"}
        self.ran: list[set[str]] = []

    def main(self, step, *extra, shard=None):
        """main() under a clean environment holding only the gtest pair for
        ``shard`` (none when ``shard`` is None); records the families run."""
        cleared = {name: "" for name in ci_shard.CHILD_ENV_EXCLUDED}
        if shard is not None:
            cleared[ci_shard.GTEST_INDEX_ENV] = str(shard)
            cleared[ci_shard.GTEST_TOTAL_ENV] = str(self.SHARDS)
        with mock.patch.dict(os.environ, cleared):
            return super().main(step, *extra)

    def reference_families(self, fams, *rest):
        self.ran.append({name for name, _ in fams})
        return super().reference_families(fams, *rest)

    def bless_all(self):
        self.bless(
            self.golden, {n: v for n, v in self.current.items() if n != "libfam"}
        )
        self.bless(self.lib_golden, {"libfam": self.current["libfam"]})

    def owner(self, family):
        [owner] = [
            index
            for index in range(self.SHARDS)
            if ci_shard.Shard(f"{index}/{self.SHARDS}").owns(family)
        ]
        return owner

    def test_the_shards_of_a_run_check_each_family_once(self):
        self.bless_all()
        for index in range(self.SHARDS):
            code, out = self.main(
                "--check-golden", ci_shard.ACTIVATION_FLAG, shard=index
            )
            self.assertEqual(code, 0, out)
            self.assertIn(f"rocke-ctest-shard: {index}/{self.SHARDS} (GTEST", out)
        self.assertTrue(all(self.ran), self.ran)
        self.assertEqual(set().union(*self.ran), set(self.current))
        self.assertEqual(sum(map(len, self.ran)), len(self.current))

    def test_a_drift_fails_only_the_shard_that_owns_its_family(self):
        self.bless_all()
        self.current["plat3"] = {"0": "drifted"}
        failed = [
            index
            for index in range(self.SHARDS)
            if self.main("--check-golden", ci_shard.ACTIVATION_FLAG, shard=index)[0]
        ]
        self.assertEqual(failed, [self.owner("plat3")])

    def test_a_blessed_family_without_an_emitter_is_missing_in_one_shard(self):
        self.bless_all()
        store = run_diff.GoldenStore.load(self.golden)
        store.record(run_diff.LL, "llvm23", {"gone": {"0": "g0"}}, families=["gone"])
        store.save(self.golden)
        outcomes = [
            self.main("--check-golden", ci_shard.ACTIVATION_FLAG, shard=index)
            for index in range(self.SHARDS)
        ]
        missing = [index for index, (_, out) in enumerate(outcomes) if "gone[0]" in out]
        self.assertEqual(missing, [self.owner("gone")])
        self.assertEqual(outcomes[self.owner("gone")][0], 1)

    def test_without_the_shard_variables_the_flag_checks_every_family(self):
        self.bless_all()
        code, out = self.main("--check-golden", ci_shard.ACTIVATION_FLAG)
        self.assertEqual(code, 0, out)
        self.assertIn("rocke-ctest-shard: off", out)
        self.assertEqual(self.ran, [set(self.current)])

    def test_the_shard_variables_alone_do_not_shard_a_check(self):
        self.bless_all()
        code, out = self.main("--check-golden", shard=1)
        self.assertEqual(code, 0, out)
        self.assertEqual(self.ran, [set(self.current)])

    def test_the_key_listing_names_each_blessed_config_by_its_family(self):
        self.current["plat0"] = {"0": "plat0-0", "1": "plat0-1"}
        self.bless_all()
        listing_path = self.golden.parent.parent / "keys.json"
        code, out = self.main(
            "--check-golden",
            ci_shard.ACTIVATION_FLAG,
            ci_shard.LIST_KEYS_FLAG,
            str(listing_path),
            shard=1,
        )
        self.assertEqual(code, 0, out)
        self.assertEqual(self.ran, [])
        listing = ci_shard.ShardKeyListing.read(listing_path)
        expected = {
            f"{name}/llvm23/{config}": name
            for name, configs in self.current.items()
            for config in configs
        }
        self.assertEqual(dict(listing.key_of_case), expected)
        self.assertEqual(
            sorted(case for case, key in expected.items() if key == "plat0"),
            ["plat0/llvm23/0", "plat0/llvm23/1"],
        )

    def test_a_listing_with_every_family_excluded_lists_no_case(self):
        self.bless_all()
        listing_path = self.golden.parent.parent / "keys.json"
        excluded = ",".join(self.current)
        code, out = self.main(
            "--check-golden",
            "--exclude",
            excluded,
            ci_shard.ACTIVATION_FLAG,
            ci_shard.LIST_KEYS_FLAG,
            str(listing_path),
        )
        self.assertEqual(code, 0, out)
        self.assertEqual(ci_shard.ShardKeyListing.read(listing_path).key_of_case, {})

    def test_the_key_listing_is_refused_without_the_shard_flag(self):
        refused = self.assertRaises(SystemExit)
        with refused, contextlib.redirect_stderr(io.StringIO()):
            self.main("--check-golden", ci_shard.LIST_KEYS_FLAG, "keys.json")
        self.assertEqual(refused.exception.code, 2)

    def test_the_flag_is_refused_outside_a_python_only_check(self):
        refused = self.assertRaises(SystemExit)
        with refused, contextlib.redirect_stderr(io.StringIO()):
            self.main("--record-golden", ci_shard.ACTIVATION_FLAG, shard=0)
        self.assertEqual(refused.exception.code, 2)


class TestUnblessedConfigs(GoldenGateTree):
    """Output with no blessed hash is checked by nothing, so it fails."""

    def test_an_unblessed_config_fails_with_its_rerecord_command(self):
        self.bless(self.golden, {"plat": {"0": "p0"}})
        self.bless(self.lib_golden, {"libfam": {"0": "l0"}})
        self.current["libfam"] = {"0": "l0", "1": "l1"}
        code, out = self.main("--check-golden")
        self.assertEqual(code, 1)
        self.assertIn("GOLDEN UNBLESSED", out)
        self.assertIn("    libfam[1]\n", out)
        self.assertIn(
            "\n    python tools/golden.py rerecord differential --only libfam\n", out
        )
