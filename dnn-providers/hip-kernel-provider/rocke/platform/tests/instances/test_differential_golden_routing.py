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


class TestGoldenRoutingGates(unittest.TestCase):
    """main()'s golden step over a temp tree: a platform family ``plat`` and a
    library family ``libfam``, each with its own golden file."""

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
            run_diff,
            "python_reference_families",
            side_effect=lambda fams, *a: ({n: self.current[n] for n, _ in fams}, []),
        )
        argv_patch = mock.patch.object(sys, "argv", argv)
        with reference, argv_patch, contextlib.redirect_stdout(out):
            code = run_diff.main()
        return code, out.getvalue()

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
