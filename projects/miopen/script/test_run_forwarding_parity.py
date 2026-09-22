#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the parts of run_forwarding_parity.py that do not need a GPU.

Where it puts its replay reports: ctest runs the installed entry from inside the
install tree, which on a shipping prefix is root-owned and read-only to whoever runs
the tests, so writing the reports beside the working directory fails the whole
harness on a permission error that has nothing to do with parity.

How it drives the two helpers: which library pair it resolves and names, and that it
holds the comparison to this run's binary. That last one is the harness's side of the
guard against two leftover reports comparing cleanly, and it is invisible from the
comparator's own tests.

The replays are stood in for by a script that writes a well-formed report, the
comparison by one that records how it was called, and the ABI check by one that
passes.

Written against the standard library's unittest rather than pytest: this runs as a
ctest entry in a wrapper-enabled build, and nothing provisions pytest for a machine
that builds MIOpen.

    python3 -m unittest test_run_forwarding_parity
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS = Path(__file__).resolve().parent / "run_forwarding_parity.py"

FAKE_GTEST = """#!/usr/bin/env python3
import sys
out = [a.split("xml:", 1)[1] for a in sys.argv if a.startswith("--gtest_output=")][0]
open(out, "w").write(
    '<?xml version="1.0"?><testsuites tests="1" failures="0" disabled="0" errors="0">'
    '<testsuite name="S" tests="1"><testcase name="T" classname="S"/></testsuite>'
    "</testsuites>"
)
"""

PASSES = "#!/usr/bin/env python3\n"

# Passes, and leaves behind what it was called with, so the harness's own wiring can
# be checked rather than assumed.
RECORDS_ARGV = """#!/usr/bin/env python3
import pathlib, sys
pathlib.Path(__file__).with_suffix(".argv").write_text("\\n".join(sys.argv[1:]))
"""


# Checked before anything here touches os.geteuid(), which off POSIX would fail at
# import rather than skip.
@unittest.skipUnless(os.name == "posix", "the harness under test is POSIX-only")
class ParityRunnerTest(unittest.TestCase):
    def setUp(self):
        """A stand-in install tree: the libraries the harness resolves, plus its helpers."""
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.tree = Path(holder.name)

        lib = self.tree / "lib"
        lib.mkdir()
        (lib / "libMIOpen.so.1.0").touch()
        (lib / "libMIOpen_private.so.1.0").touch()
        for name, body in (
            ("fake_gtest.py", FAKE_GTEST),
            ("fake_compare.py", RECORDS_ARGV),
            ("fake_abi.py", PASSES),
        ):
            path = self.tree / name
            path.write_text(body)
            path.chmod(0o755)

    def run_harness(self, cwd, *extra):
        return subprocess.run(
            [
                sys.executable,
                str(HARNESS),
                "--gtest",
                str(self.tree / "fake_gtest.py"),
                "--filter",
                "*",
                "--lib-dir",
                str(self.tree / "lib"),
                "--compare",
                str(self.tree / "fake_compare.py"),
                "--abi-check",
                str(self.tree / "fake_abi.py"),
                "--baseline",
                os.devnull,
                "--excluded",
                os.devnull,
                *extra,
            ],
            cwd=str(cwd),
            capture_output=True,
            text=True,
        )

    def test_nothing_is_written_into_the_working_directory(self):
        workdir = self.tree / "bin" / "MIOpen"
        workdir.mkdir(parents=True)
        result = self.run_harness(workdir)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(list(workdir.iterdir()), [])

    @unittest.skipIf(
        os.name == "posix" and os.geteuid() == 0,
        "root writes through the mode bits, so the read-only case cannot be created",
    )
    def test_a_read_only_working_directory_still_passes(self):
        """The shipping case: an artifact the runner may not write to."""
        workdir = self.tree / "bin" / "MIOpen"
        workdir.mkdir(parents=True)
        workdir.chmod(0o555)
        try:
            result = self.run_harness(workdir)
        finally:
            workdir.chmod(0o755)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_reports_land_where_output_dir_names(self):
        """The build tree passes one explicitly and keeps its reports."""
        out = self.tree / "test_results"
        result = self.run_harness(self.tree, "--output-dir", str(out))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            sorted(p.name for p in out.iterdir()),
            [
                "fake_gtest.py_forwarding_disabled.xml",
                "fake_gtest.py_forwarding_enabled.xml",
            ],
        )

    def test_the_report_directory_is_reported(self):
        """A failing replay is only diagnosable if its reports can be found."""
        result = self.run_harness(self.tree)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("replay reports:", result.stdout)

    def test_the_comparison_is_held_to_this_run_s_binary(self):
        """--newer-than is what stops a leftover pair of reports comparing cleanly."""
        result = self.run_harness(self.tree)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = (self.tree / "fake_compare.argv").read_text().splitlines()
        self.assertIn("--newer-than", argv)
        self.assertEqual(
            argv[argv.index("--newer-than") + 1], str(self.tree / "fake_gtest.py")
        )

    def test_helpers_run_without_their_exec_bit(self):
        """They are launched through this interpreter, not their shebang lines."""
        (self.tree / "fake_compare.py").chmod(0o644)
        (self.tree / "fake_abi.py").chmod(0o644)
        result = self.run_harness(self.tree)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_two_versioned_libraries_are_refused_rather_than_picked_between(self):
        """One of them is an earlier build, and filename order is not version order."""
        (self.tree / "lib" / "libMIOpen.so.10.0").touch()
        result = self.run_harness(self.tree)
        self.assertEqual(result.returncode, 1)
        self.assertIn("more than one libMIOpen.so.*", result.stdout)

    def test_a_pair_split_across_two_directories_is_refused(self):
        """lib and lib64 both present, with one half of the pair in each.

        Only one directory can go first on LD_LIBRARY_PATH, so the replays would load
        mismatched halves -- what the co-versioning check exists to catch.
        """
        lib64 = self.tree / "lib64"
        lib64.mkdir()
        (self.tree / "lib" / "libMIOpen_private.so.1.0").rename(
            lib64 / "libMIOpen_private.so.1.0"
        )

        # Copied in so the harness's default search -- lib* beside its parent directory --
        # lands here. That search is the only way to reach a split pair; --lib-dir names
        # one directory and cannot express one.
        bindir = self.tree / "bin"
        bindir.mkdir(exist_ok=True)
        harness = bindir / HARNESS.name
        shutil.copy(HARNESS, harness)

        result = subprocess.run(
            [
                sys.executable,
                str(harness),
                "--gtest",
                str(self.tree / "fake_gtest.py"),
                "--filter",
                "*",
                "--compare",
                str(self.tree / "fake_compare.py"),
                "--abi-check",
                str(self.tree / "fake_abi.py"),
                "--baseline",
                os.devnull,
                "--excluded",
                os.devnull,
            ],
            cwd=str(self.tree),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("different directories", result.stdout)

    def test_the_libraries_under_test_are_named(self):
        """A co-versioning failure downstream has to be tied back to concrete files."""
        result = self.run_harness(self.tree)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(str(self.tree / "lib" / "libMIOpen.so.1.0"), result.stdout)
        self.assertIn(
            str(self.tree / "lib" / "libMIOpen_private.so.1.0"), result.stdout
        )


if __name__ == "__main__":
    unittest.main()
