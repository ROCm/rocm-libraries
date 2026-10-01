#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for check_wrapper_abi.py.

It has no GPU-dependent behavior to hide from -- the whole script only resolves
a library pair and forwards them to check_public_abi.py -- so unlike
test_run_forwarding_parity.py this covers the entire thing, not just the parts
that do not need a GPU.

check_public_abi.py itself is stood in for by a script that records how it was
called; its own ABI logic has its own tests.

Written against the standard library's unittest rather than pytest: this runs as a
ctest entry in a wrapper-enabled build, and nothing provisions pytest for a machine
that builds MIOpen.

    python3 -m unittest test_check_wrapper_abi
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS = Path(__file__).resolve().parent / "check_wrapper_abi.py"

# Passes, and leaves behind what it was called with, so the harness's own wiring can
# be checked rather than assumed.
RECORDS_ARGV = """#!/usr/bin/env python3
import pathlib, sys
pathlib.Path(__file__).with_suffix(".argv").write_text("\\n".join(sys.argv[1:]))
"""

FAILS = """#!/usr/bin/env python3
import sys
sys.exit(1)
"""


# Checked before anything here touches os.geteuid(), which off POSIX would fail at
# import rather than skip.
@unittest.skipUnless(os.name == "posix", "the harness under test is POSIX-only")
class WrapperAbiCheckTest(unittest.TestCase):
    def setUp(self):
        """A stand-in install tree: the libraries the harness resolves, plus its helper."""
        holder = tempfile.TemporaryDirectory()
        self.addCleanup(holder.cleanup)
        self.tree = Path(holder.name)

        lib = self.tree / "lib"
        lib.mkdir()
        (lib / "libMIOpen.so.1.0").touch()
        (lib / "libMIOpen_private.so.1.0").touch()

        fake_abi = self.tree / "fake_abi_check.py"
        fake_abi.write_text(RECORDS_ARGV)
        fake_abi.chmod(0o755)

    def run_harness(self, cwd, *extra):
        return subprocess.run(
            [
                sys.executable,
                str(HARNESS),
                "--lib-dir",
                str(self.tree / "lib"),
                "--abi-check",
                str(self.tree / "fake_abi_check.py"),
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

    def test_the_libraries_under_test_are_named(self):
        """A co-versioning failure downstream has to be tied back to concrete files."""
        result = self.run_harness(self.tree)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(str(self.tree / "lib" / "libMIOpen.so.1.0"), result.stdout)
        self.assertIn(
            str(self.tree / "lib" / "libMIOpen_private.so.1.0"), result.stdout
        )

    def test_wrapper_and_private_lib_are_forwarded_to_the_abi_check(self):
        result = self.run_harness(self.tree)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = (self.tree / "fake_abi_check.argv").read_text().splitlines()
        self.assertIn("check-wrapper", argv)
        self.assertIn(str(self.tree / "lib" / "libMIOpen.so.1.0"), argv)
        self.assertEqual(
            argv[argv.index("--private-lib") + 1],
            str(self.tree / "lib" / "libMIOpen_private.so.1.0"),
        )

    def test_baseline_and_excluded_are_forwarded_to_the_abi_check(self):
        baseline = self.tree / "baseline.txt"
        excluded = self.tree / "excluded.txt"
        baseline.touch()
        excluded.touch()
        result = subprocess.run(
            [
                sys.executable,
                str(HARNESS),
                "--lib-dir",
                str(self.tree / "lib"),
                "--abi-check",
                str(self.tree / "fake_abi_check.py"),
                "--baseline",
                str(baseline),
                "--excluded",
                str(excluded),
            ],
            cwd=str(self.tree),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = (self.tree / "fake_abi_check.argv").read_text().splitlines()
        self.assertEqual(argv[argv.index("--baseline") + 1], str(baseline))
        self.assertEqual(argv[argv.index("--excluded") + 1], str(excluded))

    def test_public_header_is_forwarded_to_the_abi_check_when_given(self):
        """Only a build-tree caller has a source tree to point this at."""
        result = self.run_harness(
            self.tree, "--public-header", str(self.tree / "miopen.h")
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = (self.tree / "fake_abi_check.argv").read_text().splitlines()
        self.assertIn("--public-header", argv)
        self.assertEqual(
            argv[argv.index("--public-header") + 1], str(self.tree / "miopen.h")
        )

    def test_public_header_is_omitted_from_the_abi_check_by_default(self):
        """The installed ctest mirror has no include directory to name."""
        result = self.run_harness(self.tree)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = (self.tree / "fake_abi_check.argv").read_text().splitlines()
        self.assertNotIn("--public-header", argv)

    def test_the_abi_check_s_exit_code_is_returned(self):
        """A caught ABI regression has to fail the ctest entry, not just log it."""
        failing = self.tree / "failing_abi_check.py"
        failing.write_text(FAILS)
        failing.chmod(0o755)
        result = self.run_harness(self.tree, "--abi-check", str(failing))
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)

    def test_helper_runs_without_its_exec_bit(self):
        """It is launched through this interpreter, not its shebang line."""
        (self.tree / "fake_abi_check.py").chmod(0o644)
        result = self.run_harness(self.tree)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_a_library_problem_fails_the_run(self):
        """Which files count is tested in test_miopen_wrapper_libs; this checks the
        script stops on what it reports, using the default lib* search.

        Copied into bin/ so that search -- lib* beside its parent directory -- lands in
        this tree. miopen_wrapper_libs.py comes with it since the script imports from
        it and only its own directory is on sys.path.
        """
        (self.tree / "lib" / "libMIOpen_private.so.1.0").unlink()
        bindir = self.tree / "bin"
        bindir.mkdir()
        harness = bindir / HARNESS.name
        shutil.copy(HARNESS, harness)
        shutil.copy(HARNESS.parent / "miopen_wrapper_libs.py", bindir)

        result = subprocess.run(
            [
                sys.executable,
                str(harness),
                "--abi-check",
                str(self.tree / "fake_abi_check.py"),
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
        self.assertIn(
            f"no libMIOpen_private.so.* found under {self.tree / 'lib'}", result.stdout
        )
        self.assertFalse((self.tree / "fake_abi_check.argv").exists())


if __name__ == "__main__":
    unittest.main()
