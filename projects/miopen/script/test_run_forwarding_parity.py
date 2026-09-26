#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for the parts of run_forwarding_parity.py that do not need a GPU.

Where it puts its replay reports: ctest runs the installed entry from inside the
install tree, which on a shipping prefix is root-owned and read-only to whoever runs
the tests, so writing the reports beside the working directory fails the whole
harness on a permission error that has nothing to do with parity.

How it drives the comparison: which library pair it resolves and names, and that it
holds the comparison to this run's binary. That last one is the harness's side of the
guard against two leftover reports comparing cleanly, and it is invisible from the
comparator's own tests.

The replays are stood in for by a script that writes a well-formed report and prints
the library's banner when forwarding is enabled, the comparison by one that records
how it was called.

Written against the standard library's unittest rather than pytest: this runs as a
ctest entry in a wrapper-enabled build, and nothing provisions pytest for a machine
that builds MIOpen.

    python3 -m unittest test_run_forwarding_parity
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS = Path(__file__).resolve().parent / "run_forwarding_parity.py"

FAKE_GTEST_TEMPLATE = """#!/usr/bin/env python3
import json, os, pathlib, sys
mode = os.environ.get("MIOPEN_HIPDNN_FORWARDING")
pathlib.Path(__file__).with_name("env_%s.json" % mode).write_text(json.dumps({{
    "MIOPEN_HIPDNN_FORWARDING": mode,
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH"),
    "MIOPEN_USER_DB_PATH": os.environ.get("MIOPEN_USER_DB_PATH"),
    "MIOPEN_CUSTOM_CACHE_DIR": os.environ.get("MIOPEN_CUSTOM_CACHE_DIR"),
}}))
if {announce}:
    sys.stderr.write(
        "[MIOpen] MIOPEN_HIPDNN_FORWARDING=%s: entry points in the forwarding set "
        "are redirected to hipDNN\\n" % mode
    )
out = [a.split("xml:", 1)[1] for a in sys.argv if a.startswith("--gtest_output=")][0]
open(out, "w").write(
    '<?xml version="1.0"?><testsuites tests="1" failures="0" disabled="0" errors="0">'
    '<testsuite name="S" tests="1"><testcase name="T" classname="S"/></testsuite>'
    "</testsuites>"
)
"""

FAKE_GTEST = FAKE_GTEST_TEMPLATE.format(announce='mode == "enabled"')

# A test binary that clears the variable before the library reads it.
LOSES_THE_SETTING = FAKE_GTEST_TEMPLATE.format(announce="False")

ALWAYS_FORWARDS = FAKE_GTEST_TEMPLATE.format(announce="True")

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
            ("loses_the_setting.py", LOSES_THE_SETTING),
            ("always_forwards.py", ALWAYS_FORWARDS),
            ("fake_compare.py", RECORDS_ARGV),
            ("failing_compare.py", FAILS),
        ):
            path = self.tree / name
            path.write_text(body)
            path.chmod(0o755)

        # Where the harness's default report directory lands, so a test can see
        # whether it was cleaned up.
        self.tmp = self.tree / "tmp"
        self.tmp.mkdir()

    def run_harness(self, cwd, *extra, gtest="fake_gtest.py", env=None):
        return subprocess.run(
            [
                sys.executable,
                str(HARNESS),
                "--gtest",
                str(self.tree / gtest),
                "--filter",
                "*",
                "--lib-dir",
                str(self.tree / "lib"),
                "--compare",
                str(self.tree / "fake_compare.py"),
                *extra,
            ],
            cwd=str(cwd),
            env=dict(os.environ, TMPDIR=str(self.tmp), **(env or {})),
            capture_output=True,
            text=True,
        )

    def test_each_replay_gets_its_mode_and_the_libraries_under_test(self):
        result = self.run_harness(self.tree, env={"LD_LIBRARY_PATH": "/already/set"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for mode in ("disabled", "enabled"):
            recorded = json.loads((self.tree / f"env_{mode}.json").read_text())
            self.assertEqual(recorded["MIOPEN_HIPDNN_FORWARDING"], mode)
            first, *rest = recorded["LD_LIBRARY_PATH"].split(os.pathsep)
            self.assertEqual(Path(first).resolve(), (self.tree / "lib").resolve())
            self.assertEqual(rest, ["/already/set"])

    def test_each_replay_gets_its_own_database_and_kernel_cache(self):
        """A cache shared between the replays lets the enabled one reuse the disabled one's
        find results and compiled kernels instead of producing its own."""
        shared = {
            "MIOPEN_USER_DB_PATH": "/shared",
            "MIOPEN_CUSTOM_CACHE_DIR": "/shared",
        }
        result = self.run_harness(self.tree, env=shared)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        recorded = [
            json.loads((self.tree / f"env_{mode}.json").read_text())
            for mode in ("disabled", "enabled")
        ]
        for variable in shared:
            values = [r[variable] for r in recorded]
            self.assertNotIn("/shared", values, variable)
            self.assertEqual(len(set(values)), 2, variable)

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

    def test_the_temporary_report_directory_is_removed_after_a_pass(self):
        result = self.run_harness(self.tree)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_the_temporary_report_directory_is_kept_after_a_failure(self):
        result = self.run_harness(
            self.tree, "--compare", str(self.tree / "failing_compare.py")
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        [kept] = self.tmp.iterdir()
        self.assertIn(f"replay reports: {kept}", result.stdout)
        self.assertEqual(
            sorted(p.name for p in kept.iterdir()),
            [
                "fake_gtest.py_forwarding_disabled.xml",
                "fake_gtest.py_forwarding_enabled.xml",
            ],
        )

    def test_the_comparison_is_held_to_this_run_s_binary(self):
        """--newer-than is what stops a leftover pair of reports comparing cleanly."""
        result = self.run_harness(self.tree)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        argv = (self.tree / "fake_compare.argv").read_text().splitlines()
        self.assertIn("--newer-than", argv)
        self.assertEqual(
            argv[argv.index("--newer-than") + 1], str(self.tree / "fake_gtest.py")
        )

    def test_an_enabled_replay_that_never_forwarded_fails(self):
        result = self.run_harness(self.tree, gtest="loses_the_setting.py")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("forwarding=enabled replay never printed", result.stdout)
        self.assertFalse((self.tree / "fake_compare.argv").exists())

    def test_a_disabled_replay_that_forwarded_fails(self):
        result = self.run_harness(self.tree, gtest="always_forwards.py")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("forwarding=disabled replay printed", result.stdout)
        self.assertFalse((self.tree / "fake_compare.argv").exists())

    def test_the_replay_s_stderr_is_passed_through(self):
        result = self.run_harness(self.tree)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("MIOPEN_HIPDNN_FORWARDING=enabled:", result.stderr)

    def test_helpers_run_without_their_exec_bit(self):
        """They are launched through this interpreter, not their shebang lines."""
        (self.tree / "fake_compare.py").chmod(0o644)
        result = self.run_harness(self.tree)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_a_library_problem_fails_the_run(self):
        """Which files count is tested in test_miopen_wrapper_libs; this checks the
        harness stops on what it reports, using the default lib* search.

        Copied into bin/ so that search -- lib* beside its parent directory -- lands in
        this tree. miopen_wrapper_libs.py comes with it since the harness imports from
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
                "--gtest",
                str(self.tree / "fake_gtest.py"),
                "--filter",
                "*",
                "--compare",
                str(self.tree / "fake_compare.py"),
            ],
            cwd=str(self.tree),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn(
            f"no libMIOpen_private.so.* found under {self.tree / 'lib'}", result.stdout
        )
        self.assertFalse((self.tree / "env_disabled.json").exists())

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
