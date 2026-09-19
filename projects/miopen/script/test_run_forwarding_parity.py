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
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

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


@pytest.fixture
def tree(tmp_path):
    """A stand-in install tree: the libraries the harness resolves, plus its helpers."""
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "libMIOpen.so.1.0").touch()
    (lib / "libMIOpen_private.so.1.0").touch()
    for name, body in (
        ("fake_gtest.py", FAKE_GTEST),
        ("fake_compare.py", RECORDS_ARGV),
        ("fake_abi.py", PASSES),
    ):
        path = tmp_path / name
        path.write_text(body)
        path.chmod(0o755)
    return tmp_path


def run(tree, cwd, *extra):
    return subprocess.run(
        [
            sys.executable,
            str(HARNESS),
            "--gtest",
            str(tree / "fake_gtest.py"),
            "--filter",
            "*",
            "--lib-dir",
            str(tree / "lib"),
            "--compare",
            str(tree / "fake_compare.py"),
            "--abi-check",
            str(tree / "fake_abi.py"),
            "--baseline",
            "/dev/null",
            "--excluded",
            "/dev/null",
            *extra,
        ],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


def test_nothing_is_written_into_the_working_directory(tree):
    workdir = tree / "bin" / "MIOpen"
    workdir.mkdir(parents=True)
    result = run(tree, workdir)
    assert result.returncode == 0, result.stdout + result.stderr
    assert list(workdir.iterdir()) == []


@pytest.mark.skipif(
    os.geteuid() == 0,
    reason="root writes through the mode bits, so the read-only case cannot be created",
)
def test_a_read_only_working_directory_still_passes(tree):
    """The shipping case: an artifact the runner may not write to."""
    workdir = tree / "bin" / "MIOpen"
    workdir.mkdir(parents=True)
    workdir.chmod(0o555)
    try:
        result = run(tree, workdir)
    finally:
        workdir.chmod(0o755)
    assert result.returncode == 0, result.stdout + result.stderr


def test_reports_land_where_output_dir_names(tree):
    """The build tree passes one explicitly and keeps its reports."""
    out = tree / "test_results"
    result = run(tree, tree, "--output-dir", str(out))
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(p.name for p in out.iterdir()) == [
        "fake_gtest.py_forwarding_disabled.xml",
        "fake_gtest.py_forwarding_enabled.xml",
    ]


def test_the_report_directory_is_reported(tree):
    """A failing replay is only diagnosable if its reports can be found."""
    result = run(tree, tree)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "replay reports:" in result.stdout


def test_the_comparison_is_held_to_this_run_s_binary(tree):
    """--newer-than is what stops a leftover pair of reports comparing cleanly."""
    result = run(tree, tree)
    assert result.returncode == 0, result.stdout + result.stderr
    argv = (tree / "fake_compare.argv").read_text().splitlines()
    assert "--newer-than" in argv
    assert argv[argv.index("--newer-than") + 1] == str(tree / "fake_gtest.py")


def test_helpers_run_without_their_exec_bit(tree):
    """They are launched through this interpreter, not their shebang lines."""
    (tree / "fake_compare.py").chmod(0o644)
    (tree / "fake_abi.py").chmod(0o644)
    result = run(tree, tree)
    assert result.returncode == 0, result.stdout + result.stderr


def test_two_versioned_libraries_are_refused_rather_than_picked_between(tree):
    """One of them is an earlier build, and filename order is not version order."""
    (tree / "lib" / "libMIOpen.so.10.0").touch()
    result = run(tree, tree)
    assert result.returncode == 1
    assert "more than one libMIOpen.so.*" in result.stdout


def test_the_libraries_under_test_are_named(tree):
    """A co-versioning failure downstream has to be tied back to concrete files."""
    result = run(tree, tree)
    assert result.returncode == 0, result.stdout + result.stderr
    assert str(tree / "lib" / "libMIOpen.so.1.0") in result.stdout
    assert str(tree / "lib" / "libMIOpen_private.so.1.0") in result.stdout
