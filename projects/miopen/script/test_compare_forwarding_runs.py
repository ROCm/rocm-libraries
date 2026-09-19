# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for compare_forwarding_runs.py.

The comparator is the only thing that can turn a forwarding divergence into a
red build, so the cases worth covering are the ones where it could report
agreement that is not there: a name that appears twice in a run, an XML that a
crashed replay left half-written, a pair of reports left over from an earlier
build, and two runs that skipped everything. Comparing two well-formed, agreeing
files only ever exercises the passing path.

Everything here works on string fixtures, so no build, toolchain or GPU is
needed and the module runs in a lint lane:

    python -m pytest projects/miopen/script/test_compare_forwarding_runs.py
"""

import os
import sys
from pathlib import Path

# Colocated with the script under test. pytest's default import mode already
# puts this directory on sys.path; do it explicitly so the module also runs
# under a bare `python -m pytest` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_forwarding_runs as cmp  # noqa: E402


def suite(*cases):
    """Build a JUnit document from (name, status) pairs, statuses as gtest emits them."""
    body = []
    for name, status in cases:
        if status == "failed":
            body.append(
                '<testcase name="{}" classname="Shim"><failure message="x"/></testcase>'.format(
                    name
                )
            )
        elif status == "skipped":
            body.append(
                '<testcase name="{}" classname="Shim" status="notrun"/>'.format(name)
            )
        else:
            body.append('<testcase name="{}" classname="Shim"/>'.format(name))
    return '<?xml version="1.0"?><testsuites>{}</testsuites>'.format("".join(body))


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def run(tmp_path, disabled_xml, enabled_xml, newer_than=None):
    return cmp.main(
        write(tmp_path, "disabled.xml", disabled_xml),
        write(tmp_path, "enabled.xml", enabled_xml),
        newer_than,
    )


def aged(path, seconds):
    """Backdate a file, so freshness can be tested without waiting for the clock."""
    stamp = os.path.getmtime(path) - seconds
    os.utime(path, (stamp, stamp))
    return path


def test_identical_runs_agree(tmp_path, capsys):
    xml = suite(("A", "passed"), ("B", "skipped"))
    assert run(tmp_path, xml, xml) == 0
    assert "2 tests identical" in capsys.readouterr().out


def test_status_divergence_is_reported(tmp_path, capsys):
    rc = run(
        tmp_path,
        suite(("A", "passed")),
        suite(("A", "failed")),
    )
    assert rc == 1
    assert "disabled=passed enabled=failed" in capsys.readouterr().err


def test_duplicate_name_divergence_is_not_collapsed(tmp_path, capsys):
    # Same name twice in each run, disagreeing on one of the two. Keyed by name
    # alone, the second entry would overwrite the first and the two runs would
    # look identical.
    rc = run(
        tmp_path,
        suite(("A", "passed"), ("A", "passed")),
        suite(("A", "passed"), ("A", "failed")),
    )
    assert rc == 1
    err = capsys.readouterr().err
    assert "2 entries" in err
    assert "failed" in err


def test_duplicate_name_agreement_still_passes(tmp_path):
    # Emission order is not part of the claim, so the same outcomes in the other
    # order are agreement, not a divergence.
    assert (
        run(
            tmp_path,
            suite(("A", "passed"), ("A", "failed")),
            suite(("A", "failed"), ("A", "passed")),
        )
        == 0
    )


def test_truncated_xml_gives_a_diagnostic_not_a_traceback(tmp_path, capsys):
    good = suite(("A", "passed"))
    truncated = good[: len(good) // 2]
    rc = run(tmp_path, good, truncated)
    assert rc == 1
    err = capsys.readouterr().err
    assert "not well-formed XML" in err
    assert "crashed partway" in err


def test_two_empty_runs_do_not_pass(tmp_path, capsys):
    empty = suite()
    assert run(tmp_path, empty, empty) == 1
    assert "neither run executed a test" in capsys.readouterr().err


def test_two_all_skipped_runs_do_not_pass(tmp_path, capsys):
    # Non-empty and identical, so every other check here is satisfied. Nothing ran,
    # so the agreement says nothing about forwarding.
    skipped = suite(("A", "skipped"), ("B", "skipped"))
    assert run(tmp_path, skipped, skipped) == 1
    assert "neither run executed a test" in capsys.readouterr().err


def test_one_executed_test_is_enough(tmp_path, capsys):
    xml = suite(("A", "passed"), ("B", "skipped"))
    assert run(tmp_path, xml, xml) == 0
    assert "1 executed, 1 skipped" in capsys.readouterr().out


def test_a_missing_report_is_not_agreement(tmp_path, capsys):
    disabled = write(tmp_path, "disabled.xml", suite(("A", "passed")))
    rc = cmp.main(disabled, str(tmp_path / "never_written.xml"))
    assert rc == 1
    assert "did not run" in capsys.readouterr().err


def test_reports_older_than_the_binary_are_rejected(tmp_path, capsys):
    # The harness's worst failure mode: both files are well-formed and agree, but
    # they are the previous build's output and neither replay ran this time.
    xml = suite(("A", "passed"))
    binary = tmp_path / "miopen_gtest"
    binary.touch()
    rc = run(tmp_path, xml, xml, newer_than=str(binary))
    assert rc == 0, "a report written after the binary is current"

    aged(tmp_path / "disabled.xml", 60)
    aged(tmp_path / "enabled.xml", 60)
    assert (
        cmp.main(
            str(tmp_path / "disabled.xml"), str(tmp_path / "enabled.xml"), str(binary)
        )
        == 1
    )
    assert "left over from an earlier build" in capsys.readouterr().err


def test_a_missing_newer_than_target_is_rejected(tmp_path, capsys):
    # Nothing to date the reports against means their freshness is unknown, which
    # is not the same as fresh.
    xml = suite(("A", "passed"))
    rc = run(tmp_path, xml, xml, newer_than=str(tmp_path / "no_such_binary"))
    assert rc == 1
    assert "cannot be shown to be current" in capsys.readouterr().err
