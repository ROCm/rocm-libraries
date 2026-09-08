# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for compare_forwarding_runs.py.

The comparator is the only thing that can turn a forwarding divergence into a
red build, so the cases worth covering are the ones where it could report
agreement that is not there: a name that appears twice in a run, and an XML that
a crashed replay left half-written. Comparing two well-formed, agreeing files
only ever exercises the passing path.

Everything here works on string fixtures, so no build, toolchain or GPU is
needed and the module runs in a lint lane:

    python -m pytest projects/miopen/script/test_compare_forwarding_runs.py
"""

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


def run(tmp_path, disabled_xml, enabled_xml):
    return cmp.main(
        write(tmp_path, "disabled.xml", disabled_xml),
        write(tmp_path, "enabled.xml", enabled_xml),
    )


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
    assert "zero tests" in capsys.readouterr().err
