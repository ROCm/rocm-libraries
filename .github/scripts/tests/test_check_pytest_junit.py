# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import sys
from pathlib import Path

sys.path.insert(0, os.fspath(Path(__file__).parent.parent))
import check_pytest_junit as cpj


def _junit(path, cases):
    """Write a junit file; ``cases`` is [(classname, name, skip_message_or_None)]."""
    rows = []
    for classname, name, skip in cases:
        body = "" if skip is None else f'<skipped message="{skip}"/>'
        rows.append(
            f'<testcase classname="{classname}" name="{name}">{body}</testcase>'
        )
    path.write_text(
        '<?xml version="1.0"?><testsuites><testsuite name="pytest">'
        + "".join(rows)
        + "</testsuite></testsuites>"
    )
    return path


def _allow(path, text):
    path.write_text(text)
    return path


def _run(junit, allow, capsys):
    rc = cpj.main(["--junit", str(junit), "--allow-skips", str(allow)])
    return rc, capsys.readouterr().out


def test_all_passing_with_empty_allow_list_is_ok(tmp_path, capsys):
    junit = _junit(tmp_path / "suite.xml", [("tests.test_a", "test_one", None)])
    rc, out = _run(junit, _allow(tmp_path / "allow.txt", "# nothing\n"), capsys)
    assert rc == 0
    assert "OK 1 testcase(s) in 1 junit file(s), 0 allow-listed skip(s)" in out


def test_unlisted_skip_fails(tmp_path, capsys):
    junit = _junit(
        tmp_path / "suite.xml", [("tests.test_a", "test_one", "no compiler")]
    )
    rc, out = _run(junit, _allow(tmp_path / "allow.txt", ""), capsys)
    assert rc == 1
    assert "unexpected-skip" in out
    assert "suite::tests.test_a::test_one" in out
    assert "no compiler" in out


def test_allow_listed_skip_passes(tmp_path, capsys):
    junit = _junit(
        tmp_path / "suite.xml",
        [("tests.test_a.TestReal", "test_x[p1]", "needs input")],
    )
    allow = _allow(
        tmp_path / "allow.txt",
        "suite::tests.test_a.TestReal::* # needs input by design\n",
    )
    rc, out = _run(junit, allow, capsys)
    assert rc == 0
    assert "1 allow-listed skip(s)" in out


def test_pattern_matching_no_skip_is_stale(tmp_path, capsys):
    junit = _junit(tmp_path / "suite.xml", [("tests.test_a", "test_one", None)])
    allow = _allow(
        tmp_path / "allow.txt", "suite::tests.test_gone::* # was skipped once\n"
    )
    rc, out = _run(junit, allow, capsys)
    assert rc == 1
    assert "stale-allow" in out


def test_pattern_is_scoped_to_its_junit_file(tmp_path, capsys):
    junit = _junit(tmp_path / "other.xml", [("tests.test_a", "test_one", "skip")])
    allow = _allow(tmp_path / "allow.txt", "suite::tests.test_a::* # wrong suite\n")
    rc, out = _run(junit, allow, capsys)
    assert rc == 1
    assert "unexpected-skip" in out
    assert "stale-allow" in out


def test_missing_junit_is_unreadable(tmp_path, capsys):
    rc, out = _run(tmp_path / "absent.xml", _allow(tmp_path / "allow.txt", ""), capsys)
    assert rc == 1
    assert "junit-unreadable" in out


def test_junit_without_testcases_is_empty(tmp_path, capsys):
    junit = _junit(tmp_path / "suite.xml", [])
    rc, out = _run(junit, _allow(tmp_path / "allow.txt", ""), capsys)
    assert rc == 1
    assert "junit-empty" in out


def test_allow_list_line_without_reason_is_malformed(tmp_path, capsys):
    junit = _junit(tmp_path / "suite.xml", [("tests.test_a", "test_one", "skip")])
    allow = _allow(tmp_path / "allow.txt", "suite::tests.test_a::*\n")
    rc, out = _run(junit, allow, capsys)
    assert rc == 1
    assert "allow-list-malformed" in out


def test_several_junit_files_are_all_checked(tmp_path, capsys):
    clean = _junit(tmp_path / "a.xml", [("tests.t", "test_one", None)])
    skipping = _junit(tmp_path / "b.xml", [("tests.t", "test_two", "skip")])
    allow = _allow(tmp_path / "allow.txt", "")
    rc = cpj.main(
        ["--junit", str(clean), "--junit", str(skipping), "--allow-skips", str(allow)]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "b::tests.t::test_two" in out
