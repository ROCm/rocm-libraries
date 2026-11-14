# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""IR-harness coverage: SLO201 (changed lines) and SLO202 (function bodies)."""
import ast
import json
import xml.etree.ElementTree as ET

import pytest
from conftest import copy_tree, src

import slo
import slo_check as K
import slo_coverage as COV
import slo_output as OUT


def test_body_lines_skip_def_lines_and_module_code():
    tree = ast.parse(
        src(
            '''
        X = 1

        def f(a):
            """doc"""
            return a

        class C:
            y = 2
            def g(self):
                return 3
    '''
        )
    )
    assert COV._body_lines(tree) == {4, 5, 10}


def test_changed_lines_below_target_are_flagged_with_ranges():
    current = {"files": {"k.py": {}}}
    cov = {"files": {"k.py": {"statements": [1, 2, 3, 4, 5, 6], "executed": [1, 2]}}}
    f = K.changed_line_findings(current, cov, {"k.py": {2, 3, 4, 6, 9}})
    assert len(f) == 1
    assert (f[0]["check"], f[0]["line"], f[0]["new"]) == ("SLO201", 3, 25.0)
    assert f[0]["message"].endswith("not run: 3-4, 6")
    assert K.changed_line_findings(current, cov, {"k.py": {1, 2}}) == []  # all ran
    assert K.changed_line_findings(current, cov, {"k.py": {9}}) == []  # no statements


def test_junit_lists_coverage_checks_only_when_coverage_ran():
    def names(xml):
        return {c.get("name").split()[0] for c in ET.fromstring(xml.split("\n", 1)[1])}

    assert "SLO202" not in names(
        OUT.junit([], {"files": {}, "coverage_ran": False}, "never")
    )
    assert "SLO202" in names(
        OUT.junit([], {"files": {}, "coverage_ran": True}, "never")
    )


@pytest.fixture(scope="module")
def tree(tmp_path_factory):
    """A copy of the real tree with a coverage baseline, and a covered kernel file."""
    pytest.importorskip("coverage")
    root = copy_tree(tmp_path_factory.mktemp("cov"))
    bl = root.parent / "baseline.json"
    args = [
        "baseline",
        "--update",
        "--coverage",
        "--root",
        str(root),
        "--baseline",
        str(bl),
    ]
    assert slo.main(args) == 0
    files = json.loads(bl.read_text())["files"]
    covered = min(p for p, r in files.items() if (r.get("cov_body") or 0) > 20)
    return root, bl, covered


def test_uncovered_new_code_is_caught(tree, tmp_path):
    root, bl, rel = tree
    path = root / rel
    orig = path.read_text()
    try:
        dead = (
            "\n\ndef _never_built(b, x):\n"
            + "".join(f"    x = b.add(x, b.const_i32({i}))\n" for i in range(50))
            + "    return x\n"
        )
        path.write_text(orig + dead)
        out = tmp_path / "f.json"
        rc = slo.main(
            [
                "check",
                "--root",
                str(root),
                "--baseline",
                str(bl),
                "--coverage",
                "--base",
                "HEAD",
                "--json",
                str(out),
            ]
        )
        k = {
            (f["check"], f["kind"], f["path"])
            for f in json.loads(out.read_text())["findings"]
        }
        assert rc == 0  # warn only
        assert ("SLO202", "worse", rel) in k  # body coverage fell
        assert ("SLO201", "over_target", rel) in k  # new lines never run
    finally:
        path.write_text(orig)
