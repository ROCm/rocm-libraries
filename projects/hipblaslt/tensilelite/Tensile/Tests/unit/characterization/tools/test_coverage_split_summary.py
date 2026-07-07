# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the characterization-vs-unit coverage summary card.

These pin the card's data handling so a green run cannot silently start
reporting wrong numbers: JUnit counts exclude skips (and degrade to ``None``
rather than crash on bad input), the line-level "who covers what" split is set
arithmetic (both / char-only / unit-only / union), the aligned table stays valid
Markdown with right-aligned numeric columns, and ``main`` wires JUnit counts and
the ``$GITHUB_STEP_SUMMARY`` sink together.
"""

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest


def _has_cell(md: str, value: str) -> bool:
    """True if a padded table cell holds exactly ``value`` (ignoring padding)."""
    return re.search(r"\|\s*" + re.escape(value) + r"\s*\|", md) is not None

_TOOLS_DIR = Path(__file__).resolve().parent
_MODULE_PATH = _TOOLS_DIR / "coverage_split_summary.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("coverage_split_summary", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


css = _load_module()

pytestmark = pytest.mark.unit


def _cov(files: dict[str, list[int]], total: float) -> dict:
    """coverage.py-shaped report from {path: executed_lines} plus a total pct."""
    return {
        "meta": {"format": 3},
        "files": {
            path: {"executed_lines": lines} for path, lines in files.items()
        },
        "totals": {"percent_covered": total},
    }


def _write(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


_JUNIT_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite tests="{n}">{cases}</testsuite></testsuites>"""


def _junit(path: Path, passed: int, skipped: int) -> Path:
    cases = ["<testcase name='p{}'/>".format(i) for i in range(passed)]
    cases += ["<testcase name='s{}'><skipped/></testcase>".format(i) for i in range(skipped)]
    path.write_text(
        _JUNIT_TEMPLATE.format(n=passed + skipped, cases="".join(cases)),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------- #
# _junit_ran                                                                   #
# --------------------------------------------------------------------------- #
def test_junit_ran_excludes_skipped(tmp_path):
    xml = _junit(tmp_path / "t.xml", passed=7, skipped=3)
    assert css._junit_ran(str(xml)) == 7


def test_junit_ran_all_passed(tmp_path):
    xml = _junit(tmp_path / "t.xml", passed=5, skipped=0)
    assert css._junit_ran(str(xml)) == 5


def test_junit_ran_missing_file_is_none(tmp_path):
    assert css._junit_ran(str(tmp_path / "nope.xml")) is None


def test_junit_ran_malformed_is_none(tmp_path):
    bad = tmp_path / "bad.xml"
    bad.write_text("<testsuite><testcase", encoding="utf-8")
    assert css._junit_ran(str(bad)) is None


# --------------------------------------------------------------------------- #
# _executed_line_set / _pct                                                    #
# --------------------------------------------------------------------------- #
def test_executed_line_set_pairs_file_and_line():
    report = _cov({"a.py": [1, 2], "b.py": [5]}, 50.0)
    assert css._executed_line_set(report) == {("a.py", 1), ("a.py", 2), ("b.py", 5)}


def test_executed_line_set_tolerates_missing_keys():
    assert css._executed_line_set({}) == set()
    assert css._executed_line_set({"files": {"a.py": {}}}) == set()


def test_pct_reads_totals():
    assert css._pct(_cov({"a.py": [1]}, 73.5)) == 73.5


# --------------------------------------------------------------------------- #
# _aligned_table                                                               #
# --------------------------------------------------------------------------- #
def test_aligned_table_columns_line_up_and_separator_marks_right():
    lines = css._aligned_table(
        ["Suite", "Pct"], [["Characterization", "9.9%"]], right={1}
    )
    # header, separator, one row
    assert len(lines) == 3
    # every rendered line is the same width (columns line up as raw text)
    assert len({len(x) for x in lines}) == 1
    # right column separator ends with ':' ; left column is plain dashes
    header, sep, row = lines
    assert sep.startswith("| -") and sep.rstrip().endswith(": |")
    # the short header cell is padded out to the widest cell in its column
    assert "Suite" in header and "Pct" in header


# --------------------------------------------------------------------------- #
# build_markdown                                                               #
# --------------------------------------------------------------------------- #
def test_build_markdown_line_level_split_is_set_arithmetic():
    # char reaches a.py:1,2 ; unit reaches a.py:2 and b.py:1
    char = _cov({"a.py": [1, 2]}, 40.0)
    unit = _cov({"a.py": [2], "b.py": [1]}, 30.0)
    md = css.build_markdown(char, unit, None, char_tests=10, unit_tests=20)

    assert "characterization vs unit" in md
    # union is {a:1, a:2, b:1} = 3 lines; both = {a:2} = 1
    assert "Union (any suite)" in md and _has_cell(md, "3")
    assert "Both suites" in md and _has_cell(md, "1")
    # test counts render, combined row absent when combined is None
    assert _has_cell(md, "10") and _has_cell(md, "20")
    assert "**Combined**" not in md


def test_build_markdown_includes_combined_and_dashes_for_missing_counts():
    char = _cov({"a.py": [1]}, 40.0)
    unit = _cov({"a.py": [1]}, 40.0)
    combined = _cov({"a.py": [1]}, 55.5)
    md = css.build_markdown(char, unit, combined, char_tests=None, unit_tests=None)
    assert "**Combined**" in md and "55.50%" in md
    # missing test counts collapse to '-'
    assert _has_cell(md, "-")


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def test_main_derives_counts_from_junit_and_writes_step_summary(tmp_path, capsys, monkeypatch):
    char = _write(tmp_path / "char.json", _cov({"a.py": [1, 2]}, 40.0))
    unit = _write(tmp_path / "unit.json", _cov({"a.py": [2], "b.py": [1]}, 30.0))
    char_xml = _junit(tmp_path / "char.xml", passed=12, skipped=1)
    unit_xml = _junit(tmp_path / "unit.xml", passed=34, skipped=0)
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    rc = css.main([
        "--characterization", str(char),
        "--unit", str(unit),
        "--characterization-junit", str(char_xml),
        "--unit-junit", str(unit_xml),
    ])

    assert rc == 0
    out = capsys.readouterr().out
    # skipped test is excluded from the derived count (12, not 13)
    assert "12" in out and "34" in out
    # the same card is appended to the job-summary sink
    assert summary.read_text(encoding="utf-8").strip() == out.strip()


def test_main_explicit_counts_override_junit(tmp_path, capsys, monkeypatch):
    char = _write(tmp_path / "char.json", _cov({"a.py": [1]}, 40.0))
    unit = _write(tmp_path / "unit.json", _cov({"a.py": [1]}, 40.0))
    char_xml = _junit(tmp_path / "char.xml", passed=999, skipped=0)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    rc = css.main([
        "--characterization", str(char),
        "--unit", str(unit),
        "--characterization-tests", "7",
        "--characterization-junit", str(char_xml),
    ])
    assert rc == 0
    out = capsys.readouterr().out
    # explicit --characterization-tests wins over the JUnit-derived 999
    assert "7" in out and "999" not in out
