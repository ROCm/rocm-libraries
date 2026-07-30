# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Unit tests for the characterization-vs-unit coverage summary card.

These pin the card's data handling so a green run cannot silently start
reporting wrong numbers: JUnit counts exclude skips (and degrade to ``None``
rather than crash on bad input), the line-level "who covers what" split is set
arithmetic (both / char-only / unit-only / union), the largest-files leaderboard
ranks by measurable statements and reports statement-level per-suite percentages
(skipping files with no honest denominator, and degrading to no table at all
rather than failing), the aligned table stays valid Markdown with right-aligned
numeric columns, and ``main`` wires JUnit counts, the ``--top-files`` limit, and
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


def _cov(
    files: dict[str, list[int]],
    total: float,
    num_statements: int | None = None,
    file_stmts: dict[str, int] | None = None,
) -> dict:
    """coverage.py-shaped report from {path: executed_lines} plus a total pct.

    ``num_statements`` populates ``totals.num_statements`` (covered + uncovered),
    which the card needs from the combined report to compute the untested surface.

    ``file_stmts`` populates each file's ``summary.num_statements``, the per-file
    denominator the largest-files leaderboard ranks and divides by. A path left
    out of it gets no ``summary`` at all, which is how a report that does not know
    a file's size is represented.
    """
    totals: dict = {"percent_covered": total}
    if num_statements is not None:
        totals["num_statements"] = num_statements
    file_entries: dict[str, dict] = {}
    for path, lines in files.items():
        entry: dict = {"executed_lines": lines}
        if file_stmts and path in file_stmts:
            entry["summary"] = {"num_statements": file_stmts[path]}
        file_entries[path] = entry
    return {
        "meta": {"format": 3},
        "files": file_entries,
        "totals": totals,
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
# _file_stmts / top_files_rows (largest-files leaderboard)                      #
# --------------------------------------------------------------------------- #
def test_file_stmts_reads_per_file_num_statements():
    report = _cov({"a.py": [1], "b.py": [2]}, 50.0, file_stmts={"a.py": 10, "b.py": 4})
    assert css._file_stmts(report) == {"a.py": 10, "b.py": 4}


def test_file_stmts_takes_largest_known_and_skips_unknown():
    # Only the second report knows a.py's size; b.py's size is nowhere.
    without = _cov({"a.py": [1], "b.py": [2]}, 50.0)
    with_size = _cov({"a.py": [1]}, 50.0, file_stmts={"a.py": 12})
    assert css._file_stmts(without, with_size) == {"a.py": 12}


def test_file_stmts_ignores_none_reports_and_null_summary():
    report = _cov({"a.py": [1]}, 50.0)
    report["files"]["a.py"]["summary"] = None
    assert css._file_stmts(None, report) == {}


def test_top_files_rows_ranks_by_statements_descending():
    stmts = {"small.py": 10, "big.py": 100, "mid.py": 50}
    char = _cov({p: [] for p in stmts}, 0.0, file_stmts=stmts)
    unit = _cov({p: [] for p in stmts}, 0.0, file_stmts=stmts)
    rows = css.top_files_rows(char, unit, None, limit=10)
    assert [r[0] for r in rows] == ["big.py", "mid.py", "small.py"]


def test_top_files_rows_honors_limit():
    stmts = {"a.py": 30, "b.py": 20, "c.py": 10}
    char = _cov({p: [] for p in stmts}, 0.0, file_stmts=stmts)
    rows = css.top_files_rows(char, char, None, limit=2)
    assert [r[0] for r in rows] == ["a.py", "b.py"]


def test_top_files_rows_percentages_are_statement_level():
    # 10 statements; unit reaches 4 of them, characterization reaches 7.
    char = _cov({"a.py": [1, 2, 3, 4, 5, 6, 7]}, 70.0, file_stmts={"a.py": 10})
    unit = _cov({"a.py": [1, 2, 3, 4]}, 40.0, file_stmts={"a.py": 10})
    (row,) = css.top_files_rows(char, unit, None, limit=10)
    path, stmts, unit_pct, char_pct, char_only = row
    assert (path, stmts) == ("a.py", "10")
    assert (unit_pct, char_pct) == ("40.0%", "70.0%")
    # char-only is per-file set arithmetic: {5,6,7} reached by char but not unit
    assert char_only == "3"


def test_top_files_rows_char_only_is_zero_when_unit_covers_everything():
    char = _cov({"a.py": [1, 2]}, 20.0, file_stmts={"a.py": 10})
    unit = _cov({"a.py": [1, 2, 3]}, 30.0, file_stmts={"a.py": 10})
    (row,) = css.top_files_rows(char, unit, None, limit=10)
    assert row[4] == "0"


def test_top_files_rows_file_missing_from_one_suite_counts_as_zero():
    # b.py exists only in the characterization report.
    char = _cov({"b.py": [1, 2, 3, 4, 5]}, 50.0, file_stmts={"b.py": 10})
    unit = _cov({"a.py": [1]}, 10.0, file_stmts={"a.py": 4})
    rows = {r[0]: r for r in css.top_files_rows(char, unit, None, limit=10)}
    assert rows["b.py"][2] == "0.0%"  # no unit coverage at all
    assert rows["b.py"][3] == "50.0%"
    assert rows["b.py"][4] == "5"  # all 5 char lines are char-only
    assert rows["a.py"][3] == "0.0%"  # and the reverse direction


def test_top_files_rows_skips_files_with_unknown_or_zero_statements():
    char = _cov(
        {"known.py": [1], "unknown.py": [1], "empty.py": [1]},
        50.0,
        file_stmts={"known.py": 5, "empty.py": 0},
    )
    rows = css.top_files_rows(char, char, None, limit=10)
    # unknown.py has no summary and empty.py has no statements, so neither has an
    # honest denominator; only known.py is rankable.
    assert [r[0] for r in rows] == ["known.py"]


def test_top_files_rows_breaks_ties_by_path_for_stable_output():
    stmts = {"z.py": 20, "a.py": 20}
    char = _cov({p: [] for p in stmts}, 0.0, file_stmts=stmts)
    rows = css.top_files_rows(char, char, None, limit=10)
    assert [r[0] for r in rows] == ["a.py", "z.py"]


def test_top_files_rows_takes_statements_from_combined_report():
    # Neither per-suite report knows the size; the combined one does.
    char = _cov({"a.py": [1, 2]}, 20.0)
    unit = _cov({"a.py": [3]}, 10.0)
    combined = _cov({"a.py": [1, 2, 3]}, 30.0, file_stmts={"a.py": 10})
    (row,) = css.top_files_rows(char, unit, combined, limit=10)
    assert row[1] == "10" and row[2] == "10.0%" and row[3] == "20.0%"


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
    # No combined report -> falls back to the union-denominator table.
    md = css.build_markdown(char, unit, None, char_tests=10, unit_tests=20)

    assert "characterization vs unit" in md
    # union is {a:1, a:2, b:1} = 3 lines; both = {a:2} = 1
    assert "Union (any suite)" in md and _has_cell(md, "3")
    assert "Both suites" in md and _has_cell(md, "1")
    # without a combined report there is no untested-surface row
    assert "No test coverage" not in md
    # test counts render, combined row absent when combined is None
    assert _has_cell(md, "10") and _has_cell(md, "20")
    assert "**Combined**" not in md


def test_build_markdown_all_statements_split_shows_untested_surface():
    # char reaches a.py:1,2 ; unit reaches a.py:2 and b.py:1 -> union covers 3
    char = _cov({"a.py": [1, 2]}, 40.0)
    unit = _cov({"a.py": [2], "b.py": [1]}, 30.0)
    # 5 measurable statements total, so 3 covered leaves 2 with no test coverage.
    combined = _cov({"a.py": [1, 2], "b.py": [1]}, 60.0, num_statements=5)
    md = css.build_markdown(char, unit, combined, char_tests=10, unit_tests=20)

    assert "**Combined**" in md and "60.00%" in md
    assert "share of all measurable statements" in md
    # covered = 3/5 = 60.00% ; untested = 2/5 = 40.00% ; they sum to 100%
    assert "Covered by any suite" in md and _has_cell(md, "3")
    assert "No test coverage" in md and _has_cell(md, "2")
    assert "60.00%" in md and "40.00%" in md
    assert "Total statements" in md
    # char-only = {a:1} = 1 is the migration debt; unit-only = {b:1} = 1
    assert "Characterization only" in md and "Unit only" in md


def test_build_markdown_renders_leaderboard_when_file_sizes_are_known():
    stmts = {"big.py": 100, "small.py": 10}
    char = _cov({"big.py": [1, 2], "small.py": [1]}, 40.0, file_stmts=stmts)
    unit = _cov({"big.py": [1], "small.py": [1, 2]}, 30.0, file_stmts=stmts)
    combined = _cov(
        {"big.py": [1, 2], "small.py": [1, 2]}, 60.0, num_statements=110,
        file_stmts=stmts,
    )
    md = css.build_markdown(char, unit, combined, 10, 20, top_files=20)

    # Header names the actual row count, and the biggest file leads.
    assert "Largest 2 files: unit vs characterization coverage" in md
    assert md.index("big.py") < md.index("small.py")
    assert _has_cell(md, "big.py") and _has_cell(md, "100")
    # big.py: unit reaches 1/100, char reaches 2/100, char-only is line 2.
    assert _has_cell(md, "1.0%") and _has_cell(md, "2.0%")


def test_build_markdown_leaderboard_omitted_when_disabled_or_sizeless():
    stmts = {"a.py": 10}
    char = _cov({"a.py": [1]}, 10.0, file_stmts=stmts)
    unit = _cov({"a.py": [2]}, 10.0, file_stmts=stmts)

    # Explicitly disabled.
    assert "Largest" not in css.build_markdown(char, unit, None, 1, 1, top_files=0)
    # No per-file sizes anywhere, so there is no honest denominator to rank by.
    sizeless = _cov({"a.py": [1]}, 10.0)
    assert "Largest" not in css.build_markdown(sizeless, sizeless, None, 1, 1)


def test_build_markdown_includes_combined_and_dashes_for_missing_counts():
    char = _cov({"a.py": [1]}, 40.0)
    unit = _cov({"a.py": [1]}, 40.0)
    combined = _cov({"a.py": [1]}, 55.5, num_statements=2)
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


def test_main_top_files_flag_limits_the_leaderboard(tmp_path, capsys, monkeypatch):
    stmts = {"big.py": 100, "small.py": 10}
    char = _write(
        tmp_path / "char.json", _cov({"big.py": [1]}, 10.0, file_stmts=stmts)
    )
    unit = _write(
        tmp_path / "unit.json", _cov({"small.py": [1]}, 10.0, file_stmts=stmts)
    )
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    rc = css.main([
        "--characterization", str(char),
        "--unit", str(unit),
        "--top-files", "1",
    ])
    assert rc == 0
    out = capsys.readouterr().out
    # Only the largest file makes the cut.
    assert "Largest 1 files" in out
    assert "big.py" in out and "small.py" not in out


def test_main_top_files_zero_omits_the_leaderboard(tmp_path, capsys, monkeypatch):
    stmts = {"a.py": 10}
    char = _write(tmp_path / "char.json", _cov({"a.py": [1]}, 10.0, file_stmts=stmts))
    unit = _write(tmp_path / "unit.json", _cov({"a.py": [2]}, 10.0, file_stmts=stmts))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)

    rc = css.main([
        "--characterization", str(char),
        "--unit", str(unit),
        "--top-files", "0",
    ])
    assert rc == 0
    assert "Largest" not in capsys.readouterr().out


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
