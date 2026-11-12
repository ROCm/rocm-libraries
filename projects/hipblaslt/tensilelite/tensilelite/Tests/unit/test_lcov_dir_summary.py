# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for scripts/lcov_dir_summary.py, the coverage-cpp directory table.

Pure Python on hand-written LCOV text; no rocisa / HIP / llvm-cov needed.
"""

import importlib.util
import pathlib
import sys

import pytest

pytestmark = pytest.mark.unit

_SCRIPT = (
    pathlib.Path(__file__).resolve().parents[3] / "scripts" / "lcov_dir_summary.py"
)
_spec = importlib.util.spec_from_file_location("lcov_dir_summary", _SCRIPT)
summary = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(summary)

ROOT = pathlib.Path("/repo/tensilelite")


@pytest.mark.parametrize(
    "relative, group",
    [
        ("src/ContractionSolution.cpp", "src"),
        ("src/hip/HipSolutionAdapter.cpp", "src"),
        ("include/Tensile/ContractionProblem.hpp", "include"),
        ("client/src/Reference.cpp", "client/src"),
        ("client/include/Reference.hpp", "client/include"),
        ("rocisa/rocisa/src/pass/cycle.cpp", "rocisa/rocisa/src"),
        ("rocisa/rocisa/include/functions/f_math.hpp", "rocisa/rocisa/include"),
        ("tools/helper.cpp", "tools"),
        ("top.cpp", "."),
    ],
)
def test_group_is_path_through_first_src_or_include(relative, group):
    assert summary.group_for(ROOT / relative, ROOT) == group


def test_file_named_src_is_not_an_anchor():
    assert summary.group_for(ROOT / "lib" / "src", ROOT) == "lib"


def test_file_outside_root_keeps_absolute_parent():
    source = pathlib.Path("/opt/other/include/x.hpp")
    assert summary.group_for(source, ROOT) == "/opt/other/include"


def _write_lcov(path: pathlib.Path, records: list[tuple[str, int, int]]) -> None:
    """Write one record per (source, lines, hit lines); lines 1..hit are hit."""
    lines = []
    for source, found, hit in records:
        lines += ["TN:", f"SF:{source}"]
        lines += [f"DA:{n},{3 if n <= hit else 0}" for n in range(1, found + 1)]
        lines += [f"LF:{found}", f"LH:{hit}", "end_of_record"]
    path.write_text("\n".join(lines) + "\n")


def test_read_lcov_counts_da_lines_not_lf_lh(tmp_path):
    report = tmp_path / "a.lcov"
    report.write_text(
        f"SF:{tmp_path / 'src' / 'A.cpp'}\n"
        "DA:10,4\nDA:11,0\nDA:12,1\n"
        "LF:7\nLH:5\nend_of_record\n"
    )
    assert dict(summary.read_lcov(report, tmp_path.resolve())) == {"src": [3, 2]}


def test_read_lcov_merges_records_for_the_same_file(tmp_path):
    source = tmp_path / "src" / "A.cpp"
    report = tmp_path / "a.lcov"
    report.write_text(
        f"SF:{source}\nDA:1,0\nDA:2,5\nend_of_record\n"
        f"SF:{source}\nDA:1,2\nDA:3,0\nend_of_record\n"
    )
    assert dict(summary.read_lcov(report, tmp_path.resolve())) == {"src": [3, 2]}


def test_read_lcov_sums_lines_per_group(tmp_path):
    report = tmp_path / "a.lcov"
    _write_lcov(
        report,
        [
            (str(tmp_path / "src" / "A.cpp"), 100, 60),
            (str(tmp_path / "src" / "hip" / "B.cpp"), 20, 5),
            (str(tmp_path / "client" / "src" / "C.cpp"), 50, 0),
        ],
    )
    totals = summary.read_lcov(report, tmp_path.resolve())
    assert dict(totals) == {"src": [120, 65], "client/src": [50, 0]}


def test_main_prints_one_column_per_report_and_totals(tmp_path, monkeypatch, capsys):
    counted = tmp_path / "combined.lcov"
    advisory = tmp_path / "advisory_with_headers.lcov"
    _write_lcov(counted, [(str(tmp_path / "src" / "A.cpp"), 100, 60)])
    _write_lcov(
        advisory,
        [
            (str(tmp_path / "src" / "A.cpp"), 100, 60),
            (str(tmp_path / "include" / "A.hpp"), 40, 10),
        ],
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["lcov_dir_summary.py", "--root", str(tmp_path), str(counted), str(advisory)],
    )

    assert summary.main() == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert rows[0] == ["directory", "combined.lcov", "advisory_with_headers.lcov"]
    assert rows[1:] == [
        ["include", "-", "10/40", "(25.00%)"],
        ["src", "60/100", "(60.00%)", "60/100", "(60.00%)"],
        ["TOTAL", "60/100", "(60.00%)", "70/140", "(50.00%)"],
    ]


def test_main_fails_when_a_report_is_missing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["lcov_dir_summary.py", "--root", str(tmp_path), str(tmp_path / "none.lcov")],
    )
    assert summary.main() == 1
    assert "none.lcov not found" in capsys.readouterr().err
