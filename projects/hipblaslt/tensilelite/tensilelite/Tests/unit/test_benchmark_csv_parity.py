# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Python side of the CSV parity check against the client's ResultFileReporter.

The golden CSVs and database dumps in tests/data/benchmark_db are produced and checked by the C++
test tests/BenchmarkCsvParity_test.cpp, which describes the exact comparison rules. Here the CSV
that writeResultsCsv writes from each dump must equal the golden CSV byte for byte.
"""

from __future__ import annotations

import difflib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from tensilelite.BenchmarkDatabase import writeResultsCsv

from benchmark_db_test_helpers import DatabaseBuilder, parse_dump

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "tests" / "data" / "benchmark_db"

SCENARIOS = [
    "default",
    "merge",
    "extra_cols",
    "cu_efficiency",
    "prediction",
    "late_columns",
    "late_columns_merge",
]


def read_fixture(name: str) -> bytes:
    path = FIXTURE_DIR / name
    if not path.is_file():
        pytest.fail(f"fixture {path} is missing; run the C++ BenchmarkCsvParity test to print it")
    return path.read_bytes()


def readable_diff(expected: bytes, actual: bytes) -> str:
    expected_lines = expected.decode("utf-8").splitlines(keepends=True)
    actual_lines = actual.decode("utf-8").splitlines(keepends=True)
    return "".join(difflib.unified_diff(expected_lines, actual_lines, "golden", "actual"))


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_python_csv_matches_golden_csv(scenario: str, tmp_path: Path):
    golden = read_fixture(f"{scenario}.csv")
    dump = read_fixture(f"{scenario}.db.txt").decode("utf-8")

    builder = DatabaseBuilder(tmp_path / "bench.db")
    builder.load_dump(dump)
    db_path = builder.finish()
    csv_path = tmp_path / "results.csv"
    writeResultsCsv(db_path, csv_path)

    actual = csv_path.read_bytes()
    assert actual == golden, f"{scenario}.csv differs:\n{readable_diff(golden, actual)}"


def test_parse_dump_reads_every_record_kind():
    text = (
        "meta\tperformance_metric\tDeviceEfficiency\n"
        "solution\t0\tSolA\n"
        "problem\t0\n"
        "problem_value\t0\tSizeI\t64\n"
        "run\t0\t0\tINDIRECTLY_FAILED\tINDIRECTLY_FAILED\t0\n"
        "run_value\t0\t0\tgflops\t1.5\n"
    )
    assert parse_dump(text) == [
        ("meta", ["performance_metric", "DeviceEfficiency"]),
        ("solution", ["0", "SolA"]),
        ("problem", ["0"]),
        ("problem_value", ["0", "SizeI", "64"]),
        ("run", ["0", "0", "INDIRECTLY_FAILED", "INDIRECTLY_FAILED", "0"]),
        ("run_value", ["0", "0", "gflops", "1.5"]),
    ]


def test_parse_dump_keeps_empty_fields():
    assert parse_dump("run\t1\t2\tNOT_BENCHMARKED\t\t\n") == [
        ("run", ["1", "2", "NOT_BENCHMARKED", "", ""])
    ]


@pytest.mark.parametrize(
    "text, message",
    [
        ("session\t1\t2\n", "line 1: unknown record kind 'session'"),
        ("problem\t0\nrun\t0\t0\tPASS\n", "line 2: run has 3 fields, expected 5"),
        ("run\t0\t0\tDONE\t\t\n", "line 1: unknown run status 'DONE'"),
        ("problem\t0", "does not end with a newline"),
        ("\n", "unknown record kind ''"),
    ],
)
def test_parse_dump_rejects_malformed_text(text: str, message: str):
    with pytest.raises(ValueError, match=message):
        parse_dump(text)
