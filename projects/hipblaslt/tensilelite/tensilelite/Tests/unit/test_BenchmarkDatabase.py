# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import math
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Dict

import pytest

pytestmark = pytest.mark.unit

from tensilelite import BenchmarkDatabase
from tensilelite.BenchmarkDatabase import (
    BenchmarkDatabaseError,
    Pair,
    RunStatus,
    countPairsByStatus,
    formatDouble,
    formatSummary,
    getLatestPairSessionId,
    getPairs,
    getSessions,
    openDatabase,
    schemaVersion,
    summarize,
    writeResultsCsv,
    _readSnapshot,
)

from benchmark_db_test_helpers import (
    DatabaseBuilder,
    extract_schema_sql,
    extract_schema_version,
    schema_sql,
    schema_version,
)

PASS = RunStatus.PASS
FAILURE = RunStatus.FAILURE
UNKNOWN = RunStatus.UNKNOWN
NOT_BENCHMARKED = RunStatus.NOT_BENCHMARKED
INDIRECTLY_FAILED = RunStatus.INDIRECTLY_FAILED
INDIRECT_VERIFICATION_FAILURE = RunStatus.INDIRECT_VERIFICATION_FAILURE
NOT_SELECTED = RunStatus.NOT_SELECTED

DEFAULT_HEADER = "GFlops, SizeI, SizeJ, LDD, LDC, LDA, LDB, TotalFlops, GbpsBW"
EXTRA_HEADER = "TilesPerCu, TotalGranularity, WinnerGFlops, WinnerTimeUS, WinnerIdx, WinnerName"


def pass_values(
    gflops: str, time_us: str = "10", bandwidth: str = "100", **extra: str
) -> Dict[str, str]:
    values = {"gflops": gflops, "time-us": time_us, "gbps-bandwidth": bandwidth}
    values.update({key.replace("_", "-"): value for key, value in extra.items()})
    return values


def generate_csv(db_path: Path, tmp_path: Path) -> str:
    csv_path = tmp_path / "out.csv"
    writeResultsCsv(db_path, csv_path)
    return csv_path.read_bytes().decode("utf-8")


@pytest.fixture
def builder(tmp_path: Path) -> DatabaseBuilder:
    return DatabaseBuilder(tmp_path / "bench.db")


def two_solution_builder(tmp_path: Path, **options) -> DatabaseBuilder:
    result = DatabaseBuilder(tmp_path / "bench.db", **options)
    result.add_solution(0, "kA")
    result.add_solution(1, "kB")
    return result


# ---------------------------------------------------------------------------
# 1. openDatabase.
# ---------------------------------------------------------------------------


def test_open_missing_file(tmp_path: Path):
    path = tmp_path / "missing.db"
    with pytest.raises(BenchmarkDatabaseError, match="does not exist") as error:
        openDatabase(path)
    assert str(path) in str(error.value)
    assert not path.exists()


def test_open_not_a_database(tmp_path: Path):
    path = tmp_path / "text.db"
    path.write_text("this is not a database, " * 100)
    with pytest.raises(BenchmarkDatabaseError, match="not a benchmark database") as error:
        openDatabase(path)
    assert str(path) in str(error.value)


def test_open_missing_tables(tmp_path: Path):
    path = tmp_path / "partial.db"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        connection.commit()
    with pytest.raises(BenchmarkDatabaseError, match="missing tables") as error:
        openDatabase(path)
    assert "runs" in str(error.value)
    assert "meta" not in str(error.value).split("missing tables")[1]


def test_open_wrong_schema_version(builder: DatabaseBuilder):
    builder.connection.execute("PRAGMA user_version = 7")
    path = builder.finish()
    with pytest.raises(BenchmarkDatabaseError) as error:
        openDatabase(path)
    assert "schema version 7" in str(error.value)
    assert "expected 1" in str(error.value)


def test_open_absent_schema_version(builder: DatabaseBuilder):
    builder.connection.execute("PRAGMA user_version = 0")
    path = builder.finish()
    with pytest.raises(BenchmarkDatabaseError) as error:
        openDatabase(path)
    assert "schema version 0" in str(error.value)


def test_open_success(builder: DatabaseBuilder):
    path = builder.finish()
    with closing(openDatabase(path)) as connection:
        assert getPairs(connection, RunStatus) == []


# ---------------------------------------------------------------------------
# 2. pairs and sessions.
# ---------------------------------------------------------------------------


def make_mixed_database(tmp_path: Path) -> Path:
    # Solutions and problems are inserted out of order on purpose.
    db = DatabaseBuilder(tmp_path / "mixed.db")
    db.add_solution(1, "kB")
    db.add_solution(0, "kA")
    db.add_problem(1, [8, 8])
    db.add_problem(0, [4, 4])
    db.add_problem(2, [2, 2])
    db.add_problem(3, [1, 1])
    db.add_session(1, 5.0)
    db.add_session(2, None)
    db.set_run(3, 0, NOT_SELECTED, session=2)
    db.set_run(1, 1, FAILURE, "FAILED", session=2)
    db.set_run(1, 0, PASS, "PASSED", pass_values("5"), session=1)
    db.set_run(0, 1, UNKNOWN, session=2)
    db.set_run(0, 0, NOT_BENCHMARKED)
    db.set_run(2, 0, INDIRECTLY_FAILED, "INDIRECTLY_FAILED", session=2, caused_by=0)
    db.set_run(2, 1, INDIRECT_VERIFICATION_FAILURE, "INDIRECT_VERIFICATION_FAILURE", caused_by=1)
    return db.finish()


def test_get_pairs_ordered(tmp_path: Path):
    with closing(openDatabase(make_mixed_database(tmp_path))) as connection:
        pairs = getPairs(connection, RunStatus)
    assert [(p.problemIndex, p.libraryIndex) for p in pairs] == [
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
        (3, 0),
    ]
    assert [p.solutionName for p in pairs] == ["kA", "kB", "kA", "kB", "kA", "kB", "kA"]
    assert [p.status for p in pairs] == [
        NOT_BENCHMARKED,
        UNKNOWN,
        PASS,
        FAILURE,
        INDIRECTLY_FAILED,
        INDIRECT_VERIFICATION_FAILURE,
        NOT_SELECTED,
    ]
    assert [p.sessionId for p in pairs] == [None, 2, 1, 2, 2, None, 2]
    assert [p.validation for p in pairs] == [
        "",
        "",
        "PASSED",
        "FAILED",
        "INDIRECTLY_FAILED",
        "INDIRECT_VERIFICATION_FAILURE",
        "",
    ]
    assert [p.causedByProblemIndex for p in pairs] == [None, None, None, None, 0, 1, None]
    assert pairs[2].startTime == 1.0


def test_get_pairs_by_status(tmp_path: Path):
    with closing(openDatabase(make_mixed_database(tmp_path))) as connection:
        for status in RunStatus:
            pairs = getPairs(connection, [status])
            assert len(pairs) == 1
            assert pairs[0].status == status


def test_caused_by_is_the_public_problem_index_not_the_table_id(tmp_path: Path):
    db = two_solution_builder(tmp_path)
    db.add_problem(0, [8, 8])
    db.add_problem(7, [8, 8])
    db.set_run(0, 0, INDIRECTLY_FAILED, caused_by=7)
    db.set_run(0, 1, PASS, "PASSED")
    db.set_run(7, 0, INDIRECT_VERIFICATION_FAILURE, caused_by=0)
    db.set_run(7, 1, PASS, "PASSED")
    with closing(openDatabase(db.finish())) as connection:
        pairs = getPairs(connection, RunStatus)
    assert [p.causedByProblemIndex for p in pairs] == [7, None, 0, None]


def test_get_sessions(tmp_path: Path):
    with closing(openDatabase(make_mixed_database(tmp_path))) as connection:
        sessions = getSessions(connection)
    assert [s.id for s in sessions] == [1, 2]
    assert [s.died for s in sessions] == [False, True]
    assert sessions[0].endTime == 5.0
    assert sessions[1].endTime is None


def test_count_pairs_by_status(tmp_path: Path, builder: DatabaseBuilder):
    with closing(openDatabase(make_mixed_database(tmp_path))) as connection:
        assert countPairsByStatus(connection) == {status: 1 for status in RunStatus}
    path = builder.finish()
    with closing(openDatabase(path)) as connection:
        assert countPairsByStatus(connection) == {status: 0 for status in RunStatus}


def test_latest_pair_session_id_of_empty_runs_is_zero(builder: DatabaseBuilder):
    with closing(openDatabase(builder.finish())) as connection:
        assert getLatestPairSessionId(connection) == 0


def test_latest_pair_session_id_counts_marked_pairs(tmp_path: Path):
    db = DatabaseBuilder(tmp_path / "bench.db")
    db.add_solution(0, "k0")
    db.add_solution(1, "k1")
    db.add_problem(0, [8, 8])
    db.add_session(3, None)
    db.set_run(0, 0, PASS, "PASSED", session=2)
    db.set_run(0, 1, INDIRECTLY_FAILED, "INDIRECTLY_FAILED", session=3, caused_by=0)
    with closing(openDatabase(db.finish())) as connection:
        assert getLatestPairSessionId(connection) == 3


def test_latest_pair_session_id_ignores_pairs_without_a_session(tmp_path: Path):
    db = DatabaseBuilder(tmp_path / "bench.db")
    db.add_solution(0, "k0")
    db.add_problem(0, [8, 8])
    db.add_session(5, None)
    db.set_run(0, 0, NOT_BENCHMARKED)
    with closing(openDatabase(db.finish())) as connection:
        assert getLatestPairSessionId(connection) == 0


# ---------------------------------------------------------------------------
# 3. default metric CSV.
# ---------------------------------------------------------------------------


def make_default_database(tmp_path: Path) -> Path:
    db = DatabaseBuilder(tmp_path / "bench.db")
    # Insertion order differs from library_index order.
    db.add_solution(2, "kC")
    db.add_solution(0, "kA")
    db.add_solution(1, "kB")
    db.add_problem(0, [512, 1024], [512, 512, 1024, 1024], 536870912)
    db.add_problem(1, [64, 32], [64, 64, 64, 64], 4096)
    db.add_problem(2, [16, 16], [16, 16, 16, 16], 100)
    db.set_run(0, 0, PASS, "PASSED", pass_values("25149.4", bandwidth="232.849"))
    db.set_run(0, 1, FAILURE, "FAILED", pass_values("999999", bandwidth="9999"))
    db.set_run(0, 2, PASS, "PASSED", pass_values("12040.3", bandwidth="300.5"))
    db.set_run(1, 0, UNKNOWN)
    db.set_run(1, 1, NOT_BENCHMARKED)
    db.set_run(1, 2, PASS, "PASSED", pass_values("7", bandwidth="1"))
    db.set_run(2, 0, FAILURE, "FAILED", pass_values("1", bandwidth="50"))
    db.set_run(2, 1, UNKNOWN)
    db.set_run(2, 2, NOT_BENCHMARKED)
    return db.finish()


def test_default_csv(tmp_path: Path):
    expected = (
        f"{DEFAULT_HEADER}, kA, kB, kC\n"
        "0, 512, 1024, 512, 512, 1024, 1024, 536870912, 300.5, 25149.4, -1, 12040.3\n"
        "1, 64, 32, 64, 64, 64, 64, 4096, 1, -1, -1, 7\n"
        "2, 16, 16, 16, 16, 16, 16, 100, -1, -1, -1, -1\n"
    )
    assert generate_csv(make_default_database(tmp_path), tmp_path) == expected


def test_cu_efficiency_csv(tmp_path: Path):
    db = two_solution_builder(tmp_path, metric="CUEfficiency")
    db.add_problem(0, [8, 8], [8, 8, 8, 8], 1)
    db.set_run(0, 0, PASS, "PASSED", pass_values("100", **{"gflops_per_cu": "3.5"}))
    db.set_run(0, 1, PASS, "PASSED", pass_values("200", **{"gflops_per_cu": "2.25"}))
    expected = (
        "GFlopsPerCU, SizeI, SizeJ, LDD, LDC, LDA, LDB, TotalFlops, GbpsBW, kA, kB\n"
        "0, 8, 8, 8, 8, 8, 8, 1, 100, 3.5, 2.25\n"
    )
    assert generate_csv(db.finish(), tmp_path) == expected


# ---------------------------------------------------------------------------
# 5. extra columns.
# ---------------------------------------------------------------------------


def test_extra_columns_winner_and_fastest(tmp_path: Path):
    db = two_solution_builder(tmp_path, extra_cols=True)
    db.add_problem(0, [8, 8], [8, 8, 8, 8], 1)
    # kA is the fastest, kB has the best gflops.
    db.set_run(
        0,
        0,
        PASS,
        "PASSED",
        pass_values("100", time_us="5", tiles_per_cu="1.5", total_gran="0.75", solution_index="40"),
    )
    db.set_run(
        0,
        1,
        PASS,
        "PASSED",
        pass_values("200", time_us="9", tiles_per_cu="2.5", total_gran="0.5", solution_index="41"),
    )
    expected = (
        f"GFlops, SizeI, SizeJ, LDD, LDC, LDA, LDB, TotalFlops, GbpsBW, {EXTRA_HEADER}, kA, kB\n"
        "0, 8, 8, 8, 8, 8, 8, 1, 100, 1.5, 0.75, 200, 5, 41, kB, 100, 200\n"
    )
    assert generate_csv(db.finish(), tmp_path) == expected


def test_extra_columns_winner_index_falls_back_to_library_index(tmp_path: Path):
    db = two_solution_builder(tmp_path, extra_cols=True)
    db.add_problem(0, [8, 8], [8, 8, 8, 8], 1)
    db.set_run(0, 0, PASS, "PASSED", pass_values("100"))
    db.set_run(0, 1, PASS, "PASSED", pass_values("300"))
    lines = generate_csv(db.finish(), tmp_path).splitlines()
    assert lines[1] == "0, 8, 8, 8, 8, 8, 8, 1, 100, -1, -1, 300, 10, 1, kB, 100, 300"


def test_extra_columns_all_failed(tmp_path: Path):
    db = two_solution_builder(tmp_path, extra_cols=True)
    db.add_problem(0, [8, 8], [8, 8, 8, 8], 1)
    db.set_run(0, 0, FAILURE, "FAILED", pass_values("100"))
    db.set_run(0, 1, UNKNOWN)
    lines = generate_csv(db.finish(), tmp_path).splitlines()
    assert lines[1] == "0, 8, 8, 8, 8, 8, 8, 1, -1, -1, -1, -1, -1, -1, , -1, -1"


def test_extra_columns_nan_time_and_gflops_are_ignored(tmp_path: Path):
    db = two_solution_builder(tmp_path, extra_cols=True)
    db.add_problem(0, [8, 8], [8, 8, 8, 8], 1)
    db.set_run(0, 0, PASS, "PASSED", pass_values("-nan", time_us="-nan", bandwidth="-nan"))
    db.set_run(0, 1, PASS, "PASSED", pass_values("50", time_us="7", bandwidth="20"))
    lines = generate_csv(db.finish(), tmp_path).splitlines()
    # The NaN gflops of kA is written verbatim but it never wins.
    assert lines[1] == "0, 8, 8, 8, 8, 8, 8, 1, 20, -1, -1, 50, 7, 1, kB, -nan, 50"


def test_extra_columns_last_row_carries_failed_pair_values(tmp_path: Path):
    db = two_solution_builder(tmp_path, extra_cols=True)
    db.add_problem(0, [8, 8], [8, 8, 8, 8], 1)
    db.set_run(0, 0, FAILURE, "FAILED", {"tiles-per-cu": "3", "total-gran": "0.25"})
    db.set_run(0, 1, PASS, "PASSED", pass_values("50", time_us="7"))
    lines = generate_csv(db.finish(), tmp_path).splitlines()
    assert lines[1] == "0, 8, 8, 8, 8, 8, 8, 1, 100, 3, 0.25, 50, 7, 1, kB, -1, 50"


# ---------------------------------------------------------------------------
# 6. merge mode.
# ---------------------------------------------------------------------------


def make_quoting_database(tmp_path: Path, merge: bool) -> Path:
    db = DatabaseBuilder(tmp_path / f"quote_{merge}.db", merge=merge)
    db.add_solution(0, 'a,"b"')
    db.add_problem(0, [8, 8], [8, 8, 8, 8], 1)
    # Values are copied verbatim, so any stored text exercises the escaping.
    db.connection.execute("UPDATE problem_values SET value = '1,5' WHERE key = 'ldd'")
    db.set_run(0, 0, PASS, "PASSED", pass_values("1"))
    return db.finish()


def test_single_escaping_without_merge(tmp_path: Path):
    expected = f'{DEFAULT_HEADER}, "a,""b"""\n0, 8, 8, "1,5", 8, 8, 8, 1, 100, 1\n'
    assert generate_csv(make_quoting_database(tmp_path, merge=False), tmp_path) == expected


def test_double_escaping_in_merge_mode(tmp_path: Path):
    # The header is escaped once, the row values twice.
    expected = f'{DEFAULT_HEADER}, "a,""b"""\n0, 8, 8, """1,5""", 8, 8, 8, 1, 100, 1\n'
    assert generate_csv(make_quoting_database(tmp_path, merge=True), tmp_path) == expected


def test_merge_mode_plain_values_are_unchanged(tmp_path: Path):
    db = two_solution_builder(tmp_path, merge=True)
    db.add_problem(0, [8, 8], [8, 8, 8, 8], 1)
    db.add_problem(1, [4, 4], [4, 4, 4, 4], 2)
    db.set_run(0, 0, PASS, "PASSED", pass_values("1"))
    db.set_run(0, 1, PASS, "PASSED", pass_values("2"))
    db.set_run(1, 0, PASS, "PASSED", pass_values("3"))
    db.set_run(1, 1, PASS, "PASSED", pass_values("4"))
    expected = (
        f"{DEFAULT_HEADER}, kA, kB\n"
        "0, 8, 8, 8, 8, 8, 8, 1, 100, 1, 2\n"
        "1, 4, 4, 4, 4, 4, 4, 2, 100, 3, 4\n"
    )
    assert generate_csv(db.finish(), tmp_path) == expected


# ---------------------------------------------------------------------------
# 7. number formatting.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        (25149.4, "25149.4"),
        (1000000.0, "1e+06"),
        (123456.0, "123456"),
        (0.5, "0.5"),
        (-1.0, "-1"),
        (0.0, "0"),
        (1.5e-07, "1.5e-07"),
        (math.inf, "inf"),
        (-math.inf, "-inf"),
        (math.nan, "nan"),
        (-math.nan, "-nan"),
    ],
)
def test_format_double(value: float, expected: str):
    assert formatDouble(value) == expected


def test_format_double_nan_sign_survives_text_round_trip():
    assert formatDouble(float("-nan")) == "-nan"
    assert formatDouble(float("nan")) == "nan"


def test_nan_bandwidth_and_time_text_are_ignored(tmp_path: Path):
    db = two_solution_builder(tmp_path, extra_cols=True)
    db.add_problem(0, [8, 8], [8, 8, 8, 8], 1)
    db.set_run(0, 0, PASS, "PASSED", pass_values("10", time_us="-nan", bandwidth="-nan"))
    db.set_run(0, 1, PASS, "PASSED", pass_values("20", time_us="nan", bandwidth="nan"))
    lines = generate_csv(db.finish(), tmp_path).splitlines()
    assert lines[1] == "0, 8, 8, 8, 8, 8, 8, 1, -1, -1, -1, 20, -1, 1, kB, 10, 20"


# ---------------------------------------------------------------------------
# 8. rows.
# ---------------------------------------------------------------------------


def test_problem_without_values_has_no_row(tmp_path: Path):
    db = two_solution_builder(tmp_path)
    db.add_problem(0)
    db.add_problem(1, [8, 8], [8, 8, 8, 8], 1)
    db.set_run(0, 0, NOT_BENCHMARKED)
    db.set_run(0, 1, NOT_BENCHMARKED)
    db.set_run(1, 0, PASS, "PASSED", pass_values("1"))
    db.set_run(1, 1, PASS, "PASSED", pass_values("2"))
    expected = f"{DEFAULT_HEADER}, kA, kB\n1, 8, 8, 8, 8, 8, 8, 1, 100, 1, 2\n"
    assert generate_csv(db.finish(), tmp_path) == expected


def test_no_rows_gives_empty_file(tmp_path: Path):
    db = two_solution_builder(tmp_path)
    db.add_problem(0)
    db.set_run(0, 0, NOT_BENCHMARKED)
    db.set_run(0, 1, NOT_BENCHMARKED)
    assert generate_csv(db.finish(), tmp_path) == ""


def test_duplicate_solution_names_share_a_column(tmp_path: Path):
    db = DatabaseBuilder(tmp_path / "bench.db")
    db.add_solution(0, "same")
    db.add_solution(1, "same")
    db.add_problem(0, [8, 8], [8, 8, 8, 8], 1)
    db.set_run(0, 0, PASS, "PASSED", pass_values("1"))
    db.set_run(0, 1, PASS, "PASSED", pass_values("2"))
    expected = f"{DEFAULT_HEADER}, same\n0, 8, 8, 8, 8, 8, 8, 1, 100, 2\n"
    assert generate_csv(db.finish(), tmp_path) == expected


def test_csv_has_failed_cells_for_indirect_statuses(tmp_path: Path):
    db = DatabaseBuilder(tmp_path / "bench.db")
    db.add_solution(0, "kA")
    db.add_solution(1, "kB")
    db.add_solution(2, "kC")
    db.add_problem(0, [8, 8])
    db.set_run(0, 0, INDIRECTLY_FAILED, "INDIRECTLY_FAILED", caused_by=1)
    db.set_run(0, 1, INDIRECT_VERIFICATION_FAILURE, "INDIRECT_VERIFICATION_FAILURE", caused_by=1)
    db.set_run(0, 2, PASS, "PASSED", pass_values("2"))
    expected = f"{DEFAULT_HEADER}, kA, kB, kC\n0, 8, 8, 8, 8, 8, 8, 0, 100, -1, -1, 2\n"
    assert generate_csv(db.finish(), tmp_path) == expected


# ---------------------------------------------------------------------------
# 9. summary.
# ---------------------------------------------------------------------------


def make_summary_database(tmp_path: Path) -> Path:
    db = DatabaseBuilder(tmp_path / "summary.db")
    for index in range(3):
        db.add_solution(index, f"k{index}")
    for index in range(2):
        db.add_problem(index, [8, 8])
    db.add_session(1, None)  # Died.
    db.add_session(2, 20.0)  # Clean, but follows a dead session.
    db.add_session(3, 30.0)  # Clean, follows a clean session.
    db.set_run(0, 0, FAILURE, "FAILED", session=1)  # Failure in the dead session itself.
    db.set_run(0, 1, FAILURE, "FAILED", session=2)  # Tainted.
    db.set_run(0, 2, PASS, "PASSED", pass_values("1"), session=2)
    db.set_run(1, 0, FAILURE, None, session=3)  # Not tainted, and has no verdict.
    db.set_run(1, 1, UNKNOWN, session=1)
    db.set_run(1, 2, NOT_BENCHMARKED)
    return db.finish()


def test_summarize(tmp_path: Path):
    summary = summarize(make_summary_database(tmp_path))
    assert summary.counts == {
        PASS: 1,
        FAILURE: 3,
        UNKNOWN: 1,
        NOT_BENCHMARKED: 1,
        INDIRECTLY_FAILED: 0,
        INDIRECT_VERIFICATION_FAILURE: 0,
        NOT_SELECTED: 0,
    }
    assert summary.indirectlyFailedPairs == []
    assert summary.indirectVerificationFailurePairs == []
    assert summary.notBenchmarkedCount == 1
    assert [(p.problemIndex, p.libraryIndex) for p in summary.unknownPairs] == [(1, 1)]
    assert len(summary.failurePairs) == 3
    assert [(p.problemIndex, p.libraryIndex) for p in summary.taintedPairs] == [(0, 1)]
    assert [s.id for s in summary.deadSessions] == [1]


def test_format_summary(tmp_path: Path):
    expected = "\n".join(
        [
            "benchmark database: 6 pairs: 1 PASS, 3 FAILURE, 1 UNKNOWN, 1 NOT_BENCHMARKED, "
            "0 INDIRECTLY_FAILED, 0 INDIRECT_VERIFICATION_FAILURE, 0 NOT_SELECTED",
            "pairs left UNKNOWN (the client died while running them):",
            "  problem 1, solution 1 k1",
            "pairs that FAILED:",
            "  problem 0, solution 0 k0: FAILED",
            "  problem 0, solution 1 k1: FAILED",
            "  problem 1, solution 0 k0: no verdict",
            "pairs that failed right after a session that died "
            "(possibly caused by an earlier GPU fault, not retried):",
            "  problem 0, solution 1 k1: FAILED",
            "sessions that died: 1",
        ]
    )
    assert formatSummary(summarize(make_summary_database(tmp_path))) == expected


def test_format_summary_of_clean_database_has_only_the_counts(tmp_path: Path):
    db = two_solution_builder(tmp_path)
    db.add_problem(0, [8, 8])
    db.add_session(1, 2.0)
    db.set_run(0, 0, PASS, "PASSED", pass_values("1"), session=1)
    db.set_run(0, 1, PASS, "PASSED", pass_values("2"), session=1)
    expected = (
        "benchmark database: 2 pairs: 2 PASS, 0 FAILURE, 0 UNKNOWN, 0 NOT_BENCHMARKED, "
        "0 INDIRECTLY_FAILED, 0 INDIRECT_VERIFICATION_FAILURE, 0 NOT_SELECTED"
    )
    assert formatSummary(summarize(db.finish())) == expected


def make_indirect_database(tmp_path: Path) -> Path:
    db = DatabaseBuilder(tmp_path / "indirect.db")
    for index in range(3):
        db.add_solution(index, f"k{index}")
    for index in range(4):
        db.add_problem(index, [8, 8])
    db.add_session(1, None)  # Died.
    db.add_session(2, 20.0)  # Clean and follows a dead session.
    db.set_run(0, 2, UNKNOWN, session=1)
    db.set_run(1, 2, INDIRECTLY_FAILED, "INDIRECTLY_FAILED", session=2, caused_by=0)
    db.set_run(2, 2, INDIRECTLY_FAILED, "INDIRECTLY_FAILED", session=2, caused_by=0)
    db.set_run(3, 2, INDIRECTLY_FAILED, "INDIRECTLY_FAILED", session=2, caused_by=0)
    db.set_run(0, 0, FAILURE, "FAILED", session=2)
    db.set_run(1, 0, INDIRECT_VERIFICATION_FAILURE, "INDIRECT_VERIFICATION_FAILURE", caused_by=0)
    db.set_run(2, 0, INDIRECT_VERIFICATION_FAILURE, "INDIRECT_VERIFICATION_FAILURE", caused_by=0)
    db.set_run(3, 0, INDIRECT_VERIFICATION_FAILURE, "INDIRECT_VERIFICATION_FAILURE", caused_by=3)
    db.set_run(0, 1, PASS, "PASSED", pass_values("1"), session=2)
    db.set_run(1, 1, PASS, "PASSED", pass_values("1"), session=2)
    db.set_run(2, 1, PASS, "PASSED", pass_values("1"), session=2)
    db.set_run(3, 1, PASS, "PASSED", pass_values("1"), session=2)
    return db.finish()


def test_summarize_separates_the_indirect_statuses(tmp_path: Path):
    summary = summarize(make_indirect_database(tmp_path))
    assert summary.counts[INDIRECTLY_FAILED] == 3
    assert summary.counts[INDIRECT_VERIFICATION_FAILURE] == 3
    assert [(p.problemIndex, p.libraryIndex) for p in summary.indirectlyFailedPairs] == [
        (1, 2),
        (2, 2),
        (3, 2),
    ]
    assert len(summary.indirectVerificationFailurePairs) == 3
    assert [(p.problemIndex, p.libraryIndex) for p in summary.failurePairs] == [(0, 0)]
    assert [(p.problemIndex, p.libraryIndex) for p in summary.taintedPairs] == [(0, 0)]


def test_format_summary_groups_indirect_pairs_by_solution_and_cause(tmp_path: Path):
    expected = "\n".join(
        [
            "benchmark database: 12 pairs: 4 PASS, 1 FAILURE, 1 UNKNOWN, 0 NOT_BENCHMARKED, "
            "3 INDIRECTLY_FAILED, 3 INDIRECT_VERIFICATION_FAILURE, 0 NOT_SELECTED",
            "pairs left UNKNOWN (the client died while running them):",
            "  problem 0, solution 2 k2",
            "pairs that FAILED:",
            "  problem 0, solution 0 k0: FAILED",
            "pairs not launched because their solution crashed, faulted or hung on another "
            "problem:",
            "  solution 2 k2: 3 pairs, caused by problem 0",
            "pairs not launched because their solution failed verification on another problem:",
            "  solution 0 k0: 2 pairs, caused by problem 0",
            "  solution 0 k0: 1 pairs, caused by problem 3",
            "pairs that failed right after a session that died "
            "(possibly caused by an earlier GPU fault, not retried):",
            "  problem 0, solution 0 k0: FAILED",
            "sessions that died: 1",
        ]
    )
    assert formatSummary(summarize(make_indirect_database(tmp_path))) == expected


def make_not_selected_database(tmp_path: Path) -> Path:
    db = DatabaseBuilder(tmp_path / "not_selected.db")
    for index in range(3):
        db.add_solution(index, f"k{index}")
    db.add_problem(0, [8, 8])
    db.add_problem(1, [8, 8])
    db.add_session(1, 10.0)
    db.set_run(0, 0, PASS, "PASSED", pass_values("1"), session=1)
    db.set_run(0, 1, NOT_SELECTED, session=1)
    db.set_run(0, 2, NOT_SELECTED, session=1)
    db.set_run(1, 0, NOT_SELECTED, session=1)
    db.set_run(1, 1, PASS, "PASSED", pass_values("2"), session=1)
    db.set_run(1, 2, NOT_SELECTED, session=1)
    return db.finish()


def test_not_selected_pairs_are_counted_last_but_are_neither_failures_nor_listed(tmp_path: Path):
    path = make_not_selected_database(tmp_path)
    summary = summarize(path)
    assert summary.counts[NOT_SELECTED] == 4
    assert summary.counts[PASS] == 2
    assert summary.notBenchmarkedCount == 0
    assert summary.failurePairs == []
    assert summary.unknownPairs == []
    assert summary.indirectlyFailedPairs == []
    assert summary.indirectVerificationFailurePairs == []
    assert summary.taintedPairs == []
    expected = (
        "benchmark database: 6 pairs: 2 PASS, 0 FAILURE, 0 UNKNOWN, 0 NOT_BENCHMARKED, "
        "0 INDIRECTLY_FAILED, 0 INDIRECT_VERIFICATION_FAILURE, 4 NOT_SELECTED"
    )
    assert formatSummary(summary) == expected
    with closing(openDatabase(path)) as connection:
        pairs = getPairs(connection, [NOT_SELECTED])
    assert [(p.problemIndex, p.libraryIndex) for p in pairs] == [(0, 1), (0, 2), (1, 0), (1, 2)]
    assert all(p.causedByProblemIndex is None and p.validation == "" for p in pairs)


def test_run_status_values_agree_with_the_schema(tmp_path: Path):
    assert [int(status) for status in RunStatus] == list(range(7))
    assert NOT_SELECTED.name == "NOT_SELECTED"
    db = DatabaseBuilder(tmp_path / "bench.db")
    db.add_solution(0, "k0")
    db.add_problem(0, [8, 8])
    for status in RunStatus:
        caused_by = 0 if status in (INDIRECTLY_FAILED, INDIRECT_VERIFICATION_FAILURE) else None
        db.connection.execute("DELETE FROM runs")
        db.set_run(0, 0, status, caused_by=caused_by)
    db.connection.execute("DELETE FROM runs")
    with pytest.raises(sqlite3.IntegrityError):
        db.set_run(0, 0, len(RunStatus))


def late_column_builder(tmp_path: Path, **options) -> DatabaseBuilder:
    db = DatabaseBuilder(tmp_path / "bench.db", **options)
    for index, name in enumerate(["kA", "kB", "kC"]):
        db.add_solution(index, name)
    for index in range(3):
        db.add_problem(index, [8, 8], [8, 8, 8, 8], 1)
    return db


def test_csv_cell_of_a_not_selected_pair_is_empty_not_failed(tmp_path: Path):
    db = two_solution_builder(tmp_path)
    db.add_problem(0, [8, 8], [8, 8, 8, 8], 1)
    db.add_problem(1, [8, 8], [8, 8, 8, 8], 1)
    db.set_run(0, 0, PASS, "PASSED", pass_values("3"))
    db.set_run(0, 1, NOT_SELECTED)
    db.set_run(1, 0, NOT_BENCHMARKED)
    db.set_run(1, 1, PASS, "PASSED", pass_values("4"))
    # kB is first visited by problem 1, so it has a cell there but no header.
    expected = (
        f"{DEFAULT_HEADER}, kA\n"
        "0, 8, 8, 8, 8, 8, 8, 1, 100, 3\n"
        "1, 8, 8, 8, 8, 8, 8, 1, 100, -1, 4\n"
    )
    assert generate_csv(db.finish(), tmp_path) == expected


def test_not_selected_pair_does_not_change_the_winner_or_the_best_of_columns(tmp_path: Path):
    db = two_solution_builder(tmp_path, extra_cols=True)
    db.add_problem(0, [8, 8], [8, 8, 8, 8], 1)
    passed = pass_values("3", time_us="10", tiles_per_cu="2", total_gran="0.5")
    # A NOT_SELECTED pair is never visited, so even faster stored values must not count.
    faster = pass_values("99", time_us="1", tiles_per_cu="7", total_gran="0.9")
    db.set_run(0, 0, PASS, "PASSED", passed)
    db.set_run(0, 1, NOT_SELECTED, values=faster)
    expected = (
        f"{DEFAULT_HEADER}, {EXTRA_HEADER}, kA\n"
        "0, 8, 8, 8, 8, 8, 8, 1, 100, 2, 0.5, 3, 10, 0, kA, 3\n"
    )
    assert generate_csv(db.finish(), tmp_path) == expected


def test_not_selected_solution_neither_gets_a_column_nor_hides_a_namesake(tmp_path: Path):
    db = DatabaseBuilder(tmp_path / "bench.db")
    db.add_solution(0, "same")
    db.add_solution(1, "same")
    db.add_solution(2, "never")
    db.add_problem(0, [8, 8], [8, 8, 8, 8], 1)
    db.set_run(0, 0, PASS, "PASSED", pass_values("1"))
    db.set_run(0, 1, NOT_SELECTED)
    db.set_run(0, 2, NOT_SELECTED)
    expected = f"{DEFAULT_HEADER}, same\n0, 8, 8, 8, 8, 8, 8, 1, 100, 1\n"
    assert generate_csv(db.finish(), tmp_path) == expected


def fill_late_columns(db: DatabaseBuilder) -> None:
    # Problem 0 visits kA only, problem 1 adds kC, and problem 2 adds kB.
    db.set_run(0, 0, PASS, "PASSED", pass_values("1"))
    db.set_run(0, 1, NOT_SELECTED)
    db.set_run(0, 2, NOT_SELECTED)
    db.set_run(1, 0, NOT_SELECTED)
    db.set_run(1, 1, NOT_SELECTED)
    db.set_run(1, 2, PASS, "PASSED", pass_values("2"))
    db.set_run(2, 0, PASS, "PASSED", pass_values("3"))
    db.set_run(2, 1, PASS, "PASSED", pass_values("4"))
    db.set_run(2, 2, NOT_SELECTED)


def test_csv_header_has_the_first_problems_solutions_and_rows_grow(tmp_path: Path):
    db = late_column_builder(tmp_path)
    fill_late_columns(db)
    expected = (
        f"{DEFAULT_HEADER}, kA\n"
        "0, 8, 8, 8, 8, 8, 8, 1, 100, 1\n"
        "1, 8, 8, 8, 8, 8, 8, 1, 100, , 2\n"
        "2, 8, 8, 8, 8, 8, 8, 1, 100, 3, , 4\n"
    )
    assert generate_csv(db.finish(), tmp_path) == expected


def test_merged_csv_has_every_column_in_every_row(tmp_path: Path):
    db = late_column_builder(tmp_path, merge=True)
    fill_late_columns(db)
    expected = (
        f"{DEFAULT_HEADER}, kA\n"
        "0, 8, 8, 8, 8, 8, 8, 1, 100, 1, , \n"
        "1, 8, 8, 8, 8, 8, 8, 1, 100, , 2, \n"
        "2, 8, 8, 8, 8, 8, 8, 1, 100, 3, , 4\n"
    )
    assert generate_csv(db.finish(), tmp_path) == expected


# ---------------------------------------------------------------------------
# 10. read-only.
# ---------------------------------------------------------------------------


def test_reading_leaves_the_database_unchanged(tmp_path: Path):
    path = make_default_database(tmp_path)
    before = path.read_bytes()
    writeResultsCsv(path, tmp_path / "out.csv")
    summarize(path)
    assert path.read_bytes() == before
    # Opening a WAL database read-only may leave its -wal and -shm files behind.
    assert sorted(p.name for p in tmp_path.iterdir() if not p.name.startswith("bench.db-")) == [
        "bench.db",
        "out.csv",
    ]


# ---------------------------------------------------------------------------
# 11. stored values that cannot be read.
# ---------------------------------------------------------------------------


def make_bad_status_database(tmp_path: Path) -> Path:
    db = two_solution_builder(tmp_path)
    db.add_problem(0, [8, 8], [8, 8, 8, 8], 1)
    # The schema CHECK forbids the status, as it would for a corrupted or newer database.
    db.connection.execute("PRAGMA ignore_check_constraints = ON")
    db.set_run(0, 0, 99, "FAILED")
    db.set_run(0, 1, PASS, "PASSED", pass_values("2"))
    return db.finish()


def test_unknown_status_is_reported_by_pair_from_row():
    # getPairs filters on valid statuses in SQL, so only the conversion itself can see this value.
    with closing(sqlite3.connect(":memory:")) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            "SELECT 4 AS problemIndex, 2 AS libraryIndex, 'kA' AS solutionName, 99 AS status, "
            "NULL AS sessionId, NULL AS validation, NULL AS startTime, NULL AS causedByProblemIndex"
        ).fetchone()
    with pytest.raises(
        BenchmarkDatabaseError, match="unknown run status 99 in problem 4, solution 2 kA"
    ):
        Pair.fromRow(row)


def test_unknown_status_is_reported_by_count_pairs_by_status(tmp_path: Path):
    with closing(openDatabase(make_bad_status_database(tmp_path))) as connection:
        with pytest.raises(BenchmarkDatabaseError, match="unknown run status 99 in the pair"):
            countPairsByStatus(connection)


def test_unknown_status_is_reported_by_summarize(tmp_path: Path):
    with pytest.raises(BenchmarkDatabaseError, match="unknown run status 99"):
        summarize(make_bad_status_database(tmp_path))


def test_unknown_status_is_reported_by_the_csv_statuses(tmp_path: Path):
    with pytest.raises(
        BenchmarkDatabaseError, match="unknown run status 99 in problem 0, solution 0"
    ):
        generate_csv(make_bad_status_database(tmp_path), tmp_path)


@pytest.mark.parametrize("status", [INDIRECTLY_FAILED, INDIRECT_VERIFICATION_FAILURE])
def test_indirect_status_without_a_cause_is_reported(tmp_path: Path, status: RunStatus):
    db = two_solution_builder(tmp_path)
    db.add_problem(0, [8, 8])
    db.connection.execute("PRAGMA ignore_check_constraints = ON")
    db.set_run(0, 0, status, status.name)
    db.set_run(0, 1, PASS, "PASSED", pass_values("2"))
    path = db.finish()
    with closing(openDatabase(path)) as connection:
        with pytest.raises(BenchmarkDatabaseError, match=f"{status.name} pair without a causing"):
            getPairs(connection, RunStatus)
    with pytest.raises(BenchmarkDatabaseError, match="without a causing problem in problem 0"):
        summarize(path)


@pytest.mark.parametrize(
    "key", ["time-us", "gbps-bandwidth", "gflops", "solution-index", "tiles-per-cu", "total-gran"]
)
def test_non_numeric_run_value_is_reported(tmp_path: Path, key: str):
    db = two_solution_builder(tmp_path, extra_cols=True)
    db.add_problem(0, [8, 8], [8, 8, 8, 8], 1)
    values = pass_values("1", tiles_per_cu="1", total_gran="1", solution_index="0")
    values[key] = "abc"
    db.set_run(0, 0, PASS, "PASSED", values)
    db.set_run(0, 1, PASS, "PASSED", pass_values("2"))
    expected = f"value 'abc' of {key} is not a number in problem 0, solution kA"
    with pytest.raises(BenchmarkDatabaseError, match=expected):
        generate_csv(db.finish(), tmp_path)


def test_non_numeric_cu_efficiency_value_is_reported(tmp_path: Path):
    db = two_solution_builder(tmp_path, metric="CUEfficiency")
    db.add_problem(0, [8, 8], [8, 8, 8, 8], 1)
    db.set_run(0, 0, PASS, "PASSED", pass_values("1", gflops_per_cu="x"))
    expected = "value 'x' of gflops-per-cu is not a number in problem 0, solution kA"
    with pytest.raises(BenchmarkDatabaseError, match=expected):
        generate_csv(db.finish(), tmp_path)


@pytest.mark.parametrize("key", ["csv_export_extra_cols", "csv_merge_same_problems"])
def test_invalid_flag_meta_is_reported(tmp_path: Path, key: str):
    db = two_solution_builder(tmp_path)
    db.add_problem(0, [8, 8])
    db.set_meta(key, "yes")
    with pytest.raises(BenchmarkDatabaseError, match=f"meta {key} 'yes'"):
        generate_csv(db.finish(), tmp_path)


def test_other_performance_metrics_are_benchmarked_as_device_efficiency(tmp_path: Path):
    # The client stores any --performance-metric name, and only CUEfficiency changes the CSV.
    db = two_solution_builder(tmp_path, metric="ExperimentalMLP")
    db.add_problem(0, [8, 8], [8, 8, 8, 8], 1)
    db.set_run(0, 0, PASS, "PASSED", pass_values("1"))
    assert generate_csv(db.finish(), tmp_path).startswith("GFlops, SizeI")


# ---------------------------------------------------------------------------
# 12. atomic CSV.
# ---------------------------------------------------------------------------


def make_failing_second_problem_database(tmp_path: Path) -> Path:
    db = two_solution_builder(tmp_path)
    db.add_problem(0, [8, 8], [8, 8, 8, 8], 1)
    db.add_problem(1, [4, 4], [4, 4, 4, 4], 1)
    db.set_run(0, 0, PASS, "PASSED", pass_values("1"))
    db.set_run(0, 1, PASS, "PASSED", pass_values("2"))
    db.set_run(1, 0, PASS, "PASSED", pass_values("3", time_us="abc"))
    db.set_run(1, 1, PASS, "PASSED", pass_values("4"))
    return db.finish()


def make_csv_dir(tmp_path: Path) -> Path:
    csv_dir = tmp_path / "csv"
    csv_dir.mkdir()
    return csv_dir


def test_failure_while_writing_keeps_the_existing_csv(tmp_path: Path):
    db_path = make_failing_second_problem_database(tmp_path)
    csv_path = make_csv_dir(tmp_path) / "out.csv"
    csv_path.write_bytes(b"previous results\n")
    with pytest.raises(BenchmarkDatabaseError, match="of time-us is not a number"):
        writeResultsCsv(db_path, csv_path)
    assert csv_path.read_bytes() == b"previous results\n"
    assert [p.name for p in csv_path.parent.iterdir()] == ["out.csv"]


def test_failure_while_writing_without_a_csv_leaves_no_file(tmp_path: Path):
    db_path = make_failing_second_problem_database(tmp_path)
    csv_dir = make_csv_dir(tmp_path)
    with pytest.raises(BenchmarkDatabaseError):
        writeResultsCsv(db_path, csv_dir / "out.csv")
    assert list(csv_dir.iterdir()) == []


def test_failure_of_the_generator_after_the_first_row_removes_the_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db_path = make_default_database(tmp_path)
    csv_dir = make_csv_dir(tmp_path)
    original = BenchmarkDatabase._loadRunValues
    calls = []

    def failing(*args):
        calls.append(args)
        if len(calls) > 1:
            raise RuntimeError("injected")
        return original(*args)

    monkeypatch.setattr(BenchmarkDatabase, "_loadRunValues", failing)
    with pytest.raises(RuntimeError, match="injected"):
        writeResultsCsv(db_path, csv_dir / "out.csv")
    assert list(csv_dir.iterdir()) == []


def test_success_leaves_only_the_csv(tmp_path: Path):
    csv_dir = make_csv_dir(tmp_path)
    writeResultsCsv(make_default_database(tmp_path), csv_dir / "out.csv")
    assert [p.name for p in csv_dir.iterdir()] == ["out.csv"]


# ---------------------------------------------------------------------------
# 13. read snapshot.
# ---------------------------------------------------------------------------


def commit_change(db_path: Path, statement: str) -> None:
    with closing(sqlite3.connect(db_path)) as writer:
        writer.execute(statement)
        writer.commit()


def read_gflops(connection: sqlite3.Connection) -> str:
    return connection.execute("SELECT value FROM run_values WHERE key = 'gflops'").fetchone()[0]


def test_read_snapshot_ignores_commits_made_meanwhile(tmp_path: Path):
    db_path = make_default_database(tmp_path)
    update = "UPDATE run_values SET value = '1' WHERE key = 'gflops'"
    with closing(openDatabase(db_path)) as connection:
        with _readSnapshot(connection):
            before = read_gflops(connection)
            commit_change(db_path, update)
            assert read_gflops(connection) == before
        assert read_gflops(connection) == "1"
        assert not connection.in_transaction


def wrap_after_first_call(monkeypatch: pytest.MonkeyPatch, name: str, action) -> None:
    original = getattr(BenchmarkDatabase, name)
    done = []

    def wrapper(*args):
        result = original(*args)
        if not done:
            done.append(True)
            action()
        return result

    monkeypatch.setattr(BenchmarkDatabase, name, wrapper)


def test_csv_is_one_snapshot_even_if_the_client_commits_midway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db_path = make_default_database(tmp_path)
    expected = generate_csv(db_path, tmp_path)
    change = (
        "UPDATE run_values SET value = '9' WHERE key = 'gflops' AND problem_id = 2; "
        "UPDATE runs SET status = 3 WHERE problem_id = 2"
    )

    def write():
        with closing(sqlite3.connect(db_path)) as writer:
            writer.executescript(change)

    wrap_after_first_call(monkeypatch, "_loadStatuses", write)
    assert generate_csv(db_path, tmp_path) == expected
    monkeypatch.undo()
    assert generate_csv(db_path, tmp_path) != expected


def test_summary_is_one_snapshot_even_if_the_client_commits_midway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db_path = make_default_database(tmp_path)
    expected = summarize(db_path)
    wrap_after_first_call(
        monkeypatch,
        "getSessions",
        lambda: commit_change(db_path, "UPDATE runs SET status = 3 WHERE status = 2"),
    )
    assert summarize(db_path) == expected
    monkeypatch.undo()
    assert summarize(db_path).counts[FAILURE] > expected.counts[FAILURE]


# ---------------------------------------------------------------------------
# 14. schema taken from the client source.
# ---------------------------------------------------------------------------


def test_reader_expects_the_schema_version_of_the_client():
    assert schema_version() == schemaVersion


def test_extracted_schema_is_accepted_by_the_reader(tmp_path: Path):
    path = tmp_path / "schema.db"
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(schema_sql())
        connection.execute(f"PRAGMA user_version = {schema_version()}")
        connection.commit()
    with closing(openDatabase(path)) as connection:
        assert getPairs(connection, RunStatus) == []


def test_schema_extraction_from_a_source_text():
    source = 'static constexpr int32_t schemaVersion = 7;\nschemaSql = R"sql(\nCREATE TABLE t(a);\n)sql";'
    assert extract_schema_version(source) == 7
    assert extract_schema_sql(source) == "\nCREATE TABLE t(a);\n"


@pytest.mark.parametrize("extract", [extract_schema_sql, extract_schema_version])
def test_schema_extraction_fails_loudly_without_the_markers(extract):
    with pytest.raises(RuntimeError, match="BenchmarkDatabase.cpp has no"):
        extract("int main() {}")
