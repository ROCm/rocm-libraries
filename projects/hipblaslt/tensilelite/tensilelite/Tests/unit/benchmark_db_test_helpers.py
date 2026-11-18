# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Synthetic benchmark databases shared by the unit tests.

Must not import tensilelite: fake_benchmark_client.py runs as a bare subprocess and imports it.
"""

from __future__ import annotations

import functools
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

CLIENT_SOURCE = Path(__file__).resolve().parents[3] / "client" / "src" / "BenchmarkDatabase.cpp"

_SCHEMA_SQL_PATTERN = re.compile(r'schemaSql\s*=\s*R"sql\((.*?)\)sql"', re.DOTALL)
_SCHEMA_VERSION_PATTERN = re.compile(r"static constexpr int32_t schemaVersion\s*=\s*(\d+)\s*;")


def _extract(pattern: re.Pattern[str], source_text: str, what: str) -> str:
    match = pattern.search(source_text)
    if match is None:
        raise RuntimeError(f"{CLIENT_SOURCE} has no {what}; update benchmark_db_test_helpers.py")
    return match.group(1)


def extract_schema_sql(source_text: str) -> str:
    """The schema is read from the client source, so there is no second copy to keep in sync."""
    return _extract(_SCHEMA_SQL_PATTERN, source_text, 'raw string schemaSql = R"sql(...)sql"')


def extract_schema_version(source_text: str) -> int:
    return int(
        _extract(
            _SCHEMA_VERSION_PATTERN,
            source_text,
            "line 'static constexpr int32_t schemaVersion = N;'",
        )
    )


@functools.lru_cache(maxsize=None)
def schema_sql() -> str:
    return extract_schema_sql(CLIENT_SOURCE.read_text())


@functools.lru_cache(maxsize=None)
def schema_version() -> int:
    return extract_schema_version(CLIENT_SOURCE.read_text())


# The record kinds of the canonical database dump that the C++ BenchmarkCsvParity test writes, with
# their number of fields after the kind. See tests/BenchmarkCsvParity_test.cpp for the format.
DUMP_FIELD_COUNTS = {
    "meta": 2,
    "solution": 2,
    "problem": 1,
    "problem_value": 3,
    "run": 5,
    "run_value": 4,
}

# The names of RunStatus in BenchmarkDatabase.py, which this module must not import.
DUMP_STATUSES = {
    "NOT_BENCHMARKED": 0,
    "UNKNOWN": 1,
    "PASS": 2,
    "FAILURE": 3,
    "INDIRECTLY_FAILED": 4,
    "INDIRECT_VERIFICATION_FAILURE": 5,
    "NOT_SELECTED": 6,
}


def parse_dump(text: str) -> List[Tuple[str, List[str]]]:
    """Split a canonical dump into (kind, fields) records, rejecting anything malformed."""
    if text and not text.endswith("\n"):
        raise ValueError("dump does not end with a newline")
    records = []
    for number, line in enumerate(text.split("\n")[:-1], start=1):
        kind, *fields = line.split("\t")
        if kind not in DUMP_FIELD_COUNTS:
            raise ValueError(f"dump line {number}: unknown record kind '{kind}'")
        if len(fields) != DUMP_FIELD_COUNTS[kind]:
            raise ValueError(
                f"dump line {number}: {kind} has {len(fields)} fields, "
                f"expected {DUMP_FIELD_COUNTS[kind]}"
            )
        if kind == "run" and fields[2] not in DUMP_STATUSES:
            raise ValueError(f"dump line {number}: unknown run status '{fields[2]}'")
        records.append((kind, fields))
    return records


class DatabaseBuilder:
    """Fills a synthetic database; table ids are the public indices plus one."""

    def __init__(
        self,
        path: Path,
        metric: str = "DeviceEfficiency",
        extra_cols: bool = False,
        merge: bool = False,
    ):
        self.path = path
        self.connection = sqlite3.connect(path)
        # Like the real database, which readers open in WAL mode.
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.executescript(schema_sql())
        self.connection.execute(f"PRAGMA user_version = {schema_version()}")
        self.connection.execute("INSERT INTO client_configs(id, ini) VALUES(1, '')")
        self.set_meta("performance_metric", metric)
        self.set_meta("csv_export_extra_cols", "true" if extra_cols else "false")
        self.set_meta("csv_merge_same_problems", "true" if merge else "false")

    def set_meta(self, key: str, value: str) -> None:
        self.connection.execute("INSERT OR REPLACE INTO meta VALUES(?, ?)", (key, value))

    def add_solution(self, library_index: int, name: str) -> None:
        self.connection.execute(
            "INSERT INTO solutions(id, library_index, name, spec, spec_format) "
            "VALUES(?, ?, ?, x'00', 'yaml')",
            (library_index + 1, library_index, name),
        )

    def add_session(self, session_id: int, end_time: Optional[float]) -> None:
        self.connection.execute(
            "INSERT INTO sessions VALUES(?, ?, ?, ?, 0)",
            (session_id, 100 + session_id, float(session_id), end_time),
        )

    def add_problem(
        self,
        index: int,
        sizes: Optional[List[int]] = None,
        ld_values: Optional[List[int]] = None,
        total_flops: int = 0,
    ) -> None:
        """Without sizes the problem has no problem_values, like a never reported problem."""
        self.connection.execute(
            "INSERT INTO init_configs(id, config_id, problem_index) VALUES(?, 1, ?)",
            (index + 1, index),
        )
        if sizes is None:
            return
        values = {"problem-index": str(index), "total-flops": str(total_flops)}
        values.update({f"Size{chr(ord('I') + i)}": str(size) for i, size in enumerate(sizes)})
        lds = ld_values or [sizes[0]] * 4
        values.update(zip(("ldd", "ldc", "lda", "ldb"), map(str, lds)))
        self.connection.executemany(
            "INSERT INTO problem_values VALUES(?, ?, ?)",
            [(index + 1, key, value) for key, value in values.items()],
        )

    def add_problem_value(self, problem: int, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO problem_values VALUES(?, ?, ?)", (problem + 1, key, value)
        )

    def add_run_value(self, problem: int, solution: int, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO run_values VALUES(?, ?, ?, ?)", (problem + 1, solution + 1, key, value)
        )

    def set_run(
        self,
        problem: int,
        solution: int,
        status: int,
        validation: Optional[str] = None,
        values: Optional[Dict[str, str]] = None,
        session: Optional[int] = None,
        caused_by: Optional[int] = None,
    ) -> None:
        caused_by_id = None if caused_by is None else caused_by + 1
        self.connection.execute(
            "INSERT INTO runs VALUES(?, ?, ?, ?, 1.0, 2.0, ?, ?)",
            (problem + 1, solution + 1, int(status), session, validation, caused_by_id),
        )
        self.connection.executemany(
            "INSERT INTO run_values VALUES(?, ?, ?, ?)",
            [(problem + 1, solution + 1, key, value) for key, value in (values or {}).items()],
        )

    def load_dump(self, text: str) -> None:
        """Fill the database from a canonical dump; an empty validation or cause is NULL."""
        for kind, fields in parse_dump(text):
            if kind == "meta":
                self.set_meta(*fields)
            elif kind == "solution":
                self.add_solution(int(fields[0]), fields[1])
            elif kind == "problem":
                self.add_problem(int(fields[0]))
            elif kind == "problem_value":
                self.add_problem_value(int(fields[0]), fields[1], fields[2])
            elif kind == "run":
                problem, solution, status, validation, caused_by = fields
                self.set_run(
                    int(problem),
                    int(solution),
                    DUMP_STATUSES[status],
                    validation or None,
                    caused_by=int(caused_by) if caused_by else None,
                )
            else:
                self.add_run_value(int(fields[0]), int(fields[1]), fields[2], fields[3])

    def finish(self) -> Path:
        self.connection.commit()
        self.connection.close()
        return self.path
