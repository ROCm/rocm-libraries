# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Reader of the SQLite benchmark database written by tensilelite-client (--benchmark-db).

The table layout is the contract with ``schemaSql`` in client/src/BenchmarkDatabase.cpp and must
be kept in sync with it: any incompatible schema change there must bump ``schemaVersion`` here.

Only the public indices are meaningful to callers: ``init_configs.problem_index`` and
``solutions.library_index``. ``runs.problem_id``, ``runs.solution_id`` and
``runs.caused_by_problem_id`` reference the table ids.

INDIRECTLY_FAILED and INDIRECT_VERIFICATION_FAILURE pairs were never launched: the client marked them
because their solution crashed or failed verification on the problem ``causedByProblemIndex``.
NOT_SELECTED pairs were never launched either: the solution iterator did not select them. They are
neither failures nor passes, and are only counted.

A stored value that cannot be converted (unknown status, non-numeric run value, unknown meta value)
is reported as a BenchmarkDatabaseError that names where it was found.
"""

from __future__ import annotations

import enum
import math
import os
import sqlite3
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple, Union

schemaVersion = 1

_requiredTables = (
    "meta",
    "solutions",
    "client_configs",
    "init_configs",
    "problem_values",
    "sessions",
    "runs",
    "run_values",
    "iterations",
)

_runValueKeys = (
    "solution-index",
    "time-us",
    "gflops",
    "gflops-per-cu",
    "gbps-bandwidth",
    "tiles-per-cu",
    "total-gran",
)

_failedCell = "-1"

PathLike = Union[str, Path]


class BenchmarkDatabaseError(Exception):
    pass


class RunStatus(enum.IntEnum):
    NOT_BENCHMARKED = 0
    UNKNOWN = 1
    PASS = 2
    FAILURE = 3
    INDIRECTLY_FAILED = 4
    INDIRECT_VERIFICATION_FAILURE = 5
    NOT_SELECTED = 6


_indirectStatuses = (RunStatus.INDIRECTLY_FAILED, RunStatus.INDIRECT_VERIFICATION_FAILURE)

# The statuses summarize lists pair by pair; PASS and NOT_SELECTED pairs are only counted.
_listedStatuses = (
    RunStatus.UNKNOWN,
    RunStatus.FAILURE,
    RunStatus.INDIRECTLY_FAILED,
    RunStatus.INDIRECT_VERIFICATION_FAILURE,
)


def _toStatus(value: Any, where: str) -> RunStatus:
    try:
        return RunStatus(value)
    except ValueError:
        raise BenchmarkDatabaseError(f"unknown run status {value} in {where}") from None


def _toNumber(text: Any, key: str, where: str) -> float:
    try:
        return float(text)
    except (ValueError, TypeError):
        raise BenchmarkDatabaseError(
            f"value '{text}' of {key} is not a number in {where}"
        ) from None


@dataclass(frozen=True)
class Pair:
    problemIndex: int
    libraryIndex: int
    solutionName: str
    status: RunStatus
    sessionId: Optional[int]
    validation: str
    startTime: Optional[float]
    # Public index of the problem whose failure made the client skip this pair, if any.
    causedByProblemIndex: Optional[int] = None

    @classmethod
    def fromRow(cls, row: sqlite3.Row) -> Pair:
        where = (
            f"problem {row['problemIndex']}, solution {row['libraryIndex']} {row['solutionName']}"
        )
        status = _toStatus(row["status"], where)
        cause = row["causedByProblemIndex"]
        # The schema forbids it, but a stored value that breaks the rule must not crash a report.
        if cause is None and status in _indirectStatuses:
            raise BenchmarkDatabaseError(f"{status.name} pair without a causing problem in {where}")
        return cls(
            row["problemIndex"],
            row["libraryIndex"],
            row["solutionName"],
            status,
            row["sessionId"],
            row["validation"] or "",
            row["startTime"],
            cause,
        )


@dataclass(frozen=True)
class Session:
    id: int
    pid: int
    startTime: float
    endTime: Optional[float]

    @property
    def died(self) -> bool:
        return self.endTime is None


@dataclass(frozen=True)
class BenchmarkSummary:
    counts: Dict[RunStatus, int]
    unknownPairs: List[Pair]
    failurePairs: List[Pair]
    indirectlyFailedPairs: List[Pair]
    indirectVerificationFailurePairs: List[Pair]
    taintedPairs: List[Pair]
    deadSessions: List[Session]

    @property
    def notBenchmarkedCount(self) -> int:
        return self.counts[RunStatus.NOT_BENCHMARKED]


# ---------------------------------------------------------------------------
# Opening and queries.
# ---------------------------------------------------------------------------


def openDatabase(dbPath: PathLike, busyTimeoutSeconds: float = 5.0) -> sqlite3.Connection:
    """Open the database read-only and check its schema; callers use contextlib.closing."""
    path = Path(dbPath)
    if not path.is_file():
        raise BenchmarkDatabaseError(f"benchmark database '{path}' does not exist")
    # Autocommit, so that _readSnapshot alone decides when a transaction starts and ends.
    connection = sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro",
        uri=True,
        isolation_level=None,
        timeout=busyTimeoutSeconds,
    )
    try:
        _checkSchema(connection, path)
    except BenchmarkDatabaseError:
        connection.close()
        raise
    except sqlite3.Error as error:
        connection.close()
        raise BenchmarkDatabaseError(f"'{path}' is not a benchmark database: {error}") from error
    return connection


@contextmanager
def _readSnapshot(connection: sqlite3.Connection) -> Iterator[None]:
    """Read everything inside from one consistent view, even if the client commits meanwhile."""
    # The transaction is deferred: the WAL snapshot is taken by the first SELECT.
    connection.execute("BEGIN")
    try:
        yield
    finally:
        if connection.in_transaction:
            connection.execute("ROLLBACK")


def _checkSchema(connection: sqlite3.Connection, path: Path) -> None:
    rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    tables = {row[0] for row in rows}
    missing = [table for table in _requiredTables if table not in tables]
    if missing:
        raise BenchmarkDatabaseError(
            f"'{path}' is not a benchmark database: missing tables {', '.join(missing)}"
        )
    found = connection.execute("PRAGMA user_version").fetchone()[0]
    if found != schemaVersion:
        raise BenchmarkDatabaseError(
            f"'{path}' has schema version {found}, expected {schemaVersion}"
        )


_pairSql = """
SELECT c.problem_index AS problemIndex, s.library_index AS libraryIndex, s.name AS solutionName,
       r.status AS status, r.session_id AS sessionId, r.validation AS validation,
       r.start_time AS startTime, cc.problem_index AS causedByProblemIndex
FROM runs r
JOIN init_configs c ON c.id = r.problem_id
JOIN solutions s ON s.id = r.solution_id
LEFT JOIN init_configs cc ON cc.id = r.caused_by_problem_id
"""


def _readPairs(
    connection: sqlite3.Connection, query: str, parameters: Tuple[Union[int, float], ...]
) -> List[Pair]:
    cursor = connection.cursor()
    cursor.row_factory = sqlite3.Row
    return [Pair.fromRow(row) for row in cursor.execute(query, parameters)]


def getPairs(connection: sqlite3.Connection, statuses: Iterable[RunStatus]) -> List[Pair]:
    codes = tuple(int(status) for status in statuses)
    placeholders = ", ".join("?" * len(codes))
    query = (
        _pairSql + f"WHERE r.status IN ({placeholders}) ORDER BY c.problem_index, s.library_index"
    )
    return _readPairs(connection, query, codes)


def getStuckPairs(
    connection: sqlite3.Connection, afterSessionId: int, startedBefore: float
) -> List[Pair]:
    """Pairs of newer sessions left UNKNOWN since before startedBefore.

    Such a pair is either still running or its process died; the caller tells them apart.
    """
    query = (
        _pairSql
        + "WHERE r.status = ? AND r.end_time IS NULL AND r.session_id > ? AND r.start_time < ? "
        + "ORDER BY c.problem_index, s.library_index"
    )
    parameters = (int(RunStatus.UNKNOWN), afterSessionId, startedBefore)
    return _readPairs(connection, query, parameters)


def getSessions(connection: sqlite3.Connection) -> List[Session]:
    rows = connection.execute("SELECT id, pid, start_time, end_time FROM sessions ORDER BY id")
    return [Session(*row) for row in rows]


def countPairsByStatus(connection: sqlite3.Connection) -> Dict[RunStatus, int]:
    counts = {status: 0 for status in RunStatus}
    for status, count in connection.execute("SELECT status, COUNT(*) FROM runs GROUP BY status"):
        counts[_toStatus(status, "the pair counts")] = count
    return counts


def getLatestPairSessionId(connection: sqlite3.Connection) -> int:
    """The newest session that began or marked a pair, 0 if none did."""
    return connection.execute("SELECT COALESCE(MAX(session_id), 0) FROM runs").fetchone()[0]


# ---------------------------------------------------------------------------
# Summary.
# ---------------------------------------------------------------------------


def _sessionsAfterDead(sessions: List[Session]) -> Set[int]:
    return {current.id for previous, current in zip(sessions, sessions[1:]) if previous.died}


def summarize(dbPath: PathLike) -> BenchmarkSummary:
    with closing(openDatabase(dbPath)) as connection, _readSnapshot(connection):
        sessions = getSessions(connection)
        counts = countPairsByStatus(connection)
        pairs: Dict[RunStatus, List[Pair]] = {status: [] for status in _listedStatuses}
        for pair in getPairs(connection, _listedStatuses):
            pairs[pair.status].append(pair)
    afterDead = _sessionsAfterDead(sessions)
    failures = pairs[RunStatus.FAILURE]
    return BenchmarkSummary(
        counts=counts,
        unknownPairs=pairs[RunStatus.UNKNOWN],
        failurePairs=failures,
        indirectlyFailedPairs=pairs[RunStatus.INDIRECTLY_FAILED],
        indirectVerificationFailurePairs=pairs[RunStatus.INDIRECT_VERIFICATION_FAILURE],
        # A fault in a dead session may have corrupted the next one; recorded, never retried.
        taintedPairs=[pair for pair in failures if pair.sessionId in afterDead],
        deadSessions=[session for session in sessions if session.died],
    )


def _pairLine(pair: Pair, emptyValidation: str = "") -> str:
    line = f"  problem {pair.problemIndex}, solution {pair.libraryIndex} {pair.solutionName}"
    validation = pair.validation or emptyValidation
    return f"{line}: {validation}" if validation else line


def _indirectLines(pairs: List[Pair]) -> List[str]:
    """One line per solution and causing problem, so a faulty kernel does not print every pair."""
    groups: Dict[Tuple[int, str, Optional[int]], int] = {}
    for pair in pairs:
        key = (pair.libraryIndex, pair.solutionName, pair.causedByProblemIndex)
        groups[key] = groups.get(key, 0) + 1
    ordered = sorted(groups.items(), key=lambda item: (item[0][0], item[0][2]))
    return [
        f"  solution {libraryIndex} {name}: {count} pairs, caused by problem {cause}"
        for (libraryIndex, name, cause), count in ordered
    ]


def _section(title: str, lines: List[str]) -> List[str]:
    return [title] + lines if lines else []


def formatSummary(summary: BenchmarkSummary) -> str:
    counts = summary.counts
    lines = [
        f"benchmark database: {sum(counts.values())} pairs: "
        f"{counts[RunStatus.PASS]} PASS, {counts[RunStatus.FAILURE]} FAILURE, "
        f"{counts[RunStatus.UNKNOWN]} UNKNOWN, {summary.notBenchmarkedCount} NOT_BENCHMARKED, "
        f"{counts[RunStatus.INDIRECTLY_FAILED]} INDIRECTLY_FAILED, "
        f"{counts[RunStatus.INDIRECT_VERIFICATION_FAILURE]} INDIRECT_VERIFICATION_FAILURE, "
        f"{counts[RunStatus.NOT_SELECTED]} NOT_SELECTED"
    ]
    lines += _section(
        "pairs left UNKNOWN (the client died while running them):",
        [_pairLine(pair) for pair in summary.unknownPairs],
    )
    lines += _section(
        "pairs that FAILED:",
        [_pairLine(pair, "no verdict") for pair in summary.failurePairs],
    )
    lines += _section(
        "pairs not launched because their solution crashed, faulted or hung on another problem:",
        _indirectLines(summary.indirectlyFailedPairs),
    )
    lines += _section(
        "pairs not launched because their solution failed verification on another problem:",
        _indirectLines(summary.indirectVerificationFailurePairs),
    )
    lines += _section(
        "pairs that failed right after a session that died "
        "(possibly caused by an earlier GPU fault, not retried):",
        [_pairLine(pair, "no verdict") for pair in summary.taintedPairs],
    )
    if summary.deadSessions:
        lines.append(f"sessions that died: {len(summary.deadSessions)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Results CSV.
# ---------------------------------------------------------------------------


def formatDouble(value: float) -> str:
    """Format like C++ ``std::ostream << double`` with the default precision of 6."""
    if math.isnan(value):
        return "-nan" if math.copysign(1.0, value) < 0 else "nan"
    return "%g" % value


def _escape(value: str) -> str:
    if '"' in value:
        return '"' + value.replace('"', '""') + '"'
    if any(char in value for char in ",\n\r"):
        return '"' + value + '"'
    return value


@dataclass(frozen=True)
class _CsvOptions:
    metricKey: str
    metricHeader: str
    extraCols: bool
    mergeSameProblems: bool


def _readMeta(connection: sqlite3.Connection, path: Path, key: str) -> str:
    row = connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    if row is None:
        raise BenchmarkDatabaseError(f"'{path}' has no meta key {key}")
    return row[0]


def _readFlag(connection: sqlite3.Connection, path: Path, key: str) -> bool:
    value = _readMeta(connection, path, key)
    if value not in ("true", "false"):
        raise BenchmarkDatabaseError(
            f"'{path}' has meta {key} '{value}', expected 'true' or 'false'"
        )
    return value == "true"


def _readCsvOptions(connection: sqlite3.Connection, path: Path) -> _CsvOptions:
    # The client stores whatever --performance-metric names, and anything but CUEfficiency is
    # benchmarked as DeviceEfficiency, so the metric is deliberately not validated.
    metric = _readMeta(connection, path, "performance_metric")
    extraCols = _readFlag(connection, path, "csv_export_extra_cols")
    merge = _readFlag(connection, path, "csv_merge_same_problems")
    if metric == "CUEfficiency":
        return _CsvOptions("gflops-per-cu", "GFlopsPerCU", extraCols, merge)
    return _CsvOptions("gflops", "GFlops", extraCols, merge)


@dataclass
class _ProblemBest:
    """Best-of state of one problem, mirroring the members of the client's ResultFileReporter."""

    gflops: float = -1.0
    timeUs: float = -1.0
    bandwidth: float = -1.0
    tilesPerCu: float = -1.0
    totalGranularity: float = -1.0
    winnerIndex: float = -1.0
    winnerName: str = ""


@dataclass(frozen=True)
class _Solution:
    id: int
    libraryIndex: int
    name: str


def _loadSolutions(connection: sqlite3.Connection) -> List[_Solution]:
    rows = connection.execute(
        "SELECT id, library_index, name FROM solutions ORDER BY library_index"
    )
    return [_Solution(*row) for row in rows]


def _loadProblems(connection: sqlite3.Connection) -> List[Tuple[int, int]]:
    # A problem without values was never reported, so the client wrote no row for it.
    rows = connection.execute(
        "SELECT id, problem_index FROM init_configs "
        "WHERE EXISTS (SELECT 1 FROM problem_values WHERE problem_id = init_configs.id) "
        "ORDER BY problem_index"
    )
    return list(rows)


def _loadProblemValues(connection: sqlite3.Connection, problemId: int) -> Dict[str, str]:
    rows = connection.execute(
        "SELECT key, value FROM problem_values WHERE problem_id = ?", (problemId,)
    )
    return dict(rows)


def _loadStatuses(
    connection: sqlite3.Connection, problemId: int, problemIndex: int
) -> Dict[int, RunStatus]:
    rows = connection.execute(
        "SELECT r.solution_id, s.library_index, r.status FROM runs r "
        "JOIN solutions s ON s.id = r.solution_id WHERE r.problem_id = ?",
        (problemId,),
    )
    return {
        solutionId: _toStatus(status, f"problem {problemIndex}, solution {libraryIndex}")
        for solutionId, libraryIndex, status in rows
    }


def _loadRunValues(connection: sqlite3.Connection, problemId: int) -> Dict[int, Dict[str, str]]:
    placeholders = ", ".join("?" * len(_runValueKeys))
    rows = connection.execute(
        f"SELECT solution_id, key, value FROM run_values "
        f"WHERE problem_id = ? AND key IN ({placeholders})",
        (problemId, *_runValueKeys),
    )
    values: Dict[int, Dict[str, str]] = {}
    for solutionId, key, value in rows:
        values.setdefault(solutionId, {})[key] = value
    return values


def _updateBest(
    best: _ProblemBest,
    lastRow: Dict[str, str],
    solution: _Solution,
    values: Dict[str, str],
    metricKey: str,
    where: str,
) -> None:
    """Fold one PASS pair into the best-of state, like ResultFileReporter::reportValue."""
    if "time-us" in values:
        timeUs = _toNumber(values["time-us"], "time-us", where)
        if not math.isnan(timeUs) and (best.timeUs < 0 or best.timeUs > timeUs):
            best.timeUs = timeUs
            best.tilesPerCu = _toNumber(lastRow.get("tiles-per-cu", -1.0), "tiles-per-cu", where)
            best.totalGranularity = _toNumber(lastRow.get("total-gran", -1.0), "total-gran", where)
    if "gbps-bandwidth" in values:
        bandwidth = _toNumber(values["gbps-bandwidth"], "gbps-bandwidth", where)
        if not math.isnan(bandwidth) and bandwidth > best.bandwidth:
            best.bandwidth = bandwidth
    if metricKey in values:
        gflops = _toNumber(values[metricKey], metricKey, where)
        if best.gflops < gflops:
            best.gflops = gflops
            best.winnerName = solution.name
            winnerIndex = values.get("solution-index", solution.libraryIndex)
            best.winnerIndex = _toNumber(winnerIndex, "solution-index", where)


def _solveProblem(
    solutions: List[_Solution],
    statuses: Dict[int, RunStatus],
    runValues: Dict[int, Dict[str, str]],
    metricKey: str,
    problemIndex: int,
) -> Tuple[_ProblemBest, Dict[str, str]]:
    """Return the best-of state and the per-solution cell of one problem, keyed by name.

    The client writes -1 for a failed validation, and for a pair skipped because the database
    already holds its result (including UNKNOWN and the indirect statuses). A pair the iterator did
    not select is never visited: it has no cell and creates no column. A pair that was never
    reached has an empty cell in the client CSV; it is written as -1 here, like a failed one.
    The cells are in library index order, which is the visiting order of a run without prediction.
    """
    best = _ProblemBest()
    cells: Dict[str, str] = {}
    lastRow: Dict[str, str] = {}
    for solution in solutions:
        status = statuses.get(solution.id, RunStatus.NOT_BENCHMARKED)
        if status == RunStatus.NOT_SELECTED:
            continue
        values = runValues.get(solution.id, {})
        # The client keeps these in the current row across solutions, whatever their verdict.
        for key in ("tiles-per-cu", "total-gran"):
            if key in values:
                lastRow[key] = values[key]
        if status != RunStatus.PASS:
            cells[solution.name] = _failedCell
            continue
        where = f"problem {problemIndex}, solution {solution.name}"
        _updateBest(best, lastRow, solution, values, metricKey, where)
        cells[solution.name] = values.get(metricKey, "")
    return best, cells


def _buildColumns(options: _CsvOptions, sizeKeys: List[str]) -> List[Tuple[str, str]]:
    columns = [("problem-index", options.metricHeader)]
    columns += [(key, key) for key in sizeKeys]
    columns += [
        ("ldd", "LDD"),
        ("ldc", "LDC"),
        ("lda", "LDA"),
        ("ldb", "LDB"),
        ("total-flops", "TotalFlops"),
        ("gbps-bandwidth", "GbpsBW"),
    ]
    if options.extraCols:
        columns += [
            ("tiles-per-cu", "TilesPerCu"),
            ("total-gran", "TotalGranularity"),
            ("fastest-gflops", "WinnerGFlops"),
            ("time-us", "WinnerTimeUS"),
            ("solution-winner-idx", "WinnerIdx"),
            ("solution-winner", "WinnerName"),
        ]
    return columns


def _buildRow(
    problemIndex: int,
    problemValues: Dict[str, str],
    best: _ProblemBest,
    cells: Dict[str, str],
    extraCols: bool,
) -> Dict[str, str]:
    row = dict(problemValues)
    row["problem-index"] = str(problemIndex)
    row["gbps-bandwidth"] = formatDouble(best.bandwidth)
    if extraCols:
        row["tiles-per-cu"] = formatDouble(best.tilesPerCu)
        row["total-gran"] = formatDouble(best.totalGranularity)
        row["fastest-gflops"] = formatDouble(best.gflops)
        row["time-us"] = formatDouble(best.timeUs)
        row["solution-winner-idx"] = formatDouble(best.winnerIndex)
        row["solution-winner"] = best.winnerName
    row.update(cells)
    return row


def _csvLine(fields: List[str]) -> str:
    return ", ".join(fields) + "\n"


@dataclass(frozen=True)
class _CsvRow:
    values: Dict[str, str]
    # How many of the solution columns, in first-visit order, the client knows after this problem.
    solutionColumns: int


def _csvRows(
    connection: sqlite3.Connection, options: _CsvOptions, solutions: List[_Solution]
) -> Tuple[List[_CsvRow], List[str]]:
    """Return the rows and the names of every solution column in first-visit order."""
    rows: List[_CsvRow] = []
    # Duplicate names share one column, like the client's map keyed by name.
    names: Dict[str, None] = {}
    for problemId, problemIndex in _loadProblems(connection):
        best, cells = _solveProblem(
            solutions,
            _loadStatuses(connection, problemId, problemIndex),
            _loadRunValues(connection, problemId),
            options.metricKey,
            problemIndex,
        )
        names.update(dict.fromkeys(cells))
        values = _buildRow(
            problemIndex, _loadProblemValues(connection, problemId), best, cells, options.extraCols
        )
        rows.append(_CsvRow(values, len(names)))
    return rows, list(names)


def _csvLines(connection: sqlite3.Connection, path: Path) -> Iterator[str]:
    """Replay the client's CSVStackFile.

    A solution gets a column the first time it is visited. The header is written with the columns
    known after the first problem, and without merging every row with the columns known after its
    problem, so a solution first visited later has a cell without a header, and the earlier rows are
    shorter. With merging, the rows are written at the end with all the columns.

    The columns that one problem adds are in library index order. A run with prediction visits the
    solutions in the order of its ranking, which the database does not store.
    """
    options = _readCsvOptions(connection, path)
    rows, names = _csvRows(connection, options, _loadSolutions(connection))
    if not rows:
        return
    # The size columns come from the first problem; its values are read again with its row.
    sizeKeys = sorted(key for key in rows[0].values if key.startswith("Size"))
    fixed = _buildColumns(options, sizeKeys)
    keys = [key for key, _ in fixed]
    headers = [header for _, header in fixed]
    yield _csvLine([_escape(header) for header in headers + names[: rows[0].solutionColumns]])
    for row in rows:
        count = len(names) if options.mergeSameProblems else row.solutionColumns
        fields = [_escape(row.values.get(key, "")) for key in keys + names[:count]]
        if options.mergeSameProblems:
            # The client's CSVStackFile::readCurrentRow escapes merged rows and writeRow escapes
            # them again; the output must stay byte-identical to the client.
            fields = [_escape(field) for field in fields]
        yield _csvLine(fields)


def writeResultsCsv(dbPath: PathLike, csvPath: PathLike) -> None:
    """Regenerate the results CSV that the client's ResultFileReporter writes.

    The file is written aside and renamed, so a failure leaves any earlier CSV untouched.
    """
    path = Path(dbPath)
    csvPath = Path(csvPath)
    tempPath = csvPath.with_name(f".{csvPath.name}.{os.getpid()}.tmp")
    try:
        with closing(openDatabase(path)) as connection, _readSnapshot(connection):
            with open(tempPath, "w", encoding="utf-8", newline="\n") as csvFile:
                csvFile.writelines(_csvLines(connection, path))
        os.replace(tempPath, csvPath)
    except BaseException:
        tempPath.unlink(missing_ok=True)
        raise
