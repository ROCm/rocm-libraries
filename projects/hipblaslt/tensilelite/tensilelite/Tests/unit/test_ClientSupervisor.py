# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Supervision of the benchmark client against a fake client that speaks the database protocol."""

from __future__ import annotations

import contextlib
import json
import os
import pickle
import shlex
import signal
import sqlite3
import stat
import subprocess
import sys
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

pytestmark = pytest.mark.unit

from tensilelite import ClientSupervisor
from tensilelite.BenchmarkDatabase import RunStatus, getPairs, openDatabase
from tensilelite.ClientSupervisor import (
    PROBE_NAME,
    _DbState,
    _exitedAbnormally,
    _judge,
    _killGroup,
    _queryState,
    _Launch,
    _terminateOnSignals,
    _Verdict,
    GpuProbe,
    SupervisionOutcome,
    SupervisionResult,
    SupervisorTerminated,
    Watchdog,
    WarmUpState,
    findGpuProbe,
    removeDatabase,
    reportDatabaseResults,
    superviseClient,
)

from benchmark_db_test_helpers import DatabaseBuilder

FAKE_CLIENT = Path(__file__).with_name("fake_benchmark_client.py")
FAKE_PROBE = Path(__file__).with_name("fake_gpu_probe.py")
DRIVER = Path(__file__).with_name("supervisor_driver.py")
TEST_POLL_SECONDS = 0.1
GRACE_SECONDS = 1.0
linux_only = pytest.mark.skipif(sys.platform != "linux", reason="needs Linux process semantics")

PASS = RunStatus.PASS
FAILURE = RunStatus.FAILURE
UNKNOWN = RunStatus.UNKNOWN
NOT_BENCHMARKED = RunStatus.NOT_BENCHMARKED
INDIRECTLY_FAILED = RunStatus.INDIRECTLY_FAILED
INDIRECT_VERIFICATION_FAILURE = RunStatus.INDIRECT_VERIFICATION_FAILURE
NOT_SELECTED = RunStatus.NOT_SELECTED

COMPLETED = SupervisionOutcome.COMPLETED
STOPPED_AFTER_SESSION = SupervisionOutcome.STOPPED_AFTER_SESSION
STOPPED_BEFORE_SESSION = SupervisionOutcome.STOPPED_BEFORE_SESSION


@pytest.fixture(autouse=True)
def reset_warm_up_state(monkeypatch, tmp_path):
    # The real default build directory may hold a probe that must never run in unit tests.
    default_client = tmp_path / "default_build" / "tensilelite-client"
    monkeypatch.setattr(ClientSupervisor, "DEFAULT_CLIENT_PATH", default_client)
    monkeypatch.setattr(ClientSupervisor, "_warmUpState", WarmUpState())


class Fixture:
    """One fake client config: its ini, database, results CSV and launch log."""

    def __init__(self, directory: Path, name: str = "bench", **plan):
        self.directory = directory
        self.db = directory / f"{name}.db"
        self.csv = directory / f"{name}.csv"
        self.log = directory / f"{name}.log"
        self.ini = directory / f"{name}.ini"
        plan["launch_log"] = str(self.log)
        self.ini.write_text(
            f"results-file={self.csv}\n"
            f"benchmark-db={self.db}\n"
            f"fake-plan={json.dumps(plan)}\n"
        )

    @property
    def command(self) -> List[str]:
        return [sys.executable, str(FAKE_CLIENT), "--config-file", str(self.ini)]

    def supervise(
        self, timeout: float = 0, stop_on_clean_exit: bool = False, probe: Optional[GpuProbe] = None
    ) -> SupervisionResult:
        return superviseClient(
            self.command,
            self.directory,
            self.db,
            Watchdog(timeout, TEST_POLL_SECONDS, GRACE_SECONDS),
            stop_on_clean_exit,
            probe=probe,
        )

    def fake_probe(self, *exit_codes: int, device: int = 0) -> GpuProbe:
        """A probe that logs into the launch log, so events() shows its order against launches."""
        command = [sys.executable, str(FAKE_PROBE), str(self.log), *map(str, exit_codes)]
        return GpuProbe(command, device)

    def events(self) -> List[str]:
        return ["probe" if launch.get("probe") else "client" for launch in self.launches()]

    def statuses(self) -> Dict[int, RunStatus]:
        with closing(openDatabase(self.db)) as connection:
            return {pair.libraryIndex: pair.status for pair in getPairs(connection, RunStatus)}

    def pair_statuses(self) -> Dict[Tuple[int, int], RunStatus]:
        """The status of every pair, keyed by problem index and solution index."""
        with closing(openDatabase(self.db)) as connection:
            return {
                (pair.problemIndex, pair.libraryIndex): pair.status
                for pair in getPairs(connection, RunStatus)
            }

    def launches(self) -> List[dict]:
        if not self.log.exists():
            return []
        return [json.loads(line) for line in self.log.read_text().splitlines()]


def all_pass(count: int) -> Dict[int, RunStatus]:
    return {index: PASS for index in range(count)}


# ---------------------------------------------------------------------------
# superviseClient.
# ---------------------------------------------------------------------------


def test_normal_completion(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=3)
    result = fixture.supervise()
    assert result == SupervisionResult(0, COMPLETED)
    assert fixture.statuses() == all_pass(3)
    assert len(fixture.launches()) == 1


def test_crash_in_the_middle_is_resumed(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=4, behaviors={"2": "segv"})
    result = fixture.supervise()
    assert result == SupervisionResult(1, COMPLETED)
    assert fixture.statuses() == {0: PASS, 1: PASS, 2: UNKNOWN, 3: PASS}
    assert len(fixture.launches()) == 2


def test_two_crashing_pairs_need_two_restarts(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=5, behaviors={"1": "segv", "3": "exit:9"})
    result = fixture.supervise()
    assert result == SupervisionResult(2, COMPLETED)
    assert fixture.statuses() == {0: PASS, 1: UNKNOWN, 2: PASS, 3: UNKNOWN, 4: PASS}
    assert len(fixture.launches()) == 3


def test_crash_before_the_database_exists_is_not_retried(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=3, early="segv")
    result = fixture.supervise()
    assert result.restarts == 0
    assert result.outcome == STOPPED_BEFORE_SESSION
    assert "before starting a benchmark session" in result.message
    assert "killed by signal 11" in result.message
    assert str(fixture.db) in result.message
    assert len(fixture.launches()) == 1


def test_failure_without_progress_on_an_existing_database_is_not_retried(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=2, early="exit:1")
    builder = DatabaseBuilder(fixture.db)
    builder.add_problem(0, [8, 8])
    builder.add_solution(0, "k0")
    builder.add_solution(1, "k1")
    builder.set_run(0, 0, PASS, "PASSED")
    builder.set_run(0, 1, NOT_BENCHMARKED)
    builder.finish()
    result = fixture.supervise()
    assert result.restarts == 0
    assert result.outcome == STOPPED_BEFORE_SESSION
    assert "exit code 1" in result.message
    assert "before starting a benchmark session" in result.message
    assert len(fixture.launches()) == 1


def test_failure_after_a_session_without_progress_is_not_retried(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=3, post_session="exit:3")
    result = fixture.supervise()
    assert result.restarts == 0
    assert result.outcome == STOPPED_AFTER_SESSION
    assert "exit code 3" in result.message
    assert "without benchmarking any pair" in result.message
    assert len(fixture.launches()) == 1


def make_database_with_indirect_pair(fixture: Fixture, status: RunStatus, pending: bool) -> None:
    """Solution 1 was marked by session 1, which died; solution 2 is pending if asked."""
    builder = DatabaseBuilder(fixture.db)
    builder.add_problem(0, [8, 8])
    for index in range(3):
        builder.add_solution(index, f"k{index}")
    builder.add_session(1, None)
    builder.set_run(0, 0, PASS, "PASSED", session=1)
    builder.set_run(0, 1, status, status.name, session=1, caused_by=0)
    builder.set_run(0, 2, NOT_BENCHMARKED if pending else PASS, None if pending else "PASSED")
    builder.finish()


@pytest.mark.parametrize("status", [INDIRECTLY_FAILED, INDIRECT_VERIFICATION_FAILURE])
def test_indirect_pairs_do_not_prevent_completion(tmp_path: Path, status):
    fixture = Fixture(tmp_path, pairs=3)
    make_database_with_indirect_pair(fixture, status, pending=False)
    assert fixture.supervise() == SupervisionResult(0, COMPLETED)
    assert len(fixture.launches()) == 1
    assert fixture.statuses()[1] == status


def test_pairs_reset_by_a_disabled_indirect_option_count_as_progress(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=4, reset_marked_pairs=True, behaviors={"2": "segv"})
    builder = DatabaseBuilder(fixture.db)
    builder.add_problem(0, [8, 8])
    for index in range(4):
        builder.add_solution(index, f"k{index}")
    builder.add_session(1, None)
    builder.set_run(0, 0, PASS, "PASSED", session=1)
    builder.set_run(0, 1, UNKNOWN, session=1)
    for index in (2, 3):
        builder.set_run(0, index, INDIRECTLY_FAILED, "INDIRECTLY_FAILED", session=1, caused_by=0)
    builder.finish()
    result = fixture.supervise()
    assert result == SupervisionResult(1, COMPLETED), result
    assert fixture.statuses() == {0: PASS, 1: UNKNOWN, 2: UNKNOWN, 3: PASS}
    assert len(fixture.launches()) == 2


def make_complete_database(fixture: Fixture) -> None:
    builder = DatabaseBuilder(fixture.db)
    builder.add_problem(0, [8, 8])
    builder.add_solution(0, "k0")
    builder.add_solution(1, "k1")
    builder.add_session(1, 2.0)
    builder.set_run(0, 0, PASS, "PASSED", session=1)
    builder.set_run(0, 1, PASS, "PASSED", session=1)
    builder.finish()


def make_pending_database(fixture: Fixture) -> None:
    builder = DatabaseBuilder(fixture.db)
    builder.add_problem(0, [8, 8])
    builder.add_solution(0, "k0")
    builder.add_solution(1, "k1")
    builder.set_run(0, 0, NOT_BENCHMARKED)
    builder.set_run(0, 1, NOT_BENCHMARKED)
    builder.finish()


def test_rejected_complete_database_is_an_error(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=2, early="exit:1")
    make_complete_database(fixture)
    result = fixture.supervise()
    assert result.restarts == 0
    assert result.outcome == STOPPED_BEFORE_SESSION
    assert "delete it to start over" in result.message
    assert str(fixture.db) in result.message
    assert len(fixture.launches()) == 1


def test_complete_database_with_a_normal_run_is_done(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=2)
    make_complete_database(fixture)
    assert fixture.supervise() == SupervisionResult(0, COMPLETED)
    assert len(fixture.launches()) == 1


# A prediction run: the iterator selects a subset of the solutions of every problem, and the
# problem marks the others NOT_SELECTED when it finishes.
PREDICTION_PLAN = {"pairs": 3, "problems": 2, "not_selected": {"0": [2], "1": [0]}}


def test_not_selected_pairs_do_not_prevent_completion(tmp_path: Path):
    fixture = Fixture(tmp_path, **PREDICTION_PLAN)
    assert fixture.supervise() == SupervisionResult(0, COMPLETED)
    assert len(fixture.launches()) == 1
    assert fixture.pair_statuses() == {
        (0, 0): PASS,
        (0, 1): PASS,
        (0, 2): NOT_SELECTED,
        (1, 0): NOT_SELECTED,
        (1, 1): PASS,
        (1, 2): PASS,
    }


def test_clean_exit_with_pairs_left_stops_even_if_pairs_were_not_selected(tmp_path: Path):
    # The fake client only runs problem 0, so problem 1 keeps its NOT_BENCHMARKED pairs.
    fixture = Fixture(tmp_path, pairs=2)
    builder = DatabaseBuilder(fixture.db)
    builder.add_problem(0, [8, 8])
    builder.add_problem(1, [8, 8])
    for index in range(2):
        builder.add_solution(index, f"k{index}")
    builder.add_session(1, 2.0)
    builder.set_run(0, 0, PASS, "PASSED", session=1)
    builder.set_run(0, 1, NOT_SELECTED, session=1)
    builder.set_run(1, 0, NOT_BENCHMARKED)
    builder.set_run(1, 1, NOT_BENCHMARKED)
    builder.finish()
    result = fixture.supervise()
    assert result.restarts == 0
    assert result.outcome == STOPPED_AFTER_SESSION
    assert "exited normally but 2 pairs" in result.message
    assert fixture.pair_statuses()[(0, 1)] == NOT_SELECTED


def test_crash_after_a_problem_was_marked_resumes_without_stopping(tmp_path: Path):
    plan = dict(PREDICTION_PLAN, behaviors={"1:1": "segv"})
    fixture = Fixture(tmp_path, **plan)
    result = fixture.supervise()
    assert result == SupervisionResult(1, COMPLETED), result
    assert len(fixture.launches()) == 2
    statuses = fixture.pair_statuses()
    # Problem 0 was marked before the crash and stays so; the crashed pair is left UNKNOWN.
    assert statuses[(0, 2)] == NOT_SELECTED
    assert statuses[(1, 1)] == UNKNOWN
    # The resumed run finishes problem 1 and marks the pair that its selection still leaves out.
    assert statuses[(1, 0)] == NOT_SELECTED
    assert statuses[(1, 2)] == PASS


def test_marking_not_selected_pairs_counts_as_progress(tmp_path: Path):
    builder = DatabaseBuilder(tmp_path / "bench.db")
    builder.add_problem(0, [8, 8])
    builder.add_problem(1, [8, 8])
    builder.add_solution(0, "k0")
    builder.add_solution(1, "k1")
    builder.add_session(1, 2.0)
    builder.set_run(0, 0, PASS, "PASSED", session=1)
    builder.set_run(0, 1, PASS, "PASSED", session=1)
    builder.set_run(1, 0, NOT_BENCHMARKED)
    builder.set_run(1, 1, NOT_BENCHMARKED)
    path = builder.finish()
    before = _queryState(path)
    # A new session only marks a pair of problem 1 as not selected, then dies.
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(
            "INSERT INTO sessions(id, pid, start_time, init_seed) VALUES(2, 1, 3.0, 0)"
        )
        connection.execute(
            "UPDATE runs SET status = ?, session_id = 2 WHERE problem_id = 2 AND solution_id = 1",
            (int(NOT_SELECTED),),
        )
        connection.commit()
    after = _queryState(path)
    verdict, message = _judge(before, after, _Launch(-11), False, path)
    assert verdict == _Verdict.RESTART, message


def test_hang_is_killed_by_the_watchdog(tmp_path: Path, leftover_pids):
    fixture = Fixture(tmp_path, pairs=3, behaviors={"1": "hang"})
    start = time.monotonic()
    result = fixture.supervise(timeout=1)
    leftover_pids.append(fixture.launches()[0]["pid"])
    assert time.monotonic() - start < 15
    assert result == SupervisionResult(1, COMPLETED)
    assert fixture.statuses() == {0: PASS, 1: UNKNOWN, 2: PASS}
    hung_pid = fixture.launches()[0]["pid"]
    with pytest.raises(ProcessLookupError):
        os.kill(hung_pid, 0)


def test_timeout_zero_disables_the_watchdog(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=3, behaviors={"1": "sleep:1.5"})
    assert fixture.supervise(timeout=0) == SupervisionResult(0, COMPLETED)
    assert fixture.statuses() == all_pass(3)


def test_validation_failure_is_not_a_crash(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=3, behaviors={"1": "fail"})
    result = fixture.supervise()
    assert result == SupervisionResult(0, COMPLETED)
    assert fixture.statuses() == {0: PASS, 1: FAILURE, 2: PASS}
    assert len(fixture.launches()) == 1


def test_deliberate_early_stop_is_not_resumed(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=4, behaviors={"1": "fail"}, stop_after_fail=True)
    result = fixture.supervise(stop_on_clean_exit=True)
    assert result.restarts == 0
    assert result.outcome == STOPPED_AFTER_SESSION
    assert "stopped early" in result.message
    assert "2 pairs not benchmarked" in result.message
    assert len(fixture.launches()) == 1


def test_early_stop_is_resumed_when_not_requested(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=4, behaviors={"1": "fail"}, stop_after_fail=True)
    result = fixture.supervise(stop_on_clean_exit=False)
    assert result == SupervisionResult(1, COMPLETED)
    assert fixture.statuses() == {0: PASS, 1: FAILURE, 2: PASS, 3: PASS}


def test_a_client_failing_every_pair_costs_one_launch_per_pair(tmp_path: Path):
    behaviors = {str(index): "exit:1" for index in range(3)}
    fixture = Fixture(tmp_path, pairs=3, behaviors=behaviors)
    assert fixture.supervise() == SupervisionResult(2, COMPLETED)
    assert len(fixture.launches()) == 3
    assert set(fixture.statuses().values()) == {UNKNOWN}


def test_a_client_failing_right_after_the_session_opens_is_launched_once(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=3, post_session="exit:1")
    result = fixture.supervise()
    assert result.restarts == 0
    assert result.outcome == STOPPED_AFTER_SESSION
    assert "without benchmarking any pair" in result.message
    assert len(fixture.launches()) == 1


def test_stale_unknown_pair_of_an_earlier_session_does_not_trigger_the_watchdog(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=2, behaviors={"1": "sleep:1.0"})
    builder = DatabaseBuilder(fixture.db)
    builder.add_problem(0, [8, 8])
    builder.add_solution(0, "k0")
    builder.add_solution(1, "k1")
    builder.add_session(1, None)
    # Started long ago and never ended: it would be killed if the session floor did not exclude it.
    builder.set_run(0, 0, UNKNOWN, session=1)
    builder.connection.execute("UPDATE runs SET end_time = NULL")
    builder.set_run(0, 1, NOT_BENCHMARKED)
    builder.finish()
    assert fixture.supervise(timeout=30) == SupervisionResult(0, COMPLETED)
    assert len(fixture.launches()) == 1
    assert fixture.statuses() == {0: UNKNOWN, 1: PASS}


# ---------------------------------------------------------------------------
# The verdict.
# ---------------------------------------------------------------------------


def state(session: int, died: bool, total: int, left: int, latest: int) -> _DbState:
    return _DbState(session, died, total, left, latest)


FIRST_SESSION = state(1, False, 3, 3, 0)
ALL_DONE = state(2, True, 3, 0, 2)

JUDGED = [
    pytest.param(
        _DbState(),
        _DbState(),
        _Launch(None, 4242),
        False,
        _Verdict.STOP,
        ["client process 4242 did not exit after SIGKILL"],
        id="unreaped_without_a_session",
    ),
    pytest.param(
        _DbState(),
        state(2, False, 3, 0, 2),
        _Launch(None, 4242),
        False,
        _Verdict.STOP,
        ["client process 4242 did not exit after SIGKILL"],
        id="unreaped_with_everything_benchmarked",
    ),
    pytest.param(
        state(1, False, 3, 1, 1),
        state(1, False, 3, 1, 1),
        _Launch(1),
        False,
        _Verdict.STOP,
        ["exit code 1", "before starting a benchmark session"],
        id="failure_without_a_new_session",
    ),
    pytest.param(
        state(1, False, 2, 0, 1),
        state(1, False, 2, 0, 1),
        _Launch(1),
        False,
        _Verdict.STOP,
        ["exit code 1", "before starting a benchmark session"],
        id="failure_without_a_new_session_on_a_complete_database",
    ),
    pytest.param(
        FIRST_SESSION,
        state(2, False, 3, 0, 2),
        _Launch(0),
        False,
        _Verdict.DONE,
        [],
        id="clean_exit_complete",
    ),
    pytest.param(
        _DbState(),
        state(1, False, 0, 0, 0),
        _Launch(0),
        False,
        _Verdict.DONE,
        [],
        id="clean_exit_without_pairs",
    ),
    pytest.param(
        FIRST_SESSION,
        state(2, False, 4, 2, 2),
        _Launch(0),
        False,
        _Verdict.STOP,
        ["exited normally but 2 pairs"],
        id="clean_exit_with_pairs_left",
    ),
    pytest.param(
        FIRST_SESSION,
        ALL_DONE,
        _Launch(-11),
        False,
        _Verdict.DONE,
        [],
        id="crash_with_every_pair_benchmarked",
    ),
    pytest.param(
        FIRST_SESSION,
        state(2, True, 3, 2, 2),
        _Launch(None),
        False,
        _Verdict.RESTART,
        ["killed by the watchdog", "2 pairs left"],
        id="watchdog_kill_with_progress",
    ),
    pytest.param(
        FIRST_SESSION,
        ALL_DONE,
        _Launch(None),
        False,
        _Verdict.DONE,
        [],
        id="watchdog_kill_of_the_last_pair",
    ),
    pytest.param(
        FIRST_SESSION,
        state(2, True, 3, 2, 2),
        _Launch(-11),
        False,
        _Verdict.RESTART,
        ["killed by signal 11 (SIGSEGV)"],
        id="wrapper_signalled",
    ),
    pytest.param(
        FIRST_SESSION,
        state(2, True, 3, 2, 2),
        _Launch(139),
        False,
        _Verdict.RESTART,
        ["killed by signal 11 (SIGSEGV)"],
        id="client_segfault_seen_by_the_wrapper",
    ),
    pytest.param(
        FIRST_SESSION,
        state(2, False, 3, 1, 2),
        _Launch(139),
        False,
        _Verdict.RESTART,
        ["exit code 139"],
        id="validation_failure_count_is_not_a_signal",
    ),
    pytest.param(
        FIRST_SESSION,
        state(2, True, 3, 2, 2),
        _Launch(255),
        False,
        _Verdict.RESTART,
        ["exit code 255"],
        id="exit_code_without_a_signal_name",
    ),
    pytest.param(
        state(1, False, 3, 3, 0),
        state(2, True, 3, 3, 0),
        _Launch(-11),
        False,
        _Verdict.STOP,
        ["without benchmarking any pair"],
        id="crash_with_a_new_session_but_no_pair",
    ),
    pytest.param(
        state(1, False, 4, 0, 1),
        state(2, True, 4, 1, 2),
        _Launch(139),
        False,
        _Verdict.RESTART,
        ["restarting to resume, 1 pairs left"],
        id="pairs_reset_by_a_disabled_indirect_option",
    ),
    pytest.param(
        FIRST_SESSION,
        state(2, False, 4, 2, 2),
        _Launch(1),
        True,
        _Verdict.STOP,
        ["stopped early", "2 pairs not benchmarked"],
        id="deliberate_early_stop",
    ),
    pytest.param(
        FIRST_SESSION,
        state(2, False, 4, 2, 2),
        _Launch(1),
        False,
        _Verdict.RESTART,
        ["exit code 1"],
        id="early_stop_that_is_resumed",
    ),
    pytest.param(
        FIRST_SESSION,
        state(2, False, 3, 0, 2),
        _Launch(3),
        False,
        _Verdict.DONE,
        [],
        id="done_with_validation_failures",
    ),
]


@pytest.mark.parametrize("before, after, launch, stopOnCleanExit, verdict, texts", JUDGED)
def test_judge(before, after, launch, stopOnCleanExit, verdict, texts):
    actual, message = _judge(before, after, launch, stopOnCleanExit, "x.db")
    assert actual == verdict
    for text in texts:
        assert text in message
    if verdict == _Verdict.STOP and "client failed" in message:
        assert message.count("delete it to start over") == 1
        assert "x.db" in message
    if verdict != _Verdict.STOP:
        assert "delete it to start over" not in message


# ---------------------------------------------------------------------------
# An unreadable database.
# ---------------------------------------------------------------------------


@pytest.fixture
def fast_state_reads(monkeypatch):
    monkeypatch.setattr(ClientSupervisor, "STATE_BUSY_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(ClientSupervisor, "STATE_READ_RETRY_SECONDS", 0.01)


@contextlib.contextmanager
def exclusive_lock(db: Path):
    """Holds the database locked against readers, which needs the rollback journal mode."""
    connection = sqlite3.connect(db, isolation_level=None)
    try:
        connection.execute("PRAGMA journal_mode = DELETE")
        connection.execute("BEGIN EXCLUSIVE")
        yield
    finally:
        connection.close()


def test_a_locked_database_stops_before_the_launch(tmp_path: Path, fast_state_reads):
    fixture = Fixture(tmp_path, pairs=2)
    make_complete_database(fixture)
    with exclusive_lock(fixture.db):
        result = fixture.supervise()
    assert result.outcome == STOPPED_BEFORE_SESSION
    assert "could not read the benchmark database" in result.message
    assert fixture.launches() == []


def test_a_database_locked_after_the_launch_is_not_judged_as_done(
    tmp_path: Path, monkeypatch, fast_state_reads
):
    fixture = Fixture(tmp_path, pairs=2)
    locks = contextlib.ExitStack()
    original = ClientSupervisor._runLaunch

    def launch_then_lock(*args, **kwargs):
        launch = original(*args, **kwargs)
        locks.enter_context(exclusive_lock(fixture.db))
        return launch

    monkeypatch.setattr(ClientSupervisor, "_runLaunch", launch_then_lock)
    with locks:
        result = fixture.supervise(probe=fixture.fake_probe())
    monkeypatch.setattr(ClientSupervisor, "_runLaunch", original)
    assert result.outcome != COMPLETED
    assert "could not read the benchmark database" in result.message
    assert fixture.statuses() == all_pass(2)
    # The exit could not be judged, so the next client waits for a warm-up.
    second = Fixture(tmp_path, name="second", pairs=2)
    assert second.supervise(probe=second.fake_probe()) == SupervisionResult(0, COMPLETED)
    assert second.events() == ["probe", "client"]


def test_a_transient_read_error_is_retried(tmp_path: Path, monkeypatch, fast_state_reads):
    fixture = Fixture(tmp_path, pairs=2)
    make_pending_database(fixture)
    original = ClientSupervisor.openDatabase
    failures = []

    def flaky_open(*args, **kwargs):
        if len(failures) < 2:
            failures.append(1)
            raise sqlite3.OperationalError("database is locked")
        return original(*args, **kwargs)

    monkeypatch.setattr(ClientSupervisor, "openDatabase", flaky_open)
    assert fixture.supervise() == SupervisionResult(0, COMPLETED)
    assert len(failures) == 2
    assert len(fixture.launches()) == 1


def test_a_malformed_database_stops_before_the_launch(tmp_path: Path, fast_state_reads):
    fixture = Fixture(tmp_path, pairs=2)
    write_garbage_database(fixture.db)
    result = fixture.supervise()
    assert result.outcome == STOPPED_BEFORE_SESSION
    assert "could not read the benchmark database" in result.message
    assert fixture.launches() == []


def test_a_zero_size_database_is_the_empty_state(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=2)
    fixture.db.touch()
    assert fixture.supervise() == SupervisionResult(0, COMPLETED)
    assert fixture.statuses() == all_pass(2)


def test_a_database_without_tables_is_the_empty_state(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=2)
    with closing(sqlite3.connect(fixture.db)) as connection:
        connection.execute("PRAGMA user_version = 7")
    assert fixture.db.stat().st_size > 0
    assert fixture.supervise() == SupervisionResult(0, COMPLETED)
    assert fixture.statuses() == all_pass(2)


def test_a_database_of_another_schema_version_stops_before_the_launch(
    tmp_path: Path, fast_state_reads
):
    fixture = Fixture(tmp_path, pairs=2)
    make_complete_database(fixture)
    with closing(sqlite3.connect(fixture.db)) as connection:
        connection.execute("PRAGMA user_version = 99")
    result = fixture.supervise()
    assert result.outcome == STOPPED_BEFORE_SESSION
    assert "could not read the benchmark database" in result.message
    assert fixture.launches() == []


# ---------------------------------------------------------------------------
# reportDatabaseResults.
# ---------------------------------------------------------------------------


def test_report_of_a_clean_database(tmp_path: Path, capsys):
    fixture = Fixture(tmp_path, pairs=3)
    assert fixture.supervise() == SupervisionResult(0, COMPLETED)
    assert reportDatabaseResults(fixture.db, fixture.csv) == 0
    lines = fixture.csv.read_text().splitlines()
    assert lines[0].endswith("k0, k1, k2")
    assert lines[1].split(", ")[-3:] == ["101", "102", "103"]
    assert "3 PASS, 0 FAILURE, 0 UNKNOWN, 0 NOT_BENCHMARKED, " in capsys.readouterr().out


def test_report_of_not_selected_pairs_is_clean(tmp_path: Path, capsys):
    fixture = Fixture(tmp_path, **PREDICTION_PLAN)
    assert fixture.supervise() == SupervisionResult(0, COMPLETED)
    assert reportDatabaseResults(fixture.db, fixture.csv) == 0
    lines = fixture.csv.read_text().splitlines()
    # Problem 0 visits k0 and k1, so the header lacks k2, which problem 1 visits; problem 1 did not
    # select k0, whose cell is empty.
    assert lines[0].endswith("k0, k1")
    assert lines[1].split(", ")[-2:] == ["101", "102"]
    assert lines[2].split(", ")[-3:] == ["", "102", "103"]
    output = capsys.readouterr().out
    assert "4 PASS, 0 FAILURE, 0 UNKNOWN, 0 NOT_BENCHMARKED, " in output
    assert output.rstrip().endswith("2 NOT_SELECTED")


def test_report_with_a_crashed_pair(tmp_path: Path, capsys):
    fixture = Fixture(tmp_path, pairs=3, behaviors={"1": "segv"})
    assert fixture.supervise() == SupervisionResult(1, COMPLETED)
    capsys.readouterr()
    assert reportDatabaseResults(fixture.db, fixture.csv) == 1
    assert fixture.csv.read_text().splitlines()[1].split(", ")[-3:] == ["101", "-1", "103"]
    output = capsys.readouterr().out
    assert "1 UNKNOWN" in output
    assert "problem 0, solution 1 k1" in output


@pytest.mark.parametrize("status", [INDIRECTLY_FAILED, INDIRECT_VERIFICATION_FAILURE])
def test_report_with_an_indirect_pair_is_not_clean(tmp_path: Path, capsys, status):
    fixture = Fixture(tmp_path, pairs=3)
    make_database_with_indirect_pair(fixture, status, pending=False)
    assert reportDatabaseResults(fixture.db, fixture.csv) == 1
    assert f"1 {status.name}" in capsys.readouterr().out


def test_report_of_a_complete_database_with_a_failure_is_not_clean(tmp_path: Path, capsys):
    fixture = Fixture(tmp_path, pairs=3, behaviors={"1": "fail"})
    fixture.supervise()
    assert set(fixture.statuses().values()) == {PASS, FAILURE}
    assert reportDatabaseResults(fixture.db, fixture.csv) == 1
    first_csv = fixture.csv.read_bytes()
    assert first_csv
    # A rerun must not change the verdict nor the results.
    fixture.csv.unlink()
    assert reportDatabaseResults(fixture.db, fixture.csv) == 1
    assert fixture.csv.read_bytes() == first_csv
    assert "1 FAILURE" in capsys.readouterr().out


def test_report_without_a_database_removes_the_stale_csv(tmp_path: Path, capsys):
    csv_path = tmp_path / "bench.csv"
    csv_path.write_text("stale\n")
    assert reportDatabaseResults(tmp_path / "missing.db", csv_path) == 1
    assert "WARNING" in capsys.readouterr().out
    assert not csv_path.exists()


# ---------------------------------------------------------------------------
# removeDatabase.
# ---------------------------------------------------------------------------


def write_garbage_database(db: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        Path(str(db) + suffix).write_bytes(b"not a database")


def test_remove_database_removes_the_log_and_shared_memory_files(tmp_path: Path):
    db = tmp_path / "results.db"
    write_garbage_database(db)
    removeDatabase(db)
    assert list(tmp_path.iterdir()) == []


def test_remove_database_accepts_missing_files(tmp_path: Path):
    db = tmp_path / "results.db"
    (tmp_path / "results.db-wal").write_text("x")
    removeDatabase(str(db))
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# GPU warm-up after an abnormal client exit.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("behavior", ["segv", "exit:134"])
def test_probe_runs_between_a_crash_and_the_restart(tmp_path: Path, behavior):
    fixture = Fixture(tmp_path, pairs=3, behaviors={"1": behavior})
    result = fixture.supervise(probe=fixture.fake_probe())
    assert result == SupervisionResult(1, COMPLETED)
    assert fixture.events() == ["client", "probe", "client"]


def test_probe_runs_after_a_watchdog_kill(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=3, behaviors={"1": "hang"})
    result = fixture.supervise(timeout=2, probe=fixture.fake_probe())
    assert result == SupervisionResult(1, COMPLETED)
    assert fixture.events() == ["client", "probe", "client"]


def test_probe_does_not_run_after_a_clean_exit(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=3)
    fixture.supervise(probe=fixture.fake_probe())
    assert fixture.events() == ["client"]


def test_probe_does_not_run_after_validation_failures_of_a_clean_session(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=4, behaviors={"1": "fail"}, stop_after_fail=True)
    result = fixture.supervise(probe=fixture.fake_probe())
    assert result.restarts == 1
    assert fixture.events() == ["client", "client"]


def test_probe_is_not_needed_without_a_probe(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=3, behaviors={"1": "segv"})
    assert fixture.supervise() == SupervisionResult(1, COMPLETED)
    assert fixture.events() == ["client", "client"]


def test_pending_warm_up_carries_over_to_the_next_supervise_call(tmp_path: Path):
    first = Fixture(tmp_path, name="first", pairs=2, behaviors={"1": "segv"})
    assert first.supervise(probe=first.fake_probe()).restarts == 0
    assert first.events() == ["client"]
    second = Fixture(tmp_path, name="second", pairs=2)
    # Both configs log into their own file, so the second probe is checked in the second log.
    assert second.supervise(probe=second.fake_probe()) == SupervisionResult(0, COMPLETED)
    assert second.events() == ["probe", "client"]


def test_warm_up_only_covers_the_device_of_the_probe(tmp_path: Path):
    ClientSupervisor._warmUpState.request(1)
    fixture = Fixture(tmp_path, pairs=2)
    fixture.supervise(probe=fixture.fake_probe(device=0))
    assert fixture.events() == ["client"]
    # The request of the other device is still pending.
    other = Fixture(tmp_path, name="other", pairs=2)
    other.supervise(probe=other.fake_probe(device=1))
    assert other.events() == ["probe", "client"]


def test_failed_probe_attempts_are_retried(tmp_path: Path, capsys):
    fixture = Fixture(tmp_path, pairs=3, behaviors={"1": "segv"})
    result = fixture.supervise(probe=fixture.fake_probe(1, 1, 0))
    assert result == SupervisionResult(1, COMPLETED)
    assert fixture.events() == ["client", "probe", "probe", "probe", "client"]
    output = capsys.readouterr()
    assert "gpu warm-up on device 0, attempt 1/3 failed (exit code 1)" in output.err + output.out
    assert "# gpu warm-up on device 0, attempt 3/3: passed" in output.out


def test_warm_up_failing_every_attempt_stops_the_config(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=3, behaviors={"1": "segv"})
    result = fixture.supervise(probe=fixture.fake_probe(1, 1, 1, 0))
    assert result.outcome == STOPPED_AFTER_SESSION
    assert "gpu warm-up failed on device 0" in result.message
    assert "after 3 attempts" in result.message
    assert fixture.events() == ["client", "probe", "probe", "probe"]
    # The failed warm-up stays pending for the next config.
    second = Fixture(tmp_path, name="second", pairs=2)
    assert second.supervise(probe=second.fake_probe()) == SupervisionResult(0, COMPLETED)
    assert second.events() == ["probe", "client"]


def test_missing_probe_executable_counts_as_a_failed_attempt(tmp_path: Path):
    fixture = Fixture(tmp_path, pairs=3, behaviors={"1": "segv"})
    ClientSupervisor._warmUpState.request(0)
    result = fixture.supervise(probe=GpuProbe([str(tmp_path / "missing-probe")], 0))
    assert result.outcome == STOPPED_BEFORE_SESSION
    assert "after 3 attempts" in result.message
    assert fixture.events() == []


@linux_only
@pytest.mark.parametrize("number, expected", [(signal.SIGSEGV, "11 (SIGSEGV)"), (60, "60")])
def test_a_signalled_probe_is_described_by_its_signal(tmp_path: Path, number, expected):
    command = [sys.executable, "-c", f"import os; os.kill(os.getpid(), {int(number)})"]
    reason = ClientSupervisor._runProbeOnce(GpuProbe(command, 0), tmp_path)
    assert reason == f"killed by signal {expected}"


def probe_command(code: str) -> List[str]:
    return [sys.executable, "-c", code]


def pid_writing_probe(pid_file: Path, prefix: str = "") -> GpuProbe:
    """A probe that records its pid and then sleeps, after running the optional prefix code."""
    code = (
        f"import os, signal, time; {prefix}"
        f"open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(3600)"
    )
    return GpuProbe(probe_command(code), 0)


@linux_only
def test_a_probe_ignoring_sigterm_is_killed_after_the_timeout(
    tmp_path: Path, monkeypatch, leftover_pids
):
    monkeypatch.setattr(ClientSupervisor, "PROBE_TIMEOUT_SECONDS", 3)
    monkeypatch.setattr(ClientSupervisor, "KILL_GRACE_SECONDS", 0.2)
    pid_file = tmp_path / "probe.pid"
    probe = pid_writing_probe(pid_file, "signal.signal(signal.SIGTERM, signal.SIG_IGN); ")
    reason = ClientSupervisor._runProbeOnce(probe, tmp_path)
    pid = read_pid(pid_file)
    assert pid is not None
    leftover_pids.append(pid)
    assert reason == "timed out after 3s"
    assert_gone(pid)


@linux_only
def test_a_probe_that_is_not_reaped_stops_the_supervision(
    tmp_path: Path, monkeypatch, leftover_pids, capsys
):
    monkeypatch.setattr(ClientSupervisor, "PROBE_TIMEOUT_SECONDS", 5)
    monkeypatch.setattr(ClientSupervisor, "KILL_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(ClientSupervisor, "_reap", lambda *args, **kwargs: False)
    pid_file = tmp_path / "probe.pid"
    fixture = Fixture(tmp_path, pairs=3, behaviors={"1": "segv"})
    result = fixture.supervise(probe=pid_writing_probe(pid_file))
    pid = read_pid(pid_file)
    assert pid is not None
    leftover_pids.append(pid)
    assert result.outcome == STOPPED_AFTER_SESSION
    assert f"probe process {pid} did not exit after SIGKILL" in result.message
    # One client launch (the crash); no second probe and no relaunch.
    assert len(fixture.launches()) == 1
    assert_gone(pid)
    output = capsys.readouterr()
    lines = (output.out + output.err).splitlines()
    # The message is only returned in the result, not also printed, and never names a client.
    assert result.message.count("did not exit after SIGKILL") == 1
    assert not any("did not exit after SIGKILL" in line for line in lines)
    assert "client process" not in result.message


@linux_only
def test_signal_during_the_probe_kills_the_probe_and_propagates(
    tmp_path: Path, sentinel_handlers, leftover_pids
):
    pid_file = tmp_path / "probe.pid"
    probe = pid_writing_probe(pid_file)

    def terminate_when_running():
        wait_until(lambda: read_pid(pid_file) is not None)
        os.kill(os.getpid(), signal.SIGTERM)

    timer = threading.Thread(target=terminate_when_running)
    timer.start()
    try:
        with pytest.raises(SupervisorTerminated):
            with _terminateOnSignals():
                ClientSupervisor._runProbeOnce(probe, tmp_path)
    finally:
        timer.join()
    pid = read_pid(pid_file)
    leftover_pids.append(pid)
    assert_gone(pid)


@linux_only
def test_signal_during_the_probe_names_an_unreaped_probe(
    tmp_path: Path, monkeypatch, sentinel_handlers, leftover_pids, capsys
):
    monkeypatch.setattr(ClientSupervisor, "KILL_GRACE_SECONDS", 0.1)
    monkeypatch.setattr(ClientSupervisor, "_reap", lambda *args, **kwargs: False)
    pid_file = tmp_path / "probe.pid"
    probe = pid_writing_probe(pid_file)

    def terminate_when_running():
        wait_until(lambda: read_pid(pid_file) is not None)
        os.kill(os.getpid(), signal.SIGTERM)

    timer = threading.Thread(target=terminate_when_running)
    timer.start()
    try:
        with pytest.raises(SupervisorTerminated):
            with _terminateOnSignals():
                ClientSupervisor._runProbeOnce(probe, tmp_path)
    finally:
        timer.join()
    pid = read_pid(pid_file)
    leftover_pids.append(pid)
    output = capsys.readouterr()
    assert f"probe process {pid} did not exit after SIGKILL" in output.out + output.err
    assert "client process" not in output.out + output.err


def test_a_probe_that_cannot_be_started_fails_the_attempt(tmp_path: Path, monkeypatch):
    def failing_preexec():
        def fail():
            raise OSError("prctl failed")

        return fail

    monkeypatch.setattr(ClientSupervisor, "_dieWithParent", failing_preexec)
    reason = ClientSupervisor._runProbeOnce(GpuProbe(probe_command("pass"), 0), tmp_path)
    assert reason is not None
    assert "preexec_fn" in reason


ENDED = _DbState(lastSessionId=2, lastSessionDied=False)
DIED = _DbState(lastSessionId=2, lastSessionDied=True)
NO_SESSION = _DbState(lastSessionId=1)


@pytest.mark.parametrize(
    "after, exitCode, expected",
    [
        pytest.param(ENDED, 0, False, id="clean_exit"),
        pytest.param(ENDED, 3, False, id="validation_failures_of_an_ended_session"),
        pytest.param(DIED, 0, False, id="exit_zero_with_a_died_session"),
        pytest.param(DIED, 134, True, id="abort_with_a_died_session"),
        pytest.param(ENDED, -11, True, id="wrapper_signalled"),
        pytest.param(ENDED, None, True, id="killed_by_the_watchdog"),
        # An exit code of a client that never started a session is not a validation failure count.
        pytest.param(NO_SESSION, 1, True, id="non_zero_without_a_session"),
    ],
)
def test_exited_abnormally(after, exitCode, expected):
    before = _DbState(lastSessionId=1)
    assert _exitedAbnormally(before, after, _Launch(exitCode)) is expected


def make_wrapper(directory: Path) -> Path:
    wrapper = directory / "client.sh"
    wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} {FAKE_CLIENT} "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    return wrapper


def make_client_with_probe(directory: Path, probe_exists: bool) -> Path:
    client = make_wrapper(directory)
    if probe_exists:
        (directory / ClientSupervisor.PROBE_NAME).write_text("")
    return client


def test_find_gpu_probe_returns_the_probe_next_to_the_client(tmp_path: Path):
    client = make_client_with_probe(tmp_path, probe_exists=True)
    probe = findGpuProbe(client, 3)
    assert probe == GpuProbe([str(tmp_path / "tensilelite-gpu-probe"), "--device-idx", "3"], 3)


def test_find_gpu_probe_warns_once_when_the_probe_is_missing(tmp_path: Path, capsys):
    client = make_client_with_probe(tmp_path, probe_exists=False)
    assert findGpuProbe(client, 0) is None
    assert findGpuProbe(client, 1) is None
    output = capsys.readouterr()
    assert (output.out + output.err).count("tensilelite-gpu-probe not found in") == 1


def test_find_gpu_probe_falls_back_to_the_default_build_directory(tmp_path: Path):
    client = make_client_with_probe(tmp_path, probe_exists=False)
    default_dir = ClientSupervisor.DEFAULT_CLIENT_PATH.parent
    default_dir.mkdir()
    (default_dir / ClientSupervisor.PROBE_NAME).write_text("")
    probe = findGpuProbe(client, 2)
    assert probe == GpuProbe([str(default_dir / "tensilelite-gpu-probe"), "--device-idx", "2"], 2)


def test_find_gpu_probe_prefers_the_probe_next_to_the_client(tmp_path: Path):
    client = make_client_with_probe(tmp_path, probe_exists=True)
    default_dir = ClientSupervisor.DEFAULT_CLIENT_PATH.parent
    default_dir.mkdir()
    (default_dir / ClientSupervisor.PROBE_NAME).write_text("")
    assert findGpuProbe(client, 0).command[0] == str(tmp_path / "tensilelite-gpu-probe")


def test_find_gpu_probe_warning_names_both_directories(tmp_path: Path, capsys):
    client = make_client_with_probe(tmp_path, probe_exists=False)
    findGpuProbe(client, 0)
    output = capsys.readouterr()
    text = output.out + output.err
    assert str(tmp_path) in text and str(ClientSupervisor.DEFAULT_CLIENT_PATH.parent) in text


def test_find_gpu_probe_accepts_odd_client_values():
    assert findGpuProbe("python client.py", 0) is None
    assert findGpuProbe(Path("/fake/client"), 0) is None


# ---------------------------------------------------------------------------
# Shutdown: kill escalation, signals, supervisor death and the bounded reap.
# ---------------------------------------------------------------------------

DEADLINE_SECONDS = 10.0


def wait_until(predicate, timeout: float = DEADLINE_SECONDS) -> bool:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0]
    except OSError:
        return True
    # A zombie is dead; its parent (or init, if it was reparented) just did not reap it yet.
    return state != "Z"


def read_pid(path: Path) -> Optional[int]:
    try:
        return int(path.read_text())
    except (OSError, ValueError):
        return None


def assert_gone(*pids: int) -> None:
    assert wait_until(lambda: not any(is_alive(pid) for pid in pids)), [
        pid for pid in pids if is_alive(pid)
    ]


@pytest.fixture
def leftover_pids():
    """Pids that a test may leave behind on failure or by design; they are killed at teardown."""
    pids: List[int] = []
    yield pids
    for pid in filter(None, pids):
        if is_alive(pid):
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)


@pytest.fixture
def sentinel_handlers():
    sentinel = lambda signal_number, frame: None
    previous = {
        number: signal.signal(number, sentinel) for number in (signal.SIGTERM, signal.SIGHUP)
    }
    yield sentinel
    for number, handler in previous.items():
        signal.signal(number, handler)


def assert_handlers_are(handler) -> None:
    assert signal.getsignal(signal.SIGTERM) is handler
    assert signal.getsignal(signal.SIGHUP) is handler


def hanging_fixture(directory: Path, behavior: str) -> Fixture:
    return Fixture(
        directory,
        pairs=3,
        behaviors={"1": behavior},
        ready_file=str(directory / "ready"),
        child_pid_file=str(directory / "child.pid"),
    )


def test_sigkill_escalation_kills_a_client_ignoring_sigterm(tmp_path: Path, leftover_pids):
    fixture = hanging_fixture(tmp_path, "hang_ignore_term")
    start = time.monotonic()
    result = fixture.supervise(timeout=1)
    leftover_pids.append(fixture.launches()[0]["pid"])
    assert time.monotonic() - start < 15
    assert result == SupervisionResult(1, COMPLETED)
    assert fixture.statuses() == {0: PASS, 1: UNKNOWN, 2: PASS}
    assert_gone(fixture.launches()[0]["pid"])


def test_watchdog_kills_the_children_of_the_client(tmp_path: Path, leftover_pids):
    fixture = hanging_fixture(tmp_path, "spawn_child_hang")
    result = fixture.supervise(timeout=1)
    child = read_pid(tmp_path / "child.pid")
    assert child is not None
    leftover_pids.append(child)
    assert result == SupervisionResult(1, COMPLETED)
    assert_gone(fixture.launches()[0]["pid"], child)


@pytest.mark.parametrize("exception", [KeyboardInterrupt(), SystemExit(3)])
def test_exception_while_waiting_kills_the_client_tree(
    tmp_path: Path, monkeypatch, leftover_pids, exception
):
    fixture = hanging_fixture(tmp_path, "spawn_child_hang")
    ready = tmp_path / "ready"

    def interrupt(*args, **kwargs):
        if ready.exists():
            raise exception

    monkeypatch.setattr(ClientSupervisor, "_findHungPair", interrupt)
    with pytest.raises(type(exception)) as raised:
        ClientSupervisor._runLaunch(
            fixture.command, tmp_path, fixture.db, 0, Watchdog(1, TEST_POLL_SECONDS, GRACE_SECONDS)
        )
    assert raised.value is exception
    child = read_pid(tmp_path / "child.pid")
    assert child is not None
    leftover_pids.append(child)
    assert_gone(fixture.launches()[0]["pid"], child)


def test_unreaped_client_is_reported_without_hiding_the_exception(
    tmp_path: Path, monkeypatch, capsys, leftover_pids
):
    fixture = hanging_fixture(tmp_path, "hang_ignore_term")
    monkeypatch.setattr(ClientSupervisor, "_reap", lambda *args, **kwargs: False)

    def interrupt(*args, **kwargs):
        if (tmp_path / "ready").exists():
            raise KeyboardInterrupt

    monkeypatch.setattr(ClientSupervisor, "_findHungPair", interrupt)
    with pytest.raises(KeyboardInterrupt):
        ClientSupervisor._runLaunch(
            fixture.command, tmp_path, fixture.db, 0, Watchdog(1, TEST_POLL_SECONDS, GRACE_SECONDS)
        )
    output = capsys.readouterr()
    pid = fixture.launches()[0]["pid"]
    leftover_pids.append(pid)
    assert f"client process {pid} did not exit after SIGKILL" in output.out + output.err
    assert_gone(pid)


def test_signal_during_the_cleanup_still_kills_the_client_group(
    tmp_path: Path, monkeypatch, leftover_pids
):
    fixture = hanging_fixture(tmp_path, "hang_ignore_term")
    ready = tmp_path / "ready"

    def interrupt(*args, **kwargs):
        if ready.exists():
            raise KeyboardInterrupt

    monkeypatch.setattr(ClientSupervisor, "_findHungPair", interrupt)
    original_wait = subprocess.Popen.wait

    def wait(self, timeout=None):
        # The grace wait is the only one with this timeout; the launch loop polls faster.
        if timeout == GRACE_SECONDS:
            raise SupervisorTerminated(signal.SIGTERM)
        return original_wait(self, timeout)

    monkeypatch.setattr(subprocess.Popen, "wait", wait)
    with pytest.raises(SupervisorTerminated):
        ClientSupervisor._runLaunch(
            fixture.command, tmp_path, fixture.db, 0, Watchdog(1, TEST_POLL_SECONDS, GRACE_SECONDS)
        )
    pid = fixture.launches()[0]["pid"]
    leftover_pids.append(pid)
    assert_gone(pid)


# ---------------------------------------------------------------------------
# Signal handlers of the supervisor.
# ---------------------------------------------------------------------------


def test_supervise_restores_the_signal_handlers(tmp_path: Path, sentinel_handlers):
    Fixture(tmp_path, pairs=2).supervise()
    assert_handlers_are(sentinel_handlers)


def test_supervise_restores_the_signal_handlers_after_an_exception(
    tmp_path: Path, monkeypatch, sentinel_handlers
):
    def fail(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(ClientSupervisor, "_runLaunch", fail)
    with pytest.raises(RuntimeError, match="boom"):
        Fixture(tmp_path, pairs=2).supervise()
    assert_handlers_are(sentinel_handlers)


@pytest.mark.parametrize("signal_number", [signal.SIGTERM, signal.SIGHUP])
def test_signal_during_supervision_raises_supervisor_terminated(
    tmp_path: Path, monkeypatch, sentinel_handlers, signal_number
):
    def receive_signal(*args, **kwargs):
        os.kill(os.getpid(), signal_number)
        time.sleep(DEADLINE_SECONDS)

    monkeypatch.setattr(ClientSupervisor, "_runLaunch", receive_signal)
    with pytest.raises(SupervisorTerminated) as raised:
        Fixture(tmp_path, pairs=2).supervise()
    assert raised.value.code == 128 + signal_number
    assert raised.value.signalNumber == signal_number
    assert_handlers_are(sentinel_handlers)


def test_second_signal_during_the_cleanup_is_ignored(sentinel_handlers):
    with _terminateOnSignals():
        with pytest.raises(SupervisorTerminated):
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(DEADLINE_SECONDS)
        assert_handlers_are(signal.SIG_IGN)
    assert_handlers_are(sentinel_handlers)


@pytest.mark.parametrize("signal_number", [signal.SIGTERM, signal.SIGHUP])
def test_a_signal_ignored_before_supervision_stays_ignored(sentinel_handlers, signal_number):
    previous = signal.signal(signal_number, signal.SIG_IGN)
    other = signal.SIGTERM if signal_number == signal.SIGHUP else signal.SIGHUP
    try:
        with _terminateOnSignals():
            assert signal.getsignal(signal_number) == signal.SIG_IGN
            assert signal.getsignal(other) not in (signal.SIG_IGN, sentinel_handlers)
            with pytest.raises(SupervisorTerminated) as raised:
                os.kill(os.getpid(), other)
                time.sleep(DEADLINE_SECONDS)
            assert raised.value.signalNumber == other
        assert signal.getsignal(signal_number) == signal.SIG_IGN
        assert signal.getsignal(other) is sentinel_handlers
    finally:
        signal.signal(signal_number, previous)


def test_supervisor_terminated_survives_pickling():
    restored = pickle.loads(pickle.dumps(SupervisorTerminated(signal.SIGHUP)))
    assert isinstance(restored, SupervisorTerminated)
    assert restored.signalNumber == signal.SIGHUP
    assert restored.code == 128 + signal.SIGHUP


def test_supervision_from_a_thread_leaves_the_signal_handlers_alone(
    tmp_path: Path, sentinel_handlers
):
    seen = []
    results = []

    def supervise():
        with _terminateOnSignals():
            seen.append((signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP)))
        results.append(Fixture(tmp_path, pairs=2).supervise())

    thread = threading.Thread(target=supervise)
    thread.start()
    thread.join(DEADLINE_SECONDS)
    assert not thread.is_alive()
    assert seen == [(sentinel_handlers, sentinel_handlers)]
    assert results == [SupervisionResult(0, COMPLETED)]
    assert_handlers_are(sentinel_handlers)


# ---------------------------------------------------------------------------
# A supervisor in its own process that gets signalled.
# ---------------------------------------------------------------------------


def start_driver(directory: Path, command: List[str], fixture: Fixture) -> subprocess.Popen:
    config = directory / "driver.json"
    config.write_text(
        json.dumps(
            {
                "command": command,
                "cwd": str(directory),
                "db": str(fixture.db),
                "grace_seconds": GRACE_SECONDS,
            }
        )
    )
    package_root = Path(ClientSupervisor.__file__).parent.parent
    env = dict(os.environ, PYTHONPATH=str(package_root))
    with open(directory / "driver.log", "w") as log:
        # Own session so that nothing the test leaks can reach pytest.
        return subprocess.Popen(
            [sys.executable, str(DRIVER), str(config)],
            stdout=log,
            stderr=log,
            env=env,
            start_new_session=True,
        )


def wait_for_hang(directory: Path, driver: subprocess.Popen) -> None:
    ready = directory / "ready"
    assert wait_until(lambda: ready.exists() or driver.poll() is not None, 30)
    assert ready.exists(), (directory / "driver.log").read_text()


@linux_only
@pytest.mark.parametrize(
    "signal_number, behavior, exit_code",
    [
        (signal.SIGTERM, "spawn_child_hang", 128 + signal.SIGTERM),
        (signal.SIGHUP, "spawn_child_hang", 128 + signal.SIGHUP),
        # An unhandled KeyboardInterrupt makes Python re-raise SIGINT on itself.
        (signal.SIGINT, "spawn_child_hang", -signal.SIGINT),
        (signal.SIGTERM, "hang_ignore_term", 128 + signal.SIGTERM),
    ],
)
def test_signalled_supervisor_kills_the_client_tree(
    tmp_path: Path, leftover_pids, signal_number, behavior, exit_code
):
    fixture = hanging_fixture(tmp_path, behavior)
    driver = start_driver(tmp_path, fixture.command, fixture)
    leftover_pids.append(driver.pid)
    wait_for_hang(tmp_path, driver)
    pids = [fixture.launches()[0]["pid"]]
    if behavior == "spawn_child_hang":
        pids.append(read_pid(tmp_path / "child.pid"))
    leftover_pids.extend(pids)

    driver.send_signal(signal_number)
    assert driver.wait(DEADLINE_SECONDS) == exit_code
    assert_gone(*pids)


@linux_only
def test_killed_supervisor_takes_the_wrapper_and_the_client_with_it(tmp_path: Path, leftover_pids):
    fixture = hanging_fixture(tmp_path, "spawn_child_hang")
    wrapper = tmp_path / "run.sh"
    wrapper_pid = tmp_path / "wrapper.pid"
    marker = tmp_path / "clocks_reset"
    # Like run.sh, it cannot exec the client because it has work to do afterwards.
    wrapper.write_text(
        "#!/bin/bash\n"
        f"echo $$ > {shlex.quote(str(wrapper_pid))}\n"
        f"{shlex.quote(sys.executable)} {shlex.quote(str(FAKE_CLIENT))} "
        f"--config-file {shlex.quote(str(fixture.ini))}\n"
        f"echo clocks reset > {shlex.quote(str(marker))}\n"
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)
    driver = start_driver(tmp_path, [str(wrapper)], fixture)
    leftover_pids.append(driver.pid)
    wait_for_hang(tmp_path, driver)
    pids = [read_pid(wrapper_pid), fixture.launches()[0]["pid"]]
    # The spawned child is a grandchild without a death signal, so it may survive.
    leftover_pids.extend([*pids, read_pid(tmp_path / "child.pid")])

    driver.send_signal(signal.SIGKILL)
    assert driver.wait(DEADLINE_SECONDS) == -signal.SIGKILL
    assert_gone(*pids)
    assert not marker.exists()


# ---------------------------------------------------------------------------
# Bounded reap.
# ---------------------------------------------------------------------------


def test_unreaped_client_stops_the_supervision_without_a_relaunch(
    tmp_path: Path, monkeypatch, leftover_pids
):
    fixture = Fixture(tmp_path, pairs=3, behaviors={"1": "hang"})
    monkeypatch.setattr(ClientSupervisor, "_reap", lambda *args, **kwargs: False)
    result = fixture.supervise(timeout=1)
    pid = fixture.launches()[0]["pid"]
    leftover_pids.append(pid)
    assert result.restarts == 0
    assert result.outcome == STOPPED_AFTER_SESSION
    assert f"client process {pid} did not exit after SIGKILL" in result.message
    assert len(fixture.launches()) == 1
    assert_gone(pid)


def sleeping_process() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(3600)"], start_new_session=True
    )


def test_kill_group_reports_a_process_that_is_not_reaped():
    process = sleeping_process()

    def never_exits(timeout=None):
        raise subprocess.TimeoutExpired("process", timeout)

    process.wait = never_exits
    try:
        start = time.monotonic()
        assert _killGroup(process, 0.1, 0.1) is False
        assert time.monotonic() - start < DEADLINE_SECONDS
    finally:
        del process.wait
        process.kill()
        process.wait()


def test_kill_group_reports_a_reaped_process():
    process = sleeping_process()
    try:
        assert _killGroup(process, GRACE_SECONDS, DEADLINE_SECONDS) is True
        assert process.returncode == -signal.SIGTERM
    finally:
        process.kill()
        process.wait()


# ---------------------------------------------------------------------------
# The contract of the real GPU probe.
# ---------------------------------------------------------------------------

REAL_PROBE = (
    Path(__file__).resolve().parents[3] / "build_tmp" / "tensilelite" / "client" / PROBE_NAME
)
PROBE_ENV = dict(os.environ, HIP_VISIBLE_DEVICES="0", ROCR_VISIBLE_DEVICES="0")
needs_probe = pytest.mark.skipif(
    not REAL_PROBE.is_file(), reason="tensilelite-gpu-probe is not built (run invoke build-client)"
)
needs_gpu = pytest.mark.skipif(not Path("/dev/kfd").exists(), reason="no AMD GPU")


def run_real_probe(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(REAL_PROBE), *args],
        env=PROBE_ENV,
        capture_output=True,
        text=True,
        timeout=60,
    )


@needs_probe
@pytest.mark.parametrize(
    "args",
    [
        ["--bogus"],
        ["--device-idx"],
        ["--device-idx", "abc"],
        ["--device-idx", "-1"],
        ["--device-idx", "2147483648"],
    ],
)
def test_real_probe_rejects_bad_arguments(args):
    result = run_real_probe(*args)
    assert result.returncode == 2
    assert "usage" in result.stderr


@needs_probe
@needs_gpu
def test_real_probe_passes_on_the_visible_device():
    result = run_real_probe("--device-idx", "0")
    assert result.returncode == 0
    assert "gpu probe passed on device 0" in result.stdout


@needs_probe
@needs_gpu
def test_real_probe_fails_on_a_missing_device():
    result = run_real_probe("--device-idx", "5")
    assert result.returncode == 1
    assert "gpu probe failed on device 5" in result.stderr
