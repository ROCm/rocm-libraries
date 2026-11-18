#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Stand-in for tensilelite-client that mimics its benchmark database protocol.

It is invoked like the client (--config-file <ini>) and reads these ini keys: benchmark-db and
fake-plan, a single-line JSON object with the fields:
  pairs: number of solutions k0..k<N-1> of every problem.
  problems: number of problems (default 1); they run in order.
  not_selected: problem index (as a string) to the solution indices the fake prediction iterator
      does not select for it. Those pairs are not run, and when the problem finishes every pair of
      it still NOT_BENCHMARKED becomes NOT_SELECTED, like the real client. A problem that is cut
      short (a crash, stop_after_fail) marks nothing.
  behaviors: solution index (as a string), or "<problem>:<solution>" for one problem, to one
      of pass (default), fail, segv, hang, exit:<n>,
      sleep:<seconds>, hang_ignore_term (hang with SIGTERM and SIGHUP ignored) and
      spawn_child_hang (hang after starting a child that ignores SIGTERM and SIGHUP and sleeps,
      in the same process group; its pid goes to the file named by child_pid_file). Every behavior
      but pass and fail runs after the pair was marked UNKNOWN.
  early: segv or exit:<n>, run on every launch before the database is opened.
  post_session: like early, but after the session was opened and before any pair starts.
  launch_log: file that gets one JSON line per launch: the pid and the database files (suffixes
      "", "-wal" and "-shm") that existed when the launch started.
  stop_after_fail: end the session and exit 1 right after a failed pair, like the client's exit-on-error.
  reset_marked_pairs: right after the session opens, reset the pairs marked INDIRECTLY_FAILED to
      NOT_BENCHMARKED, like the real client does at session start when the indirect option is off.
  child_pid_file: where spawn_child_hang writes the pid of its child.
  ready_file: created by hang_ignore_term and spawn_child_hang right before they sleep.
Like the real benchmark-db client, it asks the kernel for SIGKILL when its parent dies (Linux) and
exits if the parent died before that took effect.
The exit code is the number of failed pairs of the launch, like the real client.
"""

import argparse
import contextlib
import ctypes
import json
import os
import signal
import sqlite3
import subprocess
import sys
import time

from benchmark_db_test_helpers import DatabaseBuilder

NOT_BENCHMARKED, UNKNOWN, PASS, FAILURE, INDIRECTLY_FAILED, NOT_SELECTED = 0, 1, 2, 3, 4, 6


def read_ini(path):
    values = {}
    with open(path) as ini:
        for line in ini:
            key, separator, value = line.rstrip("\n").partition("=")
            if separator:
                values[key] = value
    return values


CHILD_CODE = (
    "import signal, time; "
    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
    "signal.signal(signal.SIGHUP, signal.SIG_IGN); "
    "time.sleep(3600)"
)


def die_with_parent():
    """Mirror dieWithParent of the real client."""
    if sys.platform != "linux":
        return
    PR_SET_PDEATHSIG = 1
    parent = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGKILL), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
    if os.getppid() != parent:
        print("benchmark-db client started after its supervisor exited", file=sys.stderr)
        sys.exit(1)


def signal_ready(plan):
    with open(plan["ready_file"], "w") as ready:
        ready.write(str(os.getpid()))


def crash(behavior, plan):
    if behavior == "segv":
        os.kill(os.getpid(), signal.SIGSEGV)
    if behavior == "hang":
        time.sleep(3600)
    if behavior == "hang_ignore_term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        signal_ready(plan)
        time.sleep(3600)
    if behavior == "spawn_child_hang":
        child = subprocess.Popen([sys.executable, "-c", CHILD_CODE])
        with open(plan["child_pid_file"], "w") as pid_file:
            pid_file.write(str(child.pid))
        signal_ready(plan)
        time.sleep(3600)
    if behavior.startswith("exit:"):
        os._exit(int(behavior.split(":", 1)[1]))


def has_tables(path):
    """Like the real client, treat a missing, empty or table-less file as a new database."""
    if not os.path.exists(path):
        return False
    with contextlib.closing(sqlite3.connect(path)) as connection:
        return (
            connection.execute(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table')"
            ).fetchone()[0]
            == 1
        )


def create_database(path, pairs, problems):
    builder = DatabaseBuilder(path)
    for problem in range(problems):
        builder.add_problem(problem, [8, 8], [8, 8, 8, 8], 1000)
    for index in range(pairs):
        builder.add_solution(index, f"k{index}")
    builder.connection.executemany(
        "INSERT INTO runs(problem_id, solution_id) VALUES(?, ?)",
        [(problem + 1, index + 1) for problem in range(problems) for index in range(pairs)],
    )
    builder.finish()


def open_session(connection):
    cursor = connection.execute(
        "INSERT INTO sessions(pid, start_time, init_seed) VALUES(?, ?, 0)",
        (os.getpid(), time.time()),
    )
    connection.commit()
    return cursor.lastrowid


def reset_marked_pairs(connection):
    connection.execute(
        "UPDATE runs SET status = ?, session_id = NULL, start_time = NULL, end_time = NULL, "
        "validation = NULL, caused_by_problem_id = NULL WHERE status = ?",
        (NOT_BENCHMARKED, INDIRECTLY_FAILED),
    )
    connection.commit()


def begin_pair(connection, session, problem_id, solution_id):
    connection.execute(
        "UPDATE runs SET status = ?, session_id = ?, start_time = ? "
        "WHERE problem_id = ? AND solution_id = ?",
        (UNKNOWN, session, time.time(), problem_id, solution_id),
    )
    connection.commit()


def end_pair(connection, problem_id, solution_id, passed):
    connection.execute(
        "UPDATE runs SET status = ?, end_time = ?, validation = ? "
        "WHERE problem_id = ? AND solution_id = ?",
        (
            PASS if passed else FAILURE,
            time.time(),
            "PASSED" if passed else "FAILED",
            problem_id,
            solution_id,
        ),
    )
    if passed:
        values = {
            "gflops": str(100 + solution_id),
            "time-us": "10",
            "gbps-bandwidth": "50",
            "solution-index": str(solution_id - 1),
        }
        connection.executemany(
            "INSERT INTO run_values VALUES(?, ?, ?, ?)",
            [(problem_id, solution_id, key, value) for key, value in values.items()],
        )
    connection.commit()


def mark_not_selected(connection, session, problem_id):
    connection.execute(
        "UPDATE runs SET status = ?, session_id = ? WHERE problem_id = ? AND status = ?",
        (NOT_SELECTED, session, problem_id, NOT_BENCHMARKED),
    )
    connection.commit()


def end_session(connection, session):
    connection.execute("UPDATE sessions SET end_time = ? WHERE id = ?", (time.time(), session))
    connection.commit()


def run_pair(connection, session, problem_id, solution_id, behavior, plan):
    begin_pair(connection, session, problem_id, solution_id)
    crash(behavior, plan)
    if behavior.startswith("sleep:"):
        time.sleep(float(behavior.split(":", 1)[1]))
    passed = behavior != "fail"
    end_pair(connection, problem_id, solution_id, passed)
    return passed


def main():
    die_with_parent()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    args, _ = parser.parse_known_args()
    ini = read_ini(args.config_file)
    plan = json.loads(ini["fake-plan"])
    db_path = ini["benchmark-db"]

    if plan.get("launch_log"):
        with open(plan["launch_log"], "a") as log:
            existing = [s for s in ("", "-wal", "-shm") if os.path.exists(db_path + s)]
            log.write(json.dumps({"pid": os.getpid(), "existing": existing}) + "\n")
    crash(plan.get("early", ""), plan)

    if not has_tables(db_path):
        create_database(db_path, plan["pairs"], plan.get("problems", 1))
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode=WAL")
    session = open_session(connection)
    if plan.get("reset_marked_pairs"):
        reset_marked_pairs(connection)
    crash(plan.get("post_session", ""), plan)

    behaviors = plan.get("behaviors", {})
    failures = 0
    for problem in range(plan.get("problems", 1)):
        problem_id = problem + 1
        unselected = plan.get("not_selected", {}).get(str(problem), [])
        pending = connection.execute(
            "SELECT solution_id FROM runs WHERE problem_id = ? AND status = ? "
            "ORDER BY solution_id",
            (problem_id, NOT_BENCHMARKED),
        ).fetchall()
        for (solution_id,) in pending:
            if solution_id - 1 in unselected:
                continue
            behavior = behaviors.get(
                f"{problem}:{solution_id - 1}", behaviors.get(str(solution_id - 1), "pass")
            )
            if run_pair(connection, session, problem_id, solution_id, behavior, plan):
                continue
            failures += 1
            if plan.get("stop_after_fail"):
                end_session(connection, session)
                sys.exit(1)
        mark_not_selected(connection, session, problem_id)
    end_session(connection, session)
    sys.exit(min(failures, 255))


if __name__ == "__main__":
    main()
