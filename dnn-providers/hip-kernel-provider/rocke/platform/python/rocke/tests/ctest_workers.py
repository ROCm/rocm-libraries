# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pytest plugin: run one CI shard's tests in several worker processes.

Load it with ``python -m pytest -p rocke.tests.ctest_shard -p
rocke.tests.ctest_workers --rocke-workers N``. CTest runs a CI shard's entries
one after another, and a pytest entry runs its tests in one process on one CPU
while the shard's machine has many. With ``--rocke-workers N`` the entry's
process starts ``N`` pytest processes on its own command line instead of running
tests itself, relays their output as it comes, each line prefixed with
``[w<index>]``, and exits with their combined status
(:func:`combined_exit_status`). With ``rocke.tests.ctest_shard`` loaded, the
run first prints that plugin's header line, unprefixed, as a run without
workers does.

Each worker collects every test and keeps the same tests the run would keep
without workers (``-k``, ``-m``, the CI shard), then only its own share of them:
the shard keys of ``rocke.tests.ctest_shard`` (:func:`~rocke.tests.ctest_shard.shard_key`),
the keys with a ``rocke_shard_cost`` mark spread over the workers heaviest
first, the others by a hash that does not repeat the CI shard's
(:func:`worker_keys`). Every test of the selection runs in exactly one worker,
the tests of one parameter set in the same worker, in the order they would run
without workers. A worker whose share is empty passes when the selection is not.

The worker count is at most the CPUs the process may use
(:func:`~rocke.tests.host_cpus.available_cpus`); with one, or for a run that
needs one process (:data:`_SINGLE_PROCESS_OPTIONS`: ``--collect-only``,
``-x``/``--maxfail``, ``--junitxml``, ``--pdb``, ``--lf``, ``--help`` and the
like), the tests run in this process as without the plugin. Each worker gets an
equal share of those CPUs: on Linux it runs on its own block of them
(:func:`worker_cpus`), so the process pools its tests start, sized by affinity,
size themselves to the share; everywhere the thread counts of the numeric
libraries (``OMP_NUM_THREADS`` and the like) are lowered to the share. A worker
gets its own ``--basetemp`` below the run's when the run names one, and no
``.pytest_cache``. A worker stops when the process
that started it is gone, so a CTest timeout that kills the entry stops its
workers too.
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import BinaryIO

import pytest

from .ci_shard import (
    ACTIVATION_FLAG,
    WORKER_ENV,
    adopt_worker,
    current_shard,
    current_worker,
    header_line,
    place_keys,
)
from .ctest_shard import shard_costs, shard_key
from .host_cpus import available_cpus

__all__ = [
    "PLUGIN_NAME",
    "WORKERS_OPTION",
    "combined_exit_status",
    "worker_count",
    "worker_cpus",
    "worker_keys",
]

PLUGIN_NAME = "rocke.tests.ctest_workers"
WORKERS_OPTION = "--rocke-workers"
#: The pid of the process that started a worker.
_PARENT_ENV = "ROCKE_TEST_WORKER_PARENT"
_SHARD_PLUGIN = "rocke.tests.ctest_shard"
#: How long the run waits for a worker's output to end once the worker has
#: exited (a process the worker left running may hold its output open).
_RELAY_DRAIN_SECONDS = 10
# Salts the worker hash so that it is independent of the CI shard's hash: the
# keys a CI shard keeps all share their shard hash modulo the shard count.
_WORKER_SALT = "rocke-ctest-workers:"
_PARENT_POLL_SECONDS = 0.5
#: The thread-count variables of the numeric libraries a test may load (OpenMP,
#: BLAS, numexpr). A worker gets each at most its share of the CPUs, so the
#: workers' threads do not oversubscribe the machine.
_THREAD_COUNT_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
#: Options whose run needs one process: it only reports, stops early, debugs
#: interactively, reads or writes the run's cache, or writes one report file.
#: Each is the ``config.option`` attribute of a pytest option whose default is
#: false.
_SINGLE_PROCESS_OPTIONS = (
    "help",
    "version",
    "showfixtures",
    "show_fixtures_per_test",
    "markers",
    "cacheshow",
    "cacheclear",
    "collectonly",
    "setuponly",
    "setupplan",
    "maxfail",
    "usepdb",
    "trace",
    "lf",
    "failedfirst",
    "newfirst",
    "stepwise",
    "stepwise_skip",
    "xmlpath",
)


def pytest_addoption(parser):
    parser.addoption(
        WORKERS_OPTION,
        type=int,
        default=1,
        metavar="N",
        help=(
            "run the selected tests in N worker processes (at most the CPUs "
            "this process may use); 1 runs them in this process"
        ),
    )


def _worker_of_run() -> tuple[int, int] | None:
    """``(index, count)`` when this process is a worker, else None."""
    try:
        return current_worker()
    except ValueError as error:
        raise pytest.UsageError(f"rocke-ctest-workers: {error}") from error


def _needs_one_process(config) -> bool:
    return any(getattr(config.option, name, None) for name in _SINGLE_PROCESS_OPTIONS)


def worker_count(requested: int, cpus: int) -> int:
    """The workers a run asking for ``requested`` starts on ``cpus`` CPUs."""
    if requested < 1:
        raise pytest.UsageError(
            f"rocke-ctest-workers failed - {WORKERS_OPTION}={requested}, "
            "expected at least 1"
        )
    return max(1, min(requested, cpus))


def worker_cpus(
    cpus: Sequence[int], usable: int, index: int, count: int, rotation: int = 0
) -> list[int]:
    """The CPUs worker ``index`` of ``count`` runs on: block
    ``(index + rotation) % blocks`` of the sorted ``cpus`` this process may run
    on, cut into blocks of ``usable // count`` (at least one), where ``usable``
    is how many of them it may use at once. The run picks ``rotation`` from its
    pid, so the workers of entries a parallel CTest runs at once mostly land on
    different blocks."""
    ordered = sorted(cpus)
    size = max(1, min(usable, len(ordered)) // count)
    blocks = max(1, len(ordered) // size)
    start = ((index + rotation) % blocks) * size
    return ordered[start : start + size]


def worker_keys(
    keys: Sequence[str], costs: dict[str, float], count: int
) -> dict[str, int]:
    """The worker of ``count`` that runs each shard key of ``keys``:
    :func:`~rocke.tests.ci_shard.place_keys` over salted keys, so the keys of
    one CI shard spread over the workers."""
    salted = {f"{_WORKER_SALT}{key}": key for key in keys}
    placement = place_keys(
        salted,
        count,
        {salt: costs[key] for salt, key in salted.items() if key in costs},
    )
    return {key: placement[salt] for salt, key in salted.items()}


def combined_exit_status(statuses: Sequence[int]) -> int:
    """The exit status of a run whose workers exited with ``statuses``: a
    status that is not pytest's (a worker killed by a signal, or one that
    exited on its own) is an internal error; then the first of interrupted,
    internal error, usage error and tests failed that some worker reports; no
    tests collected when every worker reports it; else OK."""
    pytest_statuses = {int(code) for code in pytest.ExitCode}
    if any(status not in pytest_statuses for status in statuses):
        return int(pytest.ExitCode.INTERNAL_ERROR)
    for status in (
        pytest.ExitCode.INTERRUPTED,
        pytest.ExitCode.INTERNAL_ERROR,
        pytest.ExitCode.USAGE_ERROR,
        pytest.ExitCode.TESTS_FAILED,
    ):
        if status in statuses:
            return int(status)
    if statuses and all(s == pytest.ExitCode.NO_TESTS_COLLECTED for s in statuses):
        return int(pytest.ExitCode.NO_TESTS_COLLECTED)
    return int(pytest.ExitCode.OK)


# ---------------------------------------------------------------- the parent


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config):
    """Run the tests in workers instead of in this process, when asked."""
    if _worker_of_run() is not None or _needs_one_process(config):
        return None
    usable = available_cpus()
    count = worker_count(config.getoption(WORKERS_OPTION), usable)
    if count == 1:
        return None
    return _run_workers(config, count, usable)


def _worker_argv(config, index: int) -> list[str]:
    argv = [sys.executable, "-m", "pytest", *config.invocation_params.args]
    argv += ["-p", "no:cacheprovider"]
    if config.option.basetemp:
        argv += ["--basetemp", str(Path(config.option.basetemp) / f"worker{index}")]
    return argv


def _worker_environment(share: int) -> dict[str, str]:
    """The environment of every worker but its index: this one, the parent
    pid, and each thread count lowered to the share."""
    environment = dict(os.environ, **{_PARENT_ENV: str(os.getpid())})
    for name in _THREAD_COUNT_ENV:
        current = environment.get(name, "")
        threads = int(current) if current.isdigit() and int(current) > 0 else share
        environment[name] = str(min(threads, share))
    return environment


@dataclasses.dataclass
class _Worker:
    index: int
    process: subprocess.Popen
    relay: threading.Thread


def _run_workers(config, count: int, usable: int) -> int:
    started = time.monotonic()
    out = sys.stdout
    share = max(1, usable // count)
    if config.pluginmanager.has_plugin(_SHARD_PLUGIN):
        # The shard plugin's header, unprefixed, as a run without workers
        # prints it: CI-log readers find an entry's shard count by it.
        try:
            shard = current_shard(from_gtest=config.getoption(ACTIVATION_FLAG))
        except ValueError as error:
            raise pytest.UsageError(f"rocke-ctest-shard: {error}") from error
        out.write(header_line(shard) + "\n")
    out.write(f"rocke-ctest-workers: {count} workers, {share} CPUs each\n")
    out.flush()
    environment = _worker_environment(share)
    if config.option.basetemp:  # pytest creates a basetemp, not its parents
        Path(config.option.basetemp).mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    workers: list[_Worker] = []
    try:
        for index in range(count):
            process = subprocess.Popen(
                _worker_argv(config, index),
                cwd=config.invocation_params.dir,
                env={**environment, WORKER_ENV: f"{index}/{count}"},
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            relay = threading.Thread(
                target=_relay_lines,
                args=(process.stdout, f"[w{index}] ".encode(), out.buffer, lock),
                name=f"rocke-ctest-workers-relay-{index}",
                daemon=True,
            )
            relay.start()
            workers.append(_Worker(index, process, relay))
        statuses = [worker.process.wait() for worker in workers]
        for worker in workers:
            worker.relay.join(_RELAY_DRAIN_SECONDS)
    finally:
        for worker in workers:
            if worker.process.poll() is None:
                worker.process.kill()
                worker.process.wait()
    status = combined_exit_status(statuses)
    out.write(
        f"rocke-ctest-workers: {count} workers exited {statuses} in "
        f"{time.monotonic() - started:.1f}s; the run exits {status}\n"
    )
    out.flush()
    return status


def _relay_lines(
    source: BinaryIO, prefix: bytes, sink: BinaryIO, lock: threading.Lock
) -> None:
    """Copy each line of ``source`` to ``sink`` behind ``prefix``, a whole line
    at a time, until ``source`` ends. When ``sink`` fails, the rest of
    ``source`` is still read, so the worker never blocks on a full pipe."""
    relaying = True
    for line in iter(source.readline, b""):
        if not relaying:
            continue
        with lock:
            try:
                sink.write(prefix + (line if line.endswith(b"\n") else line + b"\n"))
                sink.flush()
            except OSError:
                relaying = False
    source.close()


# ---------------------------------------------------------------- a worker


#: The pid of the process that started this worker; 0 in any other process.
_parent_pid = 0


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests(early_config, parser, args):
    """In a worker, before any conftest runs: take the worker's identity and
    its parent's pid out of the environment (:func:`~rocke.tests.ci_shard.adopt_worker`),
    so a pytest process a test starts is no worker, and run on the worker's
    block of the CPUs (Linux), before a library sizes its thread pool to them."""
    global _parent_pid
    try:
        worker = adopt_worker()
    except ValueError as error:
        raise pytest.UsageError(f"rocke-ctest-workers: {error}") from error
    parent = os.environ.pop(_PARENT_ENV, "")
    if worker is None:
        return
    _parent_pid = int(parent) if parent.isascii() and parent.isdigit() else 0
    setaffinity = getattr(os, "sched_setaffinity", None)
    if setaffinity is None:
        return
    index, count = worker
    block = worker_cpus(
        os.sched_getaffinity(0), available_cpus(), index, count, rotation=_parent_pid
    )
    setaffinity(0, block)


def pytest_configure(config):
    if _worker_of_run() is not None:
        _stop_with_parent()


def _stop_with_parent() -> None:
    """End this worker when the process that started it is gone."""
    parent = _parent_pid
    if parent <= 0:
        return
    wait_for_parent = _parent_exit_waiter(parent)

    def watch() -> None:
        wait_for_parent()
        os._exit(int(pytest.ExitCode.INTERRUPTED))

    threading.Thread(
        target=watch, name="rocke-ctest-workers-parent", daemon=True
    ).start()


def _parent_exit_waiter(parent: int) -> Callable[[], None]:
    """A callable that returns once process ``parent``, alive now, has exited.

    Windows: a handle to the process, opened now, so a later process reusing
    the pid is not mistaken for it. POSIX: the parent pid changes when the
    parent exits and the worker is re-parented."""
    if os.name == "nt":
        import _winapi

        synchronize = 0x00100000
        try:
            handle = _winapi.OpenProcess(synchronize, False, parent)
        except OSError:
            return lambda: None
        return lambda: _winapi.WaitForSingleObject(handle, _winapi.INFINITE)

    def wait() -> None:
        while os.getppid() == parent:
            time.sleep(_PARENT_POLL_SECONDS)

    return wait


@dataclasses.dataclass
class _WorkerState:
    before: int = 0
    kept: int = 0


_STATE = pytest.StashKey[_WorkerState]()


@pytest.hookimpl(hookwrapper=True)
def pytest_collection_modifyitems(config, items):
    """After every other selection (the CI shard's included), keep this
    worker's share. The other workers' items are dropped, not deselected: the
    worker that runs an item reports it, and ``rocke.tests.run_records``
    records each item once per run."""
    yield
    worker = _worker_of_run()
    if worker is None:
        return
    index, count = worker
    state = config.stash[_STATE] = _WorkerState(before=len(items))
    placement = worker_keys(
        [shard_key(item) for item in items], shard_costs(items), count
    )
    items[:] = [item for item in items if placement[shard_key(item)] == index]
    state.kept = len(items)


def pytest_report_header(config):
    worker = _worker_of_run()
    if worker is not None:
        return f"rocke-ctest-workers: worker {worker[0]}/{worker[1]}"
    return None


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    state = config.stash.get(_STATE, None)
    worker = _worker_of_run()
    if state is not None and worker is not None:
        terminalreporter.write_line(
            f"rocke-ctest-workers: worker {worker[0]}/{worker[1]} kept "
            f"{state.kept} of {state.before}"
        )


def pytest_sessionfinish(session, exitstatus):
    state = session.config.stash.get(_STATE, None)
    if (
        state is not None
        and exitstatus == pytest.ExitCode.NO_TESTS_COLLECTED
        and state.before
        and not state.kept
    ):
        session.exitstatus = pytest.ExitCode.OK
