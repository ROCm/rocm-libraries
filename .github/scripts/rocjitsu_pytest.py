# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Bound a Linux emulation run and preserve pytest results as tests finish.

The supervisor becomes a child subreaper so it can clean up descendants even
when pytest workers exit or test helpers create their own process groups.
Load this module in pytest with ``-p rocjitsu_pytest`` to record progress.
"""

import argparse
import contextlib
import ctypes
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

import pytest


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


class Progress:
    def __init__(self, directory):
        self.path = directory / "progress.json"
        self.items = []
        self.reports = {}

    def save(self):
        counts = dict(passed=0, failed=0, skipped=0, incomplete=0)
        results = {}
        for nodeid in self.items:
            phases = self.reports.get(nodeid, {})
            if any(p["outcome"] == "failed" for p in phases.values()):
                outcome = "failed"
            elif "teardown" not in phases:
                outcome = "incomplete"
            elif any(p["outcome"] == "skipped" for p in phases.values()):
                outcome = "skipped"
            elif phases.get("call", {}).get("outcome") == "passed":
                outcome = "passed"
            else:
                outcome = "incomplete"
            counts[outcome] += 1
            results[nodeid] = dict(outcome=outcome, phases=phases)
        finished = sum("teardown" in self.reports.get(item, {}) for item in self.items)
        write_json(
            self.path,
            dict(selected=len(self.items), finished=finished, **counts, tests=results),
        )

    def pytest_collection_finish(self, session):
        if session.items:
            self.items = [item.nodeid for item in session.items]
            self.save()

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids):
        self.items = ids
        self.save()

    def pytest_runtest_logreport(self, report):
        phases = self.reports.setdefault(report.nodeid, {})
        phases[report.when] = dict(
            outcome=report.outcome,
            seconds=report.duration,
            wasxfail=getattr(report, "wasxfail", None),
            detail=str(report.longrepr) if report.failed or report.skipped else None,
        )
        self.save()


STALL_SAMPLE_SECONDS = 5


def descendants(pid):
    """Return the processes below pid, found through /proc parent links."""
    children = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        with contextlib.suppress(OSError):
            ppid = int(stat.read_text().rpartition(")")[2].split()[1])
            children.setdefault(ppid, []).append(int(stat.parent.name))
    found, pending = [], [pid]
    while pending:
        for child in children.get(pending.pop(), []):
            found.append(child)
            pending.append(child)
    return found


def cpu_seconds(stat):
    fields = stat.read_text().rpartition(")")[2].split()
    return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")


def describe_process(pid):
    """Per-thread state, wait channel and CPU use over a short sample.

    Threads blocked near 0% point at a deadlock; threads near 100% at a spin.
    """
    proc = Path(f"/proc/{pid}")
    tasks = sorted(proc.glob("task/*"), key=lambda task: int(task.name))
    before = {}
    for task in tasks:
        with contextlib.suppress(OSError):
            before[task] = cpu_seconds(task / "stat")
    time.sleep(STALL_SAMPLE_SECONDS)
    command = (
        (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    )
    lines = [f"pid {pid}: {command}"]
    for task, start in before.items():
        with contextlib.suppress(OSError):
            busy = 100 * (cpu_seconds(task / "stat") - start) / STALL_SAMPLE_SECONDS
            state = (task / "stat").read_text().rpartition(")")[2].split()[0]
            wchan = (task / "wchan").read_text() or "-"
            comm = (task / "comm").read_text().strip()
            lines.append(
                f"  tid {task.name} {comm}: state {state} cpu {busy:.0f}% wchan {wchan}"
            )
    return lines


def native_stacks(pid):
    command = ["gdb", "-p", str(pid), "-batch", "-nx", "-ex", "thread apply all bt"]
    if not shutil.which("gdb"):
        return "gdb not found"
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except subprocess.TimeoutExpired:
        return "gdb timed out"
    return result.stdout + result.stderr


def record_stall(nodeid, directory):
    """Save what a still-running test's processes are doing before its timeout."""
    directory.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^\w.-]+", "_", nodeid)[-150:]
    with (directory / f"{name}.txt").open("w", buffering=1) as output:
        output.write(f"{nodeid} still running; worker pid {os.getpid()}\n")
        for pid in descendants(os.getpid()):
            try:
                output.write("\n".join(describe_process(pid)) + "\n")
                # Shell wrappers are single-threaded; the emulating client is not.
                if len(list(Path(f"/proc/{pid}/task").iterdir())) > 1:
                    output.write(native_stacks(pid) + "\n")
            except OSError as error:
                output.write(f"pid {pid}: {error}\n")


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    after = item.config.getoption("--rocjitsu-stall-dump-after")
    directory = item.config.getoption("--rocjitsu-report-dir")
    if not after or not directory:
        return (yield)
    timer = threading.Timer(
        float(after), record_stall, (item.nodeid, Path(directory) / "stalls")
    )
    timer.daemon = True
    timer.start()
    try:
        return (yield)
    finally:
        timer.cancel()


def duration_key(nodeid):
    """Drop the file path so recorded durations survive a different rootdir."""
    return nodeid.rpartition("::")[2]


def longest_paired_with_shortest(items, durations):
    """Order tests as longest, shortest, next longest, next shortest, ...

    pytest-xdist's load scheduler hands each worker consecutive tests and keeps
    one queued behind the running test. Pairing every long test with a short one
    starts the long tests first and keeps them from queuing behind each other,
    which otherwise sets the suite's wall time. Tests without a recorded duration
    count as the longest.
    """
    longest = max(durations.values(), default=0)
    ranked = sorted(
        items,
        key=lambda item: (
            -durations.get(duration_key(item.nodeid), longest),
            item.nodeid,
        ),
    )
    order = []
    while ranked:
        order.append(ranked.pop(0))
        if ranked:
            order.append(ranked.pop())
    return order


def pytest_addoption(parser):
    parser.addoption("--rocjitsu-report-dir")
    parser.addoption(
        "--rocjitsu-durations",
        help="JSON of test id to seconds from an earlier run, used to order tests",
    )
    parser.addoption(
        "--rocjitsu-stall-dump-after",
        help="seconds after which a running test's process states and stacks "
        "are saved under the report directory's stalls/",
    )


# Every xdist worker collects separately, so this runs in each of them and must
# produce the same order; it runs last to order only the selected tests.
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config, items):
    path = config.getoption("--rocjitsu-durations")
    if path:
        durations = json.loads(Path(path).read_text())
        items[:] = longest_paired_with_shortest(items, durations)


def pytest_configure(config):
    directory = config.getoption("--rocjitsu-report-dir")
    if directory and not hasattr(config, "workerinput"):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        progress = Progress(directory)
        progress.save()
        config.pluginmanager.register(progress)


def reap_descendants():
    """Kill and reap only children owned/adopted by this supervisor."""
    children_file = Path(f"/proc/self/task/{os.getpid()}/children")
    deadline = time.monotonic() + 10
    while True:
        children = [int(pid) for pid in children_file.read_text().split()]
        if not children:
            return
        for pid in children:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
        # Killing an intermediate parent reparents its descendants to us.
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == 0:
                break
        if time.monotonic() >= deadline:
            raise RuntimeError("emulation descendants did not exit after SIGKILL")
        time.sleep(0.01)


def supervise(command, timeout, grace):
    if sys.platform != "linux":
        raise RuntimeError("the rocjitsu CI supervisor requires Linux")
    # PR_SET_CHILD_SUBREAPER applies only to this process and needs no privilege.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")

    def interrupt(signum, frame):
        raise InterruptedError(signum)

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupt)
    start = time.monotonic()
    timed_out = False
    proc = subprocess.Popen(command, start_new_session=True)
    try:
        try:
            status = proc.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, InterruptedError) as error:
            timed_out = isinstance(error, subprocess.TimeoutExpired)
            status = 124 if timed_out else 128 + error.args[0]
            # Give pytest time to stop workers and write its normal JUnit file.
            for signum in (signal.SIGTERM, signal.SIGINT):
                signal.signal(signum, signal.SIG_IGN)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGINT)
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
    finally:
        # This also handles a successful parent that left children behind.
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, signal.SIG_IGN)
        reap_descendants()
    return dict(
        exit_code=status if status >= 0 else 128 - status,
        timed_out=timed_out,
        wall_seconds=time.monotonic() - start,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--grace", type=float, default=30)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if any(
        not math.isfinite(value) or value <= 0 for value in (args.timeout, args.grace)
    ):
        parser.error("timeout and grace must be finite and positive")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a command is required")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    # A reused report directory must never turn an interrupted run into a pass.
    for name in ("progress.json", "execution.json", "junit/tensilelite.xml"):
        (args.report_dir / name).unlink(missing_ok=True)
    try:
        result = supervise(command, args.timeout, args.grace)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        result = dict(exit_code=1, timed_out=False, error=str(error))
    progress_path = args.report_dir / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    result["complete"] = bool(progress.get("selected")) and (
        progress.get("selected") == progress.get("finished") and not result["timed_out"]
    )
    if result["exit_code"] == 0 and (
        not result["complete"]
        or progress.get("passed", 0) == 0
        or not (args.report_dir / "junit/tensilelite.xml").is_file()
    ):
        result["exit_code"] = 1
        result["error"] = "pytest did not complete any passing emulation tests"
    write_json(args.report_dir / "execution.json", result)
    counts = {key: value for key, value in progress.items() if key != "tests"}
    print(f"rocjitsu pytest results: {counts}", flush=True)
    print(f"rocjitsu pytest execution: {result}", flush=True)
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
