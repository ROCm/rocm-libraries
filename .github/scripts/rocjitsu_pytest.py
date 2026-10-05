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
import signal
import subprocess
import sys
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
            detail=str(report.longrepr) if report.failed else None,
        )
        self.save()


def pytest_addoption(parser):
    parser.addoption("--rocjitsu-report-dir")


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
