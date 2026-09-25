# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for process-tree handling in the combined common-config runner."""

import importlib
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


_COMMON_DIR = Path(__file__).resolve().parents[1] / "common"


@pytest.fixture(scope="module")
def config_test_module():
    sys.path.insert(0, str(_COMMON_DIR))
    try:
        yield importlib.import_module("Tensile.Tests.common.test_config")
    finally:
        sys.path.remove(str(_COMMON_DIR))


def _running(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
    except FileNotFoundError:
        return False
    return state != "Z"


def test_nonzero_helper_preserves_called_process_error(config_test_module):
    command = [sys.executable, "-c", "raise SystemExit(7)"]

    with pytest.raises(subprocess.CalledProcessError) as exc:
        config_test_module._run_in_process_group(command, os.environ.copy())

    assert exc.value.returncode == 7
    assert exc.value.cmd == command


def test_non_posix_cleanup_terminates_then_kills(
    monkeypatch, config_test_module
):
    events = []

    class Process:
        pid = 123

        def terminate(self):
            events.append("terminate")

        def kill(self):
            events.append("kill")

        def wait(self, timeout=None):
            events.append(("wait", timeout))
            if timeout is not None and "kill" not in events:
                raise subprocess.TimeoutExpired("helper", timeout)
            return -9

        def poll(self):
            return None

    monkeypatch.setattr(config_test_module.os, "name", "nt")
    monkeypatch.setattr(
        config_test_module, "_PROCESS_TERMINATION_GRACE_SECONDS", 0.25
    )

    config_test_module._terminate_process_tree(Process())

    assert events == ["terminate", ("wait", 0.25), "kill", ("wait", 0.25)]


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Linux /proc process-group behavior",
)
def test_interruption_kills_helper_process_group(
    tmp_path, monkeypatch, config_test_module
):
    parentPidPath = tmp_path / "parent.pid"
    childPidPath = tmp_path / "child.pid"
    childScript = (
        "import os, pathlib, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(childPidPath)!r}).write_text(str(os.getpid())); "
        "time.sleep(60)"
    )
    parentScript = (
        "import os, pathlib, signal, subprocess, sys, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(parentPidPath)!r}).write_text(str(os.getpid())); "
        f"subprocess.Popen([sys.executable, '-c', {childScript!r}]); "
        "time.sleep(60)"
    )

    realPopen = subprocess.Popen

    class InterruptedWait:
        def __init__(self, *args, **kwargs):
            self.process = realPopen(*args, **kwargs)
            self.pid = self.process.pid
            self.interrupted = False

        def __getattr__(self, name):
            return getattr(self.process, name)

        def wait(self, timeout=None):
            if timeout is None and not self.interrupted:
                self.interrupted = True
                deadline = time.monotonic() + 5
                while not childPidPath.exists():
                    if time.monotonic() >= deadline:
                        raise AssertionError("grandchild did not start")
                    time.sleep(0.01)
                raise KeyboardInterrupt
            return self.process.wait(timeout=timeout)

    monkeypatch.setattr(
        config_test_module.subprocess, "Popen", InterruptedWait
    )
    monkeypatch.setattr(
        config_test_module, "_PROCESS_TERMINATION_GRACE_SECONDS", 0.05
    )

    with pytest.raises(KeyboardInterrupt):
        config_test_module._run_in_process_group(
            [sys.executable, "-c", parentScript], os.environ.copy()
        )

    pids = (int(parentPidPath.read_text()), int(childPidPath.read_text()))
    deadline = time.monotonic() + 2
    while any(_running(pid) for pid in pids) and time.monotonic() < deadline:
        time.sleep(0.01)

    assert not any(_running(pid) for pid in pids)
