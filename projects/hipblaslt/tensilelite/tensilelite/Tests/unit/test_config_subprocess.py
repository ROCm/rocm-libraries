# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exercise real helper/client lifetimes without GPU work."""

import importlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

with pytest.MonkeyPatch.context() as patch:
    patch.syspath_prepend(Path(__file__).parents[1] / "common")
    test_config = importlib.import_module("test_config")


@pytest.fixture
def helper(tmp_path, monkeypatch):
    monkeypatch.setattr(test_config, "_COMMON_DIR", str(tmp_path))

    def run(body):
        (tmp_path / "phase.py").write_text(
            "def run(config, output_dir, artifact_dir, args):\n"
            + "\n".join("    " + line for line in body.splitlines()) + "\n"
        )
        test_config._call_helper_in_subprocess(
            "phase", "run", str(tmp_path / "client.pid"), "", "", []
        )

    yield run
    pidfile = tmp_path / "client.pid"
    if pidfile.exists():
        try:
            os.kill(int(pidfile.read_text()), signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_success_and_failure(helper):
    helper("pass")
    with pytest.raises(subprocess.CalledProcessError) as error:
        helper("raise SystemExit(3)")
    assert error.value.returncode == 3


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-group cleanup")
@pytest.mark.parametrize("interrupted", [False, True])
def test_reaps_client_after_helper_exit(helper, tmp_path, interrupted):
    def interrupt(signum, frame):
        raise InterruptedError("test timeout")

    previous = signal.signal(signal.SIGUSR1, interrupt)
    body = """import os, signal, subprocess, sys, time
from pathlib import Path
client = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
Path(config).write_text(str(client.pid))
"""
    if interrupted:
        body += "os.kill(os.getppid(), signal.SIGUSR1)\ntime.sleep(60)\n"
    try:
        if interrupted:
            with pytest.raises(InterruptedError, match="test timeout"):
                helper(body)
        else:
            helper(body)
    finally:
        signal.signal(signal.SIGUSR1, previous)

    pid = int((tmp_path / "client.pid").read_text())
    stat = Path(f"/proc/{pid}/stat")
    deadline = time.monotonic() + 2
    while stat.exists():
        # A killed orphan can await reaping by init/the suite supervisor.
        try:
            state = stat.read_text().split(") ", 1)[1]
        except FileNotFoundError:
            break
        if state.startswith("Z"):
            break
        assert time.monotonic() < deadline, "client remained alive after helper exit"
        time.sleep(0.01)
