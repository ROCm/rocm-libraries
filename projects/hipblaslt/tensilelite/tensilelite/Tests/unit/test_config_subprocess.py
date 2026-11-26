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


def wait_until_gone(pid, seconds=2):
    """Wait for pid to exit; a killed orphan may linger as a zombie, which counts as gone."""
    stat = Path(f"/proc/{pid}/stat")
    deadline = time.monotonic() + seconds
    while stat.exists():
        try:
            state = stat.read_text().split(") ", 1)[1]
        # The process can exit between exists() and the read; the read then
        # fails with ESRCH (ProcessLookupError) rather than ENOENT.
        except (FileNotFoundError, ProcessLookupError):
            return
        if state.startswith("Z"):
            return
        assert time.monotonic() < deadline, "client remained alive after helper exit"
        time.sleep(0.01)


# The client sleeps far longer than any wait below, so it is gone only if the
# harness killed its process group (it outlived the helper on purpose).
LEAKED_CLIENT = """import os, signal, subprocess, sys, time
from pathlib import Path
client = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])
Path(config).write_text(str(client.pid))
"""


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-group cleanup")
def test_warns_and_reaps_a_client_that_outlives_a_successful_helper(helper, tmp_path):
    with pytest.warns(UserWarning, match="client outlived its helper"):
        helper(LEAKED_CLIENT)
    wait_until_gone(int((tmp_path / "client.pid").read_text()))


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-group cleanup")
def test_interrupt_returns_promptly_and_reaps_the_client(helper, tmp_path):
    def interrupt(signum, frame):
        raise InterruptedError("test timeout")

    previous = signal.signal(signal.SIGUSR1, interrupt)
    # The helper signals the test, then would sleep 30 s: the call returns
    # within 10 s only because the harness kills the helper's group.
    body = LEAKED_CLIENT + "os.kill(os.getppid(), signal.SIGUSR1)\ntime.sleep(30)\n"
    start = time.monotonic()
    try:
        with pytest.raises(InterruptedError, match="test timeout"):
            helper(body)
    finally:
        signal.signal(signal.SIGUSR1, previous)
    assert time.monotonic() - start < 10
    wait_until_gone(int((tmp_path / "client.pid").read_text()))


PYTEST_TIMEOUT_TEST = """
import sys
sys.path.insert(0, {common!r})
import test_config

def test_hangs(tmp_path, monkeypatch):
    monkeypatch.setattr(test_config, "_COMMON_DIR", str(tmp_path))
    (tmp_path / "phase.py").write_text(
        "import os, subprocess, sys, time\\n"
        "def run(config, output_dir, artifact_dir, args):\\n"
        "    client = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\\n"
        "    open(config, 'w').write(str(client.pid))\\n"
        "    time.sleep(30)\\n"
    )
    test_config._call_helper_in_subprocess("phase", "run", {pidfile!r}, "", "", [])
"""


@pytest.mark.skipif(sys.platform != "linux", reason="Linux process-group cleanup")
def test_pytest_timeout_under_xdist_reaps_the_client(tmp_path):
    """The path a real CI timeout takes: pytest-timeout's signal in an xdist worker."""
    pidfile = tmp_path / "client.pid"
    (tmp_path / "test_hang.py").write_text(
        PYTEST_TIMEOUT_TEST.format(common=str(Path(__file__).parents[1] / "common"), pidfile=str(pidfile))
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "xdist.plugin", "-n", "1",
         "-p", "pytest_timeout", "--timeout=3", "--timeout-method=signal", str(tmp_path / "test_hang.py")],
        cwd=tmp_path, capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Timeout" in result.stdout, result.stdout
    wait_until_gone(int(pidfile.read_text()))
