# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for test_config._call_helper_in_subprocess process-group reaping.

The run phase is launched via subprocess in its own process group so a
pytest-timeout interrupt can kill the whole tree (the tensilelite-client
grandchild) instead of orphaning it and stalling pytest-xdist shutdown.
"""

import os
import signal
import subprocess

import pytest

import test_config


class _FakeProc:
    def __init__(self, wait_results, poll_result):
        self.pid = 4242
        self.args = ["python", "-c", "..."]
        self._wait_results = list(wait_results)
        self._poll_result = poll_result

    def wait(self):
        result = self._wait_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result

    def poll(self):
        return self._poll_result


def test_reaps_process_group_on_interrupt(monkeypatch):
    # First wait() is interrupted (pytest-timeout); poll() shows the child alive.
    proc = _FakeProc([KeyboardInterrupt(), -9], poll_result=None)
    killed = {}
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(os, "killpg", lambda pid, sig: killed.update(pid=pid, sig=sig))
    with pytest.raises(KeyboardInterrupt):
        test_config._call_helper_in_subprocess("m", "f", "c", "o", "a", [])
    assert killed == {"pid": 4242, "sig": signal.SIGKILL}


def test_no_kill_on_clean_exit(monkeypatch):
    proc = _FakeProc([0], poll_result=0)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)
    monkeypatch.setattr(os, "killpg", lambda *a: pytest.fail("must not kill on clean exit"))
    test_config._call_helper_in_subprocess("m", "f", "c", "o", "a", [])


def test_raises_on_nonzero(monkeypatch):
    proc = _FakeProc([3], poll_result=3)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: proc)
    with pytest.raises(subprocess.CalledProcessError):
        test_config._call_helper_in_subprocess("m", "f", "c", "o", "a", [])
