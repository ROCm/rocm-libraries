# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import signal
import subprocess
import pytest

from geko.concurrency import utils as cutils


class _Proc:
    def __init__(self, pid=123, poll_seq=None, wait_raises=None):
        self.pid = pid
        self._poll_seq = list(poll_seq or [0])
        self._wait_raises = list(wait_raises or [])
        self.wait_calls = []
        self.terminated = 0
        self.killed = 0

    def poll(self):
        if len(self._poll_seq) > 1:
            return self._poll_seq.pop(0)
        return self._poll_seq[0]

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if self._wait_raises:
            exc = self._wait_raises.pop(0)
            if exc is not None:
                raise exc
        return 0

    def terminate(self):
        self.terminated += 1

    def kill(self):
        self.killed += 1


class _StopEvent:
    def __init__(self, waits):
        self._waits = list(waits)
        self.was_set = False

    def wait(self, timeout=None):
        if self._waits:
            return self._waits.pop(0)
        return False

    def set(self):
        self.was_set = True


def test_parallel_for_wiring(monkeypatch):
    class _Parallel:
        def __init__(self, n_jobs):
            self.n_jobs = n_jobs

        def __call__(self, tasks):
            return [task() for task in tasks]

    monkeypatch.setattr(cutils.joblib, "Parallel", _Parallel)
    monkeypatch.setattr(cutils.joblib, "delayed", lambda fn: lambda el: (lambda: fn(el)))

    out = cutils.parallel_for(lambda x: x * 2, [1, 2, 3], n_jobs=2)
    assert out == [2, 4, 6]


def test_parallel_for_empty_sequence_returns_early(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(cutils.joblib, "Parallel", lambda **_k: (_ for _ in ()).throw(AssertionError("should not be called")))
    out = cutils.parallel_for(lambda x: x, [])
    assert out == []
    assert called["n"] == 0


def test_parallel_for_clamps_workers_to_sequence_length(monkeypatch):
    seen = {}

    class _Parallel:
        def __init__(self, n_jobs):
            seen["n_jobs"] = n_jobs

        def __call__(self, tasks):
            return [task() for task in tasks]

    monkeypatch.setattr(cutils.joblib, "Parallel", _Parallel)
    monkeypatch.setattr(cutils.joblib, "delayed", lambda fn: lambda el: (lambda: fn(el)))
    monkeypatch.setattr(cutils.os, "cpu_count", lambda: 32)

    # n_jobs=64 and cpu_count=32 but only 2 items: workers must clamp to 2
    cutils.parallel_for(lambda x: x, [1, 2], n_jobs=64)
    assert seen["n_jobs"] == 2


def test_terminate_process_tree_windows_timeout_fallback(monkeypatch):
    proc = _Proc(wait_raises=[subprocess.TimeoutExpired("cmd", 1), None])
    called = {}

    def _run(cmd, **kwargs):
        called["cmd"] = cmd
        called["kwargs"] = kwargs

    monkeypatch.setattr(cutils.subprocess, "run", _run)
    cutils._terminate_process_tree_windows(proc, terminate_timeout=1.0)

    assert called["cmd"][0] == "taskkill"
    assert proc.killed == 1
    assert len(proc.wait_calls) == 2


def test_terminate_process_tree_posix_lookup_failure(monkeypatch):
    if os.name == "nt":
        pytest.skip("POSIX-only test")
    proc = _Proc()
    monkeypatch.setattr(cutils.os, "getpgid", lambda _pid: (_ for _ in ()).throw(ProcessLookupError()))
    cutils._terminate_process_tree_posix(proc, proc_name="cfg", terminate_timeout=1.0)
    assert proc.terminated == 0
    assert proc.killed == 0


def test_terminate_process_tree_posix_group_leader_sigterm_then_sigkill(monkeypatch):
    if os.name == "nt":
        pytest.skip("POSIX-only test")
    proc = _Proc(wait_raises=[subprocess.TimeoutExpired("cmd", 1), None])
    monkeypatch.setattr(cutils.os, "getpgid", lambda _pid: proc.pid)

    signals = []

    def _killpg(pgid, sig):
        signals.append((pgid, sig))

    monkeypatch.setattr(cutils.os, "killpg", _killpg)
    cutils._terminate_process_tree_posix(proc, proc_name="cfg", terminate_timeout=1.0)

    assert signals[0][1] == signal.SIGTERM
    assert signals[1][1] == signal.SIGKILL


def test_terminate_process_tree_posix_non_group_leader_timeout_kill(monkeypatch):
    if os.name == "nt":
        pytest.skip("POSIX-only test")
    proc = _Proc(wait_raises=[subprocess.TimeoutExpired("cmd", 1), None])
    monkeypatch.setattr(cutils.os, "getpgid", lambda _pid: proc.pid + 1)
    cutils._terminate_process_tree_posix(proc, proc_name="cfg", terminate_timeout=1.0)
    assert proc.terminated == 1
    assert proc.killed == 1


def test_wait_process_or_stop_continues_until_process_exits(monkeypatch):
    proc = _Proc(poll_seq=[None, None, 0])
    stop_event = _StopEvent([False, False])

    called = {"nt": 0, "posix": 0}
    monkeypatch.setattr(cutils, "_terminate_process_tree_windows", lambda *_a, **_k: called.__setitem__("nt", called["nt"] + 1))
    monkeypatch.setattr(cutils, "_terminate_process_tree_posix", lambda *_a, **_k: called.__setitem__("posix", called["posix"] + 1))

    cutils.wait_process_or_stop(proc, stop_event, proc_name="cfg", poll_interval=0.0, terminate_timeout=0.1)
    assert called == {"nt": 0, "posix": 0}


def test_wait_process_or_stop_terminates_nt(monkeypatch):
    proc = _Proc(poll_seq=[None, None])
    stop_event = _StopEvent([True])

    monkeypatch.setattr(cutils.os, "name", "nt", raising=False)
    called = {"nt": 0}
    monkeypatch.setattr(cutils, "_terminate_process_tree_windows", lambda *_a, **_k: called.__setitem__("nt", called["nt"] + 1))

    cutils.wait_process_or_stop(proc, stop_event, proc_name="cfg", poll_interval=0.0, terminate_timeout=0.1)
    assert called["nt"] == 1


def test_wait_process_or_stop_terminates_posix(monkeypatch):
    proc = _Proc(poll_seq=[None, None])
    stop_event = _StopEvent([True])

    monkeypatch.setattr(cutils.os, "name", "posix", raising=False)
    called = {"posix": 0}
    monkeypatch.setattr(cutils, "_terminate_process_tree_posix", lambda *_a, **_k: called.__setitem__("posix", called["posix"] + 1))

    cutils.wait_process_or_stop(proc, stop_event, proc_name="cfg", poll_interval=0.0, terminate_timeout=0.1)
    assert called["posix"] == 1


def test_install_stop_handlers_noop_off_main_thread(monkeypatch):
    monkeypatch.setattr(cutils, "current_thread", lambda: object())
    monkeypatch.setattr(cutils, "main_thread", lambda: object())
    prev = cutils.install_stop_handlers(_StopEvent([]))
    assert prev == (None, None)


def test_install_and_restore_stop_handlers_main_thread(monkeypatch):
    token = object()
    monkeypatch.setattr(cutils, "current_thread", lambda: token)
    monkeypatch.setattr(cutils, "main_thread", lambda: token)

    prev_calls = {"get": [], "set": []}

    def _getsignal(sig):
        prev_calls["get"].append(sig)
        return f"prev-{sig}"

    handlers = {}

    def _signal(sig, handler):
        prev_calls["set"].append((sig, handler))
        handlers[sig] = handler

    monkeypatch.setattr(cutils.signal, "getsignal", _getsignal)
    monkeypatch.setattr(cutils.signal, "signal", _signal)

    stop_event = _StopEvent([])
    prev = cutils.install_stop_handlers(stop_event)
    assert prev[0] == f"prev-{signal.SIGINT}"

    handlers[signal.SIGINT](signal.SIGINT, None)
    assert stop_event.was_set is True

    cutils.restore_stop_handlers(prev)
    assert any(sig == signal.SIGINT for sig, _h in prev_calls["set"])


def test_restore_stop_handlers_noop_for_none(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(cutils.signal, "signal", lambda *_a, **_k: called.__setitem__("n", called["n"] + 1))
    cutils.restore_stop_handlers((None, None))
    assert called["n"] == 0