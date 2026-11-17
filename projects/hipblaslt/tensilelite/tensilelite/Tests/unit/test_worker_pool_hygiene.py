# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""shutdownWorkerPool() ends the loky workers and releaseFreedMemory() is a no-op without glibc."""

import ctypes
import os
import time

import pytest

import importlib

# The module itself: tensilelite.Common re-exports joblib's Parallel class under
# the same name, so attribute access would return the class.
Parallel = importlib.import_module("tensilelite.Common.Parallel")
from tensilelite.Common.GlobalParameters import globalParameters


def _pid(_):
    return os.getpid()


def _gone(pid, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not os.path.exists(f"/proc/{pid}"):
            return True
        time.sleep(0.05)
    return False


def test_loky_still_exposes_the_executor_the_shutdown_reaches():
    # shutdownWorkerPool() reads this private name; a joblib upgrade that
    # moves it would otherwise turn the recycling into a silent no-op.
    from joblib.externals.loky import reusable_executor

    assert hasattr(reusable_executor, "_executor")


def test_shutdown_ends_the_workers_and_the_next_call_respawns(monkeypatch):
    monkeypatch.setitem(globalParameters, "CpuThreads", 2)
    monkeypatch.setitem(globalParameters, "ShowProgressBar", False)
    first = set(Parallel.ParallelMap2(_pid, [(i,) for i in range(8)], "pids", procs=2))
    assert first and os.getpid() not in first
    Parallel.shutdownWorkerPool()
    assert all(_gone(pid) for pid in first), f"workers survived shutdown: {first}"
    second = set(Parallel.ParallelMap2(_pid, [(i,) for i in range(8)], "pids", procs=2))
    assert second and not (second & first)
    Parallel.shutdownWorkerPool()
    Parallel.shutdownWorkerPool()  # a second shutdown without a run in between is harmless


def test_shutdown_warns_when_loky_has_no_executor_attribute(monkeypatch, capsys):
    from joblib.externals.loky import reusable_executor

    monkeypatch.delattr(reusable_executor, "_executor")
    Parallel.shutdownWorkerPool()
    assert "worker pool is not recycled" in capsys.readouterr().out


def test_release_freed_memory_is_a_noop_without_glibc(monkeypatch):
    monkeypatch.setattr(Parallel, "_libc", None)

    def no_libc(name):
        raise OSError(f"{name}: cannot open shared object file")

    monkeypatch.setattr(ctypes, "CDLL", no_libc)
    Parallel.releaseFreedMemory()
    assert Parallel._libc is False
    Parallel.releaseFreedMemory()  # still a no-op, without retrying the load


@pytest.mark.skipif(
    not os.path.exists("/lib/x86_64-linux-gnu/libc.so.6")
    and not os.path.exists("/lib64/libc.so.6"),
    reason="glibc only",
)
def test_release_freed_memory_calls_malloc_trim(monkeypatch):
    monkeypatch.setattr(Parallel, "_libc", None)
    Parallel.releaseFreedMemory()
    assert Parallel._libc is not None and Parallel._libc is not False
    Parallel.releaseFreedMemory()
