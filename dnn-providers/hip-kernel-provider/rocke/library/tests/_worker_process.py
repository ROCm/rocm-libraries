# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Process ownership for file-isolated library tests: each file worker runs in
its own POSIX process group or Windows Job, killed as a whole on timeout."""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from collections.abc import Mapping
from typing import TextIO

__all__ = ["run_bounded_to_stream", "stop_all"]

_lock = threading.RLock()
_running: set[subprocess.Popen[str]] = set()
_windows_jobs: dict[subprocess.Popen[str], int] = {}
_stopped = False
_spawning = 0


def _is_windows() -> bool:
    return os.name == "nt"


def run_bounded_to_stream(
    argv: list[str],
    *,
    timeout: float,
    cwd: Path | str,
    stream: TextIO,
    env: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run within ``timeout`` under a POSIX process group or Windows Job."""
    with _lock:
        if _stopped:
            raise KeyboardInterrupt("file-worker execution has been stopped")
    windows_gate = None
    command = argv
    if _is_windows():
        windows_job = _create_windows_job()
        if windows_job is None:
            raise OSError("could not create a kill-on-close Windows Job Object")
        descriptor, gate_name = tempfile.mkstemp(
            prefix="rocke-worker-", suffix=".start"
        )
        os.close(descriptor)
        windows_gate = Path(gate_name)
        windows_gate.unlink()
        command = [
            sys.executable,
            str(Path(__file__).with_name("_windows_worker_wrapper.py")),
            str(windows_gate),
            *argv,
        ]
        group = {
            "creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
        }
    else:
        windows_job = None
        group = {"start_new_session": True}

    global _spawning
    with _lock:
        if _stopped:
            if windows_job is not None:
                _close_windows_job(windows_job)
            if windows_gate is not None:
                windows_gate.unlink(missing_ok=True)
            raise KeyboardInterrupt("file-worker execution has been stopped")
        _spawning += 1
        process = None
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                **group,
            )
            if windows_job is not None:
                if not _assign_windows_job(windows_job, process):
                    _close_windows_job(windows_job)
                    windows_job = None
                    process.kill()
                    raise OSError("could not assign Windows worker to its Job Object")
                _windows_jobs[process] = windows_job
            _running.add(process)
            if windows_gate is not None and not _stopped:
                _release_windows_gate(windows_gate)
        except BaseException:
            if process is not None:
                _running.discard(process)
                if process in _windows_jobs:
                    _kill_tree(process)
                else:
                    if windows_job is not None:
                        _close_windows_job(windows_job)
                    with contextlib.suppress(ProcessLookupError):
                        process.kill()
            elif windows_job is not None:
                _close_windows_job(windows_job)
            if windows_gate is not None:
                windows_gate.unlink(missing_ok=True)
            raise
        finally:
            _spawning -= 1
        if _stopped:
            _running.discard(process)
            _kill_tree(process)
            raise KeyboardInterrupt("file-worker execution stopped during spawn")
    try:
        try:
            returncode = process.wait(timeout=timeout)
        except BaseException:
            _kill_tree(process)
            with contextlib.suppress(Exception):
                process.wait(timeout=5)
            raise
        else:
            # A test can leave helpers or compiler children behind even when its
            # immediate pytest process exits normally or with an assertion failure.
            _kill_tree(process)
    finally:
        with _lock:
            _running.discard(process)
        if windows_gate is not None:
            windows_gate.unlink(missing_ok=True)
    return subprocess.CompletedProcess(argv, returncode)


def _release_windows_gate(path: Path) -> None:
    path.touch()


def stop_all() -> bool:
    """Kill active trees; return whether a same-thread spawn must finish first."""
    global _stopped
    with _lock:
        _stopped = True
        running = list(_running)
        spawning = _spawning > 0
    for process in running:
        _kill_tree(process)
    return spawning


def _create_windows_job() -> int | None:
    """Create a kill-on-close Job Object, or ``None`` outside/without Windows."""
    if not _is_windows():
        return None
    import ctypes
    from ctypes import wintypes

    class BasicLimit(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_ulonglong)
            for name in (
                "ReadOperationCount",
                "WriteOperationCount",
                "OtherOperationCount",
                "ReadTransferCount",
                "WriteTransferCount",
                "OtherTransferCount",
            )
        ]

    class ExtendedLimit(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimit),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None
    information = ExtendedLimit()
    information.BasicLimitInformation.LimitFlags = 0x00002000
    if not kernel32.SetInformationJobObject(
        job, 9, ctypes.byref(information), ctypes.sizeof(information)
    ):
        kernel32.CloseHandle(job)
        return None
    return int(job)


def _assign_windows_job(job: int, process: subprocess.Popen[str]) -> bool:
    if not _is_windows():
        return False
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    return bool(
        kernel32.AssignProcessToJobObject(
            wintypes.HANDLE(job), wintypes.HANDLE(int(process._handle))
        )
    )


def _close_windows_job(job: int) -> None:
    if _is_windows():
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle(wintypes.HANDLE(job))


def _kill_tree(process: subprocess.Popen[str]) -> None:
    """Kill the owned POSIX process group or Windows Job Object."""
    if _is_windows():
        if (job := _windows_jobs.pop(process, None)) is not None:
            _close_windows_job(job)
        else:
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                    capture_output=True,
                    check=False,
                    timeout=10,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        return
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
