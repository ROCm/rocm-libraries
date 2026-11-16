# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Quota-aware process ownership for file-isolated library tests."""

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

__all__ = ["available_cpus", "run_bounded_to_stream", "stop_all"]

_lock = threading.RLock()
_running: set[subprocess.Popen[str]] = set()
_windows_jobs: dict[subprocess.Popen[str], int] = {}
_stopped = False
_spawning = 0


def _is_windows() -> bool:
    return os.name == "nt"


def _ancestors(path: Path, root: Path) -> list[Path]:
    """Return ``path`` through ``root`` without escaping the cgroup mount."""
    result = []
    current = path
    while current == root or root in current.parents:
        result.append(current)
        if current == root:
            break
        current = current.parent
    return result


def _cgroup_mounts(
    cgroup_root: Path, mountinfo: Path
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Return ``(hierarchy root, mount point)`` for v2 and CPU v1."""
    v2 = [(Path("/"), cgroup_root)]
    v1 = [
        (Path("/"), cgroup_root / "cpu"),
        (Path("/"), cgroup_root / "cpu,cpuacct"),
        (Path("/"), cgroup_root / "cpuacct,cpu"),
    ]
    try:
        lines = mountinfo.read_text(encoding="utf-8").splitlines()
    except OSError:
        return v2, v1
    discovered_v2 = []
    discovered_v1 = []
    for line in lines:
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        fields = before.split()
        filesystem = after.split()
        if len(fields) < 5 or len(filesystem) < 3:
            continue
        hierarchy_root = Path(fields[3])
        mount = Path(fields[4])
        kind = filesystem[0]
        options = set(filesystem[2].split(","))
        if kind == "cgroup2":
            discovered_v2.append((hierarchy_root, mount))
        elif kind == "cgroup" and "cpu" in options:
            discovered_v1.append((hierarchy_root, mount))
    return discovered_v2 or v2, discovered_v1 or v1


def _process_cgroup_dirs(
    cgroup_root: Path, proc_cgroup: Path, mountinfo: Path
) -> tuple[list[Path], list[Path]]:
    """Return active cgroup-v2 and CPU-controller v1 directories."""
    v2_mounts, v1_mounts = _cgroup_mounts(cgroup_root, mountinfo)
    v2 = [mount for _root, mount in v2_mounts]
    v1 = [mount for _root, mount in v1_mounts]
    try:
        lines = proc_cgroup.read_text(encoding="utf-8").splitlines()
    except OSError:
        return v2, v1
    for line in lines:
        try:
            _hierarchy, controllers, member = line.split(":", 2)
        except ValueError:
            continue
        membership = Path(member)

        def mounted_dirs(mounts):
            directories = []
            for hierarchy_root, mount in mounts:
                try:
                    relative = membership.relative_to(hierarchy_root)
                except ValueError:
                    continue
                directories.extend(_ancestors(mount / relative, mount))
            return directories

        if not controllers:
            v2 = mounted_dirs(v2_mounts)
        elif "cpu" in controllers.split(","):
            v1 = mounted_dirs(v1_mounts)
    return v2, v1


def available_cpus(
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    mountinfo: Path = Path("/proc/self/mountinfo"),
) -> int:
    """Return the narrowest CPU limit visible through affinity or cgroups."""
    counts = []
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is not None:
        try:
            counts.append(len(getaffinity(0)))
        except OSError:
            pass

    v2_dirs, v1_dirs = _process_cgroup_dirs(cgroup_root, proc_cgroup, mountinfo)
    for directory in v2_dirs:
        try:
            quota_s, period_s = (directory / "cpu.max").read_text().split()[:2]
            if quota_s != "max":
                quota, period = int(quota_s), int(period_s)
                if quota > 0 and period > 0:
                    counts.append(max(1, -(-quota // period)))
        except (OSError, ValueError):
            continue
    for directory in v1_dirs:
        try:
            quota = int((directory / "cpu.cfs_quota_us").read_text().strip())
            period = int((directory / "cpu.cfs_period_us").read_text().strip())
            if quota > 0 and period > 0:
                counts.append(max(1, -(-quota // period)))
        except (OSError, ValueError):
            continue

    counts.append(os.cpu_count() or 4)
    return max(1, min(counts))


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
