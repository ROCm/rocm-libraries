# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Run a command under a time limit, and when the limit is hit, end everything
# the command started -- not just the command. Shared by the tools under this
# directory that give a compiler a time budget.
#
# Not a CLI entry point -- hence the leading underscore, as with _hostcaps.py.

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
from pathlib import Path

# Every command `run_bounded` is running, so `stop_all` can end them from
# another thread. `_stopped` refuses new ones after that.
_lock = threading.Lock()
_running: set[subprocess.Popen[str]] = set()
_stopped = False


def run_bounded(
    argv: list[str], *, timeout: float, cwd: Path | str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run `argv` to completion, capturing text output, within `timeout` seconds.

    Behaves like ``subprocess.run(argv, capture_output=True, text=True,
    check=False, timeout=timeout, cwd=cwd)`` with one difference: on timeout
    every process the command started is killed, then
    ``subprocess.TimeoutExpired`` is raised as usual.

    Why `subprocess.run` is not enough: it kills only the process it started.
    The clang driver does the compile in a child (`clang -cc1`), so a compile
    that never finishes -- llvm20 on `raw.ptr.buffer.load.async.lds` at -O0 is
    the live case -- leaves that child behind, orphaned and spinning at full
    CPU for as long as the host stays up. Every arch-domain sweep added one per
    hanging probe.

    The command therefore runs as the leader of its own process group (a new
    session on POSIX, a new process group on Windows), and a timeout ends the
    whole group. A descendant that deliberately leaves the group escapes this;
    clang does not.
    """
    if os.name == "nt":
        group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        group = {"start_new_session": True}

    with _lock:
        if _stopped:
            raise KeyboardInterrupt("run_bounded: stopped by stop_all()")
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd,
            **group,
        )
        _running.add(proc)
    try:
        with proc:
            try:
                stdout, stderr = proc.communicate(timeout=timeout)
            except BaseException:
                # Timeout, or an interrupt while waiting: either way nothing may
                # outlive this call. The output is not needed any more, and an
                # error decoding it must not replace the exception being raised.
                _kill_group(proc)
                with contextlib.suppress(UnicodeDecodeError):
                    proc.communicate()
                raise
    finally:
        with _lock:
            _running.discard(proc)
    return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)


def stop_all() -> None:
    """End every command `run_bounded` is running, and refuse new ones.

    For an interrupt in another thread. A Ctrl-C reaches only the main thread,
    and the commands run in their own sessions, so the terminal's interrupt does
    not reach them either: a sweep running its probes in worker threads would
    otherwise wait out each probe's full timeout before stopping. Call this
    from the main thread's `KeyboardInterrupt` handler, before waiting for the
    workers. A worker that then calls `run_bounded` gets `KeyboardInterrupt`.
    """
    global _stopped
    with _lock:
        _stopped = True
        running = list(_running)
    for proc in running:
        _kill_group(proc)


def _kill_group(proc: subprocess.Popen[str]) -> None:
    """Kill `proc` and every process in the group it leads."""
    if os.name == "nt":
        # A Windows process group cannot be killed as a unit; `taskkill /T`
        # walks the tree from the leader instead.
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
        proc.kill()
        return
    try:
        # The leader of a new session leads its process group, so the group id
        # is the leader's pid. The group outlives the leader as long as any
        # member is alive, which is exactly the case this exists for.
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass  # every member had already exited
