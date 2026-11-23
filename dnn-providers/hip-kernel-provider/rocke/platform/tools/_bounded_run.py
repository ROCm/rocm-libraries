# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Run a command under a time limit, and when the limit is hit -- or, on Linux,
# when the caller dies -- end everything the command started, not just the
# command. Shared by the tools under this directory that give a compiler a time
# budget.
#
# Not a CLI entry point -- hence the leading underscore, as with _flavor_gate.py.
# `run_bounded` itself runs this file as a script, as the session leader of each
# command on Linux (see `_lead_session`); nothing else should.

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, NoReturn

# Every command `run_bounded` is running, so `stop_all` can end them from
# another thread. `_stopped` refuses new ones after that.
_lock = threading.Lock()
_running: set[subprocess.Popen[str]] = set()
_stopped = False

# Whether a command can be made to die with a caller that is killed outright:
# that rests on the parent-death signal of `prctl(2)`, which only Linux has.
_HAS_PARENT_DEATH_SIGNAL = sys.platform.startswith("linux")
_PR_SET_PDEATHSIG = 1
_PR_SET_DUMPABLE = 4

# What the leader writes to its report pipe: the command started, or it could
# not be started (followed by the errno).
_STARTED = "started"
_START_FAILED = "failed "


def run_bounded(
    argv: list[str], *, timeout: float, cwd: Path | str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run `argv` to completion, capturing text output, within `timeout` seconds.

    Behaves like ``subprocess.run(argv, capture_output=True, text=True,
    check=False, timeout=timeout, cwd=cwd)`` with one difference: on timeout
    every process the command started is killed, then
    ``subprocess.TimeoutExpired`` is raised as usual. A command that cannot be
    started raises the same ``OSError`` as there.

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

    The same isolation keeps the group alive when the caller is killed outright
    (a CTest timeout, an outer `timeout`, SIGKILL) and so never gets to end it.
    On Linux the group's leader is `_lead_session`, which ends the group when
    the caller dies. Elsewhere the command keeps running after such a death.
    """
    if os.name == "nt":
        group = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    else:
        group = {"start_new_session": True}

    with contextlib.closing(_Launch(argv)) as launch:
        with _lock:
            if _stopped:
                raise KeyboardInterrupt("run_bounded: stopped by stop_all()")
            # This thread must also be the one that waits for the command: the
            # leader's death signal fires when the thread that started it ends.
            proc = launch.popen(
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
                    # Timeout, or an interrupt while waiting: either way nothing
                    # may outlive this call. The output is not needed any more,
                    # and an error decoding it must not replace the exception
                    # being raised.
                    _kill_group(proc)
                    with contextlib.suppress(UnicodeDecodeError):
                        proc.communicate()
                    raise
        finally:
            with _lock:
                _running.discard(proc)
        launch.raise_start_failure(proc.returncode, stderr)
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


class _Launch:
    """Starts one `run_bounded` command: under `_lead_session` where the
    parent-death signal exists, directly elsewhere.

    Under the leader it is the leader that starts the command, so a command
    that cannot be started (no such file, not executable) fails there, not
    here. The leader reports the outcome through a pipe, and
    `raise_start_failure` turns a failure into the same ``OSError``
    `subprocess.Popen` would have raised. A leader that exits before it reports
    at all raises ``RuntimeError`` instead of passing for the command's result;
    one killed by a signal (a timeout, `stop_all`) is passed through as before.
    """

    def __init__(self, argv: list[str]) -> None:
        self._argv = argv
        self._report: int | None = None
        self._popened = False

    def popen(self, **options: Any) -> subprocess.Popen[str]:
        """Start the command with `subprocess.Popen` `options`; once only."""
        if self._popened:
            raise RuntimeError("_Launch.popen failed - already started its command")
        self._popened = True
        if not _HAS_PARENT_DEATH_SIGNAL:
            return subprocess.Popen(self._argv, **options)
        self._report, report_write = os.pipe()
        command = [
            sys.executable,
            # No environment, user site or `site` import: the leader needs only
            # the standard library, and starts once per command.
            "-I",
            "-S",
            str(Path(__file__).resolve()),
            str(os.getpid()),
            str(report_write),
            *self._argv,
        ]
        try:
            return subprocess.Popen(command, pass_fds=(report_write,), **options)
        finally:
            # Only the leader may hold the write end, so that reading the pipe
            # once the leader has exited cannot block.
            os.close(report_write)

    def raise_start_failure(self, leader_returncode: int, leader_stderr: str) -> None:
        """Raise if the leader did not start the command. Call only after the
        leader has exited; its exit status and stderr explain a silent death."""
        if self._report is None:
            return
        report = os.read(self._report, 64).decode()
        if report == _STARTED or leader_returncode < 0:
            return
        if report.startswith(_START_FAILED):
            error_number = int(report.removeprefix(_START_FAILED))
            raise OSError(error_number, os.strerror(error_number), self._argv[0])
        raise RuntimeError(
            f"run_bounded failed - the session leader for {self._argv[0]!r} exited "
            f"with status {leader_returncode} before starting it, expected it to "
            f"start the command: {leader_stderr.strip()[-2000:]}"
        )

    def close(self) -> None:
        if self._report is not None:
            os.close(self._report)
            self._report = None


def _lead_session(caller_pid: int, report_fd: int, argv: list[str]) -> NoReturn:
    """Run `argv` in this session, and end the whole session if the caller dies.

    `run_bounded` runs this file as the leader of each command's session on
    Linux. The session exists so that a timeout can end the clang driver and
    its `clang -cc1` child together -- and it also puts them out of reach of
    any signal sent to the caller's own process group. A caller that is killed
    outright then leaves them running, unbounded.

    The kernel's parent-death signal (`PR_SET_PDEATHSIG`) closes the gap, but
    only for the process that asks for it, and a child does not inherit it:
    armed on the clang driver it would end the driver and leave `clang -cc1`
    running. So this leader arms it on itself, catches it, and kills its whole
    process group. It fires when the *thread* that started the leader ends, not
    the whole caller; `run_bounded` holds that thread until the command is
    done, so the two coincide.

    The command's standard streams are the leader's, and the leader exits as
    the command did, with its status or by the same signal. Whether the
    command started is written to `report_fd` (see `_Launch`).
    """
    signal.signal(signal.SIGTERM, _kill_own_group)
    _prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
    if os.getppid() != caller_pid:
        _kill_own_group()  # the caller died before the death signal was armed
    os.set_inheritable(report_fd, False)
    try:
        # Python ignores SIGPIPE and SIGXFSZ; give the command the defaults,
        # as `subprocess.Popen` does.
        pid = os.posix_spawnp(
            argv[0], argv, os.environ, setsigdef=(signal.SIGPIPE, signal.SIGXFSZ)
        )
    except OSError as error:
        if error.errno is None:
            raise  # no errno to report: the caller sees a leader that died
        os.write(report_fd, f"{_START_FAILED}{error.errno}".encode())
        os._exit(127)
    os.write(report_fd, _STARTED.encode())
    os.close(report_fd)
    _, status = os.waitpid(pid, 0)
    _exit_as(os.waitstatus_to_exitcode(status))


def _kill_own_group(*_signal_handler_args: object) -> None:
    """Kill this process's whole group, this process included.

    Does not return: a signal a process sends itself is delivered before
    `kill` returns."""
    os.killpg(os.getpgrp(), signal.SIGKILL)


def _exit_as(exit_code: int) -> NoReturn:
    """Exit with `exit_code`, or die by signal ``-exit_code`` if negative.

    Dying by the command's signal keeps the caller's ``returncode`` negative,
    which the probes read as a toolchain crash."""
    if exit_code >= 0:
        os._exit(exit_code)
    signum = -exit_code
    # No core of the leader's own: the command left one already, if the host
    # keeps cores. Only a nicety, so failing to turn it off must not replace
    # the command's status with the leader's error.
    with contextlib.suppress(OSError):
        _prctl(_PR_SET_DUMPABLE, 0)
    if signum != signal.SIGKILL:  # SIGKILL has no handler to reset
        signal.signal(signum, signal.SIG_DFL)
    os.kill(os.getpid(), signum)
    os._exit(128 + signum)  # only if the signal's default is not to terminate


def _prctl(option: int, value: int) -> None:
    """Call Linux `prctl(2)`, raising ``OSError`` on failure."""
    # Only the leader needs ctypes, and only on Linux; importing it here keeps
    # it off every other platform and out of every caller.
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(option, value, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(
            error_number,
            f"prctl failed - option={option}, value={value}: "
            f"{os.strerror(error_number)}",
        )


if __name__ == "__main__":
    _lead_session(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:])
