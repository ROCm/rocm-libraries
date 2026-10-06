# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Benchmark processes of a stage: process groups, interrupts and the stall
watchdog.

Every bench runs as the leader of its own session (`ChildGroups.popen`), so a
terminal interrupt or the orchestrator's signal reaches the stage and not the
benches. `stop_children_on_exit` turns SIGINT, SIGTERM and SIGHUP into one
`StageInterrupted` in the main thread and, whenever the stage leaves the
block, stops every bench group still running (SIGTERM, then SIGKILL after a
grace period), so no bench outlives its stage.
"""
from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from . import ui

STOP_SIGNALS: Tuple[int, ...] = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


class StageInterrupted(KeyboardInterrupt):
    """SIGINT, SIGTERM or SIGHUP received by the stage."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = int(signum)


class ChildrenStopped(RuntimeError):
    """A bench was about to start after `ChildGroups.stop_all`."""


def _signal_group(proc: Any, sig: int) -> bool:
    """Send `sig` to the process group led by `proc`. Only an unreaped
    process is signalled: its pid, which is also the group id, cannot have
    been reused."""
    if proc.poll() is not None:
        return False
    try:
        os.killpg(proc.pid, sig)
    except (ProcessLookupError, PermissionError):
        return False
    return True


class ChildGroups:
    """The bench processes of one stage, each in its own process group."""

    def __init__(self, grace_s: float = 10.0) -> None:
        self.grace_s = float(grace_s)
        self._lock = threading.Lock()
        self._procs: List[subprocess.Popen] = []
        self._stopped = False

    def popen(self, cmd: Sequence[str], **kwargs: Any) -> subprocess.Popen:
        with self._lock:
            if self._stopped:
                raise ChildrenStopped("the stage is stopping; no bench is started")
            self._procs = [p for p in self._procs if p.returncode is None]
            proc = subprocess.Popen(list(cmd), start_new_session=True, **kwargs)
            self._procs.append(proc)
        return proc

    def stop_all(self) -> int:
        """Refuse new benches and stop every running group: SIGTERM, then
        SIGKILL after `grace_s`. Returns the number of groups signalled."""
        with self._lock:
            self._stopped = True
            procs = list(self._procs)
        live = [p for p in procs if _signal_group(p, signal.SIGTERM)]
        deadline = time.monotonic() + self.grace_s
        for p in live:
            try:
                p.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                pass
        for p in live:
            if _signal_group(p, signal.SIGKILL):
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
        return len(live)


@contextlib.contextmanager
def stop_children_on_exit(children: ChildGroups) -> Iterator[None]:
    """Run the block with SIGINT, SIGTERM and SIGHUP raising one
    `StageInterrupted` in the main thread (later signals are ignored).
    Leaving the block, for any reason, stops every child group; an interrupt
    then leaves as SystemExit(128 + signal number)."""
    fired: List[int] = []

    def handler(signum: int, _frame: Any) -> None:
        if fired:
            return
        fired.append(signum)
        raise StageInterrupted(signum)

    previous: Dict[int, Any] = {}
    if threading.current_thread() is threading.main_thread():
        previous = {s: signal.signal(s, handler) for s in STOP_SIGNALS}
    try:
        yield
    except StageInterrupted as e:
        ui.err(
            "interrupt",
            f"{signal.Signals(e.signum).name}: stopping the benchmark processes",
        )
        raise SystemExit(128 + e.signum) from None
    finally:
        fired.append(0)
        children.stop_all()
        for s, h in previous.items():
            signal.signal(s, h)


def kill_group(proc: Any) -> int:
    """SIGKILL `proc` with its process group (`proc` alone when it does not
    lead one) and reap it; returns a non-zero return code."""
    try:
        grouped = _signal_group(proc, signal.SIGKILL)
    except Exception:
        grouped = False
    if not grouped:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        rc = proc.wait(timeout=30)
    except Exception:
        rc = -9
    return rc if rc not in (None, 0) else -9


# ── stall watchdog ───────────────────────────────────────────────────────────
#
# A bench whose GPU work failed does not reliably exit: it can report the
# error and then stop making progress with its output still open. The
# watchdog therefore watches the log for growth; a healthy process keeps
# writing.


def file_size(path: Optional[Path]) -> int:
    try:
        return path.stat().st_size if path is not None else 0
    except OSError:
        return 0


def poll_interval(stall_s: float) -> float:
    """Watchdog poll period: a tenth of the stall timeout, within [0.2, 15] s."""
    return min(15.0, max(0.2, float(stall_s) / 10.0))


def _probe(fn: Optional[Callable[[], Any]], default: Any) -> Any:
    if fn is None:
        return default
    try:
        return fn()
    except Exception:
        return default


def wait_with_stall_watchdog(
    proc: Any,
    probe_size: Callable[[], int],
    stall_s: float,
    *,
    poll_s: float = 15.0,
    probe_fatal: Optional[Callable[[], bool]] = None,
    startup_grace_s: float = 0.0,
    probe_progress: Optional[Callable[[], bool]] = None,
    clock: Callable[[], float] = time.monotonic,
) -> Tuple[int, Optional[str]]:
    """Wait for `proc`; returns (returncode, reason) with reason None for a
    normal exit, "fatal" or "stall" for a kill (`kill_group`).

    `probe_fatal()` true kills at once. Otherwise the process is killed when
    its output (`probe_size()`) has not grown for `stall_s`, and, until
    `probe_progress()` first reports a finished problem, also not before
    `startup_grace_s` after spawn: a fresh process loads kernel modules
    lazily and silently, for longer than the stall timeout when several GPUs
    load at once. `stall_s <= 0` disables the watchdog."""
    if stall_s <= 0:
        return proc.wait(), None
    started = clock()
    last_change = started
    last_size = _probe(probe_size, -1)
    progressed = False
    while True:
        try:
            return proc.wait(timeout=poll_s), None
        except subprocess.TimeoutExpired:
            pass
        if _probe(probe_fatal, False):
            return kill_group(proc), "fatal"
        now = clock()
        if not progressed and _probe(probe_progress, False):
            progressed = True
        size = _probe(probe_size, last_size)
        if size != last_size:
            last_size, last_change = size, now
            continue
        if now - last_change < stall_s:
            continue
        if not progressed and now - started < startup_grace_s:
            continue
        return kill_group(proc), "stall"
