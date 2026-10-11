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

A sub-stage that a stage runs in its own process group (stage03 under
stage03c and stage03d, stage08, the seed bootstrap and the miss re-bench)
starts with `die_with_parent()`: it gets SIGTERM when the stage dies, also
when the stage dies by a signal it does not handle, and stops its benches.
stage05's training workers call `die_with` from their pool's initializer.

No process stuck in uninterruptible sleep (D state) in every one of
STUCK_SAMPLES readings STUCK_INTERVAL_S apart (`stuck_in_driver`) is sent
SIGKILL: such a process waits in a driver call, typically of a GPU that
stopped answering, so the signal only takes effect once that call returns,
and the exit it then starts can leave the process, and the GPU, stuck
until the host reboots. `kill_group` gives a group with such a process
STUCK_WAIT_S to leave the call, as a call that is only slow does, and
kills it once it has; a group still stuck then is left running
(STUCK_RC). `ChildGroups.stop_all` sends such a group SIGTERM only.
"""
from __future__ import annotations

import contextlib
import ctypes
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from . import ui

STOP_SIGNALS: Tuple[int, ...] = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
# Signals that end a process from outside (an operator, a dying parent, the
# OOM killer); a death by another one, such as SIGSEGV, is its own crash.
KILL_SIGNALS: Tuple[int, ...] = (*STOP_SIGNALS, signal.SIGKILL)
PR_SET_PDEATHSIG = 1
PROC = Path("/proc")
# A process counts as stuck in a driver call when every one of STUCK_SAMPLES
# readings, STUCK_INTERVAL_S apart, finds it in uninterruptible sleep.
STUCK_SAMPLES = 4
STUCK_INTERVAL_S = 0.5
# Seconds `kill_group` waits for a process group with a process stuck in a
# driver call to leave it: a group left running is held as wedged, so a
# call that is only slow, or the teardown of a process that is ending, must
# end within it.
STUCK_WAIT_S = 60.0
# What `kill_group` returns for a process group it left alone because a
# process of it stayed stuck in a driver call for STUCK_WAIT_S: not a return
# code of the process, which has not ended.
STUCK_RC = -1000
# The exit status of a stage that benches and stopped because a device is
# wedged: processes stuck in the GPU driver on two or more GPUs or the
# host, or on every GPU it had. Only a reboot of the host ends that, so no
# attempt is retried on it.
WEDGED_EXIT = 3
WEDGED_TEXT = "device wedged, reboot"


def _load_prctl() -> Optional[Callable[..., int]]:
    try:
        return ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None


_PRCTL = _load_prctl()


def die_with(parent: int, sig: int = signal.SIGTERM) -> None:
    """Have this process sent `sig` when its parent, process `parent`, dies
    (Linux PR_SET_PDEATHSIG), even by SIGKILL; exit at once when that parent
    died already. Nothing where the call is unavailable. For a child that
    takes no `preexec_fn`, such as a process-pool worker: call it from the
    pool's initializer. The kernel sends `sig` when the parent's thread that
    started the child exits, so start it from one that outlives it (the main
    thread)."""
    if _PRCTL is None:
        return
    _PRCTL(PR_SET_PDEATHSIG, int(sig), 0, 0, 0)
    if os.getppid() != parent:
        os._exit(128 + int(sig))


def die_with_parent(sig: int = signal.SIGTERM) -> Optional[Callable[[], None]]:
    """A `preexec_fn` that has the child sent `sig` when this process dies
    (`die_with`); None where PR_SET_PDEATHSIG is unavailable."""
    if _PRCTL is None:
        return None
    parent = os.getpid()

    def preexec() -> None:
        die_with(parent, sig)

    return preexec


def killed_by(rc: int) -> Optional[signal.Signals]:
    """The signal of a return code that reports a death by one of
    KILL_SIGNALS: -N from subprocess, 128 + N from a process that exits on
    it; None for any other return code."""
    sig = -rc if rc < 0 else rc - 128
    return signal.Signals(sig) if sig in KILL_SIGNALS else None


def retry_status(rc: int, bench: bool = False) -> int:
    """The exit status of a stage whose sub-stage ended with `rc`: 2 (a
    retry may cure it) for a death by one of KILL_SIGNALS or, from a
    sub-stage that benches (`bench`), for its exit 2, which reports trouble
    with its GPUs; WEDGED_EXIT for that sub-stage's WEDGED_EXIT (a wedged
    device, which no retry cures); 1 for any other failure."""
    if rc == 0:
        return 0
    if bench and rc == WEDGED_EXIT:
        return WEDGED_EXIT
    return 2 if killed_by(rc) is not None or (bench and rc == 2) else 1


class StageInterrupted(KeyboardInterrupt):
    """SIGINT, SIGTERM or SIGHUP received by the stage."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = int(signum)


class ChildrenStopped(RuntimeError):
    """A bench was about to start after `ChildGroups.stop_all`."""


# ── processes stuck in a driver call ─────────────────────────────────────────


def process_state(pid: int) -> Optional[str]:
    """The state letter of `pid` in /proc (R, S, D, Z, ...); None when it
    has no entry or cannot be read."""
    try:
        text = (PROC / str(int(pid)) / "stat").read_text()
    except (OSError, ValueError):
        return None
    head, _, tail = text.rpartition(")")
    fields = tail.split()
    return fields[0] if head and fields else None


def session_members(sid: int) -> List[int]:
    """The processes of session `sid` (/proc), its leader among them while
    it runs: a bench started in a session of its own and whatever it
    started."""
    out: List[int] = []
    try:
        entries = [e.name for e in os.scandir(PROC) if e.name.isdigit()]
    except OSError:
        return out
    for name in entries:
        try:
            text = (PROC / name / "stat").read_text()
        except OSError:
            continue
        fields = text.rpartition(")")[2].split()
        try:
            if int(fields[3]) == int(sid):
                out.append(int(name))
        except (IndexError, ValueError):
            continue
    return sorted(out)


def descendants(pid: int) -> List[int]:
    """The processes `pid` started and those they started, at any depth
    (/proc parent links), while they run and their parent has not ended."""
    children: Dict[int, List[int]] = {}
    try:
        entries = [e.name for e in os.scandir(PROC) if e.name.isdigit()]
    except OSError:
        return []
    for name in entries:
        try:
            text = (PROC / name / "stat").read_text()
        except OSError:
            continue
        fields = text.rpartition(")")[2].split()
        try:
            children.setdefault(int(fields[1]), []).append(int(name))
        except (IndexError, ValueError):
            continue
    out: Set[int] = set()
    todo = [int(pid)]
    while todo:
        for child in children.get(todo.pop(), []):
            if child not in out:
                out.add(child)
                todo.append(child)
    return sorted(out)


def stuck_in_driver(
    pids: Iterable[int],
    samples: int = STUCK_SAMPLES,
    interval_s: float = STUCK_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
) -> List[int]:
    """The `pids` in uninterruptible sleep (D state) in every one of
    `samples` readings `interval_s` apart; [] at once when the first reading
    finds none."""
    stuck = {int(p) for p in pids if process_state(p) == "D"}
    for _ in range(max(0, int(samples) - 1)):
        if not stuck:
            break
        sleep(interval_s)
        stuck = {p for p in stuck if process_state(p) == "D"}
    return sorted(stuck)


def stuck_members(proc: Any) -> List[int]:
    """The processes of the session `proc` leads (`session_members`, `proc`
    itself without a session) that are stuck in a driver call
    (`stuck_in_driver`); [] for a process that ended or has no pid."""
    pid = getattr(proc, "pid", None)
    if not isinstance(pid, int):
        return []
    try:
        if proc.poll() is not None:
            return []
    except Exception:
        return []
    return stuck_in_driver(session_members(pid) or [pid])


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

    @property
    def stopped(self) -> bool:
        """Whether `stop_all` ran: the stage is stopping."""
        with self._lock:
            return self._stopped

    def stop_all(self) -> int:
        """Refuse new benches and stop every running group: SIGTERM, then
        SIGKILL after `grace_s`, except for a group with a process stuck in
        a driver call (`stuck_members`), which is left running. Returns the
        number of groups signalled."""
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
            stuck = stuck_members(p)
            if stuck:
                ui.err(
                    "stuck",
                    f"bench pid {p.pid}: process(es) {stuck} stuck in a driver "
                    f"call are left running, not killed",
                )
                continue
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
    lead one) and reap it; returns a non-zero return code. A group with a
    process stuck in a driver call (`stuck_members`) gets the signal once no
    process of it is; one that stays stuck for STUCK_WAIT_S gets none and is
    left running: STUCK_RC."""
    deadline = time.monotonic() + STUCK_WAIT_S
    while stuck_members(proc):
        if time.monotonic() >= deadline:
            return STUCK_RC
        time.sleep(STUCK_INTERVAL_S)
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
    probe_contaminated: Optional[Callable[[], bool]] = None,
    probe_abort: Optional[Callable[[], Optional[str]]] = None,
    clock: Callable[[], float] = time.monotonic,
) -> Tuple[int, Optional[str]]:
    """Wait for `proc`; returns (returncode, reason) with reason None for a
    normal exit, "contaminated", "fatal", "stall" or the reason
    `probe_abort()` gave for a kill (`kill_group`; STUCK_RC when a process
    of it stayed stuck in a driver call and is left running).

    `probe_contaminated()` (another process is on the bench's GPU),
    `probe_abort()` returning a reason (the stage's guard stops its benches)
    or `probe_fatal()` true kills at once. Otherwise the process is killed
    when its output (`probe_size()`) has not grown for `stall_s`, and, until
    `probe_progress()` first reports a finished problem, also not before
    `startup_grace_s` after spawn: a fresh process loads kernel modules
    lazily and silently, for longer than the stall timeout when several GPUs
    load at once. `stall_s <= 0` disables the stall and fatal rules."""
    if stall_s <= 0 and probe_contaminated is None and probe_abort is None:
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
        if _probe(probe_contaminated, False):
            return kill_group(proc), "contaminated"
        why = _probe(probe_abort, None)
        if why:
            return kill_group(proc), str(why)
        if stall_s <= 0:
            continue
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
