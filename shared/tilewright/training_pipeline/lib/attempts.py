# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The processes of a stage attempt, so that none outlives it.

The orchestrator starts every stage with a fresh attempt token in the
stage's environment (`ATTEMPT_VAR`); every process the stage starts inherits
it: sub-stages, benches in their own sessions, cotenants, training workers.
`processes_of(token)` finds them by reading `/proc/<pid>/environ` of the
user's processes, however they were reparented or regrouped, and
`stop(token)` ends them: SIGTERM (a bench stage then stops its benches),
SIGKILL after a grace period.

While a stage runs, `<run dir>/.running.json` (`Running`) names its attempt:
token, pid, process group, command, the run directory and the orchestrator
that started it (pid and start time). The orchestrator removes the entry
when the stage ends and stops whatever still carries the token. An
orchestrator that dies without doing so (SIGKILL, the OOM killer) leaves the
entry; the next invocation on the run directory, which holds its lock,
stops every process of the attempts it names (`stop_survivors`) before it
moves any stage output aside or takes a GPU lease, so a stale attempt never
writes into a new one. It stops only what an orchestrator of this run
directory left behind: an entry of another directory (a copy of a run
directory made while a stage ran carries the file) is dropped untouched, and
one whose orchestrator still runs is reported, never signalled.
"""
from __future__ import annotations

import json
import os
import secrets
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import kfd, procs
from .fs import write_json

# Not a name lib/bench_env.py drops from every bench's environment
# (`is_tilewright_var`): the benches must carry the token too.
ATTEMPT_VAR = "TW_PIPELINE_ATTEMPT"
RUNNING_FILE = ".running.json"
STOP_GRACE_S = 30.0
KILL_WAIT_S = 10.0
POLL_S = 0.2


def new_token() -> str:
    return secrets.token_hex(8)


def environ_with(token: str, env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """`env` (default this process's environment) with the attempt token."""
    out = dict(os.environ if env is None else env)
    out[ATTEMPT_VAR] = token
    return out


def processes_of(token: str, proc_root: Optional[Path] = None) -> List[int]:
    """pids of this user's processes, this one excepted, whose environment
    carries the attempt token `token`."""
    root = Path(kfd.PROC if proc_root is None else proc_root)
    needle = f"{ATTEMPT_VAR}={token}".encode()
    uid, me = os.getuid(), os.getpid()
    out: List[int] = []
    try:
        entries = list(os.scandir(root))
    except OSError:
        return out
    for entry in entries:
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
            data = (root / entry.name / "environ").read_bytes()
        except OSError:
            continue
        if needle in data.split(b"\0"):
            out.append(int(entry.name))
    return sorted(out)


def _signal(pids: Sequence[int], sig: int) -> None:
    for pid in pids:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def _wait_gone(token: str, seconds: float, resend: Optional[int]) -> List[int]:
    """The attempt's processes left after up to `seconds`; a process that
    appears meanwhile is sent `resend` too."""
    sent = set(processes_of(token))
    deadline = time.monotonic() + max(0.0, seconds)
    while True:
        left = processes_of(token)
        if not left or time.monotonic() >= deadline:
            return left
        new = [p for p in left if p not in sent]
        if new and resend is not None:
            _signal(new, resend)
            sent.update(new)
        time.sleep(POLL_S)


def stop(token: str, grace_s: Optional[float] = None) -> Tuple[List[int], List[int]]:
    """Stop every process of the attempt: SIGTERM, SIGKILL after `grace_s`
    (default STOP_GRACE_S) to all but those stuck in a driver call
    (lib/procs.py `stuck_in_driver`), which are left running. Returns (the
    pids found, the pids still alive after SIGKILL)."""
    found = processes_of(token)
    if not found:
        return [], []
    _signal(found, signal.SIGTERM)
    grace = STOP_GRACE_S if grace_s is None else grace_s
    left = _wait_gone(token, grace, signal.SIGTERM)
    if left:
        stuck = set(procs.stuck_in_driver(left))
        _signal([p for p in left if p not in stuck], signal.SIGKILL)
        left = _wait_gone(token, KILL_WAIT_S, signal.SIGKILL)
    return found, left


class Running:
    """`<run dir>/.running.json`: {token: record} of the stage attempts that
    are running. Only the orchestrator holding the run directory's lock
    writes it."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / RUNNING_FILE

    def records(self) -> Dict[str, Dict[str, Any]]:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, recs: Dict[str, Dict[str, Any]]) -> None:
        if recs:
            write_json(self.path, recs)
        else:
            self.path.unlink(missing_ok=True)

    def add(
        self,
        token: str,
        *,
        pid: int,
        cmd: Sequence[str],
        label: str,
    ) -> None:
        recs = self.records()
        try:
            pgid: Optional[int] = os.getpgid(pid)
        except OSError:
            pgid = None
        me = os.getpid()
        recs[token] = {
            "label": label,
            "pid": int(pid),
            "pgid": pgid,
            "run_dir": str(self.run_dir.resolve()),
            "orchestrator_pid": me,
            "orchestrator_start_s": kfd.proc_info(me).start_s,
            "started": datetime.now().isoformat(timespec="seconds"),
            "cmd": [str(c) for c in cmd],
        }
        self._write(recs)

    def remove(self, token: str) -> None:
        recs = self.records()
        if recs.pop(token, None) is not None:
            self._write(recs)


def orchestrator_running(rec: Dict[str, Any]) -> bool:
    """Whether the orchestrator that wrote `rec` still runs, this process
    excepted: a live, non-zombie process with its pid and start time. A
    start time that cannot be compared counts as running."""
    try:
        pid = int(rec["orchestrator_pid"])
    except (KeyError, TypeError, ValueError):
        return False
    info = kfd.proc_info(pid)
    if info.exited or pid == os.getpid():
        return False
    try:
        started = float(rec["orchestrator_start_s"])
    except (KeyError, TypeError, ValueError):
        return True
    return info.start_s is None or started == info.start_s


@dataclass
class Survivors:
    """What `stop_survivors` found in `.running.json`, as (label, detail)
    pairs: `stopped` and `stuck` (pids that do not stop) of attempts an
    orchestrator of this run directory left behind; `elsewhere` (the run
    directory they belong to), dropped untouched; `alive` (the pid of their
    orchestrator, which still runs on this run directory's path), kept."""

    stopped: List[Tuple[str, List[int]]] = field(default_factory=list)
    stuck: List[Tuple[str, List[int]]] = field(default_factory=list)
    elsewhere: List[Tuple[str, str]] = field(default_factory=list)
    alive: List[Tuple[str, int]] = field(default_factory=list)


def stop_survivors(run_dir: Path, grace_s: Optional[float] = None) -> Survivors:
    """Stop the processes of every attempt `<run dir>/.running.json` names
    that an orchestrator of this run directory started and left behind when
    it died, and drop their entries once none is left. The entry of another
    run directory is dropped without a signal: its attempt belongs to the run
    it was copied from."""
    running = Running(run_dir)
    here = Path(run_dir).resolve()
    out = Survivors()
    for token, rec in running.records().items():
        label = str(rec.get("label") or token)
        where = rec.get("run_dir")
        if where is None or Path(str(where)) != here:
            out.elsewhere.append((label, str(where or "an unnamed run directory")))
            running.remove(token)
            continue
        if orchestrator_running(rec):
            out.alive.append((label, int(rec["orchestrator_pid"])))
            continue
        found, left = stop(token, grace_s)
        if found:
            out.stopped.append((label, found))
        if left:
            out.stuck.append((label, left))
        else:
            running.remove(token)
    return out
