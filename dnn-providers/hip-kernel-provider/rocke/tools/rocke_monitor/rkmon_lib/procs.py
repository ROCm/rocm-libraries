# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The processes the monitor started, and only those (rk stop, rk kill, rk status).

- Jobs (rkmon_lib/jobs.py) run in a session of their own. Their leader's pid and start time are in
  state/jobs/<id>/meta.json, and every process of a job carries RKMON_JOB=<the job's folder> in its environment, even one
  whose parent has died.
- The other long-running monitor processes register themselves while they run: <RUN_DIR>/procs/<pid>.json with their
  pid, start time and what they are (the review driver, `rk serve`, `rk tick`, `rk review-advance`, `rk stop --drain`).

A process is the monitor's only when its pid is alive with the start time recorded for it (a pid the system has
handed to another process since has another start time), or when its environment carries the folder of one of our
jobs; the processes it started count with it. Never the caller itself or its ancestors, never pid 1, never another
user's process: the Ollama server, Claude Code and the operator's shells are never touched. Everything reads /proc,
so a test passes its own `table` and `environ`.

`signals_exit()` is for a process that must clean up when it is ended: while it is entered, SIGTERM and SIGHUP end
the process as an exit, so that its cleanup runs (a drain's flag lifted, a staging copy removed), where Python's own
handling would end it with none.
"""

from __future__ import annotations

import contextlib
import json
import os
import pathlib
import signal
import time

from . import paths, whole_file

PROC = pathlib.Path("/proc")


def stat(pid: int, proc: pathlib.Path = PROC) -> dict | None:
    """/proc/<pid>/stat: {"pid", "ppid", "pgid", "sid", "start" (clock ticks after boot), "state", "comm", "uid"};
    None when the process is gone."""
    try:
        raw = (proc / str(pid) / "stat").read_text()
        uid = (proc / str(pid)).stat().st_uid
    except (OSError, ValueError):
        return None
    comm = raw[raw.find("(") + 1 : raw.rfind(")")]
    rest = raw[raw.rfind(")") + 2 :].split()
    try:
        return {
            "pid": pid,
            "state": rest[0],
            "ppid": int(rest[1]),
            "pgid": int(rest[2]),
            "sid": int(rest[3]),
            "start": int(rest[19]),
            "comm": comm,
            "uid": uid,
        }
    except (IndexError, ValueError):
        return None


def start_time(pid: int) -> int | None:
    s = stat(pid)
    return s["start"] if s else None


def table(proc: pathlib.Path = PROC) -> dict[int, dict]:
    """Every process this user can see: pid -> stat."""
    out = {}
    for d in proc.iterdir() if proc.is_dir() else []:
        if d.name.isdigit():
            s = stat(int(d.name), proc)
            if s and s["state"] != "Z":
                out[s["pid"]] = s
    return out


def environ(pid: int, proc: pathlib.Path = PROC) -> dict:
    """A process's environment at its start ({} when it cannot be read: another user's process, or gone)."""
    try:
        raw = (proc / str(pid) / "environ").read_bytes()
    except OSError:
        return {}
    out = {}
    for item in raw.split(b"\0"):
        k, sep, v = item.partition(b"=")
        if sep:
            out[k.decode("utf-8", "replace")] = v.decode("utf-8", "replace")
    return out


def cmdline(pid: int, proc: pathlib.Path = PROC) -> str:
    try:
        return (
            (proc / str(pid) / "cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode("utf-8", "replace")
            .strip()
        )
    except OSError:
        return ""


# --- the registry -------------------------------------------------------------------------------------------------------


def _dir(run_dir) -> pathlib.Path:
    return pathlib.Path(run_dir) / "procs"


def register(
    run_dir, kind: str, what: str = "", pid: int | None = None
) -> pathlib.Path | None:
    """Record a running monitor process (default: this one) with its start time, written whole (whole_file) at the mode
    of the files the monitor and the writer share (worker/perms.py). None when it cannot be written.
    """
    pid = pid or os.getpid()
    st = start_time(pid)
    if st is None:
        return None
    d = _dir(run_dir)
    try:
        d.mkdir(parents=True, exist_ok=True)
        f = d / f"{pid}.json"
        whole_file.write(
            f,
            json.dumps(
                {
                    "pid": pid,
                    "start": st,
                    "kind": kind,
                    "what": what,
                    "since": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            ),
            paths.worker_module("perms").SHARED_FILE_MODE,
        )
        return f
    except OSError:
        return None


def unregister(run_dir, pid: int | None = None) -> None:
    try:
        (_dir(run_dir) / f"{pid or os.getpid()}.json").unlink(missing_ok=True)
    except OSError:
        pass


def registered(run_dir, procs: dict | None = None, prune: bool = True) -> list[dict]:
    """The registered processes still running as recorded (same pid, same start time); stale entries are removed."""
    d = _dir(run_dir)
    if not d.is_dir():
        return []
    procs = table() if procs is None else procs
    out = []
    for f in sorted(d.glob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            rec = {}
        live = procs.get(rec.get("pid"))
        if live and live["start"] == rec.get("start"):
            out.append(rec)
        elif prune:
            f.unlink(missing_ok=True)
    return out


# --- jobs ---------------------------------------------------------------------------------------------------------------


def job_dirs(state_dir) -> dict[str, dict]:
    """Every job folder of this monitor's state: its path -> meta.json (pid, pstart, id)."""
    out = {}
    root = pathlib.Path(state_dir) / "jobs"
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        try:
            out[str(d)] = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            out[str(d)] = {}
    return out


# --- ours -----------------------------------------------------------------------------------------------------------------


def _ancestors(pid: int, procs: dict) -> set[int]:
    out, cur = set(), pid
    while cur and cur not in out:
        out.add(cur)
        cur = (procs.get(cur) or {}).get("ppid") or 0
    return out


def ours(
    run_dir, state_dir, procs: dict | None = None, env=environ, me: int | None = None
) -> list[dict]:
    """Every process the monitor started that runs now: [{"pid", "kind", "what", "cmd"}], parents before their
    children. Registered processes and job leaders count when their start time matches; job processes by the job
    folder in their environment; and every process descended from one of them. Never `me` (this process) or its
    ancestors, pid 1, or another user's process."""
    procs = table() if procs is None else procs
    me = me or os.getpid()
    uid = os.getuid()
    spare = _ancestors(me, procs) | {1}
    roots: dict[int, tuple[str, str]] = {}
    for rec in registered(run_dir, procs):
        roots[rec["pid"]] = (rec.get("kind") or "monitor", rec.get("what") or "")
    jobs = job_dirs(state_dir)
    for path, meta in jobs.items():
        pid, pstart = meta.get("pid"), meta.get("pstart")
        live = procs.get(pid)
        if (
            live
            and pstart is not None
            and live["start"] == pstart
            and not (pathlib.Path(path) / "rc").exists()
        ):
            roots[pid] = ("job", meta.get("id") or pathlib.Path(path).name)
    if jobs:
        for pid, s in procs.items():
            if pid in roots or s["uid"] != uid:
                continue
            mark = env(pid).get("RKMON_JOB")
            if mark and mark in jobs:
                roots[pid] = (
                    "job",
                    (jobs[mark] or {}).get("id") or pathlib.Path(mark).name,
                )
    children: dict[int, list[int]] = {}
    for pid, s in procs.items():
        children.setdefault(s["ppid"], []).append(pid)
    out, seen = [], set()
    for root in sorted(roots):
        stack = [(root, roots[root][0], roots[root][1], None)]
        while stack:
            pid, kind, what, parent = stack.pop(0)
            s = procs.get(pid)
            if pid in seen or not s or s["uid"] != uid or pid in spare:
                continue
            seen.add(pid)
            out.append(
                {
                    "pid": pid,
                    "kind": kind,
                    "what": what if parent is None else f"child of {parent}",
                    "cmd": cmdline(pid)[:160] or s["comm"],
                }
            )
            # A child that is a root of its own (the review driver the orchestrator started) is listed as itself,
            # under its own kind, not as a child of its parent.
            stack += [
                (c, kind, what, pid)
                for c in sorted(children.get(pid, []))
                if c not in roots
            ]
    return out


def send(entries: list[dict], sig: int, kill=os.kill) -> list[dict]:
    """Signal each process; the ones the signal reached."""
    done = []
    for e in entries:
        try:
            kill(e["pid"], sig)
            done.append(e)
        except (ProcessLookupError, PermissionError):
            pass
    return done


def wait_gone(pids, timeout: float, sleep=time.sleep, alive=None) -> list[int]:
    """Wait up to `timeout` seconds for the processes to end; the ones still there."""
    alive = alive or (lambda p: (stat(p) or {}).get("state") not in (None, "Z"))
    left = [p for p in pids if alive(p)]
    end = time.monotonic() + timeout
    while left and time.monotonic() < end:
        sleep(0.2)
        left = [p for p in left if alive(p)]
    return left


def terminate(
    entries: list[dict], grace: float = 10.0, kill=os.kill, sleep=time.sleep, alive=None
) -> list[dict]:
    """SIGTERM, then SIGKILL for whatever is still there after `grace` seconds; the processes signalled."""
    done = send(entries, signal.SIGTERM, kill)
    left = set(wait_gone([e["pid"] for e in done], grace, sleep, alive))
    send([e for e in done if e["pid"] in left], signal.SIGKILL, kill)
    return done


@contextlib.contextmanager
def signals_exit():
    """SIGTERM and SIGHUP end this process as an exit while entered: see the module's docstring."""

    def bye(signum, _frame):
        raise SystemExit(128 + signum)

    old = {}
    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            old[sig] = signal.signal(sig, bye)
        except ValueError:  # not the main thread (a test): nothing to install
            pass
    try:
        yield
    finally:
        for sig, h in old.items():
            signal.signal(sig, h)
