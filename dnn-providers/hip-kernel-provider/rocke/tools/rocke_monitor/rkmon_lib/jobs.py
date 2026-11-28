# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Detached background jobs (the reviewer model's worker, check scripts, later sanity runs) that outlive the rk
command that starts them.

Each job has a directory `state/jobs/<id>/` containing:
- `cmd.sh`: the script;
- `log`: combined output;
- `pid`;
- `rc`: the exit code, written when the script ends;
- `meta.json`.

A job gets gh.job_env: no GitHub login but the read one, no token and no atlassian CLI settings; one that asks no
model (every kind but MODEL_KINDS) no model or Claude credential either. In the container the PR-code commands of a job
that runs a PR's code (PR_CODE_KINDS) run in PR runners' slots (rkmon_lib/prcode.py), which `kill` frees: every process
in them stopped through the launcher.

A job keeps running in its own session after the rk command that started it ends. Status comes from the pid and the
rc file. A job whose process is gone without an rc was killed ("lost"), for example by memory pressure or the
container's stop.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import shlex
import signal
import subprocess
import time

from . import gh, idle
from .store import minutes_since, now

# The jobs that build or run a PR's code, whose PR-code commands `kill` stops through the launcher.
PR_CODE_KINDS = ("checks", "gate", "batch", "sanity", "bisect", "drift")
# The jobs that ask a model (the reviewer model's worker, the first pass, the drafts and the shadow ones, the offload's
# passes, the code quality pass, a role's run): only they get the model credentials (gh.job_env `model`), which a host's
# model calls read from the environment.
MODEL_KINDS = ("worker", "first_pass", "draft", "shadow", "offload", "quality", "role")

META_KEYS = {
    "id",
    "pid",
    "started",
    "timeout_min",
}  # a job's meta has these keys (start): how `referenced` finds one
JOB_ENV = "RKMON_JOB"  # every job process carries its job's folder, so a stray is known as ours after its leader


def _dir(store, job_id: str) -> pathlib.Path:
    return store.dir / "jobs" / job_id


def start(
    store, kind: str, name: str, script: str, timeout_min: int, env: dict | None = None
) -> dict:
    base = f"{kind}-{name}-{dt.datetime.now(dt.timezone.utc):%Y%m%d%H%M%S}"
    # A second start of the same kind and name in the same second gets a folder of its own (-2, -3, ...): two jobs
    # never share a log, pid, meta or rc.
    job_id, i = base, 1
    while True:
        d = _dir(store, job_id)
        d.parent.mkdir(parents=True, exist_ok=True)
        try:
            d.mkdir()
            break
        except FileExistsError:
            i += 1
            job_id = f"{base}-{i}"
    # The job body runs in its own bash so that `exit` or `set -e` inside it still reaches the rc line; the paths are
    # quoted, so a folder whose path has a space runs as any other.
    (d / "script.sh").write_text(f"#!/bin/bash\n{script}\n")
    (d / "cmd.sh").write_text(
        f"#!/bin/bash\nbash {shlex.quote(str(d / 'script.sh'))}\n"
        f"echo $? > {shlex.quote(str(d / 'rc'))}\n"
    )
    # Every process of the job carries its folder (RKMON_JOB), so rk stop and rk kill find them all, and only them
    # (rkmon_lib/procs.py), even one whose parent has ended.
    with open(d / "log", "wb") as log, open(os.devnull, "rb") as devnull:
        p = subprocess.Popen(
            ["bash", str(d / "cmd.sh")],
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=devnull,
            start_new_session=True,
            env={
                **gh.job_env({**os.environ, **(env or {})}, model=kind in MODEL_KINDS),
                JOB_ENV: str(d),
            },
            close_fds=True,
        )
    (d / "pid").write_text(str(p.pid))
    from . import procs

    meta = {
        "id": job_id,
        "kind": kind,
        "name": name,
        "pid": p.pid,
        "pstart": procs.start_time(p.pid),
        "started": now(),
        "timeout_min": timeout_min,
        "dir": str(d),
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    store.journal("job", f"start {job_id} (pid {p.pid})")
    return meta


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie (exited, not reaped) counts as gone.
    try:
        return pathlib.Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return False


def status(store, job: dict | None) -> str:
    if not job:
        return "none"
    d = _dir(store, job["id"])
    rc = d / "rc"
    if rc.exists():
        txt = rc.read_text().strip()
        return "done" if txt == "0" else f"failed(rc={txt})"
    if alive(job["pid"]):
        from . import procs

        if job.get("pstart") is not None and procs.start_time(job["pid"]) not in (
            None,
            job["pstart"],
        ):
            return "lost"  # the pid belongs to another process now
        return (
            "timeout"
            if minutes_since(job["started"]) > job["timeout_min"]
            else "running"
        )
    return "lost"


def tail(store, job: dict, n: int = 20) -> str:
    p = _dir(store, job["id"]) / "log"
    return (
        "\n".join(p.read_text(errors="replace").splitlines()[-n:]) if p.exists() else ""
    )


KILL_GRACE_S = 10  # SIGTERM, then SIGKILL for what is left after this long


def _procs():
    """(pid, session id) of every live process but this one, from /proc."""
    me = os.getpid()
    for p in pathlib.Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == me:
            continue
        try:
            stat = (p / "stat").read_text()
        except OSError:
            continue
        f = stat[stat.rfind(")") + 2 :].split()
        if len(f) > 3 and f[0] != "Z":
            yield int(p.name), int(f[3])


def _mark(pid: int) -> str | None:
    """The job folder process `pid` carries as JOB_ENV, or None."""
    try:
        env = pathlib.Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return None
    key = JOB_ENV.encode() + b"="
    return next(
        (e[len(key) :].decode(errors="replace") for e in env if e.startswith(key)),
        None,
    )


def _leader(job: dict) -> bool:
    """Is the job's leader (cmd.sh) alive: its pid, and its start time when the job recorded one (a pid reused by
    another process is not the leader)."""
    from . import procs

    pid = int(job["pid"])
    return alive(pid) and (
        job.get("pstart") is None or procs.start_time(pid) == job["pstart"]
    )


def members(store, job: dict) -> list[int]:
    """The live processes of a job: its session (each job runs in its own session, led by its cmd.sh; a `timeout`
    inside it makes a process group of its own, so the leader's group alone misses it), and any process that carries
    the job's folder as JOB_ENV. Once the leader is gone its pid may be another process's, so then only the marked
    processes count."""
    pid, mark = int(job["pid"]), str(_dir(store, job["id"]))
    leader = _leader(job)
    return [q for q, sid in _procs() if (leader and sid == pid) or _mark(q) == mark]


def terminate(
    pids: list[int], grace_s: float = KILL_GRACE_S, sleep=time.sleep
) -> list[int]:
    """SIGTERM each process, then SIGKILL the ones still there after `grace_s`; the pids that needed SIGKILL."""
    for q in pids:
        try:
            os.kill(q, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    end = time.monotonic() + grace_s
    left = [q for q in pids if alive(q)]
    while left and time.monotonic() < end:
        sleep(0.2)
        left = [q for q in left if alive(q)]
    for q in left:
        try:
            os.kill(q, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    return left


def kill(store, job: dict, grace_s: float = KILL_GRACE_S) -> None:
    """Stop a job and everything it started (members): SIGTERM, then SIGKILL after `grace_s`. Read again after each
    round (a process forked while the first round ran), at most three rounds. A pid the system has handed to another
    process since (another start time) is never a member."""
    hard: list[int] = []
    for _ in range(3):
        pids = members(store, job)
        if not pids:
            break
        hard += terminate(pids, grace_s)
    if job.get("kind", "checks") in PR_CODE_KINDS:
        # In the container its PR code ran in PR runners' slots, whose processes this user cannot signal: the slots its
        # processes took are freed (every process stopped by the launcher; a slot kept for its review stays kept).
        from . import prcode

        for line in prcode.kill_job(str(_dir(store, job["id"]))):
            store.journal("job", f"{job['id']}: {line}")
    store.journal(
        "job",
        f"killed {job['id']}"
        + (f" ({len(hard)} process(es) needed SIGKILL)" if hard else ""),
    )


def stop_unfinished(store, job: dict, status_now: str) -> None:
    """A job its owner is done with whose status is not a finish (timeout, lost): kill what is left of it, so it holds
    no tree, no slot and no CPU after its owner lets it go."""
    if status_now in ("timeout", "lost"):
        kill(store, job)


def orphans(store, st: dict) -> list[dict]:
    """Running jobs the state does not name that were started after the state was last saved: the tick or command that
    started them died before it saved (a tick past its time limit, an error), so nothing would ever read, wait for or
    stop them, and the next tick would start the same work again beside them. A job started before the last save that
    the state no longer names was let go on purpose, and is left alone."""
    try:
        saved = store.state_path.stat().st_mtime
    except OSError:
        return []
    known, out = referenced(st), []
    root = store.dir / "jobs"
    for d in root.iterdir() if root.is_dir() else []:
        meta = d / "meta.json"
        if d.name in known or (d / "rc").exists():
            continue
        try:
            if meta.stat().st_mtime <= saved:
                continue
            m = json.loads(meta.read_text())
        except (OSError, ValueError):
            continue
        if _leader(m):
            out.append(m)
    return out


def reap_orphans(store, st: dict) -> list[str]:
    """Kill every orphan (orphans): the next steps start their work again, tracked."""
    lines = []
    for m in orphans(store, st):
        kill(store, m)
        lines.append(
            f"JOB_ORPHAN {m['id']}: started by a tick or command that never saved; killed (its work starts "
            "again, tracked)"
        )
    return lines


def tree_holders(paths: list[str], st: dict, store) -> list[int]:
    """Processes of jobs the state does not name (JOB_ENV) whose working folder or command line is in one of these
    folders: strays a job left that would build in a tree the next job checks out."""
    known = {str(_dir(store, k)) for k in referenced(st)}
    roots = [os.path.realpath(p) for p in paths]
    out = []
    for q, _sid in _procs():
        mark = _mark(q)
        if not mark or mark in known:
            continue
        try:
            cwd = os.path.realpath(os.readlink(f"/proc/{q}/cwd"))
            cmd = (
                pathlib.Path(f"/proc/{q}/cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode(errors="replace")
            )
        except OSError:
            continue
        if any(cwd == r or cwd.startswith(r + "/") or r in cmd for r in roots):
            out.append(q)
    return out


def free_tree(paths: list[str], st: dict, store, what: str) -> list[str]:
    """Kill the strays holding these folders (tree_holders) before a job checks them out again; the lines to print."""
    pids = tree_holders(paths, st, store)
    if not pids:
        return []
    terminate(pids)
    store.journal(
        "job",
        f"{what}: killed {len(pids)} stray process(es) still in {', '.join(paths)}: {pids[:10]}",
    )
    return [
        f"JOB_STRAY {what}: killed {len(pids)} stray process(es) of an earlier job still in its tree"
    ]


# --- the heavy slots ------------------------------------------------------------------------------------------------------
# Heavy local jobs (native builds and GPU tests: review checks, gates, develop sanity, bisects) run at most the scaler's
# current heavy setting at once (rkmon_lib/scaler.py: within 1..config.checks_max(), starting at `checks.limit`; the
# parallel reviews never change it). State `jobs.heavy` holds the jobs that took a slot: one job as itself (as before
# the limit was a setting), several as a list. Only a running holder takes a slot.


def heavy_holders(st: dict) -> list[dict]:
    h = (st.get("jobs") or {}).get("heavy")
    if not h:
        return []
    return [h] if isinstance(h, dict) else [x for x in h if isinstance(x, dict)]


def _set_holders(st: dict, holders: list[dict]) -> None:
    st.setdefault("jobs", {})["heavy"] = (
        (holders[0] if len(holders) == 1 else holders) if holders else None
    )


def heavy_running(st: dict, store) -> list[dict]:
    return [h for h in heavy_holders(st) if status(store, h) == "running"]


def heavy_busy(cfg: dict, st: dict, store) -> dict | None:
    """A running holder when every heavy slot is taken (the scaler's current heavy setting, which starts at
    checks.limit), else None: a heavy job may start. A lower setting stops nothing: the jobs running finish, and none
    starts until fewer run than it allows."""
    from . import scaler

    running = heavy_running(st, store)
    return running[0] if len(running) >= scaler.current(cfg, store, "heavy") else None


def take_heavy(st: dict, job: dict, store=None) -> None:
    """`job` takes a heavy slot. With `store`, holders that no longer run are dropped first."""
    holders = [
        h
        for h in heavy_holders(st)
        if h.get("id") != job.get("id")
        and (store is None or status(store, h) == "running")
    ]
    _set_holders(st, holders + [job])


def release_heavy(st: dict, job: dict | None) -> None:
    """Free the heavy slot `job` holds. A job that ends never frees the slot for another job that took it since
    (else a gate that finishes clears the slot a develop sanity just took, and a second heavy job runs beside it).
    """
    if not job:
        return
    holders = heavy_holders(st)
    kept = [h for h in holders if h.get("id") != job.get("id")]
    if len(kept) != len(holders):
        _set_holders(st, kept)


def referenced(st: dict) -> dict[str, dict]:
    """Every job the state refers to, wherever it is kept (a job's meta: META_KEYS), by id."""
    out: dict[str, dict] = {}
    todo = [st]
    while todo:
        o = todo.pop()
        if isinstance(o, dict):
            if META_KEYS <= o.keys():
                out[str(o["id"])] = o
                continue
            todo.extend(o.values())
        elif isinstance(o, list):
            todo.extend(o)
    return out


def _keep_start(start: dict | None, end: dict) -> dict:
    """The jobs a full tick ends with, each with the status it had when the tick started; one the tick started is
    running. A job that ended during the tick, after its step read it, is still news for the next tick.
    """
    return {k: (start or {}).get(k, "running") for k in end}


@idle.part("jobs", end=True, merge=_keep_start)
def _jobs(cfg: dict, st: dict, store) -> dict:
    """The status of every job the state refers to: a job that ends, fails or runs past its time makes the next tick
    full. A new job kind needs nothing here."""
    return {k: status(store, j) for k, j in sorted(referenced(st).items())}
