# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Stopping the monitor: pausing it, stopping the active monitor, or killing every monitor
process are told apart. Four levels, each its own command (the one table of them: docs/restart.md, "The four stop
levels"):

- Pause, `rk pause` (`rk resume` ends it): the tick makes no decisions and starts nothing new; running jobs finish.
- Stop after the work in flight, `rk stop --drain` (`rk start` resumes): first waits as `rk drain` does
  (rkmon_lib/drain.py): nothing new starts, and the work in flight finishes, for a bounded time; then stops as Stop
  does.
- Stop, `rk stop` (`rk start` resumes): stops the running pieces now: the review driver, the parallel reviews' drafts,
  the gate, sanity and check jobs, the admin dashboard server.
- Kill, `rk kill` (`rk start` resumes): kills every process the monitor started, at once, and prints what it killed;
  never a process it did not start (the Ollama server, Claude Code, the operator's shells).

`rk status` shows which state the monitor is in. A stop or a kill is recorded in the state (`stopped`, with the boot it
happened in): while it lasts, the tick, the review driver and new jobs do nothing, and the guarded GitHub writes are
refused, until `rk start`; a reboot ends it (rkmon_lib/store.py stopped). The Claude side (the Claude Code
loop) is stopped from Claude Code: `/rocke-monitor off` ends the loop and runs `rk pause`.

Every outward GitHub write is one `rk` command that keeps the write lock (`write_lock`) from before its first write to
after its ledger entry. Stop and kill take that lock first, waiting a short, bounded time, so no write is cut in half;
a write still in flight after the wait is left alone to finish (its process is never one of the monitor's own), and
the lock keeps a new one from starting meanwhile. The processes come from rkmon_lib/procs.py; in the container, the
PR runners' processes, which this user cannot signal, are stopped by the launcher as each runner (rkmon_lib/prcode.py
stop_all).

A tick at once, `rk tick-now` (`ask_tick`): a request in the state folder (TICK_NOW), which the tick loop that runs
here takes (`take_tick`) within seconds, and then starts a tick through its own path, so its lines go where every
tick's go. The shell tick loop takes it through `rk tick-wait` (`wait_tick`), which marks that loop for rk tick-now.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

from . import config, jobs, outward, paths, prcode, procs, review, whole_file
from . import drain as _drain
from . import store as _store
from .store import draining, flock_within, halted, now, stopped

WRITE_LOCK = "write.lock"
# How long an outward write command waits for the write lock (another write, or rk stop or rk kill), as the writer
# waits too: rkmon.py run_command.
WRITE_WAIT_S = 120
STATE_WAIT_S = (
    120  # the most a stop waits for a tick in progress to finish before it stops it
)
GRACE_S = 10  # SIGTERM, then SIGKILL after this
DRAIN_WHY = (
    "rk stop --drain"  # the reason rk stop --drain gives its drain flag (drain.wait)
)
# What the container says as it starts, in its log: each line the entry point says there (docker/entrypoint.sh, say)
# comes after its time and ENTRY_POINT_SAYS, rk start's lines among them, and rk start's last line begins with
# CLAUDE_SIDE (start), as each line on the Claude side does. The host's start reads that log up to that line
# (docker/rkmon, Wrapper.start_said); the entry point, shell, keeps its own copy of ENTRY_POINT_SAYS, which a test holds
# to this one.
ENTRY_POINT_SAYS = "rkmon-container: "
CLAUDE_SIDE = "The Claude side: "
# The commands that send and run whole in one process (rkmon_lib/outward.py: gate OWN): each takes the write lock from
# before its first write to after its ledger entry. The writer's sends for another process (gate GATED) take none:
# the command that asks for one may have it already (rk teams-breakage), and the writer would wait on it. A stop
# refuses both, but a protective one (PROTECTIVE: it only takes auto-merge off, so it goes even then).
WRITES = outward.names(gate=outward.OWN)
PROTECTIVE = outward.names(protective=True)
# Commands that start work: refused while stopped (each says so and exits 0). Remote GPU jobs already in a cluster's
# queue are left to finish (a stop never waits on the network); `rk gpu-cancel` gives up on them.
STARTERS = (
    "review-start",
    "review-recheck",
    "gate",
    "sanity",
    "bisect",
    "queue-advance",
    "guard-advance",
    "gpu-run",
    "gpu-poll",
)


def run_dir(cfg: dict) -> str:
    return paths.get(cfg, "run")


# --- the write lock -----------------------------------------------------------------------------------------------------


@contextlib.contextmanager
def write_lock(state_dir: pathlib.Path, timeout: float | None = None):
    """The outward-write lock (state/write.lock). Without `timeout` it waits as long as it takes; with one it yields
    False after that many seconds without the lock (the holder's pid stays in the file) instead of raising.
    """
    path = pathlib.Path(state_dir) / WRITE_LOCK
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o664)
    held = False
    try:
        try:
            if timeout is None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                flock_within(fd, timeout, "the write lock")
            held = True
        except TimeoutError:
            held = False
        if held:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()} {procs.start_time(os.getpid())}\n".encode())
        yield held
    finally:
        if held:
            os.ftruncate(fd, 0)
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def write_lock_holder(state_dir: pathlib.Path) -> int | None:
    """The pid that has the write lock (an outward write in flight), or None."""
    path = pathlib.Path(state_dir) / WRITE_LOCK
    try:
        pid, start = path.read_text().split()[:2]
        pid = int(pid)
    except (OSError, ValueError):
        return None
    return pid if str(procs.start_time(pid)) == start else None


# --- the state ----------------------------------------------------------------------------------------------------------


def mark_stopped(st: dict, how: str) -> None:
    st["stopped"] = {"at": now(), "how": how, "boot": _store.boot_id()}
    st.pop("draining", None)


def cleanup(cfg: dict, st: dict, store) -> list[str]:
    """After the jobs are gone, the state says so, without counting anything as a failure: each review in flight is
    recorded as interrupted and starts again at its head after `rk start` (review.interrupt_pipe; a draft awaiting
    verification stays), a gate or batch gate is dropped (re-gated later), a develop sanity run goes back to pending, a
    bisect is dropped (run `rk bisect` again), every heavy slot is free.
    """
    out = []
    for p in list(review.pipes(st)):
        out.append(review.interrupt_pipe(st, store, p))
    for k in list((st.get("rechecks") or {})):
        del st["rechecks"][k]
        out.append(f"checks re-run of #{k} stopped")
    qs = st.setdefault("queue_state", {})
    if qs.get("gate"):
        out.append(
            f"gate on #{qs['gate']['pr']} stopped: it is gated again after rk start"
        )
        qs["gate"] = None
    if qs.get("batch_gate"):
        out.append(
            f"batch gate {' -> '.join('#%d' % n for n in qs['batch_gate']['members'])} stopped"
        )
        qs["batch_gate"] = None
    g = st.setdefault("guard", {})
    if g.get("running"):
        sha = g["running"]["sha"]
        pend = st["develop"].setdefault("pending_sanity", [])
        if sha not in pend:
            pend.append(sha)
        out.append(f"develop sanity on {sha[:10]} stopped: pending again")
        g["running"] = None
    if g.get("bisect"):
        out.append(
            f"bisect of {g['bisect'].get('incident')} stopped: run rk bisect again"
        )
        g["bisect"] = None
    js = st.setdefault("jobs", {})
    js["heavy"] = None
    if js.get("drift"):
        js["drift"] = None
        out.append("drift check stopped")
    return out


def state_line(st: dict) -> str:
    """The monitor's state in one line: on or off, and stopped, draining or running."""
    on = "ON" if st.get("enabled") else "OFF (paused until rk resume)"
    s, d = stopped(st), draining(st)
    if s:
        return f"{on}; STOPPED since {s.get('at')} by rk {s.get('how') or 'stop'} (rk start resumes)"
    if d:
        stale = _store.drain_stale(d)
        if stale:  # new work starts as usual; the next tick clears it
            return (
                f"{on}; DRAIN STALE since {d.get('since')} ({d.get('why') or 'rk stop --drain'}): {stale}; new work "
                "starts as usual, and the next tick clears it"
            )
        return f"{on}; DRAINING since {d.get('since')} ({d.get('why') or 'rk stop --drain'}: nothing new starts)"
    return f"{on}; running" if st.get("enabled") else on


def state_summary(cfg: dict) -> str:
    """The stop state for the setup summary, read without the lock (no state yet: not started)."""
    try:
        st = json.loads(
            (pathlib.Path(config.state_dir()) / "state.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError):
        return "not started yet (rk init, then rk resume)"
    return state_line(st)


# --- the processes ------------------------------------------------------------------------------------------------------


def processes(
    cfg: dict, store, table=None, env=procs.environ, me: int | None = None
) -> list[dict]:
    return procs.ours(run_dir(cfg), store.dir, table, env, me)


# What `rk orchestrate` registers its process as while its Claude session runs (rkmon.py REGISTERED); its --check
# and --running only ask, and register nothing.
ORCHESTRATOR = "orchestrator"


def session_pids(cfg: dict, table=None) -> list[int]:
    """The pids of the Claude sessions (rk orchestrate) that run here now, as registered; [] for none, or when the
    run folder is not known. Nothing is removed (no pruning: a question). A session reads its key and its settings
    as it starts: one that runs takes what changed at its next start, one that has not started yet as it starts.
    """
    try:
        run = run_dir(cfg)
    except paths.PathsError:
        return []
    return [
        r["pid"]
        for r in procs.registered(run, table, prune=False)
        if r.get("kind") == ORCHESTRATOR
    ]


def describe(entries: list[dict]) -> list[str]:
    names = {
        "job": "job",
        "driver": "review driver",
        "serve": "admin dashboard server",
        "tick": "tick",
        "review-advance": "review advance",
        "stop": "rk stop --drain",
        _drain.KIND: "rk drain",
    }
    return [
        f"pid {e['pid']} ({names.get(e['kind'], e['kind'])}{': ' + e['what'] if e.get('what') else ''}): "
        f"{e['cmd']}"
        for e in entries
    ]


# --- a tick at once -----------------------------------------------------------------------------------------------------

# rk tick-now's request: a file in the state folder with the time it was asked. The tick loop that runs here takes it:
# the orchestrator (rk orchestrate) at its next look, tools/tick_loop.py, which finds the file in `rk loop-config`, or
# the shell tick loop through `rk tick-wait`.
TICK_NOW = "tick_now"
# The shell tick loop, docker/entrypoint.sh's tick loop that runs without the orchestrator, waits between its ticks
# with `rk tick-wait --loop PID` (wait_tick), which marks the loop here: its pid, in the file `rk loop-config` names
# (SHELL_LOOP_PID). While that pid runs, rk tick-now names the loop as a taker.
SHELL_LOOP_PID = "shell_tick_loop.pid"
TICK_LOOK_S = 2  # rk tick-wait tries to take the request this often


def shell_loop_name(pid: int | None = None) -> str:
    """The shell tick loop as rk tick-now names it, with its pid when one is given."""
    return "the tick loop that runs without the orchestrator (docker/entrypoint.sh" + (
        f", pid {pid})" if pid else ")"
    )


def shell_loop(cfg: dict, store) -> int | None:
    """The pid of the shell tick loop while it takes rk tick-now's request: the pid its mark (SHELL_LOOP_PID) names
    runs, and no orchestrator session runs here (one that runs takes the request itself, as a tick the loop ran would
    not reach the session). None otherwise."""
    try:
        pid = int((store.dir / SHELL_LOOP_PID).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    if pid <= 0 or not jobs.alive(pid) or session_pids(cfg):
        return None
    return pid


def tick_loops(cfg: dict, store, table=None) -> list[str]:
    """The tick loops that run here now, each named as rk tick-now and rk status name it: the orchestrator (rk
    orchestrate), each session that runs; a host install's tools/tick_loop.py (idle.loop_pid); the container's shell
    tick loop, while no orchestrator runs (shell_loop). [] when none runs."""
    from . import idle  # a call-time import, as status_lines makes it

    sessions = session_pids(cfg, table)
    out = [f"the orchestrator (rk orchestrate, pid {pid})" for pid in sessions]
    loop = idle.loop_pid(store)
    if loop is not None:
        out.append("tools/tick_loop.py" + (f" (pid {loop})" if loop else ""))
    shell = None if sessions else shell_loop(cfg, store)
    if shell:
        out.append(shell_loop_name(shell))
    return out


def ask_tick(cfg: dict, store) -> int:
    """`rk tick-now`: the request written, and who takes it, in outcome words. ASKED, exit 0: a tick loop runs here
    (tick_loops: the orchestrator, tools/tick_loop.py or the shell tick loop) and takes it within seconds;
    WAITING, exit 3: none of them runs now, and the next of them to start takes it at once; REFUSED, exit 4: the
    monitor is stopped (rk start resumes it), and nothing is asked; ERROR, exit 1: the request could not be written.
    """
    try:
        st = store.load()
    except ValueError:
        st = {}
    if stopped(st):
        print(f"REFUSED: {halted(st)}; no tick asked")
        return 4
    path = store.dir / TICK_NOW
    try:
        earlier = path.read_text(encoding="utf-8").strip()
    except OSError:
        earlier = ""
    at = now()
    try:
        whole_file.write(path, at + "\n", paths.worker_module("perms").SHARED_FILE_MODE)
    except OSError as exc:
        print(f"ERROR: {path} could not be written ({exc}); no tick asked")
        return 1
    takers = tick_loops(cfg, store)
    asked = f"a tick at once ({at}" + (
        f"; the request of {earlier} was not taken yet)" if earlier else ")"
    )
    if not takers:
        print(
            f"WAITING: {asked}: none of the orchestrator (rk orchestrate), tools/tick_loop.py and {shell_loop_name()} "
            "runs here now; the next of them to start takes the request at once"
        )
        return 3
    print(
        f"ASKED: {asked}: {' or '.join(takers)} takes the request within seconds and starts the tick (after the one in "
        "progress, if any)"
    )
    return 0


def take_tick(state_dir: pathlib.Path) -> bool:
    """A tick loop's side of rk tick-now: True when a request waited in STATE_DIR. Taking it removes it, so one loop
    takes it, once."""
    try:
        (state_dir / TICK_NOW).unlink()
    except OSError:
        return False
    return True


def wait_tick(
    cfg: dict, store, seconds: float, loop: int, sleep=time.sleep, clock=time.monotonic
) -> int:
    """`rk tick-wait --loop PID SECONDS`: the shell tick loop's wait between its ticks, cut short by rk tick-now's
    request. It marks the loop (PID, in SHELL_LOOP_PID), then every TICK_LOOK_S seconds takes the request (take_tick)
    while the loop takes it (shell_loop names PID, as rk tick-now does). TAKEN, exit 0, once it took one: the loop
    starts a tick at once. Exit 1, nothing printed, once SECONDS went by with none taken; a request it cannot take (a
    folder at its path) leaves the wait as it is. A mark that cannot be written is an ERROR line, and the wait goes
    on: the loop takes the request only while the mark there names it.
    """
    mark = store.dir / SHELL_LOOP_PID
    try:
        whole_file.write(
            mark, f"{loop}\n", paths.worker_module("perms").SHARED_FILE_MODE
        )
    except OSError as exc:
        print(
            f"ERROR: {mark} could not be written ({exc}); the loop takes rk tick-now's request only while the mark "
            "there names it"
        )
    request = store.dir / TICK_NOW
    end = clock() + seconds
    while True:
        if request.exists() and shell_loop(cfg, store) == loop:
            try:
                asked = request.read_text(encoding="utf-8").strip()
            except OSError:
                asked = ""
            if take_tick(store.dir):
                print(
                    f"TAKEN at {now()}: rk tick-now's request"
                    + (f" of {asked}" if asked else "")
                    + "; the loop starts a tick at once"
                )
                return 0
        left = end - clock()
        if left <= 0:
            return 1
        sleep(min(TICK_LOOK_S, left))


# --- the levels -----------------------------------------------------------------------------------------------------------


def stop(
    cfg: dict,
    store,
    drain: bool = False,
    timeout_minutes: float | None = None,
    write_wait: float = 30,
    sleep=time.sleep,
    out=print,
    kill=os.kill,
    table=None,
    env=procs.environ,
    grace: float = GRACE_S,
) -> int:
    """`rk stop [--drain [--timeout-minutes N]]`: stop the running pieces now; with drain, first the wait `rk drain`
    makes (drain.wait, its lines said: nothing new starts, and the work in flight finishes, the reviews advancing
    meanwhile), with the limit and the looks drain.limits gives, as rk drain's (`timeout_minutes` minutes when given),
    after which what is still in flight is stopped. The drain stays in force until the stop records itself, which
    lifts it in the same write (mark_stopped), so that nothing starts in between. REFUSED, exit 4, nothing stopped,
    while another drain is in force; UNDRAINED, exit 1, nothing stopped, when rk start lifts the drain meanwhile. rk
    drain is left alone: it sees the stop at its next look and ends, nothing being left to drain
    (rkmon_lib/drain.py)."""
    if drain:
        limit, poll_s = _drain.limits(cfg, timeout_minutes)
        try:
            waited = _drain.wait(
                cfg,
                store,
                DRAIN_WHY,
                limit,
                poll_s,
                keep=True,
                say=out,
                sleep=sleep,
            )
        except _drain.Busy as exc:
            out(f"REFUSED: {exc}; nothing stopped")
            return 4
        if waited.lifted:
            out(
                f"{_drain.UNDRAINED}: rk start lifted the drain meanwhile; nothing stopped"
            )
            return 1
        out(
            _drain.outcome_line(waited)
            + (
                ""
                if waited.drained
                else "; what is left is stopped now, and the reviews start again at their heads after rk start"
            )
        )
    with write_lock(store.dir, write_wait) as held:
        if not held:
            out(
                f"an outward write is still in flight after {write_wait:.0f} s (pid {write_lock_holder(store.dir)}): "
                "it is left alone to finish"
            )
        spare = {write_lock_holder(store.dir)} if not held else set()
        try:
            with store.locked(timeout=STATE_WAIT_S):
                st = store.load()
                mark_stopped(st, "stop")
                store.save(st)
        except (
            TimeoutError
        ):  # a tick, an advance or the review driver still has it: stop it, then mark
            busy = [
                e
                for e in processes(cfg, store, table, env)
                if e["kind"] in ("tick", "review-advance", "driver")
                and e["pid"] not in spare
            ]
            procs.terminate(busy, grace, kill, sleep)
            try:
                with store.locked(timeout=STATE_WAIT_S):
                    st = store.load()
                    mark_stopped(st, "stop")
                    store.save(st)
            except TimeoutError:
                kept = (
                    "; the drain stays in force, so nothing new starts (rk start lifts it)"
                    if drain
                    else ""
                )
                out(
                    f"NOT STOPPED: the state lock is still in use after {2 * STATE_WAIT_S} s (by another rk stop, or "
                    "by a process this stop cannot see): the monitor is not marked stopped, and nothing but its tick, "
                    f"review advance and review driver was ended{kept}; run rk stop again, or rk kill"
                )
                return 1
        store.journal("mode", "STOPPED (rk stop)")
        # rk drain sees the stop at its next look, and ends as it says (DRAINED: nothing is left to drain).
        victims = [
            e
            for e in processes(cfg, store, table, env)
            if e["pid"] not in spare and e["kind"] != _drain.KIND
        ]
        done = procs.terminate(victims, grace, kill, sleep)
        pr_lines = prcode.stop_all()
        try:
            with store.locked(timeout=STATE_WAIT_S):
                st = store.load()
                notes = cleanup(cfg, st, store)
                store.save(st)
        except TimeoutError as exc:
            notes = [f"the state was not tidied: {exc} (run rk stop again)"]
    for line in describe(done):
        out(f"STOPPED {line}")
    for line in pr_lines:
        out(line)
    for line in notes:
        out(f"  {line}")
    out(
        f"STOPPED: {len(done)} process(es); nothing new starts until rk start. {_claude_side(cfg, done)}"
    )
    return 0


def _claude_side(cfg: dict, done: list[dict]) -> str:
    """What rk stop says of the Claude side. In the container, the orchestrator's session (rk orchestrate): ended by
    the stop (DONE, the processes it ended), or left idle because it ran the stop itself (a stop spares the processes
    it runs under: procs.ours); either way back after rk start, which the host's start runs in the container too, as
    the entrypoint keeps the session down until then. On a host, Claude Code's loop, which /rocke-monitor off ends.
    """
    if os.environ.get("RKMON_CONTAINER") != "1":
        return "The Claude side is stopped from Claude Code (/rocke-monitor off ends its loop)."
    # a call-time import: doctor is a large module, read here only for the host's command
    from .doctor import wrapper

    back = f"rk start, or {wrapper()} start on the host, brings it back"
    if any(e["kind"] == ORCHESTRATOR for e in done):
        return f"{CLAUDE_SIDE}this stop ended the orchestrator's session (rk orchestrate); {back}."
    if session_pids(cfg):
        return f"{CLAUDE_SIDE}the orchestrator's session ran this stop itself and is left idle; {back}."
    return f"{CLAUDE_SIDE}no orchestrator session ran."


def kill_all(
    cfg: dict,
    store,
    wait: float = 30,
    out=print,
    kill=os.kill,
    table=None,
    env=procs.environ,
    sleep=time.sleep,
) -> int:
    """`rk kill`: SIGKILL every process the monitor started, at once, after waiting up to `wait` seconds for an
    outward write in flight (which is then left alone if it still runs); print what it killed.
    """
    with write_lock(store.dir, wait) as held:
        holder = write_lock_holder(store.dir) if not held else None
        if holder:
            out(
                f"an outward write is still in flight after {wait:.0f} s (pid {holder}): it is left alone to finish"
            )
        victims = [e for e in processes(cfg, store, table, env) if e["pid"] != holder]
        done = procs.send(victims, signal.SIGKILL, kill)
        for line in describe(done):
            out(f"KILLED {line}")
        for line in prcode.stop_all(now=True):
            out(line)
        try:
            with store.locked(timeout=STATE_WAIT_S):
                st = store.load()
                mark_stopped(st, "kill")
                notes = cleanup(cfg, st, store)
                store.save(st)
            store.journal("mode", f"KILLED {len(done)} process(es) (rk kill)")
        except TimeoutError as exc:
            notes = [f"the state was not updated: {exc} (run rk kill again)"]
    for line in notes:
        out(f"  {line}")
    out(
        f"KILLED {len(done)} process(es) the monitor started"
        + ("" if done else " (none was running)")
        + ". Nothing else was touched: not the Ollama server, Claude Code or your shells. rk start resumes."
    )
    return 0


def start(
    cfg: dict, store, run=subprocess.run, out=print, popen=subprocess.Popen
) -> int:
    """`rk start`: clear a stop or a drain (the state saved only then: a home with neither gets no state file), record
    your confirmed settings it loaded as applied, the work volume's one-time move made first when one is due
    (rkmon_lib/applied.py), then start the admin dashboard.
    """
    # a call-time import: only the start records your confirmed settings as applied
    from . import applied

    with store.locked(wait=True):
        st = store.load()
        was = st.get("stopped")
        if "stopped" in st or "draining" in st:
            st.pop("stopped", None)
            st.pop("draining", None)
            store.save(st)
    store.journal(
        "mode",
        "STARTED (rk start)"
        + (f" after rk {was.get('how')} at {was.get('at')}" if was else ""),
    )
    out(
        "STARTED: the monitor runs again"
        + (
            f" (it was stopped by rk {was.get('how')} at {was.get('at')})"
            if was
            else ""
        )
    )
    for line in applied.at_start():
        out(line)
    if not runs_web(cfg):
        out("no web server: features.dashboards.admin is off")
    else:
        rkweb = paths.PACKAGE / "tools" / "rkweb.py"
        p = run(
            [sys.executable, "-B", str(rkweb)],
            capture_output=True,
            text=True,
            timeout=60,
            env=dict(os.environ, RKMON_HOME=str(config.home())),
        )
        out((p.stdout or p.stderr or "").strip() or f"{rkweb}: exit {p.returncode}")
    if os.environ.get("RKMON_CONTAINER") == "1":
        out(
            f"{CLAUDE_SIDE}the container's orchestrator (rk orchestrate) sends /rocke-monitor resume when it starts."
        )
    else:
        out(
            f"{CLAUDE_SIDE}in Claude Code, run /rocke-monitor resume: it starts the tick loop (tools/tick_loop.py) and "
            "the watchdog cron, and the loop's first tick restarts the review driver."
        )
    return 0


def _ago(ts: str | None) -> str:
    """ "2026-11-11T10:02Z (4 min ago)", or "never"."""
    if not ts:
        return "never"
    try:
        return f"{ts[:16]}Z ({_store.minutes_since(ts):.0f} min ago)"
    except ValueError:
        return str(ts)


def _job(store, job: dict | None) -> str:
    return f"{job['id']} ({jobs.status(store, job)})" if job else "no job"


def status_lines(
    cfg: dict,
    st: dict,
    store,
    table=None,
    env=procs.environ,
    web_running=None,
) -> list[str]:
    """`rk status`: how the monitor operates right now, from local files only (no remote call): its state; the pieces
    running (the review driver, the tick loop that runs here as rk tick-now finds it (tick_loops) and the last tick,
    the admin dashboard's service, the orchestrator, the gate, develop sanity); each current setting against its range
    with the scaler's reason and its last changes; the review queue and the work in flight; the RAM disk's use.
    """
    from . import doctor, features, gpu, idle, ramdisk, scaler
    from .review import FIRST, pipes

    from . import rules_tick

    out = [f"monitor: {state_line(st)}"]
    # The features' modes, and the lines waiting for a person, before the RAM disk's line, the last.
    extra = [
        f"features: {features.headline(cfg)}"
        + (
            "; each tick is run by fixed rules (no Claude session)"
            if features.rules_only(cfg)
            else ""
        )
    ] + rules_tick.status_lines(st)
    entries = processes(cfg, store, table, env)
    roots = [e for e in entries if not str(e.get("what") or "").startswith("child of")]
    kinds: dict[str, list[dict]] = {}
    for e in roots:
        kinds.setdefault(e["kind"], []).append(e)

    def pids(kind: str) -> str:
        return ", ".join(f"pid {e['pid']}" for e in kinds.get(kind, []))

    out.append(
        f"processes: {len(entries)}"
        + (
            " ("
            + "; ".join(
                f"{e['kind']} {e['what'] or ''} pid {e['pid']}".replace("  ", " ")
                for e in roots
            )
            + ")"
            if roots
            else ""
        )
    )
    out.append("running:")
    out.append(
        f"  review driver: {'running (' + pids('driver') + ')' if kinds.get('driver') else 'not running'}"
    )
    loops = tick_loops(cfg, store, table)
    full = (st.get(idle.KEY) or {}).get("full_at")
    out.append(
        "  tick loop: "
        + (f"running: {' and '.join(loops)}" if loops else "not running")
        + (f"; a tick runs now ({pids('tick')})" if kinds.get("tick") else "")
        + f"; last tick {_ago(st.get('last_tick'))}; last full tick {_ago(full)}"
    )
    web = doctor._service_running() if web_running is None else web_running
    out.append(
        "  admin dashboard's service: "
        + (
            "running"
            if web
            else "not running" if web is False else "never started here"
        )
    )
    if kinds.get(ORCHESTRATOR) or doctor.deployment_image():
        orch = (
            f"running ({pids(ORCHESTRATOR)})"
            if kinds.get(ORCHESTRATOR)
            else "not running"
        )
        out.append(f"  orchestrator (rk orchestrate): {orch}")
    qs = st.get("queue_state") or {}
    gate = qs.get("gate")
    out.append(
        "  gate: "
        + (
            f"#{gate['pr']} at {str(gate.get('head'))[:10]}: {_job(store, gate.get('job'))}"
            if gate
            else "none"
        )
        + (
            f"; batch gate {' -> '.join('#%d' % n for n in qs['batch_gate']['members'])}"
            if qs.get("batch_gate")
            else ""
        )
    )
    g = st.get("guard") or {}
    run_ = g.get("running")
    pend = (st.get("develop") or {}).get("pending_sanity") or []
    out.append(
        "  develop sanity: "
        + (
            f"on {str(run_.get('sha'))[:10]}: {_job(store, run_.get('job'))}"
            if run_
            else "none running"
        )
        + (f"; pending: {', '.join(s[:10] for s in pend)}" if pend else "")
        + (
            f"; bisect of {g['bisect'].get('incident')}: {_job(store, g['bisect'].get('job'))}"
            if g.get("bisect")
            else ""
        )
    )
    # The settings as the monitor sets them for its load: rkmon_lib/scaler.py.
    out.append("settings, as the monitor sets them for its load:")
    out += [f"  {line}" for line in scaler.status_lines(cfg, store)]
    out.append("reviews:")
    queue = [
        f"#{e['pr']}" + (" (again)" if e.get("again") else "")
        for e in st.get(FIRST) or []
    ]
    out.append(f"  review queue: {', '.join(queue) or 'empty'}")
    flying = [
        f"#{p['pr']} {p.get('kind')} ({p.get('stage')}: worker {jobs.status(store, p.get('worker'))}, checks "
        f"{jobs.status(store, p.get('checks'))})"
        for p in pipes(st)
    ]
    out.append(
        f"  work in flight: {len(flying)} of {scaler.pipes_in_effect(cfg, store)} parallel reviews: "
        f"{', '.join(flying) or 'none'}"
    )
    ready = [
        f"#{r['pr']} {r.get('kind')} since {str(r.get('at'))[11:16]}Z"
        for r in review.waiting_on_verifier(st)
    ]
    out.append(f"  awaiting verification: {', '.join(ready) or 'none'}")
    busy = jobs.heavy_running(st, store)
    out.append(
        f"  builds running: {len(busy)} of {scaler.current(cfg, store, 'heavy')} parallel builds: "
        f"{', '.join(h['id'] for h in busy) or 'none'}"
    )
    if features.on(
        cfg, gpu.FEATURE
    ):  # remote GPU jobs run on the cluster: a stop or a kill leaves them to finish
        act = gpu.records(st, active=True)
        out.append(
            f"  remote GPU jobs: {len(act)} queued or running"
            + (
                " ("
                + ", ".join(
                    f"#{j['pr'] or 'develop'} {j['arch']} {j['state']}" for j in act
                )
                + "; they "
                "finish on the cluster, rk gpu-cancel all gives up on them)"
                if act
                else ""
            )
        )
    u = ramdisk.usage(cfg)
    if not u["enabled"]:
        out.append("ramdisk: off (host.resources.ramdisk_gib 0)")
    elif not u["mounted"]:
        out.append(
            f"ramdisk: NOT MOUNTED at {u['path'] or '(no ramdisk.path)'}: nothing that goes on it starts "
            "(rk doctor)"
        )
    else:
        per = ramdisk.on_ramdisk(cfg)
        ready = scaler.awaiting(st)
        held = scaler.reservation(st, store, scaler.load_state(store), u)
        out.append(
            f"ramdisk: {u['used_gb']:.1f} of {u['total_gb']:.1f} GiB used at {u['path']} ({u['free_gb']:.1f} GiB "
            f"free); PRs on it: {', '.join('#%d' % n for n in sorted(per)) or 'none'}; held for the work running: "
            f"{held:.1f} GiB"
            + (
                f" ({len(ready)} awaiting verification, at most {scaler.READY_RESERVE_GB:g} GiB each)"
                if ready
                else ""
            )
        )
    at = len(out) - 1 if out and out[-1].startswith("ramdisk:") else len(out)
    return out[:at] + extra + out[at:]


def runs_web(cfg: dict) -> bool:
    from . import features

    return features.on(cfg, "admin-dashboard")


def refusal(cmd: str, st: dict) -> str:
    """Why rk CMD is refused while the monitor is stopped ("" when it is not): a command that sends, but a protective
    one (rkmon_lib/outward.py: `stops`), and one that starts work (STARTERS)."""
    row = outward.row(cmd)
    if row and row.stops or cmd in STARTERS:
        s = stopped(st)
        if s:
            return f"{halted(st)}: rk {cmd} is refused until then"
    return ""
