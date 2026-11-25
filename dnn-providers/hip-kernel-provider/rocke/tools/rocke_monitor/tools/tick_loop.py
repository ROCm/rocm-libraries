#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The tick loop: `rk tick` every tick_interval_min minutes, counted from one tick's start to the next one's, a restarted
loop included. Every tick's output goes to the loop log. The loop exits, which wakes the orchestrator that started it:
  - when a tick prints a line the orchestrator acts on (`rk tick-filter`): LOOP_WAKE, then those lines. A standing
    line (one the tick repeats while its condition holds, such as PROPOSE) wakes it once, not on every tick;
  - when a tick fails, or runs past tick_timeout_min and is stopped: LOOP_WAKE, then why (for a stopped tick, the
    gh calls that wait for the GitHub request cap as it stops, if any: rkmon_lib/gh_cap.py waiting_text);
  - loop_heartbeat_min minutes after it started: LOOP_HEARTBEAT, a sign of life with the tick counts, then the
    standing lines still in force.
The orchestrator acts on what it printed, then starts the loop again: the loop's own output holds every line to act
on (the timeout's and the filter's failure with the tick's last lines), so the loop log need not be read. Idle and
routine ticks wake nobody.
`rk tick-now` asks for a tick at once: while the loop waits, it tries to take that request (TICK_NOW) every ASK_S
seconds, and once it took it, starts a tick; a request it cannot take (a folder at its path) leaves the wait as it is.
Each tick the loop starts takes a request that waits too, and the log names it.
One loop at a time: a second one prints LOOP_RUNNING and exits 2. The lock, the log, the request and every setting
come from `rk loop-config` (config: tick_interval_min, loop_heartbeat_min, tick_timeout_min, loop_log_max_mb).
TICK_LOOP_RK replaces rk, for tests.

Usage: python3 -B tools/tick_loop.py   (or tools/tick_loop.py)
A SIGTERM, SIGINT or SIGHUP ends the loop at once with the signal's own status (143 for SIGTERM): the pid file goes,
the lock with the process, and a tick already running is left to finish on its own.
"""

import fcntl
import os
import shlex
import signal
import subprocess
import sys
import time

ENC = "utf-8"
ERR = "surrogateescape"  # every byte the tick printed goes through unchanged
PROG = sys.argv[0]
ASK_S = 2  # while the loop waits, it looks for rk tick-now's request this often
_pid_file = None  # set once this loop owns the lock: removed on every exit, as the EXIT trap did


def say(text: str) -> None:
    sys.stdout.buffer.write(text.encode(ENC, ERR))
    sys.stdout.buffer.flush()


def line(text: str) -> None:
    say(text + "\n")


def subst(raw: bytes) -> str:
    """A command substitution's value: the output with its trailing newlines dropped."""
    return raw.decode(ENC, ERR).rstrip("\n")


def run(argv: list[str], stdin_text=None, lineno: int = 0) -> tuple[int, str]:
    """argv's stdout and stderr together, as `$(cmd 2>&1)`; (127, the shell's message) when it cannot start."""
    try:
        p = subprocess.run(
            argv,
            input=None if stdin_text is None else stdin_text.encode(ENC, ERR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except FileNotFoundError:
        return 127, f"{PROG}: line {lineno}: {argv[0]}: No such file or directory"
    except PermissionError:
        return 126, f"{PROG}: line {lineno}: {argv[0]}: Permission denied"
    return p.returncode, subst(p.stdout)


def tail(text: str, n: int) -> list[str]:
    """`printf '%s\\n' "$text" | tail -n N`, as lines."""
    return text.split("\n")[-n:]


def num(value) -> int:
    """A shell arithmetic operand: blank is 0."""
    value = str(value).strip()
    return int(value) if value else 0


def hhmm(t: int) -> str:
    return time.strftime("%H:%MZ", time.gmtime(t))


def next_wait(t0: int, interval: int, now: int) -> int:
    """Seconds until the next tick, counted from this tick's start `t0`: never more than one interval, so a clock
    stepped back (an NTP correction, a WSL resync) never stretches the wait past the heartbeat.
    """
    return max(0, min(t0 + interval - now, interval))


def wait(seconds: int, ask_f: str) -> bool:
    """SECONDS of waiting, cut short once rk tick-now's request (the file ASK_F) is taken: tried every ASK_S. True
    when it was taken, so the tick that starts next is the one asked for; False when the time ran out.
    """
    end = time.monotonic() + seconds
    while not take(ask_f):
        left = end - time.monotonic()
        if left <= 0:
            return False
        time.sleep(min(ASK_S, left))
    return True


def take(ask_f: str) -> bool:
    """rk tick-now's request taken: True when it was there. Taking it removes it, so one loop takes it, once
    (rkmon_lib/control.py take_tick)."""
    if not ask_f:
        return False
    try:
        os.unlink(ask_f)
    except OSError:
        return False
    return True


def cap_waits() -> str:
    """The gh calls that wait for the GitHub request cap now, as the clause a stopped tick's line ends with ("" for
    none): the record of the monitor's home (RKMON_HOME), as the counter keeps it."""
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    from rkmon_lib import gh_cap

    state = gh_cap.state_folder()
    text = gh_cap.waiting_text(state, time.time()) if state else ""
    return f" {text}" if text else ""


def cleanup() -> None:
    global _pid_file
    if _pid_file:
        try:
            os.unlink(_pid_file)
        except OSError:
            pass
        _pid_file = None


def on_signal(sig, _frame) -> None:
    """As bash with an EXIT trap: the trap's cleanup, then death by the same signal (children keep running)."""
    cleanup()
    signal.signal(sig, signal.SIG_DFL)
    os.kill(os.getpid(), sig)


def main() -> int:
    global _pid_file
    rk = (
        os.environ.get("TICK_LOOP_RK")
        or os.path.dirname(os.path.realpath(__file__)) + "/../rk"
    )
    rc, conf = run([rk, "loop-config"], lineno=18)
    if rc != 0:
        line(f"LOOP_WAKE: rk loop-config failed, so no tick ran: {conf}")
        return 1
    v = {}
    for word in shlex.split(
        conf, comments=True
    ):  # TICK_INTERVAL_S ... and the LOOP_* files
        if "=" in word:
            k, _, val = word.partition("=")
            v[k] = val
    interval, heartbeat = num(v.get("TICK_INTERVAL_S", "")), num(
        v.get("LOOP_HEARTBEAT_S", "")
    )
    timeout_s, log_max = num(v.get("TICK_TIMEOUT_S", "")), num(
        v.get("LOOP_LOG_MAX_BYTES", "")
    )
    lock_f, log_f, pid_f = (
        v.get("LOOP_LOCK", ""),
        v.get("LOOP_LOG", ""),
        v.get("LOOP_PID", ""),
    )
    last_f, seen_f = v.get("LOOP_LAST", ""), v.get("LOOP_SEEN", "")
    ask_f = v.get("TICK_NOW", "")  # rk tick-now's request
    try:
        lock = open(
            lock_f, "wb"
        )  # not inherited by any child (close-on-exec): a tick never holds it
    except OSError as exc:
        sys.stderr.write(f"{PROG}: line 20: {lock_f}: {exc.strerror}\n")
        return 1
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            with open(pid_f, "rb") as f:
                pid = subst(f.read())
        except OSError:
            pid = "unknown"
        line(
            f"LOOP_RUNNING: a tick loop is already running (pid {pid}); nothing started"
        )
        return 2
    try:
        with open(pid_f, "w", encoding=ENC) as f:
            f.write(f"{os.getpid()}\n")
    except OSError as exc:
        sys.stderr.write(f"{PROG}: line 26: {pid_f}: {exc.strerror}\n")
    _pid_file = pid_f
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, on_signal)
    started = int(time.time())
    ticks = idle = full = skipped = 0
    try:
        with open(last_f, "rb") as f:
            last = subst(f.read())
    except OSError:
        last = "0"
    if not last or not all(c in "0123456789" for c in last):
        last = "0"
    wait_s = int(last) + interval - started  # a restarted loop keeps the tick interval
    asked = wait(wait_s, ask_f) if 0 < wait_s <= interval else False
    while True:
        t0 = int(time.time())
        at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        # the request taken in the wait, or one that waits now
        asked = take(ask_f) or asked
        with open(last_f, "w", encoding=ENC) as f:
            f.write(f"{t0}\n")
        if os.path.isfile(log_f) and os.lstat(log_f).st_size > log_max:
            os.replace(log_f, log_f + ".1")
        rc, out = run(["timeout", "-k", "30", str(timeout_s), rk, "tick"], lineno=41)
        what = "rk tick (rk tick-now asked for it)" if asked else "rk tick"
        with open(log_f, "ab") as f:
            f.write(f"=== {at} {what}: exit {rc}\n{out}\n".encode(ENC, ERR))
        ticks += 1
        if "TICK idle: " in out:
            idle += 1
        elif "another tick is running" in out:
            skipped += 1
        else:
            full += 1
        if rc in (124, 137):
            # The lines for the orchestrator from what it printed before it was stopped, then its last lines: no need
            # to read the log.
            _frc, acts = run([rk, "tick-filter"], stdin_text=out + "\n", lineno=51)
            acts = (
                "\n".join(
                    x for x in acts.split("\n") if not x.startswith("TICK_EMPTY")
                ).rstrip("\n")
                if acts
                else ""
            )
            line(
                f"LOOP_WAKE {at}: the tick ran past tick_timeout_min ({int(timeout_s / 60)} min) and was stopped "
                f"(exit {rc}){cap_waits()}; its lines for the orchestrator so far, then its last 5 lines"
            )
            if acts:
                line(acts)
            say("".join(f"  | {x}\n" for x in tail(out, 5)))
            return 0
        frc, lines = run(
            [rk, "tick-filter", "--rc", str(rc), "--seen", seen_f],
            stdin_text=out + "\n",
            lineno=58,
        )
        if frc > 1:  # the filter itself failed: its message, then the tick's last lines
            line(
                f"LOOP_WAKE {at}: the tick (exit {rc}) needs the orchestrator, and the filter failed (exit {frc}): "
                "its message, then the tick's last 15 lines"
            )
            line(lines)
            say("".join(f"  | {x}\n" for x in tail(out, 15)))
            return 0
        if frc == 0 or lines:  # lines for the orchestrator
            k = sum(1 for x in lines.split("\n") if x)
            what = "the 1 line below" if k == 1 else f"the {k} lines below"
            line(
                f"LOOP_WAKE {at}: the tick (exit {rc}) needs the orchestrator for {what}"
            )
            line(lines)
            return 0
        now = int(time.time())
        if now - started >= heartbeat:
            first = next((x for x in out.split("\n") if x.startswith("TICK ")), "")
            line(
                f"LOOP_HEARTBEAT {at}: {ticks} ticks since {hhmm(started)} ({idle} idle, {full} full, {skipped} "
                f"skipped while another tick ran), none for the orchestrator; the last: {first}"
            )
            if os.path.exists(seen_f) and os.path.getsize(seen_f) > 0:
                line("Still standing (reported before):")
                with open(seen_f, "rb") as f:
                    sys.stdout.buffer.write(f.read())
                sys.stdout.buffer.flush()
            return 0
        wait_s = next_wait(t0, interval, now)
        asked = wait(wait_s, ask_f) if wait_s > 0 else False


if __name__ == "__main__":
    signal.signal(
        signal.SIGINT, signal.SIG_DFL
    )  # as bash: an interrupt before the lock ends the loop quietly
    try:
        code = main()
    finally:
        cleanup()
    sys.exit(code)
