#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The local service behind the HTML dashboard's "Request PR priority" box: `rk serve` in the background, logging to
state/web.log: the admin dashboard (where it listens and who can reach it: README.md, "The admin dashboard") and, with
team_dashboard.mode serve, the team dashboard too.
Safe to run twice: the service holds state/web.lock while it runs, and this script starts nothing while that lock is
held.
  tools/rkweb.py [PORT]   start it (default port: config.json's admin_dashboard.port)
  tools/rkweb.py status   say whether it runs
  tools/rkweb.py stop     stop it
It serves the checkout it lives in (that code) and the monitor's home (`rk home`: its state/ and config.json).
"""

import fcntl
import os
import re
import signal
import subprocess
import sys
import time

PROG = sys.argv[0]


def say(text: str) -> None:
    print(text, flush=True)


def running(lock: str) -> bool:
    """The lock file is there and another process holds it (`[ -e LOCK ] && ! flock -n LOCK true`)."""
    if not os.path.exists(lock):
        return False
    try:
        fd = os.open(lock, os.O_RDONLY | os.O_NOCTTY | os.O_CREAT, 0o666)
    except OSError:
        return True  # flock could not open it: it fails, so "held"
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True
    finally:
        os.close(fd)
    return False


def read_pid(pidf: str) -> str:
    try:
        with open(pidf, "rb") as f:
            return f.read().decode("utf-8", "surrogateescape").rstrip("\n")
    except OSError:
        return "?"


def tail(path: str, n: int) -> None:
    """`tail -N FILE` to stdout, bytes as they are."""
    sys.stdout.flush()
    try:
        with open(path, "rb") as f:
            data = f.read()
    except OSError as exc:
        sys.stderr.write(f"tail: cannot open '{path}' for reading: {exc.strerror}\n")
        return
    lines = data.split(b"\n")
    if data.endswith(b"\n"):
        lines = [x + b"\n" for x in lines[:-1]]
    else:
        lines = [x + b"\n" for x in lines[:-1]] + [lines[-1]]
    sys.stdout.buffer.write(b"".join(lines[-n:]))
    sys.stdout.buffer.flush()


def background_signals() -> None:
    """A shell's `cmd &` without job control: the command ignores SIGINT and SIGQUIT."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGQUIT, signal.SIG_IGN)


def main(argv: list[str]) -> int:
    root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    rk = root + "/rk"
    try:
        out = subprocess.run([rk, "home"], stdout=subprocess.PIPE).stdout
    except OSError as exc:
        sys.stderr.write(f"{PROG}: line 14: {rk}: {exc.strerror}\n")
        out = b""
    home_dir = out.split(b"\n", 1)[0].decode("utf-8", "surrogateescape")
    state = f"{home_dir}/state"
    lock, pidf, log = f"{state}/web.lock", f"{state}/web.pid", f"{state}/web.log"
    try:
        os.makedirs(state, exist_ok=True)
    except OSError as exc:
        sys.stderr.write(f"mkdir: cannot create directory '{state}': {exc.strerror}\n")

    what = argv[1] if len(argv) > 1 and argv[1] != "" else "start"
    # With the admin dashboard off (features.dashboards.admin) no web server runs: nothing to start or watch.
    try:
        unneeded = (
            subprocess.run(
                [rk, "features", "--runs", "web"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 1
        )
    except OSError:
        unneeded = False
    if unneeded and what in ("status", "start"):
        say("RKWEB not needed: features.dashboards.admin is off, so no web server runs")
        return 0
    if what == "status":
        say(
            f"RKWEB running (pid {read_pid(pidf)}); log {log}"
            if running(lock)
            else "RKWEB not running"
        )
        return 0
    if what == "stop":
        if not running(lock):
            say("RKWEB not running")
            return 0
        p = read_pid(pidf)
        if not re.fullmatch(
            r"[0-9]+", p
        ):  # as the shell's kill with a word that is no pid
            sys.stderr.write(
                f"{PROG}: line 31: kill: `{p}': not a pid or valid job spec\n"
            )
            return 1
        try:
            os.kill(int(p), signal.SIGTERM)
        except OSError as exc:
            sys.stderr.write(f"{PROG}: line 31: kill: ({p}) - {exc.strerror}\n")
            return 1
        for _ in range(10):
            if not running(lock):
                say(f"RKWEB stopped (pid {p})")
                return 0
            time.sleep(0.5)
        say(f"RKWEB pid {p} did not stop: {lock} is still locked")
        return 1
    if what == "start":
        port_args = []
    elif re.fullmatch(r"[0-9]+", what):
        port_args = ["--port", what]
    else:
        say(f"usage: {PROG} [PORT|status|stop]")
        return 2

    if running(lock):
        say(f"RKWEB already running (pid {read_pid(pidf)}); log {log}")
        return 0
    try:
        with open(log, "ab") as logf, open(os.devnull, "rb") as null:
            subprocess.Popen(
                ["nohup", "setsid", rk, "serve", *port_args],
                stdin=null,
                stdout=logf,
                stderr=subprocess.STDOUT,
                preexec_fn=background_signals,
            )
    except OSError as exc:
        sys.stderr.write(f"{PROG}: line 40: {log}: {exc.strerror}\n")
    for _ in range(10):
        time.sleep(0.5)
        if running(lock):
            say(f"RKWEB started (pid {read_pid(pidf)}); log {log}")
            tail(log, 1)
            return 0
    say(f"RKWEB did not start; the end of {log}:")
    tail(log, 5)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
