# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The web host's half of `rkmon web-host install` (rkmon_lib/web_host_install.py), which runs it there over the web
host's key as `python3 - --status|--yes REPLACE BASE SITE REQUESTS PORT`, this file on its standard input. On a Python
older than OLDEST_PYTHON it refuses first, with --status or --yes, so the install stops at its look, before it copies
anything. It then shows what serves the page's port; with --yes it installs or restarts rocke-monitor.service in its
place, from the copies the install put in BASE, serving SITE (made when it is missing, as the page's first push needs
it).

The host may have no user unit to replace (the page served some other way). This half looks before it acts. It stops
only this login's own `http.server` on the port, and only with REPLACE yes; and an older package's user service for the
page (OLDER_UNITS) whose server holds the port, as this service's predecessor, which it stops and disables. If the new
service does not come up, it starts the old server or that older service again as it was.

Its exit code: 0 the service runs and answers (with --status: the look is done); 1 the service did not come up; 2 a
usage error; REFUSED (4) a step refused (with --status, its Python), which its last line names, and nothing changed.

Standard library only, Python 3.6 or later (the web host's Python, the one that runs webhost_server.py).
"""

import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

UNIT = "rocke-monitor.service"
# The user service an older package installs for the page under its own name: one whose server holds the port is
# replaced by UNIT, with no REPLACE, and runs again when UNIT does not come up.
OLDER_UNITS = ("rocke-merge-monitor.service",)
USAGE, REFUSED = 2, 4  # exit codes: a usage error; a step refused, and nothing changed
OLDEST_PYTHON = (3, 6)  # the oldest Python that runs webhost_server.py


def say(text=""):
    print(text, flush=True)


def run(argv, quiet_err=False):
    """(exit code, stdout) of argv; its stderr goes to ours unless quiet_err. A program that is not there: 127, ""."""
    sys.stdout.flush()
    try:
        p = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL if quiet_err else None,
            stdin=subprocess.DEVNULL,
            universal_newlines=True,
        )
    except OSError:
        if not quiet_err:
            sys.stderr.write("%s: command not found\n" % argv[0])
        return 127, ""
    return p.returncode, p.stdout


def call(argv):
    """Run argv with our stdout and stderr; its exit code."""
    sys.stdout.flush()
    try:
        return subprocess.run(argv, stdin=subprocess.DEVNULL).returncode
    except OSError:
        sys.stderr.write("%s: command not found\n" % argv[0])
        return 127


def quietly(argv):
    """Run argv with no output."""
    subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def sub(argv, quiet_err=False):
    """A command substitution: stdout without its trailing newlines."""
    return run(argv, quiet_err)[1].rstrip("\n")


def listen(port):
    return sub(["ss", "-Hltnp", "sport = :%s" % port], quiet_err=True)


def pids(port):
    found = re.findall(r"pid=([0-9]*)", listen(port))
    return sorted(set(found))


def answers(port):
    """What this host's port PORT answers on 127.0.0.1, asked past any proxy the login's environment names: "answers:
    ..." for an HTTP answer of any status (the server runs, whatever the site folder holds yet), else "does not
    answer: ...".
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open("http://127.0.0.1:%s/" % port, timeout=5) as r:
            return "answers: HTTP %d, Server %s" % (r.status, r.headers.get("Server"))
    except urllib.error.HTTPError as exc:
        exc.close()
        return "answers: HTTP %d, Server %s" % (exc.code, exc.headers.get("Server"))
    except Exception as exc:
        return "does not answer: %s" % exc


def cwd_of(pid):
    try:
        return os.readlink("/proc/%s/cwd" % pid)
    except OSError:
        return None


def unit_file(home, unit):
    """The path of the user service UNIT's unit file in the login's HOME."""
    return home + "/.config/systemd/user/" + unit


def unit_state(home, unit):
    """(systemd's word for the user service UNIT: active, inactive, ...; the path of its unit file, or "none")."""
    active = sub(["systemctl", "--user", "is-active", unit], quiet_err=True)
    u = unit_file(home, unit)
    return active, u if os.path.exists(u) else "none"


def main_pid(unit):
    """The main PID of the user service UNIT, as systemd gives it ("0" while it does not run)."""
    return sub(
        ["systemctl", "--user", "show", "-p", "MainPID", "--value", unit],
        quiet_err=True,
    )


def main(argv):
    if len(argv) < 6:
        sys.stderr.write(
            "usage: python3 - --status|--yes REPLACE BASE SITE REQUESTS PORT < webhost_remote.py\n"
        )
        return USAGE
    mode, replace, base, site, requests, port = argv[:6]
    # refused at the look, before the install copies anything
    if sys.version_info[:2] < OLDEST_PYTHON:
        say(
            "python3 here is %d.%d, and the request service needs %d.%d or newer; nothing changed"
            % (sys.version_info[:2] + OLDEST_PYTHON)
        )
        return REFUSED
    home = os.path.expanduser("~")
    user = os.environ.get("USER", "")
    u = unit_file(home, UNIT)
    new = home + "/" + base + "/" + UNIT + ".new"

    say("== %s: port %s" % (socket.gethostname(), port))
    listening = listen(port)
    say(listening or "(nothing listens on %s)" % port)
    found = pids(port)
    for p in found:
        cwd = cwd_of(p)
        say(
            "  PID %s: %s (cwd %s)"
            % (
                p,
                sub(["ps", "-o", "user=,args=", "-p", p]),
                "?" if cwd is None else cwd,
            )
        )
    if listening and not found:
        say(
            "  (the listener is not this login's process: another user or root runs it)"
        )
    if listening:
        say("  " + answers(port))
    say("== %s: %s; unit file: %s" % ((UNIT,) + unit_state(home, UNIT)))
    for unit in OLDER_UNITS:
        active, file = unit_state(home, unit)
        if active == "active" or file != "none":
            say("== %s, an older package's: %s; unit file: %s" % (unit, active, file))
    rc, linger = run(
        ["loginctl", "show-user", user, "-p", "Linger", "--value"], quiet_err=True
    )
    if rc != 0:
        linger += "unknown\n"
    say("== linger (user services keep running after logout): %s" % linger.rstrip("\n"))
    rx = re.compile(r"http\.server|%s|%s" % (port, base))
    for line in run(["crontab", "-l"], quiet_err=True)[1].splitlines():
        if rx.search(line):
            say("== crontab: " + line)
    version = subprocess.run(
        ["python3", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    ).stdout.rstrip("\n")
    site_dir = home + "/" + site
    say(
        "== python3: %s; site: %s"
        % (version, site_dir if os.path.isdir(site_dir) else "missing ~/" + site)
    )
    if mode != "--yes":
        return 0

    say("== deploy")
    if not (
        os.path.exists(new) and os.path.exists(home + "/" + base + "/webhost_server.py")
    ):
        say("the new unit or webhost_server.py did not arrive; nothing changed")
        return REFUSED
    if listening and not found:
        say(
            "port %s is held by a process this login cannot see or stop; nothing changed"
            % port
        )
        return REFUSED
    own = main_pid(UNIT)
    # an older package's service whose server holds the port: this one's predecessor, replaced as this one is
    older = {}
    for unit in OLDER_UNITS:
        p = main_pid(unit)
        if p in found:
            older[p] = unit
    old = []
    for p in found:
        if p == own or p in older:
            continue
        if "http.server" in sub(["ps", "-o", "args=", "-p", p]):
            old.append(p)
        else:
            say(
                "port %s is held by PID %s, which is not an http.server; nothing changed"
                % (port, p)
            )
            return REFUSED
    if old and replace != "yes":
        say(
            "port %s is served by this login's own http.server (PID%s, above): stop it and install again, and %s "
            "takes its place, on the same folder and port; nothing changed"
            % (port, "".join(" " + p for p in old), UNIT)
        )
        return REFUSED
    linger = sub(
        ["loginctl", "show-user", user, "-p", "Linger", "--value"], quiet_err=True
    )
    if (
        linger != "yes"
        and run(["loginctl", "enable-linger", user], quiet_err=True)[0] != 0
    ):
        say(
            "user services stop at logout here (Linger=%s), and this login cannot turn lingering on, so %s would stop "
            "when the ssh session ends: the host's admin runs `loginctl enable-linger %s`; nothing changed"
            % (linger, UNIT, user)
        )
        return REFUSED
    call(
        [
            "mkdir",
            "-p",
            home + "/" + (os.path.dirname(requests) or "."),
            site_dir,
            os.path.dirname(u),
        ]
    )
    if os.path.exists(u):
        call(["cp", "-p", u, u + ".bak-" + time.strftime("%Y%m%d%H%M")])
    call(["mv", new, u])
    call(["systemctl", "--user", "daemon-reload"])
    replaced = sorted(set(older.values()))
    for unit in replaced:
        say(
            "%s, an older package's, serves port %s: %s takes its place"
            % (unit, port, UNIT)
        )
        call(["systemctl", "--user", "stop", unit])
        quietly(["systemctl", "--user", "disable", unit])
    args, cwds = {}, {}
    for p in old:
        args[p] = sub(["ps", "-o", "args=", "-p", p])
        cwds[p] = cwd_of(p) or ""
        try:
            os.kill(int(p), signal.SIGTERM)
        except OSError as exc:
            sys.stderr.write("kill: (%s) - %s\n" % (p, exc.strerror))
    for _ in range(10):
        if not [p for p in pids(port) if p != (own or "x")]:
            break
        time.sleep(1)
    quietly(["systemctl", "--user", "enable", UNIT])
    call(["systemctl", "--user", "restart", UNIT])
    time.sleep(3)
    if sub(["systemctl", "--user", "is-active", UNIT]) == "active" and answers(
        port
    ).startswith("answers"):
        say("%s is running: %s" % (UNIT, answers(port)))
        for line in run(["systemctl", "--user", "--no-pager", "status", UNIT])[
            1
        ].splitlines()[:5]:
            say(line)
        return 0
    say("%s did not come up:" % UNIT)
    for line in run(["systemctl", "--user", "--no-pager", "status", UNIT])[
        1
    ].splitlines()[:12]:
        say(line)
    out = run(
        ["journalctl", "--user", "-u", UNIT, "-n", "10", "--no-pager"], quiet_err=True
    )[1]
    if out:
        sys.stdout.write(out)
        sys.stdout.flush()
    if replaced:
        # the port goes back to the older service, which systemd would otherwise share with this one at each start
        call(["systemctl", "--user", "stop", UNIT])
        quietly(["systemctl", "--user", "disable", UNIT])
    for unit in replaced:
        quietly(["systemctl", "--user", "enable", unit])
        call(["systemctl", "--user", "start", unit])
        say("started %s again, and %s is stopped" % (unit, UNIT))
    for p in old:
        try:
            subprocess.Popen(
                args[p].split(),
                cwd=cwds[p] or None,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
                preexec_fn=lambda: signal.signal(signal.SIGHUP, signal.SIG_IGN),
            )
        except OSError:
            pass
        say("started the old server again: %s (in %s)" % (args[p], cwds[p]))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
