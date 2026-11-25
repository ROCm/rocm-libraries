#!/usr/bin/python3 -I
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The container's write broker: the writer's rk commands, run as the writer, with code that no user can change.

The writer's secrets live in /secrets-write, which only the writer (`rkwrite`) can read: the GitHub write login in
/secrets-write/gh, and each other secret the secrets registry gives the writer to keep (rkmon_lib/secrets_registry.py,
each entry's keeper). The monitor (`rkmon`), the Claude session, its agents and every job see only the read login. One
sudo rule lets `rkmon` run this file, and only this file, as `rkwrite` (/etc/sudoers.d/rkmon-write; sudo resets the
environment):

  sudo -n -u rkwrite /opt/rocke_monitor/docker/rk_write.py <rk command> [its arguments]

and the container's entry point gives the writer's own `rk` to it, without sudo (`rkmon secret NAME` and `rkmon
web-host install` on the host).

The broker loads Python's standard library and, by their paths from the image's own code (CODE, root's and read-only),
two files of the package that load the standard library only: the container's layout (rkmon_lib/container_layout.py:
the writer's user, the working folder, the code volume's folder and the file of the image's code digest) and the code
volume's names and the launcher's rule (rkmon_lib/code_volume.py). It picks the code the command runs by that rule
(rk, at the package's top), and never by a caller's word (RKMON_CODE, any other variable, an argument): the release
that `rkmon apply` installed (the code volume's CURRENT) while it was installed over this very image and is whole, else
the image's own code. Before it runs that code, it checks the whole tree, its code and its data (`problems`): each
folder from `/` down to the tree is root's and no other user may write it; each file and folder in the tree is root's
and no other user may write it; no link leaves the tree; and the code volume's links, CURRENT and KEPT, are root's.
Any other user could change the code the writer runs otherwise. When a check fails, it refuses and says what is wrong
(the first code_volume.MOST things, then how many more) and the fix.

Then it runs that tree's writer (rkmon_lib/writer_run.py) as its child: `python3 -I -B`, with that tree alone of the
package's code on the path (BOOT), in a new folder of the writer's own for the run's copies of the file arguments
(ARGS_DIR, mode 700), with none of the caller's environment but sudo's word of who called (SUDO_VARS) and, checked,
the name the host's wrapper was run by, which the fixes show (container_layout.shown_wrapper). The writer checks the
command against its list, its file arguments and the configuration's folders, and runs it. The broker passes the
signals it gets on to its child (run_command), as sudo passes them on to the broker, and when the child ends, removes
the run's copies and exits with the child's code. Each refusal is one line of the writer's log (LOG, in the working
folder's logs), as each command the writer runs is.

The broker's one copy of the package's words, the writer's refusal line (REFUSAL: rkmon_lib/writer.py, which loads
more than the standard library, says it), and the name of the writer's log, which rkmon_lib/writer_run.py names too,
are held to their homes by tests/test_code_volume.py.
"""

import datetime as dt
import os
import pwd
import runpy
import shutil
import signal
import stat
import sys
import tempfile
import time
import types

CODE = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)  # the image's own code: root's, read-only
LAYOUT = types.SimpleNamespace(
    **runpy.run_path(os.path.join(CODE, "rkmon_lib", "container_layout.py"))
)
VOLUME = types.SimpleNamespace(
    **runpy.run_path(os.path.join(CODE, "rkmon_lib", "code_volume.py"))
)
WRITE_USER = LAYOUT.WRITER.name
LOG = f"{LAYOUT.WORK_DIR}/{LAYOUT.LOGS_FOLDER}/rk-write.log"  # rkmon_lib/writer_run.py LOG
REFUSAL = "REFUSED: the writer: {}; nothing was run"  # rkmon_lib/writer.py refusal
CODE_DIR, IMAGE_ID_FILE = LAYOUT.CODE_DIR, LAYOUT.IMAGE_ID_FILE
# The child: the writer of the tree on the path (sys.argv[1]), with the command's words after it.
BOOT = (
    "import sys; sys.path.insert(0, sys.argv[1]); from rkmon_lib import writer_run; "
    "sys.exit(writer_run.main(sys.argv[2:]))"
)
PATH = "/usr/local/bin:/usr/bin:/bin"
SUDO_VARS = (
    "SUDO_UID",
    "SUDO_GID",
    "SUDO_USER",
)  # sudo's word of who called: the writer's checks read it
ARGS_DIR = ".rk-write-args"  # in the write user's home: the checked copies of the file arguments, one folder a run
ARGS_KEEP_S = 3600  # the copies of a run whose broker was killed outright go with a later run this long after
# The signals the broker passes on to its child, as sudo passes them on to the broker: a keyboard's SIGINT and SIGQUIT
# only when a process sent them, since the keyboard's reach the child too (they go to the terminal's process group).
RELAYED = (signal.SIGHUP, signal.SIGINT, signal.SIGQUIT, signal.SIGTERM)
KEYBOARD = (signal.SIGINT, signal.SIGQUIT)
SI_KERNEL = 0x80  # siginfo's si_code for a signal the kernel sent, as a terminal sends a keyboard's (Linux)


def _inside(path: str, folder: str) -> bool:
    return path == folder or path.startswith(folder.rstrip("/") + "/")


def picked() -> str:
    """The code tree the writer runs, by the launcher's rule (code_volume.running): CURRENT's release while it was
    installed over this very image (its record's base_image is the image's code digest) and is whole, else the image's
    own code. No variable and no argument changes it."""
    try:
        with open(IMAGE_ID_FILE, encoding="utf-8") as f:
            image = f.read().strip()
    except OSError:
        return CODE
    tree = VOLUME.running(CODE_DIR, image)
    return CODE if tree is None else str(tree)


def _owned(path: str, s: os.stat_result) -> str:
    """Why PATH, which S describes, is not root's alone to write; "" when it is. With an ACL, the group's bits are its
    mask, which caps each user and group the ACL names."""
    if s.st_uid != 0:
        return f"{path} is another user's (uid {s.st_uid})"
    if s.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return f"{path} may be written by another user (mode {stat.S_IMODE(s.st_mode):04o})"
    return ""


def problems(tree: str, lstat=os.lstat, kept=(), skip=()) -> list[str]:
    """What keeps the code tree TREE from being root's and read-only all the way, a line each: a folder from `/` down to
    it, or a file or folder in it, that is another user's or that another user may write; a link in it that is another
    user's or that leaves it (one at its top may name a folder of KEPT, as the code volume's KEPT names the image's
    code); anything in it that is neither a file nor a folder; and what cannot be read. [] for a tree that is root's
    and read-only all the way. A path in it that SKIP names is left out, with all it holds: the doctor's check of the
    code volume leaves out what a change of the code leaves there while it runs (docker/rk_root.py Volume.problems).
    LSTAT is os.lstat (the tests give their own).
    """
    real, out, way = os.path.realpath(tree), [], "/"
    for part in real.split("/"):
        way = os.path.join(way, part)
        try:
            s = lstat(way)
        except OSError as exc:
            return out + [f"{way} cannot be read ({exc.strerror})"]
        if not stat.S_ISDIR(s.st_mode):
            return out + [f"{way} is not a folder"]
        out += [why for why in [_owned(way, s)] if why]
    todo = [real]
    while todo:
        folder = todo.pop()
        try:
            names = sorted(os.listdir(folder))
        except OSError as exc:
            out.append(f"{folder} cannot be read ({exc.strerror})")
            continue
        for name in names:
            path = os.path.join(folder, name)
            if path in skip:
                continue
            try:
                s = lstat(path)
            except OSError as exc:
                out.append(f"{path} cannot be read ({exc.strerror})")
                continue
            if stat.S_ISLNK(s.st_mode):
                to = os.path.realpath(path)
                if s.st_uid != 0:
                    out.append(f"{path} is a link of another user's (uid {s.st_uid})")
                elif not (_inside(to, real) or (folder == real and to in kept)):
                    out.append(f"{path} is a link that leaves the code (to {to})")
            elif stat.S_ISDIR(s.st_mode) or stat.S_ISREG(s.st_mode):
                out += [why for why in [_owned(path, s)] if why]
                if stat.S_ISDIR(s.st_mode):
                    todo.append(path)
            else:
                out.append(f"{path} is neither a file nor a folder")
    return out


def links(lstat=os.lstat) -> list[str]:
    """What is wrong with the code volume's links, CURRENT and KEPT, each where it is: another user's link."""
    out = []
    for name in (VOLUME.CURRENT, VOLUME.KEPT):
        path = os.path.join(CODE_DIR, name)
        try:
            s = lstat(path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            out.append(f"{path} cannot be read ({exc.strerror})")
            continue
        if s.st_uid != 0:
            out.append(f"{path} is another user's (uid {s.st_uid})")
    return out


def copy_dir(home: str, now: float | None = None) -> str:
    """A new folder of this user's for one run's checked copies, under `home`/ARGS_DIR (mode 700); the folders of runs
    older than ARGS_KEEP_S are removed first."""
    base = os.path.join(home, ARGS_DIR)
    os.makedirs(base, mode=0o700, exist_ok=True)
    os.chmod(base, 0o700)
    t = time.time() if now is None else now
    for e in os.scandir(base):
        try:
            if e.is_dir(follow_symlinks=False) and t - e.stat().st_mtime > ARGS_KEEP_S:
                shutil.rmtree(e.path, ignore_errors=True)
        except OSError:
            pass
    return tempfile.mkdtemp(prefix="run-", dir=base)


def log(line: str) -> None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{stamp} {line}\n")
    except OSError:
        print(f"rk-write: {line}", file=sys.stderr)


def refuse(why: str, argv: list[str]) -> int:
    """Refuse rk ARGV for WHY: one line of the log, and the writer's one refusal line (REFUSAL); exit 4."""
    log(f"REFUSED {' '.join(argv)[:300]}: {why}")
    print(REFUSAL.format(why))
    return 4


def run_command(argv: list[str], env: dict, cwd: str) -> int:
    """Run ARGV (its program by path) as this process's child, in CWD with ENV only, and wait for it to end. Each
    RELAYED signal this process gets goes on to it, but for a KEYBOARD one the kernel sent, which reached the child
    too; none of them ends this process meanwhile. The child's exit code: 128 plus the signal's number when a signal
    ended it, 127 when it could not start.
    """
    waited = {*RELAYED, signal.SIGCHLD}
    reaping = signal.signal(
        signal.SIGCHLD, signal.SIG_DFL
    )  # an inherited "ignore" would reap the child unseen
    before = signal.pthread_sigmask(signal.SIG_BLOCK, waited)
    try:
        pid = os.fork()
        if pid == 0:
            try:
                signal.pthread_sigmask(signal.SIG_SETMASK, before)
                os.chdir(cwd)
                os.execve(argv[0], argv, env)
            except OSError as exc:
                print(
                    f"rk-write: {argv[0]} did not start ({exc.strerror})",
                    file=sys.stderr,
                )
            finally:
                os._exit(127)
        while True:
            got = signal.sigwaitinfo(waited)
            if got.si_signo != signal.SIGCHLD:
                if got.si_signo not in KEYBOARD or got.si_code != SI_KERNEL:
                    os.kill(pid, got.si_signo)
                continue
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                code = os.waitstatus_to_exitcode(status)
                return code if code >= 0 else 128 - code
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, before)
        signal.signal(signal.SIGCHLD, reaping)


def main(argv: list[str], run=run_command, environ=None, lstat=os.lstat) -> int:
    e = os.environ if environ is None else environ
    me = pwd.getpwuid(os.geteuid())
    if me.pw_name != WRITE_USER:
        return refuse(
            f"it runs as {WRITE_USER} only (sudo -u {WRITE_USER}), not {me.pw_name}",
            argv,
        )
    tree = picked()
    bad = problems(tree, lstat) + links(lstat)
    if bad:
        fix = (
            "build the image again (QUICK_BUILD_GUIDE.md) and start the container from it"
            if tree == CODE
            else "rkmon apply, on the host, installs this package's code again, as root"
        )
        return refuse(
            f"the code it would run ({tree}) is not root's and read-only all the way: {VOLUME.listed(bad)}; the fix: "
            f"{fix}",
            argv,
        )
    try:
        copies = copy_dir(me.pw_dir)
    except OSError as exc:
        return refuse(f"no folder for the arguments' copies ({exc})", argv)
    env = {"PATH": PATH, "LANG": "C.UTF-8", "HOME": me.pw_dir, "USER": me.pw_name}
    env.update({k: e[k] for k in SUDO_VARS if k in e})
    env[LAYOUT.WRAPPER_VARIABLE] = LAYOUT.shown_wrapper(e)  # shown in a fix only
    try:  # in the copies' folder: a relative name the writer opens resolves among the checked copies only
        return run([sys.executable, "-I", "-B", "-c", BOOT, tree, *argv], env, copies)
    finally:
        shutil.rmtree(copies, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
