#!/usr/bin/python3 -I
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The container's PR-code launcher: it works as a PR runner, in that runner's slot.

One sudo rule lets the monitor's user run this file, and only this file, as any PR runner
(/etc/sudoers.d/rkmon-prcode; sudo resets the environment and closes every other descriptor). It works in the slot of
the runner it runs as, SLOTS_DIR/<runner> (rkmon_lib/container_layout.py), which root makes at the container's start
(docker/rk_layout.py): the runner's, group rkmon, mode 2751. Which job holds which slot is the monitor's to say
(rkmon_lib/slots.py); the launcher does the work in it:

  sudo -n -u RUNNER LAUNCHER begin      every process of the runner stopped at once, the slot's folder emptied (its
                                        ACLs too) and its four folders (SLOT_FOLDERS) made fresh: the runner's, group
                                        rkmon, mode 2750
  env -0 | sudo -n -u RUNNER LAUNCHER run [-C DIR] -- CMD [ARGS...]
                                        CMD in the slot (below); when it ends, every process it left is stopped
  sudo -n -u RUNNER LAUNCHER kill [--now]
                                        every process of the runner stopped: SIGTERM, then SIGKILL after GRACE_S
                                        (--now: SIGKILL at once). One line
  sudo -n -u RUNNER LAUNCHER end        every process stopped at once, the slot's folder emptied (its ACLs too)
  sudo -n -u RUNNER LAUNCHER keep       every process stopped at once; the slot's folder emptied but for its `build`
                                        (its ACLs too) and its other three folders made fresh, as `begin` makes them;
                                        then the `build` folder made readable by the other runners (o+rX on it and on
                                        everything in it) and writable by none of them, with no set-user-ID or
                                        set-group-ID file, never following a link

`begin`, `keep` and `end` also remove what the runner left in what every user of the container shares, once its
processes are stopped: its own entries in /dev/shm and /dev/mqueue (container_layout SHM_DIR and MQUEUE_DIR, the
folders outside its slot a runner writes), never through a link, and the System V shared memory, semaphores and
message queues it owns or made (/proc/sysvipc). Another user's are left as they are; a line says how many went, and how
many could not. The container makes no namespace of a runner's own, so while two jobs run, one can open what the other
made there when the other's modes let it.

`run` reads the job's environment from stdin (NUL-separated NAME=VALUE, at most worker/pr_env.py's ENV_MAX bytes)
and keeps only what pr_env.py lists; HOME, TMPDIR, RKMON_OUT and RKMON_BUILD (where the native builds go:
worker/build_native.py) are the slot's `home`, `tmp`, `out` and `build`, and USER and LOGNAME the runner. Git fetches
nothing (GIT_NO_LAZY_FETCH: the monitor fetched what the job reads before it started), and takes as safe
(safe.directory) the job's trees only: the one DIR is in and the ones RKMON_TREES names, each a folder of the monitor's
user, which no runner writes. Git refuses any other tree, one a runner made included. CMD runs with stdin /dev/null,
umask 027 and no new privileges (a set-user-ID program runs as the runner, never as its owner), in DIR (absolute, or
relative to the slot's folder; by default its `home`). The launcher stays CMD's parent and the subreaper of what CMD
starts, and hands it SIGTERM, SIGINT and SIGHUP. When CMD ends, every process it left is stopped (SIGKILL): each is the
launcher's descendant, as their subreaper, and the launcher waits until none is left. It signals no other process: a
`kill`, `begin`, `end` or `keep` of the slot, a process of the runner too, stops the runner's other processes. The
result reader reads nothing of the slot while a process of the runner runs (worker/pr_results.py): nothing changes
what the monitor reads there.

PR code runs as the runner, as the launcher does, so it can end the launcher; what it leaves then runs until the
slot's next `kill`, `begin`, `end` or `keep`. A process of the runner that ends every process of the runner's as it
starts (kill(-1) in a loop) ends those too: they fail, the slot's `begin` and `end` included. The monitor then marks
the slot broken (rkmon_lib/slots.py: `rk pr-slot status` and rk doctor show it), and jobs take another slot until the
container's next start stops what is left.

The slot's folder is the runner's, so PR code can change its mode or its group, and the runner, in no group of the
monitor's, cannot set them back: the launcher then refuses the slot until the container's next start makes it again.
So it is with an access ACL on the folder: for a user in no group of the folder's, the kernel takes the folder's
set-group-ID bit with any access ACL removed, even one that is not there, so the launcher removes one only when the
folder has one.
A kept slot (its `build` readable by the other runners) is refused by `run` too: no job runs in a kept slot. Nothing
else of a kept slot is left to read: what PR code put beside its build, or opened to the other runners, is removed.

Exit: 0, or CMD's code for `run` (128 + the signal that ended it); 125 when a process of the runner outlived GONE_S
after SIGKILL; 127 when CMD or DIR cannot be used, the launcher cannot be the subreaper of what CMD starts, or the
monitor's group is missing (one line says which); 3 when the slot cannot be used (not as root made it, not begun,
kept, or not emptied); 2 for a usage error, or a user that is no PR runner.
"""

import ctypes
import errno
import grp
import os
import pwd
import runpy
import signal
import stat
import sys
import time
import types

CODE = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)  # the image's code: read-only, owned by root
LAYOUT = types.SimpleNamespace(
    **runpy.run_path(os.path.join(CODE, "rkmon_lib", "container_layout.py"))
)
PR_ENV = types.SimpleNamespace(
    **runpy.run_path(os.path.join(CODE, "worker", "pr_env.py"))
)
RESULTS = types.SimpleNamespace(
    **runpy.run_path(os.path.join(CODE, "worker", "pr_results.py"))
)
PERMS = types.SimpleNamespace(
    **runpy.run_path(os.path.join(CODE, "worker", "perms.py"))
)
GRACE_S = 10  # `kill`: SIGKILL after this
GONE_S = 30  # a stop waits this long at most for the processes to be gone
# The deepest folder emptying opens (the result reader walks no deeper either); a deeper one is moved up first.
DEPTH_MAX = RESULTS.DEPTH_MAX
# The slot's four folders: read by their group (rkmon), as a tree is.
FOLDER_MODE = PERMS.TREE_DIR_MODE
UMASK = PERMS.UMASK
PATH = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"  # when the job's environment names none
PR_SET_CHILD_SUBREAPER = 36
PR_SET_NO_NEW_PRIVS = 38
FORWARDED = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)  # handed to CMD
_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
# Read once, for kill_all and ipc_rmid.
RUNNER_IDS = frozenset(u.uid for u in LAYOUT.PR_RUNNERS)
IPC_RMID = 0
# The files of /proc/sysvipc, and each one's id column.
SYSV_IPC = (("shm", "shmid"), ("sem", "semid"), ("msg", "msqid"))
USAGE = "usage: begin | run [-C DIR] -- CMD [ARGS...] | kill [--now] | end | keep"


class Refused(Exception):
    """The slot cannot be used: exit 3."""


class Left(Exception):
    """Processes of the runner outlived the stop: exit 125."""


class Missing(Exception):
    """What the start makes is missing (the monitor's group): exit 127."""


def runner(uid: int, getpwuid=pwd.getpwuid):
    """The PR runner (container_layout.PR_RUNNERS) with id `uid`, by its name and its id; None for any other user."""
    try:
        name = getpwuid(uid).pw_name
    except KeyError:
        return None
    return next((u for u in LAYOUT.PR_RUNNERS if u.name == name and u.uid == uid), None)


def monitor_gid(getgrnam=grp.getgrnam) -> int:
    """The monitor's group's id, which each start sets to the deployment's (docker/rk_ids.py); Missing without that
    group."""
    try:
        return getgrnam(LAYOUT.MONITOR.group).gr_gid
    except KeyError:
        raise Missing(
            f"no group {LAYOUT.MONITOR.group} (the monitor's): the container's start makes it"
        ) from None


def monitor_uid(getpwnam=pwd.getpwnam) -> int | None:
    """The monitor's user's id, which each start sets to the deployment's (docker/rk_ids.py); None without that user."""
    try:
        return getpwnam(LAYOUT.MONITOR.name).pw_uid
    except KeyError:
        return None


def open_slot(path: str, uid: int, gid: int) -> int:
    """The slot's folder, opened without following a link; Refused unless it is as root made it: the runner's (`uid`),
    group `gid` (rkmon), container_layout's SLOT_MODE."""
    try:
        fd = os.open(path, _DIR)
    except OSError as exc:
        raise Refused(f"{path}: {exc.strerror}") from None
    s = os.fstat(fd)
    got = (s.st_uid, s.st_gid, stat.S_IMODE(s.st_mode))
    if got != (uid, gid, LAYOUT.SLOT_MODE):
        os.close(fd)
        raise Refused(
            f"{path} is {got[0]}:{got[1]} mode {got[2]:o}, not {uid}:{gid} mode "
            f"{LAYOUT.SLOT_MODE:o}: PR code changed it, and the container's next start makes it again"
        )
    return fd


# --- the runner's processes ------------------------------------------------------------------------------------------


def others(uid: int, me: int, proc: str = "/proc") -> list[int]:
    """The processes of user `uid` that still run (a zombie has ended), but `me`."""
    return [p for p in RESULTS.running(uid, proc) if p != me]


def reap() -> None:
    """Collect the children that ended: CMD's, and the orphans the launcher took over as their subreaper."""
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def kill_all(sig: int) -> None:
    """kill(-1, sig): every process this one may signal, but itself: a PR runner's own. Only with a PR runner's id
    (RUNNER_IDS): as any other user, it would reach every process of that user's."""
    if os.geteuid() not in RUNNER_IDS:
        raise RuntimeError(
            f"kill(-1) is the PR runners' only, not user {os.geteuid()}'s"
        )
    try:
        os.kill(-1, sig)
    except ProcessLookupError:
        pass  # none was left to signal


def descendants(root: int, proc: str = "/proc") -> list[int]:
    """The processes that descend from process `root` (its children, theirs, and so on, by their parents in `proc`) and
    still run: a zombie has ended."""
    below: dict[int, list[tuple[int, str]]] = {}
    for pid, _ids, parent, state in RESULTS.processes(proc):
        below.setdefault(parent, []).append((pid, state))
    out, seen, todo = [], {root}, [root]
    while todo:
        for pid, state in below.get(todo.pop(), ()):
            if pid in seen:
                continue  # a pid used again while `proc` was read
            seen.add(pid)
            todo.append(pid)
            if state != "Z":
                out.append(pid)
    return sorted(out)


def kill_each(pids, sig: int) -> None:
    """`sig` to each process of `pids`: one that has ended, or another user's (its pid used again), is passed over."""
    for pid in pids:
        try:
            os.kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def stop(
    now: bool, left, send=None, sleep=time.sleep, clock=time.monotonic
) -> tuple[int, int]:
    """The processes `left()` lists stopped: `send(sig)` (by default kill_all: every other process of the runner)
    signals them, SIGTERM first and SIGKILL to what `left()` still lists after GRACE_S, or SIGKILL at once (`now`); then
    waits, GONE_S at most, until `left()` is empty. (how many there were, how many needed SIGKILL); Left when some
    outlived it.
    """
    send = send or kill_all
    first = left()
    if not first:
        return 0, 0
    if not now:
        send(signal.SIGTERM)
        end = clock() + GRACE_S
        while left() and clock() < end:
            reap()
            sleep(0.1)
    rest = left()
    hard = len(rest)
    end = clock() + GONE_S
    while rest and clock() < end:
        send(signal.SIGKILL)
        reap()
        sleep(0.1)
        rest = left()
    if rest:
        raise Left(
            f"{len(rest)} process(es) still run {GONE_S} s after SIGKILL: {' '.join(map(str, rest[:8]))}"
        )
    return len(first), hard


# --- the slot's folders ----------------------------------------------------------------------------------------------


def _aside(top: int) -> str:
    """A name no entry of the folder `top` has."""
    n = 0
    while True:
        name = f".deep-{n}"
        try:
            os.stat(name, dir_fd=top, follow_symlinks=False)
        except FileNotFoundError:
            return name
        n += 1


def _chmod(parent: int, name: str, mode: int) -> None:
    """The mode of `name` in the folder `parent` set, never through a link (fchmodat, AT_SYMLINK_NOFOLLOW): on a link,
    OSError (ELOOP)."""
    try:
        os.chmod(name, mode, dir_fd=parent, follow_symlinks=False)
    except NotImplementedError:  # what Python raises when fchmodat refuses a link
        raise OSError(errno.ELOOP, "a link", name) from None


def _remove(parent: int, name: str, top: int, later: list, depth: int) -> None:
    try:
        s = os.stat(name, dir_fd=parent, follow_symlinks=False)
    except FileNotFoundError:
        return
    if not stat.S_ISDIR(s.st_mode):
        os.unlink(name, dir_fd=parent)
        return
    mode = stat.S_IMODE(s.st_mode)
    # PR code may leave a folder its owner cannot list, enter or change.
    if mode & 0o700 != 0o700:
        _chmod(parent, name, mode | 0o700)
    if depth > DEPTH_MAX:
        aside = _aside(top)
        os.rename(name, aside, src_dir_fd=parent, dst_dir_fd=top)
        later.append(aside)
        return
    fd = os.open(name, _DIR, dir_fd=parent)
    try:
        for child in os.listdir(fd):
            _remove(fd, child, top, later, depth + 1)
    finally:
        os.close(fd)
    os.rmdir(name, dir_fd=parent)


def empty(top: int, keep=()) -> None:
    """Remove everything in the folder `top` (a descriptor) but its entries named in `keep`, never following a link. A
    folder deeper than DEPTH_MAX is moved up into `top` first, so the walk never holds more than DEPTH_MAX descriptors.
    """
    later: list[str] = []
    for name in os.listdir(top):
        if name not in keep:
            _remove(top, name, top, later, 1)
    while later:
        _remove(top, later.pop(), top, later, 1)


def clear_shm(uid: int, path: str = "") -> tuple[int, int]:
    """The entries of user `uid` at the top of SHM_DIR (or `path`), the folder every user writes, removed with what they
    hold, never through a link; another user's are left as they are. (entries removed, entries that could not be).
    """
    try:
        fd = os.open(path or LAYOUT.SHM_DIR, _DIR)
    except OSError:
        return 0, 0
    gone = failed = 0
    try:
        for name in os.listdir(fd):
            try:
                if os.stat(name, dir_fd=fd, follow_symlinks=False).st_uid != uid:
                    continue
                later: list[str] = []
                _remove(fd, name, fd, later, 1)
                while later:
                    _remove(fd, later.pop(), fd, later, 1)
                gone += 1
            except OSError:
                failed += 1
    finally:
        os.close(fd)
    return gone, failed


def clear_ipc(uid: int, proc: str = "/proc", remove=None) -> tuple[int, int]:
    """The System V shared memory, semaphores and message queues (`proc`/sysvipc) that user `uid` owns or made, removed
    by `remove(kind, id)` (by default ipc_rmid): every user of the container shares them, as SHM_DIR; another user's
    are left as they are. (objects removed, objects that could not be).
    """
    remove = remove or ipc_rmid
    gone = failed = 0
    for kind, column in SYSV_IPC:
        try:
            with open(
                f"{proc}/sysvipc/{kind}", encoding="ascii", errors="replace"
            ) as f:
                head, *rows = f.read().splitlines() or [""]
        except OSError:
            continue
        names = head.split()
        if not {column, "uid", "cuid"} <= set(names):
            continue
        at = [names.index(n) for n in (column, "uid", "cuid")]
        for row in rows:
            cols = row.split()
            try:
                ident, owner, maker = (int(cols[i]) for i in at)
            except (IndexError, ValueError):
                continue
            if uid not in (owner, maker):
                continue
            try:
                remove(kind, ident)
                gone += 1
            except OSError:
                failed += 1
    return gone, failed


def ipc_rmid(kind: str, ident: int) -> None:
    """The System V object `ident` of `kind` (SYSV_IPC: shm, sem or msg) removed (IPC_RMID); one removed already is
    passed over. Only with a PR runner's id (RUNNER_IDS), as kill_all."""
    if os.geteuid() not in RUNNER_IDS:
        raise RuntimeError(
            f"IPC_RMID is the PR runners' only, not user {os.geteuid()}'s"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    if kind == "shm":
        rc = libc.shmctl(ident, IPC_RMID, None)
    elif kind == "sem":
        rc = libc.semctl(ident, 0, IPC_RMID)
    else:
        rc = libc.msgctl(ident, IPC_RMID, None)
    if rc != 0:
        e = ctypes.get_errno()
        if e not in (errno.EINVAL, errno.EIDRM):
            raise OSError(e, os.strerror(e), f"{kind} {ident}")


def drop_acls(top: int) -> None:
    """The ACLs PR code set on the slot's folder `top` removed (worker/perms.py drop_acls: its access ACL only when it
    has one), so that none reaches what is made in it next. Removing an access ACL takes the folder's set-group-ID bit
    from a user in no group of the folder's, as a runner is, and the runner cannot set it again: Refused when the
    folder has lost it.
    """
    PERMS.drop_acls(top)
    if not os.fstat(top).st_mode & stat.S_ISGID:
        raise Refused(
            "the slot's folder had an access ACL, and removing it took the folder's set-group-ID bit, which a runner "
            "cannot set again: the container's next start makes the folder again"
        )


def make_folders(top: int, gid: int, names=None) -> None:
    """The four SLOT_FOLDERS (or those of them in `names`) in the emptied slot folder `top`: the runner's, with the
    group `gid` and the setgid bit of the slot's folder (a runner, in no group of the monitor's, cannot set them
    itself), mode 2750 under umask 027.
    """
    for name in LAYOUT.SLOT_FOLDERS if names is None else names:
        os.mkdir(name, FOLDER_MODE & 0o777, dir_fd=top)
        s = os.stat(name, dir_fd=top, follow_symlinks=False)
        if (s.st_gid, stat.S_IMODE(s.st_mode)) != (gid, FOLDER_MODE):
            raise Refused(
                f"{name} was made {s.st_gid} mode {stat.S_IMODE(s.st_mode):o}, not {gid} mode {FOLDER_MODE:o}"
            )


def begun(top: int, uid: int) -> None:
    """Refused unless the slot was begun: its four folders are there, folders of the runner's (`uid`)."""
    for name in LAYOUT.SLOT_FOLDERS:
        try:
            s = os.stat(name, dir_fd=top, follow_symlinks=False)
        except OSError:
            raise Refused(f"no {name} folder: the slot was not begun") from None
        if not stat.S_ISDIR(s.st_mode) or s.st_uid != uid:
            raise Refused(f"{name} is not a folder of the runner's")


def fresh_but_build(top: int, gid: int) -> None:
    """The slot's folder `top` emptied but for its `build`, its ACLs dropped, and its other SLOT_FOLDERS made again
    (`make_folders`): what PR code left in the slot's other folders or beside them, or opened to the other runners there,
    is gone. OSError, or Refused, when that cannot be done."""
    empty(top, keep=(LAYOUT.SLOT_BUILD,))
    drop_acls(top)  # none of PR code's reaches the folders made again
    make_folders(top, gid, [n for n in LAYOUT.SLOT_FOLDERS if n != LAYOUT.SLOT_BUILD])


def kept(top: int) -> bool:
    """Whether the slot is kept: its `build` folder is open to the other runners."""
    return bool(
        os.stat(LAYOUT.SLOT_BUILD, dir_fd=top, follow_symlinks=False).st_mode & 0o007
    )


def _readable(parent: int, name: str, depth: int, count: list) -> None:
    s = os.stat(name, dir_fd=parent, follow_symlinks=False)
    # Writable by no other runner: not by the others, and not by its group, whose bits are an ACL's mask: no named
    # entry of an ACL writes either.
    mode = stat.S_IMODE(s.st_mode) & ~0o022
    if stat.S_ISREG(s.st_mode):
        # and whoever runs it runs it as themselves
        mode &= ~(stat.S_ISUID | stat.S_ISGID)
        _chmod(parent, name, mode | 0o004 | (0o001 if mode & 0o100 else 0))
        count[0] += 1
        return
    if not stat.S_ISDIR(s.st_mode):  # a link, a FIFO, a socket
        count[1] += 1
        return
    _chmod(parent, name, mode | 0o505)
    count[0] += 1
    if depth >= DEPTH_MAX:
        count[1] += 1
        return
    fd = os.open(name, _DIR, dir_fd=parent)
    try:
        for child in os.listdir(fd):
            try:
                _readable(fd, child, depth + 1, count)
            except OSError:
                count[1] += 1
    finally:
        os.close(fd)


def readable(top: int) -> tuple[int, int]:
    """The slot's `build` folder and everything in it readable by the other runners: a folder o+rx (and u+rx), a file
    o+r (o+x when its owner may run it); none writable by its group or the others (g-w, o-w), and no file set-user-ID or
    set-group-ID; each changed without following a link. A link, and anything that is neither a folder nor a file, is
    left as it is, and so is what lies deeper than DEPTH_MAX. It runs while no process of the runner runs, so nothing
    changes the tree under it. (entries made readable, entries left).
    """
    count = [0, 0]
    _readable(top, LAYOUT.SLOT_BUILD, 1, count)
    return count[0], count[1]


# --- run -------------------------------------------------------------------------------------------------------------


def git_top(path: str) -> str:
    """The top of the git tree `path` is in, resolved: the nearest folder up from it that holds a `.git` (a folder, a
    file or a link: never followed); "" when none does."""
    d = os.path.realpath(path)
    while True:
        try:
            os.lstat(os.path.join(d, ".git"))
            return d
        except OSError:
            up = os.path.dirname(d)
            if up == d:
                return ""
            d = up


def trees(paths, owner: int | None) -> list[str]:
    """Of `paths`, the trees git may take as safe for the job: each absolute one, resolved, that is a folder of the user
    `owner`'s (the monitor's, which no runner writes), once, in their order; none when `owner` is None.
    """
    out: list[str] = []
    if owner is None:
        return out
    for p in paths:
        if not p or not os.path.isabs(p):
            continue
        real = os.path.realpath(p)
        try:
            s = os.lstat(real)
        except OSError:
            continue
        if stat.S_ISDIR(s.st_mode) and s.st_uid == owner and real not in out:
            out.append(real)
    return out


def job_env(given: dict, slot: str, name: str, safe=()) -> dict:
    """The environment CMD runs with: what worker/pr_env.py lists of `given` (the job's), the slot's home, temporary
    folder, results folder and build folder, the runner's name; git that fetches nothing, with safe.directory for each
    tree of `safe` (`trees`) and for no other folder.
    """
    env = PR_ENV.allowed(given)
    if not env.get("PATH"):
        env["PATH"] = PATH
    env.update(
        HOME=f"{slot}/{LAYOUT.SLOT_HOME}",
        TMPDIR=f"{slot}/{LAYOUT.SLOT_TMP}",
        RKMON_OUT=f"{slot}/{LAYOUT.SLOT_OUT}",
        RKMON_BUILD=f"{slot}/{LAYOUT.SLOT_BUILD}",
        USER=name,
        LOGNAME=name,
        GIT_NO_LAZY_FETCH="1",
    )
    if safe:
        env["GIT_CONFIG_COUNT"] = str(len(safe))
        for i, tree in enumerate(safe):
            env[f"GIT_CONFIG_KEY_{i}"] = "safe.directory"
            env[f"GIT_CONFIG_VALUE_{i}"] = tree
    return env


def parse_run(args: list[str]) -> tuple[str, list[str]] | None:
    """(DIR or "", CMD) of `[-C DIR] -- CMD...`; None when that is not what `args` is."""
    where = ""
    if args[:1] == ["-C"] and len(args) >= 2:
        where, args = args[1], args[2:]
    if args[:1] != ["--"] or len(args) < 2:
        return None
    return where, args[1:]


def prctl_on(option: int, what: str) -> None:
    """prctl(option, 1): an OSError that names `what` when the kernel refuses."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(option, 1, 0, 0, 0) != 0:
        e = ctypes.get_errno()
        raise OSError(e, os.strerror(e), what)


def subreaper() -> None:
    """This process the subreaper of what it starts: an orphan of CMD's becomes its child, not PID 1's. OSError when
    the kernel refuses."""
    prctl_on(PR_SET_CHILD_SUBREAPER, "the subreaper of what CMD starts")


def no_new_privs() -> None:
    """This process, and every one it starts, gains no privilege by exec: a set-user-ID or set-group-ID program, or one
    with file capabilities, runs as the runner, with the runner's rights. OSError when the kernel refuses.
    """
    prctl_on(PR_SET_NO_NEW_PRIVS, "no new privileges")


def run_cmd(
    cmd: list[str],
    cwd: str,
    env: dict,
    left=None,
    send=None,
    execvpe=os.execvpe,
    become_subreaper=subreaper,
    sleep=time.sleep,
    seal=no_new_privs,
) -> int:
    """CMD in `cwd` with `env`, stdin /dev/null and no new privileges (`seal`), as this process's child, which gets the
    SIGTERM, SIGINT and SIGHUP this process gets; then every process CMD left stopped (SIGKILL): `left()` lists them,
    by default this process's descendants (it is their subreaper), and `send(sig)` signals them, by default each of
    those and no other process. Its exit code, 128 + the signal that ended it, 127 when it could not start or this
    process cannot be the subreaper of what it starts (`become_subreaper`: CMD is then not run), 125 when processes
    outlived the stop.
    """
    me = os.getpid()
    left = left or (lambda: descendants(me))
    send = send or (lambda sig: kill_each(left(), sig))
    try:
        become_subreaper()
    except OSError as exc:
        print(
            f"rk_prcode.py: {exc.filename}: {exc.strerror}: CMD not run",
            file=sys.stderr,
        )
        return 127
    child = [0]

    def forward(sig, _frame):
        if child[0]:
            try:
                os.kill(child[0], sig)
            except ProcessLookupError:
                pass

    saved = {s: signal.signal(s, forward) for s in FORWARDED}
    try:
        # Blocked until the child's id is known: one that comes before is handed over then, never lost.
        signal.pthread_sigmask(signal.SIG_BLOCK, FORWARDED)
        try:
            pid = os.fork()
            if (
                pid == 0
            ):  # the child: CMD, with the default handlers, and what came before it starts ends it
                code = 127
                try:
                    for s in FORWARDED:
                        signal.signal(s, signal.SIG_DFL)
                    signal.pthread_sigmask(signal.SIG_UNBLOCK, FORWARDED)
                    fd = os.open(os.devnull, os.O_RDONLY)
                    os.dup2(fd, 0)
                    os.close(fd)
                    os.chdir(cwd)
                    seal()
                    execvpe(cmd[0], cmd, env)
                except OSError as exc:
                    os.write(
                        2,
                        f"rk_prcode.py: {exc.filename or cmd[0]}: {exc.strerror}\n".encode(),
                    )
                finally:
                    os._exit(code)
            child[0] = pid
        finally:
            signal.pthread_sigmask(signal.SIG_UNBLOCK, FORWARDED)
        _pid, status = os.waitpid(pid, 0)
        code = os.waitstatus_to_exitcode(status)
        code = 128 - code if code < 0 else code
        try:
            stop(True, left, send=send, sleep=sleep)
        except Left as exc:
            print(f"rk_prcode.py: {exc}", file=sys.stderr)
            return 125
        reap()
        return code
    finally:
        for s, h in saved.items():
            signal.signal(s, h)


# --- the actions -----------------------------------------------------------------------------------------------------


def main(argv: list[str], stdin=None) -> int:
    uid = os.geteuid()
    me = runner(uid)
    if me is None:
        print(
            f"rk_prcode.py: runs as a PR runner only ({LAYOUT.PR_RUNNER.name} to {LAYOUT.PR_RUNNERS[-1].name}, "
            f"through sudo), not as user {uid}",
            file=sys.stderr,
        )
        return 2
    os.umask(UMASK)
    act, rest = (argv[0], argv[1:]) if argv else ("", [])
    slot = os.path.join(LAYOUT.SLOTS_DIR, me.name)
    pid = os.getpid()

    def left():
        return others(uid, pid)

    def say(text: str) -> None:
        print(f"PR_SLOT {me.name}: {text}", flush=True)

    def removed(where: str, what: str, gone: int, failed: int) -> None:
        if gone or failed:
            say(
                f"{where}: removed {gone} of the runner's {what}"
                + (f"; {failed} could not be removed" if failed else "")
            )

    try:
        if act == "kill" and rest in ([], ["--now"]):
            n, hard = stop(rest == ["--now"], left)
            say(
                f"stopped {n} process(es)"
                + (f"; {hard} needed SIGKILL" if hard else "")
            )
            return 0
        if act in ("begin", "end", "keep") and not rest:
            n, _hard = stop(True, left)
            if n:
                say(f"stopped {n} process(es)")
            for where in (LAYOUT.SHM_DIR, LAYOUT.MQUEUE_DIR):
                removed(where, "entries", *clear_shm(uid, where))
            removed("System V IPC", "objects", *clear_ipc(uid))
            top = open_slot(slot, uid, monitor_gid())
            try:
                if act == "keep":
                    begun(top, uid)
                    try:
                        fresh_but_build(top, monitor_gid())
                    except OSError as exc:
                        raise Refused(f"not kept: {exc}") from None
                    done, skipped = readable(top)
                    say(
                        f"kept: its build is readable by the other runners ({done} entries"
                        + (f"; {skipped} left as they are" if skipped else "")
                        + ")"
                    )
                    return 0
                try:
                    empty(top)
                    # no ACL PR code set reaches what is made in it next
                    drop_acls(top)
                except OSError as exc:
                    raise Refused(f"not emptied: {exc}") from None
                if act == "begin":
                    try:
                        make_folders(top, monitor_gid())
                    except OSError as exc:
                        raise Refused(f"its folders were not made: {exc}") from None
                return 0
            finally:
                os.close(top)
        if act == "run" and (parsed := parse_run(rest)):
            where, cmd = parsed
            top = open_slot(slot, uid, monitor_gid())
            try:
                begun(top, uid)
                if kept(top):
                    raise Refused("the slot is kept: no job runs in a kept slot")
            finally:
                os.close(top)
            try:
                given = PR_ENV.read(
                    sys.stdin.buffer if stdin is None else stdin, PR_ENV.ENV_MAX
                )
            except ValueError as exc:
                print(f"rk_prcode.py: {exc}", file=sys.stderr)
                return 2
            cwd = os.path.join(slot, where) if where else f"{slot}/{LAYOUT.SLOT_HOME}"
            named = given.get(PR_ENV.TREES, "").split(":")
            safe = trees([git_top(cwd), *named], monitor_uid())
            return run_cmd(cmd, cwd, job_env(given, slot, me.name, safe))
    except Refused as exc:
        print(f"PR_SLOT {me.name}: refused: {exc}", file=sys.stderr, flush=True)
        return 3
    except Left as exc:
        print(f"PR_SLOT {me.name}: {exc}", file=sys.stderr, flush=True)
        return 125
    except Missing as exc:
        print(f"PR_SLOT {me.name}: {exc}", file=sys.stderr, flush=True)
        return 127
    print(USAGE, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
