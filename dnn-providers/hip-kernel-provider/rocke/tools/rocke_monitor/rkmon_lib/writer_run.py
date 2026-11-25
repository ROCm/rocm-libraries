# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The writer's run of one rk command in the container, as the write broker (docker/rk_write.py) starts it: from the
code tree the broker checked (TREE, this module's own), as the writer, in the folder the broker made for the run's
copies of its file arguments.

`main(argv)` runs the writer's commands alone, as they are listed (rkmon_lib/writer.py refused): the commands of
rkmon_lib/outward.py's table that name the writer, rk's GitHub write commands and the sends to the team chat, the web
host and JIRA among them (none approves a review, merges at once or dismisses one). Two of them, `rk secret-set` for a
secret the writer keeps and `rk web-host-install`, only the host asks for, never through sudo (writer.FROM_HOST,
writer.sudo_refusal). It runs TREE's rkmon.py in its place (`python3 -I -B`), as the writer, with a fixed environment:
the image's (BASE_ENV: the monitor's home, and RKMON_CONTAINER, by which rk takes the write login's folder to be
container_layout.WRITE_LOGIN_DIR), TREE as the code every rk it starts runs (RKMON_CODE), and of the caller's, only
the name the host's wrapper was run by, which the fixes show, checked again (container_layout.shown_wrapper). Every
guard of the command runs there, as it does outside the container, and so does the one outward gate:
nothing goes out for a feature that is off or in prototype, or while the monitor is in standby (taking off an
auto-merge the monitor armed aside, in prototype), however the request came (the table's gate column).
A request that starts with the word outward.OWN_WORD is a write of the monitor's own (rkmon_lib/writer.py routed): the
word is taken off, and the command runs with outward.OWN_VAR set, so that the gate keeps what it stops as an unsent
item rather than refuse it. Each command it runs or refuses is one line of LOG.

The writer can read what the caller cannot (its secrets above all), so it also refuses:
- for a command whose arguments may name a file (writer.FILE_ARGS), a file named in the arguments that the caller
  could not read itself, or one in the writer's secrets volume or its home; a folder; and an absolute path that names
  nothing yet;
- a configuration whose folders lie in those two places (rk would write there as the writer).

A file argument is opened once, by a walk from `/` that follows no symbolic link (each folder opened under the one
before it, the file last, all with O_NOFOLLOW), and checked on what was opened (fstat: the caller may read it, no folder
of the walk is a protected one). Its content is then copied into the run's folder, the working folder, and rk gets that
copy's path: whatever the caller changes afterwards (a link swapped in where the file was, a file made under a name
that did not exist yet) is never what rk reads.
"""

from __future__ import annotations

import datetime as dt
import os
import pwd
import stat
import sys

from rkmon_lib import container_layout, outward, writer

TREE = os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))
)  # the code tree the broker checked
WRITE_DIR = container_layout.WRITE_LOGIN_DIR
LOG = f"{container_layout.WORK_DIR}/{container_layout.LOGS_FOLDER}/rk-write.log"
# The image's environment the commands need (docker/Dockerfile ENV), set here: the broker passes none of the caller's
# but sudo's word of who called and the wrapper's name (docker/rk_write.py).
# Its PATH starts, as the image's does, with the GitHub request cap's counter (rkmon_lib/gh_cap.py).
BASE_ENV = {
    "PATH": f"{container_layout.GH_COUNTER_DIR}:/usr/local/bin:/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "PYTHONDONTWRITEBYTECODE": "1",
    "RKMON_CONTAINER": "1",
    "RKMON_HOME": container_layout.MONITOR_HOME,
    "SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt",
    "REQUESTS_CA_BUNDLE": "/etc/ssl/certs/ca-certificates.crt",
}
ARG_MAX_BYTES = (
    8 << 20
)  # a file argument larger than this is refused (a post or a notice is a few kB)


def _inside(path: str, folders: list[str]) -> bool:
    return any(path == f or path.startswith(f.rstrip("/") + "/") for f in folders)


def caller_may(s: os.stat_result, environ, read: bool = True) -> bool:
    """Whether the user sudo ran this for (SUDO_UID, SUDO_GID, its groups) may read (`read`) or search (a folder on the
    way) what `s` describes."""
    try:
        uid, gid = int(environ["SUDO_UID"]), int(environ["SUDO_GID"])
        groups = set(os.getgrouplist(pwd.getpwuid(uid).pw_name, gid))
    except (OSError, KeyError, ValueError):
        return False
    want = stat.S_IRUSR if read else stat.S_IXUSR
    if stat.S_ISDIR(s.st_mode) and read:
        want |= stat.S_IXUSR
    if s.st_uid == uid:
        bits = want
    elif s.st_gid in groups:
        bits = want >> 3
    else:
        bits = want >> 6
    return s.st_mode & bits == bits


def open_checked(path: str, protected: list[str], environ) -> int:
    """A read-only descriptor of the regular file `path` names, opened by a walk from `/` that follows no symbolic link
    (the path is resolved first; a link swapped in on the way since then fails the walk), after checking what was
    opened: the caller may search every folder of the walk and read the file, and no folder of the walk is a
    `protected` one (by device and inode, so another name for it is caught too). ValueError otherwise.
    """
    real = os.path.realpath(path)
    if _inside(real, protected):
        raise ValueError(f"{path}: not a file the caller may read")
    guard = set()
    for p in protected:
        try:
            s = os.stat(p)
            guard.add((s.st_dev, s.st_ino))
        except OSError:
            pass
    parts = [p for p in real.split("/") if p]
    if not parts:
        raise ValueError(f"{path}: not a file")
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for i, name in enumerate(parts):
            last = i == len(parts) - 1
            flags = os.O_NOFOLLOW | os.O_RDONLY | os.O_CLOEXEC
            flags |= os.O_NONBLOCK if last else os.O_DIRECTORY
            try:
                nxt = os.open(name, flags, dir_fd=fd)
            except OSError as exc:  # a link (ELOOP, ENOTDIR), gone, or not ours to open
                raise ValueError(
                    f"{path}: not a file the caller may read ({exc.strerror})"
                ) from None
            os.close(fd)
            fd = nxt
            s = os.fstat(fd)
            if (s.st_dev, s.st_ino) in guard:
                raise ValueError(f"{path}: not a file the caller may read")
            if not last and not caller_may(s, environ, read=False):
                raise ValueError(f"{path}: not a file the caller may read")
        if not stat.S_ISREG(s.st_mode):
            raise ValueError(f"{path}: not a file (no rk write command takes another)")
        if not caller_may(s, environ):
            raise ValueError(f"{path}: not a file the caller may read")
        if s.st_size > ARG_MAX_BYTES:
            raise ValueError(f"{path}: larger than {ARG_MAX_BYTES} bytes")
        out, fd = fd, -1
        return out
    finally:
        if fd >= 0:
            os.close(fd)


def _copy(fd: int, folder: str, name: str) -> str:
    """The content of `fd` (closed here) as a new file `name` in `folder`, mode 600; its path."""
    dest = os.path.join(folder, os.path.basename(name) or "arg")
    n = 1
    while os.path.lexists(dest):
        n += 1
        dest = os.path.join(folder, f"{n}-{os.path.basename(name) or 'arg'}")
    with os.fdopen(fd, "rb") as src, open(
        os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600),
        "wb",
    ) as out:
        out.write(src.read(ARG_MAX_BYTES + 1)[:ARG_MAX_BYTES])
    return dest


def guarded_args(
    argv: list[str], protected: list[str], environ, copies: str
) -> list[str]:
    """`argv`, each argument that names a file (alone, or as an option's =value) as a checked copy in `copies`
    (open_checked: the file the caller could read when the writer opened it, never what that name names later).
    ValueError for a file the caller could not read itself, one in a `protected` folder, a folder, or an absolute path
    that names nothing (rk's working folder is `copies`, so a relative one can only name a copy). The command's name
    and plain numbers are never files.
    """
    out = []
    for i, a in enumerate(argv):
        opt, eq, val = (
            a.partition("=") if a.startswith("--") and "=" in a else ("", "", a)
        )
        if i and val and not val.isdigit():
            if os.path.lexists(val):
                fd = open_checked(val, protected, environ)
                val = _copy(fd, copies, val)
            elif val.startswith("/"):  # a relative name resolves among the copies only
                raise ValueError(
                    f"{val}: names no file (a file argument must exist when the writer reads it)"
                )
        out.append(f"{opt}{eq}{val}" if eq else val)
    return out


def config_folders_outside(protected: list[str]) -> str:
    """ "" when the configuration's folders lie outside `protected`, else the first that does not. It is read as the
    writer's rk reads it (BASE_ENV's home); this process's environment is left as it was.
    """
    from rkmon_lib import config, paths

    keys = ("RKMON_HOME", "RKMON_CONTAINER")
    before = {k: os.environ.get(k) for k in keys}
    os.environ.update({k: BASE_ENV[k] for k in keys})
    try:
        cfg = config.load(check=False)
        folders = paths.resolve(cfg)
    except Exception:  # noqa: BLE001 - rk itself says what is wrong with it
        return ""
    finally:
        for k, v in before.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    for key, value in folders.items():
        if value and _inside(os.path.realpath(value), protected):
            return f"paths.{key} ({value})"
    return ""


def log(line: str) -> None:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(f"{stamp} {line}\n")
    except OSError:
        print(f"rk-write: {line}", file=sys.stderr)


def refuse(why: str, argv: list[str]) -> int:
    """Refuse rk ARGV for WHY: one line of the log, and the writer's one refusal line (writer.refusal); exit 4."""
    log(f"REFUSED {' '.join(argv)[:300]}: {why}")
    print(writer.refusal(why))
    return 4


def main(argv: list[str], environ=None, execve=os.execve) -> int:
    """Run rk ARGV as the writer, in this process's place, or refuse it (exit 4): see the module's docstring."""
    from rkmon_lib import paths

    e = os.environ if environ is None else environ
    me = pwd.getpwuid(os.geteuid())
    own = argv[:1] == [outward.OWN_WORD]  # a write of the monitor's own (writer.routed)
    argv = argv[1:] if own else argv
    why = writer.refused(argv) or writer.sudo_refusal(argv, e)
    if why:
        return refuse(why, argv)
    protected = [
        os.path.realpath(os.path.dirname(WRITE_DIR)),
        os.path.realpath(me.pw_dir),
    ]
    copies = os.getcwd()  # the broker's folder for this run
    try:
        if argv[0] in writer.FILE_ARGS:
            argv = guarded_args(argv, protected, e, copies)
    except ValueError as exc:
        return refuse(str(exc), argv)
    except OSError as exc:
        return refuse(f"a file argument could not be copied ({exc})", argv)
    bad = config_folders_outside(protected)
    if bad:
        return refuse(
            f"the configuration puts {bad} where only the writer may write", argv
        )
    env = dict(
        BASE_ENV,
        HOME=me.pw_dir,
        USER=me.pw_name,
        RKMON_GH_BROKER="1",
        RKMON_CODE=TREE,
        **{container_layout.WRAPPER_VARIABLE: container_layout.shown_wrapper(e)},
        **({outward.OWN_VAR: "1"} if own else {}),
    )
    whose = ", the monitor's own" if own else ""
    log(f"rk {' '.join(argv)[:300]} (as {me.pw_name}{whose})")
    # what it writes beside the monitor's files takes their folder's default ACL (docker/rk_layout.py): 0660, 2770
    os.umask(paths.worker_module("perms").UMASK)
    execve(sys.executable, [sys.executable, "-I", "-B", f"{TREE}/rkmon.py", *argv], env)
    return 127  # execve returns only in the tests
