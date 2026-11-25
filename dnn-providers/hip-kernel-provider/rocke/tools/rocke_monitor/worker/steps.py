# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The worker scripts' steps (mkwt.py, review_agent_run.py, build_native.py, run_suite.py, kill_runner.py, and
tools/draft_run.py and tools/quality_run.py): run a program as a shell script runs it, with its output straight to
this process's stdout and stderr in order, and the few file chores the scripts do (the last lines of a log, the lines
of a log that match, a folder made writable and removed).

`run` returns the program's exit code; `must` ends the script with it when it is not 0 (a shell's `set -e`); a
program that cannot start reads as a shell's 127 ("not found") or 126 (not executable), with the shell's message on
stderr. Nothing here starts a shell.
"""

import contextlib
import datetime as dt
import glob
import os
import pathlib
import re
import shutil
import stat
import subprocess
import sys

PROG = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "worker"


def echo(*parts: str) -> None:
    """One line on stdout, at once (a child's output that follows comes after it)."""
    print(*parts, flush=True)


def _flush() -> None:
    sys.stdout.flush()
    sys.stderr.flush()


def run(argv: list, **kw) -> int:
    """argv as a program, its output to ours unless `kw` redirects it; its exit code."""
    _flush()
    try:
        return subprocess.run([str(a) for a in argv], **kw).returncode
    except FileNotFoundError:
        why, rc = (
            "No such file or directory" if "/" in str(argv[0]) else "command not found"
        ), 127
    except PermissionError:
        why, rc = "Permission denied", 126
    # The shell's message goes where the program's stderr would have gone.
    where = kw.get("stderr")
    if where == subprocess.STDOUT:
        where = kw.get("stdout")
    if where != subprocess.DEVNULL:
        line = f"{PROG}: {argv[0]}: {why}\n"
        if hasattr(where, "write"):
            where.write(line.encode() if "b" in getattr(where, "mode", "") else line)
            where.flush()
        else:
            sys.stderr.write(line)
            sys.stderr.flush()
    return rc


def must(argv: list, **kw) -> int:
    """run(), and the script ends with the program's exit code when it is not 0 (the shell's `set -e`)."""
    rc = run(argv, **kw)
    if rc:
        sys.exit(rc)
    return rc


def capture(argv: list, **kw) -> tuple[int, str]:
    """argv's stdout as a shell's $(...) gives it (its trailing newlines dropped), and its exit code; its stderr goes
    to ours."""
    _flush()
    try:
        p = subprocess.run([str(a) for a in argv], stdout=subprocess.PIPE, **kw)
    except FileNotFoundError:
        print(
            f"{PROG}: {argv[0]}: {'No such file or directory' if '/' in str(argv[0]) else 'command not found'}",
            file=sys.stderr,
            flush=True,
        )
        return 127, ""
    return p.returncode, p.stdout.decode("utf-8", "surrogateescape").rstrip("\n")


# The review clone is shared: two stagings that fetch into its shallow history at once fail ("shallow file has changed
# since we read it"). Every fetch into it, and every staging of worktrees from it, holds this lock (an exclusive flock
# in the clone's git folder; rkmon_lib/gitops.py takes the same file for the toolkit's fetches).
CLONE_LOCK = "rkmon-clone.lock"
CLONE_LOCK_WAIT_S = 1800


def clone_lock_path(repo) -> str | None:
    """The lock file in the clone's git folder (`.git`, or the folder itself when it is a bare repository); None when
    `repo` is no git repository."""
    git_dir = os.path.join(str(repo), ".git")
    if os.path.isdir(git_dir):
        return os.path.join(git_dir, CLONE_LOCK)
    if os.path.isdir(os.path.join(str(repo), "objects")):
        return os.path.join(str(repo), CLONE_LOCK)
    return None


@contextlib.contextmanager
def clone_lock(repo, wait_s: float = CLONE_LOCK_WAIT_S):
    """Hold the clone's lock (waiting at most `wait_s` seconds; past it, go on without it, saying so)."""
    import fcntl
    import time

    path = clone_lock_path(repo)
    try:
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644) if path else None
    except OSError:
        fd = None
    if fd is None:  # no repository there, or its git folder is not writable
        yield False
        return
    held, end = False, time.monotonic() + wait_s
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = True
                break
            except BlockingIOError:
                if time.monotonic() >= end:
                    echo(
                        f"(the clone's lock was still held after {wait_s:.0f} s: going on without it)"
                    )
                    break
                time.sleep(0.5)
        yield held
    finally:
        os.close(fd)


SHALLOW_RACE = "shallow file has changed since we read it"


def must_fetch(argv: list, retry_s: float = 5) -> int:
    """must() for a fetch into the shared clone: one that fails on the shallow-file race (another git changed the
    clone's shallow history while this one read it) runs once more after `retry_s` seconds.
    """
    _flush()
    p = subprocess.run([str(a) for a in argv], stderr=subprocess.PIPE)
    err = p.stderr.decode("utf-8", "replace")
    if p.returncode and SHALLOW_RACE in err:
        sys.stderr.write(err)
        echo(f"(the clone changed under the fetch: once more in {retry_s:g} s)")
        import time

        time.sleep(retry_s)
        p = subprocess.run([str(a) for a in argv], stderr=subprocess.PIPE)
        err = p.stderr.decode("utf-8", "replace")
    sys.stderr.write(err)
    sys.stderr.flush()
    if p.returncode:
        sys.exit(p.returncode)
    return 0


def write_raw(data: bytes) -> None:
    _flush()
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()


def last_lines(data: bytes, n: int) -> bytes:
    """`tail -n N` of `data`."""
    if not data or n <= 0:
        return b""
    lines = data.split(b"\n")
    if data.endswith(b"\n"):
        lines = [x + b"\n" for x in lines[:-1]]
    else:
        lines = [x + b"\n" for x in lines[:-1]] + [lines[-1]]
    return b"".join(lines[-n:])


def tail(path, n: int) -> int:
    """`tail -n N FILE`: its last lines on stdout; 1 when it cannot be read."""
    try:
        data = pathlib.Path(path).read_bytes()
    except OSError as exc:
        _flush()
        print(
            f"tail: cannot open '{path}' for reading: {exc.strerror}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    write_raw(last_lines(data, n))
    return 0


def grep(pattern: str, path) -> int:
    """`grep -E PATTERN FILE`: the matching lines on stdout; 0 with a match, 1 without, 2 when it cannot be read."""
    try:
        data = pathlib.Path(path).read_bytes()
    except OSError as exc:
        _flush()
        print(f"grep: {path}: {exc.strerror}", file=sys.stderr, flush=True)
        return 2
    rx = re.compile(pattern.encode())
    lines = data.split(b"\n")
    if data.endswith(b"\n"):
        lines = lines[:-1]
    hits = [x + b"\n" for x in lines if rx.search(x)]
    write_raw(b"".join(hits))
    return 0 if hits else 1


def found(pattern: str, path) -> bool:
    """`grep -q PATTERN FILE 2>/dev/null` (a basic pattern made of plain characters)."""
    try:
        data = pathlib.Path(path).read_bytes()
    except OSError:
        return False
    rx = re.compile(pattern.encode())
    return any(rx.search(x) for x in data.split(b"\n"))


def utc_stamp() -> str:
    """`date -u +%FT%TZ`."""
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_writable(path) -> None:
    """`chmod -R u+w PATH 2>/dev/null`: links met inside are left alone."""
    top = str(path)
    if not os.path.exists(top):
        return
    todo = [top]
    if os.path.isdir(top):
        for d, dirs, files in os.walk(top, onerror=lambda e: None):
            todo += [os.path.join(d, x) for x in dirs + files]
    for p in todo:
        try:
            st = os.stat(p) if p == top else os.lstat(p)
            if not stat.S_ISLNK(st.st_mode) and not st.st_mode & stat.S_IWUSR:
                os.chmod(p, st.st_mode | stat.S_IWUSR)
        except OSError:
            pass


def remove(path) -> None:
    """`rm -rf PATH` (a link is removed, never followed)."""
    p = str(path)
    if os.path.islink(p) or (os.path.lexists(p) and not os.path.isdir(p)):
        try:
            os.unlink(p)
        except OSError as exc:
            print(
                f"rm: cannot remove '{p}': {exc.strerror}", file=sys.stderr, flush=True
            )
    elif os.path.isdir(p):

        def failed(_func, q, exc):
            err = exc[1] if isinstance(exc, tuple) else exc
            print(
                f"rm: cannot remove '{q}': {getattr(err, 'strerror', err)}",
                file=sys.stderr,
                flush=True,
            )

        if sys.version_info >= (3, 12):
            shutil.rmtree(p, onexc=failed)
        else:
            shutil.rmtree(p, onerror=failed)


def make_dirs(*paths) -> int:
    """`mkdir -p PATH...`: 0, or 1 (with mkdir's message) when one cannot be made."""
    rc = 0
    for p in paths:
        try:
            os.makedirs(str(p), exist_ok=True)
        except OSError as exc:
            _flush()
            print(
                f"mkdir: cannot create directory '{p}': {exc.strerror}",
                file=sys.stderr,
                flush=True,
            )
            rc = 1
    return rc


def link(target, name) -> None:
    """`ln -sfn TARGET NAME`: NAME becomes a link to TARGET, replacing a file or a link there."""
    name = str(name)
    if os.path.islink(name) or (os.path.lexists(name) and not os.path.isdir(name)):
        os.unlink(name)
    os.symlink(str(target), name)


def words(text: str) -> list[str]:
    """An unquoted shell expansion: split on white space, each word with a wildcard expanded to the names it matches
    (sorted), or kept as it is when it matches none."""
    out = []
    for w in (text or "").split():
        hits = sorted(glob.glob(w)) if any(c in w for c in "*?[") else []
        out += hits or [w]
    return out


def cpu_count() -> int:
    """The CPUs this process may run on (its CPU affinity), else the machine's."""
    try:
        return len(os.sched_getaffinity(0)) or 1
    except (AttributeError, OSError):
        return os.cpu_count() or 1


JOB_MEM_GB = 1.5  # what one C++ compile job of a native build takes at its peak, about


def _read(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def cgroup_cpus(root: str = "/sys/fs/cgroup") -> float | None:
    """The CPUs a cgroup quota allows (cgroup v2 cpu.max "QUOTA PERIOD", v1 cpu.cfs_quota_us / cpu.cfs_period_us), or
    None when there is no quota (a container run with --cpus has one; the host's CPU count says nothing about it).
    """
    v2 = _read(f"{root}/cpu.max").split()
    if len(v2) == 2 and v2[0] != "max":
        try:
            return int(v2[0]) / int(v2[1])
        except (ValueError, ZeroDivisionError):
            return None
    quota, period = _read(f"{root}/cpu/cpu.cfs_quota_us"), _read(
        f"{root}/cpu/cpu.cfs_period_us"
    )
    try:
        if int(quota) > 0 and int(period) > 0:
            return int(quota) / int(period)
    except ValueError:
        pass
    return None


def memory_available_gb(
    root: str = "/sys/fs/cgroup", meminfo: str = "/proc/meminfo"
) -> float | None:
    """The memory a new job may use: the cgroup's limit less what it uses now (v2 memory.max and memory.current, v1
    memory.limit_in_bytes and memory.usage_in_bytes), else the machine's MemAvailable; None when neither reads.
    """
    for limit, used in (
        (f"{root}/memory.max", f"{root}/memory.current"),
        (
            f"{root}/memory/memory.limit_in_bytes",
            f"{root}/memory/memory.usage_in_bytes",
        ),
    ):
        lim, cur = _read(limit), _read(used)
        if lim.isdigit() and int(lim) < 2**60:
            return max(int(lim) - (int(cur) if cur.isdigit() else 0), 0) / 2**30
    m = re.search(r"^MemAvailable:\s+(\d+)\s+kB", _read(meminfo), re.M)
    return int(m.group(1)) / 2**20 if m else None


def build_jobs(
    root: str = "/sys/fs/cgroup",
    meminfo: str = "/proc/meminfo",
    cpus: int | None = None,
    mem_per_job_gb: float = JOB_MEM_GB,
) -> int:
    """The parallel jobs a native build and its ctest may run here: the CPUs this process may run on (affinity), within
    the cgroup's CPU quota, and no more than the memory available holds at `mem_per_job_gb` each; at least 1.
    `$RKMON_JOBS` (a positive number), when the job's environment sets it, is the answer instead.
    """
    env = os.environ.get("RKMON_JOBS", "")
    if env.isdigit() and int(env) > 0:
        return int(env)
    n = cpus or cpu_count()
    quota = cgroup_cpus(root)
    if quota:
        n = min(n, max(1, int(quota)))
    mem = memory_available_gb(root, meminfo)
    if mem is not None and mem_per_job_gb > 0:
        n = min(n, int(mem // mem_per_job_gb))
    return max(1, n)


if (
    __name__ == "__main__"
):  # `python3 steps.py jobs`: the build's job count, for a shell line
    if sys.argv[1:] == ["jobs"]:
        print(build_jobs())
        sys.exit(0)
    print("usage: steps.py jobs", file=sys.stderr)
    sys.exit(2)
