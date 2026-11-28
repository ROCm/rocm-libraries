# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The results a job left in its slot, read as data: what the monitor keeps of them is copied into its own folders.

A slot's folder (rkmon_lib/container_layout.py SLOTS_DIR/<runner>) and everything in it are the runner's: what a job
left there may be a link to a file of the monitor's, a FIFO that blocks its reader, a hard link, or a file of any size.
The reader trusts none of it:
- The walk starts at the slot's folder, whose own name the runner cannot change (its parent is root's), or a folder in
  it, and opens each folder below it under the one before, with O_NOFOLLOW: a link on the way, or at the start's own
  name, is refused, never followed.
- A file is opened with O_NOFOLLOW and O_NONBLOCK, then checked on what was opened (fstat): a regular file, with one
  link, owned by the slot's owner. Anything else is refused and named, never read.
- A file is read up to FILE_MAX bytes. A copy stops at TOTAL_MAX bytes and at ENTRIES_MAX names (a folder's
  included), and opens no folder more than DEPTH_MAX folders below its start: what is past any of them is refused.
- The copies are new files in the caller's folder, which no runner writes, with explicit permissions: files 0660
  (0770 when the slot's copy is executable), folders 2770, as everything the monitor shares. A
  name the folder holds already is refused, never replaced: a job's results never take the place of the monitor's.
It runs after the launcher stopped every process the job's command left (docker/rk_prcode.py), and a folder of a user
that still has a process running is refused, not read (`running`), so nothing changes the files while they are read.
On a host, with no slots, the same reader reads the monitor's own build folder.

- `read(root, rel)`: the bytes of the file `rel` under `root` (Refused when it is not a file the rules allow).
- `read_json(root, rel)`: that file parsed as JSON, the one reader of the JSON a job wrote (Refused when it holds no
  JSON, one nested deeper than the parser goes included).
- `copy(root, rel, dest)`: the file or the tree `rel` under `root` copied into the folder `dest` (made when missing),
  what the rules allow; the lines that name what was refused.
- `python3 -I -B pr_results.py copy ROOT REL DEST` (a job script's `prr`) and `python3 -I -B pr_results.py read ROOT
  REL` (the file on stdout). Exit 0 when nothing was refused, 1 when something was, 2 when nothing could be read.

Standard library only, with worker/perms.py beside it (the shared modes): the worker tools import it too.
"""

import itertools
import json
import os
import stat
import sys

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import perms  # noqa: E402  (the modes of what the monitor shares)

FILE_MAX = 64 << 20  # one file's most
TOTAL_MAX = 512 << 20  # one copy's most
ENTRIES_MAX = 20000  # one copy's most names, a folder's included
# The most folders below its start a walk of a slot's files goes down: the copy refuses a folder deeper, and the
# launcher (docker/rk_prcode.py) moves one deeper up before it empties it.
DEPTH_MAX = 64
_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC


class Refused(OSError):
    """A path the rules refuse: a link, not a regular file, more than one link, not the slot owner's, too large."""


def _parts(rel: str) -> list[str]:
    parts = [p for p in str(rel).split("/") if p not in ("", ".")]
    if ".." in parts:
        raise Refused(f"{rel}: a path with '..'")
    return parts


def processes(proc: str = "/proc") -> list[tuple[int, tuple[int, ...], int, str]]:
    """Each process in `proc`, from its /proc/PID/status: (its id, its real and effective user ids, its parent's id, its
    state's letter: Z when it has ended). One that is gone, or that cannot be read, is passed over.
    """
    out = []
    try:
        names = os.listdir(proc)
    except OSError:
        return out
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(f"{proc}/{name}/status", encoding="utf-8", errors="replace") as f:
                st = dict(ln.split(":", 1) for ln in f.read().splitlines() if ":" in ln)
            ids = tuple(int(x) for x in st["Uid"].split()[:2])
            parent = int(st.get("PPid") or 0)
        except (OSError, KeyError, ValueError):
            continue
        out.append((int(name), ids, parent, st.get("State", "").strip()[:1]))
    return out


def running(uid: int, proc: str = "/proc") -> list[int]:
    """The processes of user `uid` (its real or effective id) that still run: a zombie has ended."""
    return sorted(
        pid
        for pid, ids, _parent, state in processes(proc)
        if uid in ids and state != "Z"
    )


def open_root(root: str, proc: str = "/proc") -> tuple[int, int]:
    """The root folder and its owner's id. Its last name is opened with O_NOFOLLOW (a link there is refused); the
    folders above it are the caller's to trust (a slot's folder sits in root's SLOTS_DIR). Another user's folder is
    read only while no process of that user runs, else Refused.
    """
    try:
        fd = os.open(root, _DIR)
    except OSError as exc:
        raise Refused(f"{root}: a link or no folder ({exc.strerror})") from None
    owner = os.fstat(fd).st_uid
    left = running(owner, proc) if owner != os.geteuid() else []
    if left:
        os.close(fd)
        raise Refused(f"{root}: {len(left)} process(es) of its user still run")
    return fd, owner


def folder(path: str) -> None:
    """`path` and the folders above it that are missing, each made a shared folder (perms.SHARED_DIR_MODE)."""
    if os.path.isdir(path):
        return
    folder(os.path.dirname(os.path.abspath(path)))
    try:
        os.mkdir(path)
    except FileExistsError:
        return
    os.chmod(path, perms.SHARED_DIR_MODE)


def _walk_to(fd: int, parts: list[str], rel: str) -> int:
    """The folder `parts` under the folder `fd`, each opened with O_NOFOLLOW: a new descriptor."""
    cur = os.dup(fd)
    try:
        for p in parts:
            try:
                nxt = os.open(p, _DIR, dir_fd=cur)
            except OSError as exc:
                raise Refused(
                    f"{rel}: {p} is a link or no folder ({exc.strerror})"
                ) from None
            os.close(cur)
            cur = nxt
        return cur
    except BaseException:
        os.close(cur)
        raise


def _open_file(dfd: int, name: str, owner: int, rel: str) -> tuple[int, os.stat_result]:
    try:
        fd = os.open(name, _FILE, dir_fd=dfd)
    except OSError as exc:
        raise Refused(f"{rel}: a link, or not readable ({exc.strerror})") from None
    s = os.fstat(fd)
    # A file's size is _read_fd's to check, on what it reads.
    why = (
        "not a regular file"
        if not stat.S_ISREG(s.st_mode)
        else (
            f"{s.st_nlink} links"
            if s.st_nlink != 1
            else ("not the slot owner's" if s.st_uid != owner else "")
        )
    )
    if why:
        os.close(fd)
        raise Refused(f"{rel}: {why}")
    return fd, s


def _read_fd(fd: int, cap: int, rel: str) -> bytes:
    chunks, left = [], cap + 1
    while left > 0:
        b = os.read(fd, min(left, 1 << 20))
        if not b:
            break
        chunks.append(b)
        left -= len(b)
    data = b"".join(chunks)
    if len(data) > cap:
        raise Refused(f"{rel}: over {cap} bytes")
    return data


def read(root: str, rel: str, cap: int = FILE_MAX) -> bytes:
    """The file `rel` under `root`, by the rules (Refused otherwise)."""
    parts = _parts(rel)
    if not parts:
        raise Refused(f"{rel}: no file named")
    rfd, owner = open_root(root)
    try:
        dfd = _walk_to(rfd, parts[:-1], rel)
    finally:
        os.close(rfd)
    try:
        fd, _s = _open_file(dfd, parts[-1], owner, rel)
    finally:
        os.close(dfd)
    try:
        return _read_fd(fd, min(cap, FILE_MAX), rel)
    finally:
        os.close(fd)


def read_json(root: str, rel: str, cap: int = FILE_MAX):
    """The file `rel` under `root`, by the rules, parsed as JSON: Refused when it holds no JSON, one nested deeper than
    the parser goes (a RecursionError) included."""
    data = read(root, rel, cap)
    try:
        return json.loads(data)
    except (ValueError, RecursionError) as exc:
        raise Refused(f"{rel}: no JSON ({str(exc)[:160]})") from None


class _Copy:
    def __init__(self, owner: int, total: int, entries: int):
        self.owner, self.total, self.left, self.refused = owner, total, total, []
        # the cap on names, and the names it leaves (-1: past it, and named)
        self.entries, self.names = entries, entries

    def file(self, dfd: int, name: str, rel: str, dest: str) -> None:
        try:
            fd, s = _open_file(dfd, name, self.owner, rel)
        except Refused as exc:
            self.refused.append(str(exc))
            return
        try:
            if s.st_size > self.left:
                self.refused.append(f"{rel}: past the copy's {self.total} bytes")
                return
            data = _read_fd(fd, min(FILE_MAX, self.left), rel)
        except Refused as exc:
            self.refused.append(str(exc))
            return
        finally:
            os.close(fd)
        self.left -= len(data)
        # a shared file (perms.SHARED_FILE_MODE), which its group runs too when its owner may
        mode = perms.SHARED_FILE_MODE | (0o110 if s.st_mode & stat.S_IXUSR else 0)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        try:
            out = os.open(dest, flags, mode)
        except OSError as exc:
            self.refused.append(f"{rel}: {dest}: {exc.strerror}")
            return
        try:
            os.fchmod(out, mode)
            view = memoryview(data)
            while view:
                view = view[os.write(out, view) :]
        finally:
            os.close(out)

    def tree(self, dfd: int, rel: str, dest: str, depth: int = 0) -> None:
        """The folder `dfd` (`rel`, `depth` folders below the copy's start) into the folder `dest`."""
        folder(dest)
        with os.scandir(dfd) as it:
            # at most one name past the cap is listed, however many the folder holds
            names = sorted(e.name for e in itertools.islice(it, max(self.names, 0) + 1))
        for name in names:
            sub = f"{rel}/{name}" if rel else name
            if self.names <= 0:
                if self.names == 0:
                    self.refused.append(
                        f"{sub}: past the copy's {self.entries} entries: it and all that follows are not copied"
                    )
                    self.names = -1
                return
            self.names -= 1
            try:
                s = os.stat(name, dir_fd=dfd, follow_symlinks=False)
            except OSError as exc:
                self.refused.append(f"{sub}: {exc.strerror}")
                continue
            if stat.S_ISDIR(s.st_mode):
                if depth >= DEPTH_MAX:
                    self.refused.append(
                        f"{sub}: more than {DEPTH_MAX} folders down: it and everything in it are not copied"
                    )
                    continue
                try:
                    cfd = os.open(name, _DIR, dir_fd=dfd)
                except OSError as exc:
                    self.refused.append(f"{sub}: {exc.strerror}")
                    continue
                try:
                    self.tree(cfd, sub, os.path.join(dest, name), depth + 1)
                finally:
                    os.close(cfd)
            elif stat.S_ISREG(s.st_mode):
                self.file(dfd, name, sub, os.path.join(dest, name))
            else:
                self.refused.append(
                    f"{sub}: {'a link' if stat.S_ISLNK(s.st_mode) else 'not a regular file'}"
                )


def copy(
    root: str,
    rel: str,
    dest: str,
    total: int = TOTAL_MAX,
    entries: int = ENTRIES_MAX,
) -> list[str]:
    """The file or tree `rel` under `root` into the folder `dest`: a file keeps its name there, a tree's content goes
    into it, up to `total` bytes and `entries` names. The lines that name what was refused (none: all of it was copied).
    Refused when `rel` is no folder or file the rules allow."""
    parts = _parts(rel)
    rfd, owner = open_root(root)
    try:
        dfd = _walk_to(rfd, parts[:-1], rel)
    finally:
        os.close(rfd)
    c = _Copy(owner, total, entries)
    try:
        if not parts:
            c.tree(dfd, "", dest)
            return c.refused
        try:
            s = os.stat(parts[-1], dir_fd=dfd, follow_symlinks=False)
        except OSError as exc:
            raise Refused(f"{rel}: {exc.strerror}") from None
        if stat.S_ISDIR(s.st_mode):
            try:
                cfd = os.open(parts[-1], _DIR, dir_fd=dfd)
            except OSError as exc:
                raise Refused(f"{rel}: {exc.strerror}") from None
            try:
                c.tree(cfd, rel, dest)
            finally:
                os.close(cfd)
        else:
            folder(dest)
            c.file(dfd, parts[-1], rel, os.path.join(dest, parts[-1]))
    finally:
        os.close(dfd)
    return c.refused


def main(argv: list[str], out=None) -> int:
    o = out or sys.stdout
    if len(argv) == 4 and argv[0] == "copy":
        try:
            refused = copy(argv[1], argv[2], argv[3])
        except (Refused, OSError) as exc:
            print(f"pr_results: {exc}", file=sys.stderr)
            return 2
        for line in refused:
            print(f"pr_results: refused {line}", file=sys.stderr)
        return 1 if refused else 0
    if len(argv) == 3 and argv[0] == "read":
        try:
            data = read(argv[1], argv[2])
        except (Refused, OSError) as exc:
            print(f"pr_results: {exc}", file=sys.stderr)
            return 2
        (getattr(o, "buffer", None) or o).write(data)
        return 0
    print("usage: pr_results.py copy ROOT REL DEST | read ROOT REL", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
