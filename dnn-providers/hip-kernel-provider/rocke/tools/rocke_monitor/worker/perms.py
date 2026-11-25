# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The modes of the container's folders: one home for the job scripts, the worker tools, rk
and the start's root step (docker/rk_layout.py), which gives the working folder these modes.

- UMASK: every process of the monitor and of the writer runs with umask 027: what they make is readable by their group
  at most, and writable by no one else.
- What the monitor and the writer share (the state, the logs, the outputs: their group, rkmon): files 0660, folders
  2770, whatever the umask. A shared folder's default ACL (SHARED: its owner and its group read and write, no one else)
  gives each file and folder made in it those modes, and its setgid bit their group. `shared_lock(path)`: the lock
  file PATH, made at SHARED_FILE_MODE when it is missing, held (flock) while the block runs: the monitor and the writer
  share it (the internal state's lock, rkmon_lib/state_store.py; the GitHub request cap's, rkmon_lib/gh_cap.py).
- The trees PR code reads (the clone, the worktrees in the working folder, the virtual environments, the probes, the
  code): the PR runners' group, folders 2750, files 0640: the umask under a setgid folder of that group with no default
  ACL, so PR code reads them and writes nothing there. `tree_folder(path, group)` makes the empty folder such a tree
  goes into (git adds a worktree, or clones, into an empty folder).
- The scratch folder (paths.tmp) is shared too, and its default ACL (SCRATCH) also lets the runners' group enter each
  folder made in it, and list, read and write none: a runner reaches only the trees made there, each with
  `tree_folder` (the job scripts' `prtree`, and `rk pr-tree` for the worktrees the session's agents add there).
  `is_tree(st, gid)` tells such a tree's folder. A file made there with an x bit gets the group's x: a runner may run
  a program the monitor built there, and never reads it.
- `default_acl(path)`, `set_default_acl(path, entries)`, `drop_acls(path)`: a folder's default ACL by its entries,
  (tag, id, perms): tag one of TAGS, id a user's or group's id for "user" and "group" (else None), perms "rwx"-style.
  `path` may be an open folder's descriptor.
- `python3 -I perms.py tree --group GROUP DIR...`: `tree_folder` for each DIR (the job scripts' `prtree`); exit 1 when
  one could not be made so (a line says why).

On a host there is no runners' group: `tree_folder` makes a plain folder. Standard library only: the worker tools and the
image's root step load it too.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import grp
import os
import stat
import struct
import sys

UMASK = 0o027
SHARED_DIR_MODE = 0o2770
SHARED_FILE_MODE = 0o660
TREE_DIR_MODE = 0o2750
TREE_FILE_MODE = 0o640
DEFAULT = "system.posix_acl_default"
ACCESS = "system.posix_acl_access"
_VERSION = 2
TAGS = {
    "user_obj": 0x01,
    "user": 0x02,
    "group_obj": 0x04,
    "group": 0x08,
    "mask": 0x10,
    "other": 0x20,
}
_NAMES = {v: k for k, v in TAGS.items()}
_NO_ID = 0xFFFFFFFF
SHARED = (("user_obj", None, "rwx"), ("group_obj", None, "rwx"), ("other", None, ""))


def scratch(pr_gid: int) -> tuple:
    """SCRATCH: SHARED, and the runners' group (`pr_gid`) enters the folders made in the folder, and lists and reads
    none of them."""
    return SHARED[:2] + (("group", pr_gid, "x"), ("mask", None, "rwx"), SHARED[2])


@contextlib.contextmanager
def shared_lock(path):
    """The lock file PATH, made at SHARED_FILE_MODE when it is missing, held while the block runs: see the module's
    docstring."""
    fd = os.open(
        path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, SHARED_FILE_MODE
    )
    try:
        if os.fstat(fd).st_uid == os.getuid():
            os.fchmod(fd, SHARED_FILE_MODE)  # the umask may have taken a bit away
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)  # which lets the lock go


def is_tree(st: os.stat_result, gid: int) -> bool:
    """Whether `st` is the folder of a tree PR code reads, as `tree_folder` makes it: group `gid` (the runners'), mode
    2750."""
    return (
        stat.S_ISDIR(st.st_mode)
        and st.st_gid == gid
        and stat.S_IMODE(st.st_mode) == TREE_DIR_MODE
    )


def _perm(text: str) -> int:
    return (
        (4 if "r" in text else 0)
        | (2 if "w" in text else 0)
        | (1 if "x" in text else 0)
    )


def _text(perm: int) -> str:
    return "".join(c for c, bit in (("r", 4), ("w", 2), ("x", 1)) if perm & bit)


def acl_bytes(entries) -> bytes:
    """The extended attribute's value for an ACL of `entries`, in the kernel's order (by tag, then id)."""
    packed = sorted(
        (
            TAGS[tag],
            _NO_ID if ident is None else int(ident),
            _perm(perms),
        )
        for tag, ident, perms in entries
    )
    return struct.pack("<I", _VERSION) + b"".join(
        struct.pack("<HHI", tag, perm, ident) for tag, ident, perm in packed
    )


def acl_entries(data: bytes) -> tuple:
    """The entries of an ACL's extended attribute value (acl_bytes' inverse)."""
    if len(data) < 4 or struct.unpack("<I", data[:4])[0] != _VERSION:
        raise ValueError("not a POSIX ACL")
    out = []
    for i in range(4, len(data) - 7, 8):
        tag, perm, ident = struct.unpack("<HHI", data[i : i + 8])
        out.append(
            (_NAMES.get(tag, str(tag)), None if ident == _NO_ID else ident, _text(perm))
        )
    return tuple(out)


def default_acl(path) -> tuple | None:
    """The folder's default ACL, as entries in the kernel's order; None when it has none."""
    try:
        return acl_entries(os.getxattr(path, DEFAULT))
    except OSError as exc:
        if exc.errno == errno.ENODATA:
            return None
        raise


def set_default_acl(path, entries) -> None:
    """Give the folder the default ACL `entries` (OSError when the file system keeps no ACL)."""
    os.setxattr(path, DEFAULT, acl_bytes(entries))


def same_acl(have, entries) -> bool:
    return have is not None and tuple(have) == acl_entries(acl_bytes(entries))


def drop_acls(path) -> None:
    """Remove the folder's default ACL and its access ACL beyond its mode, so its mode alone says who may do what
    (`path`: a path, or an open folder's descriptor). The access ACL only when it has one: for a user in no group of
    the folder's, the kernel takes the folder's set-group-ID bit with any access ACL removed, even one that is not
    there."""
    for name in (DEFAULT, ACCESS):
        try:
            if name == ACCESS:
                os.getxattr(path, name)  # ENODATA: none to remove
            os.removexattr(path, name)
        except OSError as exc:
            if exc.errno not in (errno.ENODATA, errno.EOPNOTSUPP):
                raise


def tree_folder(path: str, group: str, getgrnam=grp.getgrnam) -> str:
    """Make `path` (missing, or an empty folder) the folder a tree PR code reads goes into: group `group` (the runners'),
    mode 2750, no ACL, never through a link. Where there is no such group (a host), a plain folder. "" when done, else
    why not."""
    try:
        gid = getgrnam(group).gr_gid
    except KeyError:
        gid = None
    try:
        os.mkdir(path, TREE_DIR_MODE & 0o777)
    except FileExistsError:
        pass
    except OSError as exc:
        return f"{path}: not made ({exc.strerror})"
    if gid is None:
        return ""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as exc:
        return f"{path}: a link, or no folder ({exc.strerror})"
    try:
        if os.listdir(fd):
            s = os.fstat(fd)
            if is_tree(s, gid):
                return ""  # a tree that is there already, as it should be
            return f"{path}: not empty, and not a tree PR code reads (group {s.st_gid}, mode {stat.S_IMODE(s.st_mode):o})"
        drop_acls(fd)
        os.fchown(fd, -1, gid)
        os.fchmod(fd, TREE_DIR_MODE)
    except OSError as exc:
        return f"{path}: not made a tree PR code reads ({exc.strerror})"
    finally:
        os.close(fd)
    return ""


def main(argv: list[str]) -> int:
    if len(argv) >= 4 and argv[0] == "tree" and argv[1] == "--group":
        bad = [why for d in argv[3:] if (why := tree_folder(d, argv[2]))]
        for why in bad:
            print(f"perms: {why}", file=sys.stderr)
        return 1 if bad else 0
    print("usage: perms.py tree --group GROUP DIR...", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
