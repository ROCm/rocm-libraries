# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A file written whole: whoever reads it reads the old file or the new one, never half of one, and two writers never
mix their texts.

- `write(path, data, mode, durable=False, owner=None)`: DATA (text, written in UTF-8, or bytes) as the file PATH: a
  new file beside it (`beside`), made new and never through a link (FLAGS), filled (`fill`), then renamed onto PATH.
  DURABLE: the new file is synced to its disk before the rename and the folder after it (`sync_folder`), so a crash
  leaves the old file or the new one. OWNER, (uid, gid): the new file's owner and group, given by its descriptor
  before it is filled, so that root writing in a folder another user can write follows no path that user planted.
  The file beside PATH goes either way, and on a failure PATH is as it was.
- `beside(path)`: the new file a write of PATH makes beside it, named NEW_FILE, unique to that write.
- `fill(fd, data, mode, durable)`: the bytes DATA as the content of the new file open at FD, at MODE whatever the
  umask, synced to its disk when DURABLE; FD is closed after.
- `sync_folder(folder)`: the entries of the folder FOLDER synced to its disk; nothing when the system cannot.

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import contextlib
import os
import pathlib

# The new file a write puts beside the file NAME, then in its place (UNIQUE: the writer's process id and a random part,
# so no two writers share one).
NEW_FILE = ".{name}.{unique}.new"

# A file made new: never one that is there, nor through a link.
FLAGS = (
    os.O_WRONLY
    | os.O_CREAT
    | os.O_EXCL
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


def beside(path) -> pathlib.Path:
    """The new file a write of PATH makes beside it: see the module's docstring."""
    p = pathlib.Path(path)
    unique = f"{os.getpid()}.{os.urandom(4).hex()}"
    return p.with_name(NEW_FILE.format(name=p.name, unique=unique))


def fill(fd: int, data: bytes, mode: int, durable: bool) -> None:
    """DATA as the content of the new file open at FD: see the module's docstring."""
    with os.fdopen(fd, "wb") as f:
        os.fchmod(f.fileno(), mode)  # the umask may have taken a bit away
        f.write(data)
        if durable:
            f.flush()
            os.fsync(f.fileno())


def write(
    path, data, mode: int, durable: bool = False, owner: tuple | None = None
) -> None:
    """DATA as the file PATH, whole: see the module's docstring."""
    p = pathlib.Path(path)
    tmp = beside(p)
    fd = os.open(tmp, FLAGS, mode)
    try:
        if owner is not None:
            try:
                os.fchown(fd, *owner)
            except OSError:
                os.close(fd)
                raise
        fill(fd, data.encode("utf-8") if isinstance(data, str) else data, mode, durable)
        os.replace(tmp, p)
    finally:
        with contextlib.suppress(OSError):  # renamed already
            os.unlink(tmp)
    if durable:
        sync_folder(p.parent)


def sync_folder(folder) -> None:
    """The entries of the folder FOLDER synced to its disk: see the module's docstring."""
    with contextlib.suppress(OSError):
        fd = os.open(folder, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
