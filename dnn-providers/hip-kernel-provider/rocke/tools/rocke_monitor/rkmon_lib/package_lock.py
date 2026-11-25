# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The package folder's lock: no package update replaces a package folder while a command of its package
runs there, and no such command starts while an update runs. One lock file, LOCK, in the host's records' folder
(records.FOLDER), taken with flock, which the kernel lets go when the process ends, however it ends: the lock is the
truth. Each command that changes the deployment, or reads the package's files after it starts, holds it shared for its
whole run, and package update holds it exclusive (the command table says which: commands.Row.lock); none waits. Each
hold names its command, and since when, in a holder's file of its own beside LOCK, which it keeps locked while it holds
LOCK: the file of a holder that is gone is one whose lock no one holds, which names nothing and blocks nothing, and the
next hold removes it.

- `hold(package_dir, command, exclusive=False) -> Held`: LOCK of the package folder PACKAGE_DIR held, shared, or
  exclusive with EXCLUSIVE; then the files of the holders that are gone removed, and COMMAND (the command as you typed
  it) named, with the time now, in a holder's file. A folder with no records' folder (no hold makes one: a package
  folder where no settings were ever confirmed, as the package itself, has none), or whose LOCK can be neither opened
  nor made (a read-only folder) or locked (a file system with no flock), has no lock: the Held holds nothing, and the
  command runs as it would without it. A holder's file that cannot be made or written leaves the hold unnamed.
  `Taken` when another holds LOCK so that this hold cannot be: nothing is held.
- `Held`: a hold, until `release()`, as leaving `with hold(...)` does: its holder's file goes, from the records'
  folder wherever that moved (an update moves the folder it replaces, and its records' folder with it), then LOCK.
  `held`: whether it holds LOCK.
- `Taken(update, holders=())`: a hold refused. UPDATE: a shared one, which only an exclusive hold, a package update's,
  refuses; else an exclusive one, refused by the holders HOLDERS (`holders`).
- `holders(package_dir) -> list`: the holders of the folder's LOCK, each a `Holder(command, since)` as its file names
  it (SINCE an aware datetime; None when the file gives no time in the records' form, records.parsed), the earliest
  first: those whose file is locked alone, so that a holder that is gone names nothing, nor one whose file is not yet
  written or names no command.
- LOCK: the lock file, empty, 600. HOLDER: how each holder's file beside it is named, `package.lock.` and letters of its
  own; 600, JSON: `command` and `since` (records.timestamp).

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import contextlib
import dataclasses
import datetime
import fcntl
import json
import os
import pathlib
import secrets
from typing import Optional

from . import private_files, records

LOCK = f"{records.FOLDER}/package.lock"
HOLDER = "package.lock."
TRIES = 5  # how many holder's files a hold makes, when another hold looks at each before it is locked


@dataclasses.dataclass(frozen=True)
class Holder:
    """A holder of a package folder's lock, as its file names it: see the module's docstring."""

    command: str
    since: Optional[datetime.datetime]


class Taken(RuntimeError):
    """A hold of a package folder's lock refused: see the module's docstring."""

    def __init__(self, update: bool, holders=()):
        self.update = update
        self.holders = tuple(holders)
        by = "a package update has" if update else "other commands have"
        super().__init__(f"{by} the package folder's lock")


class Held:
    """A hold of a package folder's lock: see the module's docstring. LOCK, the descriptor of the lock file; OWN, its
    holder's file's, named NAME in the records' folder whose descriptor is AT."""

    def __init__(self, lock=None, own=None, at=None, name=""):
        self._lock, self._own, self._at, self._name = lock, own, at, name

    @property
    def held(self) -> bool:
        return self._lock is not None

    def release(self) -> None:
        if self._own is not None:
            _gone(self._name, self._at)
            os.close(self._own)
            os.close(self._at)
            self._own = self._at = None
        if self._lock is not None:
            os.close(self._lock)
            self._lock = None

    def __enter__(self) -> Held:
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def hold(package_dir, command: str, exclusive: bool = False) -> Held:
    """LOCK of PACKAGE_DIR held, and COMMAND named beside it: see the module's docstring."""
    folder = pathlib.Path(package_dir) / records.FOLDER
    if not folder.is_dir():
        return Held()
    fd = _opened(pathlib.Path(package_dir) / LOCK)
    if fd is None:
        return Held()
    try:
        fcntl.flock(fd, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        raise Taken(not exclusive, holders(package_dir) if exclusive else ()) from None
    except OSError:
        os.close(fd)
        return Held()
    _sweep(folder)
    return Held(fd, *_named(folder, command))


def holders(package_dir) -> list:
    """The holders of PACKAGE_DIR's LOCK, as their files name them: see the module's docstring."""
    found = []
    for path in _files(pathlib.Path(package_dir) / records.FOLDER):
        try:
            fd = os.open(path, os.O_RDONLY | private_files.NOFOLLOW)
        except OSError:
            continue
        try:
            holder = _holder(fd) if _locked(fd) else None
        finally:
            os.close(fd)
        if holder is not None:
            found.append(holder)
    never = datetime.datetime.max.replace(tzinfo=datetime.timezone.utc)
    return sorted(found, key=lambda h: (h.since or never, h.command))


def _opened(path) -> Optional[int]:
    """LOCK's descriptor, the file made (empty, 600) when it is not there; None when it can be neither opened nor
    made."""
    try:
        return os.open(path, os.O_RDONLY | os.O_CREAT | private_files.NOFOLLOW, 0o600)
    except OSError:
        return None


def _named(folder: pathlib.Path, command: str) -> tuple:
    """A holder's file of this hold's own in FOLDER, locked, then naming COMMAND and the time now: (its descriptor,
    FOLDER's, its name); (None, None, "") when none can be made, locked or written."""
    named = {"command": command, "since": records.timestamp()}
    data = json.dumps(named).encode("utf-8")
    try:
        at = os.open(folder, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        return None, None, ""
    for _ in range(TRIES):
        name = HOLDER + secrets.token_hex(8)
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | private_files.NOFOLLOW
        try:
            fd = os.open(name, flags, 0o600, dir_fd=at)
        except FileExistsError:
            continue
        except OSError:
            break
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if _same(fd, name, at):  # no other hold removed it before it was locked
                os.write(fd, data)
                return fd, at, name
        except BlockingIOError:
            pass  # another hold looks at it, as it looks for the files of holders that are gone: a new one
        except OSError:
            _gone(name, at)
            os.close(fd)
            break
        if _same(fd, name, at):
            _gone(name, at)
        os.close(fd)
    os.close(at)
    return None, None, ""


def _sweep(folder: pathlib.Path) -> None:
    """The holders' files in FOLDER whose lock no one holds removed: their holders are gone."""
    for path in _files(folder):
        try:
            fd = os.open(path, os.O_RDONLY | private_files.NOFOLLOW)
        except OSError:
            continue
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if _same(fd, path):
                os.unlink(path)
        except OSError:
            pass  # its holder runs (BlockingIOError), or it cannot go
        finally:
            os.close(fd)


def _files(folder: pathlib.Path) -> list:
    """The holders' files in FOLDER, by name."""
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return []
    return [folder / n for n in names if n.startswith(HOLDER)]


def _locked(fd: int) -> bool:
    """Whether another holds the lock of the file FD has open: a look, which takes it a moment when no one does."""
    try:
        fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
    except BlockingIOError:
        return True
    except OSError:
        return False
    fcntl.flock(fd, fcntl.LOCK_UN)
    return False


def _holder(fd: int) -> Optional[Holder]:
    """The holder the holder's file FD has open names; None when it names no command."""
    try:
        with open(fd, "rb", closefd=False) as f:
            data = json.loads(f.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("command"), str):
        return None
    if not data["command"]:
        return None
    return Holder(data["command"], records.parsed(data.get("since")))


def _same(fd: int, name, at: Optional[int] = None) -> bool:
    """Whether NAME (in the folder whose descriptor is AT; else a path) is still the file FD has open."""
    try:
        there = os.stat(name, dir_fd=at, follow_symlinks=False)
        return os.path.samestat(os.fstat(fd), there)
    except OSError:
        return False


def _gone(name, at: int) -> None:
    """The file NAME in the folder whose descriptor is AT removed, if it is there."""
    with contextlib.suppress(OSError):
        os.unlink(name, dir_fd=at)
