# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Atomic file writes and JSON helpers shared by the stages."""
from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from typing import Any, Optional


def _fsync_dir(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write `data` to `path` so that readers see either the previous file or
    the complete new one, never a torn write.

    The data goes to a temp file in the same directory, which is fsynced and
    renamed over `path`; the directory is then fsynced so the rename survives
    a crash too. The file gets the permissions a plain `open()` would give it
    under the current umask."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(8)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _fsync_dir(path.parent)


def atomic_write_text(path: Path, contents: str) -> None:
    """Text (UTF-8) counterpart of `atomic_write_bytes`."""
    atomic_write_bytes(path, contents.encode("utf-8"))


def write_json(path: Path, obj: Any, *, indent: Optional[int] = 2) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent) + "\n")


def read_json(path: Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)
