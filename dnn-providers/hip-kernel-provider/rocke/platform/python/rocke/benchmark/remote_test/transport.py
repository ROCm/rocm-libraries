# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Passwordless SSH + rsync wrappers.

All calls pin the configured private key and BatchMode=yes so a missing
key surfaces as an immediate error (never a password prompt). A persistent
ssh ControlMaster is reused across calls to amortize handshake cost.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from . import config

_CTRL_DIR = Path(tempfile.gettempdir()) / f"rocke_ssh_{os.getuid()}"
_CTRL_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
_CTRL_PATH = _CTRL_DIR / "cm-%r@%h:%p"


def _ssh_base() -> List[str]:
    base: List[str] = ["ssh"]
    if config.REMOTE.ssh_key:
        base += ["-i", config.REMOTE.ssh_key]
    base += [
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        f"ControlPath={_CTRL_PATH}",
        "-o",
        "ControlMaster=auto",
        "-o",
        "ControlPersist=300",
    ]
    return base


_SSH_BASE: List[str] = _ssh_base()


def _target() -> str:
    if not config.REMOTE.host:
        raise RuntimeError(
            "Remote host not configured. Set ROCKE_REMOTE_HOST to either an "
            "SSH alias from ~/.ssh/config (recommended) or a real hostname. "
            "If passing a real hostname, also set ROCKE_REMOTE_USER."
        )
    # Alias mode: user/key/etc come from ~/.ssh/config.
    if not config.REMOTE.user:
        return config.REMOTE.host
    return f"{config.REMOTE.user}@{config.REMOTE.host}"


def ssh_run(
    remote_cmd: str | Sequence[str],
    *,
    check: bool = True,
    capture: bool = False,
    stream: bool = False,
    timeout: Optional[int] = None,
) -> subprocess.CompletedProcess:
    """Run a command on the remote host. `remote_cmd` is a shell string or argv."""
    if isinstance(remote_cmd, (list, tuple)):
        remote_cmd = " ".join(shlex.quote(str(a)) for a in remote_cmd)
    argv = [*_SSH_BASE, _target(), remote_cmd]
    if stream:
        # Inherit stdout/stderr for live output.
        return subprocess.run(argv, check=check, timeout=timeout)
    return subprocess.run(
        argv,
        check=check,
        timeout=timeout,
        capture_output=capture,
        text=True,
    )


def _rsync_ssh_opt() -> str:
    # rsync -e wants a single string; reuse the same controlmaster.
    return " ".join(shlex.quote(a) for a in _SSH_BASE)


def rsync_push(
    local: Path,
    remote_path: str,
    *,
    extra: Iterable[str] = (),
    delete: bool = False,
) -> None:
    """Push local file/dir to <user>@<host>:<remote_path>. Creates parent dirs."""
    parent = remote_path.rsplit("/", 1)[0] if "/" in remote_path else "."
    if parent:
        ssh_run(["mkdir", "-p", parent])
    src = str(local)
    if local.is_dir() and not src.endswith("/"):
        src += "/"
    argv: List[str] = ["rsync", "-az", "--info=stats1", "-e", _rsync_ssh_opt()]
    if delete:
        argv.append("--delete")
    argv.extend(extra)
    argv.extend([src, f"{_target()}:{remote_path}"])
    subprocess.run(argv, check=True)


def rsync_pull(remote_path: str, local: Path, *, extra: Iterable[str] = ()) -> None:
    local.parent.mkdir(parents=True, exist_ok=True)
    argv: List[str] = [
        "rsync",
        "-az",
        "-e",
        _rsync_ssh_opt(),
        *extra,
        f"{_target()}:{remote_path}",
        str(local),
    ]
    subprocess.run(argv, check=True)


def check_connectivity() -> str:
    """Return remote `hostname` or raise."""
    r = ssh_run("hostname", capture=True)
    return r.stdout.strip()


_REMOTE_HOME: Optional[str] = None


def remote_home() -> str:
    """Return the remote `$HOME`, cached. We resolve `~` ourselves because
    `shlex.quote('~/foo')` produces `'~/foo'` which the remote shell will
    not tilde-expand inside single quotes."""
    global _REMOTE_HOME
    if _REMOTE_HOME is None:
        r = ssh_run('printf %s "$HOME"', capture=True)
        _REMOTE_HOME = r.stdout.strip() or ""
        if not _REMOTE_HOME:
            raise RuntimeError("remote $HOME resolved to empty string")
    return _REMOTE_HOME


def expand_remote(path: str) -> str:
    """Expand a leading `~` against the remote `$HOME`. Non-tilde paths
    pass through unchanged."""
    if path == "~":
        return remote_home()
    if path.startswith("~/"):
        return f"{remote_home()}{path[1:]}"
    return path
