# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""Common utility functions for the GEKO framework."""

import subprocess
import shutil
import sys
import logging
import hashlib
import os

from typing import List
from pathlib import Path
from datetime import date
from importlib.util import find_spec
from datetime import datetime, timezone

logger = logging.getLogger("GEKO")


def get_utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def compute_file_sha256(path: str | Path) -> str:
    """Return the SHA256 hex digest of a file (streamed; handles large files)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def _rocm_subprocess_env() -> dict[str, str]:
    """Return env with ROCm tools on PATH for invoke / Tensile subprocesses."""
    env = os.environ.copy()
    rocm = env.get("ROCM_PATH", "/opt/rocm")
    env["ROCM_PATH"] = rocm
    rocm_bins = f"{rocm}/bin:{rocm}/hip/bin:{rocm}/llvm/bin"
    if rocm_bins not in env.get("PATH", ""):
        env["PATH"] = f"{rocm_bins}:{env.get('PATH', '')}"
    return env


def run_silent_command(cmd: List[str], cwd: str | Path = None, env: dict[str, str] | None = None) -> None:
    """Execute a shell command with silent stdout and error handling.

    Args:
        cmd (List[str]): Shell command to execute as a list of strings.
        cwd (str | Path, optional): Current working directory override.

    Raises:
        ValueError: If command returns non-zero exit code, with stderr as message.
    """
    logger.debug(f"Running silent command: cmd={cmd} cwd={cwd}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        cwd=cwd,
        text=True,
        env=env if env is not None else _rocm_subprocess_env(),
    )
    _, err = proc.communicate()
    logger.debug(f"Silent command completed: returncode={proc.returncode}")
    if proc.returncode != 0:
        if err:
            logger.debug(f"Silent command stderr (truncated): {err[:500]}")
        raise ValueError(err)


def build_tensilelite_client(
    hipblaslt_path: str | Path,
    build_dir: str | Path = None,
    gpu_targets: str | None = None,
) -> Path | None:
    """Builds the tensilelite client if not found or outdated.

    Args:
        hipblaslt_path (str | Path): Path to hipBLASLt installation directory.
        build_dir (str | Path, optional): Target tensilelite client build directory name.
            Defaults to None.

    Returns:
        Path | None: Path to tensilelite client if custom build_dir used, None otherwise.

    Raises:
        FileNotFoundError: If hipblaslt_path does not exist.
    """

    def get_git_revision_hash(path: str | Path) -> str:
        try:
            git_dir = Path(path).resolve() / ".git"

            with (git_dir / "HEAD").open("r") as head:
                line = head.readline().strip()

            # Detached HEAD: the file holds the hash itself, not a ref.
            if not line.startswith("ref:"):
                return line

            ref = line.split(" ")[-1].strip()

            loose = git_dir / ref
            if loose.is_file():
                with loose.open("r") as git_hash:
                    return git_hash.readline().strip()

            # Packed refs: `git gc`/`git clone` moves refs out of .git/refs into
            # .git/packed-refs, where the loose file no longer exists. Reading
            # only loose refs made this fall back to today's date, which is then
            # used as the tensilelite client cache key -- so a same-day hipBLASLt
            # rebuild silently reused a stale client.
            packed = git_dir / "packed-refs"
            if packed.is_file():
                with packed.open("r") as f:
                    for entry in f:
                        entry = entry.strip()
                        if not entry or entry.startswith(("#", "^")):
                            continue
                        sha, _, name = entry.partition(" ")
                        if name.strip() == ref:
                            return sha

            raise FileNotFoundError(f"no loose or packed ref for '{ref}'")

        except OSError:
            logger.warning("Error while retrieving repository information, using current date instead of commit hash")

        return str(date.today())

    hipblaslt_path = Path(hipblaslt_path)

    if not hipblaslt_path.is_dir():
        raise FileNotFoundError(f"hipBLASLt path not found: '{hipblaslt_path}'")

    tensilelite_path = hipblaslt_path / "tensilelite"
    default_build_dir = tensilelite_path / "build_tmp"

    current_hash = get_git_revision_hash(hipblaslt_path.parent.parent)
    logger.debug(
        f"Client build context: hipblaslt_path={hipblaslt_path} tensilelite_path={tensilelite_path} "
        f"default_build_dir={default_build_dir}"
    )

    if build_dir is None:
        build_dir = default_build_dir

    build_dir = Path(build_dir).resolve()
    client_path = build_dir / "tensilelite/client/tensilelite-client"
    hash_file_path = build_dir / "hash.txt"

    build = True
    client_exists = client_path.is_file()
    hash_exists = hash_file_path.is_file()
    hash_matches = hash_exists and open(hash_file_path).read().strip() == current_hash
    logger.debug(
        f"Client cache state: build_dir={build_dir} client_exists={client_exists} "
        f"hash_exists={hash_exists} hash_matches={hash_matches}"
    )
    if client_exists and hash_exists and hash_matches:
        build = False

    if build:
        if not find_spec("invoke"):
            raise RuntimeError(
                "'invoke' package not found. It is required by tensilelite's "
                "build system. Install it via the tensilelite setup "
                "(hipBLASLt/tensilelite/requirements.txt) or: pip install invoke"
            )

        shutil.rmtree(build_dir, ignore_errors=True)

        logger.info(f"Building tensilelite client in '{build_dir}'")
        cmd = ["invoke", "build-client", "--build-dir", str(build_dir)]
        if gpu_targets:
            cmd.extend(["--gpu-targets", gpu_targets])
        run_silent_command(cmd, cwd=tensilelite_path)

        Path(hash_file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(hash_file_path, "w") as f:
            f.write(current_hash)
    else:
        logger.debug(f"Skipping tensilelite client build, using cached client at '{client_path}'")

    return client_path if build_dir != default_build_dir else None


def parse_devices(devices: str | list[int]) -> List[int]:
    """Parse device specification into a list of device IDs.

    Args:
        devices (str | list[int]): Either a comma-separated string
            (e.g., "0,1,2,3") or list of device IDs.

    Returns:
        List[int]: List of unique device IDs as integers.

    Raises:
        ValueError: If devices string cannot be parsed, wrong type provided,
            or no devices specified.
    """
    if isinstance(devices, str):
        try:
            devices = list(set([int(d) for d in devices.split(",")]))
        except ValueError:
            raise ValueError(f"Error parsing devices {devices}")
    elif not isinstance(devices, (list, tuple)):
        raise ValueError(f"Type {type(devices)} not supported")

    if len(devices) == 0:
        raise ValueError(f"Need at least 1 device to run the optimization")

    return devices


def ensure_tensile_importable(hipblaslt_path: "str | Path") -> str:
    """Make Tensile *and* rocisa importable in THIS process, and in children.

    ``geko`` imports Tensile lazily from inside functions, and every worker it
    spawns inherits ``PYTHONPATH``. Both need the same two entries, so resolve
    them once here rather than relying on whatever the invoking shell happens to
    export -- that ambient value is what silently bound a run to a stale hipBLASLt
    clone and produced::

        ImportError: cannot import name 'rocIsa' from 'rocisa' (unknown location)

    The "unknown location" wording is the giveaway: ``<build>/tensilelite/rocisa``
    is a CMake directory with no ``__init__.py``, so Python binds it as a namespace
    package that shadows the real extension module nested one level below it.

    Args:
        hipblaslt_path: Root of the hipBLASLt checkout to bind this run to.

    Returns:
        The resolved ``PYTHONPATH`` string, also exported to ``os.environ`` so
        subprocesses inherit it.
    """
    resolved = tensile_pythonpath(hipblaslt_path, inherit=False)
    for entry in reversed(resolved.split(os.pathsep)):
        if entry and entry not in sys.path:
            sys.path.insert(0, entry)
    existing = os.environ.get("PYTHONPATH", "")
    resolved_parts = set(resolved.split(os.pathsep))
    parts = [p for p in existing.split(os.pathsep) if p and p not in resolved_parts]
    os.environ["PYTHONPATH"] = os.pathsep.join([resolved, *parts])
    return os.environ["PYTHONPATH"]


def tensile_pythonpath(hipblaslt_path, inherit: bool = True) -> str:
    """PYTHONPATH that lets a child process import BOTH Tensile and rocisa.

    Tensile's ``Common.Utilities`` does ``from rocisa import rocIsa`` at import time,
    so a child given only ``<hipblaslt>/tensilelite`` dies with

        ImportError: cannot import name 'rocIsa' from 'rocisa' (unknown location)

    "unknown location" is the giveaway: Python found a *namespace* package -- some
    directory literally named ``rocisa`` with no ``__init__.py`` -- instead of the
    built one. Every hipBLASLt build tree has such a decoy, because the CMake build
    directory is itself called ``rocisa`` and the real package sits one level deeper:

        build/release/tensilelite/rocisa/            <- CMake artifacts, the decoy
        build/release/tensilelite/rocisa/rocisa/     <- __init__.py + _rocisa*.so

    So the entry to put on PYTHONPATH is the PARENT of the package, and it must be
    verified rather than assumed -- an unbuilt tree has the decoy but not the package.

    Args:
        hipblaslt_path: hipBLASLt checkout root.
        inherit: Append the caller's existing PYTHONPATH after the resolved
            entries, so a child still sees a rocisa or geko checkout the caller
            pointed it at.

    Returns:
        A ``os.pathsep``-joined PYTHONPATH string.
    """
    hip = Path(hipblaslt_path)
    parts = [str(hip / "tensilelite")]

    for rel in (
        "build/release/tensilelite/rocisa",
        "build/tensilelite/rocisa",
        "build_tmp/tensilelite/rocisa",
    ):
        candidate = hip / rel
        if (candidate / "rocisa" / "__init__.py").is_file():
            parts.append(str(candidate))
            break

    if inherit and os.environ.get("PYTHONPATH"):
        parts.append(os.environ["PYTHONPATH"])
    return os.pathsep.join(parts)
