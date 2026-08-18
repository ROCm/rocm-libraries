# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

"""Common utility functions for the GEKO framework."""

import subprocess
import shutil
import sys
import logging
import hashlib

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



def run_silent_command(cmd: List[str], cwd: str | Path = None) -> None:
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
    )
    _, err = proc.communicate()
    logger.debug(f"Silent command completed: returncode={proc.returncode}")
    if proc.returncode != 0:
        if err:
            logger.debug(f"Silent command stderr (truncated): {err[:500]}")
        raise ValueError(err)


def build_tensilelite_client(hipblaslt_path: str | Path, build_dir: str | Path = None) -> Path | None:
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
                ref = head.readline().split(" ")[-1].strip()

            with (git_dir / ref).open("r") as git_hash:
                return git_hash.readline().strip()

        except FileNotFoundError:
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
        run_silent_command(["invoke", "build-client", "--build-dir", build_dir], cwd=tensilelite_path)

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
