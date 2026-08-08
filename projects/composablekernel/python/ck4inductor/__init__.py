# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from .util import check_headers, include_roots


__all__ = ["check_headers", "include_roots", "__version__"]


# Needs to be manually updated at each release; it cannot be derived here. Release
# tags (therock-*) exist only in the monorepo, but the wheel is built from the
# filtered mirror, whose reachable tags stop at therock-7.10.
BASE_VERSION = "7.14"

_HASH_WIDTH = 7


def _build_hash():
    """Short commit hash of this source tree, or "" if it cannot be determined.

    Resolved against the directory holding this file rather than the working
    directory, so an installed copy cannot report the hash of whatever repository
    the caller happens to be standing in.
    """
    import os
    import subprocess

    try:
        return subprocess.run(
            [
                "git",
                "-C",
                os.path.dirname(os.path.abspath(__file__)),
                "rev-parse",
                f"--short={_HASH_WIDTH}",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        # No git, no .git (source tarball), or the call failed - all legitimate.
        return ""


def _build_version():
    build_hash = _build_hash()
    # "+unknown" keeps the label honest when the hash is unavailable, rather than
    # emitting a plausible-looking one that would hide the failure.
    return f"{BASE_VERSION}+g{build_hash}" if build_hash else f"{BASE_VERSION}+unknown"


# A string, not a callable: setuptools accepts either, but consumers reading
# ck4inductor.__version__ at runtime need the value.
__version__ = _build_version()
