# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Locate COMGR owned by the installed SDK distribution without importing it.

Distribution RECORD entries identify both ordinary and nonce-suffixed payloads.
Native discovery mirrors these ownership and filename rules in compiler_version.cpp.
"""

from __future__ import annotations

import os
import re
from importlib import metadata
from pathlib import PurePosixPath


def _distribution():
    try:
        return metadata.distribution("rocm-sdk-core")
    except metadata.PackageNotFoundError:
        return None


def _payloads(distribution, windows: bool) -> list[str]:
    directory = "bin" if windows else "lib"
    filename = r"amd_comgr\d*\.dll" if windows else r"libamd_comgr\.so(?:\.\d+)*"
    paths = []
    for entry in distribution.files or ():
        parts = PurePosixPath(str(entry)).parts
        if (
            len(parts) != 3
            or not re.fullmatch(r"_rocm_sdk_core(?:_.+)?", parts[0])
            or parts[1] != directory
            or not re.fullmatch(filename, parts[2])
        ):
            continue
        path = os.path.abspath(distribution.locate_file(entry))
        if os.path.isfile(path):
            paths.append(path)
    return sorted(set(paths), key=lambda path: (len(os.path.basename(path)), path))


def candidates(*, windows: bool = False) -> list[str]:
    """The selected SDK-owned file, or no candidate when its payload is absent."""
    distribution = _distribution()
    if distribution is None:
        return []
    return _payloads(distribution, windows)[:1]


def release_for_path(path: str) -> tuple[int, int] | None:
    """Informational SDK release only when RECORD owns this specific library."""
    distribution = _distribution()
    if distribution is None:
        return None
    for candidate in _payloads(distribution, False) + _payloads(distribution, True):
        try:
            owned = os.path.samefile(path, candidate)
        except OSError:
            owned = False
        if owned:
            version = re.match(r"(\d+)\.(\d+)", distribution.version)
            return tuple(map(int, version.groups())) if version else None
    return None
