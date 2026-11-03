# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Locate COMGR owned by the installed SDK distribution without importing it.

Distribution RECORD entries identify both ordinary and nonce-suffixed payloads.
The compiler/runtime owner consumes these paths; native IR lowering receives its flavor.
"""

from __future__ import annotations

import os
import re
from importlib import metadata
from pathlib import Path, PurePosixPath


def _distribution():
    try:
        return metadata.distribution("rocm-sdk-core")
    except metadata.PackageNotFoundError:
        return None


def _payloads(distribution, windows: bool, *, hip: bool = False) -> list[str]:
    directory = "bin" if windows else "lib"
    if hip:
        filename = (
            r"amdhip64(?:_?\d+)?\.dll" if windows else r"libamdhip64\.so(?:\.\d+)*"
        )
    else:
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
        # Metadata sites may contain symlink-sensitive '..'; anchor, do not normalize.
        path = str(Path(distribution.locate_file(entry)).absolute())
        if os.path.isfile(path):
            paths.append(path)
    return sorted(set(paths), key=lambda path: (len(os.path.basename(path)), path))


def candidates(*, windows: bool = False) -> list[str]:
    """The selected SDK-owned file, or no candidate when its payload is absent."""
    distribution = _distribution()
    if distribution is None:
        return []
    return _payloads(distribution, windows)[:1]


def hip_candidates(*, windows: bool = False) -> list[str]:
    """HIP owned by the same SDK distribution, without initializing that SDK."""
    distribution = _distribution()
    return (
        _payloads(distribution, windows, hip=True)[:1]
        if distribution is not None
        else []
    )


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
