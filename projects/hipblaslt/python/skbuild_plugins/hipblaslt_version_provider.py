# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""scikit-build-core metadata provider: the package version is hipBLASLt's version.

It reads hipblaslt-version.h from the hipBLASLt the bindings are built against. The search order
must match CMakeLists.txt, which checks that find_package(hipblaslt) agrees.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping

__all__ = ["dynamic_metadata", "find_version_header", "parse_version_header"]

_DEFINE = re.compile(
    r"^\s*#\s*define\s+HIPBLASLT_VERSION_(MAJOR|MINOR|PATCH)\s+(\d+)\s*$", re.M
)


def build_roots(environ: Mapping[str, str], platform: str) -> list[Path]:
    """Install roots to search, in order: hipblaslt_ROOT, ROCM_PATH, ROCM_HOME, then /opt/rocm."""
    roots = [
        Path(environ[v])
        for v in ("hipblaslt_ROOT", "ROCM_PATH", "ROCM_HOME")
        if environ.get(v)
    ]
    if platform != "win32":
        roots.append(Path("/opt/rocm"))
    return roots


def find_version_header(environ: Mapping[str, str], platform: str) -> Path:
    tried = []
    for root in build_roots(environ, platform):
        header = root / "include" / "hipblaslt" / "hipblaslt-version.h"
        if header.is_file():
            return header
        tried.append(str(header))
    raise RuntimeError(
        "Could not find hipblaslt-version.h. Set ROCM_PATH (or hipblaslt_ROOT) to the ROCm "
        "install the bindings should build against. Tried: "
        + ", ".join(tried or ["nothing"])
    )


def parse_version_header(text: str) -> str:
    parts = dict(_DEFINE.findall(text))
    missing = {"MAJOR", "MINOR", "PATCH"} - parts.keys()
    if missing:
        raise RuntimeError(
            f"hipblaslt-version.h has no HIPBLASLT_VERSION_{sorted(missing)[0]}"
        )
    return f"{int(parts['MAJOR'])}.{int(parts['MINOR'])}.{int(parts['PATCH'])}"


def dynamic_metadata(
    settings: Mapping[str, Any], project: Mapping[str, Any]
) -> dict[str, str]:
    if settings:
        raise RuntimeError("hipblaslt_version_provider takes no settings")
    header = find_version_header(os.environ, sys.platform)
    return {
        "version": parse_version_header(
            header.read_text(encoding="utf-8", errors="replace")
        )
    }
