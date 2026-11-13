# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The files of a hipBLASLt build that hold TensileLite's host code. In the
build stage 7 runs, run_orchestrator.py's preflight looks in them for the
strings that show what the build's engine and launch planner serve
(`stage07_capabilities`, `stage07_build_errors`), and a pass of stage 7
records them by name, size and time, so that a rebuild shows
(`build_libraries`).
"""
from __future__ import annotations

import mmap
from pathlib import Path
from typing import Any, List


# Files of a hipBLASLt build that hold TensileLite's host code: its shared
# TensileLite library, else hipBLASLt's own library (a static TensileLite).
TENSILELITE_LIBRARIES = (
    "tensilelite/libtensilelite-host.so*",
    "library/libhipblaslt.so*",
)


def tensilelite_libraries(build_dir: Path) -> List[Path]:
    for pattern in TENSILELITE_LIBRARIES:
        found = {p.resolve() for p in Path(build_dir).glob(pattern) if p.is_file()}
        if found:
            return sorted(found)
    return []


def build_libraries(build_dir: Path) -> List[List[Any]]:
    """[name, size, mtime] of the libraries a build's bench loads (its
    TensileLite and hipBLASLt), which change when the build is rebuilt."""
    out = []
    for pattern in TENSILELITE_LIBRARIES:
        for p in sorted({p.resolve() for p in Path(build_dir).glob(pattern)}):
            if p.is_file():
                st = p.stat()
                out.append([p.name, st.st_size, st.st_mtime_ns])
    return out


def holds(path: Path, marker: bytes) -> bool:
    try:
        with path.open("rb") as f, mmap.mmap(
            f.fileno(), 0, access=mmap.ACCESS_READ
        ) as m:
            return m.find(marker) >= 0
    except (OSError, ValueError):
        return False
