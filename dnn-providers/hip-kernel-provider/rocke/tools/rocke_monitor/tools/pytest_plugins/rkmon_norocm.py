# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""pytest plugin: ROCm's tools hidden, as in CI's test image (no /opt/rocm, no ROCm LLVM tools on PATH).

The runner (tools/ci_condition.py) gives a PATH without the folders that hold ROCm's tools and unsets ROCM_PATH and its
kin; this plugin hides the rest from the test's own lookups: every folder RKMON_NOROCM_HIDE names (os.pathsep-separated;
/opt/rocm, the former ROCM_PATH, the dropped PATH folders) reads as absent to os.stat, os.lstat and os.access, so
`os.path.exists`, `pathlib.Path.is_file` and `shutil.which` find nothing there. A program the test starts by its
absolute path still runs: the operating system's exec is not hidden. A tool the test writes itself is still found.
"""

import os

HIDE = tuple(
    os.path.normpath(p)
    for p in os.environ.get("RKMON_NOROCM_HIDE", "").split(os.pathsep)
    if p.strip()
)


def hidden(path) -> bool:
    if isinstance(path, int):  # a file descriptor
        return False
    try:
        p = os.path.normpath(os.fsdecode(os.fspath(path)))
    except TypeError:
        return False
    return any(p == h or p.startswith(h.rstrip("/") + "/") for h in HIDE)


def _wrap(fn):
    def inner(path, *a, **kw):
        if hidden(path):
            raise FileNotFoundError(
                2, "No such file or directory (rkmon: hidden as in CI)", os.fspath(path)
            )
        return fn(path, *a, **kw)

    inner.__wrapped__ = fn
    return inner


def _access(fn):
    def inner(path, *a, **kw):
        return False if hidden(path) else fn(path, *a, **kw)

    inner.__wrapped__ = fn
    return inner


if HIDE:
    os.stat, os.lstat, os.access = _wrap(os.stat), _wrap(os.lstat), _access(os.access)
    for _k in ("ROCM_PATH", "ROCM_HOME", "HIP_PATH"):
        os.environ.pop(_k, None)
