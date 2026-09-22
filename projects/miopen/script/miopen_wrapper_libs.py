#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Locate the two libraries a hipDNN wrapper build produces.

Shared by run_forwarding_parity.py, which replays against the pair, and
check_wrapper_abi.py, which inspects it. Kept in one place so the "exactly one
versioned file per stem, both sides in the same directory" rule cannot drift
between the two callers.
"""

from pathlib import Path


def find_library(lib_dirs, stem):
    """Return the one versioned shared object for `stem`, or (None, reason).

    Symlinks are skipped so the versioned file is what gets inspected. A second
    versioned file beside it is a half-replaced earlier build -- exactly what the
    co-versioning check downstream exists to catch -- and choosing between them
    would pick by filename order, which is not version order. So both the zero
    and the many case are reported rather than resolved.
    """
    searched = ", ".join(str(d) for d in lib_dirs) or "<no lib directory>"
    for lib_dir in lib_dirs:
        matches = [
            p
            for p in sorted(lib_dir.glob(f"{stem}.so.*"))
            if p.is_file() and not p.is_symlink()
        ]
        if len(matches) == 1:
            return matches[0], None
        if matches:
            listed = ", ".join(p.name for p in matches)
            return None, (
                f"{lib_dir} holds more than one {stem}.so.*: {listed}. One of them is "
                "left over from an earlier build; remove it and run again."
            )
    return None, f"no {stem}.so.* found under {searched}"


def resolve_pair(lib_dirs):
    """Return (wrapper_lib, private_lib, problems); problems is empty on success.

    Each stem is resolved independently, so a tree carrying both lib and lib64 can
    split the pair. That split matters to callers who load the pair (only one
    directory can go first on LD_LIBRARY_PATH) and callers who just inspect it
    (the two halves of one build are expected to ship side by side), so it is
    caught here rather than left to each caller to notice on its own.
    """
    wrapper_lib, wrapper_problem = find_library(lib_dirs, "libMIOpen")
    private_lib, private_problem = find_library(lib_dirs, "libMIOpen_private")
    problems = [p for p in (wrapper_problem, private_problem) if p]
    if problems:
        return None, None, problems
    if wrapper_lib.parent != private_lib.parent:
        return (
            None,
            None,
            [
                "wrapper and private library are in different directories:\n"
                f"  {wrapper_lib}\n"
                f"  {private_lib}\n"
                "They are halves of one build and must be installed side by side."
            ],
        )
    return wrapper_lib, private_lib, []
