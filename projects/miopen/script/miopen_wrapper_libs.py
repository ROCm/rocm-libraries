#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Locate the two libraries a hipDNN wrapper build produces.

Shared by run_forwarding_parity.py, which replays against the pair, and
check_wrapper_abi.py, which inspects it. Kept in one place so the "exactly one
file per library, both sides in the same directory" rule cannot drift between
the two callers.

Linux and Windows name and place the libraries differently. Every function takes
the platform as an argument, defaulting to the host, so that each layout can be
tested on either host.
"""

import sys

ON_WINDOWS = sys.platform == "win32"

WRAPPER = "MIOpen"
PRIVATE = "MIOpen_private"


def library_pattern(name, windows=ON_WINDOWS):
    """Return the file name pattern of library `name` (WRAPPER or PRIVATE).

    Linux installs a versioned libMIOpen.so.1.0 beside its unversioned links;
    Windows installs a single MIOpen.dll, with no prefix and no version.
    """
    return f"{name}.dll" if windows else f"lib{name}.so.*"


def default_lib_dirs(script_dir, windows=ON_WINDOWS):
    """Return the directories of the install prefix that hold the libraries.

    The scripts install to <prefix>/bin/miopen_wrapper_tests/. Linux puts the
    libraries in the prefix's lib*/; Windows puts DLLs in bin/, beside the programs
    that load them.
    """
    if windows:
        return [script_dir.parent]
    return sorted(script_dir.parent.parent.glob("lib*"))


def loader_path_variable(windows=ON_WINDOWS):
    """Return the environment variable the loader searches for shared libraries."""
    return "PATH" if windows else "LD_LIBRARY_PATH"


def find_library(lib_dirs, name, windows=ON_WINDOWS):
    """Return the one file of library `name`, or (None, reason).

    On Linux, symlinks are skipped so the versioned file is what gets inspected.
    A second versioned file beside it is a half-replaced earlier build -- exactly
    what the co-versioning check downstream exists to catch -- and choosing between
    them would pick by filename order, which is not version order. So both the
    zero and the many case are reported rather than resolved.
    """
    pattern = library_pattern(name, windows)
    searched = ", ".join(str(d) for d in lib_dirs) or "<no lib directory>"
    for lib_dir in lib_dirs:
        matches = [
            p
            for p in sorted(lib_dir.glob(pattern))
            if p.is_file() and (windows or not p.is_symlink())
        ]
        if len(matches) == 1:
            return matches[0], None
        if matches:
            listed = ", ".join(p.name for p in matches)
            return None, (
                f"{lib_dir} holds more than one {pattern}: {listed}. One of them is "
                "left over from an earlier build; remove it and run again."
            )
    return None, f"no {pattern} found under {searched}"


def resolve_pair(lib_dirs, windows=ON_WINDOWS):
    """Return (wrapper_lib, private_lib, problems); problems is empty on success.

    Each library is resolved independently, so a tree carrying both lib and lib64
    can split the pair. That split matters to callers who load the pair (only one
    directory can go first on the loader's search path) and callers who just
    inspect it (the two halves of one build are expected to ship side by side), so
    it is caught here rather than left to each caller to notice on its own.
    """
    wrapper_lib, wrapper_problem = find_library(lib_dirs, WRAPPER, windows)
    private_lib, private_problem = find_library(lib_dirs, PRIVATE, windows)
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
