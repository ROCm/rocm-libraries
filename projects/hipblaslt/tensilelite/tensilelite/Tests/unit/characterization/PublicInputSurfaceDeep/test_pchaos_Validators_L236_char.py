################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

# Characterization of Tensile/Toolchain/Validators.py:236 [if]
#   branch_id: 9a47d378ae60f3b7013ec37c14e996c9825948cf
#   function: _validateExecutable
#   predicate: Path(file).is_absolute()
#     true_branch  -> check _exeExists(Path(file)) (absolute path fast path)
#     false_branch -> fall through to search-path loop
#
# Classification: solver-backed-under-assumptions (POSIX flavour).
# pathlib.Path(file).is_absolute() dispatches on os.name; on POSIX a path is
# absolute iff it starts with '/'. The Windows branch (PureWindowsPath) would
# treat drive-letter roots (C:/...) as absolute, but in-container os.name=='posix'
# so the POSIX semantics are the deployed behavior.
#
# Pure helper is_absolute_posix captures the invariant for property-level testing.
# Witnesses confirmed in-container (tl-char) by the Verify phase:
#   TRUE  -> /opt/rocm/bin/amdclang++                           -> True
#   FALSE -> amdclang++                                         -> False
#   FALSE -> ./amdclang++                                       -> False
#   FALSE -> C:/Program Files/AMD/ROCm/5.7/bin/clang++.exe     -> False (POSIX)
#
# CPU-only. No GPU. Deterministic.

from pathlib import Path, PurePosixPath, PureWindowsPath

import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Pure helper: POSIX semantics of Path(file).is_absolute()
# ---------------------------------------------------------------------------

def is_absolute_posix(file: str) -> bool:
    """POSIX semantics of pathlib.Path(file).is_absolute(): absolute iff starts with '/'."""
    return file.startswith("/")


# ---------------------------------------------------------------------------
# Pure-helper tests (predicate characterization, no filesystem)
# ---------------------------------------------------------------------------

def test_is_absolute_posix_true_rocm_path():
    """Absolute ROCm toolchain path is absolute under POSIX (true_branch witness)."""
    file = "/opt/rocm/bin/amdclang++"
    assert is_absolute_posix(file) is True


def test_is_absolute_posix_false_bare_name():
    """Bare executable name without leading slash is NOT absolute (false_branch witness)."""
    file = "amdclang++"
    assert is_absolute_posix(file) is False


def test_is_absolute_posix_false_relative_dot_slash():
    """Relative path starting with ./ is NOT absolute (false_branch witness)."""
    file = "./amdclang++"
    assert is_absolute_posix(file) is False


def test_is_absolute_posix_false_windows_drive_letter():
    """Windows drive-letter path is NOT absolute under POSIX semantics (false_branch witness).

    This is the platform-divergence case: PureWindowsPath would call this True,
    but os.name=='posix' (deployed/in-container), so Path.is_absolute() returns False.
    """
    file = "C:/Program Files/AMD/ROCm/5.7/bin/clang++.exe"
    assert is_absolute_posix(file) is False


# ---------------------------------------------------------------------------
# Real-predicate tests: confirm pathlib.Path.is_absolute() matches on POSIX
# ---------------------------------------------------------------------------

def test_pathlib_agrees_true_rocm_path():
    """pathlib.Path.is_absolute() returns True for the absolute ROCm path (POSIX)."""
    file = "/opt/rocm/bin/amdclang++"
    assert Path(file).is_absolute() is True


def test_pathlib_agrees_false_bare_name():
    """pathlib.Path.is_absolute() returns False for a bare executable name."""
    file = "amdclang++"
    assert Path(file).is_absolute() is False


def test_pathlib_agrees_false_relative_dot_slash():
    """pathlib.Path.is_absolute() returns False for a ./ relative path."""
    file = "./amdclang++"
    assert Path(file).is_absolute() is False


def test_pathlib_agrees_false_windows_drive_letter_on_posix():
    """pathlib.Path.is_absolute() returns False for a Windows drive-letter path on POSIX."""
    file = "C:/Program Files/AMD/ROCm/5.7/bin/clang++.exe"
    assert Path(file).is_absolute() is False


# ---------------------------------------------------------------------------
# Cross-flavour divergence: confirm Windows flavour would differ for C:/ path
# ---------------------------------------------------------------------------

def test_pure_windows_path_would_be_true_for_drive_letter():
    """PureWindowsPath.is_absolute() returns True for the drive-letter path.

    This confirms the solver-backed-under-assumptions classification: the Windows
    code path would branch True, but POSIX (deployed) branches False.
    """
    file = "C:/Program Files/AMD/ROCm/5.7/bin/clang++.exe"
    assert PureWindowsPath(file).is_absolute() is True
    assert PurePosixPath(file).is_absolute() is False


# ---------------------------------------------------------------------------
# Helper-vs-pathlib agreement check across all 4 witnesses
# ---------------------------------------------------------------------------

def test_helper_agrees_with_pathlib_all_witnesses():
    """is_absolute_posix() and Path.is_absolute() agree on all 4 confirmed witnesses."""
    witnesses = [
        "/opt/rocm/bin/amdclang++",
        "C:/Program Files/AMD/ROCm/5.7/bin/clang++.exe",
        "amdclang++",
        "./amdclang++",
    ]
    for file in witnesses:
        assert is_absolute_posix(file) == Path(file).is_absolute(), (
            f"Helper and pathlib disagree on: {file!r}"
        )
