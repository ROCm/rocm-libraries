################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Mutation-targeted characterization tests for
``Tensile.Common.Utilities.locateExe``.

These tests pin the current behavior of the PATH lookup separator and the
"not found" error message, distinguishing the original implementation from
surviving mutants.
"""

import importlib
import os

import pytest

U = importlib.import_module("Tensile.Common.Utilities")

pytestmark = pytest.mark.unit


def _make_exe(directory, name):
    exe = directory / name
    exe.write_text("#!/bin/sh\n")
    os.chmod(str(exe), 0o755)
    return exe


def test_locateExe_splits_path_on_pathsep(tmp_path, monkeypatch):
    """PATH is split on ``os.pathsep`` so an exe in a later (non-first) PATH
    entry is found.

    Kills mutant_7: ``split(os.pathsep)`` -> ``split(None)``. With ``split(None)``
    the colon-joined PATH collapses to a single whitespace-free token and the
    real directory is never isolated, so the exe is not located.
    """
    # whitespace-free decoy directories before the real one
    decoy_a = tmp_path / "a"
    decoy_b = tmp_path / "b"
    real = tmp_path / "real"
    for d in (decoy_a, decoy_b, real):
        d.mkdir()
    exe = _make_exe(real, "tool")

    path_value = os.pathsep.join([str(decoy_a), str(decoy_b), str(real)])
    # ensure no whitespace in any component (so split(None) would NOT find it)
    assert not any(c.isspace() for c in path_value)
    monkeypatch.setenv("PATH", path_value)

    assert U.locateExe("", "tool") == str(exe)


def test_locateExe_not_found_message_contains_exeName(tmp_path, monkeypatch):
    """The OSError raised when the exe is not found includes the exe name.

    Kills mutant_16: ``raise OSError(f"Failed to locate {exeName}")`` ->
    ``raise OSError(None)``.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    with pytest.raises(OSError) as excinfo:
        U.locateExe("", "missing-tool-xyz")

    assert "missing-tool-xyz" in str(excinfo.value)
