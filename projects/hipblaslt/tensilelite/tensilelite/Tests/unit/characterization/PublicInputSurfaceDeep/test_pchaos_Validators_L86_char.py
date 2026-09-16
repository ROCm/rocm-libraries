################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""PublicInputSurfaceDeep characterization: the ``if not os.name == "nt"``
guard predicate in ``Tensile/Toolchain/Validators.py`` at line 86, inside
``_windowsWithExtensions``.

Branch cc98dba04c70eae953a89ae32af1df2b3f42fb17.  The predicate tests the
CPython runtime attribute ``os.name`` against the sentinel ``"nt"`` (Windows).

  * TRUE branch  -> ``os.name != "nt"`` (POSIX/Linux).  Raises ``ValueError``
                    immediately: "These extensions should not be added on
                    anything but Windows".
  * FALSE branch -> ``os.name == "nt"`` (Windows).  Proceeds to build and
                    return a list of the exe name with PATHEXT extensions.

Domain is exhaustive over {posix, nt}.  Tests pin ACTUAL observed behavior
via monkeypatching ``os.name`` in the real function.  CPU-only, no GPU probe.
"""

import os

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Pure-helper: extract the predicate logic inline for isolated testing
# ---------------------------------------------------------------------------


def _guard_raises_on_posix(os_name: str) -> bool:
    """Mirror the predicate at Validators.py:86.

    Returns True when the guard would raise (i.e., os.name != 'nt').
    Returns False when os.name == 'nt' (Windows, guard skipped).
    """
    return not os_name == "nt"


# ---------------------------------------------------------------------------
# Pure-helper tests (no I/O, no imports needed)
# ---------------------------------------------------------------------------


def test_posix_os_name_guard_is_true():
    """os.name='posix' -> predicate True -> guard raises (TRUE branch)."""
    assert _guard_raises_on_posix("posix") is True


def test_nt_os_name_guard_is_false():
    """os.name='nt' -> predicate False -> guard skipped (FALSE branch)."""
    assert _guard_raises_on_posix("nt") is False


# ---------------------------------------------------------------------------
# Integration tests: call the real _windowsWithExtensions with monkeypatched
# os.name to pin ACTUAL function behavior for each domain value.
# ---------------------------------------------------------------------------


def _import_windows_with_extensions():
    """Import the private _windowsWithExtensions function."""
    import importlib
    M = importlib.import_module("Tensile.Toolchain.Validators")
    return M._windowsWithExtensions


def test_real_function_posix_raises_value_error(monkeypatch):
    """TRUE branch (os.name='posix'): _windowsWithExtensions raises ValueError.

    On POSIX the guard at line 86 fires immediately.  The exact message is
    "These extensions should not be added on anything but Windows".
    """
    _windowsWithExtensions = _import_windows_with_extensions()
    monkeypatch.setattr(os, "name", "posix")
    with pytest.raises(ValueError, match="These extensions should not be added on anything but Windows"):
        _windowsWithExtensions("hipcc")


def test_real_function_nt_returns_list(monkeypatch):
    """FALSE branch (os.name='nt'): _windowsWithExtensions returns list with extensions.

    On Windows the guard is skipped and the function reads PATHEXT to build
    a list of exe variants.  We monkeypatch both os.name and PATHEXT to
    exercise the false branch deterministically in container.
    """
    _windowsWithExtensions = _import_windows_with_extensions()
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT")

    result = _windowsWithExtensions("hipcc")

    # Must be a list starting with the bare exe name
    assert isinstance(result, list)
    assert result[0] == "hipcc"
    # Must include lower-cased PATHEXT variants
    assert "hipcc.com" in result
    assert "hipcc.exe" in result
    assert "hipcc.bat" in result
