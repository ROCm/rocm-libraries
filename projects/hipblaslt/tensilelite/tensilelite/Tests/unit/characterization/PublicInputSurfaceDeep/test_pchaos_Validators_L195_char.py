################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""PublicInputSurfaceDeep characterization: the ``if os.name == "nt"``
guard predicate in ``Tensile/Toolchain/Validators.py`` at line 195, inside
``supportedDeviceEnumerator``.

Branch ffb27402fcf84b1a7fc60ace96f00b277b0d29b7.  The predicate tests the
CPython runtime attribute ``os.name`` against the sentinel ``"nt"`` (Windows).

  * TRUE branch  -> ``os.name == "nt"`` (Windows).  Delegates to
                    ``_supportedComponent(enumerator, ["hipinfo", "hipInfo"])``.
  * FALSE branch -> ``os.name != "nt"`` (POSIX/Linux).  Delegates to
                    ``_supportedComponent(enumerator, ["rocm_agent_enumerator",
                    "amdgpu-arch"])``.

Domain is exhaustive over {posix, nt}.  Tests pin ACTUAL observed behavior
via monkeypatching ``os.name`` in the real function.  CPU-only, no GPU probe.
"""

import os

import pytest

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# Pure-helper: extract the predicate logic inline for isolated testing
# ---------------------------------------------------------------------------


def _nt_branch_selected(os_name: str) -> bool:
    """Mirror the predicate at Validators.py:195.

    Returns True when the Windows (NT) branch is selected (os.name == 'nt').
    Returns False when the POSIX branch is selected (os.name != 'nt').
    """
    return os_name == "nt"


# ---------------------------------------------------------------------------
# Pure-helper tests (no I/O, no imports needed)
# ---------------------------------------------------------------------------


def test_nt_os_name_selects_windows_branch():
    """os.name='nt' -> predicate True -> Windows hipinfo branch selected."""
    assert _nt_branch_selected("nt") is True


def test_posix_os_name_selects_posix_branch():
    """os.name='posix' -> predicate False -> POSIX rocm_agent_enumerator branch selected."""
    assert _nt_branch_selected("posix") is False


# ---------------------------------------------------------------------------
# Integration tests: call the real supportedDeviceEnumerator with monkeypatched
# os.name to pin ACTUAL function behavior for each domain value.
# ---------------------------------------------------------------------------


def _import_supported_device_enumerator():
    """Import the public supportedDeviceEnumerator function."""
    import importlib
    M = importlib.import_module("Tensile.Toolchain.Validators")
    return M.supportedDeviceEnumerator


def test_real_function_nt_accepts_hipinfo(monkeypatch):
    """TRUE branch (os.name='nt'): supportedDeviceEnumerator returns True for 'hipinfo'.

    On Windows the function routes to _supportedComponent with targets
    ['hipinfo', 'hipInfo'], so 'hipinfo' must be accepted.
    """
    supportedDeviceEnumerator = _import_supported_device_enumerator()
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT")

    assert supportedDeviceEnumerator("hipinfo") is True


@pytest.mark.nt_path_simulation
def test_real_function_nt_rejects_rocm_agent_enumerator(monkeypatch):
    """TRUE branch (os.name='nt'): supportedDeviceEnumerator returns False for 'rocm_agent_enumerator'.

    On Windows the function routes to _supportedComponent with targets
    ['hipinfo', 'hipInfo'], so 'rocm_agent_enumerator' must be rejected.
    """
    supportedDeviceEnumerator = _import_supported_device_enumerator()
    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setenv("PATHEXT", ".COM;.EXE;.BAT")

    assert supportedDeviceEnumerator("rocm_agent_enumerator") is False


def test_real_function_posix_accepts_rocm_agent_enumerator(monkeypatch):
    """FALSE branch (os.name='posix'): supportedDeviceEnumerator returns True for 'rocm_agent_enumerator'.

    On POSIX the function routes to _supportedComponent with targets
    ['rocm_agent_enumerator', 'amdgpu-arch'], so 'rocm_agent_enumerator' must be accepted.
    """
    supportedDeviceEnumerator = _import_supported_device_enumerator()
    monkeypatch.setattr(os, "name", "posix")

    assert supportedDeviceEnumerator("rocm_agent_enumerator") is True


def test_real_function_posix_accepts_amdgpu_arch(monkeypatch):
    """FALSE branch (os.name='posix'): supportedDeviceEnumerator returns True for 'amdgpu-arch'.

    On POSIX the function routes to _supportedComponent with targets
    ['rocm_agent_enumerator', 'amdgpu-arch'], so 'amdgpu-arch' must be accepted.
    """
    supportedDeviceEnumerator = _import_supported_device_enumerator()
    monkeypatch.setattr(os, "name", "posix")

    assert supportedDeviceEnumerator("amdgpu-arch") is True


def test_real_function_posix_rejects_hipinfo(monkeypatch):
    """FALSE branch (os.name='posix'): supportedDeviceEnumerator returns False for 'hipinfo'.

    On POSIX the function routes to _supportedComponent with targets
    ['rocm_agent_enumerator', 'amdgpu-arch'], so 'hipinfo' must be rejected.
    """
    supportedDeviceEnumerator = _import_supported_device_enumerator()
    monkeypatch.setattr(os, "name", "posix")

    assert supportedDeviceEnumerator("hipinfo") is False
