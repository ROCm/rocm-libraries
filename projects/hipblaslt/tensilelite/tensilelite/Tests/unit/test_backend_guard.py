################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
"""Unit tests for the rocisa backend guard in Tensile.TensileCreateLibrary.Run."""
import sys
import types

import pytest

import rocisa
from Tensile.TensileCreateLibrary.Run import _assertCorrectBackendForKernel, _GFX1250

pytestmark = pytest.mark.unit

_GFX1250_KERNEL = {"ISA": (_GFX1250[0], _GFX1250[1], 0), "SolutionIndex": 1}
_NON_GFX1250_KERNEL = {"ISA": (9, 4, 2), "SolutionIndex": 2}


def _set_backend(monkeypatch, backend, *, stinky_loaded, native_loaded):
    # Pin the active backend string and which nanobind extension looks loaded.
    monkeypatch.setattr(rocisa, "_BACKEND", backend, raising=False)
    for name, loaded in (("stinkytofu._stinkytofu", stinky_loaded), ("rocisa._rocisa", native_loaded)):
        if loaded:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
        else:
            monkeypatch.delitem(sys.modules, name, raising=False)


def test_env_set_backend_mismatch_raises(monkeypatch):
    # ROCISA_BACKEND requested but the active backend disagrees.
    monkeypatch.setenv("ROCISA_BACKEND", "stinkytofu")
    _set_backend(monkeypatch, "rocisa", stinky_loaded=True, native_loaded=True)
    with pytest.raises(RuntimeError, match="active backend"):
        _assertCorrectBackendForKernel(_GFX1250_KERNEL)


def test_env_stinky_extension_not_loaded_raises(monkeypatch):
    # Requested stinkytofu, backend agrees, but the extension is not loaded.
    monkeypatch.setenv("ROCISA_BACKEND", "stinkytofu")
    _set_backend(monkeypatch, "stinkytofu", stinky_loaded=False, native_loaded=True)
    with pytest.raises(RuntimeError, match="_stinkytofu"):
        _assertCorrectBackendForKernel(_GFX1250_KERNEL)


def test_env_native_extension_not_loaded_raises(monkeypatch):
    # Requested native rocisa, backend agrees, but the extension is not loaded.
    monkeypatch.setenv("ROCISA_BACKEND", "rocisa")
    _set_backend(monkeypatch, "rocisa", stinky_loaded=True, native_loaded=False)
    with pytest.raises(RuntimeError, match="_rocisa"):
        _assertCorrectBackendForKernel(_NON_GFX1250_KERNEL)


def test_env_set_consistent_passes(monkeypatch):
    # Requested and active backend match with the extension loaded: no raise.
    monkeypatch.setenv("ROCISA_BACKEND", "stinkytofu")
    _set_backend(monkeypatch, "stinkytofu", stinky_loaded=True, native_loaded=True)
    _assertCorrectBackendForKernel(_GFX1250_KERNEL)


def test_auto_gfx1250_wrong_backend_raises(monkeypatch):
    # Auto-detected regime: a gfx1250 kernel must not go through native rocisa.
    monkeypatch.delenv("ROCISA_BACKEND", raising=False)
    _set_backend(monkeypatch, "rocisa", stinky_loaded=False, native_loaded=True)
    with pytest.raises(RuntimeError, match="gfx1250 kernel"):
        _assertCorrectBackendForKernel(_GFX1250_KERNEL)


def test_auto_gfx1250_stinkytofu_passes(monkeypatch):
    # Auto-detected regime: gfx1250 on stinkytofu with its extension loaded is fine.
    monkeypatch.delenv("ROCISA_BACKEND", raising=False)
    _set_backend(monkeypatch, "stinkytofu", stinky_loaded=True, native_loaded=False)
    _assertCorrectBackendForKernel(_GFX1250_KERNEL)


def test_auto_non_gfx1250_wrong_backend_raises(monkeypatch):
    # Auto-detected regime: a non-gfx1250 kernel must not go through the adapter.
    monkeypatch.delenv("ROCISA_BACKEND", raising=False)
    _set_backend(monkeypatch, "stinkytofu", stinky_loaded=True, native_loaded=False)
    with pytest.raises(RuntimeError, match="non-gfx1250 kernel"):
        _assertCorrectBackendForKernel(_NON_GFX1250_KERNEL)


def test_auto_non_gfx1250_native_passes(monkeypatch):
    # Auto-detected regime: non-gfx1250 on native rocisa with its extension loaded is fine.
    monkeypatch.delenv("ROCISA_BACKEND", raising=False)
    _set_backend(monkeypatch, "rocisa", stinky_loaded=False, native_loaded=True)
    _assertCorrectBackendForKernel(_NON_GFX1250_KERNEL)
