################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
################################################################################
"""When Tests/unit may turn the gfx1250 codegen guard into a skip."""
import importlib.util
import os
import subprocess

import pytest

pytestmark = pytest.mark.unit

_GUARD_LOADED_TRUE = RuntimeError(
    "gfx1250 kernel None is about to be generated with rocisa._BACKEND='' "
    "(stinkytofu._stinkytofu loaded=True); "
    "expected the stinkytofu backend with its native extension loaded."
)
_GUARD_LOADED_FALSE = RuntimeError(
    "gfx1250 kernel None is about to be generated with rocisa._BACKEND='' "
    "(stinkytofu._stinkytofu loaded=False); "
    "expected the stinkytofu backend with its native extension loaded."
)
_EXPLICIT_MISMATCH = RuntimeError(
    "kernel 1 (ISA=(12, 5, 0)): ROCISA_BACKEND='stinkytofu' was requested but "
    "the active backend is rocisa._BACKEND='rocisa'"
)


def _tests_conftest():
    path = os.path.join(os.path.dirname(__file__), os.pardir, "conftest.py")
    spec = importlib.util.spec_from_file_location("tensile_tests_conftest", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def skip_mod():
    return _tests_conftest()


def test_skip_when_host_is_not_gfx1250_and_backend_is_unset(monkeypatch, skip_mod):
    monkeypatch.delenv("ROCISA_BACKEND", raising=False)
    monkeypatch.setattr(skip_mod, "_host_sees_gfx1250", lambda: False)
    assert skip_mod._skip_gfx1250_codegen_on_this_host(_GUARD_LOADED_TRUE)
    assert skip_mod._skip_gfx1250_codegen_on_this_host(_GUARD_LOADED_FALSE)


def test_keep_failure_on_a_gfx1250_host(monkeypatch, skip_mod):
    monkeypatch.delenv("ROCISA_BACKEND", raising=False)
    monkeypatch.setattr(skip_mod, "_host_sees_gfx1250", lambda: True)
    assert not skip_mod._skip_gfx1250_codegen_on_this_host(_GUARD_LOADED_TRUE)


def test_keep_failure_when_backend_is_explicit(monkeypatch, skip_mod):
    monkeypatch.setenv("ROCISA_BACKEND", "stinkytofu")
    monkeypatch.setattr(skip_mod, "_host_sees_gfx1250", lambda: False)
    assert not skip_mod._skip_gfx1250_codegen_on_this_host(_GUARD_LOADED_TRUE)


def test_keep_other_runtime_errors(monkeypatch, skip_mod):
    monkeypatch.delenv("ROCISA_BACKEND", raising=False)
    monkeypatch.setattr(skip_mod, "_host_sees_gfx1250", lambda: False)
    assert not skip_mod._skip_gfx1250_codegen_on_this_host(_EXPLICIT_MISMATCH)
    assert not skip_mod._skip_gfx1250_codegen_on_this_host(ValueError("gfx1250 kernel"))


def test_host_sees_gfx1250_reads_the_enumerator(monkeypatch, skip_mod):
    previous = getattr(skip_mod._host_sees_gfx1250, "_cached", None)
    monkeypatch.delattr(skip_mod._host_sees_gfx1250, "_cached", raising=False)
    monkeypatch.setattr(os.path, "exists", lambda _path: True)
    monkeypatch.setattr(subprocess, "check_output", lambda *_a, **_k: b"gfx942\ngfx1250\n")
    try:
        assert skip_mod._host_sees_gfx1250() is True
    finally:
        if previous is None:
            if hasattr(skip_mod._host_sees_gfx1250, "_cached"):
                delattr(skip_mod._host_sees_gfx1250, "_cached")
        else:
            skip_mod._host_sees_gfx1250._cached = previous
