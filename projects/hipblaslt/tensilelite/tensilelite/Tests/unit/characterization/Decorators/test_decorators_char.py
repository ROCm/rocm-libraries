################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``Tensile.Utilities.Decorators``: ``envVariableIsSet``
+ ``CallableGuard`` (Shared), the ``@timing`` decorator (Timing), and the
``@profile`` decorator + ``initProfileArtifacts`` (Profile)."""

import pytest

from Tensile.Utilities.Decorators.Shared import envVariableIsSet, CallableGuard
from Tensile.Utilities.Decorators.Timing import timing, TIMING_ENV_VAR
from Tensile.Utilities.Decorators.Profile import profile, initProfileArtifacts, PROFILE_ENV_VAR

pytestmark = pytest.mark.unit


# --- Shared -----------------------------------------------------------------

@pytest.mark.parametrize(
    "value,expected",
    [("YES", True), ("on", True), ("True", True), ("1", True),
     ("no", False), ("0", False), ("", False)],
    ids=["yes", "on", "true", "one", "no", "zero", "empty"],
)
def test_env_variable_is_set(monkeypatch, value, expected):
    monkeypatch.setenv("TENSILE_TEST_VAR", value)
    assert envVariableIsSet("TENSILE_TEST_VAR") is expected


def test_env_variable_unset(monkeypatch):
    monkeypatch.delenv("TENSILE_TEST_VAR", raising=False)
    assert envVariableIsSet("TENSILE_TEST_VAR") is False


def test_callable_guard_call():
    g = CallableGuard(lambda x: x + 1)
    assert g(41) == 42
    assert g.__name__ == "<lambda>"


def test_callable_guard_bool_raises():
    g = CallableGuard(lambda: True)
    with pytest.raises(TypeError):
        bool(g)


# --- Timing -----------------------------------------------------------------

def test_timing_disabled_returns_original(monkeypatch):
    monkeypatch.delenv(TIMING_ENV_VAR, raising=False)

    def f():
        return 7

    assert timing(f) is f  # unchanged when env not set


def test_timing_enabled_wraps(monkeypatch, capsys):
    monkeypatch.setenv(TIMING_ENV_VAR, "ON")

    @timing
    def f(a, b):
        return a + b

    assert f(2, 3) == 5
    assert "took" in capsys.readouterr().out


# --- Profile ----------------------------------------------------------------

def test_profile_disabled_returns_original(monkeypatch):
    monkeypatch.delenv(PROFILE_ENV_VAR, raising=False)

    def f():
        return 1

    assert profile(f) is f


def test_profile_enabled_runs_and_writes(monkeypatch, tmp_path):
    monkeypatch.setenv(PROFILE_ENV_VAR, "ON")
    monkeypatch.chdir(tmp_path)

    @profile
    def f(x):
        return x * 2

    assert f(21) == 42
    # A profiling-results-* dir with a .prof file is created under cwd.
    dirs = list(tmp_path.glob("profiling-results-*"))
    assert dirs and any(p.suffix == ".prof" for p in dirs[0].iterdir())


def test_init_profile_artifacts(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    path, filename = initProfileArtifacts("myfunc")
    assert filename.startswith("myfunc-") and filename.endswith(".prof")
    assert path.exists() and path.name.startswith("profiling-results-")
