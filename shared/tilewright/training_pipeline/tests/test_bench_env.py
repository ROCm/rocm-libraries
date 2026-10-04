# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os

import pytest

from lib import bench_env
from lib.bench_env import bench_env_with, is_tilewright_var


@pytest.fixture
def tilewright_environ(monkeypatch):
    monkeypatch.setenv("TENSILE_USE_TILEWRIGHT", "1")
    monkeypatch.setenv("TILEWRIGHT_DIAG", "1")
    monkeypatch.setenv("TILEWRIGHT_PICK_LOG", "1")
    monkeypatch.setenv("UNRELATED_SETTING", "keep-me")


def test_inherited_tilewright_variables_are_stripped(tilewright_environ):
    env = bench_env_with()
    assert not [k for k in env if is_tilewright_var(k)]
    assert env["UNRELATED_SETTING"] == "keep-me"
    for key, value in bench_env.REQUIRED_BENCH_ENV.items():
        assert env[key] == value
    assert os.environ["TENSILE_USE_TILEWRIGHT"] == "1"


def test_library_path_only_when_passed(tilewright_environ, monkeypatch):
    monkeypatch.setenv("HIPBLASLT_TENSILE_LIBPATH", "/stale/library")
    assert "HIPBLASLT_TENSILE_LIBPATH" not in bench_env_with()
    env = bench_env_with({"HIPBLASLT_TENSILE_LIBPATH": "/build/library/gfx950"})
    assert env["HIPBLASLT_TENSILE_LIBPATH"] == "/build/library/gfx950"
    assert os.environ["HIPBLASLT_TENSILE_LIBPATH"] == "/stale/library"


def test_extra_overrides_required_knobs(tilewright_environ):
    env = bench_env_with({"TENSILE_DB": 65536, "TENSILE_PREDICTION_LIB": "0"})
    assert env["TENSILE_DB"] == "65536"
    assert env["TENSILE_PREDICTION_LIB"] == "0"


def test_tilewright_knobs_only_through_the_explicit_parameter(tilewright_environ):
    env = bench_env_with(
        {"TENSILE_DB": "0x10000"},
        tilewright={"TENSILE_USE_TILEWRIGHT": 1, "TILEWRIGHT_DIAG": "1"},
    )
    assert env["TENSILE_USE_TILEWRIGHT"] == "1"
    assert env["TILEWRIGHT_DIAG"] == "1"
    assert "TILEWRIGHT_PICK_LOG" not in env
    with pytest.raises(ValueError, match="tilewright="):
        bench_env_with({"TENSILE_USE_TILEWRIGHT": "1"})
    with pytest.raises(ValueError, match="TILEWRIGHT_"):
        bench_env_with(tilewright={"TENSILE_DB": "1"})


@pytest.mark.parametrize(
    "name, expected",
    [
        ("TENSILE_USE_TILEWRIGHT", True),
        ("TILEWRIGHT_FORCE_CELL", True),
        ("TILEWRIGHT_BENCH_ROOT", True),
        ("TENSILE_PREDICTION_LIB", False),
        ("MY_TILEWRIGHT_X", False),
    ],
)
def test_is_tilewright_var(name, expected):
    assert is_tilewright_var(name) is expected
