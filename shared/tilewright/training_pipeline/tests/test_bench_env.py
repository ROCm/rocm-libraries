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


LAUNCH_OVERRIDES = (
    "TENSILE_FIXED_WGM",
    "TENSILE_FIXED_WGMXCCCHUNK",
    "TENSILE_FIXED_STAGGERU_MAPPING",
    "TENSILE_PERSISTENT_FIXED_GRID",
    "TENSILE_PERSISTENT_DYNAMIC_GRID",
    "TENSILE_PERSISTENT_HYBRID_FORCE_MODE",
    "TENSILE_STREAMK_FULL_TILES",
    "TENSILE_STREAMK_GRID_MULTIPLIER",
    "TENSILE_STREAMK5_FORCE_MODE",
    "TENSILE_DISABLE_STAGGERU",
    "TENSILE_PRECISE_SM_TARGET",
    "TENSILE_AUTO_GSU_ALGO",
)


def test_launch_overrides_are_not_inherited(tilewright_environ, monkeypatch):
    for name in LAUNCH_OVERRIDES:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("TENSILE_DB", "0x2")
    monkeypatch.setenv("TENSILE_ADAPTIVE_GEMM_NTAB_ALGO", "1")
    for env in (
        bench_env_with(),
        bench_env_with({"TENSILE_DB": "0x10000"}, tilewright={"TILEWRIGHT_DIAG": 1}),
    ):
        assert not set(LAUNCH_OVERRIDES) & set(env)
        assert bench_env.launch_overrides(env) == {}
        assert env["UNRELATED_SETTING"] == "keep-me"
        assert env["TENSILE_ADAPTIVE_GEMM_NTAB_ALGO"] == "1"
    assert bench_env_with()["TENSILE_DB"] == "0x2"
    assert os.environ["TENSILE_STREAMK5_FORCE_MODE"] == "1"


def test_runtime_env_sets_launch_overrides_deliberately(monkeypatch):
    monkeypatch.setenv("TENSILE_FIXED_WGM", "4")
    monkeypatch.setenv("TENSILE_PRECISE_SM_TARGET", "0")
    cfg = {"runtime": {"env": {"TENSILE_PRECISE_SM_TARGET": 1, "MY_KNOB": "on"}}}
    assert bench_env.runtime_env_errors(cfg["runtime"]["env"]) == []
    env = bench_env_with(bench_env.runtime_env(cfg))
    assert "TENSILE_FIXED_WGM" not in env
    assert env["TENSILE_PRECISE_SM_TARGET"] == "1" and env["MY_KNOB"] == "on"
    assert bench_env.launch_overrides(env) == {"TENSILE_PRECISE_SM_TARGET": "1"}


@pytest.mark.parametrize("name", LAUNCH_OVERRIDES)
def test_the_launch_override_family(name):
    assert bench_env.is_launch_override(name)
    assert not bench_env.managed_var(name)


@pytest.mark.parametrize(
    "name",
    ["TENSILE_DB", "TENSILE_STREAMKX", "TENSILE_ADAPTIVE_GEMM_NTAB_ALGO", "FIXED_WGM"],
)
def test_names_outside_the_launch_override_family(name):
    assert not bench_env.is_launch_override(name)


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


def test_tensile_tilewright_settings_are_tilewright_variables(monkeypatch):
    settings = {"TENSILE_TILEWRIGHT_BANK": "1", "TENSILE_TILEWRIGHT_PREPARE": "0"}
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    for env in (
        bench_env_with(),
        bench_env_with(tilewright={"TENSILE_USE_TILEWRIGHT": "1"}),
    ):
        assert not set(settings) & set(env)
    env = bench_env_with(
        tilewright={"TENSILE_USE_TILEWRIGHT": "1", "TENSILE_TILEWRIGHT_BANK": "0"}
    )
    assert env["TENSILE_TILEWRIGHT_BANK"] == "0"
    assert "TENSILE_TILEWRIGHT_PREPARE" not in env
    with pytest.raises(ValueError, match="tilewright="):
        bench_env_with({"TENSILE_TILEWRIGHT_BANK": "1"})


@pytest.mark.parametrize(
    "name, expected",
    [
        ("TENSILE_USE_TILEWRIGHT", True),
        ("TILEWRIGHT_FORCE_CELL", True),
        ("TILEWRIGHT_BENCH_ROOT", True),
        ("TENSILE_TILEWRIGHT_BANK", True),
        ("TENSILE_TILEWRIGHT_PREPARE", True),
        ("TENSILE_PREDICTION_LIB", False),
        ("MY_TILEWRIGHT_X", False),
    ],
)
def test_is_tilewright_var(name, expected):
    assert is_tilewright_var(name) is expected


def test_runtime_env_is_the_configs_variables_as_strings():
    assert bench_env.runtime_env({}) == {}
    assert bench_env.runtime_env({"runtime": {"devices": [0]}}) == {}
    cfg = {"runtime": {"env": {"HSA_DISABLE_GFX12_STRICT": 0, "MY_KNOB": "on"}}}
    assert bench_env.runtime_env(cfg) == {
        "HSA_DISABLE_GFX12_STRICT": "0",
        "MY_KNOB": "on",
    }


@pytest.mark.parametrize(
    "value, fragment",
    [
        ([("A", "1")], "expected a mapping"),
        ({"1A": "x"}, "is not an environment variable name"),
        ({"A B": "x"}, "is not an environment variable name"),
        ({"MY_KNOB": True}, "MY_KNOB: expected a non-empty string or an integer"),
        ({"MY_KNOB": ""}, "expected a non-empty string"),
        ({"MY_KNOB": 1.5}, "expected a non-empty string"),
        ({"MY_KNOB": None}, "expected a non-empty string"),
        ({"ROCR_VISIBLE_DEVICES": "0"}, "the pipeline sets this variable itself"),
        ({"HIP_VISIBLE_DEVICES": "0"}, "the pipeline sets this variable itself"),
        ({"TILEWRIGHT_DIAG": "1"}, "the pipeline sets this variable itself"),
        ({"TENSILE_USE_TILEWRIGHT": "1"}, "the pipeline sets this variable itself"),
        ({"HIPBLASLT_TENSILE_LIBPATH": "/x"}, "the pipeline sets this variable"),
        ({"TENSILE_PREDICTION_LIB": "0"}, "the pipeline sets this variable"),
    ],
)
def test_runtime_env_errors(value, fragment):
    errors = bench_env.runtime_env_errors(value, "runtime.env")
    assert any(fragment in e for e in errors), errors
    assert all(e.startswith("runtime.env") for e in errors)


def test_runtime_env_accepts_strings_and_integers():
    ok = {"HSA_DISABLE_GFX12_STRICT": "0", "MY_LEVEL": 3}
    assert bench_env.runtime_env_errors(ok) == []
    assert bench_env.runtime_env_errors(None) == []


def test_recorded_env_names_the_strict_switch_and_the_runtime_variables():
    from lib.arch import STRICT_SWITCH

    cfg = {"runtime": {"env": {"MY_KNOB": "on"}}}
    environ = {"MY_KNOB": "on", "OTHER": "x"}
    assert bench_env.recorded_env(cfg, environ) == {
        "MY_KNOB": "on",
        STRICT_SWITCH: None,
    }
    assert bench_env.recorded_env({}, {STRICT_SWITCH: "0"}) == {STRICT_SWITCH: "0"}


def test_a_production_bench_runs_without_the_prediction_library_switch(monkeypatch):
    monkeypatch.setenv("TENSILE_PREDICTION_LIB", "1")
    env = bench_env_with(
        {"TENSILE_DB": "0x10000"},
        tilewright={"TENSILE_USE_TILEWRIGHT": "1"},
        prediction_lib=False,
    )
    assert bench_env.PREDICTION_LIB_VAR == "TENSILE_PREDICTION_LIB"
    assert "TENSILE_PREDICTION_LIB" not in env
    assert env["TENSILE_USE_TILEWRIGHT"] == "1" and env["HSA_HOTSWAP_DISABLE"] == "1"
    assert env["TENSILE_DB"] == "0x10000"
    assert bench_env_with()["TENSILE_PREDICTION_LIB"] == "1"
