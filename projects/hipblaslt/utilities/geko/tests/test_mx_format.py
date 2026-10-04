# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""One MX format per GEMM: input keys, hipBLASLt scale values, logs and emitted fields."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from geko.config_generator.config_generator import mi_design_mx_options
from geko.config_generator.config_sections_generator import ConfigSectionGenerator
from geko.config_generator.constants import mx_format, mx_format_from_scale_code, mx_scale_code
from geko.config_generator.load_input_config import (
    apply_input_config_defaults,
    gemm_configs_from_gemm_dataframe,
    get_gemm_problem,
    load_prepared_config_from_yaml,
    resolve_mx_defaults,
    validate_input_config,
)
from geko.schemas import GemmConfig, GemmType

_F4 = GemmType.from_tensile("T", "N", "F4", "B", "S")
_F8 = GemmType.from_tensile("T", "N", "F8", "B", "S")
_B = GemmType.from_tensile("T", "N", "B", "B", "S")
_SIZES = [[256, 256, 1, 256]]


@pytest.mark.parametrize(
    "fmt,code",
    [((32, "E8"), 3), ((16, "E8"), 4), ((32, "F8"), 5), ((16, "F8"), 6), ((32, "E5M3"), 7), ((16, "E5M3"), 8)],
)
def test_scale_values_round_trip(fmt: tuple, code: int) -> None:
    assert mx_scale_code("gfx1250", fmt) == code
    assert mx_format_from_scale_code(code) == fmt


@pytest.mark.parametrize(
    "arch,fmt,code",
    [
        ("gfx950", (32, "E8"), 1001),
        ("gfx950_128cu", (32, "E8"), 1001),
        ("gfx950", (16, "F8"), 6),
        ("gfx1250-strict_96cu", (32, "E8"), 3),
        (None, (32, "E8"), 3),
        ("gfx1250", None, 0),
    ],
)
def test_scale_value_per_arch(arch: str | None, fmt: tuple | None, code: int) -> None:
    assert mx_scale_code(arch, fmt) == code


def test_scale_values_without_block_scaling() -> None:
    assert [mx_format_from_scale_code(c) for c in (0, 1, 2)] == [None, None, None]
    assert mx_format_from_scale_code(1001) == (32, "E8")
    with pytest.raises(ValueError, match="Unknown hipBLASLt scale value"):
        mx_format_from_scale_code(9)


@pytest.mark.parametrize(
    "gt,kw,expected",
    [
        (_F4, {}, (True, None, None)),
        (_F8, {}, (False, None, None)),
        (_F8, {"mx": True}, (True, None, None)),
        (_F8, {"mx_block": 32}, (True, 32, None)),
        (_F8, {"mx_block": 0}, (False, None, None)),
        (_F4, {"mx_block": 16, "mx_scale_type": "f8"}, (True, 16, "F8")),
        (_F4, {"mx_scale_type": "E8"}, (True, None, None)),
    ],
)
def test_gemm_config_mx_spellings(gt: GemmType, kw: dict, expected: tuple) -> None:
    gc = GemmConfig(gt, _SIZES, **kw)
    assert (gc.mx, gc.mx_block, gc.mx_scale_type) == expected


@pytest.mark.parametrize(
    "gt,kw,match",
    [
        (_F4, {"mx_block": 0}, "always MX"),
        (_F8, {"mx": True, "mx_block": 0}, "conflicts"),
        (_F8, {"mx_block": 64}, "MX_BLOCK must be"),
        (_F8, {"mx_scale_type": "F8"}, "needs MX"),
        (_F4, {"mx_scale_type": "UE8M0"}, "MX_SCALE_TYPE must be"),
        (_B, {"mx_block": 32}, "not compatible"),
    ],
)
def test_gemm_config_rejects_contradictions(gt: GemmType, kw: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        GemmConfig(gt, _SIZES, **kw)


@pytest.mark.parametrize(
    "gt,kw,name",
    [
        (_F4, {}, "F4BS_TN"),
        (_F4, {"mx_block": 32}, "F4BS_TN"),
        (_F4, {"mx_block": 16, "mx_scale_type": "F8"}, "F4BS_MXF8B16_TN"),
        (_F8, {}, "F8BS_TN"),
        (_F8, {"mx_block": 32}, "F8BS_MXE8B32_TN"),
    ],
)
def test_names_tell_mx_formats_apart(gt: GemmType, kw: dict, name: str) -> None:
    assert GemmConfig(gt, _SIZES, **kw).name == name


def test_mx_format_takes_the_arch_block() -> None:
    gc = GemmConfig(_F8, _SIZES, mx=True)
    assert mx_format(gc, "gfx1250") == (32, "E8")
    assert mx_format(gc, "gfx950") == (32, "E8")
    assert mx_format(gc, None) == (32, "E8")
    with pytest.raises(ValueError, match="not supported on ARCH 'gfx942'"):
        mx_format(gc, "gfx942")
    assert mx_format(GemmConfig(_F8, _SIZES), "gfx942") is None


def _row(a_type: str, scale_a: int, scale_b: int | None = None, m: int = 256) -> dict:
    return {
        "transA": "T", "transB": "N", "a_type": a_type, "b_type": a_type, "c_type": "bf16_r",
        "d_type": "bf16_r", "compute_type": "c_f32_r", "M": m, "N": 256, "K": 256, "batch_count": 1,
        "scaleA": scale_a, "scaleB": scale_a if scale_b is None else scale_b,
    }


def test_log_rows_group_by_mx_format() -> None:
    df = pd.DataFrame([
        _row("f8_r", 0), _row("f8_r", 1, m=512), _row("f8_r", 3, m=1024),
        _row("f4_r", 6), _row("f4_r", 1001), _row("f4_r", 3, m=512),
    ])
    got = {(gc.name, gc.mx_block, gc.mx_scale_type, len(gc.sizes)) for gc in gemm_configs_from_gemm_dataframe(df)}
    assert got == {
        ("F8BS_TN", None, None, 2),
        ("F8BS_MXE8B32_TN", 32, None, 1),
        ("F4BS_MXF8B16_TN", 16, "F8", 1),
        ("F4BS_TN", 32, None, 2),
    }


def test_log_rows_need_one_mx_format_for_a_and_b() -> None:
    with pytest.raises(ValueError, match="share one MX format"):
        gemm_configs_from_gemm_dataframe(pd.DataFrame([_row("f8_r", 3, scale_b=0)]))


def _log_config(tmp_path: Path, rows: list, **keys) -> dict:
    template = yaml.safe_load((Path(__file__).parent / "test_data" / "workload.yaml").read_text())[0]
    log = tmp_path / "log.yaml"
    log.write_text(yaml.safe_dump([{**template, **row} for row in rows]))
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"ARCH": "gfx1250", "SIZE_OPTION": 2, "GEMM_LOG_PATH": str(log), **keys}))
    return load_prepared_config_from_yaml(cfg)


def test_config_mx_keys_fill_log_rows_without_a_format(tmp_path: Path) -> None:
    rows = [
        {"a_type": "f8_r", "b_type": "f8_r", "scaleA": 0, "scaleB": 0},
        {"a_type": "f8_r", "b_type": "f8_r", "scaleA": 3, "scaleB": 3, "M": 1024},
        {"a_type": "bf16_r", "b_type": "bf16_r", "scaleA": 0, "scaleB": 0},
    ]
    config = _log_config(tmp_path, rows, MX_BLOCK=32)
    got = sorted((gc.name, len(gc.sizes)) for gc in config["GemmProblems"])
    assert got == [("BBS_TN", 1), ("F8BS_MXE8B32_TN", 2)]


def test_equal_formats_merge_into_one_gemm_config() -> None:
    df = pd.DataFrame([_row("f4_r", 0), _row("f4_r", 3, m=512), _row("f4_r", 3)])
    (gc,) = resolve_mx_defaults(gemm_configs_from_gemm_dataframe(df), "gfx1250")
    assert (gc.name, gc.mx_block, gc.sizes) == ("F4BS_TN", 32, [[256, 256, 1, 256], [512, 256, 1, 256]])


def test_config_mx_keys_must_agree_with_the_log(tmp_path: Path) -> None:
    rows = [{"a_type": "f4_r", "b_type": "f4_r", "scaleA": 6, "scaleB": 6}]
    with pytest.raises(ValueError, match="asks for"):
        _log_config(tmp_path, rows, MX_BLOCK=32)


def _config(arch: str, data_type: str = "F8", **keys) -> dict:
    cfg = {
        "TRANSA": "T", "TRANSB": "N", "DataType": data_type, "DestDataType": "B",
        "ComputeDataType": "S", "ARCH": arch, "Sizes": [[256, 256, 1, 256]], **keys,
    }
    validate_input_config(cfg)
    apply_input_config_defaults(cfg)
    get_gemm_problem(cfg)
    cfg["GemmProblem"] = cfg["GemmProblems"][0]
    return cfg


@pytest.mark.parametrize(
    "arch,keys,sav,sab",
    [
        ("gfx1250", {"MX_BLOCK": 32, "LIBRARY_TYPE": "OOB"}, True, False),
        ("gfx1250", {"MX_BLOCK": 32, "LIBRARY_TYPE": "Equality"}, False, False),
        ("gfx1250", {"MX_BLOCK": 0, "LIBRARY_TYPE": "Equality"}, True, True),
        ("gfx950", {"MX_BLOCK": 32}, True, True),
        ("gfx950", {"MX": True, "LIBRARY_TYPE": "Equality"}, True, True),
    ],
)
def test_mx_epilogue_fields_follow_the_target_library(arch: str, keys: dict, sav: bool, sab: bool) -> None:
    pt = ConfigSectionGenerator(_config(arch, **keys))._problem_type
    assert ("UseScaleAlphaVec" in pt) is sav
    assert ("UseScaleAB" in pt) is sab


def test_mx_problem_type_and_global_parameters() -> None:
    nvfp4 = ConfigSectionGenerator(_config("gfx1250", "F4", MX_BLOCK=16, MX_SCALE_TYPE="F8"))
    assert (nvfp4._problem_type["MXBlockA"], nvfp4._problem_type["DataTypeMXSA"]) == (16, "F8")
    assert "MXScaleFormat" not in nvfp4._global_params_base
    assert nvfp4._bias_type_args == "[S]"

    mxfp4 = ConfigSectionGenerator(_config("gfx950", "F4"))
    assert mxfp4._problem_type["MXBlockA"] == 32
    assert "DataTypeMXSA" not in mxfp4._problem_type
    assert mxfp4._global_params_base["MXScaleFormat"] == 1

    with pytest.raises(ValueError, match="not supported on ARCH 'gfx942'"):
        ConfigSectionGenerator(_config("gfx942", "F4"))


@pytest.mark.parametrize(
    "arch,data_type,keys,expected",
    [
        ("gfx950", "F4", {}, ((32, 32), True)),
        ("gfx950", "F8", {"MX": True}, ((32, 32), True)),
        ("gfx950", "B", {"search_space": "subtile"}, (None, True)),
        ("gfx950", "B", {}, (None, False)),
        ("gfx1250", "F4", {}, (None, False)),
        ("gfx1250", "F4", {"MX_BLOCK": 16, "MX_SCALE_TYPE": "F8"}, (None, False)),
    ],
)
def test_subtile_mi_checks_only_for_gfx950(arch: str, data_type: str, keys: dict, expected: tuple) -> None:
    assert mi_design_mx_options(_config(arch, data_type, **keys)) == expected
