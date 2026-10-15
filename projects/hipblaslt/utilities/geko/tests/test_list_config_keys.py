# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Input-config keys reach ``geko --tune --list``; the environment does not set them."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from geko.cli import _rows_from_gemm_config_yaml
from geko.config_generator.constants import LIST_FORWARDED_KEYS
from geko.config_generator.load_input_config import (
    apply_input_config_defaults,
    gemm_configs_from_gemm_dataframe,
)
from geko.optim import optim
from geko.schemas import GemmConfig, GemmType

_TN = {
    "TRANSA": "T",
    "TRANSB": "N",
    "DestDataType": "B",
    "ComputeDataType": "S",
    "ARCH": "gfx1250-strict",
    "Sizes": [[512, 512, 1, 32768]],
}


def _write(tmp_path: Path, cfg: dict) -> Path:
    path = tmp_path / "list.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.mark.parametrize(
    "keys,scale,fmt,name",
    [
        ({"DataType": "F8", "MX_BLOCK": 32}, 3, (32, None), "F8BS_MXE8B32_TN"),
        ({"DataType": "F8", "MX": True}, 3, (32, None), "F8BS_MXE8B32_TN"),
        ({"DataType": "F8", "MX_BLOCK": 0}, 0, (None, None), "F8BS_TN"),
        ({"DataType": "F4"}, 3, (32, None), "F4BS_TN"),
        ({"DataType": "F4", "MX_BLOCK": 16, "MX_SCALE_TYPE": "F8"}, 6, (16, "F8"), "F4BS_MXF8B16_TN"),
        ({"DataType": "F4", "ARCH": "gfx950"}, 1001, (32, None), "F4BS_TN"),
    ],
)
def test_list_rows_carry_the_mx_format(tmp_path: Path, keys: dict, scale: int, fmt: tuple, name: str) -> None:
    cfg = {**_TN, **keys}
    rows, _ = _rows_from_gemm_config_yaml(_write(tmp_path, cfg), cfg["ARCH"])
    assert [(r["scaleA"], r["scaleB"]) for r in rows] == [(scale, scale)]

    (gc,) = gemm_configs_from_gemm_dataframe(pd.DataFrame(rows))
    assert (gc.mx_block, gc.mx_scale_type) == fmt
    assert gc.name == name


def test_list_yaml_forwards_epilogue_and_library_keys(tmp_path: Path) -> None:
    path = _write(tmp_path, {**_TN, "DataType": "F8", "MX_BLOCK": 32, "EPILOGUES": False})
    _, overrides = _rows_from_gemm_config_yaml(path, "gfx1250-strict")
    assert overrides == {"EPILOGUES": False, "LIBRARY_TYPE": "OOB"}


def test_list_yaml_mx_keys_leave_other_log_types_alone(tmp_path: Path) -> None:
    log = Path(__file__).resolve().parent / "test_data" / "workload.yaml"
    cfg = {"ARCH": "gfx1250-strict", "SIZE_OPTION": 2, "GEMM_LOG_PATH": str(log), "MX_BLOCK": 32}
    rows, overrides = _rows_from_gemm_config_yaml(_write(tmp_path, cfg), "gfx1250-strict")
    assert rows and {r["a_type"] for r in rows} == {"bf16_r"}
    assert {r["scaleA"] for r in rows} == {0}
    assert set(overrides) == set(LIST_FORWARDED_KEYS)


def test_configure_applies_overrides_below_explicit_args(tmp_path: Path, monkeypatch) -> None:
    seen = {}
    monkeypatch.setattr(optim.cg, "run", lambda config, *a, **k: seen.update(config))
    gc = GemmConfig(GemmType.from_tensile("T", "N", "F8", "B", "S"), [[512, 512, 1, 32768]])
    optim.configure(
        "/unused",
        gc,
        tmp_path,
        arch="gfx1250-strict",
        config_overrides={"LIBRARY_TYPE": "Equality", "ARCH": "gfx942"},
    )
    assert seen["LIBRARY_TYPE"] == "Equality"
    assert seen["ARCH"] == "gfx1250-strict"
    assert seen["EPILOGUES"] is True


@pytest.mark.parametrize(
    "key,value,default",
    [
        ("MX", "1", False),
        ("MX_BLOCK", "32", None),
        ("MX_SCALE_TYPE", "F8", None),
        ("EPILOGUES", "0", True),
        ("LIBRARY_TYPE", "Equality", "OOB"),
    ],
)
def test_environment_does_not_override_input_config_keys(key: str, value: str, default, monkeypatch) -> None:
    monkeypatch.setenv(key, value)
    cfg = {"ARCH": "gfx1250-strict"}
    apply_input_config_defaults(cfg)
    assert cfg[key] == default
