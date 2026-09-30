# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for amd-smi based device-spec collection in rrperf.specs."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import rrperf.specs as specs


def _bytes_completed(stdout_str, returncode=0):
    return SimpleNamespace(
        stdout=stdout_str.encode("ascii"),
        returncode=returncode,
    )


def _static_json():
    return json.dumps(
        {
            "gpu_data": [
                {
                    "gpu": 0,
                    "asic": {
                        "market_name": "AMD Instinct MI350X",
                        "device_id": "0x75a0",
                    },
                    "ifwi": {"part_number": "113-M350-01-1K1-030A"},
                }
            ]
        }
    )


def _metric_json(perf_level="AMDSMI_DEV_PERF_LEVEL_AUTO", sclk=1877, mclk=1900):
    return json.dumps(
        {
            "gpu_data": [
                {
                    "gpu": 0,
                    "mem_usage": {"total_vram": {"value": 294896, "unit": "MB"}},
                    "perf_level": perf_level,
                    "clock": {
                        "gfx_0": {
                            "clk": {"value": sclk, "unit": "MHz"},
                            "max_clk": {"value": 2200, "unit": "MHz"},
                        },
                        "mem_0": {"clk": {"value": mclk, "unit": "MHz"}},
                    },
                }
            ]
        }
    )


def _make_side_effect(static_out, metric_out):
    def side_effect(cmd, *args, **kwargs):
        if "static" in cmd:
            return _bytes_completed(static_out)
        if "metric" in cmd:
            return _bytes_completed(metric_out)
        raise AssertionError(f"unexpected command {cmd}")

    return side_effect


def test_parses_all_fields():
    se = _make_side_effect(_static_json(), _metric_json())
    with patch.object(specs.subprocess, "run", side_effect=se):
        r = specs.get_amdsmi_specs(0)
    assert r["gpuid"] == "0x75a0"
    assert r["market_name"] == "AMD Instinct MI350X"
    assert r["vbios_version"] == "113-M350-01-1K1-030A"
    assert r["performance_level"] == "auto"
    assert r["system_clk"] == "1877Mhz"
    assert r["memory_clk"] == "1900Mhz"
    assert r["vram"] == 294896 * 1024 * 1024


def test_vram_converts_to_expected_gib():
    se = _make_side_effect(_static_json(), _metric_json())
    with patch.object(specs.subprocess, "run", side_effect=se):
        r = specs.get_amdsmi_specs(0)
    gib = r["vram"] / 1024**3
    assert round(gib, 2) == 287.98


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("AMDSMI_DEV_PERF_LEVEL_AUTO", "auto"),
        ("AMDSMI_DEV_PERF_LEVEL_HIGH", "high"),
        ("AMDSMI_DEV_PERF_LEVEL_MANUAL", "manual"),
    ],
)
def test_perf_level_normalized(raw, expected):
    se = _make_side_effect(_static_json(), _metric_json(perf_level=raw))
    with patch.object(specs.subprocess, "run", side_effect=se):
        r = specs.get_amdsmi_specs(0)
    assert r["performance_level"] == expected


def test_device_index_passed_to_amdsmi():
    se = _make_side_effect(_static_json(), _metric_json())
    with patch.object(specs.subprocess, "run", side_effect=se) as m:
        specs.get_amdsmi_specs(5)
    for call in m.call_args_list:
        cmd = call.args[0]
        assert "-g" in cmd and cmd[cmd.index("-g") + 1] == "5"


def test_selects_matching_gpu_entry():
    static = json.dumps(
        {
            "gpu_data": [
                {"gpu": 0, "asic": {"device_id": "0x1111"}},
                {"gpu": 5, "asic": {"device_id": "0x5555", "market_name": "GPU5"}},
            ]
        }
    )
    metric = json.dumps({"gpu_data": [{"gpu": 5, "perf_level": "AMDSMI_DEV_PERF_LEVEL_HIGH"}]})
    se = _make_side_effect(static, metric)
    with patch.object(specs.subprocess, "run", side_effect=se):
        r = specs.get_amdsmi_specs(5)
    assert r["gpuid"] == "0x5555"
    assert r["market_name"] == "GPU5"
    assert r["performance_level"] == "high"


def test_nonzero_exit_returns_none_fields():
    def side_effect(cmd, *args, **kwargs):
        if "static" in cmd:
            return _bytes_completed(_static_json())
        if "metric" in cmd:
            return _bytes_completed("{}", returncode=1)
        raise AssertionError(f"unexpected command {cmd}")

    with patch.object(specs.subprocess, "run", side_effect=side_effect):
        r = specs.get_amdsmi_specs(0)
    assert r["gpuid"] == "0x75a0"
    assert r["vram"] is None
    assert r["performance_level"] is None


def test_all_none_when_amdsmi_unavailable():
    with patch.object(
        specs.subprocess, "run", side_effect=FileNotFoundError("no amd-smi")
    ):
        r = specs.get_amdsmi_specs(0)
    assert r == {
        "vbios_version": None,
        "gpuid": None,
        "market_name": None,
        "vram": None,
        "performance_level": None,
        "memory_clk": None,
        "system_clk": None,
    }


def test_get_machine_specs_uses_amdsmi_fields():
    se = _make_side_effect(_static_json(), _metric_json())
    with patch.object(specs.shutil, "which", return_value="/usr/bin/amd-smi"), patch.object(
        specs.subprocess, "run", side_effect=se
    ):
        m = specs.get_machine_specs(0)
    assert m.gpuid == "0x75a0"
    assert m.deviceinfo == "AMD Instinct MI350X"
    assert m.vbios == "113-M350-01-1K1-030A"
    assert m.perflevel == "auto"
    assert m.vram == "287.98 GiB"


def test_get_machine_specs_without_amdsmi():
    with patch.object(specs.shutil, "which", return_value=None):
        m = specs.get_machine_specs(0)
    assert m.gpuid == "no amd-smi"
    assert m.deviceinfo == "no amd-smi"
    assert m.vram == "0.00 GiB"


def test_parses_vbios_key_and_na_clocks():
    static = json.dumps(
        {
            "gpu_data": [
                {
                    "gpu": 0,
                    "asic": {"market_name": "AMD Instinct MI355X", "device_id": "0x75a3"},
                    "vbios": {"part_number": "113-M355-01-1K1-010C"},
                }
            ]
        }
    )
    metric = json.dumps(
        {
            "gpu_data": [
                {
                    "gpu": 0,
                    "mem_usage": {"total_vram": {"value": 294896, "unit": "MB"}},
                    "perf_level": "AMDSMI_DEV_PERF_LEVEL_AUTO",
                    "clock": {
                        "gfx_0": {"clk": "N/A", "min_clk": "N/A", "max_clk": "N/A"},
                        "mem_0": {"clk": "N/A", "min_clk": "N/A", "max_clk": "N/A"},
                    },
                }
            ]
        }
    )
    se = _make_side_effect(static, metric)
    with patch.object(specs.shutil, "which", return_value="/usr/bin/amd-smi"), patch.object(
        specs.subprocess, "run", side_effect=se
    ):
        m = specs.get_machine_specs(0)
    assert m.vbios == "113-M355-01-1K1-010C"
    assert m.mclk == ""
    assert m.sclk == ""
