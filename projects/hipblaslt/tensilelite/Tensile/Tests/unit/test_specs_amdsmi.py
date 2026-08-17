# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
Unit tests for clients/scripts/performance/specs.py get_amdsmi_specs.

These verify the amd-smi (JSON) based device-spec collection that
replaced rocm-smi text scraping, without requiring real GPUs.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# specs.py lives outside the Tensile package, under the hipBLASLt client scripts.
# parents[4] == projects/hipblaslt (unit -> Tests -> Tensile -> tensilelite -> hipblaslt)
# In an installed/packaged test tree (e.g. CI's build/share/hipblaslt/...), the
# client scripts are not shipped next to the Tensile package, so specs.py is not
# importable there. Skip gracefully in that case rather than erroring at collection.
specs_dir = Path(__file__).resolve().parents[4] / "clients" / "scripts" / "performance"
if str(specs_dir) not in sys.path:
    sys.path.insert(0, str(specs_dir))

try:
    import specs  # noqa: E402
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    specs = None
    _IMPORT_ERROR = exc

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        specs is None,
        reason=f"clients/scripts/performance/specs.py not importable: {_IMPORT_ERROR}",
    ),
]


def _bytes_completed(stdout_str):
    """specs._run_amdsmi_json reads .stdout and calls .decode('ascii')."""
    return SimpleNamespace(stdout=stdout_str.encode("ascii"))


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
                        "gfx_0": {"clk": {"value": sclk, "unit": "MHz"},
                                  "max_clk": {"value": 2200, "unit": "MHz"}},
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
    assert r["vbios_version"] == "113-M350-01-1K1-030A"
    assert r["performance_level"] == "auto"
    assert r["system_clk"] == "1877Mhz"
    assert r["memory_clk"] == "1900Mhz"
    # 294896 MB expressed in bytes (downstream divides by 1024**3 to get GiB)
    assert r["vram"] == 294896 * 1024 * 1024


def test_vram_converts_to_expected_gib():
    se = _make_side_effect(_static_json(), _metric_json())
    with patch.object(specs.subprocess, "run", side_effect=se):
        r = specs.get_amdsmi_specs(0)
    gib = r["vram"] / 1024 ** 3
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
    # every amd-smi invocation should carry "-g 5"
    for call in m.call_args_list:
        cmd = call.args[0]
        assert "-g" in cmd and cmd[cmd.index("-g") + 1] == "5"


def test_all_none_when_amdsmi_unavailable():
    with patch.object(specs.subprocess, "run", side_effect=FileNotFoundError("no amd-smi")):
        r = specs.get_amdsmi_specs(0)
    assert r == {
        "vbios_version": None,
        "gpuid": None,
        "vram": None,
        "performance_level": None,
        "memory_clk": None,
        "system_clk": None,
    }
