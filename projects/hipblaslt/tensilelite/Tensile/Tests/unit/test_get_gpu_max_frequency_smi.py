# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
Unit tests for Tensile.Tensile.get_gpu_max_frequency_smi.

These verify that the amd-smi based max-GFX-clock lookup parses the
`amd-smi metric --clock --json` output correctly, without real GPUs.
"""

import json
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# Importing Tensile.Tensile pulls in the full code-generation toolchain
# (rocisa bindings, etc.). If that chain is unavailable in the current
# environment, skip rather than error at collection time.
try:
    from Tensile.Tensile import get_gpu_max_frequency_smi
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    get_gpu_max_frequency_smi = None
    _IMPORT_ERROR = exc

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        get_gpu_max_frequency_smi is None,
        reason=f"Tensile.Tensile import unavailable: {_IMPORT_ERROR}",
    ),
]


def _completed(stdout="", returncode=0):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr="")


def _clock_json(gfx_max_clocks, include_mem=True):
    """Build a stand-in for `amd-smi metric --clock --json` output.

    gfx_max_clocks: list of max_clk MHz values, one per gfx engine.
    """
    clock = {}
    for i, mhz in enumerate(gfx_max_clocks):
        clock[f"gfx_{i}"] = {
            "clk": {"value": 114, "unit": "MHz"},
            "min_clk": {"value": 500, "unit": "MHz"},
            "max_clk": {"value": mhz, "unit": "MHz"},
            "clk_locked": "DISABLED",
            "deep_sleep": "DISABLED",
        }
    if include_mem:
        clock["mem_0"] = {
            "clk": {"value": 1900, "unit": "MHz"},
            "max_clk": {"value": 1900, "unit": "MHz"},
        }
    return json.dumps({"gpu_data": [{"gpu": 0, "clock": clock}]})


def test_returns_max_gfx_clock_across_engines():
    out = _clock_json([2200, 2200, 2100, 2200])
    with patch.object(subprocess, "run", return_value=_completed(out)) as m:
        assert get_gpu_max_frequency_smi(0) == 2200
    # Confirm the command shape and device selection
    called_cmd = m.call_args_list[0].args[0]
    assert called_cmd == ["amd-smi", "metric", "-g", "0", "--clock", "--json"]


def test_device_id_is_passed_through():
    out = _clock_json([2100])
    with patch.object(subprocess, "run", return_value=_completed(out)) as m:
        assert get_gpu_max_frequency_smi(3) == 2100
    assert m.call_args_list[0].args[0][3] == "3"


def test_ignores_mem_clock():
    # mem_0 max_clk (1900) must not be chosen over gfx max_clk (2200)
    out = _clock_json([2200], include_mem=True)
    with patch.object(subprocess, "run", return_value=_completed(out)):
        assert get_gpu_max_frequency_smi(0) == 2200


def test_returns_none_on_nonzero_returncode():
    with patch.object(subprocess, "run", return_value=_completed("", returncode=1)):
        assert get_gpu_max_frequency_smi(0) is None


def test_returns_none_on_exception():
    with patch.object(subprocess, "run", side_effect=FileNotFoundError("no amd-smi")):
        assert get_gpu_max_frequency_smi(0) is None


def test_returns_none_when_no_gfx_entries():
    out = json.dumps({"gpu_data": [{"gpu": 0, "clock": {"mem_0": {"max_clk": {"value": 1900}}}}]})
    with patch.object(subprocess, "run", return_value=_completed(out)):
        assert get_gpu_max_frequency_smi(0) is None
