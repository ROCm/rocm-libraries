# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
from pathlib import Path
import shutil
import subprocess

import pytest


pytestmark = pytest.mark.unit
_HIPBLASLT_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    ("option", "diagnostic"),
    [
        (
            "TENSILELITE_ENABLE_HOST",
            "HIPBLASLT_ENABLE_DEVICE=ON requires TENSILELITE_ENABLE_HOST=ON",
        ),
        (
            "TENSILELITE_ENABLE_CLIENT",
            "HIPBLASLT_ENABLE_DEVICE=ON requires TENSILELITE_ENABLE_CLIENT=ON",
        ),
    ],
)
def test_device_generation_rejects_disabled_prerequisite(tmp_path, option, diagnostic):
    compiler = Path(shutil.which("amdclang")).resolve()
    rocm_root = next(parent for parent in compiler.parents if (parent / ".info/version").is_file())
    result = subprocess.run(
        [
            "cmake",
            "-S",
            str(_HIPBLASLT_ROOT),
            "-B",
            str(tmp_path / option),
            "-DHIPBLASLT_ENABLE_DEVICE=ON",
            f"-D{option}=OFF",
            f"-DCMAKE_PREFIX_PATH={rocm_root}",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert diagnostic in result.stderr


def test_hipblaslt_clients_preset_selects_complete_device_graph():
    presets = json.loads((_HIPBLASLT_ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
    preset = next(item for item in presets["configurePresets"] if item["name"] == "hipblaslt-clients")
    assert preset["cacheVariables"]["HIPBLASLT_ENABLE_DEVICE"] == "ON"
    assert preset["cacheVariables"]["TENSILELITE_ENABLE_HOST"] == "ON"
    assert preset["cacheVariables"]["TENSILELITE_ENABLE_CLIENT"] == "ON"
