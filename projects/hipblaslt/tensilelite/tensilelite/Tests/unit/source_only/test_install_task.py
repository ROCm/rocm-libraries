# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import subprocess

import pytest

import tasks


pytestmark = pytest.mark.unit

_SOURCE_ROOT = Path(__file__).resolve().parents[4]


def test_invoke_install_is_a_discoverable_developer_workflow():
    result = subprocess.run(
        ["invoke", "--help", "install"],
        cwd=_SOURCE_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--build-dir" in result.stdout
    assert "--gpu-targets" in result.stdout
    assert "--rocm-path" in result.stdout


def test_install_binds_the_actual_cmake_client_output(tmp_path):
    expected = tmp_path / "build/tensilelite/client/tensilelite-client"
    assert tasks._built_client_path(tmp_path / "build") == expected
