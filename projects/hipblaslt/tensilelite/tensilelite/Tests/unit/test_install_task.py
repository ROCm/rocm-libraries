# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import subprocess

import pytest


pytestmark = pytest.mark.unit

_SOURCE_ROOT = Path(__file__).resolve().parents[3]


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
