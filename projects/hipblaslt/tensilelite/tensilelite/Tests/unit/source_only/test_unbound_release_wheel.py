# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


pytestmark = pytest.mark.unit
_SOURCE_ROOT = Path(__file__).resolve().parents[4]


def test_configuring_installation_does_not_modify_original_wheel(tmp_path):
    wheel_dir = tmp_path / "wheelhouse"
    wheel_dir.mkdir()
    rocm_root = tmp_path / "rocm"
    (rocm_root / ".info").mkdir(parents=True)
    (rocm_root / ".info/version").write_text("7.2.4\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--disable-pip-version-check",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            str(_SOURCE_ROOT),
        ],
        cwd=_SOURCE_ROOT,
        env=dict(
            os.environ,
            ROCM_PATH=str(rocm_root),
            TENSILELITE_ROCM_VERSION="7.2.4",
        ),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    wheel = next(wheel_dir.glob("tensilelite-*.whl"))
    before = hashlib.sha256(wheel.read_bytes()).digest()
    with zipfile.ZipFile(wheel) as archive:
        assert not any(name.endswith("client.json") for name in archive.namelist())
        assert "_tensilelite_client_binding.py" in archive.namelist()
        assert "tensilelite_configure_client.py" in archive.namelist()
    assert hashlib.sha256(wheel.read_bytes()).digest() == before
