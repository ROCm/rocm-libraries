# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The `gpu` marker policy in conftest.py, run against a throwaway test in a fresh pytest.

A GPU lane sets HIPBLASLT_PY_REQUIRE_GPU=1, so losing its device must fail the lane instead of
skipping every GPU test and passing. HIP_VISIBLE_DEVICES=-1 hides the device for the check.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

_CONFTEST = Path(__file__).with_name("conftest.py")
_PROBE = """
import pytest

@pytest.mark.gpu
def test_needs_a_gpu():
    pass
"""


def _run(tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    shutil.copy(_CONFTEST, tmp_path / "conftest.py")
    (tmp_path / "test_probe.py").write_text(_PROBE, encoding="utf-8")
    full_env = {k: v for k, v in os.environ.items() if k != "HIPBLASLT_PY_REQUIRE_GPU"}
    full_env.update(HIP_VISIBLE_DEVICES="-1", **env)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-rsf",
            "-m",
            "gpu",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=full_env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_no_device_skips_by_default(tmp_path):
    r = _run(tmp_path)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "1 skipped" in r.stdout and "no HIP device visible" in r.stdout


def test_no_device_fails_when_a_gpu_is_required(tmp_path):
    r = _run(tmp_path, HIPBLASLT_PY_REQUIRE_GPU="1")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "HIPBLASLT_PY_REQUIRE_GPU=1, but no HIP device is visible" in r.stdout
