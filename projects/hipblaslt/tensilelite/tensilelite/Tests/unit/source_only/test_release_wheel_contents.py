# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


pytestmark = pytest.mark.unit
_SOURCE_ROOT = Path(__file__).resolve().parents[4]
_VALIDATOR = _SOURCE_ROOT / "scripts/check_release_wheel_contents.py"


def test_canonical_and_compatibility_release_wheels_validate_independently(tmp_path):
    environment = dict(
        os.environ,
        TENSILELITE_ROCM_VERSION="7.2.4",
    )
    environment.pop("ROCM_PATH", None)

    for mode, source, pattern in (
        ("canonical", _SOURCE_ROOT, "tensilelite-*.whl"),
        ("compatibility", _SOURCE_ROOT / "compat", "tensilelite_tensile_compat-*.whl"),
    ):
        wheel_dir = tmp_path / mode
        wheel_dir.mkdir()
        build = subprocess.run(
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
                str(source),
            ],
            cwd=_SOURCE_ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        assert build.returncode == 0, build.stderr
        wheel = next(wheel_dir.glob(pattern))
        validation = subprocess.run(
            [
                sys.executable,
                str(_VALIDATOR),
                "--mode",
                mode,
                "--wheel",
                str(wheel),
                "--expected-version",
                "5.0.0+rocm7.2.4",
                "--source-root",
                str(_SOURCE_ROOT),
            ],
            capture_output=True,
            text=True,
        )
        assert validation.returncode == 0, validation.stderr

        with zipfile.ZipFile(wheel, "a") as archive:
            metadata = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
            archive.writestr(metadata.rsplit("/", 1)[0] + "/client.json", "/tmp/client")
        bound_validation = subprocess.run(
            [
                sys.executable,
                str(_VALIDATOR),
                "--mode",
                mode,
                "--wheel",
                str(wheel),
                "--expected-version",
                "5.0.0+rocm7.2.4",
                "--source-root",
                str(_SOURCE_ROOT),
            ],
            capture_output=True,
            text=True,
        )
        assert bound_validation.returncode != 0
        assert "wheel must not contain client bindings" in bound_validation.stderr
