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
_INSTALLER = _SOURCE_ROOT / "scripts/install_release_wheels.py"
_VALIDATOR = _SOURCE_ROOT / "scripts/check_release_wheel_contents.py"


def test_canonical_and_compatibility_release_wheels_validate_independently(tmp_path):
    rocm_root = tmp_path / "rocm"
    (rocm_root / ".info").mkdir(parents=True)
    (rocm_root / ".info/version").write_text("7.2.4\n", encoding="utf-8")
    environment = dict(
        os.environ,
        ROCM_PATH=str(rocm_root),
        TENSILELITE_ROCM_VERSION="7.2.4",
    )

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
        if mode == "canonical":
            custom_kernel_root = _SOURCE_ROOT / "tensilelite/CustomKernels"
            assert any(
                path.parent != custom_kernel_root for path in custom_kernel_root.rglob("*.s")
            ), "the release-wheel check must exercise nested custom-kernel resources"
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

        installed_root = tmp_path / "installed"
        installation = subprocess.run(
            [
                sys.executable,
                str(_INSTALLER),
                "--destination",
                str(installed_root),
                str(wheel),
            ],
            capture_output=True,
            text=True,
        )
        assert installation.returncode == 0, installation.stderr
        expected_module = {
            "canonical": installed_root / "tensilelite/KernelWriterAssembly.py",
            "compatibility": installed_root / "tensilelite_tensile_compat/commands.py",
        }[mode]
        assert expected_module.is_file()
        module_name = {
            "canonical": "tensilelite.KernelWriterAssembly",
            "compatibility": "tensilelite_tensile_compat.commands",
        }[mode]
        import_check = subprocess.run(
            [
                sys.executable,
                "-c",
                "import importlib.util; "
                f"spec = importlib.util.find_spec({module_name!r}); "
                "print(spec.origin if spec else '')",
            ],
            cwd=tmp_path,
            env=dict(environment, PYTHONPATH=str(installed_root)),
            capture_output=True,
            text=True,
        )
        assert import_check.returncode == 0, import_check.stderr
        assert Path(import_check.stdout.strip()) == expected_module

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
