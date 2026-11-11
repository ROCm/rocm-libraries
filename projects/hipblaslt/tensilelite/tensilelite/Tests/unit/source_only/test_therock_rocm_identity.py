# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""hipblaslt_resolve_build_rocm_version(): which builds stamp a ROCm identity."""

from pathlib import Path
import shutil
import subprocess

import pytest


pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(shutil.which("cmake") is None, reason="needs cmake"),
]

_TENSILELITE_ROOT = Path(__file__).resolve().parents[4]
_PYTHON_CMAKE = _TENSILELITE_ROOT.parent / "cmake" / "hipblaslt_python.cmake"


def _resolve_version(tmp_path, *, therock=True, package_version=None):
    lines = [f"set(HIPBLASLT_ENABLE_THEROCK {'ON' if therock else 'OFF'})"]
    if package_version is not None:
        lines.append(f'set(THEROCK_PACKAGE_VERSION "{package_version}")')
    lines += [
        f'include("{_PYTHON_CMAKE.as_posix()}")',
        "hipblaslt_resolve_build_rocm_version(resolved)",
        'message(STATUS "resolved=[${resolved}]")',
    ]
    script = tmp_path / "resolve.cmake"
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = subprocess.run(["cmake", "-P", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "WARNING" not in result.stderr
    return result.stdout.split("resolved=[", 1)[1].split("]", 1)[0]


@pytest.mark.parametrize(
    "package_version",
    ["7.12.0", "7.12.0rc1", "7.12.0a20261110", "7.12.0.dev0+" + "a" * 40 + ".asan"],
)
def test_therock_forwards_its_package_version(tmp_path, package_version):
    assert _resolve_version(tmp_path, package_version=package_version) == package_version


@pytest.mark.parametrize("package_version", [None, "", "git"])
def test_therock_without_a_real_version_is_a_development_build(tmp_path, package_version):
    """TheRock main passes nothing and local TheRock builds pass "git"."""
    assert _resolve_version(tmp_path, package_version=package_version) == ""


@pytest.mark.parametrize("package_version", [None, "7.12.0"])
def test_standalone_builds_never_stamp_an_identity(tmp_path, package_version):
    assert _resolve_version(tmp_path, therock=False, package_version=package_version) == ""
