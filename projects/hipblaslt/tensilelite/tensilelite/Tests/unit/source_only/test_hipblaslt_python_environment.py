# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.unit

_HIPBLASLT_ROOT = Path(__file__).resolve().parents[5]
_CMAKE_HELPER = _HIPBLASLT_ROOT / "cmake" / "hipblaslt_python.cmake"
_CODEGEN_HELPER = _HIPBLASLT_ROOT / "tensilelite" / "cmake" / "hipblaslt_codegen.cmake"


def _configured_environment(
    tmp_path: Path, *, therock: bool, version: str = "10.1.0.dev0+abcdef"
) -> str:
    script = tmp_path / "environment.cmake"
    script.write_text(
        "\n".join(
            [
                f'set(HIPBLASLT_ENABLE_THEROCK {"ON" if therock else "OFF"})',
                'set(HIPBLASLT_BUILD_ROCM_ROOT "/graph/rocm")',
                f'set(HIPBLASLT_BUILD_ROCM_VERSION "{version}")',
                'set(HIPBLASLT_TENSILELITE_WHEEL_ROOT "/build/tensilelite-wheel")',
                'set(hipblaslt_SOURCE_DIR "/src/hipblaslt")',
                'set(ENV{PATH} "/graph/bin:/ambient/bin")',
                f'include("{_CMAKE_HELPER.as_posix()}")',
                "hipblaslt_tensilelite_python_environment(environment)",
                'message(STATUS "environment=${environment}")',
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["cmake", "-P", str(script)],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


def _configured_codegen_command(tmp_path: Path) -> str:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    (source_dir / "CMakeLists.txt").write_text(
        "\n".join(
            [
                "cmake_minimum_required(VERSION 3.25)",
                "project(codegen_command NONE)",
                'set(Python3_EXECUTABLE "/fallback/python")',
                'set(HIPBLASLT_PYTHON_COMMAND "configured-python" "--sentinel")',
                'set(HIPBLASLT_PYTHON_DEPS "")',
                f'include("{_CODEGEN_HELPER.as_posix()}")',
                "create_device_library(",
                f'  LOGIC_PATH "{(_HIPBLASLT_ROOT / "library").as_posix()}"',
                f'  OUTPUT_DIR "{(tmp_path / "output").as_posix()}"',
                f'  CODEGEN_ROOT "{(_HIPBLASLT_ROOT / "tensilelite").as_posix()}"',
                '  CXX_COMPILER "/compiler"',
                "  ARCHES gfx942",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        ["cmake", "-S", str(source_dir), "-B", str(tmp_path / "build")],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout + result.stderr


def _build_codegen_with_configured_dependency(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    ready_stamp = tmp_path / "python-environment-ready"
    runner = tmp_path / "configured_python.py"
    runner.write_text(
        "\n".join(
            [
                "from pathlib import Path",
                f'ready_stamp = Path(r"{ready_stamp}")',
                "if not ready_stamp.exists():",
                '    raise SystemExit("configured Python dependency was not built")',
            ]
        ),
        encoding="utf-8",
    )
    (source_dir / "CMakeLists.txt").write_text(
        "\n".join(
            [
                "cmake_minimum_required(VERSION 3.25)",
                "project(codegen_dependency NONE)",
                "add_custom_target(_rocisa)",
                "add_custom_command(",
                f'  OUTPUT "{ready_stamp.as_posix()}"',
                f'  COMMAND "${{CMAKE_COMMAND}}" -E touch "{ready_stamp.as_posix()}"',
                ")",
                "add_custom_target(configured-python-dependency",
                f'  DEPENDS "{ready_stamp.as_posix()}"',
                ")",
                f'set(HIPBLASLT_PYTHON_COMMAND "{sys.executable}" "{runner.as_posix()}")',
                'set(HIPBLASLT_PYTHON_DEPS "configured-python-dependency")',
                f'include("{_CODEGEN_HELPER.as_posix()}")',
                "create_device_library(",
                f'  LOGIC_PATH "{(_HIPBLASLT_ROOT / "library").as_posix()}"',
                f'  OUTPUT_DIR "{(tmp_path / "output").as_posix()}"',
                f'  CODEGEN_ROOT "{(_HIPBLASLT_ROOT / "tensilelite").as_posix()}"',
                '  CXX_COMPILER "/compiler"',
                "  ARCHES gfx942",
                ")",
            ]
        ),
        encoding="utf-8",
    )
    build_dir = tmp_path / "build"
    subprocess.run(
        ["cmake", "-S", str(source_dir), "-B", str(build_dir)],
        check=True,
        capture_output=True,
        text=True,
    )
    return subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "tensilelite-device-libraries"],
        capture_output=True,
        text=True,
    )


def test_therock_environment_marks_the_adapter_and_passes_a_real_version(tmp_path):
    environment = _configured_environment(tmp_path, therock=True)

    assert "TENSILELITE_ROCM_ADAPTER=therock-ci" in environment
    assert "THEROCK_PACKAGE_VERSION=10.1.0.dev0+abcdef" in environment
    assert "PATH=/graph/bin:/ambient/bin" in environment
    assert "ROCM_PATH=" not in environment
    assert "TENSILELITE_ROCM_VERSION" not in environment


def test_therock_environment_without_a_version_still_marks_the_adapter(tmp_path):
    environment = _configured_environment(tmp_path, therock=True, version="")

    assert "TENSILELITE_ROCM_ADAPTER=therock-ci" in environment
    assert "THEROCK_PACKAGE_VERSION" not in environment


def test_standalone_environment_retains_selected_rocm_path(tmp_path):
    environment = _configured_environment(tmp_path, therock=False)

    assert "ROCM_PATH=/graph/rocm" in environment
    assert "TENSILELITE_ROCM_ADAPTER" not in environment
    assert "THEROCK_PACKAGE_VERSION=" not in environment
    assert "TENSILELITE_ROCM_VERSION" not in environment


@pytest.mark.parametrize("therock", [True, False])
def test_codegen_imports_tensilelite_from_the_unpacked_wheel(tmp_path, therock):
    environment = _configured_environment(tmp_path, therock=therock)
    python_path = environment.split("PYTHONPATH=", 1)[1].split(";", 1)[0]

    entries = python_path.split(":")
    assert entries[0] == "/build/tensilelite-wheel"
    # The source tree's tensilelite package must never shadow the wheel; only
    # the adaptor, which the wheel does not ship yet, comes from the source.
    source_entries = [entry for entry in entries if entry.startswith("/src/hipblaslt")]
    assert source_entries == ["/src/hipblaslt/tensilelite/rocisa_stinkytofu_adaptor"]


def test_codegen_depends_on_the_wheel_not_on_source_globs():
    codegen = _CODEGEN_HELPER.read_text(encoding="utf-8")
    python_cmake = _CMAKE_HELPER.read_text(encoding="utf-8")

    assert "_codegen_sources" not in codegen
    assert "${HIPBLASLT_PYTHON_INPUTS}" in codegen
    assert "install_release_wheels.py" in python_cmake
    assert "pip install" not in python_cmake


def test_device_codegen_uses_the_configured_python_command(tmp_path):
    output = _configured_codegen_command(tmp_path)

    assert "configured-python --sentinel -m tensilelite create-library" in output
    assert "/fallback/python" not in output


def test_device_codegen_builds_the_configured_python_dependency(tmp_path):
    result = _build_codegen_with_configured_dependency(tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
