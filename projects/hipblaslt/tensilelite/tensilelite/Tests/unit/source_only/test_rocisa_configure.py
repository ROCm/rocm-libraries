# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exercise native Python target selection and package layout without a ROCm SDK."""

from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not all(shutil.which(tool) for tool in ("cmake", "ninja", "c++")),
        reason="requires CMake, Ninja and a host C++ compiler",
    ),
]
_HIPBLASLT = Path(__file__).resolve().parents[5]


def _run(*args):
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def _configure(tmp_path, options, generator="Ninja"):
    settings = dict.fromkeys(
        (
            "HIPBLASLT_ENABLE_DEVICE",
            "HIPBLASLT_PREPARE_TENSILELITE_GENERATOR",
            "HIPBLASLT_ENABLE_CLIENT",
            "TENSILELITE_ENABLE_CLIENT",
            "HIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS",
            "ROCISA_BUILD_PYTHON",
            "ROCISA_USE_STABLE_ABI",
        ),
        "OFF",
    )
    settings.update(options)
    (tmp_path / "native.cpp").write_text("int native() { return 1; }\n", encoding="utf-8")
    # Execute the production discovery macro and the entire rocisa CMake file.
    # Only SDK/dependency discovery and native source compilation are replaced.
    # Target selection, links, package staging and output properties stay real.
    (tmp_path / "CMakeLists.txt").write_text(
        r"""
cmake_minimum_required(VERSION 3.25)
project(rocisa_configuration LANGUAGES CXX)
set(ROCM_LIBRARIES_ROOT "@REPO@")
set(CMAKE_INSTALL_LIBDIR lib)
set(TENSILELITE_DEVEL_COMPONENT devel)
set(Python3_EXECUTABLE "@PYTHON@")
set(Python_EXECUTABLE "@PYTHON@")
file(WRITE "${CMAKE_BINARY_DIR}/python-requests.txt" "")
macro(find_package package)
    if("${package}" MATCHES "^Python3?$")
        file(APPEND "${CMAKE_BINARY_DIR}/python-requests.txt" "${ARGV}\n")
    endif()
endmacro()
include("@HIPBLASLT@/cmake/hipblaslt_python.cmake")
hipblaslt_find_build_python()
configure_file("${CMAKE_BINARY_DIR}/python-requests.txt"
               "${CMAKE_BINARY_DIR}/root-python-requests.txt" COPYONLY)

add_library(roc::origami INTERFACE IMPORTED)
add_library(hip::host INTERFACE IMPORTED)
macro(include module)
    if(NOT "${module}" STREQUAL "FetchContent")
        _include(${ARGV})
    endif()
endmacro()
macro(FetchContent_Declare)
endmacro()
macro(FetchContent_MakeAvailable)
endmacro()
function(nanobind_add_module target)
    add_library(${target} MODULE "${CMAKE_SOURCE_DIR}/native.cpp")
endfunction()
function(target_sources target)
    if(NOT "${target}" STREQUAL "_rocisa")
        _target_sources(${ARGV})
    endif()
endfunction()
function(add_subdirectory source)
    if(NOT "${source}" STREQUAL "${ROCM_LIBRARIES_ROOT}/shared/stinkytofu")
        message(FATAL_ERROR "Unexpected dependency: ${source}")
    endif()
    add_library(stinkytofu STATIC "${CMAKE_SOURCE_DIR}/native.cpp")
    add_library(stinkytofu::stinkytofu ALIAS stinkytofu)
    # Like StinkyTofu itself, build its standalone Python module only on request.
    if(STINKYTOFU_BUILD_PYTHON)
        add_library(stinkytofu_python MODULE "${CMAKE_SOURCE_DIR}/native.cpp")
    endif()
endfunction()
_add_subdirectory("@HIPBLASLT@/tensilelite/rocisa" rocisa)
file(WRITE "${CMAKE_BINARY_DIR}/extensions.txt" "")
foreach(_target IN ITEMS _rocisa stinkytofu_python)
    if(TARGET ${_target})
        file(APPEND "${CMAKE_BINARY_DIR}/extensions.txt" "${_target}\n")
        file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/${_target}-$<CONFIG>.txt"
             CONTENT "$<TARGET_FILE:${_target}>")
    endif()
endforeach()
""".replace("@REPO@", _HIPBLASLT.parents[1].as_posix())
        .replace("@HIPBLASLT@", _HIPBLASLT.as_posix())
        .replace("@PYTHON@", Path(sys.executable).as_posix()),
        encoding="utf-8",
    )
    build = tmp_path / "build"
    _run(
        "cmake",
        "-G",
        generator,
        "-S",
        str(tmp_path),
        "-B",
        str(build),
        f"-DCMAKE_CXX_COMPILER={shutil.which('c++')}",
        f"-DCMAKE_MAKE_PROGRAM={shutil.which('ninja')}",
        "-DCMAKE_BUILD_TYPE=Release",
        *(f"-D{name}={value}" for name, value in settings.items()),
    )
    return build


@pytest.mark.parametrize(
    "options,minimum_version,components,extensions",
    [
        pytest.param({}, None, [], [], id="host-only"),
        pytest.param({"TENSILELITE_ENABLE_CLIENT": "ON"}, "3.10", [], [], id="tensilelite-client"),
        pytest.param({"HIPBLASLT_ENABLE_CLIENT": "ON"}, "3.10", [], [], id="hipblaslt-client"),
        pytest.param(
            {"HIPBLASLT_ENABLE_DEVICE": "ON"},
            "3.10",
            ["Development.Module"],
            ["_rocisa"],
            id="device-generation",
        ),
        pytest.param(
            {"HIPBLASLT_PREPARE_TENSILELITE_GENERATOR": "ON"},
            "3.10",
            ["Development.Module"],
            ["_rocisa"],
            id="generator-without-devices-or-tests",
        ),
        pytest.param(
            {"ROCISA_BUILD_PYTHON": "ON"},
            "3.10",
            ["Development.Module"],
            ["_rocisa"],
            id="rocisa-only",
        ),
        pytest.param(
            {"HIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS": "ON"},
            "3.10",
            ["Development.Module"],
            ["_rocisa"],
            id="test-artifacts-without-devices",
        ),
        pytest.param(
            {"HIPBLASLT_INSTALL_TENSILELITE_TEST_ARTIFACTS": "ON", "ROCISA_USE_STABLE_ABI": "ON"},
            "3.12",
            ["Development.Module", "Development.SABIModule"],
            ["_rocisa"],
            id="stable-abi-test-artifacts-without-devices",
        ),
    ],
)
def test_native_python_requirements(tmp_path, options, minimum_version, components, extensions):
    # All capabilities are off unless this case explicitly requests them.
    # In particular, installing test artifacts must not require a device build
    # or a second opt-in via ROCISA_BUILD_PYTHON.
    build = _configure(tmp_path, options)
    requests = (build / "root-python-requests.txt").read_text(encoding="utf-8").splitlines()
    expected = []
    if minimum_version:
        expected = [
            ";".join(
                [package, minimum_version, "COMPONENTS", "Interpreter", *components, "REQUIRED"]
            )
            for package in ("Python3", "Python")
        ]
    assert requests == expected
    assert (build / "extensions.txt").read_text(encoding="utf-8").splitlines() == extensions


def test_native_python_packages_stay_beside_initializers_in_all_configurations(tmp_path):
    build = _configure(tmp_path, {"ROCISA_BUILD_PYTHON": "ON"}, generator="Ninja Multi-Config")
    # Build each configuration: generated paths alone do not prove that the
    # linker emits the modules beside the staged package initializers.
    for config in ("Debug", "Release", "RelWithDebInfo"):
        _run("cmake", "--build", str(build), "--config", config, "--target", "_rocisa")
        output = Path((build / f"_rocisa-{config}.txt").read_text(encoding="utf-8"))
        assert output.parent == build / "rocisa" / "rocisa"
        assert output.is_file()
        assert (output.parent / "__init__.py").is_file()
