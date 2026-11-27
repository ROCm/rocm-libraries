# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exercise production CMake generation dependencies without GPU compilation."""

from pathlib import Path
import shutil
import subprocess
import sys
import time

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        not shutil.which("cmake") or not shutil.which("ninja"),
        reason="requires CMake and Ninja",
    ),
]
_ROOT = Path(__file__).resolve().parents[5]
_EXTOPS = _ROOT / "device-library/extops/CMakeLists.txt"


def _run(*args):
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _build(build, target):
    return _run("cmake", "--build", str(build), "--target", target)


def _runs(path):
    return path.read_text(encoding="utf-8").splitlines()


def _next_change():
    # Give even filesystems with whole-second timestamps a distinct build input.
    time.sleep(1.05)


@pytest.mark.parametrize("changed_input", ["native", "wheel"])
def test_extops_regenerate_after_native_or_wheel_change(tmp_path, changed_input):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("requires a host C++ compiler")
    native = tmp_path / "native.cpp"
    native.write_text("int fixture() { return 1; }\n", encoding="utf-8")
    stamp = tmp_path / "wheel.stamp"
    stamp.touch()
    runs = tmp_path / "runs"
    runner = tmp_path / "runner.py"
    runner.write_text(
        "import sys\nfrom pathlib import Path\n"
        "output = Path(sys.argv[sys.argv.index('-o') + 1])\n"
        "output.write_text('// generated\\n', encoding='utf-8')\n"
        f"with Path({str(runs)!r}).open('a', encoding='utf-8') as out:\n"
        "    out.write(output.name + '\\n')\n",
        encoding="utf-8",
    )
    (tmp_path / "CMakeLists.txt").write_text(
        "\n".join(
            [
                "cmake_minimum_required(VERSION 3.25)",
                "project(extops_dependencies LANGUAGES CXX ASM)",
                "add_library(_rocisa SHARED native.cpp)",
                'file(GENERATE OUTPUT "${CMAKE_BINARY_DIR}/native-path"',
                '  CONTENT "$<TARGET_FILE:_rocisa>")',
                f'include("{(_ROOT / "tensilelite/cmake/TensileLiteGenerator.cmake").as_posix()}")',
                "add_custom_target(tensilelite-generator)",
                "set_target_properties(tensilelite-generator PROPERTIES",
                f'  TENSILELITE_GENERATOR_INPUTS "{stamp.as_posix()};_rocisa"',
                f'  TENSILELITE_GENERATOR_COMMAND "{Path(sys.executable).as_posix()};{runner.as_posix()}")',
                "set(GPU_TARGETS gfx942)",
                f'set(HIPBLASLT_TENSILE_LIBPATH "{(tmp_path / "out").as_posix()}")',
                "function(tensilelite_offload_target output arch)",
                '  set(${output} "${arch}" PARENT_SCOPE)',
                "endfunction()",
                f'include("{_EXTOPS.as_posix()}")',
            ]
        ),
        encoding="utf-8",
    )
    build = tmp_path / "build"
    _run(
        "cmake",
        "-G",
        "Ninja",
        "-S",
        str(tmp_path),
        "-B",
        str(build),
        f"-DCMAKE_CXX_COMPILER={compiler}",
    )
    # Build the real production assembly commands without assembling for a GPU.
    target = "L_256_4_1_gfx942.s"
    _build(build, target)
    initial = _runs(runs)
    assert initial
    library = Path((build / "native-path").read_text(encoding="utf-8"))
    before_native = library.stat().st_mtime_ns
    _build(build, target)
    assert _runs(runs) == initial
    _next_change()
    if changed_input == "native":
        native.write_text("int fixture() { return 2; }\n", encoding="utf-8")
    else:
        stamp.touch()
    _build(build, target)
    assert _runs(runs) == initial * 2
    assert (library.stat().st_mtime_ns > before_native) == (changed_input == "native")
    _build(build, target)
    assert _runs(runs) == initial * 2
