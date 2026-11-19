# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Exercise production CMake generation dependencies without GPU compilation."""

import os
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
_CODEGEN = _ROOT / "tensilelite/cmake/hipblaslt_codegen.cmake"
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


def _generator_package(root, runs, identity):
    package = root / "tensilelite"
    (package / "logic").mkdir(parents=True)
    (package / "create_library").mkdir()
    for relative in ("__init__.py", "cli.py", "logic/run.py", "create_library/run.py"):
        (package / relative).write_text("", encoding="utf-8")
    (package / "logic/known_bugs.yaml").write_text("{}\n", encoding="utf-8")
    (package / "__main__.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        f"with Path({str(runs)!r}).open('a', encoding='utf-8') as out:\n"
        f"    out.write({identity!r} + ':' + sys.argv[1] + '\\n')\n",
        encoding="utf-8",
    )
    (package / "header.h").write_text("// fixture\n", encoding="utf-8")
    (package / "kernel.s").write_text("// fixture\n", encoding="utf-8")
    return package


@pytest.mark.parametrize(
    "command_mode",
    ["source", "source_override", "source_root_override", "interpreter_override", "wheel"],
)
def test_codegen_tracks_inputs_used_by_selected_command(tmp_path, command_mode):
    runs = tmp_path / "runs"
    source = tmp_path / "source"
    package = _generator_package(source, runs, "source")
    wheel = tmp_path / "wheel"
    _generator_package(wheel, runs, "wheel")
    stamp = tmp_path / "wheel.stamp"
    stamp.touch()
    logic = tmp_path / "logic"
    logic.mkdir()
    (logic / "fixture.yaml").write_text("{}\n", encoding="utf-8")
    configuration = []
    if command_mode != "source":
        configuration = [
            "set(HIPBLASLT_PYTHON_COMMAND",
            '  "${CMAKE_COMMAND}" -E env',
            f'  "PYTHONPATH={wheel.as_posix()}" "{Path(sys.executable).as_posix()}")',
            f'set(HIPBLASLT_PYTHON_INPUTS "{stamp.as_posix()}")',
        ]
    overrides = []
    if command_mode in ("source", "source_override", "source_root_override"):
        overrides.append(f'CODEGEN_ROOT "{source.as_posix()}"')
    if command_mode in ("source", "source_override", "interpreter_override"):
        overrides.append(f'PYTHON_EXECUTABLE "{Path(sys.executable).as_posix()}"')
    (tmp_path / "CMakeLists.txt").write_text(
        "\n".join(
            [
                "cmake_minimum_required(VERSION 3.25)",
                "project(generator_dependencies NONE)",
                *configuration,
                # The normal hipBLASLt wheel caller inherits its source root;
                # only a per-call root explicitly selects source generation.
                f'set(HIPBLASLT_CODEGEN_ROOT "{source.as_posix()}")',
                f'include("{_CODEGEN.as_posix()}")',
                "create_device_library(",
                f'  LOGIC_PATH "{logic.as_posix()}"',
                f'  OUTPUT_DIR "{(tmp_path / "out").as_posix()}"',
                *overrides,
                '  CXX_COMPILER "/compiler" ARCHES gfx942)',
            ]
        ),
        encoding="utf-8",
    )
    build = tmp_path / "build"
    _run("cmake", "-G", "Ninja", "-S", str(tmp_path), "-B", str(build))
    target = "tensilelite-device-libraries"
    _build(build, target)
    identity = "wheel" if command_mode == "wheel" else "source"
    commands = [f"{identity}:logic", f"{identity}:create-library"]
    expected = commands.copy()
    assert _runs(runs) == expected
    _build(build, target)
    assert _runs(runs) == expected

    # Python and both types of packaged generator resource invalidate source
    # consumers, including a root or interpreter overriding a configured wheel.
    for relative in ("__main__.py", "header.h", "kernel.s"):
        _next_change()
        path = package / relative
        with path.open("a", encoding="utf-8") as out:
            out.write("# changed\n" if path.suffix == ".py" else "// changed\n")
        _build(build, target)
        if command_mode != "wheel":
            expected += commands
        assert _runs(runs) == expected

    # Adding an old resource and removing it must invalidate membership even
    # though neither action leaves a file newer than the previous output stamp.
    _next_change()
    added = package / "added.s"
    added.write_text("// added\n", encoding="utf-8")
    os.utime(added, (1, 1))
    _build(build, target)
    if command_mode != "wheel":
        expected += commands
    assert _runs(runs) == expected
    _next_change()
    added.unlink()
    _build(build, target)
    if command_mode != "wheel":
        expected += commands
    assert _runs(runs) == expected

    _next_change()
    stamp.touch()
    _build(build, target)
    if command_mode == "wheel":
        expected += commands
    assert _runs(runs) == expected
    _build(build, target)
    assert _runs(runs) == expected


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
                "add_custom_target(tensilelite-python-build-environment",
                f'  DEPENDS _rocisa "{stamp.as_posix()}")',
                "set(HIPBLASLT_PYTHON_DEPS tensilelite-python-build-environment)",
                f'set(HIPBLASLT_PYTHON_INPUTS "{stamp.as_posix()}")',
                "set(HIPBLASLT_PYTHON_COMMAND",
                f'  "{Path(sys.executable).as_posix()}" "{runner.as_posix()}")',
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
