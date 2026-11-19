# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Configure optional client targets and execute their production RPATH commands."""

import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(sys.platform != "linux", reason="RPATH enforcement is Linux-specific"),
    pytest.mark.skipif(
        not shutil.which("cmake") or not shutil.which("c++"),
        reason="requires CMake and a host C++ compiler",
    ),
]
_CLIENT = Path(__file__).resolve().parents[5] / "tensilelite/client/CMakeLists.txt"


def _run(*args):
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "tensile_testing,build_testing", [(False, False), (True, False), (False, True)]
)
def test_default_rpath_setup_handles_optional_cpu_driver(tmp_path, tensile_testing, build_testing):
    calls = tmp_path / "patchelf-calls.jsonl"
    patchelf = tmp_path / "patchelf"
    patchelf.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\nfrom pathlib import Path\n"
        f"with Path({str(calls)!r}).open('a', encoding='utf-8') as out:\n"
        "    out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "if sys.argv[1] == '--print-rpath':\n"
        "    print('/fixture/lib')\n",
        encoding="utf-8",
    )
    patchelf.chmod(0o755)
    for source in ("main.cpp", "cpu_gemm_driver.cpp", "common.cpp"):
        (tmp_path / source).write_text("int main() { return 0; }\n", encoding="utf-8")

    # Execute the entire production file. Replace SDK discovery and the client
    # source subdirectories with a tiny host fixture; target creation and the
    # default-enabled POST_BUILD setup must still configure, generate and run.
    (tmp_path / "CMakeLists.txt").write_text(
        "\n".join(
            [
                "cmake_minimum_required(VERSION 3.25)",
                "project(client_rpath LANGUAGES CXX)",
                "macro(find_package)",
                "endmacro()",
                "macro(add_subdirectory)",
                "endmacro()",
                "function(hipblaslt_attach_tensilelite_metadata)",
                "endfunction()",
                f'set(TENSILELITE_BUILD_TESTING {"ON" if tensile_testing else "OFF"})',
                f'set(BUILD_TESTING {"ON" if build_testing else "OFF"})',
                f'set(TENSILELITE_PATCHELF "{patchelf.as_posix()}" CACHE FILEPATH "")',
                "foreach(dep IN ITEMS hip::device hip::host roc::tensilelite-host",
                "    rocisa::rocisa-cpp OpenMP::OpenMP_CXX)",
                "  add_library(${dep} INTERFACE IMPORTED)",
                "endforeach()",
                f'include("{_CLIENT.as_posix()}")',
                "target_sources(tensilelite-client-common PRIVATE common.cpp)",
            ]
        ),
        encoding="utf-8",
    )
    build = tmp_path / "build"
    _run(
        "cmake",
        "-S",
        str(tmp_path),
        "-B",
        str(build),
        f"-DCMAKE_CXX_COMPILER={shutil.which('c++')}",
    )
    _run("cmake", "--build", str(build))
    recorded = [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]
    expected_targets = ["tensilelite-client"]
    if tensile_testing or build_testing:
        expected_targets.append("cpu-gemm-driver")
    expected = []
    for target in expected_targets:
        binary = str(build / target)
        expected.extend(
            [
                ["--print-rpath", binary],
                ["--force-rpath", "--set-rpath", "/fixture/lib", binary],
            ]
        )
    assert sorted(recorded) == sorted(expected)
