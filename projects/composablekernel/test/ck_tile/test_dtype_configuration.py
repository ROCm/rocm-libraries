# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Check CK Tile target filtering without a HIP compiler or GPU."""

from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import unittest


CK_ROOT = Path(__file__).resolve().parents[2]
# CTest supplies the configuring CMake; direct invocations use PATH.
CMAKE_COMMAND = os.environ.get("CMAKE_COMMAND") or shutil.which("cmake")
SUBDIRS = ("data_type", "add_rmsnorm2d_rdquant", "batched_gemm", "grouped_gemm")


class DtypeConfigurationTest(unittest.TestCase):
    def test_filtered_targets_and_dependencies(self):
        self.assertIsNotNone(CMAKE_COMMAND, "requires CMake on PATH or CMAKE_COMMAND")
        # Load the production helpers, including their dtype and architecture
        # filtering. Only installation is stubbed; no ROCm package is needed.
        helpers = re.findall(
            r"^function\(add_(?:gtest|test)_executable\b.*?^endfunction\(\)",
            (CK_ROOT / "test/CMakeLists.txt").read_text(),
            re.MULTILINE | re.DOTALL,
        )
        self.assertEqual(len(helpers), 2)
        for archs in ("gfx1030;gfx1102", "gfx90a", "gfx1102", "gfx1200"):
            for dtype in (None, "fp32", "fp16", "bf16"):
                with self.subTest(archs=archs, dtype=dtype):
                    with tempfile.TemporaryDirectory() as tmp:
                        source = Path(tmp)
                        build = source / "build"
                        (source / "helpers.cmake").write_text("\n".join(helpers))
                        subdirs = "\n".join(
                            f'add_subdirectory("{CK_ROOT / "test/ck_tile" / name}" {name})'
                            for name in SUBDIRS
                        )
                        (
                            source / "CMakeLists.txt"
                        ).write_text(f'''cmake_minimum_required(VERSION 3.21)
project(dtype_configuration LANGUAGES NONE)
enable_testing()
set(GPU_TARGETS "{archs}")
set(SUPPORTED_GPU_TARGETS "{archs}")
set(CK_USE_FNUZ_FP8 {"OFF" if archs == "gfx1200" else "ON"})
set(CK_USE_OCP_FP8 {"ON" if archs == "gfx1200" else "OFF"})
set(CK_TILE_USE_CUSTOM_DATA_TYPE ON)
set(CK_TILE_USE_LLVM_BUILTIN_BF16 ON)
function(rocm_install)
endfunction()
foreach(t tests check smoke regression)
    add_custom_target(${{t}})
endforeach()
add_library(getopt::getopt INTERFACE IMPORTED)
add_library(gtest_main INTERFACE IMPORTED)
set(CMAKE_CXX_LINK_EXECUTABLE "unused")
include("${{CMAKE_CURRENT_SOURCE_DIR}}/helpers.cmake")
{subdirs}
# Inspect actual targets and properties. Suppress compilation only after the
# production helpers and subdirectories have completed all target operations.
file(WRITE "${{CMAKE_BINARY_DIR}}/registered_tests.txt" "")
foreach(dir {" ".join(SUBDIRS)})
    get_property(registered_tests DIRECTORY "{CK_ROOT}/test/ck_tile/${{dir}}" PROPERTY TESTS)
    file(APPEND "${{CMAKE_BINARY_DIR}}/registered_tests.txt" "${{registered_tests}};")
    get_property(targets DIRECTORY "{CK_ROOT}/test/ck_tile/${{dir}}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(t IN LISTS targets)
        get_target_property(kind ${{t}} TYPE)
        if(kind STREQUAL "EXECUTABLE")
            get_target_property(sources ${{t}} SOURCES)
            foreach(src IN LISTS sources)
                get_filename_component(src "${{src}}" ABSOLUTE BASE_DIR "{CK_ROOT}/test/ck_tile/${{dir}}")
                set_source_files_properties("${{src}}" DIRECTORY "{CK_ROOT}/test/ck_tile/${{dir}}"
                    PROPERTIES HEADER_FILE_ONLY TRUE)
            endforeach()
            set_property(TARGET ${{t}} PROPERTY LINKER_LANGUAGE CXX)
        endif()
        file(GENERATE OUTPUT "${{CMAKE_BINARY_DIR}}/${{t}}.txt" CONTENT
"$<TARGET_PROPERTY:${{t}},MANUALLY_ADDED_DEPENDENCIES>\n$<TARGET_PROPERTY:${{t}},COMPILE_OPTIONS>\n$<TARGET_PROPERTY:${{t}},COMPILE_DEFINITIONS>\n")
    endforeach()
endforeach()
''')
                        command = [
                            CMAKE_COMMAND,
                            "-S",
                            str(source),
                            "-B",
                            str(build),
                            "-G",
                            "Unix Makefiles",
                            # No build is performed, so a make installation is unnecessary.
                            f"-DCMAKE_MAKE_PROGRAM={CMAKE_COMMAND}",
                        ]
                        if dtype is not None:
                            command.append(f"-DDTYPES={dtype}")
                        result = subprocess.run(
                            command, capture_output=True, text=True, timeout=60
                        )
                        self.assertEqual(
                            result.returncode, 0, result.stdout + result.stderr
                        )
                        expected = {"test_ck_tile_pk_int4"}
                        if dtype is None:
                            expected.update(("test_ck_tile_tf32", "test_ck_tile_fp8"))
                        if dtype in (None, "bf16"):
                            expected.add("test_ck_tile_bf16")
                        for selected in ("fp16", "bf16"):
                            if dtype in (None, selected):
                                name = (
                                    f"test_ck_tile_add_rmsnorm2d_rdquant_fwd_{selected}"
                                )
                                expected.add(name)
                                self.assertIn(
                                    "-Wno-undefined-func-template",
                                    (build / f"{name}.txt").read_text(),
                                )
                        for op in ("batched_gemm", "grouped_gemm"):
                            umbrella = f"test_ck_tile_{op}"
                            children = set()
                            if dtype in (None, "fp16"):
                                children.add(f"{umbrella}_f16")
                            if dtype in (None, "bf16"):
                                children.add(f"{umbrella}_bf16")
                            if dtype is None:
                                if archs in ("gfx90a", "gfx1200"):
                                    children.update(
                                        (f"{umbrella}_f8", f"{umbrella}_bf8")
                                    )
                                if archs != "gfx90a":
                                    children.add(f"{umbrella}_int8")
                            if dtype is None and archs == "gfx1200":
                                for selected in ("f8", "bf8"):
                                    self.assertIn(
                                        "-DCK_TILE_USE_OCP_FP8",
                                        (
                                            build / f"{umbrella}_{selected}.txt"
                                        ).read_text(),
                                    )
                            expected.update(children | {umbrella})
                            dependencies = (
                                (build / f"{umbrella}.txt").read_text().splitlines()[0]
                            )
                            self.assertEqual(
                                set(filter(None, dependencies.split(";"))), children
                            )
                        actual = {p.stem for p in build.glob("test_ck_tile_*.txt")}
                        self.assertEqual(actual, expected)
                        registered = (
                            (build / "registered_tests.txt").read_text().split(";")
                        )
                        self.assertEqual(
                            set(filter(None, registered)),
                            expected
                            - {
                                "test_ck_tile_batched_gemm",
                                "test_ck_tile_grouped_gemm",
                            },
                        )
                        if dtype in (None, "bf16"):
                            properties = (build / "test_ck_tile_bf16.txt").read_text()
                            for flag in (
                                "-Wno-float-equal",
                                "CK_TILE_USE_LLVM_BUILTIN_BF16=1",
                                "CK_TILE_USE_CUSTOM_DATA_TYPE=1",
                                "GTEST_HAS_RTTI=0",
                            ):
                                self.assertIn(flag, properties)
                        if dtype is None:
                            self.assertIn(
                                "-Wno-float-equal",
                                (build / "test_ck_tile_fp8.txt").read_text(),
                            )


if __name__ == "__main__":
    unittest.main()
