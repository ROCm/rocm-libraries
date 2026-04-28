# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Regression coverage for hipdnn_flatbuffers_version_check.cmake.

The helper is the single-sourced version validation invoked by both the
in-tree flatbuffers_sdk/CMakeLists.txt and the installed imported Config
template (hipdnn_flatbuffers_sdkConfig_imported.cmake.in). These tests
drive it via `cmake -P` script mode so the comparison logic is covered
without needing a full configure of the project.
"""

import os
import shutil
import subprocess
import textwrap

import pytest


_HIPDNN_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_HELPER = os.path.join(
    _HIPDNN_DIR,
    "flatbuffers_sdk",
    "cmake",
    "hipdnn_flatbuffers_version_check.cmake",
)


def _cmake_or_skip():
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake not on PATH")
    return cmake


def _run_helper(tmp_path, snippet):
    """Write a small driver script that includes the helper, sets the
    ambient flatbuffers_VERSION/flatbuffers_FOUND state, and invokes
    hipdnn_check_flatbuffers_version. Returns (returncode, combined_output).
    """
    cmake = _cmake_or_skip()
    script = tmp_path / "driver.cmake"
    script.write_text(f'include("{_HELPER}")\n' + textwrap.dedent(snippet))
    proc = subprocess.run(
        [cmake, "-P", str(script)],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_helper_module_exists():
    assert os.path.isfile(_HELPER), f"Helper not found at {_HELPER}"


class TestVersionCheck:
    def test_silent_on_exact_match(self, tmp_path):
        rc, output = _run_helper(
            tmp_path,
            """
            set(flatbuffers_VERSION "25.9.23")
            set(flatbuffers_FOUND TRUE)
            hipdnn_check_flatbuffers_version(
                EXPECTED "25.9.23"
                CONTEXT  "test"
            )
            """,
        )
        assert rc == 0, output
        assert "FATAL" not in output
        assert "CMake Warning" not in output

    def test_silent_when_neither_version_nor_found(self, tmp_path):
        # The FetchContent path: find_package was never called, so the
        # version is guaranteed by the pinned git tag.
        rc, output = _run_helper(
            tmp_path,
            """
            unset(flatbuffers_VERSION)
            unset(flatbuffers_FOUND)
            hipdnn_check_flatbuffers_version(
                EXPECTED "25.9.23"
                CONTEXT  "test"
            )
            """,
        )
        assert rc == 0, output
        assert "CMake Warning" not in output

    def test_fatal_on_version_mismatch(self, tmp_path):
        rc, output = _run_helper(
            tmp_path,
            """
            set(flatbuffers_VERSION "24.12.23")
            set(flatbuffers_FOUND TRUE)
            set(flatbuffers_DIR "/fake/path")
            hipdnn_check_flatbuffers_version(
                EXPECTED "25.9.23"
                CONTEXT  "test ctx"
                DETAIL   "extra remediation hint"
            )
            """,
        )
        assert rc != 0, output
        assert "FlatBuffers version mismatch" in output
        assert "24.12.23" in output
        assert "25.9.23" in output
        assert "test ctx" in output
        # DETAIL must propagate so the call site can offer remediation.
        assert "extra remediation hint" in output
        # Surfacing flatbuffers_DIR helps the user locate the wrong package.
        assert "/fake/path" in output

    def test_warning_when_found_but_version_unset(self, tmp_path):
        rc, output = _run_helper(
            tmp_path,
            """
            unset(flatbuffers_VERSION)
            set(flatbuffers_FOUND TRUE)
            hipdnn_check_flatbuffers_version(
                EXPECTED "25.9.23"
                CONTEXT  "warn ctx"
            )
            """,
        )
        assert rc == 0, output
        assert "CMake Warning" in output
        assert "warn ctx" in output
        assert "could not validate FlatBuffers version" in output

    def test_missing_required_arguments_is_fatal(self, tmp_path):
        rc, output = _run_helper(
            tmp_path,
            """
            hipdnn_check_flatbuffers_version(CONTEXT "no expected")
            """,
        )
        assert rc != 0, output
        assert "EXPECTED" in output
