# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Unit tests for the helpers in scripts/run_flatc.py.

These cover the two single-source-of-truth readers added during the
ALMIOPEN-1796 follow-up so a future rename of the underlying CMake cache
variable or shared flag file fails loudly with a useful message instead of
silently producing wrong output.
"""

import re

import pytest

import run_flatc


class TestReadRequiredVersion:
    def test_parses_version_from_real_cmakelists(self):
        """The script must be able to read the version from the live tree."""
        version = run_flatc._read_required_version()
        assert re.match(
            r"^\d+\.\d+\.\d+$", version
        ), f"Expected semver-like version, got: {version!r}"

    def test_parses_version_from_synthetic_cmakelists(self, tmp_path):
        cmake = tmp_path / "CMakeLists.txt"
        cmake.write_text(
            "project(foo)\n"
            'set(HIPDNN_FLATBUFFERS_VERSION "26.1.2" CACHE STRING "...")\n'
        )
        assert run_flatc._read_required_version(str(cmake)) == "26.1.2"

    def test_case_insensitive_match(self, tmp_path):
        cmake = tmp_path / "CMakeLists.txt"
        cmake.write_text('SET(hipdnn_flatbuffers_version "9.9.9")\n')
        assert run_flatc._read_required_version(str(cmake)) == "9.9.9"

    def test_raises_when_variable_absent(self, tmp_path):
        """Failure path: missing variable must raise with the file path so a
        future rename produces an actionable error, not a silent fallback."""
        cmake = tmp_path / "CMakeLists.txt"
        cmake.write_text('project(foo)\nset(SOMETHING_ELSE "1.2.3")\n')
        with pytest.raises(RuntimeError) as excinfo:
            run_flatc._read_required_version(str(cmake))
        assert "HIPDNN_FLATBUFFERS_VERSION" in str(excinfo.value)
        assert str(cmake) in str(excinfo.value)


class TestReadFlatcFlags:
    def test_reads_real_flag_file(self):
        flags = run_flatc._read_flatc_flags()
        assert flags, "Real flag file must contribute at least one flag"
        # Every line that survives the strip+filter must look like a flag, not
        # a stray value or comment that slipped through.
        for flag in flags:
            assert flag.startswith("-"), f"Unexpected non-flag entry: {flag!r}"
        # --cpp is the consumer's responsibility (build_flatbuffers adds it,
        # run_flatc.py adds it explicitly). Catch a regression where someone
        # promotes it into the shared file.
        assert (
            "--cpp" not in flags
        ), "--cpp must not live in the shared flag file; the consumer adds it."

    def test_strips_blank_lines_and_comments(self, tmp_path):
        flags_file = tmp_path / "flatc_flags.txt"
        flags_file.write_text(
            "# top-level comment\n"
            "\n"
            "  --gen-object-api  \n"
            "    \n"
            "# another comment\n"
            "--scoped-enums\n"
        )
        assert run_flatc._read_flatc_flags(str(flags_file)) == [
            "--gen-object-api",
            "--scoped-enums",
        ]

    def test_raises_on_empty_flag_file(self, tmp_path):
        flags_file = tmp_path / "flatc_flags.txt"
        flags_file.write_text("# only a comment\n\n")
        with pytest.raises(RuntimeError) as excinfo:
            run_flatc._read_flatc_flags(str(flags_file))
        assert str(flags_file) in str(excinfo.value)
