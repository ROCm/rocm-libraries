"""Unit tests for hkp_pack.toolchain hip version parsing."""

import pytest

from hkp_pack.toolchain import _read_hip_build_version, hip_version


@pytest.fixture(autouse=True)
def _clear_lru_cache():
    """Clear hip_version's lru_cache so each test gets a fresh call."""
    hip_version.cache_clear()
    yield
    hip_version.cache_clear()


class TestReadHipBuildVersion:
    def test_key_value_format(self, tmp_path):
        f = tmp_path / "version"
        f.write_text(
            "HIP_VERSION_MAJOR = 6\n"
            "HIP_VERSION_MINOR = 4\n"
            "HIP_VERSION_PATCH = 0\n"
        )
        assert _read_hip_build_version(f) == "6.4.0"

    def test_define_format(self, tmp_path):
        f = tmp_path / "hip_version.h"
        f.write_text(
            "#define HIP_VERSION_MAJOR 6\n"
            "#define HIP_VERSION_MINOR 4\n"
            "#define HIP_VERSION_PATCH 0\n"
        )
        assert _read_hip_build_version(f) == "6.4.0"

    def test_githash_suffix_key_value(self, tmp_path):
        f = tmp_path / "version"
        f.write_text(
            "HIP_VERSION_MAJOR = 6\n"
            "HIP_VERSION_MINOR = 4\n"
            "HIP_VERSION_PATCH = 0\n"
            'HIP_VERSION_GITHASH = "abc123"\n'
        )
        assert _read_hip_build_version(f) == "6.4.0-abc123"

    def test_githash_suffix_define(self, tmp_path):
        f = tmp_path / "hip_version.h"
        f.write_text(
            "#define HIP_VERSION_MAJOR 6\n"
            "#define HIP_VERSION_MINOR 4\n"
            "#define HIP_VERSION_PATCH 0\n"
            '#define HIP_VERSION_GITHASH "abc123"\n'
        )
        assert _read_hip_build_version(f) == "6.4.0-abc123"

    def test_missing_file_returns_none(self, tmp_path):
        assert _read_hip_build_version(tmp_path / "nonexistent") is None

    def test_partial_file_returns_none(self, tmp_path):
        f = tmp_path / "version"
        f.write_text("HIP_VERSION_MAJOR = 6\n" "HIP_VERSION_MINOR = 4\n")
        assert _read_hip_build_version(f) is None


class TestHipVersion:
    def test_share_hip_version_preferred(self, tmp_path):
        (tmp_path / "share" / "hip").mkdir(parents=True)
        (tmp_path / "share" / "hip" / "version").write_text(
            "HIP_VERSION_MAJOR = 6\n"
            "HIP_VERSION_MINOR = 4\n"
            "HIP_VERSION_PATCH = 0\n"
        )
        (tmp_path / "include" / "hip").mkdir(parents=True)
        (tmp_path / "include" / "hip" / "hip_version.h").write_text(
            "#define HIP_VERSION_MAJOR 5\n"
            "#define HIP_VERSION_MINOR 0\n"
            "#define HIP_VERSION_PATCH 0\n"
        )
        assert hip_version(str(tmp_path)) == "6.4.0"

    def test_falls_back_to_header(self, tmp_path):
        (tmp_path / "include" / "hip").mkdir(parents=True)
        (tmp_path / "include" / "hip" / "hip_version.h").write_text(
            "#define HIP_VERSION_MAJOR 5\n"
            "#define HIP_VERSION_MINOR 7\n"
            "#define HIP_VERSION_PATCH 1\n"
        )
        assert hip_version(str(tmp_path)) == "5.7.1"

    def test_none_hip_path_returns_none(self):
        assert hip_version(None) is None

    def test_no_version_files_returns_none(self, tmp_path):
        assert hip_version(str(tmp_path)) is None
