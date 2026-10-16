# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for rocisa's source-staleness detection (_find_stale_sources).

These tests exercise the scanning logic directly using tmp_path + os.utime to
simulate a "touched" source file, without requiring a real source tree or
mocking any internals.
"""

import os
import pytest
from pathlib import Path

from rocisa import _find_stale_sources
from rocisa import _rocisa as _rocisa_mod


@pytest.fixture()
def so_path():
    return Path(_rocisa_mod.__file__)


@pytest.fixture()
def so_mtime(so_path):
    return so_path.stat().st_mtime


def _touch(path: Path, delta: float):
    """Set path's mtime to so_mtime + delta."""
    t = path.stat().st_mtime + delta
    os.utime(path, (t, t))


class TestFindStaleSources:
    def test_empty_source_root_returns_nothing(self, tmp_path, so_path):
        assert _find_stale_sources(so_path, [tmp_path], tmp_path / "build") == []

    def test_older_file_is_not_stale(self, tmp_path, so_path, so_mtime):
        src = tmp_path / "old.cpp"
        src.write_text("// old")
        os.utime(src, (so_mtime - 100, so_mtime - 100))

        assert _find_stale_sources(so_path, [tmp_path], tmp_path / "build") == []

    def test_newer_file_is_detected(self, tmp_path, so_path, so_mtime):
        src = tmp_path / "touched.cpp"
        src.write_text("// touched")
        os.utime(src, (so_mtime + 100, so_mtime + 100))

        stale = _find_stale_sources(so_path, [tmp_path], tmp_path / "build")
        assert str(src) in stale

    def test_subsecond_edit_in_install_second_is_not_stale(self, tmp_path):
        # cmake --install writes a whole-second mtime; a source edited in that
        # same second keeps sub-second precision and must not read as stale.
        fake_so = tmp_path / "_rocisa.so"
        fake_so.write_bytes(b"")
        whole = float(int(fake_so.stat().st_mtime))
        os.utime(fake_so, (whole, whole))

        src = tmp_path / "same_second.cpp"
        src.write_text("// edited in the same second as the install")
        os.utime(src, (whole + 0.75, whole + 0.75))

        assert _find_stale_sources(fake_so, [tmp_path], tmp_path / "build") == []

    def test_edit_after_install_second_is_still_stale(self, tmp_path):
        # Rounding up must not swallow a genuine edit made a second later.
        fake_so = tmp_path / "_rocisa.so"
        fake_so.write_bytes(b"")
        whole = float(int(fake_so.stat().st_mtime))
        os.utime(fake_so, (whole, whole))

        src = tmp_path / "later.cpp"
        src.write_text("// edited well after the install")
        os.utime(src, (whole + 2, whole + 2))

        assert str(src) in _find_stale_sources(fake_so, [tmp_path], tmp_path / "build")

    def test_build_dir_files_are_excluded(self, tmp_path, so_path, so_mtime):
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        generated = build_dir / "generated.h"
        generated.write_text("// generated")
        os.utime(generated, (so_mtime + 100, so_mtime + 100))

        assert _find_stale_sources(so_path, [tmp_path], build_dir) == []

    def test_only_source_extensions_are_checked(self, tmp_path, so_path, so_mtime):
        for name in ("readme.txt", "script.py", "data.json"):
            f = tmp_path / name
            f.write_text("content")
            os.utime(f, (so_mtime + 100, so_mtime + 100))

        assert _find_stale_sources(so_path, [tmp_path], tmp_path / "build") == []

    def test_all_tracked_extensions_detected(self, tmp_path, so_path, so_mtime):
        files = {
            tmp_path / "a.cpp",
            tmp_path / "b.h",
            tmp_path / "c.def",
            tmp_path / "d.inc",
        }
        for f in files:
            f.write_text("// content")
            os.utime(f, (so_mtime + 100, so_mtime + 100))

        stale = set(_find_stale_sources(so_path, [tmp_path], tmp_path / "build"))
        assert stale == {str(f) for f in files}

    def test_multiple_roots_scanned(self, tmp_path, so_path, so_mtime):
        root_a = tmp_path / "a"
        root_b = tmp_path / "b"
        root_a.mkdir()
        root_b.mkdir()

        for root in (root_a, root_b):
            f = root / "file.cpp"
            f.write_text("// content")
            os.utime(f, (so_mtime + 100, so_mtime + 100))

        stale = _find_stale_sources(so_path, [root_a, root_b], tmp_path / "build")
        assert len(stale) == 2

    def test_nested_stale_file_detected(self, tmp_path, so_path, so_mtime):
        nested = tmp_path / "src" / "deep"
        nested.mkdir(parents=True)
        f = nested / "deep.h"
        f.write_text("// deep")
        os.utime(f, (so_mtime + 100, so_mtime + 100))

        stale = _find_stale_sources(so_path, [tmp_path], tmp_path / "build")
        assert str(f) in stale
