################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################

import pytest

pytestmark = pytest.mark.unit

import hashlib
import os
import shutil
from collections import namedtuple
from pathlib import Path

MockVersion = namedtuple("MockVersion", ["major", "minor", "patch"])


class MockCompiler:
    def __init__(self, version=(6, 0, 0), rocm_version=(6, 0, 0), asan=False):
        self.version = MockVersion(*version)
        self.rocm_version = MockVersion(*rocm_version)
        self.default_args = ["amdclang++", "-O3"]
        if asan:
            self.default_args.append("-fsanitize=address")


def _write_test_files(tmp_path, cpp_content="void f(){}", h_content="#pragma once"):
    """Create minimal source + header files for cache key tests."""
    (tmp_path / "Kernels.cpp").write_text(cpp_content)
    (tmp_path / "Kernels.h").write_text(h_content)
    for name in [
        "KernelHeader.h", "TensileTypes.h", "tensile_bfloat16.h",
        "tensile_float8_bfloat8.h", "ReductionTemplate.h", "memory_gfx.h",
    ]:
        (tmp_path / name).write_text(f"// {name}")
    return tmp_path / "Kernels.cpp"


class TestComputeCacheKey:
    def test_deterministic(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        compiler = MockCompiler()
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        assert k1 == k2
        assert len(k1) == 64  # sha256 hex digest

    def test_different_source_different_key(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _computeCacheKey
        kernel_path = _write_test_files(tmp_path, cpp_content="void f(){}")
        compiler = MockCompiler()
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        (tmp_path / "Kernels.cpp").write_text("void g(){}")
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        assert k1 != k2

    def test_different_arch_different_key(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        compiler = MockCompiler()
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx1100"], compiler)
        assert k1 != k2

    def test_arch_order_irrelevant(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        compiler = MockCompiler()
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942", "gfx1100"], compiler)
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx1100", "gfx942"], compiler)
        assert k1 == k2

    def test_different_compiler_version_different_key(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], MockCompiler(version=(6, 0, 0)))
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], MockCompiler(version=(6, 1, 0)))
        assert k1 != k2

    def test_different_rocm_version_different_key(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], MockCompiler(rocm_version=(6, 0, 0)))
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], MockCompiler(rocm_version=(6, 1, 0)))
        assert k1 != k2

    def test_asan_changes_key(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], MockCompiler(asan=False))
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], MockCompiler(asan=True))
        assert k1 != k2

    def test_static_header_change_changes_key(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        compiler = MockCompiler()
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        (tmp_path / "TensileTypes.h").write_text("// modified")
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        assert k1 != k2


class TestCheckCache:
    def test_returns_none_when_dir_missing(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _checkCache
        assert _checkCache(tmp_path, "nonexistent_hash") is None

    def test_returns_none_when_dir_empty(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _checkCache
        (tmp_path / "some_hash").mkdir()
        assert _checkCache(tmp_path, "some_hash") is None

    def test_returns_none_when_file_zero_size(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _checkCache
        entry = tmp_path / "some_hash" / "gfx942"
        entry.mkdir(parents=True)
        (entry / "Kernels.so-000-gfx942.hsaco").write_bytes(b"")
        assert _checkCache(tmp_path, "some_hash") is None

    def test_returns_files_on_valid_entry(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _checkCache
        entry = tmp_path / "some_hash" / "gfx942"
        entry.mkdir(parents=True)
        (entry / "Kernels.so-000-gfx942.hsaco").write_bytes(b"\x7fELF")
        (entry / "Kernels.so-000-gfx942-xnack+.hsaco").write_bytes(b"\x7fELF")
        result = _checkCache(tmp_path, "some_hash")
        assert result is not None
        assert len(result) == 2
        assert all(f.suffix == ".hsaco" for f in result)

    def test_ignores_non_hsaco_files(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _checkCache
        entry = tmp_path / "some_hash" / "gfx942"
        entry.mkdir(parents=True)
        (entry / "Kernels.so-000-gfx942.hsaco").write_bytes(b"\x7fELF")
        (entry / "metadata.json").write_text("{}")
        result = _checkCache(tmp_path, "some_hash")
        assert len(result) == 1


class TestPopulateCache:
    def test_populates_empty_cache(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _populateCache
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Source hsacoFiles live under <root>/<base-arch>/; _populateCache mirrors
        # that parent name into the cache layout.
        src_dir = tmp_path / "src" / "gfx942"
        src_dir.mkdir(parents=True)
        f1 = src_dir / "Kernels.so-000-gfx942.hsaco"
        f1.write_bytes(b"\x7fELF_data_1")
        _populateCache(cache_dir, "abc123", [f1])
        cached = cache_dir / "abc123" / "gfx942" / "Kernels.so-000-gfx942.hsaco"
        assert cached.exists()
        assert cached.read_bytes() == b"\x7fELF_data_1"

    def test_skips_when_entry_exists(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _populateCache
        cache_dir = tmp_path / "cache"
        entry = cache_dir / "abc123" / "gfx942"
        entry.mkdir(parents=True)
        (entry / "Kernels.so-000-gfx942.hsaco").write_bytes(b"original")
        src_dir = tmp_path / "src" / "gfx942"
        src_dir.mkdir(parents=True)
        src = src_dir / "new.hsaco"
        src.write_bytes(b"different")
        _populateCache(cache_dir, "abc123", [src])
        assert (entry / "Kernels.so-000-gfx942.hsaco").read_bytes() == b"original"

    def test_cleans_up_tmp_on_race(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _populateCache
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Pre-create the final dir to simulate a race
        (cache_dir / "abc123" / "gfx942").mkdir(parents=True)
        (cache_dir / "abc123" / "gfx942" / "f.hsaco").write_bytes(b"winner")
        src_dir = tmp_path / "src" / "gfx942"
        src_dir.mkdir(parents=True)
        src = src_dir / "f.hsaco"
        src.write_bytes(b"loser")
        _populateCache(cache_dir, "abc123", [src])
        # No leftover tmp dirs
        tmp_dirs = list(cache_dir.glob(".tmp_*"))
        assert len(tmp_dirs) == 0

    def test_creates_cache_dir_if_missing(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _populateCache
        cache_dir = tmp_path / "cache" / "subdir"
        src_dir = tmp_path / "src" / "gfx942"
        src_dir.mkdir(parents=True)
        src = src_dir / "f.hsaco"
        src.write_bytes(b"\x7fELF")
        _populateCache(cache_dir, "abc123", [src])
        assert (cache_dir / "abc123" / "gfx942" / "f.hsaco").exists()


class TestEvictStale:
    def test_removes_old_entries(self, tmp_path):
        import time
        from Tensile.Toolchain.HelperKernelCache import _evictStale
        cache_dir = tmp_path / "cache"
        old_entry = cache_dir / "old_hash"
        old_entry.mkdir(parents=True)
        (old_entry / "f.hsaco").write_bytes(b"\x7fELF")
        # Backdate mtime by 31 days
        old_time = time.time() - 31 * 24 * 60 * 60
        os.utime(old_entry, (old_time, old_time))

        _evictStale(cache_dir, 30)
        assert not old_entry.exists()

    def test_keeps_recent_entries(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _evictStale
        cache_dir = tmp_path / "cache"
        recent = cache_dir / "recent_hash"
        recent.mkdir(parents=True)
        (recent / "f.hsaco").write_bytes(b"\x7fELF")
        # mtime is now, well within 30 days

        _evictStale(cache_dir, 30)
        assert recent.exists()

    def test_skips_tmp_dirs(self, tmp_path):
        import time
        from Tensile.Toolchain.HelperKernelCache import _evictStale
        cache_dir = tmp_path / "cache"
        tmp_dir = cache_dir / ".tmp_abc_1234"
        tmp_dir.mkdir(parents=True)
        old_time = time.time() - 31 * 24 * 60 * 60
        os.utime(tmp_dir, (old_time, old_time))

        _evictStale(cache_dir, 30)
        assert tmp_dir.exists()

    def test_noop_when_cache_dir_missing(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _evictStale
        # Should not raise
        _evictStale(tmp_path / "nonexistent", 30)

    def test_mixed_old_and_recent(self, tmp_path):
        import time
        from Tensile.Toolchain.HelperKernelCache import _evictStale
        cache_dir = tmp_path / "cache"
        old = cache_dir / "old_hash"
        old.mkdir(parents=True)
        (old / "f.hsaco").write_bytes(b"\x7fELF")
        old_time = time.time() - 31 * 24 * 60 * 60
        os.utime(old, (old_time, old_time))

        recent = cache_dir / "recent_hash"
        recent.mkdir(parents=True)
        (recent / "f.hsaco").write_bytes(b"\x7fELF")

        _evictStale(cache_dir, 30)
        assert not old.exists()
        assert recent.exists()


class TestRestoreRobustness:
    def test_restore_updates_mtime(self, tmp_path, monkeypatch):
        """Cache hit should touch mtime so the entry is not evicted."""
        import time
        from Tensile.Toolchain.HelperKernelCache import HelperKernelCache, _computeCacheKey
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TENSILE_HELPER_CACHE_DIR", str(cache_dir))
        monkeypatch.delenv("TENSILE_DISABLE_HELPER_CACHE", raising=False)

        kernel_path = _write_test_files(tmp_path)
        compiler = MockCompiler()
        key = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)

        entry = cache_dir / key
        archDir = entry / "gfx942"
        archDir.mkdir(parents=True)
        (archDir / "Kernels.so-000-gfx942.hsaco").write_bytes(b"\x7fELF")
        old_time = time.time() - 20 * 24 * 60 * 60
        os.utime(entry, (old_time, old_time))

        dest = tmp_path / "dest"
        dest.mkdir()
        cache = HelperKernelCache()
        hit, coPaths = cache.restore(kernel_path, tmp_path, ["gfx942"], compiler, dest)
        assert hit
        # mtime should be refreshed to now, not 20 days ago
        assert time.time() - entry.stat().st_mtime < 60

    def test_restore_recovers_from_deleted_entry(self, tmp_path, monkeypatch):
        """If cache entry is deleted mid-copy, restore returns a miss."""
        from Tensile.Toolchain.HelperKernelCache import HelperKernelCache, _computeCacheKey, _checkCache
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TENSILE_HELPER_CACHE_DIR", str(cache_dir))
        monkeypatch.delenv("TENSILE_DISABLE_HELPER_CACHE", raising=False)

        kernel_path = _write_test_files(tmp_path)
        compiler = MockCompiler()
        key = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)

        entry = cache_dir / key
        archDir = entry / "gfx942"
        archDir.mkdir(parents=True)
        (archDir / "Kernels.so-000-gfx942.hsaco").write_bytes(b"\x7fELF")

        dest = tmp_path / "dest"
        dest.mkdir()

        # Simulate deletion by making the source file unreadable after _checkCache
        original_copy2 = shutil.copy2
        def failing_copy2(src, dst):
            raise OSError("file deleted")
        monkeypatch.setattr(shutil, "copy2", failing_copy2)

        cache = HelperKernelCache()
        hit, coPaths = cache.restore(kernel_path, tmp_path, ["gfx942"], compiler, dest)
        assert not hit
        assert coPaths == []

    def test_restore_cleans_partial_copies(self, tmp_path, monkeypatch):
        """If copy fails mid-loop, already-copied files are cleaned up."""
        from Tensile.Toolchain.HelperKernelCache import HelperKernelCache, _computeCacheKey
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TENSILE_HELPER_CACHE_DIR", str(cache_dir))
        monkeypatch.delenv("TENSILE_DISABLE_HELPER_CACHE", raising=False)

        kernel_path = _write_test_files(tmp_path)
        compiler = MockCompiler()
        key = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)

        entry = cache_dir / key
        (entry / "gfx942").mkdir(parents=True)
        (entry / "gfx942" / "Kernels.so-000-gfx942.hsaco").write_bytes(b"\x7fELF")
        (entry / "gfx1100").mkdir(parents=True)
        (entry / "gfx1100" / "Kernels.so-000-gfx1100.hsaco").write_bytes(b"\x7fELF")

        dest = tmp_path / "dest"
        dest.mkdir()

        # Let the first copy succeed, fail on the second
        call_count = 0
        original_copy2 = shutil.copy2
        def copy2_fail_second(src, dst):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OSError("deleted mid-copy")
            return original_copy2(src, dst)
        monkeypatch.setattr(shutil, "copy2", copy2_fail_second)

        cache = HelperKernelCache()
        hit, coPaths = cache.restore(kernel_path, tmp_path, ["gfx942"], compiler, dest)
        assert not hit
        # No leftover partial files in dest
        assert list(dest.rglob("*.hsaco")) == []


class TestBuildSourceCodeObjectFilesCache:
    """Test cache integration via the env var and file-system side effects."""

    def test_cache_miss_creates_entry(self, tmp_path, monkeypatch):
        """On cache miss, after compilation, cache dir should be populated."""
        from Tensile.Toolchain.HelperKernelCache import _computeCacheKey, _checkCache
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TENSILE_HELPER_CACHE_DIR", str(cache_dir))
        monkeypatch.delenv("TENSILE_DISABLE_HELPER_CACHE", raising=False)

        # Set up source files
        (tmp_path / "output").mkdir()
        kernel_path = _write_test_files(tmp_path / "output")
        output = tmp_path / "output"
        compiler = MockCompiler()
        key = _computeCacheKey(kernel_path, output, ["gfx942"], compiler)

        # Before build, cache is empty
        assert _checkCache(cache_dir, key) is None

    def test_cache_disabled_skips_cache(self, tmp_path, monkeypatch):
        """When TENSILE_DISABLE_HELPER_CACHE=1, no cache dir should be created."""
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TENSILE_HELPER_CACHE_DIR", str(cache_dir))
        monkeypatch.setenv("TENSILE_DISABLE_HELPER_CACHE", "1")

        # Cache dir should not be created when disabled
        assert not cache_dir.exists()

    def test_cache_hit_copies_files(self, tmp_path):
        """Pre-populate cache, verify _checkCache finds it."""
        from Tensile.Toolchain.HelperKernelCache import _checkCache
        cache_dir = tmp_path / "cache"
        entry = cache_dir / "test_key" / "gfx942"
        entry.mkdir(parents=True)
        (entry / "Kernels.so-000-gfx942.hsaco").write_bytes(b"\x7fELF")
        result = _checkCache(cache_dir, "test_key")
        assert result is not None
        assert len(result) == 1
