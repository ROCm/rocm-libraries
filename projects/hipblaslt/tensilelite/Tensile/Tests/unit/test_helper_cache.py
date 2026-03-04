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
        from Tensile.Toolchain.Source import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        compiler = MockCompiler()
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        assert k1 == k2
        assert len(k1) == 64  # sha256 hex digest

    def test_different_source_different_key(self, tmp_path):
        from Tensile.Toolchain.Source import _computeCacheKey
        kernel_path = _write_test_files(tmp_path, cpp_content="void f(){}")
        compiler = MockCompiler()
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        (tmp_path / "Kernels.cpp").write_text("void g(){}")
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        assert k1 != k2

    def test_different_arch_different_key(self, tmp_path):
        from Tensile.Toolchain.Source import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        compiler = MockCompiler()
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx1100"], compiler)
        assert k1 != k2

    def test_arch_order_irrelevant(self, tmp_path):
        from Tensile.Toolchain.Source import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        compiler = MockCompiler()
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942", "gfx1100"], compiler)
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx1100", "gfx942"], compiler)
        assert k1 == k2

    def test_different_compiler_version_different_key(self, tmp_path):
        from Tensile.Toolchain.Source import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], MockCompiler(version=(6, 0, 0)))
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], MockCompiler(version=(6, 1, 0)))
        assert k1 != k2

    def test_different_rocm_version_different_key(self, tmp_path):
        from Tensile.Toolchain.Source import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], MockCompiler(rocm_version=(6, 0, 0)))
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], MockCompiler(rocm_version=(6, 1, 0)))
        assert k1 != k2

    def test_asan_changes_key(self, tmp_path):
        from Tensile.Toolchain.Source import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], MockCompiler(asan=False))
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], MockCompiler(asan=True))
        assert k1 != k2

    def test_static_header_change_changes_key(self, tmp_path):
        from Tensile.Toolchain.Source import _computeCacheKey
        kernel_path = _write_test_files(tmp_path)
        compiler = MockCompiler()
        k1 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        (tmp_path / "TensileTypes.h").write_text("// modified")
        k2 = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)
        assert k1 != k2


class TestCheckCache:
    def test_returns_none_when_dir_missing(self, tmp_path):
        from Tensile.Toolchain.Source import _checkCache
        assert _checkCache(tmp_path, "nonexistent_hash") is None

    def test_returns_none_when_dir_empty(self, tmp_path):
        from Tensile.Toolchain.Source import _checkCache
        (tmp_path / "some_hash").mkdir()
        assert _checkCache(tmp_path, "some_hash") is None

    def test_returns_none_when_file_zero_size(self, tmp_path):
        from Tensile.Toolchain.Source import _checkCache
        entry = tmp_path / "some_hash"
        entry.mkdir()
        (entry / "Kernels.so-000-gfx942.hsaco").write_bytes(b"")
        assert _checkCache(tmp_path, "some_hash") is None

    def test_returns_files_on_valid_entry(self, tmp_path):
        from Tensile.Toolchain.Source import _checkCache
        entry = tmp_path / "some_hash"
        entry.mkdir()
        (entry / "Kernels.so-000-gfx942.hsaco").write_bytes(b"\x7fELF")
        (entry / "Kernels.so-000-gfx942-xnack+.hsaco").write_bytes(b"\x7fELF")
        result = _checkCache(tmp_path, "some_hash")
        assert result is not None
        assert len(result) == 2
        assert all(f.suffix == ".hsaco" for f in result)

    def test_ignores_non_hsaco_files(self, tmp_path):
        from Tensile.Toolchain.Source import _checkCache
        entry = tmp_path / "some_hash"
        entry.mkdir()
        (entry / "Kernels.so-000-gfx942.hsaco").write_bytes(b"\x7fELF")
        (entry / "metadata.json").write_text("{}")
        result = _checkCache(tmp_path, "some_hash")
        assert len(result) == 1


class TestPopulateCache:
    def test_populates_empty_cache(self, tmp_path):
        from Tensile.Toolchain.Source import _populateCache
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        f1 = src_dir / "Kernels.so-000-gfx942.hsaco"
        f1.write_bytes(b"\x7fELF_data_1")
        _populateCache(cache_dir, "abc123", [f1])
        cached = cache_dir / "abc123" / "Kernels.so-000-gfx942.hsaco"
        assert cached.exists()
        assert cached.read_bytes() == b"\x7fELF_data_1"

    def test_skips_when_entry_exists(self, tmp_path):
        from Tensile.Toolchain.Source import _populateCache
        cache_dir = tmp_path / "cache"
        entry = cache_dir / "abc123"
        entry.mkdir(parents=True)
        (entry / "Kernels.so-000-gfx942.hsaco").write_bytes(b"original")
        src = tmp_path / "new.hsaco"
        src.write_bytes(b"different")
        _populateCache(cache_dir, "abc123", [src])
        assert (entry / "Kernels.so-000-gfx942.hsaco").read_bytes() == b"original"

    def test_cleans_up_tmp_on_race(self, tmp_path):
        from Tensile.Toolchain.Source import _populateCache
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Pre-create the final dir to simulate a race
        (cache_dir / "abc123").mkdir()
        (cache_dir / "abc123" / "f.hsaco").write_bytes(b"winner")
        src = tmp_path / "f.hsaco"
        src.write_bytes(b"loser")
        _populateCache(cache_dir, "abc123", [src])
        # No leftover tmp dirs
        tmp_dirs = list(cache_dir.glob(".tmp_*"))
        assert len(tmp_dirs) == 0

    def test_creates_cache_dir_if_missing(self, tmp_path):
        from Tensile.Toolchain.Source import _populateCache
        cache_dir = tmp_path / "cache" / "subdir"
        src = tmp_path / "f.hsaco"
        src.write_bytes(b"\x7fELF")
        _populateCache(cache_dir, "abc123", [src])
        assert (cache_dir / "abc123" / "f.hsaco").exists()
