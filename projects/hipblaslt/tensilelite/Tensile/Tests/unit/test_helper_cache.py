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

import json
import os
import shutil
import threading
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


def _populate_test_entry(cache_dir, cache_key, files):
    """Populate a manifest-backed entry from {relative_path: contents}."""
    from Tensile.Toolchain.HelperKernelCache import _populateCache

    source_root = cache_dir.parent / f"{cache_dir.name}_inputs_{cache_key}"
    source_files = []
    for relative_path, contents in files.items():
        source = source_root / relative_path
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(contents)
        source_files.append(source)
    _populateCache(cache_dir, cache_key, source_files)
    return cache_dir / cache_key


def test_expected_cache_files_matches_source_output_layout():
    from Tensile.Toolchain.HelperKernelCache import _expectedCacheFiles

    assert _expectedCacheFiles(
        Path("Kernels.cpp"),
        ["gfx942:sramecc+:xnack+", "gfx942:xnack-", "gfx950"],
    ) == {
        "gfx942/Kernels.so-000-gfx942-xnack+.hsaco",
        "gfx942/Kernels.so-000-gfx942-xnack-.hsaco",
        "gfx950/Kernels.so-000-gfx950.hsaco",
    }


def test_valid_artifact_name_rejects_non_string():
    from Tensile.Toolchain.HelperKernelCache import _validArtifactName

    assert not _validArtifactName(Path("gfx942/Kernels.so-000-gfx942.hsaco"))


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

    def test_returns_none_for_legacy_entry_without_manifest(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _checkCache
        entry = tmp_path / "some_hash" / "gfx942"
        entry.mkdir(parents=True)
        (entry / "Kernels.so-000-gfx942.hsaco").write_bytes(b"\x7fELF")
        assert _checkCache(tmp_path, "some_hash") is None

    def test_returns_none_when_file_zero_size(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _checkCache
        entry = _populate_test_entry(
            tmp_path,
            "some_hash",
            {Path("gfx942/Kernels.so-000-gfx942.hsaco"): b"\x7fELF"},
        )
        (entry / "gfx942" / "Kernels.so-000-gfx942.hsaco").write_bytes(b"")
        assert _checkCache(tmp_path, "some_hash") is None

    def test_returns_files_on_valid_entry(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _checkCache
        _populate_test_entry(
            tmp_path,
            "some_hash",
            {
                Path("gfx942/Kernels.so-000-gfx942.hsaco"): b"\x7fELF",
                Path("gfx942/Kernels.so-000-gfx942-xnack+.hsaco"): b"\x7fELF",
            },
        )
        result = _checkCache(tmp_path, "some_hash")
        assert result is not None
        assert len(result) == 2
        assert all(f.suffix == ".hsaco" for f in result)

    def test_ignores_non_hsaco_files(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _checkCache
        entry = _populate_test_entry(
            tmp_path,
            "some_hash",
            {Path("gfx942/Kernels.so-000-gfx942.hsaco"): b"\x7fELF"},
        )
        (entry / "gfx942" / "metadata.json").write_text("{}")
        result = _checkCache(tmp_path, "some_hash")
        assert len(result) == 1

    def test_returns_none_when_manifest_version_is_stale(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _checkCache
        entry = _populate_test_entry(
            tmp_path,
            "some_hash",
            {Path("gfx942/Kernels.so-000-gfx942.hsaco"): b"\x7fELF"},
        )
        manifest_path = entry / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["version"] += 1
        manifest_path.write_text(json.dumps(manifest))
        assert _checkCache(tmp_path, "some_hash") is None

    def test_returns_none_when_manifest_is_not_an_object(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _checkCache
        entry = tmp_path / "some_hash"
        entry.mkdir()
        (entry / "manifest.json").write_text("[]")
        assert _checkCache(tmp_path, "some_hash") is None

    def test_returns_none_when_manifest_artifact_is_missing(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _checkCache
        entry = _populate_test_entry(
            tmp_path,
            "some_hash",
            {
                Path("gfx942/Kernels.so-000-gfx942.hsaco"): b"\x7fELF",
                Path("gfx950/Kernels.so-000-gfx950.hsaco"): b"\x7fELF",
            },
        )
        (entry / "gfx950" / "Kernels.so-000-gfx950.hsaco").unlink()
        assert _checkCache(tmp_path, "some_hash") is None

    def test_returns_none_when_file_content_changes_without_size_change(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _checkCache
        entry = _populate_test_entry(
            tmp_path,
            "some_hash",
            {Path("gfx942/Kernels.so-000-gfx942.hsaco"): b"original"},
        )
        (entry / "gfx942" / "Kernels.so-000-gfx942.hsaco").write_bytes(b"modified")
        assert _checkCache(tmp_path, "some_hash") is None

    @pytest.mark.parametrize(
        "relative_name",
        ["flat.hsaco", "outer/gfx942/Kernels.so-000-gfx942.hsaco"],
    )
    def test_returns_none_when_manifest_path_has_wrong_shape(self, tmp_path, relative_name):
        from Tensile.Toolchain.HelperKernelCache import _checkCache, _fileDigest
        entry = tmp_path / "some_hash"
        artifact = entry / relative_name
        artifact.parent.mkdir(parents=True)
        artifact.write_bytes(b"\x7fELF")
        manifest = {
            "version": 1,
            "cache_key": "some_hash",
            "files": {
                relative_name: {
                    "size": artifact.stat().st_size,
                    "sha256": _fileDigest(artifact),
                }
            },
        }
        (entry / "manifest.json").write_text(json.dumps(manifest))

        assert _checkCache(tmp_path, "some_hash") is None


class TestPopulateCache:
    def test_populates_empty_cache(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _checkCache, _populateCache
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
        assert (cache_dir / "abc123" / "manifest.json").is_file()
        assert _checkCache(cache_dir, "abc123") == [cached]

    def test_published_entry_inherits_cache_root_permissions(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _populateCache
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cache_dir.chmod(0o770)
        source = tmp_path / "src" / "gfx942" / "f.hsaco"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"\x7fELF")

        _populateCache(cache_dir, "abc123", [source])

        assert (cache_dir / "abc123").stat().st_mode & 0o777 == 0o770

    def test_skips_when_entry_exists(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _populateCache
        cache_dir = tmp_path / "cache"
        entry = _populate_test_entry(
            cache_dir,
            "abc123",
            {Path("gfx942/Kernels.so-000-gfx942.hsaco"): b"original"},
        )
        src_dir = tmp_path / "src" / "gfx942"
        src_dir.mkdir(parents=True)
        src = src_dir / "new.hsaco"
        src.write_bytes(b"different")
        _populateCache(cache_dir, "abc123", [src])
        assert (entry / "gfx942" / "Kernels.so-000-gfx942.hsaco").read_bytes() == b"original"
        assert not (entry / "gfx942" / "new.hsaco").exists()

    def test_replaces_incomplete_existing_entry(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _checkCache, _populateCache
        cache_dir = tmp_path / "cache"
        stale_entry = cache_dir / "abc123" / "gfx942"
        stale_entry.mkdir(parents=True)
        (stale_entry / "Kernels.so-000-gfx942.hsaco").write_bytes(b"stale")

        src_dir = tmp_path / "src" / "gfx942"
        src_dir.mkdir(parents=True)
        src = src_dir / "Kernels.so-000-gfx942.hsaco"
        src.write_bytes(b"replacement")
        _populateCache(cache_dir, "abc123", [src])

        cached = cache_dir / "abc123" / "gfx942" / src.name
        assert cached.read_bytes() == b"replacement"
        assert _checkCache(cache_dir, "abc123") == [cached]
        assert not list(cache_dir.glob(".tmp_*"))
        assert not list(cache_dir.glob(".stale_*"))

    def test_does_not_publish_duplicate_destinations(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _populateCache
        cache_dir = tmp_path / "cache"
        sources = []
        for parent in ("first", "second"):
            source = tmp_path / parent / "same.hsaco"
            source.parent.mkdir()
            source.write_text(parent)
            sources.append(source)

        _populateCache(
            cache_dir,
            "abc123",
            sources,
            storedArchNames={"first": "gfx950", "second": "gfx950"},
        )

        assert not (cache_dir / "abc123").exists()
        assert not list(cache_dir.glob(".tmp_*"))

    def test_does_not_stage_outside_cache_for_invalid_arch_name(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _populateCache
        cache_dir = tmp_path / "cache"
        source = tmp_path / "src" / "gfx950" / "f.hsaco"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"\x7fELF")

        _populateCache(
            cache_dir,
            "abc123",
            [source],
            storedArchNames={"gfx950": "../outside"},
        )

        assert not (cache_dir / "abc123").exists()
        assert not (cache_dir / "outside" / "f.hsaco").exists()

    @pytest.mark.parametrize("contents", [b"", None])
    def test_does_not_publish_empty_entries(self, tmp_path, contents):
        from Tensile.Toolchain.HelperKernelCache import _populateCache
        cache_dir = tmp_path / "cache"
        sources = []
        if contents is not None:
            source = tmp_path / "src" / "gfx950" / "empty.hsaco"
            source.parent.mkdir(parents=True)
            source.write_bytes(contents)
            sources.append(source)

        _populateCache(cache_dir, "abc123", sources)

        assert not (cache_dir / "abc123").exists()
        assert not list(cache_dir.glob(".tmp_*"))

    def test_cleans_up_tmp_on_race(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _populateCache
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        # Pre-create a complete final entry to simulate a concurrent winner.
        _populate_test_entry(
            cache_dir, "abc123", {Path("gfx942/f.hsaco"): b"winner"}
        )
        src_dir = tmp_path / "src" / "gfx942"
        src_dir.mkdir(parents=True)
        src = src_dir / "f.hsaco"
        src.write_bytes(b"loser")
        _populateCache(cache_dir, "abc123", [src])
        # No leftover tmp dirs
        tmp_dirs = list(cache_dir.glob(".tmp_*"))
        assert len(tmp_dirs) == 0
        assert (cache_dir / "abc123" / "gfx942" / "f.hsaco").read_bytes() == b"winner"

    def test_concurrent_writers_publish_one_complete_entry(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _checkCache, _populateCache
        cache_dir = tmp_path / "cache"
        sources = []
        for index in range(8):
            src_dir = tmp_path / f"src_{index}" / "gfx950"
            src_dir.mkdir(parents=True)
            src = src_dir / "Kernels.so-000-gfx950.hsaco"
            src.write_bytes(f"writer-{index}".encode())
            sources.append(src)

        barrier = threading.Barrier(len(sources))
        errors = []

        def populate(source):
            try:
                barrier.wait()
                _populateCache(cache_dir, "abc123", [source])
            except Exception as error:
                errors.append(error)

        threads = [threading.Thread(target=populate, args=(source,)) for source in sources]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == []
        result = _checkCache(cache_dir, "abc123")
        assert result is not None and len(result) == 1
        assert result[0].read_bytes() in {source.read_bytes() for source in sources}
        assert not list(cache_dir.glob(".tmp_*"))
        assert not list(cache_dir.glob(".stale_*"))

    def test_keeps_valid_entry_published_by_concurrent_writer(self, tmp_path, monkeypatch):
        from Tensile.Toolchain.HelperKernelCache import _checkCache, _populateCache
        cache_dir = tmp_path / "cache"
        source = tmp_path / "src" / "gfx950" / "Kernels.so-000-gfx950.hsaco"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"winner")
        final_dir = cache_dir / "abc123"
        real_rename = Path.rename

        def publish_winner_then_fail(path, target):
            if Path(path).name.startswith(".tmp_") and Path(target) == final_dir:
                shutil.copytree(path, final_dir)
                raise OSError("another writer won")
            return real_rename(path, target)

        monkeypatch.setattr(Path, "rename", publish_winner_then_fail)
        _populateCache(cache_dir, "abc123", [source])

        assert _checkCache(cache_dir, "abc123") is not None
        assert not list(cache_dir.glob(".tmp_*"))

    @pytest.mark.parametrize("restore_succeeds", [True, False])
    def test_handles_concurrent_winner_when_replacement_publish_fails(
        self, tmp_path, monkeypatch, restore_succeeds
    ):
        from Tensile.Toolchain.HelperKernelCache import _checkCache, _populateCache
        cache_dir = tmp_path / "cache"
        stale = cache_dir / "abc123" / "gfx950" / "old.hsaco"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"stale")

        source = tmp_path / "src" / "gfx950" / "new.hsaco"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"replacement")

        winner_cache = tmp_path / "winner-cache"
        winner_entry = _populate_test_entry(
            winner_cache,
            "abc123",
            {Path("gfx950/new.hsaco"): b"winner"},
        )
        prepared_winner = tmp_path / "prepared-winner"
        winner_entry.rename(prepared_winner)

        final_dir = cache_dir / "abc123"
        real_rename = Path.rename
        publish_attempts = 0

        def publish_winner_then_fail_replacement(path, target):
            nonlocal publish_attempts
            path = Path(path)
            target = Path(target)
            if path.name.startswith(".tmp_") and target == final_dir:
                publish_attempts += 1
                if publish_attempts == 2:
                    raise OSError("replacement publish failed")
            if (
                not restore_succeeds
                and path.name.startswith(".stale_")
                and target == final_dir
            ):
                raise OSError("winner restore failed")
            if path == final_dir and target.name.startswith(".stale_"):
                shutil.rmtree(final_dir)
                real_rename(prepared_winner, final_dir)
            return real_rename(path, target)

        monkeypatch.setattr(Path, "rename", publish_winner_then_fail_replacement)
        _populateCache(cache_dir, "abc123", [source])

        cached = _checkCache(cache_dir, "abc123")
        if restore_succeeds:
            assert cached is not None and len(cached) == 1
            assert cached[0].read_bytes() == b"winner"
        else:
            assert cached is None
        assert not list(cache_dir.glob(".tmp_*"))
        assert not list(cache_dir.glob(".stale_*"))

    def test_failed_replacement_leaves_no_partial_entry(self, tmp_path, monkeypatch):
        from Tensile.Toolchain.HelperKernelCache import _populateCache
        cache_dir = tmp_path / "cache"
        stale = cache_dir / "abc123" / "gfx950" / "old.hsaco"
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"stale")
        source = tmp_path / "src" / "gfx950" / "new.hsaco"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"replacement")
        real_rename = Path.rename
        calls = 0

        def fail_publish_after_quarantine(path, target):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("publish failed")
            return real_rename(path, target)

        monkeypatch.setattr(Path, "rename", fail_publish_after_quarantine)
        _populateCache(cache_dir, "abc123", [source])

        assert not (cache_dir / "abc123").exists()
        assert not list(cache_dir.glob(".tmp_*"))
        assert not list(cache_dir.glob(".stale_*"))

    def test_staging_failure_is_ignored_and_cleaned_up(self, tmp_path, monkeypatch):
        from Tensile.Toolchain.HelperKernelCache import _populateCache
        cache_dir = tmp_path / "cache"
        source = tmp_path / "src" / "gfx950" / "new.hsaco"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"replacement")
        monkeypatch.setattr(shutil, "copy2", lambda *_args: (_ for _ in ()).throw(OSError("full")))

        _populateCache(cache_dir, "abc123", [source])

        assert not (cache_dir / "abc123").exists()
        assert not list(cache_dir.glob(".tmp_*"))

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

    def test_ignores_entry_that_disappears_during_stat(self, tmp_path, monkeypatch):
        from Tensile.Toolchain.HelperKernelCache import _evictStale
        cache_dir = tmp_path / "cache"
        entry = cache_dir / "entry"
        entry.mkdir(parents=True)
        real_stat = Path.stat
        calls = 0

        def fail_second_stat(path, *args, **kwargs):
            nonlocal calls
            if path == entry:
                calls += 1
                if calls == 2:
                    raise OSError("entry disappeared")
            return real_stat(path, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", fail_second_stat)
        _evictStale(cache_dir, 30)
        assert calls == 2


class TestRemovePath:
    def test_removes_file(self, tmp_path):
        from Tensile.Toolchain.HelperKernelCache import _removePath
        path = tmp_path / "stale"
        path.write_text("data")
        _removePath(path)
        assert not path.exists()

    def test_ignores_file_removal_failure(self, tmp_path, monkeypatch):
        from Tensile.Toolchain.HelperKernelCache import _removePath
        path = tmp_path / "stale"
        path.write_text("data")
        monkeypatch.setattr(Path, "unlink", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("busy")))
        _removePath(path)


class TestRestoreRobustness:
    def test_store_does_not_publish_missing_requested_arch(self, tmp_path, monkeypatch):
        from Tensile.Toolchain.HelperKernelCache import HelperKernelCache
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TENSILE_HELPER_CACHE_DIR", str(cache_dir))
        monkeypatch.delenv("TENSILE_DISABLE_HELPER_CACHE", raising=False)

        kernel_path = _write_test_files(tmp_path)
        compiler = MockCompiler()
        archs = ["gfx942", "gfx950"]
        cache = HelperKernelCache()
        assert cache.restore(
            kernel_path, tmp_path, archs, compiler, tmp_path / "restore"
        ) == (False, [])
        output = tmp_path / "built" / "gfx942" / "Kernels.so-000-gfx942.hsaco"
        output.parent.mkdir(parents=True)
        output.write_bytes(b"partial")

        cache.store([output])

        assert not any(cache_dir.glob("*/manifest.json"))

    def test_manifest_backed_entry_missing_requested_arch_is_a_miss(
        self, tmp_path, monkeypatch
    ):
        from Tensile.Toolchain.HelperKernelCache import HelperKernelCache, _computeCacheKey
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TENSILE_HELPER_CACHE_DIR", str(cache_dir))
        monkeypatch.delenv("TENSILE_DISABLE_HELPER_CACHE", raising=False)

        kernel_path = _write_test_files(tmp_path)
        compiler = MockCompiler()
        archs = ["gfx942", "gfx950"]
        key = _computeCacheKey(kernel_path, tmp_path, archs, compiler)
        _populate_test_entry(
            cache_dir,
            key,
            {Path("gfx942/Kernels.so-000-gfx942.hsaco"): b"partial"},
        )

        hit, paths = HelperKernelCache().restore(
            kernel_path, tmp_path, archs, compiler, tmp_path / "restore"
        )

        assert not hit
        assert paths == []
        assert not (tmp_path / "restore" / "gfx942").exists()

    def test_incomplete_multi_arch_entry_is_rebuilt(self, tmp_path, monkeypatch):
        """A legacy gfx942-only entry must not hide a missing gfx950 helper."""
        from Tensile.Toolchain.HelperKernelCache import HelperKernelCache, _computeCacheKey
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TENSILE_HELPER_CACHE_DIR", str(cache_dir))
        monkeypatch.delenv("TENSILE_DISABLE_HELPER_CACHE", raising=False)

        kernel_path = _write_test_files(tmp_path)
        compiler = MockCompiler()
        archs = ["gfx942", "gfx950"]
        key = _computeCacheKey(kernel_path, tmp_path, archs, compiler)
        legacy = cache_dir / key / "gfx942" / "Kernels.so-000-gfx942.hsaco"
        legacy.parent.mkdir(parents=True)
        legacy.write_bytes(b"legacy")

        cache = HelperKernelCache()
        hit, paths = cache.restore(kernel_path, tmp_path, archs, compiler, tmp_path / "first")
        assert not hit and paths == []

        outputs = []
        for arch in archs:
            output = tmp_path / "built" / arch / f"Kernels.so-000-{arch}.hsaco"
            output.parent.mkdir(parents=True)
            output.write_bytes(f"compiled-{arch}".encode())
            outputs.append(output)
        cache.store(outputs)

        hit, paths = HelperKernelCache().restore(
            kernel_path, tmp_path, archs, compiler, tmp_path / "second"
        )
        assert hit
        assert {Path(path).parent.name for path in paths} == set(archs)
        assert {Path(path).name for path in paths} == {
            "Kernels.so-000-gfx942.hsaco",
            "Kernels.so-000-gfx950.hsaco",
        }

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

        entry = _populate_test_entry(
            cache_dir,
            key,
            {Path("gfx942/Kernels.so-000-gfx942.hsaco"): b"\x7fELF"},
        )
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
        from Tensile.Toolchain.HelperKernelCache import HelperKernelCache, _computeCacheKey
        cache_dir = tmp_path / "cache"
        monkeypatch.setenv("TENSILE_HELPER_CACHE_DIR", str(cache_dir))
        monkeypatch.delenv("TENSILE_DISABLE_HELPER_CACHE", raising=False)

        kernel_path = _write_test_files(tmp_path)
        compiler = MockCompiler()
        key = _computeCacheKey(kernel_path, tmp_path, ["gfx942"], compiler)

        _populate_test_entry(
            cache_dir,
            key,
            {Path("gfx942/Kernels.so-000-gfx942.hsaco"): b"\x7fELF"},
        )

        dest = tmp_path / "dest"
        dest.mkdir()

        # Simulate deletion by making the source file unreadable after _checkCache
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
        archs = ["gfx942", "gfx1100"]
        key = _computeCacheKey(kernel_path, tmp_path, archs, compiler)

        _populate_test_entry(
            cache_dir,
            key,
            {
                Path("gfx942/Kernels.so-000-gfx942.hsaco"): b"\x7fELF",
                Path("gfx1100/Kernels.so-000-gfx1100.hsaco"): b"\x7fELF",
            },
        )

        dest = tmp_path / "dest"
        dest.mkdir()

        # Let the first copy succeed, fail on the second
        call_count = 0
        original_copy2 = shutil.copy2
        def copy2_fail_second(src, dst):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                Path(dst).write_bytes(b"partial")
                raise OSError("deleted mid-copy")
            return original_copy2(src, dst)
        monkeypatch.setattr(shutil, "copy2", copy2_fail_second)

        cache = HelperKernelCache()
        hit, coPaths = cache.restore(kernel_path, tmp_path, archs, compiler, dest)
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
        _populate_test_entry(
            cache_dir,
            "test_key",
            {Path("gfx942/Kernels.so-000-gfx942.hsaco"): b"\x7fELF"},
        )
        result = _checkCache(cache_dir, "test_key")
        assert result is not None
        assert len(result) == 1
