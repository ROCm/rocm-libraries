################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``Tensile.Toolchain.HelperKernelCache`` — the pure
filesystem cache helpers (`_computeCacheKey`/`_checkCache`/`_populateCache`/
`_evictStale`) and the `HelperKernelCache` restore/store surface (cache
enabled/disabled, hit/miss), driven over tmp dirs with a fake compiler (no real
compilation)."""

import os
import time
import types
from pathlib import Path

import pytest

import Tensile.Toolchain.HelperKernelCache as HKC
from Tensile.Toolchain.HelperKernelCache import (
    _computeCacheKey, _checkCache, _populateCache, _evictStale, HelperKernelCache,
    _STATIC_HEADER_FILES,
)

pytestmark = pytest.mark.unit


def _fake_compiler():
    ver = types.SimpleNamespace(major=6, minor=2, patch=0)
    return types.SimpleNamespace(version=ver, rocm_version=ver, default_args=[])


def _make_include_dir(d):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    (d / "Kernels.h").write_text("// kernels")
    for name in _STATIC_HEADER_FILES:
        (d / name).write_text(f"// {name}")
    return d


def _make_kernel(d):
    p = Path(d) / "kern.cpp"
    p.write_text("// kernel source")
    return p


def test_compute_cache_key_deterministic(tmp_path):
    inc = _make_include_dir(tmp_path / "inc")
    kern = _make_kernel(tmp_path)
    comp = _fake_compiler()
    k1 = _computeCacheKey(str(kern), str(inc), ["gfx942"], comp)
    k2 = _computeCacheKey(str(kern), str(inc), ["gfx942"], comp)
    assert k1 == k2 and len(k1) == 64
    # Different archs -> different key.
    assert _computeCacheKey(str(kern), str(inc), ["gfx90a"], comp) != k1


def test_compute_cache_key_asan_differs(tmp_path):
    inc = _make_include_dir(tmp_path / "inc")
    kern = _make_kernel(tmp_path)
    base = _computeCacheKey(str(kern), str(inc), ["gfx942"], _fake_compiler())
    asan = _fake_compiler()
    asan.default_args = ["-fsanitize=address"]
    assert _computeCacheKey(str(kern), str(inc), ["gfx942"], asan) != base


def test_check_cache_missing_empty_valid(tmp_path):
    cacheDir = tmp_path / "cache"
    cacheDir.mkdir()
    assert _checkCache(cacheDir, "nope") is None          # no entry dir
    entry = cacheDir / "key"; entry.mkdir()
    assert _checkCache(cacheDir, "key") is None            # empty dir
    (entry / "a.hsaco").write_text("x")
    assert _checkCache(cacheDir, "key") is None            # flat layout -> miss
    arch = entry / "gfx942"; arch.mkdir()
    (arch / "a.hsaco").write_text("x")
    assert _checkCache(cacheDir, "key") is not None        # valid <key>/<arch>/*.hsaco
    (arch / "b.hsaco").write_text("")                      # zero-size -> invalid
    assert _checkCache(cacheDir, "key") is None


def test_populate_cache_and_already_exists(tmp_path):
    cacheDir = tmp_path / "cache"; cacheDir.mkdir()
    archDir = tmp_path / "gfx942"; archDir.mkdir()
    src = archDir / "a.hsaco"; src.write_text("data")
    _populateCache(cacheDir, "k", [src])
    assert (cacheDir / "k" / "gfx942" / "a.hsaco").read_text() == "data"
    # Second call -> finalDir exists -> early return (no error).
    _populateCache(cacheDir, "k", [src])
    assert (cacheDir / "k" / "gfx942" / "a.hsaco").exists()


def test_evict_stale(tmp_path):
    cacheDir = tmp_path / "cache"; cacheDir.mkdir()
    _evictStale(tmp_path / "does_not_exist", 30)           # no-op on missing dir
    fresh = cacheDir / "fresh"; fresh.mkdir()
    stale = cacheDir / "stale"; stale.mkdir()
    tmpd = cacheDir / ".tmp_x"; tmpd.mkdir()               # skipped (.tmp_ prefix)
    old = time.time() - 40 * 24 * 60 * 60
    os.utime(stale, (old, old))
    _evictStale(cacheDir, 30)
    assert fresh.exists() and tmpd.exists() and not stale.exists()


def test_populate_cache_rename_failure_cleans_up(tmp_path, monkeypatch):
    # Force the tmpDir.rename() OSError arm -> tmp dir is cleaned up, no entry.
    cacheDir = tmp_path / "cache"; cacheDir.mkdir()
    src = tmp_path / "a.hsaco"; src.write_text("data")
    monkeypatch.setattr(Path, "rename", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))
    _populateCache(cacheDir, "k", [src])
    assert not (cacheDir / "k").exists()
    assert not list(cacheDir.glob(".tmp_*"))


def test_restore_copy_failure_falls_through_to_miss(tmp_path, monkeypatch):
    # A populated cache, but the copy into destDir fails -> the except OSError
    # arm unlinks partials and returns a miss.
    monkeypatch.delenv("TENSILE_DISABLE_HELPER_CACHE", raising=False)
    monkeypatch.setenv("TENSILE_HELPER_CACHE_DIR", str(tmp_path / "cache"))
    inc = _make_include_dir(tmp_path / "inc")
    kern = _make_kernel(tmp_path)
    comp = _fake_compiler()
    dest = tmp_path / "dest"; dest.mkdir()

    cache = HelperKernelCache()
    cache.restore(str(kern), str(inc), ["gfx942"], comp, str(dest))  # miss, sets key
    co = tmp_path / "out.hsaco"; co.write_text("compiled")
    cache.store([str(co)])

    monkeypatch.setattr(HKC.shutil, "copy2", lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    cache2 = HelperKernelCache()
    hit, paths = cache2.restore(str(kern), str(inc), ["gfx942"], comp, str(dest))
    assert hit is False and paths == []


def test_cache_disabled_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSILE_DISABLE_HELPER_CACHE", "1")
    monkeypatch.setenv("TENSILE_HELPER_CACHE_DIR", str(tmp_path / "c"))
    cache = HelperKernelCache()
    assert cache.enabled is False
    # restore returns (False, []) without touching the fs; store is a no-op.
    assert cache.restore("k", "inc", ["gfx942"], _fake_compiler(), str(tmp_path)) == (False, [])
    cache.store(["x.hsaco"])  # no-op


def test_cache_miss_then_store_then_hit(tmp_path, monkeypatch):
    monkeypatch.delenv("TENSILE_DISABLE_HELPER_CACHE", raising=False)
    monkeypatch.setenv("TENSILE_HELPER_CACHE_DIR", str(tmp_path / "cache"))
    inc = _make_include_dir(tmp_path / "inc")
    kern = _make_kernel(tmp_path)
    comp = _fake_compiler()
    dest = tmp_path / "dest"; dest.mkdir()

    cache = HelperKernelCache()
    assert cache.enabled is True

    # MISS (nothing cached yet).
    hit, paths = cache.restore(str(kern), str(inc), ["gfx942"], comp, str(dest))
    assert hit is False and paths == []

    # Build artifacts and store them.
    co = tmp_path / "out.hsaco"; co.write_text("compiled")
    cache.store([str(co)])

    # HIT on a fresh instance with the same inputs.
    cache2 = HelperKernelCache()
    cache2._cacheKey = None
    hit2, paths2 = cache2.restore(str(kern), str(inc), ["gfx942"], comp, str(dest))
    assert hit2 is True
    assert [Path(p).name for p in paths2] == ["out.hsaco"]
