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

import hashlib
import os
import shutil
import time

from pathlib import Path

from ..Common import print1


_STATIC_HEADER_FILES = [
    "KernelHeader.h",
    "TensileTypes.h",
    "tensile_bfloat16.h",
    "tensile_float8_bfloat8.h",
    "ReductionTemplate.h",
    "memory_gfx.h",
]


def _computeCacheKey(kernelPath, includeDir, cmdlineArchs, compiler):
    """Compute SHA256 cache key from source contents + build metadata."""
    h = hashlib.sha256()
    h.update(Path(kernelPath).read_bytes())
    h.update(Path(includeDir, "Kernels.h").read_bytes())
    for name in _STATIC_HEADER_FILES:
        h.update(Path(includeDir, name).read_bytes())
    h.update(",".join(sorted(cmdlineArchs)).encode())
    v = compiler.version
    h.update(f"{v.major}.{v.minor}.{v.patch}".encode())
    rv = compiler.rocm_version
    h.update(f"{rv.major}.{rv.minor}.{rv.patch}".encode())
    h.update(b"asan" if "-fsanitize=address" in compiler.default_args else b"no-asan")
    return h.hexdigest()


def _checkCache(cacheDir, cacheKey):
    """Check if a valid cache entry exists. Returns list of .hsaco Paths or None.

    Walks one level: cache entries are organized as <key>/<base-arch>/<*.hsaco>
    to mirror the on-disk install layout. A flat <key>/<*.hsaco> entry from an
    older cache version is treated as missing so it gets rewritten in the new
    structure on the next store.
    """
    entryDir = Path(cacheDir) / cacheKey
    if not entryDir.is_dir():
        return None
    hsacoFiles = list(entryDir.glob("*/*.hsaco"))
    if not hsacoFiles or any(f.stat().st_size == 0 for f in hsacoFiles):
        return None
    return hsacoFiles


def _populateCache(cacheDir, cacheKey, hsacoFiles, storedArchNames=None):
    """Atomically populate a cache entry. Safe under concurrent writes.

    storedArchNames maps each file's parent subtree back to the base arch the
    entry is keyed on (the key is the compiler target, so a stepping and its base
    share an entry). Empty/identity for ordinary builds.
    """
    storedNames = storedArchNames or {}
    cacheDir = Path(cacheDir)
    finalDir = cacheDir / cacheKey
    if finalDir.exists():
        return

    tmpDir = cacheDir / f".tmp_{cacheKey}_{os.getpid()}"
    tmpDir.mkdir(parents=True, exist_ok=True)
    for f in hsacoFiles:
        src = Path(f)
        archSubdir = tmpDir / storedNames.get(src.parent.name, src.parent.name)
        archSubdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, archSubdir / src.name)

    try:
        tmpDir.rename(finalDir)
    except OSError:
        shutil.rmtree(tmpDir, ignore_errors=True)


def _evictStale(cacheDir, maxAgeDays):
    """Remove cache entries whose directories are older than maxAgeDays."""
    cacheDir = Path(cacheDir)
    if not cacheDir.is_dir():
        return
    maxAgeSecs = maxAgeDays * 24 * 60 * 60
    now = time.time()
    for entry in cacheDir.iterdir():
        if not entry.is_dir() or entry.name.startswith(".tmp_"):
            continue
        try:
            age = now - entry.stat().st_mtime
            if age > maxAgeSecs:
                shutil.rmtree(entry, ignore_errors=True)
        except OSError:
            pass


class HelperKernelCache:
    """Filesystem cache for compiled helper kernel .hsaco files.

    Construct a fresh instance per build call to pick up the current environment.
    """

    _DEFAULT_DIR = Path.home() / ".tensile" / "helper_cache"
    _MAX_AGE_DAYS = 30

    def __init__(self):
        disabled = os.environ.get("TENSILE_DISABLE_HELPER_CACHE", "").upper() \
                   in ("1", "YES", "ON", "TRUE")
        self.enabled = not disabled
        self.dir = Path(os.environ.get("TENSILE_HELPER_CACHE_DIR",
                                       str(self._DEFAULT_DIR)))
        self._cacheKey = None
        _evictStale(self.dir, self._MAX_AGE_DAYS)

    def restore(self, kernelPath, includeDir, cmdlineArchs, compiler, destRoot, outputArchNames=None):
        """Restore cached .hsaco files (organized as <key>/<base-arch>/<*.hsaco>)
        into <destRoot>/<subtree>/<name>. outputArchNames maps each base-arch
        subdir to the subtree it ships under, so a stepping lands in its own tree;
        identity/empty for ordinary builds.

        Returns (hit, coPaths): copied paths on hit, [] on miss or when disabled.
        """
        outArchNames = outputArchNames or {}
        if not self.enabled:
            return False, []

        self._cacheKey = _computeCacheKey(kernelPath, includeDir, cmdlineArchs, compiler)
        cachedFiles = _checkCache(self.dir, self._cacheKey)

        if cachedFiles:
            coPaths = []
            try:
                os.utime(Path(self.dir) / self._cacheKey)
                for f in cachedFiles:
                    archSubdir = Path(destRoot) / outArchNames.get(f.parent.name, f.parent.name)
                    archSubdir.mkdir(parents=True, exist_ok=True)
                    dst = archSubdir / f.name
                    shutil.copy2(f, dst)
                    coPaths.append(str(dst))
                return True, coPaths
            except OSError:
                for p in coPaths:
                    Path(p).unlink(missing_ok=True)
                # fall through to cache miss

        print1(f"# Helper kernel cache MISS ({self._cacheKey[:12]}...)")
        return False, []

    def store(self, coPaths, outputArchNames=None):
        """Populate cache after a successful build. No-op if disabled or no key.

        outputArchNames is restore()'s base -> subtree map, inverted here to store
        under the base arch (the inverse is well defined: two steppings sharing an
        ISA is rejected by the capability guard).
        """
        if not self.enabled or not self._cacheKey:
            return
        storedArchNames = {out: base for base, out in (outputArchNames or {}).items()}
        self.dir.mkdir(parents=True, exist_ok=True)
        _populateCache(
            self.dir, self._cacheKey, [Path(p) for p in coPaths], storedArchNames
        )
