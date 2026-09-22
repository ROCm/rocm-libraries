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
import json
import os
import shutil
import stat
import tempfile
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

_CACHE_FORMAT_VERSION = 1
_CACHE_MANIFEST = "manifest.json"


def _fileDigest(path):
    """Return the SHA256 digest of a file without loading it all into memory."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _expectedCacheFiles(kernelPath, cmdlineArchs):
    """Return the cache-relative helper artifacts required by a build."""
    kernelName = Path(kernelPath).stem
    expected = set()
    for arch in cmdlineArchs:
        parts = arch.split(":")
        baseArch = parts[0]
        xnack = next(
            (feature for feature in parts[1:] if feature.startswith("xnack")), None
        )
        filenameArch = baseArch + ("-" + xnack if xnack else "")
        expected.add(
            (Path(baseArch) / f"{kernelName}.so-000-{filenameArch}.hsaco").as_posix()
        )
    return expected


def _validArtifactName(relativeName):
    """Whether a manifest name has the canonical <arch>/<file>.hsaco shape."""
    if not isinstance(relativeName, str):
        return False
    path = Path(relativeName)
    return (
        not path.is_absolute()
        and len(path.parts) == 2
        and all(part not in ("", ".", "..") for part in path.parts)
        and path.as_posix() == relativeName
        and path.suffix == ".hsaco"
    )


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


def _checkCacheEntry(entryDir, cacheKey, expectedFiles=None):
    """Validate one cache entry directory against its key and expected files."""
    entryDir = Path(entryDir)
    if not entryDir.is_dir():
        return None

    try:
        manifest = json.loads((entryDir / _CACHE_MANIFEST).read_text())
        if not isinstance(manifest, dict):
            return None
        records = manifest.get("files")
        if (
            manifest.get("version") != _CACHE_FORMAT_VERSION
            or manifest.get("cache_key") != cacheKey
            or not isinstance(records, dict)
            or not records
            or not all(_validArtifactName(name) for name in records)
        ):
            return None

        recordNames = set(records)
        if expectedFiles is not None and recordNames != set(expectedFiles):
            return None

        actualPaths = {
            p.relative_to(entryDir).as_posix() for p in entryDir.rglob("*.hsaco")
        }
        if recordNames != actualPaths:
            return None

        hsacoFiles = []
        for relativeName in sorted(records):
            cachedFile = entryDir / relativeName
            record = records[relativeName]
            if (
                cachedFile.is_symlink()
                or not cachedFile.is_file()
                or record["size"] <= 0
                or cachedFile.stat().st_size != record["size"]
                or _fileDigest(cachedFile) != record["sha256"]
            ):
                return None
            hsacoFiles.append(cachedFile)
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        return None

    return hsacoFiles


def _checkCache(cacheDir, cacheKey, expectedFiles=None):
    """Check if a valid cache entry exists. Returns list of .hsaco Paths or None.

    Cache entries are organized as <key>/<base-arch>/<*.hsaco> and contain a
    versioned manifest written only after every artifact has been staged.  The
    manifest makes a missing, truncated, or stale subset a miss instead of a
    false hit. Entries from before the manifest format are intentionally misses
    and are replaced on the next store. expectedFiles, when supplied, is the
    exact cache-relative artifact set required by the current build.
    """
    entryDir = Path(cacheDir) / cacheKey
    return _checkCacheEntry(entryDir, cacheKey, expectedFiles)


def _removePath(path):
    """Best-effort removal for an internal cache file or directory."""
    path = Path(path)
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _populateCache(
    cacheDir, cacheKey, hsacoFiles, storedArchNames=None, expectedFiles=None
):
    """Atomically populate or repair a cache entry.

    storedArchNames maps each file's parent subtree back to the base arch the
    entry is keyed on (the key is the compiler target, so a stepping and its base
    share an entry). Empty/identity for ordinary builds. Each writer stages in a
    unique directory. A valid concurrent winner is kept; an invalid entry is
    quarantined before the complete staged entry is renamed into place.
    expectedFiles prevents a partial build output from being published.
    """
    storedNames = storedArchNames or {}
    cacheDir = Path(cacheDir)
    finalDir = cacheDir / cacheKey
    cacheDir.mkdir(parents=True, exist_ok=True)

    if _checkCache(cacheDir, cacheKey, expectedFiles) is not None:
        return

    tmpDir = Path(tempfile.mkdtemp(prefix=f".tmp_{cacheKey}_", dir=cacheDir))
    try:
        manifestFiles = {}
        for f in hsacoFiles:
            src = Path(f)
            archName = storedNames.get(src.parent.name, src.parent.name)
            relativeName = (Path(archName) / src.name).as_posix()
            if not _validArtifactName(relativeName) or relativeName in manifestFiles:
                return

            dst = tmpDir / archName / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            size = dst.stat().st_size
            if size == 0:
                return
            manifestFiles[relativeName] = {
                "size": size,
                "sha256": _fileDigest(dst),
            }

        if not manifestFiles or (
            expectedFiles is not None and set(manifestFiles) != set(expectedFiles)
        ):
            return

        manifest = {
            "version": _CACHE_FORMAT_VERSION,
            "cache_key": cacheKey,
            "files": manifestFiles,
        }
        (tmpDir / _CACHE_MANIFEST).write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        )
        # mkdtemp deliberately starts at 0700. Match the configured cache root
        # only after staging is complete so the published entry retains shared-
        # cache access without exposing a partially written entry.
        tmpDir.chmod(stat.S_IMODE(cacheDir.stat().st_mode))

        # A directory rename publishes the artifacts and their manifest as one
        # unit. If an old/incomplete entry occupies the key, move it aside first;
        # readers see either that invalid entry (a miss), no entry (a miss), or
        # the complete replacement. A concurrent valid writer always wins.
        try:
            tmpDir.rename(finalDir)
            tmpDir = None
            return
        except OSError:
            if _checkCache(cacheDir, cacheKey, expectedFiles) is not None:
                return

        staleDir = cacheDir / f".stale_{cacheKey}_{tmpDir.name}"
        try:
            finalDir.rename(staleDir)
        except OSError:
            return

        try:
            tmpDir.rename(finalDir)
            tmpDir = None
        except OSError:
            # The entry moved aside above may be a valid writer that won after
            # our last check. If our publication fails, put that winner back.
            if (
                not finalDir.exists()
                and _checkCacheEntry(staleDir, cacheKey, expectedFiles) is not None
            ):
                try:
                    staleDir.rename(finalDir)
                except OSError:
                    pass
            return
        finally:
            _removePath(staleDir)
    except OSError:
        # The cache is an optimization; a cache filesystem failure must not
        # turn a successful helper-kernel build into a failed build.
        return
    finally:
        if tmpDir is not None:
            _removePath(tmpDir)


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
        self._expectedFiles = None
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
        self._expectedFiles = _expectedCacheFiles(kernelPath, cmdlineArchs)
        cachedFiles = _checkCache(self.dir, self._cacheKey, self._expectedFiles)

        if cachedFiles:
            coPaths = []
            try:
                os.utime(Path(self.dir) / self._cacheKey)
                for f in cachedFiles:
                    archSubdir = Path(destRoot) / outArchNames.get(f.parent.name, f.parent.name)
                    archSubdir.mkdir(parents=True, exist_ok=True)
                    dst = archSubdir / f.name
                    coPaths.append(str(dst))
                    shutil.copy2(f, dst)
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
            self.dir,
            self._cacheKey,
            [Path(p) for p in coPaths],
            storedArchNames,
            self._expectedFiles,
        )
