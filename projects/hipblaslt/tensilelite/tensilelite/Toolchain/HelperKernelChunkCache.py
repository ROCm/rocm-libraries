# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Cache of the compiled chunks of the helper-kernel source (see HelperKernelChunks).

A chunk holds the helper kernels whose names hash to it, so most chunks of a build have the same
content as in an earlier build (of another config, or of another benchmark step) that shared
those kernels, even when the whole Kernels.cpp, and with it HelperKernelCache's key, differs.
Each chunk's compiled object (the output of the driver's compile job) is cached with
HelperKernelCache's pattern next to it: a SHA-256 key, the entry written to a temporary
directory unique to the store and renamed into place, entries refreshed on use and evicted after
30 days.

The key is the chunk's source and generated header, the static headers, the compile job with the
names that differ between build directories masked, and the full compiler version.

TENSILE_HELPER_CHUNK_CACHE_DIR overrides the location (default ~/.tensile/helper_chunk_cache).
TENSILE_DISABLE_HELPER_CHUNK_CACHE=1/YES/ON/TRUE compiles every chunk.
"""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from subprocess import CalledProcessError
from typing import List, Optional, Sequence, Tuple, Union

from ..Common import print1
from .HelperKernelCache import _STATIC_HEADER_FILES, _evictStale
from .HelperKernelPch import _compilerVersion

_OBJECT_FILE = "chunk.o"
# A store's temporary directory older than this was left by a killed process.
_TMP_MAX_AGE_DAYS = 1


def _sha256(path: Union[Path, str]) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _keyedCompileJob(compileJob: List[str], chunkPath: str, includeDir: str) -> List[str]:
    """The driver's compile job for a chunk without what differs between two builds of the same
    chunk and does not change its code: the output (a temporary file), the -cuid (a hash of the
    source path, only used to name a symbol), and the source path, include directory and working
    directory (only recorded in debug information, which helper compiles do not emit)."""
    keyed = []
    args = iter(compileJob)
    for arg in args:
        if arg == "-o":
            next(args, None)
        elif arg.startswith("-cuid="):
            continue
        elif arg.startswith(("-fdebug-compilation-dir=", "-fcoverage-compilation-dir=")):
            keyed.append(arg.split("=", 1)[0] + "=<cwd>")
        elif arg == chunkPath:
            keyed.append("<source>")
        elif arg == includeDir:
            keyed.append("<include>")
        else:
            keyed.append(arg)
    return keyed


def _computeChunkKey(
    chunkPath: str,
    compileJob: List[str],
    includeDir: str,
    staticHeaders: dict,
    compilerVersion: str,
) -> str:
    """SHA-256 of everything a chunk's object depends on: its source and its generated header
    (Kernels_chunk<i>.h next to it, writeHelpers), the static headers in includeDir, the
    compile job (every flag, the target, the PCH and the header search paths) and the exact
    compiler build."""
    return hashlib.sha256(
        json.dumps(
            {
                "source": _sha256(chunkPath),
                "header": _sha256(Path(chunkPath).with_suffix(".h")),
                "staticHeaders": staticHeaders,
                "compileJob": _keyedCompileJob(compileJob, chunkPath, includeDir),
                "compilerVersion": compilerVersion,
            }
        ).encode()
    ).hexdigest()


class HelperKernelChunkCache:
    """Filesystem cache of compiled helper-kernel chunks.

    Construct a fresh instance per build call to pick up the current environment.
    """

    _DEFAULT_DIR = Path.home() / ".tensile" / "helper_chunk_cache"
    _MAX_AGE_DAYS = 30

    def __init__(self):
        disabled = os.environ.get("TENSILE_DISABLE_HELPER_CHUNK_CACHE", "").upper() in (
            "1",
            "YES",
            "ON",
            "TRUE",
        )
        self.enabled = not disabled
        self.dir = Path(os.environ.get("TENSILE_HELPER_CHUNK_CACHE_DIR", str(self._DEFAULT_DIR)))
        if self.enabled:
            _evictStale(self.dir, self._MAX_AGE_DAYS, _TMP_MAX_AGE_DAYS)

    def keys(
        self,
        compilerPath: Union[Path, str],
        includeDir: str,
        chunks: Sequence[Tuple[str, List[str]]],
    ) -> List[Optional[str]]:
        """The key of each (chunk source path, driver compile job) in chunks; all None (compile
        every chunk, store nothing) when disabled or when a key cannot be computed."""
        if not self.enabled:
            return [None] * len(chunks)
        try:
            staticHeaders = {name: _sha256(Path(includeDir, name)) for name in _STATIC_HEADER_FILES}
            version = _compilerVersion(compilerPath)
            return [
                _computeChunkKey(path, job, includeDir, staticHeaders, version)
                for path, job in chunks
            ]
        except (OSError, CalledProcessError) as err:
            print1(f"# Helper kernel chunk cache not used, computing its keys failed: {err}")
            return [None] * len(chunks)

    def restore(self, key: str, objPath: Union[Path, str]) -> bool:
        """Copies the cached object of key to objPath. False (nothing written) on a miss."""
        entry = self.dir / key
        try:
            cached = entry / _OBJECT_FILE
            if cached.stat().st_size == 0:
                return False
            shutil.copyfile(cached, objPath)
            os.utime(entry)
            return True
        except OSError:
            Path(objPath).unlink(missing_ok=True)
            return False

    def store(self, key: str, objPath: Union[Path, str]):
        """Stores objPath as the object of key, unless an entry exists. Safe under concurrent
        stores of any key from any thread or process: each writes its own temporary directory
        and renames it into place; the first rename wins. Never raises: a chunk that cannot be
        stored is compiled again next time."""
        finalDir = self.dir / key
        if finalDir.exists():
            return
        tmpDir = None
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmpDir = Path(tempfile.mkdtemp(prefix=f".tmp_{key}_{os.getpid()}_", dir=self.dir))
            shutil.copyfile(objPath, tmpDir / _OBJECT_FILE)
            tmpDir.rename(finalDir)
        except OSError:
            # Another store of this key renamed first, or the cache is not writable.
            if tmpDir is not None:
                shutil.rmtree(tmpDir, ignore_errors=True)

    def discard(self, key: str):
        """Removes the entry of key (e.g. its object did not link). Renamed away first, so no
        other build reads it half removed."""
        try:
            staleDir = Path(tempfile.mkdtemp(prefix=f".tmp_stale_{key}_", dir=self.dir))
        except OSError:
            return
        try:
            (self.dir / key).rename(staleDir / key)
        except OSError:
            pass  # Already removed by another build.
        shutil.rmtree(staleDir, ignore_errors=True)
