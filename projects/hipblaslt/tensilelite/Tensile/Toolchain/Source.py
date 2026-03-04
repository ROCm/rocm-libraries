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
import re
import shutil

from pathlib import Path
from timeit import default_timer as timer
from typing import List, Union, NamedTuple

from ..Common import print1, ensurePath
from ..Common.TimingInstrumentation import timing_context

from .Component import Compiler, Bundler

class SourceToolchain(NamedTuple):
   compiler: Compiler
   bundler: Bundler


def makeSourceToolchain(compiler_path, bundler_path, asan_build=False, build_id_kind="sha1", save_temps=False):
   compiler = Compiler(compiler_path, build_id_kind, asan_build, save_temps)
   bundler = Bundler(bundler_path)
   return SourceToolchain(compiler, bundler)


_HELPER_CACHE_DIR_DEFAULT = Path.home() / ".tensile" / "helper_cache"

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
    """Check if a valid cache entry exists. Returns list of .hsaco Paths or None."""
    entryDir = Path(cacheDir) / cacheKey
    if not entryDir.is_dir():
        return None
    hsacoFiles = list(entryDir.glob("*.hsaco"))
    if not hsacoFiles or any(f.stat().st_size == 0 for f in hsacoFiles):
        return None
    return hsacoFiles


def _populateCache(cacheDir, cacheKey, hsacoFiles):
    """Atomically populate a cache entry. Safe under concurrent writes."""
    cacheDir = Path(cacheDir)
    finalDir = cacheDir / cacheKey
    if finalDir.exists():
        return

    tmpDir = cacheDir / f".tmp_{cacheKey}_{os.getpid()}"
    tmpDir.mkdir(parents=True, exist_ok=True)
    for f in hsacoFiles:
        shutil.copy2(Path(f), tmpDir / Path(f).name)

    try:
        tmpDir.rename(finalDir)
    except OSError:
        shutil.rmtree(tmpDir, ignore_errors=True)


def _computeSourceCodeObjectFilename(target: str, base: str, buildPath: Union[Path, str], arch: str) -> Union[Path, None]:
    """Generates a code object file path using the target, base, and build path.

    Args:
        target: The target triple.
        base: The base name for the output file (name without extension).
        buildPath: The build directory path.

    Returns:
        Path to the code object file.
    """
    coPath = None
    buildPath = Path(buildPath)
    if "TensileLibrary" in base and "fallback" in base:
        coPath = buildPath / "{0}_{1}.hsaco.raw".format(base, arch)
    elif "TensileLibrary" in base:
        variant = [t for t in ["", "xnack-", "xnack+"] if t in target][-1]
        baseVariant = base + "-" + variant if variant else base
        if arch in baseVariant:
            coPath = buildPath / (baseVariant + ".hsaco.raw")
    else:
        coPath= buildPath / "{0}.so-000-{1}.hsaco.raw".format(base, arch)

    return coPath


def buildSourceCodeObjectFiles(
        compiler: Compiler,
        bundler: Bundler,
        destDir: Union[Path, str],
        tmpObjDir: Union[Path, str],
        includeDir: Union[Path, str],
        kernelPath: Union[Path, str],
        cmdlineArchs: List[str]
    ) -> List[str]:
    """Compiles a HIP source code file into a code object file.

    Args:
        toolchain: The source toolchain.
        destDir: The destination directory where HSA code object files are placed.
        tmpObjDir: The directory where HIP source object files are created.
        includeDir: The include directory path.
        kernelPath: The path to the kernel source file.

    Returns:
        List of paths to the created code objects.
    """
    start = timer()

    cacheEnabled = os.environ.get("TENSILE_DISABLE_HELPER_CACHE", "").upper() \
                   not in ("1", "YES", "ON", "TRUE")
    cacheDir = Path(os.environ.get("TENSILE_HELPER_CACHE_DIR",
                                   str(_HELPER_CACHE_DIR_DEFAULT)))
    cacheKey = None

    with timing_context("python_kernel_build_src_co.setup"):
        tmpObjDir = Path(ensurePath(tmpObjDir))
        destDir = Path(ensurePath(destDir))
        kernelPath = Path(kernelPath)

        objFilename = kernelPath.stem + '.o'
        coPathsRaw = []
        coPaths= []

    if cacheEnabled:
        with timing_context("python_kernel_build_src_co.cache_check"):
            cacheKey = _computeCacheKey(kernelPath, includeDir, cmdlineArchs, compiler)
            cachedFiles = _checkCache(cacheDir, cacheKey)

        if cachedFiles:
            with timing_context("python_kernel_build_src_co.cache_hit"):
                for f in cachedFiles:
                    dst = destDir / f.name
                    shutil.copy2(f, dst)
                    coPaths.append(str(dst))
            stop = timer()
            print1(f"buildSourceCodeObjectFile time (s): {(stop-start):3.2f}  [cache hit]")
            return coPaths
        else:
            print1(f"# Helper kernel cache MISS ({cacheKey[:12]}...)")

    objPath = str(tmpObjDir / objFilename)
    with timing_context("python_kernel_build_src_co.compile"):
        compiler(str(includeDir), cmdlineArchs, str(kernelPath), objPath)

    with timing_context("python_kernel_build_src_co.unbundle"):
        for target in bundler.targets(objPath):
          match = re.search("gfx.*$", target)
          if match:
            arch = re.sub(":", "-", match.group())
            coPathRaw = _computeSourceCodeObjectFilename(target, kernelPath.stem, tmpObjDir, arch)
            if not coPathRaw: continue
            bundler(target, objPath, str(coPathRaw))

            coPath = str(destDir / coPathRaw.stem)
            coPathsRaw.append(coPathRaw)
            coPaths.append(coPath)

    with timing_context("python_kernel_build_src_co.move"):
        for src, dst in zip(coPathsRaw, coPaths):
            shutil.move(src, dst)

    if cacheEnabled and cacheKey:
        with timing_context("python_kernel_build_src_co.cache_populate"):
            cacheDir.mkdir(parents=True, exist_ok=True)
            _populateCache(cacheDir, cacheKey, [Path(p) for p in coPaths])

    stop = timer()
    print1(f"buildSourceCodeObjectFile time (s): {(stop-start):3.2f}")

    return coPaths
