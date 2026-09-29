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

import re
import shutil

from pathlib import Path
from timeit import default_timer as timer
from typing import List, Union, NamedTuple

from ..Common import print1, ensurePath
from ..Common.Architectures import toolchainTargetOf
from ..Common.TimingInstrumentation import timing_context

from .Component import Compiler, Bundler
from .HelperKernelCache import HelperKernelCache

class SourceToolchain(NamedTuple):
   compiler: Compiler
   bundler: Bundler


def makeSourceToolchain(compiler_path, bundler_path, asan_build=False, build_id_kind="sha1", save_temps=False):
   compiler = Compiler(compiler_path, build_id_kind, asan_build, save_temps)
   bundler = Bundler(bundler_path)
   return SourceToolchain(compiler, bundler)


def _archNamesFromBundlerTarget(rawArch: str):
    """Split a bundler arch token into (filenameArch, baseArch).

    The bundler emits gcn arch tokens of the form ``gfx942:sramecc+:xnack+``.
    Per-base layout requires:

      * The directory uses only the base arch (`gfx942`) so every target-feature
        variant co-locates in one subdir. Splitting at the first ':' is the
        single source of truth — callers that strip with `split("-xnack")[0]`
        AFTER ':' -> '-' conversion leave `gfx942-sramecc+` as the directory
        and silently place files in the wrong subdir.
      * The filename keeps only the base arch and the xnack feature
        (`gfx942`, `gfx942-xnack+`, `gfx942-xnack-`). The runtime helper-kernel
        loader (HipSolutionAdapter / tensile_host) probes only
        ``{"", "-xnack-", "-xnack+"}`` appended to ``<base>``; sramecc and any
        other feature are never part of the loaded name, so keeping them here
        produces a file the runtime can never find. xnack+/xnack- still get
        distinct filenames, so they do not collide.

    Returns ``(filenameArch, baseArch)`` — both extracted from the same source
    token so they cannot drift apart.
    """
    baseArch     = rawArch.split(":", 1)[0]
    xnack        = next((f for f in rawArch.split(":")[1:] if f.startswith("xnack")), None)
    filenameArch = baseArch + ("-" + xnack if xnack else "")
    return filenameArch, baseArch


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
        destRoot: Union[Path, str],
        tmpObjDir: Union[Path, str],
        includeDir: Union[Path, str],
        kernelPath: Union[Path, str],
        cmdlineArchs: List[str],
    ) -> List[str]:
    """Compiles a HIP source code file into a code object file.

    Args:
        toolchain: The source toolchain.
        destRoot: The library/ root directory. Per-arch outputs are written to
            destRoot/<arch>/, named for the target the bundler reports back;
            target features (xnack+/xnack-) are stripped from the directory path
            and survive only in the filename suffix.
        tmpObjDir: The directory where HIP source object files are created.
        includeDir: The include directory path.
        kernelPath: The path to the kernel source file.

    Returns:
        List of paths to the created code objects.
    """
    start = timer()
    cache = HelperKernelCache()

    with timing_context("python_kernel_build_src_co.setup"):
        tmpObjDir = Path(ensurePath(tmpObjDir))
        destRoot = Path(ensurePath(destRoot))
        kernelPath = Path(kernelPath)

        objFilename = kernelPath.stem + '.o'
        coPathsRaw = []
        coPaths= []

    # Try to restore pre-built code objects from the helper-kernel cache.
    # On a hit we skip compilation/unbundling entirely and return early.
    # The cache restore routes each file to its per-base subdir under destRoot.
    with timing_context("python_kernel_build_src_co.cache_check"):
        hit, coPaths = cache.restore(kernelPath, includeDir, cmdlineArchs, compiler, destRoot)
    if hit:
        stop = timer()
        print1(f"buildSourceCodeObjectFile time (s): {(stop-start):3.2f}  [cache hit]")
        return coPaths

    objPath = str(tmpObjDir / objFilename)
    with timing_context("python_kernel_build_src_co.compile"):
        compiler(str(includeDir), cmdlineArchs, str(kernelPath), objPath)

    # The architectures asked for, keyed by the target they were compiled as.
    #
    # A stepping compiles as the architecture it steps (toolchainTargetOf), so
    # the bundler reports the base name back and everything derived from that
    # token loses the stepping. The assembly path does not have this problem
    # because it is handed archNames outright; here the destination came from
    # the bundler, so a gfx1250-strict build wrote its source kernels into
    # gfx1250/ and shipped a strict subtree with no Kernels.so of its own --
    # which the runtime then selected on an A0 and could not load.
    #
    # Keyed rather than zipped because one compile can serve several requested
    # names: asking for an architecture and its stepping together yields one
    # bundle entry and two subtrees, each getting its own copy.
    requestedByTarget = {}
    for requested in cmdlineArchs:
        requestedByTarget.setdefault(toolchainTargetOf(requested), []).append(requested)

    with timing_context("python_kernel_build_src_co.unbundle"):
        for target in bundler.targets(objPath):
          match = re.search("gfx.*$", target)
          if match:
            arch, baseArch = _archNamesFromBundlerTarget(match.group())
            # Fall back to the reported name where nothing asked for it, which
            # keeps every architecture without a stepping exactly as it was.
            for requestedArch in requestedByTarget.get(baseArch, [baseArch]):
                # The directory is the requested name; the filename is it plus
                # whatever xnack variant the bundler token carried, since that
                # is the pair the runtime helper-kernel loader probes for.
                fileArch = requestedArch + arch[len(baseArch):]
                coPathRaw = _computeSourceCodeObjectFilename(
                    target, kernelPath.stem, tmpObjDir, fileArch
                )
                if not coPathRaw: continue
                bundler(target, objPath, str(coPathRaw))

                destDir = Path(ensurePath(destRoot / requestedArch))
                coPath = str(destDir / coPathRaw.stem)
                coPathsRaw.append(coPathRaw)
                coPaths.append(coPath)

    with timing_context("python_kernel_build_src_co.move"):
        for src, dst in zip(coPathsRaw, coPaths):
            shutil.move(src, dst)

    # Save the freshly built code objects into the cache so subsequent
    # builds with the same inputs can skip recompilation.
    with timing_context("python_kernel_build_src_co.cache_populate"):
        cache.store(coPaths)

    stop = timer()
    print1(f"buildSourceCodeObjectFile time (s): {(stop-start):3.2f}")

    return coPaths
