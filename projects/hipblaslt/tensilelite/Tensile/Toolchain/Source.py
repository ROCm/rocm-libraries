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

from .Component import Compiler, Bundler

class SourceToolchain(NamedTuple):
   compiler: Compiler
   bundler: Bundler


def makeSourceToolchain(compiler_path, bundler_path, asan_build=False, build_id_kind="sha1", save_temps=False):
   compiler = Compiler(compiler_path, build_id_kind, asan_build, save_temps)
   bundler = Bundler(bundler_path)
   return SourceToolchain(compiler, bundler)


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

    tmpObjDir = Path(ensurePath(tmpObjDir))
    destDir = Path(ensurePath(destDir))
    kernelPath = Path(kernelPath)

    objFilename = kernelPath.stem + '.o'
    coPathsRaw = []
    coPaths= []

    objPath = str(tmpObjDir / objFilename)
    compiler(str(includeDir), cmdlineArchs, str(kernelPath), objPath)

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

    for src, dst in zip(coPathsRaw, coPaths):
        shutil.move(src, dst)

    stop = timer()
    print1(f"buildSourceCodeObjectFile time (s): {(stop-start):3.2f}")

    return coPaths
