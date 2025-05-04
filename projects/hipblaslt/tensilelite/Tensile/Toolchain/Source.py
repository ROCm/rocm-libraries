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

import os
import re
import shutil

from pathlib import Path
from timeit import default_timer as timer
from typing import List, Union, NamedTuple

from ..Common import print1, ensurePath

from .Component import Compiler, Bundler, RocObjLs, RocObjExtract

class SourceToolchain(NamedTuple):
   compiler: Compiler
   bundler: Bundler
   rocObjLs: RocObjLs
   rocObjExtract: RocObjExtract


def makeSourceToolchain(compiler_path, bundler_path, ls_path, extract_path, cpu_threads, asan_build=False, build_id_kind="sha1", save_temps=False):
   compiler = Compiler(compiler_path, build_id_kind, cpu_threads, asan_build, save_temps)
   bundler = Bundler(bundler_path)
   ls = RocObjLs(ls_path)
   extract = RocObjExtract(extract_path)
   return SourceToolchain(compiler, bundler, ls, extract)


def buildSourceCodeObjectFiles(
        toolchain: SourceToolchain,
        destPath: Path,
        sharedObjPath: Union[Path, str]
    ):
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

    for target, filename in toolchain.rocObjLs(sharedObjPath):
      match = re.search("gfx.*$", target)
      if match:
        print(f"Generating Kernels.so for {target.split('-')[-1]}")
        arch = re.sub(":", "-", match.group())
        toolchain.rocObjExtract(filename)
        src = str(Path(sharedObjPath).parent / (str(Path(filename).name).replace("#","-").replace("=","").replace("&","-") + ".co"))
        dst = str(destPath / f"Kernels.so-000-{arch}.hsaco")
        shutil.move(src, dst)
