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

import collections
import math
import shutil
import subprocess

from pathlib import Path
from typing import Dict, List, Optional, Union, NamedTuple

from Tensile.Common import ensurePath, print2
from Tensile.Common.Architectures import isaToGfx
from Tensile.Common.Types import IsaVersion
from ..SolutionStructs import Solution

from .Component import Assembler, Linker, Bundler

class AssemblyToolchain(NamedTuple):
   assembler: Assembler
   linker: Linker
   bundler: Bundler


def makeAssemblyToolchain(assembler_path, bundler_path, co_version, build_id_kind="sha1", debug=False):
   compiler = Assembler(assembler_path, co_version, debug)
   linker = Linker(assembler_path, build_id_kind)
   bundler = Bundler(bundler_path)
   return AssemblyToolchain(compiler, linker, bundler)


def buildAssemblyCodeObjectFiles(
      linker: Linker,
      bundler: Bundler,
      kernels: List[Solution],
      destRoot: Union[Path, str],
      asmDir: Union[Path, str],
      compress: bool=True,
      archNames: Optional[Dict[IsaVersion, str]]=None,
    ):
    """Builds code object files from assembly files.

    Args:
        toolchain: The assembly toolchain object to use for building.
        kernels: A list of the kernel objects to build.
        writer: The KernelWriterAssembly object to use.
        destRoot: The library/ root directory. Per-arch outputs are written to
            destRoot/<arch>/.
        asmDir: The directory containing the assembly files.
        compress: Whether to compress the code object files.
        archNames: ISA version -> the architecture that ISA is being built as,
            from archNamesByIsa. It names the compiler target, the code object,
            and the output subtree; two architectures can share an ISA, so none
            of the three can be derived from the ISA the kernels carry.
    """

    extObj = ".o"
    extCo = ".co"
    extCoRaw = ".co.raw"

    archNames = archNames or {}
    destRoot = Path(destRoot)
    archKernelMap = collections.defaultdict(list)
    for k in kernels:
      archKernelMap[tuple(k['ISA'])].append(k)

    coFiles = []
    for arch, archKernels in archKernelMap.items():
      if len(archKernels) == 0:
        continue

      name = archNames.get(arch) or isaToGfx(arch)
      destDir = Path(ensurePath(destRoot / name))

      objectFiles = [str(asmDir / (k["BaseName"] + extObj)) for k in archKernels if 'codeObjectFile' not in k]
      coFileMap = collections.defaultdict(set)
      if len(objectFiles):
        coFileMap[asmDir / ("TensileLibrary_"+ name + extCoRaw)] = objectFiles
      for kernel in archKernels:
        coName = kernel.get("codeObjectFile", None)
        if coName:
          coFileMap[asmDir / (coName + extCoRaw)].add(str(asmDir / (kernel["BaseName"] + extObj)))

      for coFileRaw, objFiles in coFileMap.items():
        # Canonicalize both the default-list and explicit-set linker input paths.
        linker(sorted(objFiles), str(coFileRaw))
        coFile = destDir / coFileRaw.name.replace(extCoRaw, extCo)
        if compress:
          bundler.compress(str(coFileRaw), str(coFile), name)
        else:
          shutil.move(coFileRaw, coFile)
        coFiles.append(coFile)

    return coFiles
