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
from typing import List, Union, NamedTuple

from Tensile.Common import print2
from Tensile.Common.Architectures import isaToGfx
from ..SolutionStructs import Solution

from .Component import Assembler, Linker, Bundler

class AssemblyToolchain(NamedTuple):
   assembler: Assembler
   linker: Linker
   bundler: Bundler


def makeAssemblyToolchain(assembler_path, bundler_path, co_version, build_id_kind="sha1"):
   compiler = Assembler(assembler_path, co_version)
   linker = Linker(assembler_path, build_id_kind)
   bundler = Bundler(bundler_path)
   return AssemblyToolchain(compiler, linker, bundler)


def _batchObjectFiles(ldPath: str, objFiles: List[str], coPathDest: Union[Path, str], maxObjFiles: int=10000) -> List[str]:
    numObjFiles = len(objFiles)

    if numObjFiles <= maxObjFiles:
      return objFiles

    batchedObjFiles = [objFiles[i:i+maxObjFiles] for i in range(0, numObjFiles, maxObjFiles)]
    numBatches = int(math.ceil(numObjFiles / maxObjFiles))

    newObjFiles = [str(coPathDest) + "." + str(i) for i in range(0, numBatches)]
    newObjFilesOutput = []

    for batch, filename in zip(batchedObjFiles, newObjFiles):
      if len(batch) > 1:
        args = [ldPath, "-r"] + batch + [ "-o", filename]
        print2(f"Linking object files into fewer object files: {' '.join(args)}")
        subprocess.check_call(args)
        newObjFilesOutput.append(filename)
      else:
        newObjFilesOutput.append(batchedObjFiles[0])

    return newObjFilesOutput


def buildAssemblyCodeObjectFiles(
      linker: Linker,
      bundler: Bundler,
      ldPath: str,
      kernels: List[Solution],
      destDir: Union[Path, str],
      asmDir: Union[Path, str],
      compress: bool=True,
    ):
    """Builds code object files from assembly files

    Args:
        toolchain: The assembly toolchain object to use for building.
        kernels: A list of the kernel objects to build.
        writer: The KernelWriterAssembly object to use.
        destDir: The destination directory for the code object files.
        asmDir: The directory containing the assembly files.
        compress: Whether to compress the code object files.
    """

    extObj = ".o"
    extCo = ".co"
    extCoRaw = ".co.raw"

    archKernelMap = collections.defaultdict(list)
    for k in kernels:
      archKernelMap[tuple(k['ISA'])].append(k)

    coFiles = []
    for arch, archKernels in archKernelMap.items():
      if len(archKernels) == 0:
        continue

      gfx = isaToGfx(arch)

      objectFiles = [str(asmDir / (k["BaseName"] + extObj)) for k in archKernels if 'codeObjectFile' not in k]
      coFileMap = collections.defaultdict(list)
      if len(objectFiles):
        coFileMap[asmDir / ("TensileLibrary_"+ gfx + extCoRaw)] = objectFiles
      for kernel in archKernels:
        coName = kernel.get("codeObjectFile", None)
        if coName:
          coFileMap[asmDir / (coName + extCoRaw)].append(str(asmDir / (kernel["BaseName"] + extObj)))

      for coFileRaw, objFiles in coFileMap.items():
        objFiles = _batchObjectFiles(ldPath, objFiles, coFileRaw)
        linker(objFiles, str(coFileRaw))
        coFile = destDir / coFileRaw.name.replace(extCoRaw, extCo)
        if compress:
          bundler.compress(str(coFileRaw), str(coFile), gfx)
        else:
          shutil.move(coFileRaw, coFile)
        coFiles.append(coFile)

    return coFiles
