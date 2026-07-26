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
from typing import Collection, List, Union, NamedTuple

from Tensile.Common import ensurePath, print2
from Tensile.Common.Architectures import gfxToIsa, isaToGfx
from ..SolutionStructs import Solution

from .Component import Assembler, Linker, Bundler

class AssemblyToolchain(NamedTuple):
   assembler: Assembler
   linker: Linker
   bundler: Bundler


def targetArchForIsa(isa, requestedArchs: Collection[str]) -> str:
    """Return the requested target ID corresponding to an ISA.

    A solution records only its numeric ISA, so converting it back with
    ``isaToGfx`` loses target features such as ``xnack+``. Preserve the full
    command-line target when there is one match. The assembly pipeline emits
    one object per ISA and cannot represent two feature variants of the same
    ISA in one invocation, so retain the established bare-target behavior when
    multiple variants are requested.
    """
    matches = sorted({arch for arch in requestedArchs if gfxToIsa(arch) == isa})
    if not matches:
        return isaToGfx(isa)
    if len(matches) > 1:
        return isaToGfx(isa)
    return matches[0]


def replaceAssemblyTarget(source: str, isa, targetArch: str) -> str:
    """Make the generated ``.amdgcn_target`` match the assembler target ID."""
    bareArch = isaToGfx(isa)
    if targetArch == bareArch:
        return source
    bareDirective = f'.amdgcn_target "amdgcn-amd-amdhsa--{bareArch}"'
    if source.count(bareDirective) != 1:
        raise RuntimeError(
            f"Expected one {bareArch} .amdgcn_target directive, found "
            f"{source.count(bareDirective)}"
        )
    return source.replace(
        bareDirective,
        f'.amdgcn_target "amdgcn-amd-amdhsa--{targetArch}"',
    )


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
      requestedArchs: Collection[str]=(),
      compress: bool=True,
    ):
    """Builds code object files from assembly files.

    Args:
        toolchain: The assembly toolchain object to use for building.
        kernels: A list of the kernel objects to build.
        writer: The KernelWriterAssembly object to use.
        destRoot: The library/ root directory. Per-arch outputs are written to
            destRoot/<gfx>/; isaToGfx() yields a bare gfx name already (no target
            features), so the routing here is the bare gfx.
        asmDir: The directory containing the assembly files.
        compress: Whether to compress the code object files.
    """

    extObj = ".o"
    extCo = ".co"
    extCoRaw = ".co.raw"

    destRoot = Path(destRoot)
    archKernelMap = collections.defaultdict(list)
    for k in kernels:
      archKernelMap[tuple(k['ISA'])].append(k)

    coFiles = []
    for arch, archKernels in archKernelMap.items():
      if len(archKernels) == 0:
        continue

      gfx = isaToGfx(arch)
      targetArch = targetArchForIsa(arch, requestedArchs)
      destDir = Path(ensurePath(destRoot / gfx))

      objectFiles = [str(asmDir / (k["BaseName"] + extObj)) for k in archKernels if 'codeObjectFile' not in k]
      coFileMap = collections.defaultdict(set)
      if len(objectFiles):
        coFileMap[asmDir / ("TensileLibrary_"+ gfx + extCoRaw)] = objectFiles
      for kernel in archKernels:
        coName = kernel.get("codeObjectFile", None)
        if coName:
          coFileMap[asmDir / (coName + extCoRaw)].add(str(asmDir / (kernel["BaseName"] + extObj)))

      for coFileRaw, objFiles in coFileMap.items():
        linker(objFiles, str(coFileRaw))
        coFile = destDir / coFileRaw.name.replace(extCoRaw, extCo)
        if compress:
          bundler.compress(str(coFileRaw), str(coFile), targetArch)
        else:
          shutil.move(coFileRaw, coFile)
        coFiles.append(coFile)

    return coFiles
