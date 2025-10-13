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

from Tensile.Common import print1, print2
from Tensile.Common.Architectures import isaToGfx
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
      destDir: Union[Path, str],
      asmDir: Union[Path, str],
      compress: bool=True,
      cofile_objects: dict=None,
    ):
    """Builds code object files from assembly files

    Args:
        linker: The linker object for combining .o files.
        bundler: The bundler object for compressing .co files.
        destDir: The destination directory for the code object files.
        asmDir: The directory containing the assembly files.
        compress: Whether to compress the code object files.
        cofile_objects: Mapping from ISA to dict of cofile_name to list of (sol.index, solution) tuples.
                       Format: {isa: {cofile_name: [(sol.index, solution), ...]}}
    """

    if cofile_objects is None:
        raise RuntimeError("cofile_objects must be provided to buildAssemblyCodeObjectFiles")

    extObj = ".o"
    extCo = ".co"
    extCoRaw = ".co.raw"
    asmDir = Path(asmDir)
    destDir = Path(destDir)

    coFiles = []

    lostSolutions = []
    # Build a global map of .o files to their reference counts and which .co files reference them
    objFileRefCount = collections.Counter()
    objFileToCoFiles = collections.defaultdict(list)  # Track which .co files reference each .o file
    for isa, cofile_map in cofile_objects.items():
        gfx = isaToGfx(isa)
        for cofile_name, sol_list in cofile_map.items():
            # Determine .co filename
            if cofile_name is None:
                coFileName = f"TensileLibrary_{gfx}.co"
            else:
                coFileName = f"{cofile_name}.co"

            # Deduplicate solutions in this .co file by basename
            seen_basenames = set()
            for sol_idx, sol in sol_list:
                basename = sol.getKernels()[0].get("BaseName", None)
                if basename is None:
                    basename = "MISSING!.o"
                    lostSolutions += [(sol_idx, cofile_name, sol)]
                if basename not in seen_basenames:
                    seen_basenames.add(basename)
                    objFilePath = asmDir / (basename + extObj)
                    objFileRefCount[str(objFilePath)] += 1
                    objFileToCoFiles[str(objFilePath)].append(coFileName)

    sharedObjFiles = {objFile: count for objFile, count in objFileRefCount.items() if count > 1}
    if sharedObjFiles:
        print1(f"Found {len(sharedObjFiles)} .o files shared across multiple code objects:")
        for objFile in list(sharedObjFiles.keys())[:10]:  # Show first 10 examples
            coFiles = objFileToCoFiles[objFile]
            basename = Path(objFile).name
            print1(f"  {basename} -> {', '.join(coFiles)}")
    
    if lostSolutions:
        for a, b, c in lostSolutions[:10]:
            print(a, b, c)
        raise Exception("Some solutions are missing a BaseName. First 10 printed.")

    # Now process each ISA and .co file
    for isa, cofile_map in cofile_objects.items():
        gfx = isaToGfx(isa)

        for cofile_name, sol_list in cofile_map.items():
            # Sort by solution index to ensure correct kernel ordering
            sol_list.sort(key=lambda x: x[0])

            # Build list of .o files, deduplicating by basename within this .co file
            objFiles = []
            seen_basenames = set()
            for sol_idx, sol in sol_list:
                basename = sol.getKernels()[0].get("BaseName", "MISSING!.o")
                if basename not in seen_basenames:
                    seen_basenames.add(basename)
                    objFilePath = asmDir / (basename + extObj)
                    objFiles.append(str(objFilePath))

            # Determine output filename
            if cofile_name is None:
                coFileRaw = asmDir / f"TensileLibrary_{gfx}{extCoRaw}"
            else:
                coFileRaw = asmDir / f"{cofile_name}{extCoRaw}"

            # Link the .o files into a .co file
            linker(objFiles, str(coFileRaw))

            # Delete .o files after linking once usage count reaches 0
            for objFile in objFiles:
                objFileRefCount[objFile] -= 1
                if objFileRefCount[objFile] == 0:
                    Path(objFile).unlink()

            # Compress/move the .co file to destination
            coFile = destDir / coFileRaw.name.replace(extCoRaw, extCo)
            if compress:
                bundler.compress(str(coFileRaw), str(coFile), gfx)
            else:
                shutil.move(coFileRaw, coFile)
            coFiles.append(coFile)

    return coFiles
