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
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Union

from .. import Utils
from ..Common import ensurePath, gfxName, globalParameters, printWarning, tPrint
from ..KernelWriterAssembly import KernelWriterAssembly
from ..SolutionStructs import Solution
from .SharedCommands import compressCodeObject


def _linkIntoCodeObject(
    objFiles: List[str],
    coPathDest: Union[Path, str],
    writer: KernelWriterAssembly,
    maxLineLength: int,
):
    """Links object files into a code object file.

    Args:
        objectFiles: A list of object files to be linked.
        coPathDest: The destination path for the code object file.
        writer: An instance of KernelWriterAssembly to get link arguments.
        maxLineLength: The maximum command line length. On Windows, this is nominally 8191,
            on posix platforms this number can be computed with ``$ getconf ARG_MAX``.

    Raises:
        RuntimeError: If linker invocation fails.
    """
    args = writer.getLinkCodeObjectArgs(objFiles, str(coPathDest))
    # It is possible for the list of `objFiles` to exceed the command line limit
    # LLVM allows the provision of compiler arguments via a "response file" (`rf` below)
    # Reference: https://llvm.org/docs/CommandLine.html#response-files
    lineLength = sum(len(arg) for arg in args) + len(args) - 1  # Account for spaces
    if lineLength > maxLineLength:
        with tempfile.NamedTemporaryFile(mode="wt", delete=False) as rf:
            # Use repr on Windows to a get raw string with non-escaped path separators
            strArgs = " ".join(map(repr, objFiles)) if os.name == "nt" else " ".join(objFiles)
            rf.write(strArgs)
            rf.flush()
            args = writer.getLinkCodeObjectArgs([f"@{rf.name}"], str(coPathDest))

    tPrint(2, "Linking objects into co files: " + " ".join(args))
    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT)
        tPrint(2, f"Output: {out}")
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"Error linking object files in to code object: {err.output}\nFailed command: {' '.join(args)}"
        )


def buildAssemblyCodeObjectFiles(
    bundler: str,
    kernels: List[Solution],
    writer: KernelWriterAssembly,
    outputPath: Union[Path, str],
    removeTemporaries: bool,
):

    countAsmKernels = lambda kernels: sum(k["KernelLanguage"] == "Assembly" for k in kernels)

    extObj = ".o"
    extCo = ".co"
    extCoRaw = ".co.raw"

    destDir = Path(ensurePath(os.path.join(outputPath, "library")))
    asmDir = Path(writer.getAssemblyDirectory())

    maxLineLength = (
        8191 if os.name == "nt" else int(subprocess.check_output(["getconf", "ARG_MAX"]).strip())
    )

    assemblyKernels = [k for k in kernels if k["KernelLanguage"] == "Assembly"]
    if len(assemblyKernels) == 0:
        return []

    archKernelMap = collections.defaultdict(list)
    for k in assemblyKernels:
        archKernelMap[tuple(k["ISA"])].append(k)

    coFiles = []
    for arch, archKernels in archKernelMap.items():
        gfx = gfxName(arch)
        objectFiles = [
            str(asmDir / (writer.getKernelFileBase(k) + extObj))
            for k in archKernels
            if "codeObjectFile" not in k
        ]

        if countAsmKernels(archKernels) == 0:
            continue

        if (
            globalParameters["MergeFiles"]
            or globalParameters["NumMergedFiles"] > 1
            or globalParameters["LazyLibraryLoading"]
        ):

            # Group kernels from placeholder libraries
            coFileMap = collections.defaultdict(list)

            if len(objectFiles):
                coFileMap[asmDir / ("TensileLibrary_" + gfx + extCoRaw)] = objectFiles

            for kernel in archKernels:
                coName = kernel.get("codeObjectFile", None)
                if coName:
                    coFileMap[asmDir / (coName + extCoRaw)].append(
                        str(asmDir / (writer.getKernelFileBase(kernel) + extObj))
                    )

            for coFileRaw, objFiles in coFileMap.items():

                _linkIntoCodeObject(objFiles, coFileRaw, writer, maxLineLength)
                coFile = destDir / coFileRaw.name.replace(extCoRaw, extCo)
                compressCodeObject(coFileRaw, coFile, gfx, bundler)
                coFiles.append(coFile)

        else:
            # no mergefiles
            assemblyKernelNames = [writer.getKernelFileBase(k) for k in archKernels]
            origCOFiles = [os.path.join(asmDir, k + ".co") for k in assemblyKernelNames]
            newCOFiles = [os.path.join(destDir, k + "_" + gfx + ".co") for k in assemblyKernelNames]

            for src, dst in (
                zip(origCOFiles, newCOFiles)
                if globalParameters["PrintLevel"] == 0
                else Utils.tqdm(zip(origCOFiles, newCOFiles), desc="Relocating code objects")
            ):
                shutil.copyfile(src, dst)
            coFiles += newCOFiles
            printWarning("Code object files are not compressed in `--no-merge-files` build mode.")

    return coFiles
