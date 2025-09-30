################################################################################
#
# Copyright (C) 2016-2024 Advanced Micro Devices, Inc. All rights reserved.
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

# This script only gets called by CMake

if __name__ == "__main__":
    print(
        "This file can no longer be run as a script.  Run 'Tensile/bin/TensileCreateLibrary' instead."
    )
    exit(1)

import collections
import itertools
import os
import re
import shutil
import sys
import time
import warnings
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from . import Common, EmbeddedData, LibraryIO, Utils
from .BuildCommands import AssemblyCommands, SourceCommands
from .Common import (
    HR,
    CHeader,
    CMakeHeader,
    assignGlobalParameters,
    ensurePath,
    getArchitectureName,
    gfxName,
    globalParameters,
    printExit,
    printWarning,
    splitArchs,
    tPrint,
)
from .KernelWriterAssembly import KernelWriterAssembly
from .KernelWriterBase import KernelWriterBase
from .KernelWriterSource import KernelWriterSource
from .SolutionLibrary import MasterSolutionLibrary
from .SolutionStructs import Solution
from .TensileCreateLib.KernelFileContext import KernelFileContextManager
from .TensileCreateLib.ParseArguments import parseArguments
from .Utilities.Profile import profile
from .Utilities.String import splitDelimitedString
from .Utilities.toFile import toFile

TENSILE_MANIFEST_FILENAME = "TensileManifest.txt"
TENSILE_LIBRARY_DIR = "library"

ProcessedKernelResult = Tuple[int, str, str, str, Optional[str]]
ProcessedKernelLookup = Dict[str, List[ProcessedKernelResult]]


def processKernelSource(kernel, kernelWriterSource, kernelWriterAssembly) -> ProcessedKernelResult:
    """Generate source for a single kernel.
    Returns (error, source, header, kernelName).
    """
    try:
        kernelWriter = (
            kernelWriterSource if kernel["KernelLanguage"] == "Source" else kernelWriterAssembly
        )
        # get kernel name
        kernelName = kernelWriter.getKernelFileBase(kernel)
        (err, src) = kernelWriter.getSourceFileString(kernel)
        header = kernelWriter.getHeaderFileString(kernel)
        # will be put in Kernels.h/cpp if None
        filename = kernel.get("codeObjectFile", None)

    except RuntimeError:
        printWarning(
            "Gracefully handling unknown runtime error when generating kernel: %s"
            % kernel["KernelName"]
        )
        return (1, "", "", kernelName, None)

    return (err, src, header, kernelName, filename)


def prepAsm(
    kernelWriterAssembly: KernelWriterAssembly,
    isLinux: bool,
    buildPath: Path,
    isa: Tuple[int, int, int],
    printLevel: int,
):
    """Create and prepare the assembly directory; called ONCE per output directory.

    This function is called once per output directory. It creates a directory
    "assembly" under the provided **buildPath**, and generates a bash script for
    compiling object files into code object files.

    Args:
        kernelWriterAssembly: Assembly writer object.
        buildPath: Path to directory where assembly files will be written.
    """
    asmPath = buildPath / "assembly"
    asmPath.mkdir(exist_ok=True)

    assemblerFileName = asmPath / f"asm-new.{'sh' if isLinux else 'bat'}"

    with open(assemblerFileName, "w") as assemblerFile:
        if isLinux:
            assemblerFile.write("#!/bin/sh{log}\n".format(log=" -x" if printLevel >= 3 else ""))
            assemblerFile.write("# usage: asm-new.sh kernelName(no extension) [--wave32]\n")

            assemblerFile.write("f=$1\n")
            assemblerFile.write("shift\n")
            assemblerFile.write('if [ ! -z "$1" ] && [ "$1" = "--wave32" ]; then\n')
            assemblerFile.write("    wave=32\n")
            assemblerFile.write("    shift\n")
            assemblerFile.write("else\n")
            assemblerFile.write("    wave=64\n")
            assemblerFile.write("fi\n")

            assemblerFile.write("h={gfxName}\n".format(gfxName=Common.gfxName(isa)))

            cArgs32 = kernelWriterAssembly.getCompileArgs("$f.s", "$f.o", isa=isa, wavefrontSize=32)
            cArgs64 = kernelWriterAssembly.getCompileArgs("$f.s", "$f.o", isa=isa, wavefrontSize=64)
            lArgs = kernelWriterAssembly.getLinkCodeObjectArgs(["$f.o"], "$f.co")

            assemblerFile.write("if [ $wave -eq 32 ]; then\n")
            assemblerFile.write(" ".join(cArgs32) + "\n")
            assemblerFile.write("else\n")
            assemblerFile.write(" ".join(cArgs64) + "\n")
            assemblerFile.write("fi\n")

            assemblerFile.write(" ".join(lArgs) + "\n")

            assemblerFile.write("cp $f.co ../../../library/${f}_$h.co\n")
            assemblerFile.write("mkdir -p ../../../asm_backup && ")
            assemblerFile.write("cp $f.s ../../../asm_backup/$f.s\n")
        else:
            assemblerFile.write("@echo off\n")
            assemblerFile.write("set f=%1\n\n")
            assemblerFile.write("set arg2=--wave64\n")
            assemblerFile.write("if [%2] NEQ [] set arg2=%2\n\n")
            assemblerFile.write("set /A wave=64\n")
            assemblerFile.write("if %arg2% EQU --wave32 set /A wave=32\n\n")

            assemblerFile.write("set h={gfxName}\n".format(gfxName=Common.gfxName(isa)))

            cArgs32 = " ".join(
                kernelWriterAssembly.getCompileArgs("%f%.s", "%f%.o", isa=isa, wavefrontSize=32)
            )
            cArgs64 = " ".join(
                kernelWriterAssembly.getCompileArgs("%f%.s", "%f%.o", isa=isa, wavefrontSize=64)
            )
            lArgs = " ".join(kernelWriterAssembly.getLinkCodeObjectArgs(["%f%.o"], "%f%.co"))

            assemblerFile.write(f"if %wave% == 32 ({cArgs32}) else ({cArgs64})\n")
            assemblerFile.write(f"{lArgs}\n")
            assemblerFile.write("copy %f%.co ..\\..\\..\\library\\%f%_%h%.co\n")
    os.chmod(assemblerFileName, 0o777)


def collectFilesToWrite(
    results: List[ProcessedKernelResult],
    outputPath: Path,
    lazyLoading: bool,
    mergeFiles: bool,
    numMergedFiles: int,
) -> ProcessedKernelLookup:
    """Collects and organizes kernel files to be written based on the provided results.

    Args:
        results: A list of processed kernel results, each containing
            error status, source code, header, kernel name, and filename.
        outputPath: The path where the output files should be written.
        lazyLoading: Flag indicating whether lazy loading is enabled.
        mergeFiles: Flag indicating whether kernel files should be merged.
        numMergedFiles: The number of files to merge into if merging is enabled.

    Returns:
        A tuple containing:
            - A dictionary mapping file paths to kernel data:
                (error status, source code, header code, kernel name).
            - A dictionary mapping kernel names to an error status.
    """
    pathJoin = lambda x: os.path.join(os.path.normcase(outputPath), x)

    filesToWrite = collections.defaultdict(list)
    validKernelCount = 0

    for err, src, header, kernelName, filename in results:
        if not src.strip():
            continue

        kernPath = pathJoin(kernelName)
        if filename:
            kernPath = pathJoin(filename)
        elif mergeFiles:
            suffix = str(validKernelCount % numMergedFiles) if numMergedFiles > 1 else ""
            kernPath = pathJoin(f"Kernels{suffix}")

        filesToWrite[kernPath].append((err, src, header, kernelName))
        validKernelCount += 1

    # Ensure there's at least one kernel file for helper kernels
    if lazyLoading or (mergeFiles and validKernelCount == 0):
        kernelSuffix = "0" if numMergedFiles > 1 else ""
        filesToWrite[pathJoin(f"Kernels{kernelSuffix}")] = []

    return filesToWrite


def generateKernelSourceAndHeaderFiles(
    filesToWrite: ProcessedKernelLookup,
) -> List[str]:
    """Generates kernel source and header files.

    Arguments:
        fileToWrite: A dictionary mapping file paths to kernel data:
            (error status, source code, header code, kernel name).

    Returns:
        A list containing source kernel filenames, and a dictionary with kernels that
            encountered build errors.
    """

    def writeHeaderPreface(hdrFile):
        hdrFile.write(CHeader)
        hdrFile.write("#pragma once\n")
        if globalParameters["RuntimeLanguage"] == "HIP":
            hdrFile.write("#include <hip/hip_runtime.h>\n")
            hdrFile.write("#include <hip/hip_ext.h>\n\n")
        hdrFile.write('#include "KernelHeader.h"\n\n')

    def writeSourcePreface(srcFile, filename):
        srcFile.write(CHeader)
        srcFile.write(f'#include "{filename}.h"\n')

    # Write kernel data to files
    for filename, kernelList in filesToWrite.items():
        # fmt: off
        with open(f"{filename}.h", "w", encoding="utf-8") as hdrFile, \
             open(f"{filename}.cpp", "w", encoding="utf-8") as srcFile:
        # fmt: on
            writeHeaderPreface(hdrFile)
            writeSourcePreface(srcFile, filename)
            for _, src, header, _ in kernelList:
                srcFile.write(src)
                hdrFile.write(header)

    return [filePrefix + ".cpp" for filePrefix in filesToWrite]


def markDuplicateKernels(
    kernels: List[Solution], kernelWriterAssembly: KernelWriterAssembly
) -> List[Solution]:
    """Marks duplicate assembly kernels based on their generated base file names.

    Kernels written in Assembly language may generate duplicate output file names,
    leading to potential race conditions. This function identifies such duplicates within
    the provided list of Solution objects and marks them to prevent issues.

    Args:
        kernels: A list of Solution objects representing kernels to be processed.

    Returns:
        A modified list of Solution objects where kernels identified as duplicates
        are marked with a `duplicate` attribute indicating their duplication status.

    Notes:
        This function sets the "duplicate" attribute on Solution objects, and thereby prepares
        kernels for **processKernelSource**, which requires "duplicate" to be set.
    """
    # Kernels may be intended for different .co files, but generate the same .o file
    # Mark duplicate kernels to avoid race condition
    # @TODO improve organization so this problem doesn't appear
    visited = set()
    count = 0
    for kernel in kernels:
        if kernel["KernelLanguage"] == "Assembly":
            curr = kernelWriterAssembly.getKernelFileBase(kernel)
            kernel.duplicate = curr in visited
            count += kernel.duplicate
            visited.add(curr)
    if count:
        printWarning(f"Found {count} duplicate kernels, these will be ignored")
    return kernels


def filterProcessingErrors(
    kernels: List[Solution],
    solutions: List[Solution],
    results: List[ProcessedKernelResult],
    errorTolerant: bool,
):
    """Filters out processing errors from lists of kernels, solutions, and results.

    This function iterates through the results of **processKernelSource** and identifies
    any errors encountered during processing. If an error is found (-2 error code),
    the corresponding kernel, solution, and result are appended to separate lists
    for removal. After processing, items identified for removal are deleted from the
    original lists of kernels, solutions, and results.

    Args:
        kernels: List of Solution objects representing kernels.
        solutions: List of Solution objects associated with kernels.
        results: List of tuples representing processing results.
        printLevel: Print level indicator.

    Returns:
        Tuple[List[Solution], List[Solution], List[Any]]: Tuple containing filtered lists
        of kernels, solutions, and results after removing items with processing errors.

    Raises:
        KeyError: If 'PrintLevel' key is not found in the params dictionary.
    """
    removeKernels = []
    removeSolutions = []
    removeResults = []
    for kernIdx, res in (
        enumerate(results)
        if globalParameters["PrintLevel"] == 0
        else Utils.tqdm(enumerate(results), desc="Filtering processing errors")
    ):
        (err, src, header, kernelName, filename) = res
        if err == -2:
            if not errorTolerant:
                print(
                    "\nKernel generation failed for kernel: {}".format(
                        kernels[kernIdx]["SolutionIndex"]
                    )
                )
                print(kernels[kernIdx]["SolutionNameMin"])
            removeKernels.append(kernels[kernIdx])
            removeSolutions.append(solutions[kernIdx])
            removeResults.append(results[kernIdx])
    if len(removeKernels) > 0 and not errorTolerant:
        printExit("** kernel generation failure **")
    for kern in removeKernels:
        kernels.remove(kern)
    for solut in removeSolutions:
        solutions.remove(solut)
    for rel in removeResults:
        results.remove(rel)


def filterBuildErrors(
    kernels: List[Solution],
    kernelsWithBuildErrors: Dict[str, int],
    writerSelectionFn: Callable[[str], Union[KernelWriterSource, KernelWriterAssembly]],
    ignoreErr: bool,
) -> List[Solution]:
    """Filters a list of kernels based on build errors and error tolerance.

    Args:
        kernels: A list of `Solution` objects representing kernels to filter.
        kernelsWithBuildErrors: A list of `Solution` objects that have build errors.
        errorTolerant: A boolean indicating whether to tolerate build errors.

    Returns:
        A filtered list of kernels (**Solution** objects) that are eligible for building.

    Raises:
        SystemExit: If **ignoreErr** is False and any kernels have build errors.
    """
    if not ignoreErr and len(kernelsWithBuildErrors) > 0:
        raise RuntimeError(
            "Kernel compilation failed in one or more subprocesses. "
            "Consider setting CpuThreads=0 and re-run to debug."
        )

    def noBuildError(kernel):
        kernelName = writerSelectionFn(kernel["KernelLanguage"]).getKernelName(kernel)
        return kernelName not in kernelsWithBuildErrors

    return list(filter(noBuildError, kernels))


def getKernelSourceAndHeaderCode(ko: KernelWriterBase) -> Tuple[int, List[str], List[str], str]:
    """Get the source and header content for a kernel object.

    Arguments:
        ko: Kernel object to extract content from.

    Returns:
        Tuple of data: (error code, source code, header code, kernel name)
    """
    name = ko.getKernelName()
    err, src = ko.getSourceFileString()
    hdr = ko.getHeaderFileString()
    return err, [CHeader, src], [CHeader, hdr], name


def writeKernelHelpers(
    kernelHelperObj: KernelWriterBase,
    kernelSourceFile: Optional[TextIOWrapper],
    kernelHeaderFile: Optional[TextIOWrapper],
    outputPath: Path,
    kernelFiles: List[str],
):
    """Writes the source and header code generated by a kernel helper object to specified files or a new file.

    Args:
        kernelHelperObj: The kernel helper object providing source and header code.
        kernelSourceFile: The file object for the kernel's source. If None, a new file is created.
        kernelHeaderFile: The file object for the kernel's header. If None, a new file is created.
        outputPath: The directory path where new files should be saved if `kernelSourceFile` and
            `kernelHeaderFile` are None.
        kernelFiles: A list of kernel file names to be updated with the new kernel name if new
            files are created.

    Notes:
        - If `kernelSourceFile` and `kernelHeaderFile` are provided, the source and header code
          are appended to these files.
        - If these file objects are None, new `.cpp` and `.h` files are created in the
          `outputPath/Kernels` directory named after the kernel.
        - The function appends the new kernel name to `kernelFiles` if new files are created.
    """
    err, srcCode, hdrCode, kernelName = getKernelSourceAndHeaderCode(kernelHelperObj)
    if err:
        printWarning(f"Invalid kernel: {kernelName} may be corrupt")
    if kernelSourceFile and kernelHeaderFile:  # Append to existing files => mergeFiles == True
        toFile(kernelSourceFile, srcCode)
        toFile(kernelHeaderFile, hdrCode)
    else:  # Write to new a file for each helper => mergeFiles == False. Default behaviour when called through rocBLAS
        srcFilename = Path(outputPath) / "Kernels" / f"{kernelName}.cpp"
        hdrFilename = Path(outputPath) / "Kernels" / f"{kernelName}.h"
        toFile(srcFilename, srcCode)
        toFile(hdrFilename, hdrCode)
        kernelFiles.append(str(srcFilename))


################################################################################
# Write Solutions and Kernels for BenchmarkClient or LibraryClient
################################################################################
def writeKernels(
    outputPath: str,
    cxxCompiler: str,
    bundler: str,
    params: Dict[str, Any],
    solutions: List[Solution],
    kernels: List[Solution],
    kernelHelperObjs: List[KernelWriterBase],
    kernelWriterSource: KernelWriterSource,
    kernelWriterAssembly: KernelWriterAssembly,
    errorTolerant: bool = False,
    removeTemporaries: bool = True,
):
    start = time.time()

    # Push working path into build_tmp folder because there may be more than
    # one process running this script. This is to avoid build directory clashing.
    # NOTE: file paths must not contain the lower case word 'kernel' or the
    # /opt/rocm/bin/extractkernel will fail.
    # See buildSourceCodeObjectFile:167 for the call to this binary.

    ## TODO: Is there a way to get this to work without changing global state?
    Common.pushWorkingPath("build_tmp")
    Common.pushWorkingPath(os.path.basename(outputPath).upper())

    tPrint(1, "# Writing Kernels...")

    ## TODO: This may be unused
    if not params["MergeFiles"] or params["NumMergedFiles"] > 1 or params["LazyLibraryLoading"]:
        ensurePath(os.path.join(outputPath, "Kernels"))

    ## This uses global state from "WorkingPath"
    prepAsm(
        kernelWriterAssembly,
        os.name != "nt",
        # Use globalParameters here, not params
        Path(globalParameters["WorkingPath"]),
        globalParameters["CurrentISA"],
        params["PrintLevel"],
    )

    kernels = markDuplicateKernels(kernels, kernelWriterAssembly)

    kIter = zip(
        kernels,
        itertools.repeat(kernelWriterSource),
        itertools.repeat(kernelWriterAssembly),
    )
    results = Common.ParallelMap(processKernelSource, list(kIter), "Generating kernels")

    filterProcessingErrors(kernels, solutions, results, errorTolerant)

    kernelsWithBuildErrors = {kernelName: err for err, _, _, kernelName, _ in results if err}
    filesToWrite = collectFilesToWrite(
        results,
        Path(outputPath),
        params["LazyLibraryLoading"],
        params["MergeFiles"],
        params["NumMergedFiles"],
    )

    kernelFiles = generateKernelSourceAndHeaderFiles(filesToWrite)

    writerSelector = lambda lang: kernelWriterAssembly if lang == "Assembly" else kernelWriterSource
    kernelsToBuild = filterBuildErrors(
        kernels, kernelsWithBuildErrors, writerSelector, errorTolerant
    )

    outPath = Path(outputPath)
    with KernelFileContextManager(
        params["LazyLibraryLoading"],
        params["MergeFiles"],
        params["NumMergedFiles"],
        outPath,
        kernelFiles,
    ) as (srcFile, hdrFile):
        for ko in kernelHelperObjs:
            writeKernelHelpers(ko, srcFile, hdrFile, outPath, kernelFiles)

    codeObjectFiles = []
    if not globalParameters["GenerateSourcesAndExit"]:
        codeObjectFiles += SourceCommands.buildSourceCodeObjectFiles(
            cxxCompiler, kernelFiles, outputPath, removeTemporaries
        )
        codeObjectFiles += AssemblyCommands.buildAssemblyCodeObjectFiles(
            bundler,
            kernelsToBuild,
            kernelWriterAssembly,
            outputPath,
            removeTemporaries,
        )

    stop = time.time()
    tPrint(1, "# Kernel Building elapsed time = %.1f secs" % (stop - start))

    Common.popWorkingPath()  # outputPath.upper()
    Common.popWorkingPath()  # build_tmp

    return codeObjectFiles, kernels, solutions


##############################################################################
# Min Naming / Solution and Kernel Writers
##############################################################################
def getKernelWriters(solutions: List[Solution], kernels: List[Solution], removeTemporaries):

    # if any kernels are assembly, append every ISA supported
    kernelSerialNaming = Solution.getSerialNaming(kernels)

    solutionMinNaming = Solution.getMinNaming(solutions)
    kernelMinNaming = Solution.getMinNaming(kernels)
    kernelWriterSource = KernelWriterSource(kernelMinNaming, kernelSerialNaming, removeTemporaries)
    kernelWriterAssembly = KernelWriterAssembly(
        kernelMinNaming, kernelSerialNaming, removeTemporaries
    )

    return (
        kernelWriterSource,
        kernelWriterAssembly,
        kernelMinNaming,
        solutionMinNaming,
    )


################################################################################
# copy static cpp files and headers
################################################################################
def copyStaticFiles(outputPath=None):
    if outputPath is None:
        outputPath = globalParameters["WorkingPath"]
    libraryStaticFiles = [
        "TensileTypes.h",
        "tensile_bfloat16.h",
        "tensile_float8_bfloat8.h",
        "hip_f8_impl.h",
        "KernelHeader.h",
    ]

    for fileName in libraryStaticFiles:
        # copy file
        shutil.copy(os.path.join(globalParameters["SourcePath"], fileName), outputPath)

    return libraryStaticFiles


def buildObjectFileNames(
    kernelWriterSource, kernelWriterAssembly, solutions, kernels, kernelHelperObjs
):

    # Build lists of output object names
    sourceKernelNames = []
    asmKernelNames = []
    kernelHelperObjNames = []

    solutionFiles = []
    sourceKernelFiles = []
    asmKernelFiles = []
    sourceLibFiles = []
    asmLibFiles = []

    sourceKernels = list([k for k in kernels if k["KernelLanguage"] == "Source"])
    asmKernels = list([k for k in kernels if k["KernelLanguage"] == "Assembly"])

    # Build a list of kernel object names.
    for kernel in sourceKernels:
        sourceKernelNames += [kernelWriterSource.getKernelFileBase(kernel)]

    for kernel in asmKernels:
        asmKernelNames += [kernelWriterAssembly.getKernelFileBase(kernel)]

    kernelHelperObjNames = [ko.getKernelName() for ko in kernelHelperObjs]

    # Source based kernels are built for all supported architectures
    sourceArchs, _ = splitArchs()

    # Asm based kernels target the configured ISA
    asmArchs = collections.defaultdict(list)
    for kernelName, kernel in zip(asmKernelNames, asmKernels):
        asmArchs[kernelName].append(gfxName(kernel["ISA"]))

    # Build a list of source files
    if not globalParameters["MergeFiles"]:
        for kernelName in sourceKernelNames + asmKernelNames + kernelHelperObjNames:
            sourceKernelFiles += ["%s.h" % (kernelName), "%s.cpp" % (kernelName)]
    elif globalParameters["NumMergedFiles"] > 1:
        for kernelIndex in range(0, globalParameters["NumMergedFiles"]):
            sourceKernelFiles += [
                "Kernels%s.h" % str(kernelIndex),
                "Kernels%s.cpp" % str(kernelIndex),
            ]
        for kernelName in kernelHelperObjNames:
            sourceKernelFiles += ["%s.h" % (kernelName), "%s.cpp" % (kernelName)]
    else:
        sourceKernelFiles += ["Kernels.h", "Kernels.cpp"]

    # Build a list of assembly files
    for asmKernelName in asmKernelNames:
        asmKernelFiles += [
            "%s.s" % (asmKernelName),
            "%s.o" % (asmKernelName),
            "%s.co" % (asmKernelName),
        ]

    # Build a list of lib names from source
    if not globalParameters["MergeFiles"]:

        allSources = sourceKernelNames + kernelHelperObjNames

        for kernelName in allSources:
            sourceLibFiles += ["%s.so-000-%s.hsaco" % (kernelName, arch) for arch in sourceArchs]
    elif globalParameters["NumMergedFiles"] > 1:
        for kernelIndex in range(0, globalParameters["NumMergedFiles"]):
            sourceLibFiles += [
                "Kernels%d.so-000-%s.hsaco" % (kernelIndex, arch) for arch in sourceArchs
            ]
    elif globalParameters["LazyLibraryLoading"]:
        fallbackLibs = list(
            set(
                [
                    kernel["codeObjectFile"]
                    for kernel in kernels
                    if "fallback" in kernel.get("codeObjectFile", "")
                ]
            )
        )
        sourceLibFiles += [
            "{0}_{1}.hsaco".format(name, arch)
            for name, arch in itertools.product(fallbackLibs, sourceArchs)
        ]
        sourceLibFiles += ["Kernels.so-000-%s.hsaco" % (arch) for arch in sourceArchs]
    else:  # Merge
        sourceLibFiles += ["Kernels.so-000-%s.hsaco" % (arch) for arch in sourceArchs]

    # Returns names for all xnack versions
    def addxnack(name, ext):
        arch = re.search(r"gfx.*$", name).group()
        if arch in sourceArchs:
            return [name + ext]
        else:
            return [name + xnack[len(arch) :] + ext for xnack in sourceArchs if arch in xnack]

    # Build a list of asm lib names
    if globalParameters["LazyLibraryLoading"]:

        # If assembly kernel with codeObjectFile specified
        cond = (
            lambda k: "codeObjectFile" in k
            and "fallback" not in k["codeObjectFile"]
            and k["KernelLanguage"] == "Assembly"
        )

        asmLibFiles += list(
            set([kernel["codeObjectFile"] + ".co" for kernel in kernels if cond(kernel)])
        )

        # If architecture specific source kernel with codeObjectFile specified
        cond = (
            lambda k: "codeObjectFile" in k
            and "fallback" not in k["codeObjectFile"]
            and k["KernelLanguage"] == "Source"
        )

        sourceLibFiles += list(
            set(
                itertools.chain.from_iterable(
                    [
                        addxnack(kernel["codeObjectFile"], ".hsaco")
                        for kernel in kernels
                        if cond(kernel)
                    ]
                )
            )
        )

    elif globalParameters["MergeFiles"]:
        # Find all unique arch values for current asm kernels
        uniqueArchs = set(itertools.chain(*asmArchs.values()))
        asmLibFiles += ["TensileLibrary_%s.co" % (arch) for arch in uniqueArchs]

    else:
        for asmKernelName, archs in asmArchs.items():
            asmLibFiles += ["%s_%s.co" % (asmKernelName, str(arch)) for arch in archs]

    return (
        solutionFiles,
        sourceKernelFiles,
        asmKernelFiles,
        sourceLibFiles,
        asmLibFiles,
    )


def buildObjectFilePaths(
    prefixDir,
    solutionFiles,
    sourceKernelFiles,
    asmKernelFiles,
    sourceLibFiles,
    asmLibFiles,
    masterLibraries,
):
    solutionPaths = []
    sourceKernelPaths = []
    asmKernelPaths = []
    sourceLibPaths = []
    asmLibPaths = []
    libMetadataPaths = []

    # Build full paths for source kernel files
    sourceKernelDir = ""
    if not globalParameters["MergeFiles"] or globalParameters["NumMergedFiles"] > 1:
        sourceKernelDir = os.path.join(prefixDir, "Kernels")
    else:
        sourceKernelDir = prefixDir

    for sourceKernelFile in sourceKernelFiles:
        sourceKernelPaths += [os.path.join(sourceKernelDir, sourceKernelFile)]

    # Build full paths for asm kernel files
    asmKernelDir = os.path.join(prefixDir, "assembly")

    for asmKernelFile in asmKernelFiles:
        asmKernelPaths += [os.path.join(asmKernelDir, asmKernelFile)]

    # Build full paths for source and asm library files
    libDir = os.path.join(prefixDir, "library")

    libraryExt = ".yaml" if globalParameters["LibraryFormat"] == "yaml" else ".dat"
    if not globalParameters["SeparateArchitectures"] and not globalParameters["LazyLibraryLoading"]:
        libMetadataPaths = [os.path.join(libDir, "TensileLibrary" + libraryExt)]

    for sourceLibFile in sourceLibFiles:
        sourceLibPaths += [os.path.join(libDir, sourceLibFile)]

    # Use set because of duplicate fallback libraries
    newMetadataPaths = set()
    if "full" not in masterLibraries.keys():
        for arch, lib in masterLibraries.items():
            if globalParameters["LazyLibraryLoading"]:
                newMetadataPaths.add(
                    os.path.join(libDir, "TensileLibrary_lazy_" + arch + libraryExt)
                )
            else:
                newMetadataPaths.add(os.path.join(libDir, "TensileLibrary_" + arch + libraryExt))
            for name, placeholder in lib.lazyLibraries.items():
                newMetadataPaths.add(os.path.join(libDir, name + libraryExt))

    libMetadataPaths += list(newMetadataPaths)

    for asmLibFile in asmLibFiles:
        # Asm lib files are enumerated in the form of
        # KernelName_gfxXXXXX.co
        asmLibPaths += [os.path.join(libDir, asmLibFile)]

    return (
        solutionPaths,
        sourceKernelPaths,
        asmKernelPaths,
        sourceLibPaths,
        asmLibPaths,
        libMetadataPaths,
    )


################################################################################
# Write CMake
################################################################################
def writeCMake(outputPath, solutionFiles, kernelFiles, libraryStaticFiles, masterLibraries):
    tPrint(1, "# Writing Custom CMake")

    # Build output file paths, using relative CMake symbol
    cmakeSrcDir = "${CMAKE_SOURCE_DIR}"
    (
        solutionPaths,
        sourceKernelPaths,
        asmKernelPaths,
        sourceLibPaths,
        asmLibPaths,
        _,
    ) = buildObjectFilePaths(cmakeSrcDir, solutionFiles, kernelFiles, [], [], [], masterLibraries)

    # Build full paths the static library files
    staticFilePaths = []
    for staticFile in libraryStaticFiles:
        staticFilePaths += [os.path.join(cmakeSrcDir, staticFile)]

    # Proceed to generate cmake file
    generatedFile = open(os.path.join(os.path.normcase(outputPath), "Generated.cmake"), "w")
    generatedFile.write(CMakeHeader)

    # write TensileClient_KERNELS symbol
    generatedFile.write("set( TensileClient_KERNELS\n")
    for kernelFile in sourceKernelPaths:
        generatedFile.write("  %s\n" % (kernelFile))
    generatedFile.write("  )\n")

    # write TensileClient_SOURCE symbol
    generatedFile.write("set( TensileClient_SOURCE\n")
    for fileName in libraryStaticFiles:
        generatedFile.write("  ${CMAKE_SOURCE_DIR}/%s\n" % fileName)
    generatedFile.write("  )\n\n")

    generatedFile.close()


################################################################################
# Generate Kernel Objects From Solutions
################################################################################
def generateKernelObjectsFromSolutions(
    solutions: List[Solution],
) -> Tuple[List[Solution], List[KernelWriterBase], List[str]]:
    # create solution writer and kernel writer
    kernels = []
    kernelHelperObjs = []
    kernelHelperNames = set()

    for solution in solutions:
        kernels.append(solution.getKernels())
        solutionHelperKernels = solution.getHelperKernelObjects()
        kernelHelperObjs += solutionHelperKernels
        for ko in solutionHelperKernels:
            kernelHelperNames.add(ko.getKernelName())

    kernelHelperObjs = list(dict.fromkeys(kernelHelperObjs))
    return (kernels, kernelHelperObjs, kernelHelperNames)


def addNewLibrary(
    masterLibraries: Dict[str, MasterSolutionLibrary],
    newLibrary: MasterSolutionLibrary,
    architectureName: str,
) -> int:
    """Adds new master solution library to a master solution libraries dict.

    For a given architecture, add the new library to a dictionary containing
    libraries for all architectures, compute the starting index for the new
    library, then remap the indexes for all of the solutions associated with
    the library.

    Args:
        masterLibraries: A dictionary containing all master solution libraries for all architectures.
        newLibrary: A master solution library to add to the dictionary.
        architectureName: The name of the architecture (or key) associated with the library.

    Returns:
        Index to the last solution of the library associated with current architecture.
    """
    masterLibraries[architectureName] = newLibrary
    archIndex = MasterSolutionLibrary.ArchitectureIndexMap(architectureName)
    masterLibraries[architectureName].remapSolutionIndicesStartingFrom(archIndex)
    return archIndex


def makeMasterLibraries(
    logicList: List[LibraryIO.LibraryLogic], separate: bool
) -> Dict[str, MasterSolutionLibrary]:
    """Creates a dictionary of master solution libraries.

    Iterates through a list of LibraryLogic objects creating
    master solution libraries and modifying the solution
    indexing as required.

    Args:
        logicFiles: List of LibraryLogic objects.
        separate: Separate libraries by architecture.

    Returns:
        An architecture separated master solution libraries
        or a single master solution library for all architectures.
    """
    masterLibraries = {}
    nextSolIndex = {}
    fullMasterLibrary = None

    for logic in logicList:
        (_, architectureName, _, solutionsForSchedule, _, newLibrary) = logic
        if separate:
            if architectureName in masterLibraries:
                nextSolIndex[architectureName] = masterLibraries[architectureName].merge(
                    newLibrary, nextSolIndex[architectureName]
                )
            else:
                nextSolIndex[architectureName] = addNewLibrary(
                    masterLibraries, newLibrary, architectureName
                )
        else:
            if fullMasterLibrary:
                fullMasterLibrary.merge(newLibrary)
            else:
                fullMasterLibrary = newLibrary

    return {"full": fullMasterLibrary} if fullMasterLibrary is not None else masterLibraries


def addFallback(masterLibraries: Dict[str, MasterSolutionLibrary]) -> None:
    """Adds fallback library.

    Given a master solution library, add a fallback and if the corresponding
    architecture is unsupported, replace the library altogether with a fallback.

    Args:
        masterLibraries: A dictionary containing the master solution libraries.
    """
    archs, _ = splitArchs()

    for key, value in masterLibraries.items():
        if key != "fallback":
            value.insert(masterLibraries["fallback"])

    for archName in archs:
        archName = archName.split("-", 1)[0]
        if archName not in masterLibraries:
            tPrint(1, "Using fallback for arch: " + archName)
            masterLibraries[archName] = masterLibraries["fallback"]

    masterLibraries.pop("fallback")


def applyNaming(masterLibraries: Dict[str, MasterSolutionLibrary]) -> None:
    """Assigns the solution code object file name for lazy libraries.

    Given a master solution library with lazy libraries, assigns the
    key associated with the lazy library (or name) as the value
    assiciated with the corresponding solution's code object file.

    Args:
        masterLibraries: A dictionary containing the master solution libraries.
    """
    for masterLibrary in masterLibraries.values():
        for name, lib in masterLibrary.lazyLibraries.items():
            for sol in lib.solutions.values():
                sol.originalSolution["codeObjectFile"] = name


def makeSolutions(
    masterLibraries: dict, separate: bool
):  # -> Generator[Solution]:# is breaking tensile
    """Extracts the solutions from the master solution library.

    Given a master solution library, forms a flattened generator that
    yields solutions by iterating over all of the solutions contained
    in the master solution libraries. If using separate architectures
    but not using lazy loading, lazyLibraries should be an empty dict.

    Args:
        masterLibraries: A dictionary containing the master solution libraries.

    Returns:
        Generator representing a sequence of library logic tuples.
    """
    gen1 = (
        sol.originalSolution
        for masterLibrary in masterLibraries.values()
        for sol in masterLibrary.solutions.values()
    )
    gen2 = (
        sol.originalSolution
        for masterLibrary in masterLibraries.values()
        for lib in masterLibrary.lazyLibraries.values()
        for sol in lib.solutions.values()
    )
    return itertools.chain(gen1, gen2)


def parseLibraryLogicFiles(logicFiles: List[str]) -> List[LibraryIO.LibraryLogic]:
    """Load and parse logic (yaml) files.

    Given a list of paths to yaml files containing library logic, load the files
    into memory and parse the data into a named tuple (i.e. LibraryLogic). This
    operation is parallelized over N processes.

    Args:
        logicFiles: List of paths to logic files.

    Returns:
        List of library logic tuples.
    """
    return Common.ParallelMap(
        LibraryIO.parseLibraryLogicFile, logicFiles, "Reading logic files", multiArg=False
    )


def generateLogicData(
    logicFiles: List[str], version: str, printLevel: int, separate: bool
) -> Dict[str, MasterSolutionLibrary]:
    """Generates a dictionary of master solution libraries.

    Args:
        logicFiles: List of paths to logic files.
        version: User provided version for the library.
        printLevel: Level of debug printing requested.
        separate: Separate libraries by architecture.

    Returns:
        For separate architectures, a dictionary of architecture
        separated master solution libraries; otherwise, a single
        master solution library for all architectures.
    """
    libraries = parseLibraryLogicFiles(logicFiles)
    logicList = libraries if not printLevel else Utils.tqdm(libraries, desc="Processing logic data")
    masterLibraries = makeMasterLibraries(logicList, separate)
    if separate and "fallback" in masterLibraries:
        addFallback(masterLibraries)
    applyNaming(masterLibraries)
    for lib in masterLibraries.values():
        lib.version = version

    return masterLibraries


def generateSolutions(
    masterLibraries: Dict[str, MasterSolutionLibrary], separate: bool
) -> List[Solution]:
    """Generates a list of solutions.

    Args:
        masterLibraries: A dictionary of master solutions libraries.
        separate: Separate libraries by architecture.

    Returns:
        A solution list.
    """
    # remove duplicates while preserving order
    return list(dict.fromkeys(makeSolutions(masterLibraries, separate)))


################################################################################
# Write Benchmark Client Files
################################################################################
def writeBenchmarkClientFiles(
    libraryWorkingPath, tensileSourcePath, solutions, cxxCompiler, removeTemporaries=False
):

    if not globalParameters["GenerateSourcesAndExit"]:
        copyStaticFiles(libraryWorkingPath)

    kernels, kernelsBetaOnly, _ = generateKernelObjectsFromSolutions(solutions)
    kernelWriterSource, kernelWriterAssembly, kernelMinNaming, _ = getKernelWriters(
        solutions,
        kernels,
        removeTemporaries,
    )

    # write solution, kernels and CMake
    codeObjectFiles, kernels, solutions = writeKernels(
        libraryWorkingPath,
        cxxCompiler,
        globalParameters["ClangOffloadBundler"],
        globalParameters,
        solutions,
        kernels,
        kernelsBetaOnly,
        kernelWriterSource,
        kernelWriterAssembly,
        errorTolerant=True,
        removeTemporaries=removeTemporaries,
    )

    newLibraryDir = ensurePath(os.path.join(libraryWorkingPath, "library"))
    newLibraryFile = os.path.join(newLibraryDir, "TensileLibrary.yaml")
    newLibrary = MasterSolutionLibrary.BenchmarkingLibrary(solutions)
    newLibrary.applyNaming(kernelMinNaming)

    LibraryIO.writeYAML(newLibraryFile, Utils.state(newLibrary))

    return (codeObjectFiles, newLibrary)


################################################################################
# Write Master Solution Index CSV
################################################################################
def writeMasterSolutionIndexCSV(outputPath, masterLibraries):
    libraryPath = os.path.join(outputPath, "library")
    ensurePath(libraryPath)
    try:
        with open(os.path.join(libraryPath, "TensileMasterSolutionIndex.csv"), "w") as indexFile:
            indexFile.write(
                "architectureName,libraryName,libraryIndex,solutionIndex,solutionName\n"
            )
            for arch, lib in masterLibraries.items():
                for lazylibname, lazylibvals in lib.lazyLibraries.items():
                    for solidx, solution in lazylibvals.solutions.items():
                        line = ",".join(
                            str(x)
                            for x in [
                                arch,
                                lazylibname,
                                solidx,
                                solution.index,
                                solution.name,
                            ]
                        )
                        indexFile.write("%s\n" % (line))
    except IOError as err:
        tPrint(1, "Error writing MasterSolutionIndex %s" % err)


def verifyManifest(manifest: Path) -> bool:
    """Verifies whether the files listed in the manifest exist on disk.

    Args:
        manifest: Path to the manifest file.

    Returns:
        True if all files exist on disk, otherwise False.
    """
    with open(manifest, mode="r") as generatedFiles:
        for f in generatedFiles.readlines():
            if not Path(f.rstrip()).exists():
                printWarning(f"File in manifest ``{f}`` not found.")
                return False
    return True


def findLogicFiles(
    path: Path,
    logicArchs: Set[str],
    lazyLoading: bool,
    experimentalDir: str,
    extraMatchers: Set[str] = {"hip"},
) -> List[str]:
    """Recursively searches the provided path for logic files.

    Args:
        path: The path to the directory to search.
        logicArchs: Target logic archiectures. These are interepreted as filename substrings
            for which logic files are to be included.
        extraMatchers: Additional directories to include for logic files.

    Returns:
        A list of Path objects representing the found YAML files.
    """
    isMatch = lambda file: any((arch in file.stem for arch in logicArchs.union(extraMatchers)))
    isExperimental = lambda path: not experimentalDir in str(path)

    extensions = ["*.yaml", "*.yml"]
    logicFiles = filter(isMatch, (file for ext in extensions for file in path.rglob(ext)))
    if not lazyLoading:
        if not experimentalDir:
            printWarning(
                "Configuration parameter `ExperimentalLogicDir` is an empty string, "
                "logic files may be filtered incorrectly."
            )
        logicFiles = filter(isExperimental, logicFiles)

    return list(str(l) for l in logicFiles)


def sanityCheck(
    srcLibPaths: List[str],
    asmLibPaths: List[str],
    codeObjectPaths: List[str],
    genSourcesAndExit: bool,
):
    """Verifies that generated code object paths match associated library paths.

    Args:
        srcLibPaths: Source library paths (.hsaco).
        asmLibPaths: Assembly library paths (.co).
        coPaths: Code object paths containing generated kernels; should contain all assembly
            and source library paths.
        genSourcesAndExit: Flag identifying whether only source file should be generated.

    Raises:
        ValueError: If code object paths do not match library paths.
    """
    libPaths = set([Path(p).resolve() for p in srcLibPaths + asmLibPaths])
    coPaths = set([Path(p).resolve() for p in codeObjectPaths])

    tPrint(2, "Library paths:\n    " + "\n    ".join(map(str, libPaths)))
    tPrint(2, "Code object paths:\n    " + "\n    ".join(map(str, coPaths)))

    extraCodeObjects = coPaths - libPaths
    if extraCodeObjects:
        raise ValueError(
            f"Sanity check failed; unexpected code object files: "
            f"{[p.name for p in extraCodeObjects]}"
        )

    if not genSourcesAndExit:
        extraLibs = libPaths - coPaths
        if extraLibs:
            raise ValueError(
                f"Sanity check failed; missing expected code object files: "
                f"{[p.name for p  in extraLibs]}"
            )


def generateClientConfig(
    outputPath: Path,
    masterFile: Path,
    codeObjectFiles: List[str],
    configFile: str = "best-solution.ini",
) -> None:
    """Generates a client config file.

    Generates a client config file corresponding to a master library file and code-object parameters
    created by a TensileCreateLibrary invocation. Also sets best-solution-mode to True.

    Args:
        outputPath: The path to the tensile output directory where output files are written.
        masterFile: Path to the master library file (.dat or .yaml).
        codeObjectFiles: List of code object files created by TensileCreateLibrary.
        configFile: Name of config file written to the output directory.
    """
    iniFile = outputPath / configFile

    def param(key, value):
        f.write(f"{key}={value}\n")

    with open(iniFile, "w") as f:
        if not masterFile.is_file():
            warnings.warn(
                UserWarning(f"{masterFile} does not exist. best-solution.ini may be invalid.")
            )

        param("library-file", masterFile)
        for coFile in codeObjectFiles:
            codeObject: Path = outputPath / coFile
            if not codeObject.is_file():
                warnings.warn(
                    UserWarning(f"{codeObject} does not exist. best-solution.ini may be invalid.")
                )

            param("code-object", outputPath / coFile)

        param("best-solution", True)


def generateLazyMasterFileList(
    masterFileList: List[Tuple[str, MasterSolutionLibrary]],
) -> List[Tuple[str, MasterSolutionLibrary]]:
    """Generates a list of tuples that represent the name and the state associated with the lazy master libraries.

    This function takes a list of MasterSolutionLibraries and traverses each lazy libraries.
    It collects the items (i.e. the name and corresponding master file) and adds them to list
    of master files.

    Args:
        masterLibraries: A list of name / master solution library pairs.

    Returns:
        List of pairs of master solutions libraries and the corresponding name.
    """
    return [t for _, lib in masterFileList for t in lib.lazyLibraries.items()]


def generateMasterFileList(
    masterLibraries: dict, archs: List[str], lazy: bool
) -> List[Tuple[str, MasterSolutionLibrary]]:
    """Generates a list of tuples that represent the name and the state associated with the master libraries.

    This function takes a dictionary with keys corresponding to a target architecture and values
    corresponding to the master solution library for that architecture. The function generates a
    tuple consisting of a MasterSolutionLibrary and the associated name. When not separating architectures,
    the key full will appear in masterLibraries indicating that all libraries are combinded into a
    single master library.

    Args:
        masterLibraries: A dictionary of architecture name / master solution library pairs.
        archs: A list of supported architectures.
        lazy: If True, add lazy library master files.

    Returns:
        List of pairs of master solutions libraries and the corresponding name.
    """
    if "full" in masterLibraries.keys():
        return [("TensileLibrary", masterLibraries["full"])]

    baseName = "TensileLibrary_lazy_" if lazy else "TensileLibrary_"
    result = [
        (baseName + arch, masterLibrary)
        for arch, masterLibrary in masterLibraries.items()
        if arch in archs
    ]

    return result + generateLazyMasterFileList(result) if lazy else result


def writeMasterFile(
    libraryPath: Path, format: str, naming: dict, name: str, lib: MasterSolutionLibrary
) -> None:
    """Writes a master file to disk as a .yaml or .dat file.

    Args:
        libraryPath: Path to library subdirectory located in the tensile output directory.
        format: Output format of written file (.dat or .yaml).
        naming: Kernel minimum naming.
        name: Name of the masterfile.
        lib: Master solution library data.
    """
    lib.applyNaming(naming)
    LibraryIO.write(str(libraryPath / name), Utils.state(lib), format)


################################################################################
# Tensile Create Library
################################################################################
@profile
def TensileCreateLibrary():

    tPrint(3, "Arguments: %s" % sys.argv)
    args = parseArguments()

    lazyLoading = args["LazyLibraryLoading"]
    separateArchs = args["SeparateArchitectures"]
    mergeFiles = args["MergeFiles"]
    embedLibrary = args["EmbedLibrary"]
    libraryFormat = args["LibraryFormat"]
    logicPath = args["LogicPath"]
    outputPath = args["OutputPath"]
    removeTemporaries = not args["KeepBuildTmp"]

    globalParameters["PrintLevel"] = args["PrintLevel"]

    tPrint(3, "OutputPath: %s" % outputPath)
    ensurePath(outputPath)
    outputPath = os.path.abspath(outputPath)

    tPrint(1, "")
    tPrint(1, HR)
    tPrint(1, "# Tensile Create Library")
    tPrint(3, HR)
    tPrint(3, "")

    assignGlobalParameters(args)

    supportedArchs = [
        gfxName(arch)
        for arch in globalParameters["SupportedISA"]
        if globalParameters["AsmCaps"][arch]["SupportedISA"]
    ]

    _, requestedArchs = splitArchs()
    if all(a.split(":")[0] not in supportedArchs for a in requestedArchs):
        printExit(
            f"No requested architecture is supported by ROCm {globalParameters['HipClangVersion']}\n  Requested {', '.join(requestedArchs)}\n  Supported {', '.join(supportedArchs)}"
        )

    manifestFile = Path(outputPath) / TENSILE_LIBRARY_DIR / TENSILE_MANIFEST_FILENAME
    manifestFile.parent.mkdir(exist_ok=True)

    if args["VerifyManifest"]:
        if verifyManifest(manifestFile):
            tPrint(1, "Successfully verified all files in manifest were generated")
            return
        else:
            printExit("Failed to verify all files in manifest")

    tPrint(1, "# CodeObjectVersion: %s" % args["CodeObjectVersion"])
    tPrint(1, "# CxxCompiler:       %s" % args["CxxCompiler"])
    tPrint(1, "# CCompiler:         %s" % args["CCompiler"])
    tPrint(1, "# Assembler:         %s" % args["Assembler"])
    tPrint(1, "# OffloadBundler:    %s" % args["OffloadBundler"])
    tPrint(1, "# Architecture:      %s" % args["Architecture"])
    tPrint(1, "# LibraryFormat:     %s" % libraryFormat)

    if not os.path.exists(logicPath):
        printExit("LogicPath %s doesn't exist" % logicPath)

    # CLI uses `;` delimiters, CMake uses `_` delimiters
    logicArchs = splitDelimitedString(args["Architecture"], {";", "_"})
    logicArchs = {name for name in (getArchitectureName(gfxName) for gfxName in logicArchs) if name}

    if lazyLoading and not (mergeFiles and separateArchs):
        printExit(
            "--lazy-library-loading requires --merge-files and --separate-architectures enabled"
        )

    logicFiles = findLogicFiles(
        Path(logicPath),
        logicArchs,
        lazyLoading=lazyLoading,
        experimentalDir=globalParameters["ExperimentalLogicDir"],
    )

    tPrint(1, f"# LibraryLogicFiles: found {len(logicFiles)} files")
    tPrint(1, "#      set --verbose=2 to view all files")
    tPrint(2, "#   " + "\n#   ".join(logicFiles))

    masterLibraries = generateLogicData(
        logicFiles, args["Version"], args["PrintLevel"], args["SeparateArchitectures"]
    )
    solutions = generateSolutions(masterLibraries, args["SeparateArchitectures"])
    if lazyLoading and args["WriteMasterSolutionIndex"]:
        writeMasterSolutionIndexCSV(outputPath, masterLibraries)

    kernels, kernelHelperObjs, _ = generateKernelObjectsFromSolutions(solutions)

    # if any kernels are assembly, append every ISA supported
    kernelWriterSource, kernelWriterAssembly, kernelMinNaming, _ = getKernelWriters(
        solutions, kernels, removeTemporaries
    )

    staticFiles = copyStaticFiles(outputPath)

    (solutionFiles, sourceKernelFiles, asmKernelFiles, sourceLibFiles, asmLibFiles) = (
        buildObjectFileNames(
            kernelWriterSource,
            kernelWriterAssembly,
            solutions,
            kernels,
            kernelHelperObjs,
        )
    )

    (_, _, _, sourceLibPaths, asmLibPaths, libMetadataPaths) = buildObjectFilePaths(
        outputPath,
        solutionFiles,
        sourceKernelFiles,
        asmKernelFiles,
        sourceLibFiles,
        asmLibFiles,
        masterLibraries,
    )

    toFile(Path(manifestFile), libMetadataPaths + sourceLibPaths + asmLibPaths)
    if args["GenerateManifestAndExit"]:
        return

    if not args["GenerateSourcesAndExit"]:
        writeCMake(outputPath, solutionFiles, sourceKernelFiles, staticFiles, masterLibraries)

    # Make sure to copy the library static files.
    for fileName in staticFiles:
        shutil.copy(os.path.join(globalParameters["SourcePath"], fileName), outputPath)

    codeObjectFiles, kernels, solutions = writeKernels(
        outputPath,
        args["CxxCompiler"],
        globalParameters["ClangOffloadBundlerPath"],
        args,
        solutions,
        kernels,
        kernelHelperObjs,
        kernelWriterSource,
        kernelWriterAssembly,
        removeTemporaries=removeTemporaries,
    )

    sanityCheck(
        sourceLibPaths,
        asmLibPaths,
        codeObjectFiles,
        globalParameters["GenerateSourcesAndExit"],
    )

    tPrint(2, f"codeObjectFiles: {codeObjectFiles}")
    tPrint(2, f"sourceLibPaths + asmLibPaths: {sourceLibPaths + asmLibPaths}")

    newLibraryDir = Path(outputPath) / "library"
    newLibraryDir.mkdir(exist_ok=True)

    masterFileList = generateMasterFileList(masterLibraries, supportedArchs, lazyLoading)

    tPrint(1, f"# Writing {len(masterFileList)} solution selection catalog(s)")
    for name, lib in masterFileList:
        writeMasterFile(newLibraryDir, libraryFormat, kernelMinNaming, name, lib)

    if embedLibrary or args["ClientConfig"]:
        masterFile, fullMasterLibrary = masterFileList[0]
        ext = ".yaml" if globalParameters["LibraryFormat"] == "yaml" else ".dat"

    if embedLibrary:
        embedFileName = Path(outputPath) / "library" / args["EmbedLibrary"]
        EmbeddedData.generateLibrary(
            embedFileName,
            args["EmbedLibraryKey"],
            (newLibraryDir / masterFile).with_suffix(ext),
            fullMasterLibrary.cpp_base_class,
            codeObjectFiles,
        )

    if args["ClientConfig"]:
        generateClientConfig(Path(outputPath), Path(masterFile).with_suffix(ext), codeObjectFiles)

    if removeTemporaries:
        buildTmp = Path(outputPath).parent / "build_tmp"
        if buildTmp.exists() and buildTmp.is_dir():
            shutil.rmtree(buildTmp)

    tPrint(1, "# Tensile Library Writer DONE")
    tPrint(1, HR)
    tPrint(1, "")
