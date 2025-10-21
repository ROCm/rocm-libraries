################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

import rocisa

import functools
import glob
import itertools
import os
import shutil
from pathlib import Path
from timeit import default_timer as timer
from typing import List, NamedTuple, Optional, Union

from Tensile import SOURCE_PATH, LibraryIO
from Tensile.Common import (
    CHeader,
    DebugConfig,
    ensurePath,
    HR,
    IsaVersion,
    ParallelMap2,
    print1,
    print2,
    printWarning,
    printExit,
    printWarning,
    state,
    tqdm,
    setVerbosity,
    getVerbosity,
)
from Tensile.Common.Architectures import gfxToIsa, isaToGfx, SUPPORTED_GFX, splitArchsFromPredicates, filterLogicFilesByPredicates
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.GlobalParameters import assignGlobalParameters, globalParameters
from Tensile.SolutionStructs.Naming import getKernelFileBase, getKeyNoInternalArgs, getKernelNameMin

from Tensile.CustomYamlLoader import load_logic_gfx_arch
from Tensile.KernelHelperNaming import kernelObjectNameCallables, initHelperKernelObjects
from Tensile.KernelWriterAssembly import KernelWriterAssembly
from Tensile.KernelWriterBase import (
    KERNEL_HELPER_FILENAME_CPP,
    KERNEL_HELPER_FILENAME_H,
)
from Tensile.SolutionLibrary import MasterSolutionLibrary
from Tensile.SolutionStructs import Solution
from Tensile.Toolchain.Assembly import makeAssemblyToolchain, buildAssemblyCodeObjectFiles
from Tensile.Toolchain.Source import makeSourceToolchain, buildSourceCodeObjectFiles
from Tensile.Toolchain.Validators import (
    ToolchainDefaults,
    validateToolchain,
)
from Tensile.Toolchain.Component import Assembler
from Tensile.Utilities.Decorators.Profile import profile
from Tensile.Utilities.Decorators.Timing import timing

from .ParseArguments import parseArguments


class KernelCodeGenResult(NamedTuple):
    err: int
    src: str
    header: Optional[str]
    name: str
    targetObjFilename: str
    isa: IsaVersion
    wavefrontSize: int
    cuoccupancy: int
    pgr: int
    mathclk: int

class KernelMinResult(NamedTuple):
    err: int
    cuoccupancy: int
    pgr: int
    mathclk: int

def processKernelSource(kernelWriterAssembly, data, outOptions, splitGSU, kernel) -> KernelCodeGenResult:
    """
    Generate source for a single kernel.
    Returns (error, source, header, kernelName).
    """
    kernelWriter = kernelWriterAssembly
    kernelWriter.setRocIsa(data, outOptions)
    asmFilename = getKernelFileBase(splitGSU, kernel)
    err, src = kernelWriter.getSourceFileString(kernel)
    header = kernelWriter.getHeaderFileString(kernel)
    objFilename = kernel._state.get("codeObjectFile", None)
    pgr = int(kernel["PrefetchGlobalRead"])
    return KernelCodeGenResult(
        err, src, header, asmFilename, objFilename, tuple(kernel["ISA"]), \
        kernel["WavefrontSize"], kernel["CUOccupancy"], \
        pgr, kernel["MathClocksUnrolledLoop"]
    )


def removeInvalidSolutionsAndKernels(results, kernels, solutions, errorTolerant, printLevel: bool, splitGSU: bool):
    removeKernels = []
    removeKernelNames = []
    removeSolutions = []
    removeResults = []

    for kernIdx, r in (
        tqdm(enumerate(results)) if printLevel > 1 else enumerate(results)
    ):
        if r.err != 0:
            if not errorTolerant:
                print(
                    "\nKernel generation failed for kernel: {}".format(
                        kernels[kernIdx]["SolutionIndex"]
                    )
                )
                print(kernels[kernIdx]["SolutionNameMin"])
            removeKernels.append(kernels[kernIdx])
            kName = getKeyNoInternalArgs(kernels[kernIdx], splitGSU)
            if kName not in removeKernelNames:
                removeKernelNames.append(kName)
            removeResults.append(results[kernIdx])

    if len(removeKernels) > 0 and not errorTolerant:
        printExit("** kernel generation failure **")

    for kern in removeKernels:
        kernels.remove(kern)

    for solution in (
        tqdm(solutions, "Finding invalid solutions")
        if printLevel > 1
        else solutions
    ):
        solutionKernels = solution.getKernels()
        for kernel in solutionKernels:
            kName = getKeyNoInternalArgs(kernel, splitGSU)
            if kName in removeKernelNames:
                removeSolutions.append(solution)
                break

    for solut in removeSolutions:
        solutions.remove(solut)

    for rel in removeResults:
        results.remove(rel)

def passPostKernelInfoToSolution(results, kernels, solutions, splitGSU: bool):
    resultDict = {}
    for kernIdx, r in enumerate(results):
        kName = getKernelNameMin(kernels[kernIdx], splitGSU)
        resultDict["%s"%kName] = r
    for solution in solutions:
        solutionKernels = solution.getKernels()
        for kernel in solutionKernels:
            kName = getKernelNameMin(kernel, splitGSU)
            result = resultDict["%s"%kName]
            solution._state["CUOccupancy"] = result.cuoccupancy
            solution._state["PrefetchGlobalRead"] = result.pgr
            solution._state["MathClocksUnrolledLoop"] = result.mathclk

def writeAssembly(asmPath: Union[Path, str], result: KernelCodeGenResult):
    if result.err:
        printExit(f"Failed to build kernel {result.name} because it has error code {result.err}")
    path = Path(asmPath) / f"{result.name}.s"
    isa = result.isa
    wfsize = result.wavefrontSize
    with open(path, "w", encoding="utf-8") as f:
        f.write(result.src)

    minResult = KernelMinResult(result.err, result.cuoccupancy, result.pgr, result.mathclk)
    return path, isa, wfsize, minResult


def writeHelpers(
    outputPath, kernelHelperObjs, KERNEL_HELPER_FILENAME_CPP, KERNEL_HELPER_FILENAME_H
):
    kernelSourceFilename = os.path.join(os.path.normcase(outputPath), KERNEL_HELPER_FILENAME_CPP)
    kernelHeaderFilename = os.path.join(os.path.normcase(outputPath), KERNEL_HELPER_FILENAME_H)

    with open(kernelHeaderFilename, "w", encoding="utf-8") as kernelHeaderFile, open(
        kernelSourceFilename, "w", encoding="utf-8"
    ) as kernelSourceFile:
        kernelSourceFile.write(CHeader)
        kernelHeaderFile.write(CHeader)
        kernelSourceFile.write('#include "Kernels.h"\n')
        kernelHeaderFile.write("#pragma once\n")
        kernelHeaderFile.write("#include <hip/hip_runtime.h>\n")
        kernelHeaderFile.write("#include <hip/hip_ext.h>\n\n")
        kernelHeaderFile.write('#include "KernelHeader.h"\n\n')
        HeaderText = ""
        for ko in kernelHelperObjs:
            kernelName = ko.getKernelName()
            (err, src) = ko.getSourceFileString()

            kernelSourceFile.write(src)
            if err:
                print("*** warning: invalid kernel#%u" % kernelName)
            HeaderText += ko.getHeaderFileString()
        kernelHeaderFile.write(HeaderText)


def writeSolutionsAndKernels(
    outputPath,
    asmToolchain,
    srcToolchain,
    solutions,
    kernels,
    kernelHelperObjs,
    kernelWriterAssembly,
    splitGSU: bool,
    cmdlineArchs: List[str],
    disableAsmComments=False,
    errorTolerant=False,
    generateSourcesAndExit=False,
    compress=True,
):
    if globalParameters["PythonProfile"]:
        globalParameters["CpuThreads"] = 0
        printWarning("Python profiling is enabled. CpuThreads set to 0.")
        import yappi
        yappi.start()

    codeObjectFiles = []

    outputPath = Path(outputPath)
    destLibPath = ensurePath(
        outputPath / "library"
    )  # Destination for code object library files (.co)
    buildTmpPath = ensurePath(outputPath / "build_tmp" / outputPath.stem.upper())  #
    assemblyTmpPath = ensurePath(
        buildTmpPath / "assembly"
    )  # Temp path for generated assembly files (.s)
    objectTmpPath = ensurePath(
        buildTmpPath / "code_object_tmp"
    )  # Temp path for HSA code object files (.hsaco)

    asmKernels = [k for k in kernels if k["KernelLanguage"] == "Assembly"]

    visited = set()
    duplicates = 0
    for k in asmKernels:
        base = getKernelFileBase(splitGSU, k)
        k.duplicate = True if base in visited else False
        if not k.duplicate:
            k["BaseName"] = base
        duplicates += k.duplicate
        print2(f"Duplicate: {base}")
        visited.add(base)
    print1(f"Number of duplicate kernels: {duplicates}")

    outOptions = rocisa.rocIsa.getInstance().getOutputOptions()
    outOptions.outputNoComment = disableAsmComments

    numAsmKernels = len(asmKernels)
    numKernels = len(asmKernels)
    assert numKernels == numAsmKernels, "Only assembly kernels are supported in TensileLite"
    asmIter = zip(
        itertools.repeat(kernelWriterAssembly),
        itertools.repeat(rocisa.rocIsa.getInstance().getData()),
        itertools.repeat(outOptions),
        itertools.repeat(splitGSU),
        asmKernels
    )
    asmResults = ParallelMap2(processKernelSource, asmIter, "Generating assembly kernels", return_as="list")
    removeInvalidSolutionsAndKernels(
        asmResults, asmKernels, solutions, errorTolerant, getVerbosity(), splitGSU
    )
    passPostKernelInfoToSolution(
        asmResults, asmKernels, solutions, splitGSU
    )

    def assemble(ret):
        p, isa, wavefrontsize, result = ret
        asmToolchain.assembler(isaToGfx(isa), wavefrontsize, str(p), str(p.with_suffix(".o")))

    unaryWriteAssembly = functools.partial(writeAssembly, assemblyTmpPath)
    compose = lambda *F: functools.reduce(lambda f, g: lambda x: f(g(x)), F)
    ret = ParallelMap2(
        compose(assemble, unaryWriteAssembly),
        asmResults,
        "Writing assembly kernels",
        return_as="list",
        multiArg=False,
    )

    writeHelpers(outputPath, kernelHelperObjs, KERNEL_HELPER_FILENAME_CPP, KERNEL_HELPER_FILENAME_H)
    srcKernelFile = Path(outputPath) / "Kernels.cpp"

    if globalParameters["PythonProfile"]:
        yappi.stop()
        yappi.get_func_stats().save("yappi_results.profile", type="callgrind")
        with open("yappi_results.txt", "w") as f:
            yappi.get_func_stats().print_all(out=f)
        if globalParameters["CpuThreads"] != 0:
            with open("yappi_thread_stats.txt", "w") as f:
                yappi.get_thread_stats().print_all(out=f)

    if not generateSourcesAndExit:
        codeObjectFiles += buildAssemblyCodeObjectFiles(
            asmToolchain.linker,
            asmToolchain.bundler,
            asmKernels,
            destLibPath,
            assemblyTmpPath,
            compress,
        )
        buildSourceCodeObjectFiles(
            srcToolchain.compiler,
            srcToolchain.bundler,
            destLibPath,
            objectTmpPath,
            outputPath,
            srcKernelFile,
            cmdlineArchs,
        )

    return codeObjectFiles, numKernels


def writeSolutionsAndKernelsTCL(
    outputPath,
    asmToolchain,
    srcToolchain,
    solutions,
    kernels,
    kernelHelperObjs,
    kernelWriterAssembly,
    cmdlineArchs: List[str],
    disableAsmComments=False,
    compress=True,
):
    outputPath = Path(outputPath)
    destLibPath = ensurePath(
        outputPath / "library"
    )  # Destination for code object library files (.co)
    buildTmpPath = ensurePath(outputPath / "build_tmp" / outputPath.stem.upper())
    assemblyTmpPath = ensurePath(
        buildTmpPath / "assembly"
    )  # Temp path for generated assembly files (.s)
    objectTmpPath = ensurePath(
        buildTmpPath / "code_object_tmp"
    )  # Temp path for HSA code object files (.hsaco)

    asmKernels = [k for k in kernels if k["KernelLanguage"] == "Assembly"]

    visited = set()
    duplicates = 0
    splitGSU = False
    for k in asmKernels:
        base = getKernelFileBase(splitGSU, k)
        k["BaseName"] = base
        k.duplicate = True if base in visited else False
        duplicates += k.duplicate
        print2(f"Duplicate: {base}")
        visited.add(base)
    print1(f"Number of duplicate kernels: {duplicates}")

    uniqueAsmKernels = [k for k in asmKernels if not k.duplicate]

    def assemble(ret):
        p, isa, wavefrontsize, result = ret
        asmToolchain.assembler(isaToGfx(isa), wavefrontsize, str(p), str(p.with_suffix(".o")))
        return result

    outOptions = rocisa.rocIsa.getInstance().getOutputOptions()
    outOptions.outputNoComment = not disableAsmComments

    unaryProcessKernelSource = functools.partial(
        processKernelSource,
        kernelWriterAssembly,
        rocisa.rocIsa.getInstance().getData(),
        outOptions,
        splitGSU,
    )

    unaryWriteAssembly = functools.partial(writeAssembly, assemblyTmpPath)
    compose = lambda *F: functools.reduce(lambda f, g: lambda x: f(g(x)), F)
    ret = ParallelMap2(
        compose(assemble, unaryWriteAssembly, unaryProcessKernelSource),
        uniqueAsmKernels,
        "Generating assembly kernels",
        multiArg=False,
        return_as="list"
    )
    passPostKernelInfoToSolution(
        ret, uniqueAsmKernels, solutions, splitGSU
    )
    # result.src is very large so let garbage collector know to clean up
    del ret
    buildAssemblyCodeObjectFiles(
        asmToolchain.linker,
        asmToolchain.bundler,
        asmKernels,
        destLibPath,
        assemblyTmpPath,
        compress,
    )

    writeHelpers(outputPath, kernelHelperObjs, KERNEL_HELPER_FILENAME_CPP, KERNEL_HELPER_FILENAME_H)
    srcKernelFile = Path(outputPath) / "Kernels.cpp"
    buildSourceCodeObjectFiles(
        srcToolchain.compiler,
        srcToolchain.bundler,
        destLibPath,
        objectTmpPath,
        outputPath,
        srcKernelFile,
        cmdlineArchs,
    )

    return len(uniqueAsmKernels)


@timing
def copyStaticFiles(outputPath):
    libraryStaticFiles = [
        "TensileTypes.h",
        "tensile_bfloat16.h",
        "tensile_float8_bfloat8.h",
        "KernelHeader.h",
        "ReductionTemplate.h",
        "memory_gfx.h",
    ]

    for fileName in libraryStaticFiles:
        shutil.copy(os.path.join(SOURCE_PATH, fileName), outputPath)

    return libraryStaticFiles


@timing
def generateKernelObjectsFromSolutions(solutions):
    kernels = []
    kernelNames = set()
    for solution in solutions:
        solutionKernels = solution.getKernels()
        for kernel in solutionKernels:
            kName = getKeyNoInternalArgs(kernel, False)
            if kName not in kernelNames:
                kernels.append(kernel)
                kernelNames.add(kName)
    return kernels


def generateKernelHelperObjects(solutions: List[Solution], cxxCompiler: str, isaInfoMap):
    """
    Generates a unique list of kernel helpers.

    Kernel helpers are used to generate hip source code kernels called
    before/after gemm kernels. This function creates a minimal list of
    kernel helpers required to support the solutions reuested in a build.
    The list of kernel helpers is then used to write Kernels.cpp/h to
    disk. To ensure the ActivationEnumHeaders are written first, the
    list is sorted such that those kernel helpers appear first.

    Args:
        solutions: a list of solutions to process.
        cxxCompiler: the full path to the cxxCompiler.

    Returns:
        List of kernel helpers.
    """
    khos = []
    visited = set()
    for solution in solutions:
        for kernelHelperType, callable in kernelObjectNameCallables():
            buildMask = []
            names = callable(solution)
            if names:
                sortByEnum = lambda x: ("Enum" in x, names.index(x))
                names = sorted(names, key=sortByEnum, reverse=True)
                for name in names:
                    if name not in visited:
                        visited.add(name)
                        buildMask.append(True)
                    else:
                        buildMask.append(False)
                if any(buildMask):
                    kho = initHelperKernelObjects(solution, kernelHelperType, cxxCompiler, isaInfoMap)
                    kho = list(itertools.compress(kho, buildMask))
                    if kho:
                        khos.extend(kho)
    khos = list(set(khos))
    sortByEnum = lambda x: ("Enum" in x.getKernelName(), khos.index(x))
    return sorted(khos, key=sortByEnum, reverse=True) # Ensure that we write Enum kernel helpers are first in list


@timing
def generateLogicDataAndSolutions(logicFiles, args, assembler: Assembler, isaInfoMap):

    if ";" in args["Architecture"]:
        archs = args["Architecture"].split(";")  # user arg list format
    else:
        archs = args["Architecture"].split("_")  # workaround for cmake list in list issue

    solutions = []
    masterLibraries = {}
    nextSolIndex = 0
    splitGSU = False
    printSolutionRejectionReason = True
    printIndexAssignmentInfo = False

    fIter = zip(
        logicFiles,
        itertools.repeat(assembler),
        itertools.repeat(splitGSU),
        itertools.repeat(printSolutionRejectionReason),
        itertools.repeat(printIndexAssignmentInfo),
        itertools.repeat(isaInfoMap),
        itertools.repeat(args["LazyLibraryLoading"]),
    )

    def libraryIter(lib: MasterSolutionLibrary):
        if len(lib.solutions):
            for i, s in enumerate(lib.solutions.items()):
                yield (i, *s)
        else:
            for _, lazyLib in lib.lazyLibraries.items():
                yield from libraryIter(lazyLib)

    for library in ParallelMap2(
        LibraryIO.parseLibraryLogicFile, fIter, "Loading Logics...", return_as="generator_unordered"
    ):
        _, architectureName, _, _, _, newLibrary = library

        if architectureName == "":
            continue

        if architectureName in masterLibraries:
            nextSolIndex = masterLibraries[architectureName].merge(newLibrary, nextSolIndex)
        else:
            masterLibraries[architectureName] = newLibrary
            masterLibraries[architectureName].version = args["CodeObjectVersion"]

    # Sort masterLibraries to make global soln index values deterministic
    solnReIndex = 0
    masterLibraries = dict(sorted(masterLibraries.items()))
    for _, masterLibrary in masterLibraries.items():
        for _, sol in masterLibrary.solutions.items():
            sol.index = solnReIndex
            solnReIndex += 1
        # Sort masterLibrary to make global soln index values deterministic
        masterLibrary.lazyLibraries = dict(sorted(masterLibrary.lazyLibraries.items()))
        for name, lib in masterLibrary.lazyLibraries.items():
            # Sort solns by the lib logic file they were generated from
            lib.solutions = {
                k: lib.solutions[k]
                for k in sorted(lib.solutions, key=lambda idx: lib.solutions[idx].srcName)
            }
            for _, sol in lib.solutions.items():
                sol.index = solnReIndex
                solnReIndex += 1

    if args["GenSolTable"]:
        matchTable = {}
        # Match yaml file solutions to solution index
        for _, masterLibrary in masterLibraries.items():
            for _, _, s in libraryIter(masterLibrary):
                matchTable[s.index] = [s.srcName, s.libraryLogicIndex]
        LibraryIO.write("MatchTable", matchTable)

    if "fallback" in masterLibraries.keys():
        for key, value in masterLibraries.items():
            if key != "fallback":
                value.merge(masterLibraries["fallback"])
        masterLibraries.pop("fallback")
    solIndex = []
    for _, masterLibrary in masterLibraries.items():
        for _, sol in masterLibrary.solutions.items():
            solutions.append(sol.originalSolution)
            solIndex.append(sol.index)
        for name, lib in masterLibrary.lazyLibraries.items():
            for _, sol in lib.solutions.items():
                sol.originalSolution._state["codeObjectFile"] = name
                solutions.append(sol.originalSolution)
                solIndex.append(sol.index)

    # Get the solution index and it's codeObjectFile name
    codeObjectFilesIndex = {}
    for solution, index in zip(solutions, solIndex):
        if "codeObjectFile" in solution._state and solution._state["codeObjectFile"] is not None:
            if solution._state["codeObjectFile"] in codeObjectFilesIndex:
                codeObjectFilesIndex[solution._state["codeObjectFile"]] = min(index, codeObjectFilesIndex[solution._state["codeObjectFile"]])
            else:
                codeObjectFilesIndex[solution._state["codeObjectFile"]] = index

    # Reorder to int: name format
    codeObjectFilesIndex = {v: k for k, v in codeObjectFilesIndex.items()}
    # Reorder to maintain ascending order by index
    codeObjectFilesIndex = dict(sorted(codeObjectFilesIndex.items()))

    # remove duplicates while preserving order
    numSoln = len(solutions)
    solutions = dict.fromkeys(solutions).keys()

    print1(f"Number of solutions parsed: {numSoln}")
    print1(f"Number of unique solutions: {len(solutions)}")

    return solutions, masterLibraries, codeObjectFilesIndex


################################################################################
# Tensile Create Library
################################################################################
@profile
def run():
    start = timer()
    print1("")
    print1(HR)
    print1("# Tensile Create Library")
    print2(HR)
    print2("")

    arguments = parseArguments()
    setVerbosity(arguments["PrintLevel"])
    outputPath = Path(ensurePath(os.path.abspath(arguments["OutputPath"])))
    cxxCompiler, _, offloadBundler, _, _ = validateToolchain(
        arguments["CxxCompiler"],
        arguments["CCompiler"],
        arguments["OffloadBundler"],
        arguments["Assembler"],
        ToolchainDefaults.HIP_CONFIG,
    )

    if ";" in arguments["Architecture"]:
        archs = arguments["Architecture"].split(";")
    else:
        archs = arguments["Architecture"].split("_")
    archs = SUPPORTED_GFX if "all" in archs else archs
    archs, requestedPredicateMap = splitArchsFromPredicates(archs)

    targetIsas = [gfxToIsa(a) for a in archs]
    isaInfoMap = makeIsaInfoMap(targetIsas, cxxCompiler)
    assignGlobalParameters(arguments, isaInfoMap)

    asmToolchain = makeAssemblyToolchain(
        cxxCompiler,
        offloadBundler,
        arguments["CodeObjectVersion"],
        arguments["BuildIdKind"],
        arguments["AsmDebug"],
    )
    srcToolchain = makeSourceToolchain(
        cxxCompiler,
        offloadBundler,
        arguments["AsanBuild"],
        arguments["BuildIdKind"],
        save_temps=False
    )

    print1(asmToolchain.assembler)
    print1(asmToolchain.bundler)

    if not os.path.exists(arguments["LogicPath"]):
        printExit(f"LogicPath {arguments['LogicPath']} doesn't exist")

    logicExtFormat = ".yaml"
    if arguments["LogicFormat"] == "yaml":
        pass
    elif arguments["LogicFormat"] == "json":
        logicExtFormat = ".json"
    else:
        printExit(f"Unrecognized LogicFormat: {arguments['LogicFormat']}")

    def archMatch(arch: str, archs: List[str]):
        return (arch in archs) or any(a.startswith(arch) for a in archs)

    def validLogicFile(p: Path):
        return p.suffix == logicExtFormat and (
            "all" in archs or archMatch(load_logic_gfx_arch(p), archs)
        )

    globPattern = os.path.join(
        arguments["LogicPath"], f"**/{arguments['LogicFilter']}{logicExtFormat}"
    )
    print1(f"# LogicFilter:       {globPattern}")
    logicFiles = [
        file for file in glob.iglob(globPattern, recursive=True)
    ]

    logicFiles = [file for file in logicFiles if validLogicFile(Path(file))]

    print1(f"# Experimental:      {arguments['Experimental']}")
    if not arguments["Experimental"]:
        logicFiles = [
            file for file in logicFiles if "experimental" not in map(str.lower, Path(file).parts)
        ]

    print1("# Archs: " + ' ,'.join(archs))
    if requestedPredicateMap:
        print1("# Predicates:\n" + "\n".join(f"#   {arch}: {', '.join(v) if v else 'all variants'}" for arch, v in requestedPredicateMap.items()))
        numPrior = len(logicFiles)
        logicFiles = filterLogicFilesByPredicates(logicFiles, requestedPredicateMap)
        print1(f"# Filtered {numPrior - len(logicFiles)} logic files not matching requested predicates")

    print1(f"# LibraryLogicFiles: {len(logicFiles)}")

    for logicFile in logicFiles:
        print2("#   %s" % logicFile)

    start_glds = timer()
    solutions, masterLibraries, libraryMapping = generateLogicDataAndSolutions(
        logicFiles, arguments, asmToolchain.assembler, isaInfoMap
    )
    stop_glds = timer()
    print(f"Time to load yaml files (s): {(stop_glds-start_glds):3.2f}")


    kernels = generateKernelObjectsFromSolutions(solutions)
    kernelHelperObjs = generateKernelHelperObjects(kernels, str(asmToolchain.assembler.path), isaInfoMap)
    kernelWriterAssembly = KernelWriterAssembly(asmToolchain.assembler, DebugConfig())

    copyStaticFiles(outputPath)

    start_wsk = timer()
    numKernels = writeSolutionsAndKernelsTCL(
        outputPath,
        asmToolchain,
        srcToolchain,
        solutions,
        kernels,
        kernelHelperObjs,
        kernelWriterAssembly,
        archs,
        arguments["DisableAsmComments"],
        compress=arguments["UseCompression"],
    )
    stop_wsk = timer()
    print(f"Time to generate kernels (s): {(stop_wsk-start_wsk):3.2f}")

    archs = [ # is this really different than the other archs above?
        isaToGfx(arch)
        for arch in targetIsas
        if isaInfoMap[arch].asmCaps["SupportedISA"]
    ]
    newLibraryDir = ensurePath(os.path.join(outputPath, "library"))
    splitGSU = False

    solDict = {}
    for solution in solutions:
        solutionKernels = solution.getKernels()
        for kernel in solutionKernels:
            kName = getKeyNoInternalArgs(kernel, False)
            if kName not in solDict:
                solDict["%s"%kName] = kernel

    def writeMsl(name, lib):
        filename = os.path.join(newLibraryDir, name)
        lib.applyNaming(splitGSU)
        LibraryIO.write(filename, state(lib), arguments["LibraryFormat"])

    filename = os.path.join(newLibraryDir, "TensileLiteLibrary_lazy_Mapping")
    LibraryIO.write(filename, libraryMapping, "msgpack")

    start_msl = timer()
    for archName, newMasterLibrary in masterLibraries.items():
        if archName in archs:
            if arguments["LazyLibraryLoading"]:
                masterFile = os.path.join(newLibraryDir, "TensileLibrary_lazy_" + archName)
            else:
                masterFile = os.path.join(newLibraryDir, "TensileLibrary_" + archName)
            newMasterLibrary.applyNaming(splitGSU)
            LibraryIO.write(masterFile, state(newMasterLibrary), arguments["LibraryFormat"])

            for name, lib in newMasterLibrary.lazyLibraries.items():
                for k, s in lib.solutions.items():
                    kName = getKeyNoInternalArgs(s.originalSolution, splitGSU)
                    s.sizeMapping.CUOccupancy = solDict["%s"%kName]["CUOccupancy"]

            ParallelMap2(writeMsl,
                         newMasterLibrary.lazyLibraries.items(),
                         "Writing master solution libraries",
                         return_as="list")
    stop_msl = timer()
    print(f"Time to write master solution libraries (s): {(stop_msl-start_msl):3.2f}")

    if not arguments["KeepBuildTmp"]:
        buildTmp = Path(arguments["OutputPath"]).parent / "library" / "build_tmp"
        if buildTmp.exists() and buildTmp.is_dir():
            shutil.rmtree(buildTmp)
        buildTmp = Path(arguments["OutputPath"]) / "build_tmp"
        if buildTmp.exists() and buildTmp.is_dir():
            shutil.rmtree(buildTmp)
        else:
            printWarning(f"Cannot remove build_tmp")

    print("# Tensile Library Writer DONE")
    print(HR)
    print("")

    stop = timer()

    print(f"Total time (s): {(stop-start):3.2f}")
    print(f"Total kernels processed: {numKernels}")
    print(f"Kernels processed per second: {(numKernels/(stop-start)):3.2f}")
    print(f"KernelHelperObjs: {len(kernelHelperObjs)}")
