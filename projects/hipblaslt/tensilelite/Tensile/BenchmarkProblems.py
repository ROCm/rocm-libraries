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

import os
import shutil
import sys
import time

from copy import deepcopy
from pathlib import Path
from typing import Dict

from Tensile import CUSTOM_KERNEL_PATH, SolutionLibrary, LibraryIO
from Tensile.KernelWriter import DebugConfig
from Tensile.KernelHelperNaming import KernelHelperEnum, initHelperKernelObjects
from Tensile.Toolchain.Component import Assembler
from Tensile.SolutionStructs.Problem import ProblemType, ProblemSizes
from Tensile.SolutionStructs.Solution import Solution
from Tensile.SolutionStructs.Validators.MatrixInstruction import matrixInstructionToMIParameters, \
                                                                 validateMIParameters
from Tensile.SolutionStructs.Naming import getKeyNoInternalArgs, getSolutionNameMin, getKernelNameMin

from .BenchmarkStructs import BenchmarkProcess, constructForkPermutations
from .Contractions import ProblemType as ContractionsProblemType
from .ClientWriter import runClient, writeClientConfig, writeClientConfigIni, getClientExecutablePath
from .KernelWriterAssembly import KernelWriterAssembly
from .TensileCreateLibrary import copyStaticFiles, writeSolutionsAndKernels
from .CustomKernels import getCustomKernelConfig
from .Toolchain.Assembly import AssemblyToolchain
from .Toolchain.Source import SourceToolchain
from Tensile.Common import HR, print1, print2, IsaInfo, IsaVersion, \
        printExit, printWarning, ensurePath, tqdm, state, \
        BENCHMARK_PROBLEMS_DIR, BENCHMARK_DATA_DIR
from Tensile.Common.Architectures import isaToGfx, gfxToVariants
from Tensile.Common.GlobalParameters import globalParameters, startTime


def _generateForkedSolutions(problemType, constantParams, forkPermutations, assembler: Assembler, \
                            debugConfig: DebugConfig, isaInfoMap: Dict[IsaVersion, IsaInfo]):
    """Creates a list with a Solution object for each parameter combination in forkPermutations"""
    print1("# Enumerating Solutions")

    solutions = []
    solutionSet = set()
    for perm in forkPermutations:
        # Expect only a single ISA in the map for the Tensile context
        # because the GPU has to be physically present for benchmarking

        solution = {
            "ProblemType": deepcopy(problemType.state),
            "ISA": next(iter(isaInfoMap.keys()))
        }
        solution.update(constantParams)
        solution.update(perm)


        mi = solution["MatrixInstruction"]
        wavefrontSize = solution["WavefrontSize"]
        workgroup = solution["WorkGroup"]
        ptype = solution["ProblemType"]
        isa = solution["ISA"]

        if len(mi) == 9:
            miParams = matrixInstructionToMIParameters(mi, isa, wavefrontSize, ptype, workgroup, isaInfoMap)
            solution.update(miParams)
        elif len(mi) == 0:
            solution["EnableMatrixInstruction"] = False

        if validateMIParameters(solution, isaInfoMap):
            solutionObject = Solution(
                solution,
                debugConfig.splitGSU,
                debugConfig.printSolutionRejectionReason,
                debugConfig.printIndexAssignmentInfo,
                assembler,
                isaInfoMap
            )
            if solutionObject["Valid"]:
                if solutionObject not in solutionSet:
                    solutionSet.add(solutionObject)
                    solutions.append(solutionObject)
        elif debugConfig.printSolutionRejectionReason:
            print1("rejecting solution " + str(solution))

    return solutions


def _getCustomKernelSolutionObj(
        kernelName,
        internalSupportParams,
        assembler: Assembler,
        debugConfig: DebugConfig,
        isaInfoMap: Dict[IsaVersion, IsaInfo],
        directory=CUSTOM_KERNEL_PATH
    ):
    """Creates the Solution object for a custom kernel"""
    sol = getCustomKernelConfig(kernelName, internalSupportParams, directory)

    mi = sol["MatrixInstruction"]
    isa = next(iter(isaInfoMap.keys()))
    wavefrontSize = sol["WavefrontSize"]
    ptype = sol["ProblemType"]
    workgroup = sol.get("WorkGroup", None)

    if len(mi) == 9:
        miParams = matrixInstructionToMIParameters(mi, isa, wavefrontSize, ptype, workgroup, isaInfoMap)
        sol.update(miParams)
    elif len(mi) == 0:
        sol["EnableMatrixInstruction"] = False

    sol = Solution(
               sol,
               debugConfig.printIndexAssignmentInfo,
               debugConfig.printSolutionRejectionReason,
               debugConfig.printIndexAssignmentInfo,
               assembler,
               isaInfoMap
           )

    return sol


def _generateCustomKernelSolutions(
        problemType,
        customKernels,
        internalSupportParams,
        failOnMismatch,
        assembler: Assembler,
        debugConfig: DebugConfig,
        isaInfoMap: Dict[str, IsaInfo]
    ):
    """Creates a list with a Solution object for each name in customKernel"""
    solutions = []
    for kernelName in customKernels:
        print1("# Processing custom kernel {}".format(kernelName))
        solution = _getCustomKernelSolutionObj(kernelName, internalSupportParams, assembler, debugConfig, isaInfoMap)
        # The ActivationType setting in YAML is meaningless in customKernel case.
        # Therefore, we override the customKernel setting with the ActivationType value from ProblemType to avoid false alarms during subsequent problemType checks.
        solution["ProblemType"]["ActivationType"] = problemType["ActivationType"]
        if solution["ProblemType"] != problemType:
            # Raise error if this kernel was specifically requested and problem type doesn't match
            if failOnMismatch:
                benchmarkSet = set([(k,tuple(v)) if type(v) is list else (k,v) \
                        for k,v in problemType.items()])
                customSet = set([(k,tuple(v)) if type(v) is list else (k,v) \
                        for k,v in solution["ProblemType"].items()])

                msg = "The problem type in the config file does not match " \
                        "that of the custom kernel, {}.".format(kernelName) \
                        + "\nDiffering parameters:\n" \
                        + "\tConfig values:\n\t" \
                        + str(sorted(benchmarkSet - (customSet & benchmarkSet))) \
                        + "\n\tCustom kernel values:\n\t" \
                        +  str(sorted(customSet - (customSet & benchmarkSet)))
                printExit(msg)
            else:
                print1("# Rejected {}: Problem Type doesn't match".format(kernelName))
        else:
            print1("# Added {} to solutions".format(kernelName))
            if solution["Valid"]:
                solutions.append(solution)
            elif debugConfig.printSolutionRejectionReason:
                print1("rejecting solution " + str(solution))

    return solutions

def writeBenchmarkFiles(
        stepBaseDir,
        solutions,
        problemSizes,
        biasTypeArgs,
        factorDimArgs,
        activationArgs,
        icacheFlushArgs,
        stepName,
        solutionSummationSizes,
        asmToolchain: AssemblyToolchain,
        srcToolchain: SourceToolchain,
        sourcePath: Path,
        debugConfig: DebugConfig,
        deviceId: int,
        gfxName: str,
        isaInfoMap: Dict[IsaVersion, IsaInfo],
        probSolMap: dict
    ):
    """Write all the files needed for a given benchmarking step"""

    ensurePath(sourcePath)
    copyStaticFiles(sourcePath)

    kernels = []
    kernelHelperObjs = []
    kernelNames = set()
    kernelHelperNames = set()

    # get unique kernels and kernel helpers
    for solution in tqdm(solutions, "Finding unique solutions"):
        solutionKernels = solution.getKernels()
        for kernel in solutionKernels:
            kName = getKeyNoInternalArgs(kernel, debugConfig.splitGSU)
            if kName not in kernelNames:
                kernels.append(kernel)
                kernelNames.add(kName)

        solutionHelperKernels = initHelperKernelObjects(solution,
                                                        KernelHelperEnum.All,
                                                        str(asmToolchain.assembler.path),
                                                        isaInfoMap)
        for ko in solutionHelperKernels:
            kname = ko.getKernelName()
            if kname not in kernelHelperNames:
                kernelHelperObjs.append(ko)
                kernelHelperNames.add(kname)

    kernelWriterAssembly = KernelWriterAssembly(asmToolchain.assembler, debugConfig)

    cmdLineArchs = [var for isa in isaInfoMap.keys() for var in gfxToVariants(isaToGfx(isa))]
    # cmdLineArchs = [variant isaToGfx(isa) for isa in isaInfoMap.keys() for gfxToVariants()]
    # write solution, kernels and CMake
    problemType = solutions[0]["ProblemType"]
    codeObjectFiles, _= writeSolutionsAndKernels( \
                            sourcePath,
                            asmToolchain,
                            srcToolchain,
                            solutions,
                            kernels,
                            kernelHelperObjs,
                            kernelWriterAssembly,
                            debugConfig.splitGSU,
                            cmdLineArchs,
                            disableAsmComments=globalParameters["DisableAsmComments"],
                            errorTolerant=True,
                            generateSourcesAndExit=globalParameters["GenerateSourcesAndExit"], # put in debug config
                            compress=False,
                        )
    # ^ this is where solutions is mutated
    for s in solutions:
        s["SolutionNameMin"] = getSolutionNameMin(solution, debugConfig.splitGSU)
        s["KernelNameMin"]   = getKernelNameMin(solution, debugConfig.splitGSU)

    newLibraryDir = ensurePath(sourcePath / 'library')
    newLibraryFile = os.path.join(newLibraryDir, "TensileLibrary")
    newLibrary = SolutionLibrary.MasterSolutionLibrary.BenchmarkingLibrary(
                     solutions,
                     asmToolchain.assembler,
                     debugConfig.splitGSU,
                     debugConfig.printSolutionRejectionReason,
                     debugConfig.printIndexAssignmentInfo,
                     isaInfoMap,
                 )
    newLibrary.applyNaming(debugConfig.splitGSU)
    LibraryIO.write(newLibraryFile, state(newLibrary), globalParameters["LibraryFormat"])

    codeObjectFiles = [os.path.relpath(f, sourcePath) \
            for f in codeObjectFiles]

    if "TileAwareSelection" in problemType and problemType["TileAwareSelection"]:
        maxMacroTile0 = 0
        maxMacroTile1 = 0
        for solution in solutions:
            macroTile0 = solution["MacroTile0"]
            macroTile1 = solution["MacroTile1"]
            if macroTile0 > maxMacroTile0:
                maxMacroTile0 = macroTile0
            if macroTile1 > maxMacroTile1:
                maxMacroTile1 = macroTile1
        idealM = 36 * maxMacroTile0
        idealN = 36 * maxMacroTile1
        idealSizes = []
        if problemType["Batched"]:
            for idealK in solutionSummationSizes:
                idealSize = {"Exact": [idealM, idealN, 1, idealK]}
                idealSizes.append(idealSize)
        else:
            for idealK in solutionSummationSizes:
                idealSize = {"Exact": [idealM, idealN, idealK]}
                idealSizes.append(idealSize)
        idealProblemSizes = ProblemSizes(problemType, idealSizes)
        writeClientConfig(True, solutions, idealProblemSizes, biasTypeArgs, \
                          factorDimArgs, activationArgs, icacheFlushArgs, stepName, stepBaseDir, \
                          newLibrary, codeObjectFiles, True, deviceId, gfxName, probSolMap=probSolMap)
    else:
        writeClientConfig(True, solutions, problemSizes, biasTypeArgs, \
                          factorDimArgs, activationArgs, icacheFlushArgs, stepName, stepBaseDir, \
                          newLibrary, codeObjectFiles, False, deviceId, gfxName, probSolMap=probSolMap)

    if len(solutions) == 0:
        printExit("write solutions and kernels results 0 valid soultion.")

    return codeObjectFiles


def _benchmarkProblemType(problemTypeConfig, problemSizeGroupConfig, problemSizeGroupIdx, useCache,
                         asmToolchain: AssemblyToolchain, srcToolchain: SourceToolchain, cCompiler: str,
                         buildTmpPath: Path, benchmarkProblemsPath: Path,
                         debugConfig: DebugConfig, deviceId: int,
                         gfxName: str, isaInfoMap: Dict[str, IsaInfo], probSolMap: dict
    ):
    """Run the benchmarking for a single entry in the BenchmarkProblems of a Tensile config"""
    benchmarkTestFails = 0

    print1("")
    print1(HR)
    print1("# Converting Config to BenchmarkProcess Object")
    print1(HR)
    print1("")
    benchmarkProcess = BenchmarkProcess(problemTypeConfig, problemSizeGroupConfig, debugConfig.printIndexAssignmentInfo)

    enableTileSelection = benchmarkProcess.problemType["TileAwareSelection"]
    groupName = "{}_{:02d}".format(str(benchmarkProcess.problemType), problemSizeGroupIdx)
    groupNamePath = benchmarkProblemsPath / groupName
    ensurePath(groupNamePath / "Data")

    totalBenchmarkSteps = len(benchmarkProcess)
    resultsFileBaseFinal = None

    print1("# NumBenchmarkSteps: {}".format(totalBenchmarkSteps))
    print1("")
    print1(HR)
    print1("# Done Creating BenchmarkProcess Object")
    print1(HR)

    for benchmarkStepIdx in range(0, totalBenchmarkSteps):
        benchmarkStep = benchmarkProcess[benchmarkStepIdx]
        stepName = str(benchmarkStep)
        shortName = stepName

        print1("\n")
        print1(HR)
        currentTime = time.time()
        elapsedTime = currentTime - startTime
        print1("# Benchmark Step: {} - {} {:.3f}s".format(groupName, stepName, elapsedTime))
        print1("# Num Sizes: {}".format(benchmarkStep.problemSizes.totalProblemSizes))
        print1("# Factor Dim steps: {}".format(benchmarkStep.factorDimArgs.totalProblemSizes))
        print1("# Bias Type steps: {}".format(benchmarkStep.biasTypeArgs.totalProblemSizes))
        print1("# Activation steps: {}".format(benchmarkStep.activationArgs.totalProblemSizes))
        print1("# ICacheFlush steps: {}".format(len(benchmarkStep.icacheFlushArgs)))
        print1("# Fork Parameters:")
        for k, v in benchmarkStep.forkParams.items():
            print1("#     {}: {}".format(k, v))
        if benchmarkStep.internalSupportParams:
            print("# InternalSupportParams: {}".format(benchmarkStep.internalSupportParams))

        shortNamePath = ensurePath(groupNamePath / shortName)
        stepBaseDir = shortNamePath
        resultsFileBase = os.path.normpath(shortNamePath / ".." / "Data" / shortName)

        if benchmarkStep.isFinal():
            resultsFileBaseFinal = resultsFileBase
        resultsFileName = resultsFileBase + ".csv"
        solutionsFileName = resultsFileBase + ".yaml"

        # check if a solution cache exists and if it matches our solution parameters
        cachePath = os.path.join(stepBaseDir, "cache.yaml")
        sourcePath = ensurePath(shortNamePath / "source")

        cacheValid = False
        if useCache and os.path.isfile(cachePath):
            c = LibraryIO.read(cachePath)
            if c["ConstantParams"] == benchmarkStep.constantParams and \
                    c["ForkParams"] == benchmarkStep.forkParams and \
                    c["ParamGroups"] == benchmarkStep.paramGroups and \
                    c["CustomKernels"] == benchmarkStep.customKernels and \
                    c["InternalSupportParams"] == benchmarkStep.internalSupportParams and \
                    c["CustomKernelWildcard"] == benchmarkStep.customKernelWildcard:
                cacheValid = True
                codeObjectFiles = c["CodeObjectFiles"]
            else:
                printWarning("Cache data does not match config: redoing solution generation")

        if not cacheValid:
            # enumerate benchmark permutations and create resulting solution objects
            forkPermutations = constructForkPermutations(benchmarkStep.forkParams, \
                    benchmarkStep.paramGroups) if problemSizeGroupConfig["ForkParameters"] else []
            maxPossibleSolutions = len(forkPermutations)

            regSolutions = _generateForkedSolutions(benchmarkProcess.problemType, \
                    benchmarkStep.constantParams, forkPermutations, asmToolchain.assembler, \
                        debugConfig, isaInfoMap)
            kcSolutions = _generateCustomKernelSolutions(benchmarkProcess.problemType, \
                    benchmarkStep.customKernels, benchmarkStep.internalSupportParams, \
                    not benchmarkStep.customKernelWildcard, asmToolchain.assembler, debugConfig, \
                        isaInfoMap)

            maxPossibleSolutions += len(kcSolutions)
            solutions = regSolutions + kcSolutions

            print1("# Actual Solutions: {} / {} after SolutionStructs\n" \
                .format(len(solutions), maxPossibleSolutions))

            # handle no valid solutions
            if len(solutions) == 0:
                msg = "Your parameters resulted in 0 valid solutions."
                if debugConfig.printSolutionRejectionReason:
                    msg += "\nExamine reject and backtrace messages above to see why" \
                            "and where solutions were rejected."
                else:
                    msg += "\nYou should re-run with \"PrintSolutionRejectionReason: True\"" \
                            "to see why each parameter combination was rejected."
                printExit(msg)

            for solution in solutions:
                print2("#    ({}:{}) {}".format(0, 0, getSolutionNameMin(solution, debugConfig.splitGSU)))
            print2(HR)

            # write benchmarkFiles
            prevCount = len(solutions)
            codeObjectFiles = writeBenchmarkFiles(stepBaseDir, solutions, \
                    benchmarkStep.problemSizes, benchmarkStep.biasTypeArgs, \
                    benchmarkStep.factorDimArgs, benchmarkStep.activationArgs, \
                    benchmarkStep.icacheFlushArgs, shortName, [], asmToolchain, srcToolchain, \
                    sourcePath, debugConfig, deviceId, gfxName, isaInfoMap, probSolMap)
            # ^ this mutates solutions

            # write cache data
            cacheData = {
                "CodeObjectFiles": codeObjectFiles,
                "ConstantParams": benchmarkStep.constantParams,
                "ForkParams": benchmarkStep.forkParams,
                "ParamGroups": benchmarkStep.paramGroups,
                "CustomKernels": benchmarkStep.customKernels,
                "CustomKernelWildcard": benchmarkStep.customKernelWildcard
            }
            LibraryIO.writeYAML(cachePath, cacheData)

            print1("# Actual Solutions: {} / {} after KernelWriter\n" \
                    .format(len(solutions), prevCount ))

            # add SolutionIndex and SolutionNameMin into benchmark yaml
            for i in range(0, len(solutions)):
                solution = solutions[i]
                solution["SolutionIndex"] = i
                solution["SolutionNameMin"] = getSolutionNameMin(solution, debugConfig.splitGSU)
                solution["KernelNameMin"]   = getKernelNameMin(solution, debugConfig.splitGSU)
        else:
            solutions = None
            print1("# Using cached solution data")

            ssProblemType = ProblemType(problemTypeConfig, debugConfig.printIndexAssignmentInfo)
            conProblemType = ContractionsProblemType.FromOriginalState(ssProblemType)
            outFile = os.path.join(sourcePath, "ClientParameters.ini")

            writeClientConfigIni(True, benchmarkStep.problemSizes, benchmarkStep.biasTypeArgs,
                                 benchmarkStep.factorDimArgs, benchmarkStep.activationArgs,
                                 benchmarkStep.icacheFlushArgs, conProblemType,
                                 stepBaseDir, codeObjectFiles, resultsFileName,
                                 outFile, deviceId, probSolMap=probSolMap)

        # I think the size portion of this yaml could be removed,
        # but for now it's needed, so we update it even in the cache case
        LibraryIO.writeSolutions(solutionsFileName, benchmarkStep.problemSizes, benchmarkStep.biasTypeArgs,
            benchmarkStep.activationArgs, solutions, cacheValid)

        # run benchmarking client
        if not os.path.exists(resultsFileName) or globalParameters["ForceRedoBenchmarkProblems"]:
            libraryLogicPath = None
            forBenchmark = True
            returncode = runClient(libraryLogicPath, forBenchmark, enableTileSelection, srcToolchain.compiler, cCompiler, shortNamePath)

            if returncode:
                benchmarkTestFails += 1
                printWarning("BenchmarkProblems: Benchmark Process exited with code {}" \
                        .format(returncode))
        else:
            print1("# Already benchmarked; skipping.")

        # End Iteration
        currentTime = time.time()
        elapsedTime = currentTime - startTime
        print1("{}\n# {}\n# {}: End - {:.3f}s\n{}\n" \
                .format(HR, groupName, shortName, elapsedTime, HR))

    return (resultsFileBaseFinal, benchmarkTestFails)


def main(
    config,
    useCache,
    asmToolchain: AssemblyToolchain,
    srcToolchain: SourceToolchain,
    cCompiler: str,
    outputPath: Path,
    buildTmpPath: Path,
    debugConfig: DebugConfig,
    deviceId: int,
    gfxName: str,
    isaInfoMap: Dict[str, IsaInfo],
    probSolMap: dict
):
    """Entry point for the "BenchmarkProblems" section of a Tensile config yaml"""
    getClientExecutablePath()

    if config is None:
        print(f'No config specified in {globalParameters["ConfigPath"]}, built client only')
        return

    benchmarkDataPath = ensurePath(outputPath / BENCHMARK_DATA_DIR)

    totalTestFails = 0
    for benchmarkProblemTypeConfig in config:
        problemTypeConfig = benchmarkProblemTypeConfig[0]
        if len(benchmarkProblemTypeConfig) < 2:
            problemSizeGroupConfigs = [{}]
        else:
            problemSizeGroupConfigs = benchmarkProblemTypeConfig[1:]

        for idx, sizeGroupConfig in enumerate(problemSizeGroupConfigs):
            print2("ProblemTypeConfig: {}".format(problemTypeConfig))
            problemTypeObj = ProblemType(problemTypeConfig, debugConfig.printIndexAssignmentInfo)

            # using a suffix to check the csv version (for later addFromCSV())
            csvSuffix = "_CSVWinner" if globalParameters["CSVExportWinner"] else ""
            # results files will be named
            newResultsFileName = os.path.join(benchmarkDataPath, "{}_{:02d}{}.csv" \
                    .format(str(problemTypeObj), idx, csvSuffix) )
            newSolutionsFileName = os.path.join(benchmarkDataPath, "{}_{:02d}{}.yaml" \
                    .format(str(problemTypeObj), idx, csvSuffix) )
            newGranularityFileName = os.path.join(benchmarkDataPath, "{}_{:02d}{}.gsp" \
                    .format(str(problemTypeObj), idx, csvSuffix) )

            # skip if possible
            if globalParameters["ForceRedoBenchmarkProblems"] \
                    or not os.path.exists(newResultsFileName):

                # benchmark problem size group
                benchmarkProblemsPath = ensurePath(outputPath / BENCHMARK_PROBLEMS_DIR)
                (resultsFileBaseFinal, benchmarkErrors) = \
                        _benchmarkProblemType(
                            problemTypeConfig,
                            sizeGroupConfig,
                            idx,
                            useCache,
                            asmToolchain,
                            srcToolchain,
                            cCompiler,
                            buildTmpPath,
                            benchmarkProblemsPath,
                            debugConfig,
                            deviceId,
                            gfxName,
                            isaInfoMap,
                            probSolMap
                        )
                totalTestFails += benchmarkErrors

                print("clientExit={} {} for {}" \
                        .format(totalTestFails, "(ERROR)" if totalTestFails else "(PASS)", \
                        globalParameters["ConfigPath"]) )

                # copy data
                resultsFileBase = resultsFileBaseFinal
                resultsFileName = resultsFileBase + ".csv"
                solutionsFileName = resultsFileBase + ".yaml"
                granularityFileName = resultsFileBase + "_Granularity.csv"
                shutil.copy(resultsFileName, newResultsFileName)
                shutil.copy(solutionsFileName, newSolutionsFileName)
                if os.path.isfile(granularityFileName):
                    shutil.copy(granularityFileName, newGranularityFileName)
            else:
                print1("# {}_{:02d} already benchmarked; skipping." \
                        .format(str(problemTypeObj), idx) )

    if globalParameters["ExitOnFails"] and totalTestFails:
        sys.exit(1)
