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

import os
import shutil
import sys
import time

from copy import deepcopy

from . import SolutionLibrary
from . import LibraryIO
from . import Utils
from .BenchmarkStructs import BenchmarkProcess, checkParametersAreValid, constructForkPermutations
from .Contractions import ProblemType as ContractionsProblemType
from .ClientWriter import runClient, writeClientConfig, writeClientConfigIni
from .Common import globalParameters, HR, pushWorkingPath, popWorkingPath, tPrint, \
        printExit, printWarning, ensurePath, startTime, validParameters
from .KernelWriterAssembly import KernelWriterAssembly
from .KernelWriterSource import KernelWriterSource
from .SolutionStructs import Solution, ProblemType, ProblemSizes
from .TensileCreateLibrary import copyStaticFiles, writeKernels
from .CustomKernels import getCustomKernelConfig

def generateForkedSolutions(problemType, constantParams, forkPermutations):
    """Creates a list with a Solution object for each parameter combination in forkPermutations"""
    tPrint(1, "# Enumerating Solutions")

    solutions = []
    solutionSet = set()
    for perm in forkPermutations:
        solution = {"ProblemType": deepcopy(problemType.state)}
        solution.update(constantParams)
        solution.update(perm)

        # TODO check if solution matches problem size for exact tile kernels
        solutionObject = Solution(solution)
        if solutionObject["Valid"]:
            if solutionObject not in solutionSet:
                solutionSet.add(solutionObject)
                solutions.append(solutionObject)
        elif globalParameters["PrintSolutionRejectionReason"]:
            tPrint(1, "rejecting solution " + str(solutionObject))

    return solutions


def getCustomKernelSolutionObj(kernelName, directory=globalParameters["CustomKernelDirectory"]):
    """Creates the Solution object for a custom kernel"""
    kernelConfig = getCustomKernelConfig(kernelName, directory)
    for k, v in kernelConfig.items():
        if k != "ProblemType":
            checkParametersAreValid((k, [v]), validParameters)
    kernelConfig["KernelLanguage"] = "Assembly"
    kernelConfig["CustomKernelName"] = kernelName

    return Solution(kernelConfig)


def generateCustomKernelSolutions(problemType, customKernels, failOnMismatch):
    """Creates a list with a Solution object for each name in customKernel"""
    solutions = []
    for kernelName in customKernels:
        tPrint(1, "# Processing custom kernel {}".format(kernelName))
        solution = getCustomKernelSolutionObj(kernelName)
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
                tPrint(1, "# Rejected {}: Problem Type doesn't match".format(kernelName))
        else:
            tPrint(1, "# Added {} to solutions".format(kernelName))
            if solution["Valid"]:
                solutions.append(solution)
            elif globalParameters["PrintSolutionRejectionReason"]:
                tPrint(1, "rejecting solution " + str(solution))

    return solutions

def writeBenchmarkFiles(stepBaseDir, solutions, problemSizes, \
        stepName, solutionSummationSizes):
    """Write all the files needed for a given benchmarking step"""
    if not globalParameters["MergeFiles"]:
        ensurePath(os.path.join(globalParameters["WorkingPath"], "Solutions"))
        ensurePath(os.path.join(globalParameters["WorkingPath"], "Kernels"))

    copyStaticFiles()

    kernels = []
    kernelHelperOjbs = []
    kernelNames = set()
    kernelHelperNames = set()

    # get unique kernels and kernel helpers
    for solution in Utils.tqdm(solutions, desc="Finding unique solutions"):
        kernel = solution.getKernels()
        kName = Solution.getNameFull(kernel)
        if kName not in kernelNames:
            kernels.append(kernel)
            kernelNames.add(kName)

        solutionHelperKernels = solution.getHelperKernelObjects()
        for ko in solutionHelperKernels:
            kname = ko.getKernelName()
            if kname not in kernelHelperNames:
                kernelHelperOjbs.append(ko)
                kernelHelperNames.add(kname)

    kernelSerialNaming = Solution.getSerialNaming(kernels)
    kernelMinNaming = Solution.getMinNaming(kernels)
    kernelWriterSource = KernelWriterSource(kernelMinNaming, kernelSerialNaming, \
                                            not globalParameters["KeepBuildTmp"])
    kernelWriterAssembly = KernelWriterAssembly(kernelMinNaming, kernelSerialNaming, \
                                                not globalParameters["KeepBuildTmp"])

    # write solution, kernels and CMake
    problemType = solutions[0]["ProblemType"]
    codeObjectFiles, kernels, solutions = writeKernels( \
            globalParameters["WorkingPath"], globalParameters["CxxCompiler"], globalParameters["ClangOffloadBundlerPath"], \
            globalParameters, solutions, kernels, kernelHelperOjbs, \
            kernelWriterSource, kernelWriterAssembly, errorTolerant=True, \
            removeTemporaries = not globalParameters["KeepBuildTmp"])
    # ^ this is where solutions is mutated
    newLibraryDir = ensurePath(os.path.join(globalParameters["WorkingPath"], 'library'))
    newLibraryFile = os.path.join(newLibraryDir, "TensileLibrary")
    newLibrary = SolutionLibrary.MasterSolutionLibrary.BenchmarkingLibrary(solutions)
    newLibrary.applyNaming(kernelMinNaming)
    LibraryIO.write(newLibraryFile, Utils.state(newLibrary), globalParameters["LibraryFormat"])

    codeObjectFiles = [os.path.relpath(f, globalParameters["WorkingPath"]) \
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
        writeClientConfig(True, solutions, idealProblemSizes, stepName, stepBaseDir, \
            newLibrary, codeObjectFiles, True)
    else:
        writeClientConfig(True, solutions, problemSizes, stepName, stepBaseDir, \
            newLibrary, codeObjectFiles, False)

    if len(solutions) == 0:
        printExit("write solutions and kernels results 0 valid soultion.")

    return codeObjectFiles


def benchmarkProblemType(problemTypeConfig, problemSizeGroupConfig, problemSizeGroupIdx, useCache):
    """Run the benchmarking for a single entry in the BenchmarkProblems of a Tensile config"""
    benchmarkTestFails = 0

    tPrint(1, "")
    tPrint(1, HR)
    tPrint(1, "# Converting Config to BenchmarkProcess Object")
    tPrint(1, HR)
    tPrint(1, "")
    benchmarkProcess = BenchmarkProcess(problemTypeConfig, problemSizeGroupConfig)

    enableTileSelection = benchmarkProcess.problemType["TileAwareSelection"]
    groupName = "{}_{:02d}".format(str(benchmarkProcess.problemType), problemSizeGroupIdx)
    pushWorkingPath(groupName)
    ensurePath(os.path.join(globalParameters["WorkingPath"], "Data"))

    totalBenchmarkSteps = len(benchmarkProcess)
    resultsFileBaseFinal = None

    tPrint(1, "# NumBenchmarkSteps: {}".format(totalBenchmarkSteps))
    tPrint(1, "")
    tPrint(1, HR)
    tPrint(1, "# Done Creating BenchmarkProcess Object")
    tPrint(1, HR)

    for benchmarkStepIdx in range(0, totalBenchmarkSteps):
        benchmarkStep = benchmarkProcess[benchmarkStepIdx]
        stepName = str(benchmarkStep)
        shortName = stepName

        tPrint(1, "\n")
        tPrint(1, HR)
        currentTime = time.time()
        elapsedTime = currentTime - startTime
        tPrint(1, "# Benchmark Step: {} - {} {:.3f}s".format(groupName, stepName, elapsedTime))
        tPrint(1, "# Num Sizes: {}".format(benchmarkStep.problemSizes.totalProblemSizes))
        tPrint(1, "# Fork Parameters:")
        for k, v in sorted(benchmarkStep.forkParams.items()):
            tPrint(1, "#     {}: {}".format(k, v))

        pushWorkingPath(shortName)
        stepBaseDir = globalParameters["WorkingPath"]

        # file paths
        resultsFileBase = os.path.normpath(os.path.join( \
                globalParameters["WorkingPath"], "../Data", shortName))
        if benchmarkStep.isFinal():
            resultsFileBaseFinal = resultsFileBase
        resultsFileName = resultsFileBase + ".csv"
        solutionsFileName = resultsFileBase + ".yaml"

        # check if a solution cache exists and if it matches our solution parameters
        cachePath = os.path.join(stepBaseDir, "cache.yaml")
        pushWorkingPath("source")

        cacheValid = False
        if useCache and os.path.isfile(cachePath):
            c = LibraryIO.readYAML(cachePath)
            if c["ConstantParams"] == benchmarkStep.constantParams and \
                    c["ForkParams"] == benchmarkStep.forkParams and \
                    c["ParamGroups"] == benchmarkStep.paramGroups and \
                    c["CustomKernels"] == benchmarkStep.customKernels and \
                    c["CustomKernelWildcard"] == benchmarkStep.customKernelWildcard:
                cacheValid = True
                codeObjectFiles = c["CodeObjectFiles"]
            else:
                printWarning("Cache data does not match config: redoing solution generation")

        if not cacheValid:
            # enumerate benchmark permutations and create resulting solution objects
            forkPermutations = constructForkPermutations(benchmarkStep.forkParams, \
                    benchmarkStep.paramGroups)
            maxPossibleSolutions = len(forkPermutations)

            regSolutions = generateForkedSolutions(benchmarkProcess.problemType, \
                    benchmarkStep.constantParams, forkPermutations)
            kcSolutions = generateCustomKernelSolutions(benchmarkProcess.problemType, \
                    benchmarkStep.customKernels, not benchmarkStep.customKernelWildcard)

            maxPossibleSolutions += len(kcSolutions)
            solutions = regSolutions + kcSolutions

            tPrint(1, "# Actual Solutions: {} / {} after SolutionStructs\n" \
                .format(len(solutions), maxPossibleSolutions))

            # handle no valid solutions
            if len(solutions) == 0:
                msg = "Your parameters resulted in 0 valid solutions."
                if globalParameters["PrintSolutionRejectionReason"]:
                    msg += "\nExamine reject and backtrace messages above to see why" \
                            "and where solutions were rejected."
                else:
                    msg += "\nYou should re-run with \"PrintSolutionRejectionReason: True\"" \
                            "to see why each parameter combination was rejected."
                printExit(msg)

            if globalParameters["PrintLevel"] >= 1:
                for solution in solutions:
                    tPrint(3, "#    ({}:{}) {}".format(0, 0, Solution.getNameFull(solution)))
                tPrint(3, HR)

            # write benchmarkFiles
            prevCount = len(solutions)
            codeObjectFiles = writeBenchmarkFiles(stepBaseDir, solutions, \
                    benchmarkStep.problemSizes, shortName, [])
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

            tPrint(1, "# Actual Solutions: {} / {} after KernelWriter\n" \
                    .format(len(solutions), prevCount ))
        else:
            solutions = None
            tPrint(1, "# Using cached solution data")

            ssProblemType = ProblemType(problemTypeConfig)
            conProblemType = ContractionsProblemType.FromOriginalState(ssProblemType)
            outFile = os.path.join(globalParameters["WorkingPath"], "ClientParameters.ini")

            writeClientConfigIni(benchmarkStep.problemSizes, conProblemType,
                                 globalParameters["WorkingPath"], codeObjectFiles, resultsFileName,
                                 outFile)

        # I think the size portion of this yaml could be removed,
        # but for now it's needed, so we update it even in the cache case
        LibraryIO.writeSolutions(solutionsFileName, benchmarkStep.problemSizes, solutions, cacheValid)

        popWorkingPath()  # source

        # run benchmarking client
        if not os.path.exists(resultsFileName) or globalParameters["ForceRedoBenchmarkProblems"]:
            libraryLogicPath = None
            forBenchmark = True
            returncode = runClient(libraryLogicPath, forBenchmark, enableTileSelection)

            if returncode:
                benchmarkTestFails += 1
                printWarning("BenchmarkProblems: Benchmark Process exited with code {}" \
                        .format(returncode))
        else:
            tPrint(1, "# Already benchmarked; skipping.")

        # End Iteration
        popWorkingPath()  # stepName
        currentTime = time.time()
        elapsedTime = currentTime - startTime
        tPrint(1, "{}\n# {}\n# {}: End - {:.3f}s\n{}\n" \
                .format(HR, groupName, shortName, elapsedTime, HR))

    popWorkingPath()  # ProblemType
    return (resultsFileBaseFinal, benchmarkTestFails)


def main(config, useCache):
    """Entry point for the "BenchmarkProblems" section of a Tensile config yaml"""
    dataPath = os.path.join(globalParameters["WorkingPath"], globalParameters["BenchmarkDataPath"])
    pushWorkingPath(globalParameters["BenchmarkProblemsPath"])
    ensurePath(dataPath)

    totalTestFails = 0
    for benchmarkProblemTypeConfig in config:
        problemTypeConfig = benchmarkProblemTypeConfig[0]
        if len(benchmarkProblemTypeConfig) < 2:
            problemSizeGroupConfigs = [{}]
        else:
            problemSizeGroupConfigs = benchmarkProblemTypeConfig[1:]

        for idx, sizeGroupConfig in enumerate(problemSizeGroupConfigs):
            tPrint(3, "ProblemTypeConfig: {}".format(problemTypeConfig))
            problemTypeObj = ProblemType(problemTypeConfig)

            # using a suffix to check the csv version (for later addFromCSV())
            csvSuffix = "_CSVWinner" if globalParameters["CSVExportWinner"] else ""
            # results files will be named
            newResultsFileName = os.path.join(dataPath, "{}_{:02d}{}.csv" \
                    .format(str(problemTypeObj), idx, csvSuffix) )
            newSolutionsFileName = os.path.join(dataPath, "{}_{:02d}{}.yaml" \
                    .format(str(problemTypeObj), idx, csvSuffix) )
            newGranularityFileName = os.path.join(dataPath, "{}_{:02d}{}.gsp" \
                    .format(str(problemTypeObj), idx, csvSuffix) )

            # skip if possible
            if globalParameters["ForceRedoBenchmarkProblems"] \
                    or not os.path.exists(newResultsFileName):

                # benchmark problem size group
                (resultsFileBaseFinal, benchmarkErrors) = \
                        benchmarkProblemType(problemTypeConfig, sizeGroupConfig, idx, useCache)
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
                tPrint(1, "# {}_{:02d} already benchmarked; skipping." \
                        .format(str(problemTypeObj), idx) )

    popWorkingPath()

    if globalParameters["ExitOnFails"] and totalTestFails:
        sys.exit(1)
