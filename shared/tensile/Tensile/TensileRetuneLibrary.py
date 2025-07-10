################################################################################
#
# Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from . import BenchmarkProblems
from . import ClientWriter
from . import LibraryIO
from . import LibraryLogic
from . import Common
from .Common import globalParameters, tPrint, printWarning, ensurePath, assignGlobalParameters, \
                    pushWorkingPath, popWorkingPath, restoreDefaultGlobalParameters, HR
from .Tensile import addCommonArguments, argUpdatedGlobalParameters
from .SolutionStructs import ProblemSizes
from . import __version__

import argparse
import copy
import os
import shutil
import sys


def parseCurrentLibrary(libPath, skipRK, sizePath):
    libYaml = LibraryIO.readYAML(libPath)
    # parseLibraryLogicData mutates the original data, so make a copy
    fields = LibraryIO.parseLibraryLogicData(copy.deepcopy(libYaml), libPath)
    (_, _, problemType, solutions, exactLogic, _) = fields

    # get performance metric
    if len(libYaml) > 10:
        Common.globalParameters["PerformanceMetric"] = libYaml[10]

    # process exactLogic into ProblemSizes
    sizes = []
    if sizePath is None:
        for (size, mapping) in exactLogic:
            if skipRK:
                sol = solutions[mapping[0]]
                if sol["ReplacementKernel"]:
                    continue

            sizes.append({"Exact": size})
    else:
        sizes = LibraryIO.readYAML(sizePath)

    if skipRK:
        solutions = [s for s in solutions if not s["ReplacementKernel"]]

    # remove duplicate solutions and reindex
    solutions = [v1 for i, v1 in enumerate(solutions) if not any(v1 == v2 for v2 in solutions[:i])]
    for i, s in enumerate(solutions):
        s["SolutionIndex"] = i

    problemSizes = ProblemSizes(problemType, sizes)

    return (libYaml, solutions, problemSizes)


def runBenchmarking(solutions, problemSizes, outPath, update):
    # TODO some copy-pasting from BenchmarkProblems.benchmarkProblemType
    # could use a refactor to elimate duplicated code
    ClientWriter.getClientExecutablePath()

    shortName = "benchmark"
    benchmarkDir = os.path.join(outPath, shortName)
    sourceDir = os.path.join(benchmarkDir, "source")
    resultsDir = os.path.normpath(os.path.join(globalParameters["WorkingPath"], "Data"))
    libraryFile = os.path.join(resultsDir, "benchmark.yaml")

    ensurePath(sourceDir)
    ensurePath(resultsDir)

    if update:
        Common.globalParameters["LibraryUpdateFile"] = os.path.join(resultsDir, "update.yaml")

    pushWorkingPath(shortName)
    pushWorkingPath("source")
    BenchmarkProblems.writeBenchmarkFiles(benchmarkDir, solutions, problemSizes, shortName, [])
    popWorkingPath() # source

    libraryLogicPath = None
    forBenchmark = True
    # TODO make this work with TileAware selection
    returncode = ClientWriter.runClient(libraryLogicPath, forBenchmark, False)
    if returncode:
        printWarning("Benchmarking Client exited with code {}. Trying to continue".format(returncode))

    # write solutions yaml file
    for sol in solutions:
        sol["ISA"] = list(sol["ISA"])
    LibraryIO.writeSolutions(libraryFile, problemSizes, solutions)

    popWorkingPath() # benchmark

    # copy results to expected directory
    out = os.path.join(globalParameters["WorkingPath"], "2_BenchmarkData")
    ensurePath(out)
    shutil.copy(os.path.join(resultsDir, "benchmark.csv"), os.path.join(out, "benchmark.csv"))
    shutil.copy(os.path.join(resultsDir, "benchmark.yaml"), os.path.join(out, "benchmark.yaml"))


def TensileRetuneLibrary(userArgs):
    tPrint(1, "")
    tPrint(1, HR)
    tPrint(1, "#")
    tPrint(1, "#  Tensile Retune Library v{}".format(__version__))

    # argument parsing and related setup
    argParser = argparse.ArgumentParser()
    argParser.add_argument("LogicFile", type=os.path.realpath,
                           help="Library logic file to retune")
    argParser.add_argument("OutputPath", type=os.path.realpath,
                           help="Where to run benchmarks and output results")
    argParser.add_argument("SizeFile", type=os.path.realpath, nargs="?",
                           help="Yaml file with sizes to tune; same format as the 'ProblemSizes' "
                           "section of a regular Tensile config "
                           "(https://github.com/ROCmSoftwarePlatform/Tensile/wiki/Benchmark-Protocol)",
                           default=None)
    argParser.add_argument("--update-method", "-u", dest="updateMethod",
                           choices=["remake", "update", "both"], default="remake",
                           help="Method for making new library logic file")
    argParser.add_argument("--skip-replacement-kernels", "-s", dest="skipRK", action="store_true",
                           help="Exclude sizes and solutions related to replacement kernels. "
                                "Forces update-method to 'remake'")

    addCommonArguments(argParser)
    args = argParser.parse_args(userArgs)

    libPath = args.LogicFile
    sizePath = args.SizeFile
    tPrint(1, "#  Library Logic: {}".format(libPath))
    tPrint(1, "#")
    tPrint(1, HR)
    tPrint(1, "")

    if args.skipRK and args.updateMethod != "remake":
        printWarning("--skip-replacement-kernels=true only compatable with --update-method=remake:"
                     "\n\tsetting --update-method=remake")
        args.updateMethod = "remake"

    if args.updateMethod == "remake":
        update = False
        remake = True
    elif args.updateMethod == "update":
        update = True
        remake = False
    else: # args.updateMethod == "both"
        update = True
        remake = True

    ##############################################
    # Retuning
    ##############################################
    outPath = ensurePath(os.path.abspath(args.OutputPath))
    restoreDefaultGlobalParameters()
    assignGlobalParameters({"LibraryFormat": "msgpack",
                            "OutputPath": outPath,
                            "WorkingPath": outPath})

    overrideParameters = argUpdatedGlobalParameters(args)
    for key, value in overrideParameters.items():
        tPrint(1, "Overriding {0}={1}".format(key, value))
        Common.globalParameters[key] = value

    # parse library logic then setup and run benchmarks
    (rawYaml, solutions, problemSizes) = parseCurrentLibrary(libPath, args.skipRK, sizePath)
    runBenchmarking(solutions, problemSizes, outPath, update)

    if remake:
        # write library logic file
        LibraryLogic.main({"ScheduleName": rawYaml[1],
                           "ArchitectureName": rawYaml[2],
                           "DeviceNames": rawYaml[3] })

    if update:
        # read update yaml from benchmark client and update logic
        tPrint(1, "")
        tPrint(1, HR)
        tPrint(1, "# Reading update file from Benchmarking Client")
        updateFile = os.path.join(outPath, "Data", "update.yaml")
        updateLogic = LibraryIO.readYAML(updateFile)
        rawYaml[7] = updateLogic

        # write updated library logic (does not overwrite original)
        libName = os.path.basename(libPath)
        outFile = os.path.join(outPath, libName)

        tPrint(1, "# Writing updated Library Logic: {}".format(outFile))
        LibraryIO.writeYAML(outFile, rawYaml, explicit_start=False, explicit_end=False)
        tPrint(1, HR)


def main():
    TensileRetuneLibrary(sys.argv[1:])
