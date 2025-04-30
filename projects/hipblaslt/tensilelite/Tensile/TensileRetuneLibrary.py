###############################################################################
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

from . import BenchmarkProblems
from . import ClientExecutable
from . import ClientWriter
from . import LibraryIO
from . import LibraryLogic
from .Common import globalParameters, print1, printWarning, ensurePath, assignGlobalParameters, \
                    restoreDefaultGlobalParameters, HR, __version__
from .Tensile import addCommonArguments, argUpdatedGlobalParameters
from .SolutionStructs import ProblemSizes
from .Toolchain.Validators import validateToolchain

from pathlib import Path

import argparse
import copy
import os
import shutil
import sys
from pathlib import Path

workingDirectoryStack = []
def pushWorkingPath( foldername ):
  # Warning: this is not thread-safe, modifies the global WorkingPath!
  globalParameters["WorkingPath"] = \
      os.path.join(globalParameters["WorkingPath"], foldername )
  return ensurePath( globalParameters["WorkingPath"] )
def popWorkingPath():
  # Warning: this is not thread-safe, modifies the global WorkingPath!
  if len(workingDirectoryStack) == 0:
    globalParameters["WorkingPath"] = \
      os.path.split(globalParameters["WorkingPath"])[0]
  else:
    globalParameters["WorkingPath"] = workingDirectoryStack.pop()
def ensurePath(path):
  try:
    os.makedirs(path)
  except FileExistsError:
    pass
  return path
def setWorkingPath( fullPathName ):
  # Warning: this is not thread-safe, modifies the global WorkingPath!
  workingDirectoryStack.append(globalParameters["WorkingPath"])
  globalParameters["WorkingPath"] = ensurePath(fullPathName)

workingDirectoryStack = []
def pushWorkingPath( foldername ):
  # Warning: this is not thread-safe, modifies the global WorkingPath!
  globalParameters["WorkingPath"] = \
      os.path.join(globalParameters["WorkingPath"], foldername )
  return ensurePath( globalParameters["WorkingPath"] )
def popWorkingPath():
  # Warning: this is not thread-safe, modifies the global WorkingPath!
  if len(workingDirectoryStack) == 0:
    globalParameters["WorkingPath"] = \
      os.path.split(globalParameters["WorkingPath"])[0]
  else:
    globalParameters["WorkingPath"] = workingDirectoryStack.pop()
def ensurePath(path):
  try:
    os.makedirs(path)
  except FileExistsError:
    pass
  return path
def setWorkingPath( fullPathName ):
  # Warning: this is not thread-safe, modifies the global WorkingPath!
  workingDirectoryStack.append(globalParameters["WorkingPath"])
  globalParameters["WorkingPath"] = ensurePath(fullPathName)


def parseCurrentLibrary(libPath, sizePath):
    libYaml = LibraryIO.read(libPath)
    # parseLibraryLogicData mutates the original data, so make a copy
    fields = LibraryIO.parseLibraryLogicData(copy.deepcopy(libYaml), libPath)
    (_, _, problemType, solutions, exactLogic, _, _) = fields

    # get performance metric
    if len(libYaml) > 10:
        GlobalParameters.globalParameters["PerformanceMetric"] = libYaml[10]

    # process exactLogic into ProblemSizes
    sizes = []
    if sizePath is None:
        for (size, mapping) in exactLogic:
            sizes.append({"Exact": size})
    else:
        sizes = LibraryIO.read(sizePath)

    # remove duplicate solutions and reindex
    solutions = [v1 for i, v1 in enumerate(solutions) if not any(v1 == v2 for v2 in solutions[:i])]
    for i, s in enumerate(solutions):
        s["SolutionIndex"] = i

    problemSizes = ProblemSizes(problemType, sizes)

    return (libYaml, solutions, problemSizes)


def runBenchmarking(solutions, problemSizes, outPath, update, cxxCompiler: str, cCompiler: str, assembler: str, offloadBundler: str):
    # TODO some copy-pasting from BenchmarkProblems.benchmarkProblemType
    # could use a refactor to elimate duplicated code
    ClientExecutable.getClientExecutable(cxxCompiler, cCompiler)



    shortName = "benchmark"
    benchmarkDir = os.path.join(outPath, shortName)
    sourceDir = os.path.join(benchmarkDir, "source")
    resultsDir = os.path.normpath(os.path.join(globalParameters["WorkingPath"], "Data"))
    libraryFile = os.path.join(resultsDir, "benchmark.yaml")

    ensurePath(sourceDir)
    ensurePath(resultsDir)

    if update:
        globalParameters["LibraryUpdateFile"] = os.path.join(resultsDir, "update.yaml")

    pushWorkingPath(shortName)
    pushWorkingPath("source")
    BenchmarkProblems.writeBenchmarkFiles(benchmarkDir, solutions, problemSizes , "", "", "", "", shortName, [], cxxCompiler, assembler, offloadBundler)
    popWorkingPath() # source

    libraryLogicPath = None
    forBenchmark = True
    # TODO make this work with TileAware selection
    returncode = ClientWriter.runClient(libraryLogicPath, forBenchmark, False, cxxCompiler, cCompiler)
    if returncode:
        printWarning("Benchmarking Client exited with code {}. Trying to continue".format(returncode))

    # write solutions yaml file
    for sol in solutions:
        sol["ISA"] = list(sol["ISA"])
    LibraryIO.writeSolutions(libraryFile, problemSizes, "", solutions)

    popWorkingPath() # benchmark

    # copy results to expected directory
    out = os.path.join(globalParameters["WorkingPath"], "2_BenchmarkData")
    ensurePath(out)
    shutil.copy(os.path.join(resultsDir, "benchmark.csv"), os.path.join(out, "benchmark.csv"))
    shutil.copy(os.path.join(resultsDir, "benchmark.yaml"), os.path.join(out, "benchmark.yaml"))


def TensileRetuneLibrary(userArgs):
    print1("")
    print1(HR)
    print1("#")
    print1("#  Tensile Retune Library v{}".format(__version__))

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

    addCommonArguments(argParser)
    args = argParser.parse_args(userArgs)

    libPath = args.LogicFile
    sizePath = args.SizeFile
    libraryFormat = args.LibraryFormat

    print1("#  Library Logic: {}".format(libPath))
    print1("#")
    print1(HR)
    print1("")

    if args.updateMethod == "remake":
        update = False
        remake = True
    elif args.updateMethod == "update":
        update = True
        remake = False
    else: # args.updateMethod == "both"
        update = True
        remake = True

    cxxCompiler, cCompiler, assembler, offloadBundler = validateToolchain(args.CxxCompiler, args.CCompiler, args.Assembler, args.OffloadBundler)

    ##############################################
    # Retuning
    ##############################################
    outputPath = Path(ensurePath(os.path.abspath(args.OutputPath)))
    restoreDefaultGlobalParameters()

    assignGlobalParameters({"LibraryFormat": libraryFormat, "OutputPath": outputPath})

    overrideParameters = argUpdatedGlobalParameters(args)
    for key, value in overrideParameters.items():
        print1("Overriding {0}={1}".format(key, value))
        globalParameters[key] = value

    # parse library logic then setup and run benchmarks
    (rawYaml, solutions, problemSizes) = parseCurrentLibrary(libPath, sizePath)
    runBenchmarking(solutions, problemSizes, outputPath, update, cxxCompiler, cCompiler, assembler, offloadBundler)

    if remake:
        # write library logic file
        LibraryLogic.main(
           {
              "ScheduleName": rawYaml[1],
              "ArchitectureName": rawYaml[2],
              "DeviceNames": rawYaml[3]
            },
            cxxCompiler,
            outputPath
        )

    if update:
        # read update yaml from benchmark client and update logic
        print1("")
        print1(HR)
        print1("# Reading update file from Benchmarking Client")
        updateFile = os.path.join(outputPath, "Data", "update.yaml")
        updateLogic = LibraryIO.read(updateFile)
        rawYaml[7] = updateLogic

        # write updated library logic (does not overwrite original)
        libName = os.path.basename(libPath)
        outFile = os.path.join(outputPath, libName)

        print1("# Writing updated Library Logic: {}".format(outFile))
        LibraryIO.writeYAML(outFile, rawYaml, explicit_start=False, explicit_end=False)
        print1(HR)


def main():
    TensileRetuneLibrary(sys.argv[1:])
