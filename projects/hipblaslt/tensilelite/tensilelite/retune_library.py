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
from . import ClientWriter
from . import LibraryIO
from . import LibraryLogic
from .Common.GlobalParameters import globalParameters, assignGlobalParameters, restoreDefaultGlobalParameters, __version__
from .Common.Utilities import print1, printWarning, printExit, ensurePath as ensurePathUtil
from .Common.Constants import HR
from .Common.Architectures import baseArchName, gfxToIsa
from .Common.Capabilities import applyArchCapOverrides, makeIsaInfoMap
from .Common.Types import makeDebugConfig
from .run import addCommonArguments, argUpdatedGlobalParameters
from .SolutionStructs import ProblemSizes
from .SolutionStructs.Solution import ActivationArgs, BiasTypeArgs, FactorDimArgs, GateTypeArgs
from .Toolchain.Assembly import makeAssemblyToolchain
from .Toolchain.Source import makeSourceToolchain
from .Toolchain.Validators import validateToolchain

from pathlib import Path

import argparse
import copy
import os
import shutil
import sys

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


def libraryArchitectureName(libYaml):
    if isinstance(libYaml, dict):
        arch = libYaml.get("ArchitectureName")
    elif isinstance(libYaml, list) and len(libYaml) > 2:
        arch = libYaml[2]
        if isinstance(arch, dict):
            arch = arch.get("Architecture")
    else:
        return None
    return baseArchName(arch) if isinstance(arch, str) else None


def parseCurrentLibrary(libPath, sizePath, assembler, isaInfoMap, lazyLibraryLoading):
    libYaml = LibraryIO.read(libPath)
    if not isaInfoMap:
        printExit(f"No ISA capability map for {libPath}")
    fields = LibraryIO.parseLibraryLogicData(
        copy.deepcopy(libYaml), libPath, assembler, False, True, False,
        isaInfoMap, lazyLibraryLoading)
    (_, _, problemType, solutions, exactLogic, _, _) = fields

    # get performance metric from list or dict logic formats
    if isinstance(libYaml, dict):
        globalParameters["PerformanceMetric"] = libYaml.get("PerfMetric")
    elif len(libYaml) > 10:
        globalParameters["PerformanceMetric"] = libYaml[10]

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


def runBenchmarking(solutions, problemSizes, outPath, update, cxxCompiler, cCompiler,
                    asmToolchain, srcToolchain, isaInfoMap, deviceId, gfxName,
                    debugConfig, problemType):
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
        globalParameters["LibraryUpdateFile"] = os.path.join(resultsDir, "update.yaml")
    biasTypes = problemType["BiasDataTypeList"]
    gateTypes = problemType["GateResidualDataTypeList"]
    biasTypeArgs = BiasTypeArgs(problemType, [biasTypes[0]]) if biasTypes else ""
    gateTypeArgs = GateTypeArgs(problemType, [gateTypes[0]]) if gateTypes else ""
    factorDimArgs = FactorDimArgs(problemType, [0])
    activationArgs = (ActivationArgs(problemType, [[{"Enum": "relu"}]])
                      if problemType["ActivationType"] in ["all", "hipblaslt_all"]
                      else "")


    pushWorkingPath(shortName)
    pushWorkingPath("source")
    BenchmarkProblems.writeBenchmarkFiles(
        benchmarkDir, solutions, problemSizes, biasTypeArgs, factorDimArgs,
        activationArgs, [False], shortName, [], asmToolchain, srcToolchain,
        Path(sourceDir), debugConfig, gateTypeArgs, deviceId, gfxName,
        isaInfoMap, {}, [gfxName])
    popWorkingPath() # source

    libraryLogicPath = None
    forBenchmark = True
    # TODO make this work with TileAware selection
    returncode = ClientWriter.runClient(
        libraryLogicPath, forBenchmark, False, cxxCompiler, cCompiler,
        Path(benchmarkDir), configPaths=[os.path.join(sourceDir, "ClientParameters.ini")])
    if returncode:
        printWarning("Benchmarking Client exited with code {}. Trying to continue".format(returncode))

    # write solutions yaml file
    for sol in solutions:
        sol["ISA"] = list(sol["ISA"])
    LibraryIO.writeSolutions(libraryFile, problemSizes, biasTypeArgs, activationArgs,
                             solutions, gateTypeArgs=gateTypeArgs)

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
    print1("#  TensileLite Retune Library v{}".format(__version__))

    # argument parsing and related setup
    argParser = argparse.ArgumentParser()
    argParser.add_argument("LogicFile", type=os.path.realpath,
                           help="Library logic file to retune")
    argParser.add_argument("OutputPath", type=os.path.realpath,
                           help="Where to run benchmarks and output results")
    argParser.add_argument("SizeFile", type=os.path.realpath, nargs="?",
                           help="Yaml file with sizes to tune; same format as the 'ProblemSizes' "
                           "section of a regular TensileLite config "
                           "(https://github.com/ROCmSoftwarePlatform/Tensile/wiki/Benchmark-Protocol)",
                           default=None)
    argParser.add_argument("--update-method", "-u", dest="updateMethod",
                           choices=["remake", "update", "both"], default="remake",
                           help="Method for making new library logic file")

    addCommonArguments(argParser)
    argParser.set_defaults(LibraryFormat="msgpack")
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

    globalParameters["WorkingPath"] = str(outputPath)
    logicArchName = libraryArchitectureName(LibraryIO.read(libPath))
    if not logicArchName:
        printExit(f"Library logic file {libPath} names no architecture")
    isaInfoMap = makeIsaInfoMap([gfxToIsa(logicArchName)], str(cxxCompiler))
    applyArchCapOverrides(isaInfoMap, [logicArchName])
    assignGlobalParameters({"LibraryFormat": libraryFormat, "OutputPath": outputPath}, isaInfoMap)
    globalParameters["SleepPercent"] = 0

    overrideParameters = argUpdatedGlobalParameters(args)
    for key, value in overrideParameters.items():
        print1("Overriding {0}={1}".format(key, value))
        globalParameters[key] = value

    asmToolchain = makeAssemblyToolchain(
        assembler, offloadBundler, globalParameters["CodeObjectVersion"],
        globalParameters["BuildIdKind"], globalParameters["AsmDebug"])
    srcToolchain = makeSourceToolchain(
        cxxCompiler, offloadBundler, globalParameters["AsanBuild"],
        globalParameters["BuildIdKind"], save_temps=False)
    debugConfig = makeDebugConfig({
        "SplitGSU": False, "PrintSolutionRejectionReason": True,
        "PrintIndexAssignmentInfo": False,
    })
    rawYaml, solutions, problemSizes = parseCurrentLibrary(
        libPath, sizePath, asmToolchain.assembler, isaInfoMap,
        globalParameters["LazyLibraryLoading"])
    if not solutions:
        printExit(f"No solutions found in {libPath}")
    runBenchmarking(
        solutions, problemSizes, outputPath, update, cxxCompiler, cCompiler,
        asmToolchain, srcToolchain, isaInfoMap, args.device, logicArchName,
        debugConfig, solutions[0]["ProblemType"])

    if remake:
        # write library logic file
        if isinstance(rawYaml, dict):
            logicHeader = {
                "ScheduleName": rawYaml.get("ScheduleName"),
                "ArchitectureName": rawYaml.get("ArchitectureName"),
                "DeviceNames": rawYaml.get("DeviceNames"),
            }
        else:
            arch = rawYaml[2]
            logicHeader = {
                "ScheduleName": rawYaml[1],
                "ArchitectureName": arch.get("Architecture") if isinstance(arch, dict) else arch,
                "DeviceNames": rawYaml[3],
            }
        LibraryLogic.main(
            logicHeader,
            asmToolchain.assembler,
            outputPath,
            debugConfig.splitGSU,
            debugConfig.printSolutionRejectionReason,
            debugConfig.printIndexAssignmentInfo,
            isaInfoMap
        )

    if update:
        # read update yaml from benchmark client and update logic
        print1("")
        print1(HR)
        print1("# Reading update file from Benchmarking Client")
        updateFile = os.path.join(outputPath, "Data", "update.yaml")
        updateLogic = LibraryIO.read(updateFile)
        if isinstance(rawYaml, dict):
            rawYaml["ExactLogic"] = updateLogic
        else:
            rawYaml[7] = updateLogic

        # write updated library logic (does not overwrite original)
        libName = os.path.basename(libPath)
        outFile = os.path.join(outputPath, libName)

        print1("# Writing updated Library Logic: {}".format(outFile))
        LibraryIO.writeYAML(outFile, rawYaml, explicit_start=False, explicit_end=False)
        print1(HR)


def main():
    TensileRetuneLibrary(sys.argv[1:])
