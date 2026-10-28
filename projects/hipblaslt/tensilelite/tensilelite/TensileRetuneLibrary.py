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
from .Common.Architectures import ARCH_BUILD_ALIASES, baseArchName, gfxToIsa
from .Common.Capabilities import applyArchCapOverrides, makeIsaInfoMap
from .Common.Types import DebugConfig, makeDebugConfig
from .SolutionStructs import ProblemSizes
from .SolutionStructs.Solution import ActivationArgs, BiasTypeArgs, FactorDimArgs, GateTypeArgs
from .Tensile import addCommonArguments, argUpdatedGlobalParameters
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
    """The architecture a parsed library logic file names.

    Returns None when the file names no architecture, so a caller that only
    needs the name for something optional can carry on; a caller that needs it
    says so itself.
    """
    if isinstance(libYaml, dict):
        archName = libYaml.get("ArchitectureName")
    elif isinstance(libYaml, list) and len(libYaml) > 2:
        archName = libYaml[2]
    else:
        archName = None
    if not isinstance(archName, str):
        return None
    return baseArchName(archName)


def logicFileArchitectureName(libPath):
    """The architecture a library logic file is for, as it names it."""
    return libraryArchitectureName(LibraryIO.read(libPath))


def logicFileArchRenames(archName):
    """The name a library is keyed under, for a build alias.

    A logic file that names a build alias's tuning architecture is keyed under
    that alias's target, the way TensileCreateLibrary keys it.  Nothing else
    is renamed.
    """
    if not archName:
        return None
    for fields in ARCH_BUILD_ALIASES.values():
        if fields["tuningArch"] == archName:
            return {archName: fields["target"]}
    return None


def parseCurrentLibrary(libPath, sizePath, assembler=None, isaInfoMap=None, lazyLibraryLoading=False, archRenames=None):
    libYaml = LibraryIO.read(libPath)
    # parseLibraryLogicData mutates the original data, so make a copy
    #
    # The parse grew six required arguments when the ISA capability map was
    # introduced and the library format split; this call site was left behind
    # and raises TypeError.  They carry what the solutions are built and
    # validated against, so they have to describe this file's architecture:
    # the capability map for its ISA, the assembler that assembles it, and the
    # lazy-loading mode the library was written in.
    if isaInfoMap is not None and not isaInfoMap:
        printExit("No ISA capability map for {}; the file's architecture could not be resolved".format(libPath))
    if archRenames is None:
        archRenames = logicFileArchRenames(libraryArchitectureName(libYaml))
    fields = LibraryIO.parseLibraryLogicData(
        copy.deepcopy(libYaml),
        libPath,
        assembler,
        False,  # splitGSU
        True,   # printSolutionRejectionReason
        False,  # printIndexAssignmentInfo
        isaInfoMap,
        lazyLibraryLoading,
        archRenames,
    )
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


def benchmarkArguments(problemType):
    """The client's problem-variant arguments for a problem type.

    A logic file names the problem type its solutions were built for, and the
    benchmark client has to be told the same thing: which bias data type to
    use, which activation to apply, and so on.  The values are the ones the
    library-client path in ClientWriter uses for the same problem type, so a
    retuned file is measured the way the shipped one is used.

    Each entry is either an argument object the client config writer
    understands, or the empty string it treats as "not present".  A problem
    type of None yields the neutral set, which is what a benchmark of a
    problem type that names no variants needs.
    """
    if problemType is None:
        return ("", "", "", [], "")

    # The bias data type the client allocates the bias tensor with.  The
    # problem type lists every bias type the kernels accept; one of them is
    # enough to measure the kernels with, and one problem per size keeps the
    # benchmark CSV and the solutions file in step.
    biasTypeList = problemType["BiasDataTypeList"]
    biasTypeArgs = BiasTypeArgs(problemType, [biasTypeList[0]]) if biasTypeList else ""

    # The gate residual data type, for a problem type that has one.
    gateTypeList = problemType["GateResidualDataTypeList"]
    gateTypeArgs = GateTypeArgs(problemType, [gateTypeList[0]]) if gateTypeList else ""

    # The factor dimension says which of the free dimensions the scale or the
    # bias is indexed by.  It only changes the bias strides of a sparse
    # problem, and 0 is the dimension the shipped libraries use.
    factorDimArgs = FactorDimArgs(problemType, [0])

    # An activation is only benchmarked for a problem type that applies one.
    # 'relu' is the activation every activation type supports.
    if problemType["ActivationType"] in ['all', 'hipblaslt_all']:
        activationArgs = ActivationArgs(problemType, [[{'Enum': 'relu'}]])
    else:
        activationArgs = ""

    # The icache flush is a client-side cache clearing step between
    # measurements; a plain benchmark does not ask for it.
    icacheFlushArgs = [False]

    return (biasTypeArgs, factorDimArgs, activationArgs, icacheFlushArgs, gateTypeArgs)


def runBenchmarking(solutions, problemSizes, outPath, update, cxxCompiler: str, cCompiler: str,
                    assembler: str = None, offloadBundler: str = None, *,
                    asmToolchain=None, srcToolchain=None, isaInfoMap=None,
                    deviceId: int = 0, gfxName: str = None, debugConfig: DebugConfig = None,
                    problemType=None):
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

    (biasTypeArgs, factorDimArgs, activationArgs, icacheFlushArgs, gateTypeArgs) = \
        benchmarkArguments(problemType)

    pushWorkingPath(shortName)
    pushWorkingPath("source")
    # writeBenchmarkFiles grew the toolchain objects, the debug configuration,
    # the device and the architecture it builds for; this call site was left
    # behind and raises TypeError.  The values are the ones the regular
    # benchmark flow passes for the same job: the assembler and the source
    # toolchain that build the kernels, the debug configuration the logic
    # step is later run with, the device to benchmark on, and the
    # architecture the file is for.
    BenchmarkProblems.writeBenchmarkFiles(
        benchmarkDir,
        solutions,
        problemSizes,
        biasTypeArgs,
        factorDimArgs,
        activationArgs,
        icacheFlushArgs,
        shortName,
        [],              # solutionSummationSizes
        asmToolchain,
        srcToolchain,
        Path(sourceDir),
        debugConfig,
        gateTypeArgs,
        deviceId,
        gfxName,
        isaInfoMap,
        {},              # probSolMap
        [gfxName] if gfxName else None)
    popWorkingPath() # source

    libraryLogicPath = None
    forBenchmark = True
    # TODO make this work with TileAware selection
    # runClient grew the path it writes its build and run script to, and the
    # configuration files it runs the client with; this call site was left
    # behind and raises TypeError.
    configPaths = [os.path.join(sourceDir, "ClientParameters.ini")]
    returncode = ClientWriter.runClient(libraryLogicPath, forBenchmark, False,
                                        cxxCompiler, cCompiler,
                                        Path(benchmarkDir), configPaths=configPaths)
    if returncode:
        printWarning("Benchmarking Client exited with code {}. Trying to continue".format(returncode))

    # write solutions yaml file
    for sol in solutions:
        sol["ISA"] = list(sol["ISA"])
        # An Assert* map that is empty states no constraint, and the solution
        # file writer renders an empty map as a key with no value.  That reads
        # back as null, and a null Assert* map is rejected when the logic step
        # reads the solutions back.  Drop the empty ones: the predicate an
        # empty map would produce is empty as well.
        solutionState = sol.getAttributes() if hasattr(sol, "getAttributes") else sol
        for key in [k for k, v in solutionState.items()
                    if isinstance(v, dict) and not v and k.startswith("Assert")]:
            del solutionState[key]
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
    # The client this tool runs is the one the hipBLASLt build makes, and that
    # build has the msgpack loader rather than the YAML one (the two are
    # mutually exclusive), so the library the client is handed has to be
    # msgpack.  --library-format still selects the YAML library for a client
    # that was built with the YAML loader.
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

    # Everything the retune step writes -- the benchmark data, the regenerated
    # logic -- lives below the output path, and the library logic step reads
    # the benchmark data back from it.
    globalParameters["WorkingPath"] = str(outputPath)

    # The capability map the logic file's architecture needs.  The global
    # parameters it carries (the wavefront size, the ISA's assembly
    # capabilities) are what the solutions are built with, so it has to be the
    # map for this file's architecture and not a default one.  assignGlobal-
    # Parameters takes it as an argument; it grew that parameter when the ISA
    # capability map was introduced, and this call site was left behind.
    logicArchName = logicFileArchitectureName(libPath)
    isaInfoMap = {}
    if logicArchName:
        isaInfoMap = makeIsaInfoMap([gfxToIsa(logicArchName)], str(cxxCompiler))
        applyArchCapOverrides(isaInfoMap, [logicArchName])
    else:
        printWarning("Library logic file {} names no architecture; "
                     "the ISA capability map is empty".format(libPath))

    assignGlobalParameters({"LibraryFormat": libraryFormat, "OutputPath": outputPath}, isaInfoMap)

    # Sleeping between data points lets the GPU cool so its clock does not
    # drift over a long run.  It does not change which kernel wins a shape:
    # every candidate for one shape is measured next to its rivals, which is
    # the comparison the grid makes.
    #
    # Set before the command line is applied, so --global-parameters still
    # overrides it.
    globalParameters["SleepPercent"] = 0

    overrideParameters = argUpdatedGlobalParameters(args)
    for key, value in overrideParameters.items():
        print1("Overriding {0}={1}".format(key, value))
        globalParameters[key] = value

    # The toolchains the benchmark step builds the kernels with.  The
    # assembler is the one the command line names (it is the c++ compiler by
    # default), and the source toolchain compiles the helper kernels.  The
    # code object version, the build id kind and the debug flag come from the
    # global parameters, so a command line override of them applies.
    asmToolchain = makeAssemblyToolchain(
        assembler,
        offloadBundler,
        globalParameters["CodeObjectVersion"],
        globalParameters["BuildIdKind"],
        globalParameters["AsmDebug"])
    srcToolchain = makeSourceToolchain(
        cxxCompiler,
        offloadBundler,
        globalParameters["AsanBuild"],
        globalParameters["BuildIdKind"],
        save_temps=False)

    # The debug configuration the benchmark and the logic step are run with:
    # the same one TensileCreateLibrary uses.
    debugConfig = makeDebugConfig({
        "SplitGSU": False,
        "PrintSolutionRejectionReason": True,
        "PrintIndexAssignmentInfo": False,
    })

    # parse library logic then setup and run benchmarks
    #
    # The library is read back in the mode it was written in, so a lazy
    # library (the default for the gfx1030 build) has to be parsed lazy.
    (rawYaml, solutions, problemSizes) = parseCurrentLibrary(
        libPath, sizePath, asmToolchain.assembler, isaInfoMap,
        globalParameters["LazyLibraryLoading"],
        logicFileArchRenames(logicArchName))

    # The problem type the solutions were built for; the benchmark client is
    # configured for it.
    problemType = solutions[0]["ProblemType"] if solutions else None

    runBenchmarking(solutions, problemSizes, outputPath, update, cxxCompiler, cCompiler,
                    assembler, offloadBundler,
                    asmToolchain=asmToolchain, srcToolchain=srcToolchain, isaInfoMap=isaInfoMap,
                    deviceId=args.device, gfxName=logicArchName, debugConfig=debugConfig,
                    problemType=problemType)

    if remake:
        # write library logic file
        if isinstance(rawYaml, dict):
            logicHeader = {
                "ScheduleName": rawYaml.get("ScheduleName"),
                "ArchitectureName": rawYaml.get("ArchitectureName"),
                "DeviceNames": rawYaml.get("DeviceNames"),
            }
        else:
            logicHeader = {
                "ScheduleName": rawYaml[1],
                "ArchitectureName": rawYaml[2],
                "DeviceNames": rawYaml[3],
            }
        # LibraryLogic.main grew the debug configuration and the ISA
        # capability map; this call site was left behind and raises
        # TypeError.  The values are the ones the logic step is run with
        # everywhere else.
        LibraryLogic.main(
            logicHeader,
            srcToolchain.compiler,
            outputPath,
            debugConfig.splitGSU,
            debugConfig.printSolutionRejectionReason,
            debugConfig.printIndexAssignmentInfo,
            isaInfoMap)

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


if __name__ == "__main__":
    main()
