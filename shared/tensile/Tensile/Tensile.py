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

if __name__ == "__main__":
    print("This file can no longer be run as a script.  Run 'Tensile/bin/Tensile' instead.")
    exit(1)

import os
import sys
import argparse
import shutil

from .Common import globalParameters, tPrint, printExit, ensurePath, \
    assignGlobalParameters, restoreDefaultGlobalParameters, HR, gfxArch
from . import BenchmarkProblems
from . import ClientWriter
from . import LibraryIO
from . import LibraryLogic
from . import __version__
from datetime import datetime
from .Utilities.Profile import profile
from .Utilities.Toolchain import validateToolchain, ToolchainDefaults

###############################################################################
# Execute Steps in Config
# called from Tensile() below
# calls
#   BenchmarkProblems.main() to run benchmark steps
#   LibraryLogic.main() to analyse final benchmark data and produce logic/yaml
#   ClientWriter.main() to create client which calls library based on above yaml
################################################################################
def executeStepsInConfig(config):

    ##############################################################################
    # Benchmark Problems
    ##############################################################################
    if "BenchmarkProblems" in config:
        BenchmarkProblems.main(config["BenchmarkProblems"], config["UseCache"])
        tPrint(1, "")

    ##############################################################################
    # Library Logic
    ##############################################################################
    libraryLogicDataPath = os.path.join(globalParameters["WorkingPath"], \
      globalParameters["LibraryLogicPath"])
    if "LibraryLogic" in config:
        if os.path.exists(libraryLogicDataPath):
            libraryLogicFiles = os.listdir(libraryLogicDataPath)
        else:
            libraryLogicFiles = []
        if len(libraryLogicFiles) < 1 or globalParameters["ForceRedoLibraryLogic"]:
            if config["LibraryLogic"] != None:
                libraryLogicConfig = config["LibraryLogic"]
            else:
                libraryLogicConfig = {}
            LibraryLogic.main(libraryLogicConfig)
            tPrint(1, "")
        else:
            tPrint(1, "# LibraryLogic already done.")
        tPrint(1, "")

    ##############################################################################
    # Write Client
    ##############################################################################
    if "LibraryClient" in config:
        if config["LibraryClient"] != None:
            libraryClientConfig = config["LibraryClient"]
        else:
            libraryClientConfig = {}
        ClientWriter.main(libraryClientConfig)
        tPrint(1, "")


def addCommonArguments(argParser):
    """
    Add a common set of arguments to `argParser`.

    Currently used by the main Tensile script and the unit tests but could also be used for TensileCreateLibrary.
    """

    def splitExtraParameters(par):
        """
        Allows the --global-parameters option to specify any parameters from the command line.
        """
        (key, value) = par.split("=")
        value = eval(value)
        return (key, value)

    argParser.add_argument("-d", "--device", dest="device", type=int, \
        help="override which device to benchmark")
    argParser.add_argument("-p", "--platform", dest="platform", type=int, \
        help="override which OpenCL platform to benchmark")
    argParser.add_argument("--runtime-language", dest="RuntimeLanguage", \
        choices=["HIP"], help="override which runtime language to use")
    argParser.add_argument("--code-object-version", dest="CodeObjectVersion", \
        choices=["default", "V4", "V5"], help="HSA code-object version")
    argParser.add_argument("--arch", dest="arch", help="override gfx arch version")
    argParser.add_argument("-v", "--verbose", action="store_true", \
        help="set PrintLevel=3")
    argParser.add_argument("--debug", dest="debug", action="store_true", \
        help="set PrintLevel=3 and CMakeBuildType=Debug")
    argParser.add_argument("--short-names", dest="shortNames", action="store_true", \
        help="use serial kernel and solution names")
    argParser.add_argument("--no-merge-files", dest="noMergeFiles", action="store_true", \
        help="kernels and solutions written to individual files")
    argParser.add_argument(
        "--cxx-compiler",
        dest="CxxCompiler",
        default=ToolchainDefaults.CXX_COMPILER,
        type=str,
        help="C++ compiler used when generating binaries."
        "On Linux, amdclang++ (default) or hipcc. On Windows clang++ (default) or hipcc. "
        "On Windows, include the file extension, or extensions will be searched according to the PATHEXT environment variable. "
        "Pass a fully-qualified path to override environment inspection when searching for the compiler."
    )
    argParser.add_argument(
        "--c-compiler",
        dest="CCompiler",
        default=ToolchainDefaults.C_COMPILER,
        type=str,
    )
    argParser.add_argument(
        "--assembler",
        dest="Assembler",
        default=ToolchainDefaults.ASSEMBLER,
        type=str,
    )
    argParser.add_argument(
        "--offload-bundler",
        dest="OffloadBundler",
        default=ToolchainDefaults.OFFLOAD_BUNDLER,
        type=str,
    )
    argParser.add_argument("--library-format", dest="LibraryFormat", choices=["yaml", "msgpack"], \
        action="store", help="select which library format to use")
    argParser.add_argument("--client-lock", default=None)
    argParser.add_argument("--prebuilt-client",
        default=os.path.abspath(os.path.join('build', 'client', 'tensile-client')),
        help="Specify the full path to a pre-built tensile-client executable")
    argParser.add_argument("--asm-cache", dest="AsmCacheFile", action="store", type=str, \
        help="Path to ASM cache YAML file. If it does not exist, generate the cache. If it does exist, use the cache file")
    argParser.add_argument("--global-parameters", nargs="+", type=splitExtraParameters, default=[])


def argUpdatedGlobalParameters(args):
    """
    Returns a dictionary with `globalParameters` keys that should be updated based on `args`.
    """
    rv = {}
    # override config with command-line options
    if args.device:
        tPrint(1, "# Command-line override: Device")
        rv["Device"] = args.device
    if args.platform:
        tPrint(1, "# Command-line override: Platform")
        rv["Platform"] = args.platform
    if args.RuntimeLanguage:
        tPrint(1, "# Command-line override: RuntimeLanguage")
        rv["RuntimeLanguage"] = args.RuntimeLanguage
    if args.CodeObjectVersion:
        tPrint(1, "# Command-line override: CodeObjectVersion")
        rv["CodeObjectVersion"] = args.CodeObjectVersion
    if args.arch:
        tPrint(1, "# Command-line override: CurrentISA")
        rv["CurrentISA"] = gfxArch(args.arch)
    if args.verbose:
        tPrint(1, "# Command-line override: PrintLevel")
        rv["PrintLevel"] = 3
    if args.debug:
        tPrint(1, "# Command-line override: Debug")
        rv["PrintLevel"] = 3
        rv["CMakeBuildType"] = "Debug"
    if args.shortNames:
        rv["ShortNames"] = True
    if args.noMergeFiles:
        rv["MergeFiles"] = False
    if args.CxxCompiler:
        rv['CxxCompiler'] = args.CxxCompiler
    tPrint(1, "")
    if args.client_lock:
        rv["ClientExecutionLockPath"] = args.client_lock
    if args.prebuilt_client:
        args.prebuilt_client = os.path.abspath(args.prebuilt_client)
        rv["PrebuiltClient"] = args.prebuilt_client

    for key, value in args.global_parameters:
        rv[key] = value

    return rv


################################################################################
# Tensile
# - below entry points call here
################################################################################
@profile
def Tensile(userArgs):
    global globalParameters

    # 1st half of splash
    tPrint(1, "")
    tPrint(1, HR)
    tPrint(1, "#")
    tPrint(1, "#  Tensile v%s" % (__version__))

    # setup argument parser
    # yapf: disable
    argParser = argparse.ArgumentParser()
    argParser.add_argument("config_file", type=os.path.realpath, nargs="+",
            help="Benchmark config.yaml file")
    argParser.add_argument("output_path", \
            help="Path to conduct benchmark and write output files")
    argParser.add_argument("--version", action="version", \
            version="%(prog)s {version}".format(version=__version__))
    argParser.add_argument("--alternate-format", dest="AlternateFormat", action="store_true",
            help="Alternate format for config_file(s): first file is alternate config "
            "and optional second file is size list")
    argParser.add_argument("--no-cache", dest="NoCache", action="store_true",
            help="Ignore cache; redo parameter forking and solution generation")
    # yapf: enable

    addCommonArguments(argParser)
    args = argParser.parse_args(userArgs)
    configPaths = args.config_file
    altFormat = args.AlternateFormat
    useCache = not args.NoCache

    if altFormat and len(configPaths) > 2:
        printExit("Only 1 or 2 config_files are accepted for the alternate config format: "
                  "the alternate config file and an optional size list")
    elif not altFormat and len(configPaths) != 1:
        printExit("Only 1 config_file is accepted for the default config format. "
                  "Did you mean to add '--alternate-formate'?")

    # 2nd half of splash
    if len(configPaths) == 1:
        tPrint(1, "#  Config: {}".format(configPaths[0]))
    else:
        tPrint(1, "#  Configs: {} and {}".format(configPaths[0], configPaths[1]))
    tPrint(1, "#  Date & Time: %s" % (datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    tPrint(1, "#")
    tPrint(1, HR)
    tPrint(1, "")

    tPrint(1, "# Restoring default globalParameters")
    restoreDefaultGlobalParameters()

    # CxxCompiler and LibraryFormat needs to be updated before assignGlobalParameters.
    if args.CxxCompiler:
        globalParameters['CxxCompiler'] = args.CxxCompiler
    if args.LibraryFormat:
        globalParameters['LibraryFormat'] = args.LibraryFormat

    # default config format
    if not altFormat:
        config = LibraryIO.readYAML(configPaths[0])
    # convert alternate format into default format
    else:
        base = LibraryIO.readYAML(configPaths[0])
        sizes = []
        if len(configPaths) == 2:
            sizes = LibraryIO.readYAML(configPaths[1])

        config = {"GlobalParameters": base.get("GlobalParameters")}
        if "LibraryLogic" in base and len(sizes) > 0:
            config["LibraryLogic"] = base["LibraryLogic"]
        if "LibraryClient" in base and len(sizes) > 0:
            config["LibraryClient"] = None

        solParams = {
            "BenchmarkCommonParameters": base.get("BenchmarkCommonParameters"),
            "ForkParameters": base.get("ForkParameters"),
            "GroupForkParameters": base.get("GroupForkParameters"),
            "BenchmarkFinalParameters": [{
                "ProblemSizes": sizes
            }]
        }
        config["BenchmarkProblems"] = [[base["ProblemType"], solParams]]

    config["UseCache"] = useCache
    globalParameters["ConfigPath"] = configPaths

    capabilitiesCache = LibraryIO.initAsmCapsCache(args.AsmCacheFile)

    (
        cxxCompiler,
        cCompiler,
        assembler,
        offloadBundler,
        hipconfig,
        deviceEnumerator
    ) = validateToolchain(
        args.CxxCompiler,
        args.CCompiler,
        args.Assembler,
        args.OffloadBundler,
        ToolchainDefaults.HIP_CONFIG,
        ToolchainDefaults.DEVICE_ENUMERATOR
    )
    params = config.get("GlobalParameters", {})
    params["CxxCompiler"] = cxxCompiler
    params["CCompiler"] = cCompiler
    params["Assembler"] = assembler
    params["OffloadBundler"] = offloadBundler
    params["HipConfig"] = hipconfig
    params["ROCmAgentEnumeratorPath"] = deviceEnumerator
    assignGlobalParameters(params, capabilitiesCache)

    if globalParameters["CacheAsmCaps"]:
        LibraryIO.writeAsmCapsCache(args.AsmCacheFile, globalParameters["AsmCaps"])

    globalParameters["OutputPath"] = ensurePath(os.path.abspath(args.output_path))
    globalParameters["WorkingPath"] = globalParameters["OutputPath"]

    overrideParameters = argUpdatedGlobalParameters(args)

    for key, value in overrideParameters.items():
        print("Overriding {0}={1}".format(key, value))
        globalParameters[key] = value

    executeStepsInConfig(config)

    if not globalParameters["KeepBuildTmp"]:
        for root, subdirs, files in os.walk(globalParameters["OutputPath"]):
            for d in subdirs:
                if d == "build_tmp":
                    shutil.rmtree(os.path.join(root, d))
                    break

def TensileConfigPath(*args):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "Configs", *args)


def TensileTestPath(*args):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "Tests", *args)


################################################################################
# Entry points
# installed "tensile" command
################################################################################

def main():
    Tensile(sys.argv[1:])
