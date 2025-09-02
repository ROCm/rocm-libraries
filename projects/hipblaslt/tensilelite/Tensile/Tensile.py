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

if __name__ == "__main__":
    print("This file can no longer be run as a script.  Run 'Tensile/bin/Tensile' instead.")
    exit(1)

import os
import subprocess
import sys
import argparse

from datetime import datetime
from pathlib import Path
from typing import Dict

from Tensile import __version__
from Tensile.Common import print1, printExit, printWarning, ensurePath, HR, isRhel8, \
                           LIBRARY_LOGIC_DIR, setVerbosity, IsaInfo, makeDebugConfig, \
                           DebugConfig, IsaVersion, coVersionMap
from Tensile.Common.Architectures import detectGlobalCurrentISA, isaToGfx
from Tensile.Common.Capabilities import makeIsaInfoMap
from Tensile.Common.GlobalParameters import globalParameters, assignGlobalParameters, \
                                            restoreDefaultGlobalParameters
from Tensile.Toolchain.Assembly import AssemblyToolchain, makeAssemblyToolchain
from Tensile.Toolchain.Source import SourceToolchain, makeSourceToolchain
from Tensile.Toolchain.Validators import validateToolchain, ToolchainDefaults
from Tensile.Utilities.Decorators.Profile import profile
from Tensile import BenchmarkProblems
from Tensile import ClientWriter
from Tensile import LibraryIO
from Tensile import LibraryLogic

TENSILE_SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
TENSILE_CLIENT_PATH = Path('build_tmp') / 'tensilelite' / 'client' / 'tensilelite-client'
TENSILE_CLIENT_PATH = TENSILE_SCRIPT_DIR.parent / TENSILE_CLIENT_PATH

###############################################################################
# Execute Steps in Config
# called from Tensile() below
# calls
#   BenchmarkProblems.main() to run benchmark steps
#   LibraryLogic.main() to analyse final benchmark data and produce logic/yaml
#   ClientWriter.main() to create client which calls library based on above yaml
################################################################################
@profile
def executeStepsInConfig(
        config: dict,
        outputPath: Path,
        asmToolchain: AssemblyToolchain,
        srcToolchain: SourceToolchain,
        isaInfoMap: Dict[str, IsaInfo],
        cCompiler: str,
        debugConfig: DebugConfig,
        deviceId: int,
        probSolDict: dict
   ):
    """Conducts the steps in the provided ``config`` according to the Tensile workflow.

    The top-level steps are:
    1. BenchmarkProblems: Runs the benchmarking steps and generates the directories
        build_tmp, 1_BenchmarkProblems, 2_BenchmarkData
    2. LibraryLogic: Analyzes the benchmark data, makes logic files, and generates
        the directory 3_LibraryLogic
    3. LibraryClient: Makes the client callable libraries and generates the
        directory 4_LibraryClient

    Args:
        config (dict): The configuration dictionary.
        outputPath (Path): The path to the top-level build directory.
        asmToolchain (AssemblyToolchain): The toolchain for making assembly kernels.
        srcToolchain (SourceToolchain): The toolchain for making source kernels.
        cCompiler (str): The C compiler to use.
    """

    buildTmpPath = outputPath / "build_tmp"
    ##############################################################################
    # Benchmark Problems
    ##############################################################################
    gfxName = isaToGfx(next(iter(isaInfoMap)))
    if "BenchmarkProblems" in config:
        BenchmarkProblems.main(
            config["BenchmarkProblems"],
            config["UseCache"],
            asmToolchain,
            srcToolchain,
            cCompiler,
            outputPath,
            buildTmpPath,
            debugConfig,
            deviceId,
            gfxName,
            isaInfoMap,
            probSolDict,
        )
        print1("")

    ##############################################################################
    # Library Logic
    ##############################################################################
    libraryLogicDataPath = os.path.join(outputPath, LIBRARY_LOGIC_DIR)
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
            LibraryLogic.main(
                libraryLogicConfig,
                srcToolchain.compiler,
                outputPath,
                debugConfig.splitGSU,
                debugConfig.printSolutionRejectionReason,
                debugConfig.printIndexAssignmentInfo,
                isaInfoMap,
            )
            print1("")
        else:
            print1("# LibraryLogic already done.")
        print1("")

    ##############################################################################
    # Write Client
    ##############################################################################
    if "LibraryClient" in config:
        if config["LibraryClient"] != None:
            libraryClientConfig = config["LibraryClient"]
        else:
            libraryClientConfig = {}
        ClientWriter.main(
            libraryClientConfig,
            asmToolchain.assembler,
            cCompiler,
            isaInfoMap,
            outputPath,
            deviceId,
            gfxName,
        )
        print1("")


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

    argParser.add_argument("-d", "--device", dest="device", default=0, type=int, \
        help="override which device to benchmark")
    argParser.add_argument("-p", "--platform", dest="platform", type=int, \
        help="override which OpenCL platform to benchmark")
    argParser.add_argument("--runtime-language", dest="RuntimeLanguage", \
        choices=["HIP", "OCL"], help="override which runtime language to use")
    argParser.add_argument("--code-object-version", dest="CodeObjectVersion", \
        choices=["4", "5", "V4", "V5", "default"], action="store", default="4", help="HSA code-object version")
    argParser.add_argument("-v", "--verbose", action="store_true", \
        help="set PrintLevel=2")
    argParser.add_argument("--debug", dest="debug", action="store_true", \
        help="set PrintLevel=2 and CMakeBuildType=Debug")
    argParser.add_argument("--cxx-compiler", dest="CxxCompiler", \
        action="store", default=ToolchainDefaults.CXX_COMPILER, help="select which C++/HIP compiler to use")
    argParser.add_argument("--c-compiler", dest="CCompiler", \
        action="store", default=ToolchainDefaults.C_COMPILER, help="select which C compiler to use")
    argParser.add_argument("--assembler", dest="Assembler", \
        action="store", default=ToolchainDefaults.ASSEMBLER, help="select which assembler to use")
    argParser.add_argument("--offload-bundler", dest="OffloadBundler", \
        action="store", default=ToolchainDefaults.OFFLOAD_BUNDLER, help="select which offload bundler to use")
    argParser.add_argument("--device-enumerator", dest="DeviceEnumerator", \
        action="store", default=ToolchainDefaults.DEVICE_ENUMERATOR, help="select which device enumerator to use")
    argParser.add_argument("--logic-format", dest="LogicFormat", choices=["yaml", "json"], \
        action="store", default="yaml", help="select which logic format to use")
    argParser.add_argument("--library-format", dest="LibraryFormat", choices=["yaml", "msgpack"], \
        action="store", default="yaml", help="select which library format to use")
    argParser.add_argument("--client-lock", default=None)
    argParser.add_argument("--prebuilt-client", default=str(TENSILE_CLIENT_PATH),
        type=os.path.abspath, help="Specify the full path to a pre-built tensilelite-client executable")

    argParser.add_argument("--global-parameters", nargs="+", type=splitExtraParameters, default=[])


def argUpdatedGlobalParameters(args):
    """
    Returns a dictionary with `globalParameters` keys that should be updated based on `args`.
    """
    rv = {}
    # override config with command-line options
    if args.platform:
        print1("# Command-line override: Platform")
        rv["Platform"] = args.platform
    if args.RuntimeLanguage:
        print1("# Command-line override: RuntimeLanguage")
        rv["RuntimeLanguage"] = args.RuntimeLanguage
    if args.CodeObjectVersion:
        print1("# Command-line override: CodeObjectVersion")
        rv["CodeObjectVersion"] = args.CodeObjectVersion
    if args.debug:
        print1("# Command-line override: Debug")
        rv["CMakeBuildType"] = "Debug"
    if args.client_lock:
        rv["ClientExecutionLockPath"] = args.client_lock
    if args.prebuilt_client:
        rv["PrebuiltClient"] = args.prebuilt_client

    for key, value in args.global_parameters:
        rv[key] = value

    PyTestBuildArchNames = os.environ.get("PyTestBuildArchNames")
    if PyTestBuildArchNames != None and len(PyTestBuildArchNames) > 0:
        rv["Architecture"] = PyTestBuildArchNames

    return rv

def get_gpu_max_frequency_smi(device_id):
    '''
    Get the maximum frequency of the specified GPU device
    '''
    try:
        # Run rocm-smi command and capture output
        result = subprocess.run(['rocm-smi', '-s'], capture_output=True, text=True)

        if result.returncode != 0:
           print(f"Error running rocm-smi: {result.stderr}")
           return None

        # Parse the output
        lines = result.stdout.split('\n')
        sclk_section = False
        frequencies = []

        # Look for the sclk section of the specified device
        for line in lines:
            line = line.split(" ")
            if 'sclk' in line and f"GPU{device_id}" in line:
                sclk_section = True
                continue

           # Parse frequencies in the sclk section
            if sclk_section:
                for part in line:
                    if part.endswith("Mhz"):
                        try:
                            frequency = part.replace("Mhz", "")
                            frequencies.append(int(frequency))
                        except ValueError:
                            print(f"Error parsing frequency: {part}")
                        break
                if "socclk" in line:
                    break

        # Return the maximum frequency found
        return max(frequencies) if frequencies else None

    except Exception as e:
       print(f"Error: {e}")
       return None

def get_gpu_max_frequency(device_id):
    try:
        from hip import hip
    except ImportError:
        print("HIP module not found. Installing it now...")
        # Install the HIP module using pip
        subprocess.run("python3 -m pip install --upgrade pip", shell=True)
        subprocess.run("python3 -m pip install --index-url https://test.pypi.org/simple/ hip-python", shell=True)

        from hip import hip
        print("HIP module successfully installed.")

    def hip_check(call_result):
        err, result = call_result[0], call_result[1]
        if isinstance(err, hip.hipError_t) and err != hip.hipError_t.hipSuccess:
            return None
        return result

    attrib = hip.hipDeviceAttribute_t.hipDeviceAttributeClockRate
    try:
        freq = hip_check(hip.hipDeviceGetAttribute(attrib, device_id))
    except:
        freq = None

    return freq // 1000 if freq else None

def get_user_max_frequency():
    '''
    Get the maximum frequency from the user when the GPU frequency cannot be determined
    '''
    while True:
        try:
            user_input = input("Please enter the maximum frequency (MHz): ")

            frequency = int(user_input)

            if frequency <= 0:
                print("Error: Frequency must be greater than 0 MHz")
                continue

            return frequency

        except ValueError:
            print("Error: Please enter a valid number")
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again")

def store_max_frequency(max_frequency):
    try:
        os.environ["MAX_FREQ"] = str(max_frequency)
        return True
    except Exception as e:
        print(f"Error setting MAX_FREQ environment variable: {e}")
        return False

def restore_prob_sol_map(logfile):

    print(f"#  Restore Previous Tuning From Log File: {str(logfile)}")

    startHint = "run,problem-progress,"
    finTuningHint = "clientExit"
    finProblemHint = "####"
    keyword = "Contraction"
    runningTuning = False

    prob_best_us_record = {}
    prob_sol_map = {}
    cur_prob_idx = -1
    cur_sol_idx = -1
    last_fin_prob = -1
    allProblems = 0
    allSolutions = 0

    try:
        with open(logfile, 'r') as sh_file:
            # Iterate over each line in the file
            for line in sh_file:
                line = line.strip()
                # run until we see a line: "run,problem-progress,...."
                if not runningTuning:
                    runningTuning = line.startswith(startHint)
                    continue

                # following lines should be the bench results
                if runningTuning:
                    # we see a line with clientExit, then finish parsing log.
                    if line.startswith(finTuningHint):
                        print1(f'#  Parsed log finished the tuning with line: {line}')
                        break

                    # a line starts and ends with "####" is printed in postProblem() which means the cur_prob completed tuning.
                    elif line.startswith(finProblemHint) and line.endswith(finProblemHint):
                        # print1(f'#  Parsed Prob-Idx: {cur_prob_idx} completed tuning.')
                        last_fin_prob = cur_prob_idx

                    # not a valid tuning result line (a valid tuning result line contains "Contraction")
                    elif keyword not in line:
                        continue

                    # a valid tuning result line
                    else:
                        tokens = line.split('"')
                        # split by '"' to extract "(problem-size-desc)"
                        # there should be 3 tokens: { "some-info," , "(problem-size-desc)" , ",rest-of-other-info"}
                        assert len(tokens) == 3, f'bench result line: {line} is not fitting the expected format'
                        # original tokens[1] looks like "(M,N,B,K)", replace it with a string without comma
                        tokens[1] = 'problem-sizes'
                        newline = tokens[0] + tokens[1] + tokens[2]
                        # now the new line = "some-info, 'problem-sizes' , rest-of-other-info"
                        # 0,29/31,8/247,Contraction....,'problem-sizes',None,,None,[SOLUTION-NAME],[VALIDATION],[53.6248(us)],[63354.7(gflops)],.........
                        tokens = newline.split(',')
                        probInfo = tokens[1].split('/')
                        solInfo = tokens[2].split('/')
                        assert len(probInfo) == 2, f'tokens[1]: {tokens[1]} should be a token as probId/AllProb'
                        assert len(solInfo) == 2, f'tokens[2]: {tokens[2]} should be a token as solId/AllSol'
                        # solName = tokens[8] # Might be another choice to put in map...
                        usTimeInfo = float(tokens[10]) # tokens[10] should be the mju-sec
                        cur_prob_idx = int(probInfo[0])
                        cur_sol_idx = int(solInfo[0])

                        # This can handle logs with or without PrintWinnerOnly
                        current_best_us_for_prob = prob_best_us_record.get(cur_prob_idx)
                        if current_best_us_for_prob is None:
                            # print(f"None, added for probID: {probInfo[0]}, time: {usTimeInfo}")
                            prob_best_us_record.update( {cur_prob_idx : usTimeInfo} )
                            prob_sol_map.update( {cur_prob_idx : cur_sol_idx} )
                        else:
                            # update the winner if better
                            if usTimeInfo < current_best_us_for_prob:
                                # print(f"Better, updated for probID: {probInfo[0]}, new time: {usTimeInfo}, old time: {current_best_us_for_prob}")
                                prob_best_us_record.update( {cur_prob_idx : usTimeInfo} )
                                prob_sol_map.update( {cur_prob_idx : cur_sol_idx} )

                        # for printing information
                        if allProblems == 0:
                            allProblems = int(probInfo[1]) + 1
                        if allSolutions == 0:
                            allSolutions = int(solInfo[1]) + 1

        # The final updated "cur_prob_idx" is not finished since we don't see its postProblem() "####" line
        # -> The problem did not complete tuning, so we need to remove it from map
        if cur_prob_idx != last_fin_prob:
            prob_sol_map.pop(cur_prob_idx)
            print1(f'#  Parsed Prob-Idx {cur_prob_idx} did not complete tuning. Remove it from map.')

    except FileNotFoundError:
        print(f'Error: The file {logfile} was not found.')

    except Exception as e:
        print(f'An error occurred: {e}')

    print(f"#  Parsed log is for a '{allProblems}-problems-vs-{allSolutions}-solutions' tuning.")

    return prob_sol_map


################################################################################
# Tensile
# - below entry points call here
################################################################################
def Tensile(userArgs):
    global globalParameters

    print1("")
    print1(HR)
    print1("#")
    print1("#  Tensile v%s" % (__version__))

    argParser = argparse.ArgumentParser()
    argParser.add_argument("ConfigFile", type=os.path.realpath, nargs="+",
            help="Benchmark config.yaml file")
    argParser.add_argument("OutputPath", \
            help="Path to conduct benchmark and write output files")
    argParser.add_argument("--version", action="version", \
            version="%(prog)s {version}".format(version=__version__))
    argParser.add_argument("--alternate-format", dest="AlternateFormat", action="store_true",
            help="Alternate format for config_file(s): first file is alternate config "
            "and optional second file is size list")
    argParser.add_argument("--use-cache", dest="useCache", action="store_true",
            help="Ignore cache; redo parameter forking and solution generation")
    argParser.add_argument("--restore-from-log", type=str, dest="RestoreLog",
            help="A log file captured in previous tuning. ONLY RELIABLE when configs yaml not changes")

    addCommonArguments(argParser)
    args = argParser.parse_args(userArgs)
    configPaths = args.ConfigFile
    altFormat = args.AlternateFormat
    useCache = args.useCache
    outputPath = Path(ensurePath(os.path.abspath(args.OutputPath)))
    print1(f"#  OutputPath: {str(outputPath)}")

    setVerbosity(2 if (args.debug or args.verbose) else 1)

    if altFormat and len(configPaths) > 2:
        printExit("Only 1 or 2 config_files are accepted for the alternate config format: "
                  "the alternate config file and an optional size list")
    elif not altFormat and len(configPaths) != 1:
        printExit("Only 1 config_file is accepted for the default config format. "
                  "Did you mean to add '--alternate-formate'?")

    prob_sol_map = {}
    if args.RestoreLog:
        restoreLogPath = Path(os.path.abspath(args.RestoreLog))
        if not os.path.exists(restoreLogPath):
            printExit(f'restore log file: {restoreLogPath} is not existing, abort. Please check')
        prob_sol_map = restore_prob_sol_map(restoreLogPath)
        for key,value in prob_sol_map.items():
            print1(f'#  Restored Prob-Solution From Log: [Prob:{key},Sol:{value}]')

    # 2nd half of splash
    if len(configPaths) == 1:
        print1("#  Config: {}".format(configPaths[0]))
    else:
        print1("#  Configs: {} and {}".format(configPaths[0], configPaths[1]))
    print1("#  Date & Time: %s" % (datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
    print1("#")
    print1(HR)
    print1("")

    print1("# Restoring default globalParameters")
    restoreDefaultGlobalParameters()

    if args.LogicFormat:
        globalParameters['LogicFormat'] = args.LogicFormat
    if args.LibraryFormat:
        globalParameters['LibraryFormat'] = args.LibraryFormat
    globalParameters['CodeObjectVersion'] = coVersionMap[args.CodeObjectVersion]
    print1(f"# Code Object Version: {globalParameters['CodeObjectVersion']}")

    # default config format
    if not altFormat:
        config = LibraryIO.read(configPaths[0])
    # convert alternate format into default format
    else:
        base = LibraryIO.read(configPaths[0])
        sizes = []
        if len(configPaths) == 2:
            sizes = LibraryIO.read(configPaths[1])

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

    asm_debug = config["GlobalParameters"].get("AsmDebug", False)
    device_id = config["GlobalParameters"].get("Device", int(args.device))
    UseEffLike = config["GlobalParameters"].get("UseEffLike", globalParameters["UseEffLike"])
    UseEffLike = False if isRhel8() else UseEffLike

    if 'LibraryLogic' in config and UseEffLike:
        max_frequency = get_gpu_max_frequency(device_id)

        if not max_frequency or max_frequency <= 0:
            max_frequency = get_gpu_max_frequency_smi(device_id) # Using rocm-smi just in case

        if not max_frequency or max_frequency <= 0:
            print(f"Could not detect valid GPU frequency for device {device_id}")
            max_frequency = get_user_max_frequency()

        print(f"Successfully retrieve Max frequency: {max_frequency} for device {device_id}")
        store_max_frequency(max_frequency)

    cxxCompiler, \
    cCompiler, \
    offloadBundler, \
    enumerator = validateToolchain(args.CxxCompiler,
                                   args.CCompiler,
                                   args.OffloadBundler,
                                   ToolchainDefaults.DEVICE_ENUMERATOR)
    asmToolchain = makeAssemblyToolchain(
        cxxCompiler,
        offloadBundler,
        args.CodeObjectVersion,
        debug=asm_debug,
    )
    srcToolchain = makeSourceToolchain(
        cxxCompiler,
        offloadBundler,
    )

    if "ISA" in config["GlobalParameters"]:
        isaList = [IsaVersion(isa[0], isa[1], isa[2]) for isa in config["GlobalParameters"]["ISA"]]

    else:
        isaList = [detectGlobalCurrentISA(device_id, enumerator)]

    if IsaVersion(9,5,0) in isaList:
        printWarning("HardwareMonitor currently disabled for gfx950")
        globalParameters["HardwareMonitor"] = False

    isaInfoMap = makeIsaInfoMap(isaList, cxxCompiler)
    assignGlobalParameters(config.get("GlobalParameters", {}), isaInfoMap)

    overrideParameters = argUpdatedGlobalParameters(args)

    debugConfig = makeDebugConfig(config["GlobalParameters"])

    for key, value in overrideParameters.items():
        print("Overriding {0}={1}".format(key, value))
        globalParameters[key] = value

    if "MaxFileName" in globalParameters or "MaxFileName" in config:
        printWarning("MaxFileName is no longer configurable, it will be automatically set to 64")

    executeStepsInConfig(config, outputPath, asmToolchain, srcToolchain, isaInfoMap, cCompiler, debugConfig, device_id, prob_sol_map)

def TensileConfigPath(*args):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "Configs", *args)


def TensileTestPath(*args):
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "Tests", *args)


################################################################################
# Entry points
# the first several of these can be deprecated, only main() is used
################################################################################


# installed "tensile_rocblas_sgemm" command
def TensileROCBLASSGEMM():
    Tensile([TensileConfigPath("rocblas_sgemm.yaml"), "."])


# installed "tensile_rocblas_dgemm" command
def TensileROCBLASDGEMM():
    Tensile([TensileConfigPath("rocblas_dgemm.yaml"), "."])


# installed "tensile_rocblas_cgemm" command
def TensileROCBLASCGEMM():
    Tensile([TensileConfigPath("rocblas_cgemm.yaml"), "."])


# installed "tensile_rocblas_zgemm" command
def TensileROCBLASZGEMM():
    Tensile([TensileConfigPath("rocblas_zgemm.yaml"), "."])


# installed "tensile_sgemm" command
def TensileSGEMM5760():
    Tensile([TensileConfigPath("sgemm_5760.yaml"), "."])


# installed "tensile" command
def main():
    Tensile(sys.argv[1:])
