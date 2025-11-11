################################################################################
#
# Copyright (C) 2022 Advanced Micro Devices, Inc. All rights reserved.
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

from . import __version__
from . import LibraryIO
from Tensile.Common.GlobalParameters import defaultBenchmarkCommonParameters
from Tensile.Common.Constants import HR
from Tensile.SolutionStructs.Problem import _defaultProblemType as defaultProblemType
from Tensile.Common.GlobalParameters import globalParameters
from Tensile.Common.DataType import DataType

import argparse
import os
import sys


def tPrint(verbosity: int, arg) -> None:
    """Conditionally prints input to stdout.

    If the global print level is greater than or equal to the verbosity,
    the argument is printed to stdout.

    Args:
        verbosity: Level to use for printing arg.
        arg: Item to print to stdout.
    """
    if globalParameters["ClientLogLevel"] >= verbosity:
        print(arg)
        sys.stdout.flush()

def TensileLibLogicToYaml(userArgs):
    tPrint(1, "")
    tPrint(1, HR)
    tPrint(1, "#")
    tPrint(1, "#  TensileLibLogicToYaml Library v{}".format(__version__))

    argParser = argparse.ArgumentParser()

    argParser.add_argument(
        "LibLogicFile",
        type=os.path.realpath,
        help="Library logic file to be converted to tensile input yaml file",
    )
    argParser.add_argument(
        "SolutionIndex",
        type=int,
        help="Solution index from library logic File",
        default=None,
    )
    argParser.add_argument(
        "OutputYaml",
        type=os.path.realpath,
        help="OutputYaml path where output tensile yaml files are placed",
    )

    argParser.add_argument(
        "--skipMI",
        "-s",
        action="store_true",
        help="Skips the MatrixInstruction field in the tensile yaml file"
        "i.e Thread Tile and Work Group parameters without MI",
        required=False,
    )

    args = argParser.parse_args(userArgs)
    logicFilePath = args.LibLogicFile
    solutionIndex = args.SolutionIndex
    tensileYamlFile = args.OutputYaml

    tPrint(1, "#  Library Logic: {}".format(logicFilePath))
    tPrint(1, "#  Solution Index: {}".format(solutionIndex))
    tPrint(1, "#")
    tPrint(1, HR)
    tPrint(1, "")
    libYaml = LibraryIO.readYAML(logicFilePath)
    if libYaml == "":
        raise RuntimeError(
            "Yaml file data is empty, read yaml file :{} failed".format(logicFilePath)
        )

    # reads library logic file. AllsolutionStates=>has solution data for all solution index
    fields = LibraryIO.rawLibraryLogic(libYaml)
    (
        versionString,
        scheduleName,
        architectureName,
        deviceNames,
        problemTypeState,
        allsolutionStates,
        indexOrder,
        exactLogic,
        rangeLogic,
        otherFields,
    ) = fields

    # Extract the solution data for the user specified solution Index
    currentIndexSolution = allsolutionStates[solutionIndex]

    if currentIndexSolution == "":
        raise RuntimeError(
            "Could not find the matching data for the solution index:{} from the library logic file, Try different solution index".format(
                solutionIndex
            )
        )

    # Skip the MI calculation if 9 bit MI is not needed or MatrixInstruction field is disabled
    isMatrixInsEnabled = False
    if "EnableMatrixInstruction" in currentIndexSolution:
        isMatrixInsEnabled = currentIndexSolution["EnableMatrixInstruction"]

        if (
            currentIndexSolution["EnableMatrixInstruction"]
            and currentIndexSolution["MatrixInstruction"]
        ):
            isMatrixInsEnabled = True
        else:
            tPrint(
                1,
                "Matrix instruction is disabled skipping the matrix instruction parameter ..",
            )

    tensileYamlFileData = []
    if args.skipMI != True and isMatrixInsEnabled:
        checkMacroTileThreadTileWorkGroupMatches(currentIndexSolution)
        MIInstruction9Bits = form9BitMIInstruction(currentIndexSolution)
    else:
        MIInstruction9Bits = "None"

    # Iterate over problem type parameters and form problem type yaml data
    problemTypeYamlData = formProblemTypeYamlData(problemTypeState, versionString)
    tensileYamlFileData.append(problemTypeYamlData)

    # Iterate over Fork parameters and form the Yaml data
    forkParameterYamlData = formForkParametersYamlData(currentIndexSolution)
    tensileYamlFileData.append(forkParameterYamlData)

    # Iterate over MIs in Group
    forkGroupData = formGroups(MIInstruction9Bits)
    tensileYamlFileData.append(forkGroupData)

    # Forms the Library logic string
    problemSizeYamlData = formProblemSizeYamlData(
        exactLogic, solutionIndex, scheduleName, deviceNames, architectureName
    )
    tensileYamlFileData.append(problemSizeYamlData)

    # Write the Formed Yaml data into the Yaml File
    writeToTensileYamlFile(tensileYamlFile, tensileYamlFileData)

def setEpilouges(version, dataType):
    res = {}
    res["MinimumRequiredVersion"] = version
    res["SleepPercent"] = 0
    res["KernelTime"] = True
    res["NumElementsToValidate"] = 0
    res["DataInitTypeBeta"] = 1
    res["DataInitTypeAlpha"] = 1
    res["DataInitTypeA"] = 12 if dataType != "I8" else 3
    res["DataInitTypeB"] = 13 if dataType != "I8" else 3
    res["DataInitTypeC"] = 12 if dataType != "I8" else 3
    res["DataInitTypeD"] = 12 if dataType != "I8" else 3
    res["PreciseKernelTime"] = 0
    res["Device"] = 0
    res["SkipSlowSolutionRatio"] = 0
    res["KeepBuildTmp"] = False
    return res


def formProblemTypeYamlData(problemTypeState, versionString):
    problemTypeYamlData = []
    if len(problemTypeState) == 0:
        raise RuntimeError(
            "Length of problem Type Parameters is empty!!, Please re-check the library logic file !"
        )

    # Form Global parameters section as well
    problemTypeYamlData.append("GlobalParameters:\n")
    epilouges = setEpilouges(
        versionString["MinimumRequiredVersion"], problemTypeState["DataType"]
    )
    for problemTypeKey, problemTypeValue in epilouges.items():
        problemTypeYamlData.append(
            "  {}: {}\n".format(problemTypeKey, problemTypeValue)
        )
    problemTypeYamlData.append("BenchmarkProblems:\n- - OperationType: GEMM\n")

    for problemTypeKey, problemTypeValue in problemTypeState.items():
        # Ignore DataTypeA, DataTypeB, DataTypeE, OperationType, UseBias
        if problemTypeKey in [
            "DataTypeA",
            "DataTypeB",
            "DataTypeE",
            "OperationType",
            "UseBeta",
            "UseBias",
            "Gradient",
        ]:
            continue
        # Convert TransposeA, TransposeB, UseBeta
        if problemTypeKey in ["TransposeA", "TransposeB", "UseBeta"]:
            problemTypeValue = True if problemTypeValue == 1 else False
        # Always print DataType, DestDataType, ComputeDataType
        if problemTypeKey in ["DataType", "DestDataType", "ComputeDataType"]:
            problemTypeYamlData.append(
                "    {}: {}\n".format(
                    problemTypeKey, DataType(problemTypeValue).toChar().upper()
                )
            )
            continue
        # Always print HighPrecisionAccumulate, OperationType TransposeA, TransposeB fields
        if problemTypeKey in [
            "HighPrecisionAccumulate",
            "TransposeA",
            "TransposeB"
        ]:
            problemTypeYamlData.append(
                "    {}: {}\n".format(problemTypeKey, problemTypeValue)
            )
            continue
        # Print default keys with no default values
        if problemTypeKey in defaultProblemType:
            if problemTypeValue != defaultProblemType[problemTypeKey]:
                problemTypeYamlData.append(
                    "    {}: {}\n".format(problemTypeKey, problemTypeValue)
                )
                continue

    return "".join(str(x) for x in problemTypeYamlData)

def formGroups(MIInstruction9Bits):
    formGroupsData = []
    formGroupsData.append("    - Groups:\n")
    formGroupsData.append("      - {}".format(MIInstruction9Bits))

    return "".join(str(x) for x in formGroupsData)

def formForkParametersYamlData(currentIndexSolution):
    forkParametersYamlData = []
    forkParametersYamlData.append("  - InitialSolutionParameters:\n")
    forkParametersYamlData.append("    BenchmarkCommonParameters:\n")
    forkParametersYamlData.append("    ForkParameters:\n")

    for forkKey, forkValue in currentIndexSolution.items():
        # Find the matching index for fork key name from list of dictionaries => defaultBenchmarkCommonParameters
        index = next(
            (i for i, d in enumerate(defaultBenchmarkCommonParameters) if forkKey in d),
            None,
        )
        if index != None:
            forkValue = [forkValue]  # convert to list
            if forkValue != defaultBenchmarkCommonParameters[index][forkKey]:
                forkParametersYamlData.append(
                    "    - {}: {}\n".format(forkKey, forkValue)
                )

    return "".join(str(x) for x in forkParametersYamlData)

def form9BitMIInstruction(currentSolutionState):
    MIBlock = currentSolutionState["MIBlock"]
    MIWaveTile = currentSolutionState["MIWaveTile"]
    MIWaveGroup = currentSolutionState["MIWaveGroup"]

    if len(MIBlock) == 0 or len(MIWaveTile) == 0 or len(MIWaveGroup) == 0:
        raise RuntimeError(
            "Length of MIBlock:{0}, MIWave Tile:{1},MIWaveGroup:{2} cannot be empty".format(
                len(MIBlock), len(MIWaveTile), len(MIWaveGroup)
            )
        )

    MIBlock1 = [MIBlock[i] for i in (0, 1, 2, 3, 4)]
    MIBlock5bits = ",".join([str(item) for item in MIBlock1])
    MIWaveTile2Bits = ",".join([str(item) for item in MIWaveTile])
    MIWaveGroup2Bits = ",".join([str(item) for item in MIWaveGroup])

    if (
        len(MIBlock5bits) == 0
        or len(MIWaveTile2Bits) == 0
        or len(MIWaveGroup2Bits) == 0
    ):
        raise RuntimeError(
            "Length of MIBlock5bits:{0}, MIWaveGroup2Bits:{1},MIWaveGroup2Bits:{2} cannot be empty".format(
                len(MIBlock5bits), len(MIWaveTile2Bits), len(MIWaveGroup2Bits)
            )
        )

    MIInstruction9Bits = (
        "- MatrixInstruction: ["
        + MIBlock5bits
        + ","
        + MIWaveTile2Bits
        + ","
        + MIWaveGroup2Bits
        + "]\n"
    )

    return MIInstruction9Bits

def formProblemSizeYamlData(
    exactLogic, solutionIndex, scheduleName, deviceNames, architectureName
):
    problemSizeYamlData = []

    problemSizeYamlData.append("    BenchmarkJoinParameters:\n")
    problemSizeYamlData.append("    BenchmarkFinalParameters:\n")
    problemSizeYamlData.append("    - ProblemSizes:\n")
    # Form the problem Size
    for size, mapping in exactLogic:
        if mapping[0] == solutionIndex:
            problemSizeYamlData.append(
                "        - Exact: %s             # Eff: %s  Solution Index: %s\n"
                % (size, mapping[1], mapping[0])
            )

    # Form final library logic string
    problemSizeYamlData.append("LibraryLogic:\n")
    problemSizeYamlData.append(f'  ScheduleName: "{scheduleName}"\n')
    problemSizeYamlData.append(f"  DeviceNames: {deviceNames}\n")
    problemSizeYamlData.append(f'  ArchitectureName: "{architectureName}"\n')

    return "".join(str(x) for x in problemSizeYamlData)

def checkMacroTileThreadTileWorkGroupMatches(currentSolution):
    MIBlock = currentSolution["MIBlock"]
    MIWaveTile = currentSolution["MIWaveTile"]
    MIWaveGroup = currentSolution["MIWaveGroup"]

    if MIBlock == "" or MIWaveTile == "" or MIWaveGroup == "":
        raise RuntimeError(
            "Length of MIBlock:{0}, MIWaveTile:{1},MIWaveGroup:{2} cannot be empty,Check the library logic file !\n".format(
                len(MIBlock), len(MIWaveTile), len(MIWaveGroup)
            )
        )

    if not currentSolution.keys() & {"WavefrontSize"}:
        waveFrontSize = 64
    else:
        waveFrontSize = int(currentSolution["WavefrontSize"])

    TT0, TT1, MT0, MT1, WG0, WG1 = calculateThreadTileMacroTileWorkGroupParameters(
        MIBlock, MIWaveTile, MIWaveGroup, waveFrontSize
    )

    if not currentSolution.keys() & {
        "MacroTile0",
        "MacroTile1",
        "ThreadTile",
        "WorkGroup",
        "MatrixInstM",
        "MatrixInstN",
    }:
        raise RuntimeError(
            "one or more of these fields MacroTile0,MacroTile1,ThreadTile, WorkGroup,MatrixInstM,MatrixInstN is missing in the library logic file!! ..\n"
        )

    if MT0 != currentSolution["MacroTile0"]:
        raise RuntimeError(
            "Macro Tile0 {0} does not match LibLogic value {1}".format(
                MT0, currentSolution["MacroTile0"]
            )
        )

    if MT1 != currentSolution["MacroTile1"]:
        raise RuntimeError(
            "Macro Tile1 {0} does not match LibLogic value {1}".format(
                MT1, currentSolution["MacroTile1"]
            )
        )

    if TT0 != int(currentSolution["ThreadTile"][0]):
        raise RuntimeError(
            "ThreadTile0 {0} does not match LibLogic value {1}".format(
                TT0, currentSolution["ThreadTile"][0]
            )
        )

    if TT1 != int(currentSolution["ThreadTile"][1]):
        raise RuntimeError(
            "ThreadTile1 {0} does not match LibLogic value {1}".format(
                TT1, currentSolution["ThreadTile"][1]
            )
        )

    if WG0 != int(currentSolution["WorkGroup"][0]):
        raise RuntimeError(
            "WorkGroup0 {0} does not match LibLogic value {1}\n".format(
                WG0, currentSolution["WorkGroup"][0]
            )
        )

    if WG1 != int(currentSolution["WorkGroup"][1]):
        raise RuntimeError(
            "WorkGroup1 {0} does not match LibLogic value {1}".format(
                WG1, currentSolution["WorkGroup"][1]
            )
        )

def calculateThreadTileMacroTileWorkGroupParameters(
    MIBlock, MIWaveTile, MIWaveGroup, waveFrontSize
):
    TT0 = int(MIWaveTile[0])
    TT1 = int(MIWaveTile[1])
    MT0 = int(MIBlock[0]) * int(MIBlock[4]) * int(MIWaveTile[0]) * int(MIWaveGroup[0])
    MT1 = int(MIBlock[1]) * int(MIWaveTile[1]) * int(MIWaveGroup[1])
    WG0 = int(MIBlock[0]) * int(MIBlock[4]) * int(MIWaveGroup[0])
    WG1 = int(MIWaveGroup[0]) * int(MIWaveGroup[1]) * waveFrontSize // int(WG0)

    return TT0, TT1, MT0, MT1, WG0, WG1

def writeToTensileYamlFile(tensileYamlFile, tensileYamlData):
    try:
        os.makedirs(os.path.dirname(tensileYamlFile), exist_ok=True)
        with open(tensileYamlFile, "w") as fileHandle:
            fileHandle.writelines("%s" % place for place in tensileYamlData)
            fileHandle.close()
            tPrint(
                1,
                "Successfully created the Tensile Input Yaml File {} from library logic file !!.\n".format(
                    tensileYamlFile
                ),
            )
    except (OSError, IOError):
        tPrint(
            1,
            "Error: Creating file {} Please provide file name in this format <filename>.yaml.".format(
                tensileYamlFile
            ),
        )

def main():
    TensileLibLogicToYaml(sys.argv[1:])
