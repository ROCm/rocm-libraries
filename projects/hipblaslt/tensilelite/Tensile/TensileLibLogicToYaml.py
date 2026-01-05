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

from Tensile import __version__
from Tensile import LibraryIO
from Tensile.Common.GlobalParameters import defaultBenchmarkCommonParameters
from Tensile.Common.Constants import HR
from Tensile.SolutionStructs.Problem import _defaultProblemType as defaultProblemType
from Tensile.Common.GlobalParameters import globalParameters

import argparse
import os
import sys
import re
import yaml
from typing import Optional, Tuple


class Quoted(str):
    pass


# print ""
def quotedPresenter(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='"')


class FlowList(list):
    pass


def makeFlow(value):
    if isinstance(value, list):
        return FlowList(value)
    return value


# print list in format of []
def flowSeq(dumper, value):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=True)


# ignore null
def representNone(self, _):
    return self.represent_scalar("tag:yaml.org,2002:null", "")


yaml.add_representer(Quoted, quotedPresenter)
yaml.add_representer(FlowList, flowSeq)
yaml.add_representer(type(None), representNone)


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


def setGlobalParams(versionString: dict, problemTypeState: dict) -> dict:
    res = {}
    res["MinimumRequiredVersion"] = versionString["MinimumRequiredVersion"]
    res["SleepPercent"] = 0
    res["KernelTime"] = True
    res["NumElementsToValidate"] = 0
    res["DataInitTypeBeta"] = 1
    res["DataInitTypeAlpha"] = 1
    res["DataInitTypeA"] = 12 if problemTypeState["DataType"] != "I8" else 3
    res["DataInitTypeB"] = 13 if problemTypeState["DataType"] != "I8" else 3
    res["DataInitTypeC"] = 12 if problemTypeState["DataType"] != "I8" else 3
    res["DataInitTypeD"] = 12 if problemTypeState["DataType"] != "I8" else 3
    res["PreciseKernelTime"] = 0
    res["Device"] = 0
    res["SkipSlowSolutionRatio"] = 0
    res["KeepBuildTmp"] = False
    return res


def formProblemTypeYamlData(problemTypeState: dict) -> dict:
    if len(problemTypeState) == 0:
        raise RuntimeError(
            "Length of problem Type Parameters is empty!!, Please re-check the library logic file !"
        )

    data = {}
    data["OperationType"] = problemTypeState["OperationType"]
    for problemTypeKey, problemTypeValue in problemTypeState.items():
        # Always print HighPrecisionAccumulate, TransposeA, TransposeB fields
        if problemTypeKey in [
            "DataType",
            "DestDataType",
            "ComputeDataType",
            "HighPrecisionAccumulate",
            "TransposeA",
            "TransposeB",
        ]:
            data[problemTypeKey] = problemTypeValue
            continue

        # Print default keys with no default values
        if problemTypeKey in defaultProblemType:
            if problemTypeValue != defaultProblemType[problemTypeKey]:
                data[problemTypeKey] = makeFlow(problemTypeValue)
                continue

    return data


def formGroups(MIInstruction9Bits: dict) -> dict:
    data = {}
    data["Groups"] = [[]]
    group = {}
    for forkKey, forkValue in MIInstruction9Bits.items():
        group[forkKey] = forkValue
    data["Groups"][0].append(group)
    return data


def form9BitMIInst(currentSolutionState: dict) -> dict:
    MIBlock = currentSolutionState["MIBlock"]
    MIWaveTile = currentSolutionState["MIWaveTile"]
    MIWaveGroup = currentSolutionState["MIWaveGroup"]

    if len(MIBlock) == 0 or len(MIWaveTile) == 0 or len(MIWaveGroup) == 0:
        raise RuntimeError(
            "Length of MIBlock:{0}, MIWave Tile:{1},MIWaveGroup:{2} cannot be empty".format(
                len(MIBlock), len(MIWaveTile), len(MIWaveGroup)
            )
        )

    MIBlock1 = MIBlock[0:5]

    MIInstruction9Bits = MIBlock1 + MIWaveTile + MIWaveGroup

    groups = {}
    groups["MatrixInstruction"] = FlowList(MIInstruction9Bits)
    groups["WorkGroup"] = FlowList(currentSolutionState["WorkGroup"])
    groups["MIArchVgpr"] = currentSolutionState["MIArchVgpr"]

    return groups


def formForkParams(currentIndexSolution: dict, skipMI: bool) -> dict:

    data = {}
    data["InitialSolutionParameters"] = None
    kernelLang = {}
    kernelLang["KernelLanguage"] = FlowList(["Assembly"])
    data["BenchmarkCommonParameters"] = [kernelLang]

    forkData = []
    for forkKey, forkValue in currentIndexSolution.items():
        temp = {}
        # # ignore MatrixInstruction
        if forkKey in ["MatrixInstruction", "WorkGroup"]:
            continue
        # Find the matching index for fork key name from list of dictionaries => defaultBenchmarkCommonParameters
        index = next(
            (i for i, d in enumerate(defaultBenchmarkCommonParameters) if forkKey in d),
            None,
        )
        if index is not None:
            forkValue = [forkValue]  # convert to list
            if forkValue != defaultBenchmarkCommonParameters[index][forkKey]:
                temp[forkKey] = FlowList(forkValue)
                forkData.append(temp)

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

    # Iterate over MIs in Group
    if skipMI != True and isMatrixInsEnabled:
        temp = form9BitMIInst(currentIndexSolution)
    else:
        temp = "None"

    forkData.append(formGroups(temp))

    data["ForkParameters"] = forkData

    return data


def formProblemSize(
    exactLogic: Optional[list[Tuple[list, list]]],
    solutionIndex: int,
    problemTypeStat: dict,
) -> dict:
    data = {}
    data["BenchmarkJoinParameters"] = None
    data["BenchmarkFinalParameters"] = []

    temp = {}
    # for origami exactLogic is not present so we need to create it
    if exactLogic is None:
        tPrint(
            1, "Warning: For Origami liblogics, Exact logic needs to be set manually"
        )
        temp["ProblemSizes"] = [{"Exact": FlowList([1, 1, 1, 1])}]
    else:
        for size, mapping in exactLogic:
            if mapping[0] == solutionIndex:
                temp["ProblemSizes"] = [{"Exact": FlowList(size)}]

    data["BenchmarkFinalParameters"].append(temp)

    temp = {}
    biasTypeArgs = problemTypeStat["BiasDataTypeList"]
    temp["BiasTypeArgs"] = FlowList(biasTypeArgs)

    data["BenchmarkFinalParameters"].append(temp)

    return data


def formLibraryLogic(
    scheduleName: str, deviceNames: list, architectureName: str
) -> dict:
    data = {}
    # Form final library logic string
    data["ScheduleName"] = Quoted(scheduleName)
    data["DeviceNames"] = FlowList([Quoted(deviceNames[0])])
    data["ArchitectureName"] = Quoted(architectureName)

    return data


def writeToTensileYamlFile(tensileYamlFile: str, tensileYamlData: str) -> Optional[str]:
    ret = None
    try:
        fileDir = os.path.dirname(tensileYamlFile)
        if fileDir:
            os.makedirs(fileDir, exist_ok=True)

        with open(tensileYamlFile, "w") as f:
            yaml.dump(
                tensileYamlData,
                f,
                default_flow_style=False,
                sort_keys=False,
                Dumper=yaml.Dumper,
            )
        tPrint(1, "Config library is written to {}".format(tensileYamlFile))
        ret = tensileYamlFile

    except (OSError, IOError):
        tPrint(
            1,
            "Error: Creating file {} Please provide file name in this format <filename>.yaml.".format(
                tensileYamlFile
            ),
        )
    return ret


def TensileLibLogicToYaml(
    logicFilePath: str, solutionIndex: int, tensileYamlFile: str, skipMI: bool
) -> Optional[str]:
    """Generate config from a library logic.

    This function generates a config yaml by extracting a solution from a given a library
    logic and a solution index.

    Args:
        logicFilePath: Yaml format file. Input library logic to extract from.
        solutionIndex: Solution index to extract solution from the library.
        tensileYamlFile: Config yaml file name. Creates the dir if path is given.
        skipMI: If False ignores the MI instruction.

    Returns:
        If generated the config file name otherwise None.

    Raises:
        RuntimeError: If logicFilePath cannot be read.
        RuntimeError: If solutionIndex is not in the logicFilePath.
        RuntimeError: If tensileYamlFile string is empty or name is not valid.

    Example:
        TensileLibLogicToYaml("gfx950_Cijk_Alik_Bljk_BSS_BH_BiasS_HAS_SAV_UserArgs.yaml", 0, "config.yaml", False)
    """

    tPrint(1, "")
    tPrint(1, HR)
    tPrint(1, "#")
    tPrint(1, "#  TensileLibLogicToYaml Library v{}".format(__version__))

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
    if solutionIndex == "":
        raise RuntimeError("At least one solution idx should be provided")

    # reads library logic file. AllSolutionStates=>has solution data for all solution index
    fields = LibraryIO.rawLibraryLogic(libYaml)
    (
        versionString,
        scheduleName,
        architectureName,
        deviceNames,
        problemTypeState,
        allSolutionStates,
        _,  # indexOrder
        exactLogic,
        _,  # rangeLogic
        _,  # otherFields
    ) = fields

    # Extract the solution data for the user specified solution Index
    currentIndexSolution = allSolutionStates[solutionIndex]

    if currentIndexSolution == "":
        raise RuntimeError(
            "Could not find the matching data for the solution index:{} from the library logic file, Try different solution index".format(
                solutionIndex
            )
        )

    tensileYamlFileData = {}

    # Iterate over problem type parameters and form problem type yaml data
    tensileYamlFileData["GlobalParameters"] = setGlobalParams(
        versionString, problemTypeState
    )
    benchmarkProblems = [[]]
    benchmarkProblemsData = formProblemTypeYamlData(problemTypeState)
    benchmarkProblems[0].append(benchmarkProblemsData)

    # # Iterate over Fork parameters and form the Yaml data
    benchmarkProblemsData = formForkParams(currentIndexSolution, skipMI)
    benchmarkProblemsData.update(
        formProblemSize(
            exactLogic,
            solutionIndex,
            problemTypeState,
        )
    )
    benchmarkProblems[0].append(benchmarkProblemsData)

    tensileYamlFileData["BenchmarkProblems"] = benchmarkProblems

    # # Forms the Library logic string
    problemSizeData = formLibraryLogic(scheduleName, deviceNames, architectureName)

    tensileYamlFileData["LibraryLogic"] = problemSizeData

    # Write the Formed Yaml data into the Yaml File
    return writeToTensileYamlFile(tensileYamlFile, tensileYamlFileData)


def parseArgs():
    argParser = argparse.ArgumentParser()
    argHelp = {
        "input": "Library logic file to be converted to tensile input yaml file.",
        "indices": "Comma-separated list of Solution indices from library logic File to extract. Ex: 0,3,4,5",
        "output": "Base Output file name.",
        "skipMI": "Skips the MatrixInstruction field in the tensile yaml file",
    }

    argParser.add_argument(
        "--input",
        "-i",
        action="store",
        type=os.path.realpath,
        required=True,
        default=None,
        help=argHelp["input"],
    )
    argParser.add_argument(
        "--indices",
        "-d",
        action="store",
        type=str,
        required=True,
        default="0",
        help=argHelp["indices"],
    )
    argParser.add_argument(
        "--output",
        "-o",
        action="store",
        type=os.path.realpath,
        required=True,
        default=None,
        help=argHelp["output"],
    )
    argParser.add_argument(
        "--skipMI",
        "-s",
        action="store_true",
        default=None,
        help=argHelp["skipMI"],
        required=False,
    )

    return argParser.parse_args()


def main():
    args = parseArgs()
    ids = [int(x.strip()) for x in args.indices.split(",")]
    tensileYamlFiles = []
    for id in ids:
        if len(ids) == 1:
            tensileYamlFile = args.output
        else:
            tensileYamlFile = re.sub(".yaml", f"_{int(id)}.yaml", args.output)
        TensileLibLogicToYaml(args.input, int(id), tensileYamlFile, args.skipMI)
        tensileYamlFiles.append(tensileYamlFile)
    tPrint(1, f"Tensile Files generated: {tensileYamlFiles}")
