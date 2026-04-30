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

from .CustomKernels import getCustomKernelConfig
from rocisa.enum import DataTypeEnum
from . import SolutionLibrary
from .CustomYamlLoader import load_yaml_stream
from Tensile import __version__
from Tensile.Common import printExit, printWarning, print2, \
                           versionIsCompatible, IsaInfo
from Tensile.Common.TimingInstrumentation import timing_context
from Tensile.Common.Architectures import gfxToIsa
from Tensile.SolutionStructs import Solution, ProblemSizes
from Tensile.SolutionStructs.Solution import getTypeMismatchCollector, resetTypeMismatchCollector
from Tensile.SolutionStructs.Problem import ProblemType, problemTypeToEnum

from typing import IO, NamedTuple, List, Dict, Optional
from Tensile.SolutionStructs.Solution import BiasTypeArgs, ActivationArgs
import io
import os
import sys
import subprocess
import re

try:
    import orjson as json
except ImportError:
    try:
        import ujson as json
        print2("orjson not installed. Fallback to ujson.")
    except ImportError:
        try:
            import simplejson as json
            print2("orjson, ujson not installed. Fallback to simplejson.")
        except ImportError:
            import json
            print2("orjson, ujson, simplejson not installed. Fallback to json.")

try:
    import yaml
except ImportError:
    printExit(
        "You must install PyYAML to use Tensile (to parse config files). See http://pyyaml.org/wiki/PyYAML for installation instructions."
    )

try:
    from yaml import CSafeLoader as yamlLoader
except ImportError:
    from yaml import SafeLoader as yamlLoader
    printWarning("CSafeLoader not installed. Fallback to SafeLoader.")

try:
    from yaml import CSafeDumper as yamlDumper
except ImportError:
    from yaml import SafeDumper as yamlDumper
    printWarning("CSafeDumper not installed. Fallback to SafeDumper.")

try:
    import msgpack
except ImportError:
    print("Message pack python library not detected. Must use YAML backend instead.")



###################
# Writing functions
###################
def write(filename_noExt, data, format="yaml"):
    """Writes data to file with specified format; extension is appended based on format."""
    if format == "yaml":
        writeYAML(filename_noExt + ".yaml", data)
    elif format == "json":
        writeJson(filename_noExt + ".json", data)
    elif format == "msgpack":
        writeMsgPack(filename_noExt + ".dat", data)
    else:
        printExit("Unrecognized write format {}".format(format))


def writeYAML(filename, data, **kwargs):
    """Writes data to file in YAML format."""
    # set default kwags for yaml dump
    if "explicit_start" not in kwargs:
        kwargs["explicit_start"] = True
    if "explicit_end" not in kwargs:
        kwargs["explicit_end"] = True
    if "default_flow_style" not in kwargs:
        kwargs["default_flow_style"] = None

    with open(filename, "w") as f:
        yaml.dump(data, f, Dumper=yamlDumper, **kwargs)

def writeJson(filename, data):
    """Writes data to file in json format."""
    with open(filename, "w") as f:
        json_object = json.dumps(data, option=json.OPT_INDENT_2).decode("utf-8") if 'orjson' in sys.modules else json.dumps(data, indent=2)
        f.write(json_object)

def writeMsgPack(filename, data):
    """Writes data to file in Message Pack format."""
    with open(filename, "wb") as f:
        msgpack.pack(data, f)

def _writeSolutionsHeader(f: IO[str], problemSizes: Optional[ProblemSizes], biasTypeArgs: Optional[BiasTypeArgs], activationArgs: Optional[ActivationArgs]) -> None:
    """Write the YAML header (version, problem sizes, bias/activation args)."""
    f.write("- MinimumRequiredVersion: {}\n".format(__version__))
    f.write("- ProblemSizes:\n")
    if problemSizes:
        for sizeRange in problemSizes.ranges:
            f.write("  - Range: {}\n".format(sizeRange))
        for problemExact in problemSizes.exacts:
            #FIXME-problem, this ignores strides:
            f.write("  - Exact: {}\n".format(problemExact))
    if biasTypeArgs:
        f.write("- BiasTypeArgs: [{}]\n".format([btype.value for btype in biasTypeArgs.biasTypes]))
    if activationArgs:
        f.write("- ActivationArgs:\n")
        for setting in activationArgs.settingList:
            f.write("  - [Enum: %s]\n"%(setting.activationEnum))

def _findBodyOffset(filename: str, headerKeys: set[str]) -> int:
    """Find the character offset where solution entries begin, skipping the header."""
    with open(filename, "r") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                return pos
            if line.startswith("- "):
                key = line[2:].split(":")[0].strip()
                if key not in headerKeys:
                    return pos

def writeSolutions(filename: str, problemSizes: Optional[ProblemSizes], biasTypeArgs: Optional[BiasTypeArgs], activationArgs: Optional[ActivationArgs], solutions: list, cache: bool = False) -> None:
    """Writes solution YAML file."""

    if cache:
        # Solutions unchanged; rewrite only the header in place
        with timing_context("python_wsol_prepare"):
            with timing_context("python_wsol_prepare_cache"):
                newHeader = io.StringIO()
                _writeSolutionsHeader(newHeader, problemSizes, biasTypeArgs, activationArgs)
                newHeader = newHeader.getvalue()
                headerKeys = {line[2:].split(":")[0].strip()
                              for line in newHeader.splitlines() if line.startswith("- ")}
                oldBodyOffset = _findBodyOffset(filename, headerKeys)
        with timing_context("python_wsol_header"):
            if len(newHeader) == oldBodyOffset:
                # Same size header; overwrite in place, body untouched
                with open(filename, "r+") as f:
                    f.write(newHeader)
            else:
                # Header size changed; must shift the body
                with open(filename, "r+") as f:
                    f.seek(oldBodyOffset)
                    solutionsBody = f.read()
                    f.seek(0)
                    f.write(newHeader)
                    f.write(solutionsBody)
                    f.truncate()
        return

    with timing_context("python_wsol_prepare"):
        with timing_context("python_wsol_prepare_nocache"):
            solutionStates: list[dict] = []
            for solution in solutions:
                solutionState = solution.getAttributes()
                solutionState["ProblemType"] = solutionState["ProblemType"].state
                problemTypeToEnum(solutionState["ProblemType"])
                isa = solutionState["ISA"]
                solutionState["ISA"] = [isa[0], isa[1], isa[2]]
                solutionStates.append(solutionState)
    with open(filename, "w") as f:
        with timing_context("python_wsol_header"):
            _writeSolutionsHeader(f, problemSizes, biasTypeArgs, activationArgs)
        with timing_context("python_wsol_dump"):
            yaml.dump(solutionStates, f, Dumper=yamlDumper, default_flow_style=None)


###############################
# Reading and parsing functions
###############################
def read(filename, customizedLoader=False):
    name, extension = os.path.splitext(filename)
    if extension == ".yaml":
        return load_yaml_stream(filename, yamlLoader) if customizedLoader else readYAML(filename)
    if extension == ".json":
        return readJson(filename)
    else:
        printExit("Unrecognized read format {}".format(extension))


def readYAML(filename):
    """Reads and returns YAML data from file."""
    with open(filename, "r") as f:
        data = yaml.load(f, yamlLoader)
    return data


def readJson(filename):
    """Reads and returns JSON data from file."""
    with open(filename, "r") as f:
        data = json.loads(f.read())
    return data


def parseSolutionsFile(
        filename,
        assembler,
        splitGSU: bool,
        printSolutionRejectionReason: bool,
        printIndexAssignmentInfo: bool,
        isaInfoMap
    ):
    """Wrapper function to read and parse a solutions file."""
    return parseSolutionsData(
               read(filename),
               filename,
               assembler,
               splitGSU,
               printSolutionRejectionReason,
               printIndexAssignmentInfo,
               isaInfoMap
            )


def parseSolutionsData(
        data,
        srcFile,
        assembler,
        splitGSU: bool,
        printSolutionRejectionReason: bool,
        printIndexAssignmentInfo: bool,
        isaInfoMap
    ):
    """Parses problem sizes and solutions from the data of a solutions file."""
    if len(data) < 3:
        printExit("Solution file {} is missing required fields (len = {} < 3" \
                .format(srcFile, len(data)))

    versionString = data[0]["MinimumRequiredVersion"]
    if not versionIsCompatible(versionString):
        printWarning("Version = {} in solution file {} does not match Tensile version = {}" \
                .format(srcFile, versionString, __version__) )

    if "ProblemSizes" not in data[1]:
        printExit("Solution file {} doesn't begin with ProblemSizes".format(srcFile))

    problemSizesConfig = data[1]["ProblemSizes"]
    solutionStartIdxInData = 2
    if (len(data) > solutionStartIdxInData) and "BiasTypeArgs" in data[solutionStartIdxInData]:
        solutionStartIdxInData += 1
    if (len(data) > solutionStartIdxInData) and "ActivationArgs" in data[solutionStartIdxInData]:
        solutionStartIdxInData += 1

    solutions = []
    for i in range(solutionStartIdxInData, len(data)):
        solutionState = data[i]
        # force redo the deriving of parameters, make sure old version logic yamls can be validated
        solutionState["AssignedProblemIndependentDerivedParameters"] = False
        solutionState["AssignedDerivedParameters"] = False
        solutionObject = Solution(
                             solutionState,
                             splitGSU,
                             printSolutionRejectionReason,
                             printIndexAssignmentInfo,
                             assembler,
                             isaInfoMap,
                             srcFile
                         )
        solutions.append(solutionObject)
    problemType = solutions[0]["ProblemType"]
    problemSizes = ProblemSizes(problemType, problemSizesConfig)
    return (problemSizes, solutions)

def getRealDataTypeA(dataType):
    if dataType == DataTypeEnum.Float8BFloat8.value:
        return DataTypeEnum.Float8.value
    elif dataType == DataTypeEnum.BFloat8Float8.value:
        return DataTypeEnum.BFloat8.value
    elif dataType == DataTypeEnum.Float8BFloat8_fnuz.value:
        return DataTypeEnum.Float8_fnuz.value
    elif dataType == DataTypeEnum.BFloat8Float8_fnuz.value:
        return DataTypeEnum.BFloat8_fnuz.value
    else:
        return dataType

def getRealDataTypeB(dataType):
    if dataType == DataTypeEnum.Float8BFloat8.value:
        return DataTypeEnum.BFloat8.value
    elif dataType == DataTypeEnum.BFloat8Float8.value:
        return DataTypeEnum.Float8.value
    elif dataType == DataTypeEnum.Float8BFloat8_fnuz.value:
        return DataTypeEnum.BFloat8_fnuz.value
    elif dataType == DataTypeEnum.BFloat8Float8_fnuz.value:
        return DataTypeEnum.Float8_fnuz.value
    else:
        return dataType
    
class LibraryLogic(NamedTuple):
    """Return tuple for parseLibraryLogicData()"""
    schedule: str
    architecture: str
    problemType: ProblemType
    solutions: list
    exactLogic: list
    library: SolutionLibrary.MasterSolutionLibrary
    typeMismatches: dict = {}

def parseLibraryLogicFile(
        filename,
        assembler,
        splitGSU: bool,
        printSolutionRejectionReason: bool,
        printIndexAssignmentInfo: bool,
        isaInfoMap: Dict[str, IsaInfo],
        lazyLibraryLoading: bool
    ):
    """Wrapper function to read and parse a library logic file."""
    return parseLibraryLogicData(
               read(filename, True),
               filename,
               assembler,
               splitGSU,
               printSolutionRejectionReason,
               printIndexAssignmentInfo,
               isaInfoMap,
               lazyLibraryLoading
           )


def parseLibraryLogicData(
        data,
        srcFile,
        assembler,
        splitGSU: bool,
        printSolutionRejectionReason: bool,
        printIndexAssignmentInfo: bool,
        isaInfoMap: Dict[str, IsaInfo],
        lazyLibraryLoading: bool
    ):
    """Parses the data of a library logic file."""
    if isinstance(data, List):
        data = parseLibraryLogicList(data, srcFile)

    if "CUCount" not in data:
        data["CUCount"] = None
    if 'MacDataTypeA' not in data["ProblemType"]: #it will either be set as d['MacDataType'] or a specified input
        data["ProblemType"]['MacDataTypeA'] = getRealDataTypeA(data["ProblemType"]['DataType'])

    if 'MacDataTypeB' not in data["ProblemType"]:
        data["ProblemType"]['MacDataTypeB'] = getRealDataTypeB(data["ProblemType"]['DataType'])

    if 'DataTypeA' not in data["ProblemType"]:
        data["ProblemType"]['DataTypeA'] = data["ProblemType"]['MacDataTypeA']
    else:
        data["ProblemType"]['DataTypeA'] = getRealDataTypeA(data["ProblemType"]['DataTypeA'])

    if 'DataTypeB' not in data["ProblemType"]:
        data["ProblemType"]['DataTypeB'] = data["ProblemType"]['MacDataTypeB']
    else:
        data["ProblemType"]['DataTypeB'] = getRealDataTypeB(data["ProblemType"]['DataTypeB'])

    if not versionIsCompatible(data["MinimumRequiredVersion"]):
        printWarning("Version = {} in library logic file {} does not match Tensile version = {}" \
                .format(srcFile, data["MinimumRequiredVersion"], __version__) )

    # unpack problemType
    problemType = ProblemType(data["ProblemType"], printIndexAssignmentInfo)

    # unpack solution
    def solutionStateToSolution(solutionState, assembler, isaInfoMap) -> Optional[Solution]:
        if solutionState["KernelLanguage"] == "Assembly":
            solutionState["ISA"] = gfxToIsa(data["ArchitectureName"])
        solutionState["CUCount"] = data["CUCount"]
        solutionState["DeviceNames"] = data.get("DeviceNames", None)
        # force redo the deriving of parameters, make sure old version logic yamls can be validated
        solutionState["AssignedProblemIndependentDerivedParameters"] = False
        solutionState["AssignedDerivedParameters"] = False
        customKernelName = None
        if "CustomKernel" in solutionState and solutionState["CustomKernel"]["name"] \
                and not solutionState["CustomKernel"].get("generated", False):
            customKernelName = solutionState["CustomKernel"]["name"]
        elif solutionState.get("CustomKernelName", ""):
            customKernelName = solutionState["CustomKernelName"]

        if customKernelName:
            isp = {}
            if "InternalSupportParams" in solutionState:
                isp = solutionState["InternalSupportParams"]
            try:
                customConfig = getCustomKernelConfig(customKernelName, isp)
            except (RuntimeError, KeyError, TypeError) as e:
                printWarning(f"Skipping custom kernel '{customKernelName}': "
                             f"missing or invalid custom.config ({e})")
                return None
            for key, value in customConfig.items():
                solutionState[key] = value

            if "MatrixInstruction" in customConfig and len(customConfig["MatrixInstruction"]) != 4:
                raise ValueError(f"Custom kernel MatrixInstruction can only be of length 4, found {customConfig['MatrixInstruction']}")
        # overwrite problemType if any
        solutionState["ProblemType"] = problemType
        if 'MacDataTypeA' not in solutionState["ProblemType"]: #it will either be set as d['MacDataType'] or a specified input
            solutionState["ProblemType"]['MacDataTypeA'] = getRealDataTypeA(solutionState["ProblemType"]['DataType'])

        if 'MacDataTypeB' not in solutionState["ProblemType"]:
            solutionState["ProblemType"]['MacDataTypeB'] = getRealDataTypeB(solutionState["ProblemType"]['DataType'])

        if 'DataTypeA' not in solutionState["ProblemType"]:
            solutionState["ProblemType"]['DataTypeA'] = solutionState["ProblemType"]['MacDataTypeA']
        else:
            solutionState["ProblemType"]['DataTypeA'] = getRealDataTypeA(solutionState["ProblemType"]['DataTypeA'])

        if 'DataTypeB' not in solutionState["ProblemType"]:
            solutionState["ProblemType"]['DataTypeB'] = solutionState["ProblemType"]['MacDataTypeB']
        else:
            solutionState["ProblemType"]['DataTypeB'] = getRealDataTypeB(solutionState["ProblemType"]['DataTypeB'])

        solutionObject = Solution(
                             solutionState,
                             splitGSU,
                             printSolutionRejectionReason,
                             printIndexAssignmentInfo,
                             assembler,
                             isaInfoMap,
                             srcFile
                         )
        return solutionObject

    resetTypeMismatchCollector()
    allSolutions = [solutionStateToSolution(solutionState, assembler, isaInfoMap) for solutionState in data["Solutions"]]
    skipped = sum(1 for s in allSolutions if s is None)
    if skipped:
        printWarning(f"Skipped {skipped} solution(s) due to missing or invalid custom.config")
    solutions = [s for s in allSolutions if s is not None]
    typeMismatches = getTypeMismatchCollector()

    newLibrary, _ = SolutionLibrary.MasterSolutionLibrary.FromOriginalState(
        data,
        solutions,
        splitGSU,
        printSolutionRejectionReason,
        printIndexAssignmentInfo,
        assembler,
        isaInfoMap,
        lazyLibraryLoading,
        logicFile=srcFile
    )

    return LibraryLogic(data["ScheduleName"], data["ArchitectureName"], problemType, solutions, \
            data.get("ExactLogic"), newLibrary, typeMismatches)


def parseLibraryLogicList(data, srcFile="?"):
    """Parses the data of a matching table style library logic file."""
    if len(data) < 9:
        printExit("Library logic file {} is missing required fields (len = {} < 9)" \
                .format(srcFile, len(data)))

    rv = {}
    rv["MinimumRequiredVersion"] = data[0]["MinimumRequiredVersion"]
    rv["ScheduleName"] = data[1]
    rv["DeviceNames"] = data[3]
    rv["ProblemType"] = data[4]
    rv["Solutions"] = data[5]

    if type(data[2]) is dict:
        rv["ArchitectureName"] = data[2]["Architecture"]
        rv["CUCount"] = data[2]["CUCount"]
    else:
        rv["ArchitectureName"] = data[2]
        rv["CUCount"] = None

    # TODOBEN: figure out what to do with these...
    rv["ExactLogic"] = data[7]
    rv["RangeLogic"] = data[8]

    # optional fields
    if len(data) > 10 and data[10]:
        rv["PerfMetric"] = data[10]

    # library logic fields
    libraryType = None
    if len(data) > 11 and data[11]:
        libraryType = data[11]
    else:
        printExit("Library logic file {} is missing required field matching property." \
                .format(srcFile))
    if libraryType == "FreeSize":
        rv["LibraryType"] = "FreeSize"
        rv["Library"] = {}
        rv["Library"]["indexOrder"] = None
        rv["Library"]["table"] = [0, len(data[5])]
        rv["Library"]["distance"] = None
    elif libraryType == "Prediction":
        rv["LibraryType"] = "Prediction"
        rv["Library"] = {}
        rv["Library"]["indexOrder"] = None
        rv["Library"]["table"] = [0, len(data[5])]
        rv["Library"]["distance"] = None
    else:
        rv["LibraryType"] = "Matching"
        rv["Library"] = {}
        rv["Library"]["indexOrder"] = data[6]
        rv["Library"]["table"] = data[7]
        rv["Library"]["distance"] = libraryType

    return rv


def rawLibraryLogic(data):
    """Returns a tuple of the data in a library logic file."""
    versionString = data[0]
    scheduleName = data[1]
    architectureName = data[2]
    deviceNames = data[3]
    problemTypeState = data[4]
    solutionStates = data[5]
    indexOrder = data[6]
    exactLogic = data[7]
    rangeLogic = data[8]
    otherFields = []

    dataLength = len(data)
    if dataLength > 9:
        for idx in range(9, dataLength):
            otherFields.append(data[idx])

    return (versionString, scheduleName, architectureName, deviceNames,\
            problemTypeState, solutionStates, indexOrder, exactLogic, rangeLogic, otherFields)


#################
# Other functions
#################
def getCUCount() -> int:
    """Return the number of CU Count in current Hardware."""
    CU = os.environ.get("CU", None)
    if CU is None:
        try:
            res = subprocess.run("rocminfo | grep Compute", stdout=subprocess.PIPE, shell=True, env={**os.environ, "ROCR_VISIBLE_DEVICES": "0"})
            CU_RE = r"Compute Unit:(?P<COMPUTE_UNIT>[\w ]+)"
            lines = res.stdout.decode("utf-8").strip().split('\n')
            if lines:
                match = re.search(CU_RE, lines[-1])
                if match:
                    CU = int(match.group('COMPUTE_UNIT').strip())
        except Exception:
            pass

    if CU is None:
        printExit("Failed to get Compute Unit count from rocminfo or env variable 'CU'")

    return int(CU)

def createLibraryLogic(schedulePrefix, architectureName, deviceNames, libraryType, logicTuple):
    """Creates the data for a library logic file suitable for writing to YAML."""
    problemType = logicTuple[0]
    solutions = logicTuple[1]
    indexOrder = logicTuple[2]
    exactLogic = logicTuple[3]
    rangeLogic = logicTuple[4]

    tileSelection = False
    if len(logicTuple) > 5 and logicTuple[5]:
        tileSelection = True

    data = []
    # Tensile version
    data.append({"MinimumRequiredVersion": __version__})
    # schedule name
    data.append(schedulePrefix)  # change from Tensile to vega10
    # schedule architecture name and get CU count
    CUCount=getCUCount()
    data.append({"Architecture": architectureName, "CUCount": CUCount} if architectureName=="gfx942" and CUCount and CUCount!=304 else architectureName)
    # schedule device names
    data.append(deviceNames)
    # problem type
    problemTypeState = problemType.state
    problemTypeState["DataType"] = \
            problemTypeState["DataType"].value
    problemTypeState["MacDataTypeA"] = \
            problemTypeState["MacDataTypeA"].value
    problemTypeState["MacDataTypeB"] = \
            problemTypeState["MacDataTypeB"].value
    problemTypeState["DataTypeA"] = \
            problemTypeState["DataTypeA"].value
    problemTypeState["DataTypeB"] = \
            problemTypeState["DataTypeB"].value
    problemTypeState["DataTypeE"] = \
            problemTypeState["DataTypeE"].value
    problemTypeState["DataTypeAmaxD"] = \
            problemTypeState["DataTypeAmaxD"].value
    problemTypeState["DestDataType"] = \
            problemTypeState["DestDataType"].value
    problemTypeState["ComputeDataType"] = \
            problemTypeState["ComputeDataType"].value
    problemTypeState["BiasDataTypeList"] = \
            [btype.value for btype in problemTypeState["BiasDataTypeList"]]
    problemTypeState["ActivationComputeDataType"] = \
            problemTypeState["ActivationComputeDataType"].value
    problemTypeState["ActivationType"] = \
            problemTypeState["ActivationType"].value
    problemTypeState["F32XdlMathOp"] = \
            problemTypeState["F32XdlMathOp"].value
    if "DataTypeMetadata" in problemTypeState:
        problemTypeState["DataTypeMetadata"] = \
                problemTypeState["DataTypeMetadata"].value
    if "DataTypeMXSA" in problemTypeState:
        problemTypeState["DataTypeMXSA"] = \
                problemTypeState["DataTypeMXSA"].value
    if "DataTypeMXSB" in problemTypeState:
        problemTypeState["DataTypeMXSB"] = \
                problemTypeState["DataTypeMXSB"].value
    data.append(problemTypeState)
    # solutions
    solutionList = []
    for solution in solutions:
        solutionState = solution.getAttributes()
        del solutionState["ProblemType"]
        isa = solutionState["ISA"]
        solutionState["ISA"] = [isa[0], isa[1], isa[2]]
        solutionList.append(solutionState)

    if tileSelection:
        tileSolutions = logicTuple[5]
        for solution in tileSolutions:
            solutionState = solution.getAttributes()
            del solutionState["ProblemType"]
            solutionList.append(solutionState)

    data.append(solutionList)
    # index order
    data.append(indexOrder)

    # exactLogic
    exactLogicList = []
    if exactLogic:
        for key in exactLogic:
            exactLogicList.append([list(key), exactLogic[key]])
        data.append(exactLogicList)
    else:
        data.append(None)

    # rangeLogic
    data.append(rangeLogic)

    if tileSelection:
        tileSelectionLogic = {}
        tileSelectionIndices = logicTuple[6]
        tileSelectionLogic["TileSelectionIndices"] = tileSelectionIndices
        data.append(tileSelectionLogic)
    else:
        data.append(None)

    data.append(logicTuple[7]) # PerfMetric
    data.append(libraryType) # LibraryType
    return data
