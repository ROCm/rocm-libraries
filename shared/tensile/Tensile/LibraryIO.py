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

from .Common import printExit, printWarning, versionIsCompatible
from .SolutionStructs import Solution, ProblemSizes, ProblemType
from . import __version__
from . import Common
from . import SolutionLibrary
from .Utilities.ConditionalImports import yamlLoader, yamlDumper

from typing import NamedTuple, Optional

import os

try:
    import yaml
except ImportError:
    printExit(
        "You must install PyYAML to use Tensile (to parse config files). See http://pyyaml.org/wiki/PyYAML for installation instructions."
    )

try:
    import msgpack
except ImportError:
    print("Message pack python library not detected. Must use YAML backend instead.")


def updateProblemDatatypes(problemType):
    problemType["DataType"] = problemType["DataType"].value
    problemType["DestDataType"] = problemType["DestDataType"].value
    problemType["ComputeDataType"] = problemType["ComputeDataType"].value
    problemType["F32XdlMathOp"] = problemType["F32XdlMathOp"].value


###################
# Writing functions
###################
def write(filename_noExt, data, format="yaml"):
    """Writes data to file with specified format; extension is appended based on format."""
    if format == "yaml":
        writeYAML(filename_noExt + ".yaml", data)
    elif format == "msgpack":
        writeMsgPack(filename_noExt + ".dat", data)
    else:
        printExit("Unrecognized format {}".format(format))


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
        yaml.dump(data, f, yamlDumper, **kwargs)


def writeMsgPack(filename, data):
    """Writes data to file in Message Pack format."""
    with open(filename, "wb") as f:
        msgpack.pack(data, f)


def writeSolutions(filename, problemSizes, solutions, cache=False):
    """Writes solution YAML file."""

    # convert objects to nested dictionaries
    solutionStates = []

    if cache:
        solYaml = readYAML(filename)
        solutionStates = solYaml[2:]
    else:
        for solution in solutions:
            solutionState = solution.getAttributes()
            solutionState["ProblemType"] = solutionState["ProblemType"].state
            updateProblemDatatypes(solutionState["ProblemType"])
            solutionStates.append(solutionState)
    # write dictionaries
    with open(filename, "w") as f:
        f.write("- MinimumRequiredVersion: {}\n".format(__version__))
        f.write("- ProblemSizes:\n")
        if problemSizes:
            for sizeRange in problemSizes.ranges:
                f.write("  - Range: {}\n".format(sizeRange))
            for problemExact in problemSizes.exacts:
                #FIXME-problem, this ignores strides:
                f.write("  - Exact: {}\n".format(problemExact))

        yaml.dump(solutionStates, f, yamlDumper, default_flow_style=None)


###############################
# Reading and parsing functions
###############################
def readYAML(filename):
    """Reads and returns YAML data from file."""
    with open(filename, "r") as f:
        data = yaml.load(f, yamlLoader)
    return data


def parseSolutionsFile(filename):
    """Wrapper function to read and parse a solutions file."""
    return parseSolutionsData(readYAML(filename), filename)


def parseSolutionsData(data, srcFile="?"):
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

    solutions = []
    for i in range(2, len(data)):
        solutionState = data[i]
        # force redo the deriving of parameters, make sure old version logic yamls can be validated
        solutionState["AssignedProblemIndependentDerivedParameters"] = False
        solutionState["AssignedDerivedParameters"] = False
        solutionObject = Solution(solutionState)
        solutions.append(solutionObject)
    problemType = solutions[0]["ProblemType"]
    problemSizes = ProblemSizes(problemType, problemSizesConfig)
    return (problemSizes, solutions)


class LibraryLogic(NamedTuple):
    """Return tuple for parseLibraryLogicData()"""
    schedule: str
    architecture: str
    problemType: ProblemType
    solutions: list
    exactLogic: list
    library: SolutionLibrary.MasterSolutionLibrary


def parseLibraryLogicFile(filename):
    """Wrapper function to read and parse a library logic file."""
    return parseLibraryLogicData(readYAML(filename), filename)


def parseLibraryLogicData(data, srcFile="?"):
    """Parses the data of a library logic file."""
    if type(data) is list:
        data = parseLibraryLogicList(data, srcFile)

    if "CUCount" not in data:
        data["CUCount"] = None

    if "IsAPU" not in data:
        data["IsAPU"] = None

    if "Fp16AltImpl" not in data:
        data["Fp16AltImpl"] = False

    if "Fp16AltImplRound" not in data:
        data["Fp16AltImplRound"] = False

    if not versionIsCompatible(data["MinimumRequiredVersion"]):
        printWarning("Version = {} in library logic file {} does not match Tensile version = {}" \
                .format(srcFile, data["MinimumRequiredVersion"], __version__) )

    # unpack problemType
    problemType = ProblemType(data["ProblemType"])
    # unpack solutions
    solutions = []
    for solutionState in data["Solutions"]:
        if solutionState["ProblemType"] == "derive":
            solutionState["ProblemType"] = problemType
        if solutionState["KernelLanguage"] == "Assembly":
            solutionState["ISA"] = Common.gfxArch(data["ArchitectureName"])
        else:
            solutionState["ISA"] = (0, 0, 0)
        # force redo the deriving of parameters, make sure old version logic yamls can be validated
        solutionState["AssignedProblemIndependentDerivedParameters"] = False
        solutionState["AssignedDerivedParameters"] = False
        solutionObject = Solution(solutionState)

        if solutionObject["ProblemType"] != problemType:
            printExit("ProblemType in library logic file {} doesn't match solution: {} != {}" \
                    .format(srcFile, problemType, solutionObject["ProblemType"]))
        solutions.append(solutionObject)

    newLibrary = SolutionLibrary.MasterSolutionLibrary.FromOriginalState(data, solutions)

    return LibraryLogic(data["ScheduleName"], data["ArchitectureName"], problemType, solutions, \
            data.get("ExactLogic"), newLibrary)


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
        if "CUCount" in data[2]:
            rv["CUCount"] = data[2]["CUCount"]
        if "IsAPU" in data[2]:
            rv["IsAPU"] = data[2]["IsAPU"]
    else:
        rv["ArchitectureName"] = data[2]
        rv["CUCount"] = None
        rv["IsAPU"] = None

    rv["ExactLogic"] = data[7]
    # data[8] previously contained range logic, which has been retired

    # optional fields
    if len(data) > 10 and data[10]:
        rv["PerfMetric"] = data[10]

    if len(data) > 11 and data[11] == "Fp16AltImpl":
        rv["Fp16AltImpl"] = True

    if len(data) > 13 and data[13] == "Fp16AltImplRound":
        rv["Fp16AltImplRound"] = True

    # library logic fields
    rv["LibraryType"] = "Matching"
    rv["Library"] = {}
    rv["Library"]["indexOrder"] = data[6]
    rv["Library"]["table"] = data[7]
    rv["Library"]["distance"] = "Euclidean"
    if len(data) > 12 and data[12]:
        rv["Library"]["distance"] = data[12]

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


########################
# Library logic creation
########################
def createLibraryLogic(schedulePrefix, architectureName, deviceNames, logicTuple, dictFormat):
    if not dictFormat:
        return createLibraryLogicList(schedulePrefix, architectureName, deviceNames, logicTuple)

    rv = {}
    rv["MinimumRequiredVersion"] = __version__
    rv["ScheduleName"] = schedulePrefix
    rv["DeviceNames"] = deviceNames
    rawProblemType = logicTuple[0]
    rawSolutions = logicTuple[1]

    # architecture
    if type(architectureName) is dict:
        rv["ArchitectureName"] = architectureName["Architecture"]
        if "CUCount" in architectureName:
            rv["CUCount"] = architectureName["CUCount"]
        if "IsAPU" in architectureName:
            rv["IsAPU"] = architectureName["IsAPU"]
    else:
        rv["ArchitectureName"] = architectureName

    # problem type
    problemTypeState = rawProblemType.state
    updateProblemDatatypes(problemTypeState)
    rv["ProblemType"] = problemTypeState

    # solutions
    solutionList = []
    for solution in rawSolutions:
        solutionState = solution.getAttributes()
        solutionState["ProblemType"] = "derive"
        solutionList.append(solutionState)

    rv["Solutions"] = solutionList

    # library logic
    exactLogic = logicTuple[3]
    exactLogicList = []
    for key in exactLogic:
        exactLogicList.append([list(key), exactLogic[key]])

    rv["LibraryType"] = "Matching"
    rv["Library"] = {}
    rv["Library"]["indexOrder"] = logicTuple[2]
    rv["Library"]["table"] = exactLogicList
    rv["Library"]["distance"] = "Euclidean"
    rv["PerfMetric"] = logicTuple[7]

    return rv


def createLibraryLogicList(schedulePrefix, architectureName, deviceNames, logicTuple):
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
    data.append(architectureName)
    # schedule device names
    data.append(deviceNames)

    # problem type
    problemTypeState = problemType.state
    updateProblemDatatypes(problemTypeState)
    data.append(problemTypeState)

    # solutions
    solutionList = []
    for solution in solutions:
        solutionState = solution.getAttributes()
        solutionState["ProblemType"] = solutionState["ProblemType"].state
        updateProblemDatatypes(solutionState["ProblemType"])
        solutionList.append(solutionState)

    if tileSelection:
        tileSolutions = logicTuple[5]
        for solution in tileSolutions:
            solutionState = solution.getAttributes()
            solutionState["ProblemType"] = solutionState["ProblemType"].state
            updateProblemDatatypes(solutionState["ProblemType"])
            solutionList.append(solutionState)

    data.append(solutionList)
    # index order
    data.append(indexOrder)

    # exactLogic
    exactLogicList = []
    for key in exactLogic:
        exactLogicList.append([list(key), exactLogic[key]])
    data.append(exactLogicList)

    # rangeLogic
    data.append(rangeLogic)

    if tileSelection:
        tileSelectionLogic = {}
        tileSelectionIndices = logicTuple[6]
        tileSelectionLogic["TileSelectionIndices"] = tileSelectionIndices
        data.append(tileSelectionLogic)
    else:
        data.append(None)

    data.append(logicTuple[7])
    return data


def initAsmCapsCache(cacheFile: str) -> Optional[dict]:
    """
    If requested and exists, reads an assembler capabilities cache from disk.

    Arguments:
      cacheFile: Cache file (YAML format).

    Returns:
      newcache:  If cachFile is empty, None. Otherwise, if file exists, a dictionary containing
                 capabilities cache; if not, an empty dict
    """
    # if cacheFile str is empty, do not use cache
    if not cacheFile:
        return None

    cacheExists = os.path.exists(cacheFile)
    if not cacheExists:
        return {}

    cache  = readYAML(cacheFile)

    # gaurd against inconsistent state
    if not cache:
        return None

    toTuple = lambda s: tuple(int(i.strip()) for i in s.split(","))
    newcache = {toTuple(k): v for k,v in cache.items()}
    return newcache

def writeAsmCapsCache(cacheFile: str, data: dict):
    """
    Writes an asm cache to disk.

    Arguments:
      cacheFile: User spedified file to write the cache.
      data:      dict containg the computed cache.
    """
    newdata = {", ".join(map(str, k)): v for k, v in data.items()}
    writeYAML(cacheFile, newdata)
