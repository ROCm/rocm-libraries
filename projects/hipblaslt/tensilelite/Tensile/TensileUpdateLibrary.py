###############################################################################
#
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
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

from . import LibraryIO
from .Tensile import addCommonArguments, argUpdatedGlobalParameters

from .Common import assignGlobalParameters, print1, restoreDefaultGlobalParameters, HR, \
                    globalParameters, architectureMap, ensurePath, ParallelMap, __version__

import argparse
import copy
import itertools
import os
import sys


def UpdateLogic(filename, logicPath, outputPath):
    libYaml = LibraryIO.readYAML(filename)
    # parseLibraryLogicData mutates the original data, so make a copy
    fields = LibraryIO.parseLibraryLogicData(copy.deepcopy(libYaml), filename)
    (_, _, problemType, solutions, _, _, _) = fields

    # problem type object to state
    problemTypeState = problemType.state
    problemTypeState["DataType"] = problemTypeState["DataType"].value
    problemTypeState["DataTypeA"] = problemTypeState["DataTypeA"].value
    problemTypeState["DataTypeB"] = problemTypeState["DataTypeB"].value
    problemTypeState["DataTypeE"] = problemTypeState["DataTypeE"].value
    problemTypeState["DataTypeAmaxD"] = problemTypeState["DataTypeAmaxD"].value
    problemTypeState["DestDataType"] = problemTypeState["DestDataType"].value
    problemTypeState["ComputeDataType"] = problemTypeState["ComputeDataType"].value
    problemTypeState["BiasDataTypeList"] = [btype.value for btype in problemTypeState["BiasDataTypeList"]]
    problemTypeState["ActivationComputeDataType"] = problemTypeState["ActivationComputeDataType"].value
    problemTypeState["ActivationType"] = problemTypeState["ActivationType"].value
    problemTypeState["F32XdlMathOp"] = problemTypeState["F32XdlMathOp"].value
    if "DataTypeMetadata" in problemTypeState:
        problemTypeState["DataTypeMetadata"] = problemTypeState["DataTypeMetadata"].value

    # solution objects to state
    solutionList = []
    for solution in solutions:
        solutionState = solution.getAttributes()
        solutionState["ProblemType"] = solutionState["ProblemType"].state
        solutionState["ProblemType"]["DataType"] = \
                solutionState["ProblemType"]["DataType"].value
        solutionState["ProblemType"]["DataTypeA"] = \
                solutionState["ProblemType"]["DataTypeA"].value
        solutionState["ProblemType"]["DataTypeB"] = \
                solutionState["ProblemType"]["DataTypeB"].value
        solutionState["ProblemType"]["DataTypeE"] = \
                solutionState["ProblemType"]["DataTypeE"].value
        solutionState["ProblemType"]["DataTypeAmaxD"] = \
                solutionState["ProblemType"]["DataTypeAmaxD"].value
        solutionState["ProblemType"]["DestDataType"] = \
                solutionState["ProblemType"]["DestDataType"].value
        solutionState["ProblemType"]["ComputeDataType"] = \
                solutionState["ProblemType"]["ComputeDataType"].value
        solutionState["ProblemType"]["BiasDataTypeList"] = \
                [btype.value for btype in solutionState["ProblemType"]["BiasDataTypeList"]]
        solutionState["ProblemType"]["ActivationComputeDataType"] = \
                solutionState["ProblemType"]["ActivationComputeDataType"].value
        solutionState["ProblemType"]["ActivationType"] = \
                solutionState["ProblemType"]["ActivationType"].value
        solutionState["ProblemType"]["F32XdlMathOp"] = \
            solutionState["ProblemType"]["F32XdlMathOp"].value
        if "DataTypeMetadata" in solutionState["ProblemType"]:
            solutionState["ProblemType"]["DataTypeMetadata"] = \
                solutionState["ProblemType"]["DataTypeMetadata"].value

        solutionState["ISA"] = list(solutionState["ISA"])
        solutionList.append(solutionState)

    # update yaml
    libYaml[0] = {"MinimumRequiredVersion":__version__}
    libYaml[4] = problemTypeState
    libYaml[5] = solutionList

    if outputPath != "":
        filename = filename.replace(logicPath, outputPath)
    ensurePath(os.path.dirname(filename))
    LibraryIO.writeYAML(filename, libYaml, explicit_start=False, explicit_end=False)

def TensileUpdateLibrary(userArgs):
    print1("")
    print1(HR)
    print1("#")
    print1("#  Tensile Update Library v{}".format(__version__))

    # argument parsing and related setup
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--logic_path",  type=os.path.realpath, help="Path to LibraryLogic.yaml files.")
    argParser.add_argument("--output_path", type=os.path.realpath, default=None, help="Where to place updated logic file.")

    addCommonArguments(argParser)
    args = argParser.parse_args(userArgs)

    libPath = args.logic_path
    print1("#  Library Logic: {}".format(libPath))
    print1("#")
    print1(HR)
    print1("")

    # setup global parameters
    restoreDefaultGlobalParameters()
    assignGlobalParameters({})
    overrideParameters = argUpdatedGlobalParameters(args)
    for key, value in overrideParameters.items():
        print1("Overriding {0}={1}".format(key, value))
        globalParameters[key] = value

    # Recursive directory search
    logicArchs = set()
    for key, name in architectureMap.items():
        logicArchs.add(name)
    logicFiles = []
    for root, dirs, files in os.walk(args.logic_path):
        logicFiles += [os.path.join(root, f) for f in files
                        if os.path.splitext(f)[1]==".yaml" \
                        and (any(logicArch in os.path.splitext(f)[0] for logicArch in logicArchs) \
                        or "hip" in os.path.splitext(f)[0]) ]

    # update logic file
    outputPath = ""
    if args.output_path:
        outputPath = ensurePath(os.path.abspath(args.output_path))

    print("# LibraryLogicFiles:" % logicFiles)
    for logicFile in logicFiles:
        print("#   %s" % logicFile)
    fIter = zip(logicFiles, itertools.repeat(args.logic_path), itertools.repeat(outputPath))
    libraries = ParallelMap(UpdateLogic, fIter, "Updating logic files", method=lambda x: x.starmap)


def main():
    TensileUpdateLibrary(sys.argv[1:])
