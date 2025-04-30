################################################################################
#
# Copyright (C) 2016-2022 Advanced Micro Devices, Inc. All rights reserved.
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

from . import Common
from . import LibraryIO
from .Common import assignGlobalParameters, tPrint, restoreDefaultGlobalParameters, HR
from .Tensile import addCommonArguments, argUpdatedGlobalParameters
from . import __version__

import argparse
import copy
import os
import sys


def TensileUpdateLibrary(userArgs):
    tPrint(1, "")
    tPrint(1, HR)
    tPrint(1, "#")
    tPrint(1, "#  Tensile Update Library v{}".format(__version__))

    # argument parsing and related setup
    argParser = argparse.ArgumentParser()
    argParser.add_argument("LogicFile", type=os.path.realpath,
                           help="Library logic file to update")
    argParser.add_argument("OutputPath", type=os.path.realpath,
                           help="Where to place updated logic file")

    addCommonArguments(argParser)
    args = argParser.parse_args(userArgs)

    libPath = args.LogicFile
    tPrint(1, "#  Library Logic: {}".format(libPath))
    tPrint(1, "#")
    tPrint(1, HR)
    tPrint(1, "")

    # setup global parameters
    restoreDefaultGlobalParameters()
    assignGlobalParameters({})
    overrideParameters = argUpdatedGlobalParameters(args)
    for key, value in overrideParameters.items():
        tPrint(1, "Overriding {0}={1}".format(key, value))
        Common.globalParameters[key] = value

    # update logic file
    outPath = Common.ensurePath(os.path.abspath(args.OutputPath))
    filename = os.path.basename(libPath)
    outFile = os.path.join(outPath, filename)

    libYaml = LibraryIO.readYAML(libPath)
    # parseLibraryLogicData mutates the original data, so make a copy
    fields = LibraryIO.parseLibraryLogicData(copy.deepcopy(libYaml), libPath)
    (_, _, problemType, solutions, _, _) = fields

    # problem type object to state
    problemTypeState = problemType.state
    problemTypeState["DataType"] = problemTypeState["DataType"].value
    problemTypeState["DestDataType"] = problemTypeState["DestDataType"].value
    problemTypeState["ComputeDataType"] = problemTypeState["ComputeDataType"].value

    # solution objects to state
    solutionList = []
    for solution in solutions:
        solutionState = solution.getAttributes()
        solutionState["ProblemType"] = solutionState["ProblemType"].state
        solutionState["ProblemType"]["DataType"] = \
                solutionState["ProblemType"]["DataType"].value
        solutionState["ProblemType"]["DestDataType"] = \
                solutionState["ProblemType"]["DestDataType"].value
        solutionState["ProblemType"]["ComputeDataType"] = \
                solutionState["ProblemType"]["ComputeDataType"].value

        solutionState["ISA"] = list(solutionState["ISA"])
        solutionList.append(solutionState)

    # update yaml
    libYaml[0] = {"MinimumRequiredVersion":__version__}
    libYaml[4] = problemTypeState
    libYaml[5] = solutionList
    LibraryIO.writeYAML(outFile, libYaml, explicit_start=False, explicit_end=False)


def main():
    TensileUpdateLibrary(sys.argv[1:])
