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

from . import ClientWriter
from . import LibraryIO
from .Contractions import ProblemType as ContractionsProblemType
from Tensile.SolutionStructs.Problem import ProblemType, ProblemSizes
from .Common import globalParameters, print1, printExit, printWarning, assignGlobalParameters, \
        restoreDefaultGlobalParameters, HR, __version__
from .Tensile import addCommonArguments, argUpdatedGlobalParameters

import argparse
import os
import sys


def getGlobalParams(config):
    """Try to parse out global parameters from the places it could be"""
    globalParams = None
    try:
        globalParams = config["GlobalParameters"]
    except (TypeError, LookupError):
        pass

    return globalParams


def getProblemDict(config):
    """Try to parse out a problem type dict from the places it could be"""
    problemDict = None
    try:  # Tensile Config file (first entry)
        problemDict = config["BenchmarkProblems"][0][0]
        if len(config["BenchmarkProblems"]) > 1:
            printWarning("More than one BenchmarkProblem in config file: only using first")
    except (TypeError, LookupError):
        pass
    else:
        return problemDict

    try:  # Solution Selection "base" file
        problemDict = config["ProblemType"]
    except (TypeError, LookupError):
        pass

    return problemDict


def getSizeList(config):
    """
    Try to parse out a size list from the places it could be
    (including the whole config being the list)
    """
    sizeList = None
    try:  # Tensile config file (first entry)
        sizeList = config["BenchmarkProblems"][0][1]["BenchmarkFinalParameters"][0]["ProblemSizes"]
    except (TypeError, LookupError):
        pass
    else:
        return sizeList

    # check if whole config looks like a size list
    keep = True
    if type(config) is list:
        for i in config:
            if not (type(i) is dict and ("Exact" in i or "Range" in i)):
                keep = False
                break
    else:
        keep = False

    if keep:
        sizeList = config
    return sizeList


def parseConfig(config):
    """Parse out all data we can get from the config"""
    globalParams = getGlobalParams(config)
    problemDict = getProblemDict(config)
    sizeList = getSizeList(config)

    return (globalParams, problemDict, sizeList)


def TensileClientConfig(userArgs):
    print1("")
    print1(HR)
    print1("#")
    print1("#  Tensile Client Config v{}".format(__version__))
    print1("#")
    print1(HR)
    print1("")

    # argument parsing
    # yapf: disable
    argParser = argparse.ArgumentParser()
    argParser.add_argument("ConfigYaml", type=os.path.realpath, nargs="+",
            help="Config yaml(s) containing parameters and problem and size information")
    argParser.add_argument("--output-config", "-o", dest="OutputConfig", type=os.path.realpath,
            required=True, help="Path to output resulting client config file")
    argParser.add_argument("--merge-sizes", dest="MergeSizes", action="store_true",
            help="Allow sizes from multiple config files")
    # yapf: enable

    addCommonArguments(argParser)
    args = argParser.parse_args(userArgs)

    # loop through all configs and parse out all the data we can
    configs = [LibraryIO.read(x) for x in args.ConfigYaml]

    globalParams = {}
    problemDict = None
    sizeList = None

    for config in configs:
        (myGlobalParams, myProblemDict, mySizeList) = parseConfig(config)

        # if we got data from the config, keep it
        # if we already had that data, warn and exit
        if myGlobalParams is not None:
            if globalParams == {} or globalParams == myGlobalParams:
                globalParams = myGlobalParams
            else:
                printExit("Multiple definitions for GlobalParameters found:\n{}\nand\n{}".format(
                    globalParams, myGlobalParams))

        if myProblemDict is not None:
            if problemDict is None or problemDict == myProblemDict:
                problemDict = myProblemDict
            else:
                printExit("Multiple definitions for ProblemType found:\n{}\nand\n{}".format(
                    problemDict, myProblemDict))

        if mySizeList is not None:
            if sizeList is None or sizeList == mySizeList:
                sizeList = mySizeList
            elif args.MergeSizes:
                sizeList += mySizeList
            else:
                printExit("Multiple size lists found:\n{}\nand\n{}\n"
                          "Run with --merge-sizes to keep all size lists found".format(
                              sizeList, mySizeList))

    if problemDict is None:
        printExit("No ProblemType found; cannot produce output")
    if sizeList is None:
        printExit("No SizeList found; cannot produce output")

    ssProblemType = ProblemType(problemDict)
    conProblemType = ContractionsProblemType.FromOriginalState(ssProblemType)
    sizes = ProblemSizes(ssProblemType, sizeList)  # TODO doesn't seem to work for range sizes

    # update globals
    restoreDefaultGlobalParameters()
    assignGlobalParameters(globalParams)

    overrideParameters = argUpdatedGlobalParameters(args)
    for key, value in overrideParameters.items():
        print1("Overriding {0}={1}".format(key, value))
        globalParameters[key] = value

    # write output
    ClientWriter.writeClientConfigIni(True, sizes, "", "", "", "", conProblemType, "", [], "", args.OutputConfig, None)


def main():
    TensileClientConfig(sys.argv[1:])
