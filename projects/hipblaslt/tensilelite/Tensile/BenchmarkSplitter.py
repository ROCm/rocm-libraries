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

import os
import copy
import yaml
import math

class BenchmarkSplitter(object):

    """
    Benchmark splitter class
    Loads in a benchmark yaml file and splits
    it into several smaller benchmarks limited
    by a number of problem sizes entries.
    """

    @staticmethod
    def __readConfigFile(benchmarkConfigFile):
        with open(benchmarkConfigFile) as f:
            data = yaml.safe_load(f)
        return data

    # data: a loaded .yaml file
    # returns: a list of yaml files that
    # are differentiated by the benchmark problem
    @staticmethod
    def __splitByProblem(data):
        rv = []
        problemKey = "BenchmarkProblems"
        for i in range(len(data[problemKey])):
            result = {}
            for k in data.keys():
                if k == problemKey:
                    result[k] = [copy.deepcopy(data[k][i])]
                else:
                    result[k] = copy.deepcopy(data[k])
            rv.append(result)
        return rv

    # data: a loaded .yaml file, containing one benchmark problem section
    # returns: a list of yaml files that are differentiated by the
    # benchmark groupings
    @staticmethod
    def __splitByBenchmarkGroup(data):
        rv = []
        problemKey = "BenchmarkProblems"

        assert len(data[problemKey]) == 1, "Config file must have one BenchmarkProblems group"

        benchmarkProblems = data[problemKey][0]

        # Find the index of the problem type group
        problemIdx = -1
        for i in range(len(benchmarkProblems)):
            if("OperationType" in benchmarkProblems[i].keys()):
                problemIdx = i
                break

        assert problemIdx != -1, "Could not find problem type group"

        # Split files on the benchmark group
        for i in range(len(benchmarkProblems)):
            if i != problemIdx:
                result = {}
                for k in data.keys():
                    if k == problemKey:
                        # Take only the problem group and one benchmarkgroup
                        result[k] = [[copy.deepcopy(benchmarkProblems[problemIdx]), \
                                    copy.deepcopy(benchmarkProblems[i])]]
                    else:
                        # copy other sections verbatim
                        result[k] = copy.deepcopy(data[k])
                rv.append(result)
        return rv

    # data: a loaded .yaml file, containing one benchmark problem section
    # and one benchmark group section
    # returns: a list of yaml files that are differentiated by the
    # benchmark sizes
    @staticmethod
    def __splitByBenchmarkSizes(data, numSizes=1):
        rv = []
        problemKey = "BenchmarkProblems"

        assert len(data[problemKey]) == 1, "Config file must have one BenchmarkProblems group"

        benchmarkProblems = data[problemKey][0]

        # Find the index of the problem type group
        # and the benchmark group
        problemIdx = -1
        benchmarkIdx = -1
        for i in range(len(benchmarkProblems)):
            groupKeys = benchmarkProblems[i].keys()
            if("OperationType" in groupKeys):
                problemIdx = i
            elif("BenchmarkFinalParameters" in groupKeys):
                benchmarkIdx = i

        assert len(benchmarkProblems) == 2 \
            and problemIdx is not -1 \
            and benchmarkIdx is not -1, \
            "Config file must have one ProblemType group and one Benchmark group"

        # Grab the problem sizes from the Benchmark group
        benchmarkGroup = benchmarkProblems[benchmarkIdx]
        assert "ProblemSizes" in benchmarkGroup["BenchmarkFinalParameters"][0] \
                and len(benchmarkGroup["BenchmarkFinalParameters"][0]["ProblemSizes"]), \
                "Benchmark group must have non-empty ProblemSizes"

        problemSizesGroup = benchmarkGroup["BenchmarkFinalParameters"][0]["ProblemSizes"]
        problemSizesCount = len(problemSizesGroup)

        numFiles = math.ceil(problemSizesCount / numSizes)

        # Split files on the benchmark sizes
        for i in range(numFiles):
            result = {}
            for k in data.keys():
                if k == problemKey:

                    # Create a new benchmark group that has all the old keys but split the sizes.
                    newBenchmarkGroup = {}
                    for bk in benchmarkGroup.keys():
                        if bk == "BenchmarkFinalParameters":
                            newBenchmarkGroup[bk] = [ {"ProblemSizes": [] } ]
                            for j in range(i*numSizes, min((i+1)*numSizes, problemSizesCount)):
                                newBenchmarkGroup[bk][0]["ProblemSizes"].append(copy.deepcopy(problemSizesGroup[j]))
                        else:
                            newBenchmarkGroup[bk] = copy.deepcopy(benchmarkGroup[bk])

                    result[k] = [[copy.deepcopy(benchmarkProblems[problemIdx]), copy.deepcopy(newBenchmarkGroup)]]
                else:
                    result[k] = copy.deepcopy(data[k])
            rv.append(result)
        return rv

    # filePath: Name of the file (can be a path)
    # suffix: Add a suffix of _## to fileName by default
    # formatting: How the stuffix string is to be treated.
    # If more indices are needed, you can increase them on the formatting.
    @staticmethod
    def __appendFileNameSuffix(filePath, suffix, separator="_", formatting="{:02}"):
        root, ext = os.path.splitext(filePath)
        suffixString = (separator + formatting).format(suffix)
        return root + suffixString + ext

    @staticmethod
    def splitBenchmarkBySizes(configFile, outputDir, numSizes=1, baseFileName="", separator="_", suffixFormat="{:02}"):

        # Use the configFile as base name if none provided
        if baseFileName == "":
            baseFileName = os.path.basename(configFile)

        # If we want to split by sizes, then we must first split files
        # into separate problems, then split those files into separate
        # benchmark groups, then split those files into sizes
        data = BenchmarkSplitter.__readConfigFile(configFile)
        benchmarksByProblem = BenchmarkSplitter.__splitByProblem(data)
        benchmarksByGroup = []
        benchmarksBySize = []
        for problem in benchmarksByProblem:
            benchmarksByGroup += BenchmarkSplitter.__splitByBenchmarkGroup(problem)
        for group in benchmarksByGroup:
            benchmarksBySize += BenchmarkSplitter.__splitByBenchmarkSizes(group, numSizes)

        # outputDir/basefileName_XX.ext
        outputFileBase = os.path.join(outputDir, baseFileName)
        for i in range(len(benchmarksBySize)):
            outFileName = BenchmarkSplitter.__appendFileNameSuffix(outputFileBase, i, separator, suffixFormat)
            with open(outFileName, "w") as f:
                yaml.safe_dump(benchmarksBySize[i], f)
