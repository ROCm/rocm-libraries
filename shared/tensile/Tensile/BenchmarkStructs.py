################################################################################
#
# Copyright (C) 2016-2023 Advanced Micro Devices, Inc. All rights reserved.
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

from copy import deepcopy
import itertools
from .Common import tPrint, printExit, \
        defaultBenchmarkCommonParameters, validParameters, globalParameters, \
        defaultBatchedBenchmarkFinalProblemSizes, defaultBenchmarkFinalProblemSizes
from .CustomKernels import getAllCustomKernelNames
from .SolutionStructs import ProblemType, ProblemSizes


def checkParametersAreValid(param, validParams):
    """Ensures paramaters in params exist and have valid values as specified by validParames"""
    (name, values) = param
    if name == "ProblemSizes":
        return

    if name not in validParams:
        printExit("Invalid parameter name: {}\nValid parameters are {}." \
                .format(name, sorted(validParameters.keys())))

    for value in values:
        if validParams[name] != -1 and value not in validParams[name]:
            msgBase = "Invalid parameter value: {} = {}\nValid values for {} are {}{}."
            msgExt = " (only first 32 combos printed)\nRefer to Common.py for more info" \
                    if len(validParams[name])>32 else ""
            printExit(msgBase.format(name, value, name, validParams[name][:32], msgExt))


def separateParameters(paramSetList):
    """Separates paramSetList into parameters with single and multiple values"""
    singleVaules = {}
    multiValues = {}
    for name, values in paramSetList.items():
        if values == None:
            printExit("You must specify value(s) for parameter \"{}\"".format(name))
        if len(values) == 1 and name != "ProblemSizes":
            singleVaules[name] = values[0]
        elif len(values) > 1 and name != "ProblemSizes":
            multiValues[name] = values

    return singleVaules, multiValues


def checkCDBufferAndStrides(problemType, problemSizes, isCEqualD):
    """Ensures ldd == ldc when CEqualD"""
    if isCEqualD and problemType["OperationType"] == "GEMM":
        for problem in problemSizes.problems:
            ldd = problem.sizes[problemType["IndexAssignmentsLD"][0]]
            ldc = problem.sizes[problemType["IndexAssignmentsLD"][1]]
            if ldd != ldc:
                printExit("LDD({}) != LDC({}) causes unpredictable result when CEqualD(True)" \
                        .format(ldd, ldc))


class BenchmarkProcess:
    """Representation of benchmarking parameters and resulting steps"""

    def __init__(self, problemTypeConfig, problemSizeGroupConfig):
        """Create from the two sections of a config for a BenchmarkProblem"""
        self.problemType = ProblemType(problemTypeConfig)
        self.isBatched = "Batched" in problemTypeConfig and problemTypeConfig["Batched"]
        tPrint(3, "# BenchmarkProcess beginning {}".format(self.problemType))

        # fill parameter values from config
        self.singleValueParams = {}
        self.multiValueParams = {}
        self.customKernels = []
        self.sizes = None
        self.getConfigParameters(self.isBatched, problemSizeGroupConfig)

        # convert parameter lists to steps
        # previously, multiple benchmark steps were possible
        # currently only 1 benchmark step is possible; more may be added back later
        self.benchmarkSteps = []
        self.benchmarkStepIdx = 0
        self.convertParametersToSteps()

    def getConfigParameters(self, isbatched, config):
        """Parse and validate benchmarking parameters in config"""
        tPrint(3, "")
        tPrint(3, "####################################################################")
        tPrint(1, "# Filling in Parameters With Defaults")
        tPrint(3, "####################################################################")
        tPrint(3, "")

        # check for no longer supported legacy benchmark steps
        badParams = ["InitialSolutionParameters", "BenchmarkForkParameters", \
                     "JoinParameters", "BenchmarkJoinParameters"]
        badsInConfig = []

        for p in badParams:
            if config.get(p) is not None:
                badsInConfig.append(p)

        if len(badsInConfig) == 1:
            printExit("Benchmark step {} is no longer supported".format("'" + badsInConfig[0] +
                                                                        "'"))
        elif len(badsInConfig) > 1:
            printExit("Benchmark steps {} are no longer supported".format(badsInConfig))

        # get supported configurations
        # value in config file may be "None", which we should ignore
        def getNonNoneFromConfig(key, default):
            if config.get(key) is not None:
                return config[key]
            else:
                return default

        # converts list of dicts into a flat dict
        benchmarkCommonParams = dict(itertools.chain(*[x.items() \
                for x in getNonNoneFromConfig("BenchmarkCommonParameters", [])]))
        forkParams = dict(itertools.chain(*[x.items() \
                for x in getNonNoneFromConfig("ForkParameters", [])]))
        self.paramGroups = forkParams.pop("Groups") if "Groups" in forkParams else []
        self.customKernels = getNonNoneFromConfig("CustomKernels", [])

        if "BenchmarkFinalParameters" in config:
            sizes = config["BenchmarkFinalParameters"][0]["ProblemSizes"]
        else:
            sizes = defaultBatchedBenchmarkFinalProblemSizes if isbatched \
                else defaultBenchmarkFinalProblemSizes

        self.problemSizes = ProblemSizes(self.problemType, sizes)
        checkCDBufferAndStrides(self.problemType, self.problemSizes, globalParameters["CEqualD"])

        # validate parameter values
        configParams = {**benchmarkCommonParams, **forkParams}
        for param in configParams.items():
            checkParametersAreValid(param, validParameters)

        # TODO other checks on groups (same params for each entry? no dups between groups?)
        for list in self.paramGroups:
            for group in list:
                for k, v in group.items():
                    checkParametersAreValid((k, [v]), validParameters)

        params = dict(itertools.chain(*[x.items() for x in defaultBenchmarkCommonParameters]))
        params.update(configParams)
        self.singleValueParams, self.multiValueParams = separateParameters(params)

        # print summary of parameter values
        tPrint(3, "Single Value Parameters:")
        for k, v in sorted(self.singleValueParams.items()):
            tPrint(3, "    {}: {}".format(k, v))

        tPrint(3, "Multi-Value Parameters:")
        for k, v in sorted(self.multiValueParams.items()):
            tPrint(3, "    {}: {}".format(k, v))

        if len(self.paramGroups) > 0:
            tPrint(3, "{} Parameter Group(s):".format(len(self.paramGroups)))
            for i, group in enumerate(self.paramGroups):
                tPrint(3, "    {} entries is group {}".format(len(group), i + 1))

    def convertParametersToSteps(self):
        """Create benchmark steps based on parsed parameters"""
        tPrint(3, "")
        tPrint(3, "####################################################################")
        tPrint(1, "# Convert Parameters to Benchmark Step(s)")
        tPrint(3, "####################################################################")
        tPrint(3, "")

        # currently only a single step is supported
        tPrint(3, "")
        tPrint(3, "####################################################################")
        tPrint(1, "# Benchmark Final")
        benchmarkStep = BenchmarkStep( \
                self.multiValueParams, \
                self.singleValueParams, \
                self.paramGroups, \
                self.customKernels, \
                self.problemSizes, \
                self.benchmarkStepIdx)
        self.benchmarkSteps.append(benchmarkStep)
        self.benchmarkStepIdx += 1

    def __len__(self):
        return len(self.benchmarkSteps)

    def __getitem__(self, key):
        return self.benchmarkSteps[key]

    def __str__(self):
        string = "BenchmarkProcess:\n"
        for step in self.benchmarkSteps:
            string += str(step)
        return string

    def __repr__(self):
        return self.__str__()


def constructForkPermutations(forkParams, paramGroups):
    """Constructs cartesian product of parameter values in forkParams and paramGroups"""

    myParams = {}
    myParams.update(forkParams)

    totalPermutations = 1
    for _, values in forkParams.items():
        totalPermutations *= len(values)

    # add groups to parameters to fork on
    for i, group in enumerate(paramGroups):
        myParams["_group" + str(i)] = group
        totalPermutations *= len(group)

    forkPermutations = []
    for i in range(0, totalPermutations):
        permutation = {}
        pIdx = i
        for name, v in myParams.items():
            values = deepcopy(v)
            valueIdx = pIdx % len(v)

            # groups have multiple parameters to update
            if "_group" in name:
                entry = values[valueIdx]
                for n2, v2 in entry.items():
                    permutation[n2] = v2
            else:
                permutation[name] = values[valueIdx]

            pIdx //= len(values)
        forkPermutations.append(permutation)

    return forkPermutations


class BenchmarkStep:
    """A single benchmark step which consists of constant and fork parameters and a set of sizes"""

    def __init__(self, forkParams, constantParams, paramGroups, customKernels, problemSizes, idx):
        """Basic constructor storing each argument"""
        self.forkParams = forkParams
        self.constantParams = constantParams
        self.paramGroups = paramGroups
        self.customKernels = customKernels
        self.problemSizes = problemSizes
        self.stepIdx = idx

        self.customKernelWildcard = False
        if self.customKernels == ["*"]:
            self.customKernels = getAllCustomKernelNames()
            self.customKernelWildcard = True

        tPrint(3, "# Creating BenchmarkStep: {} fork params and {} sizes" \
                .format( len(forkParams), problemSizes.totalProblemSizes))

    def isFinal(self):
        """Legacy. Currently always returns true since only one benchmark step is possible"""
        return True

    def __str__(self):
        string = "{:02d}".format(self.stepIdx)
        if self.isFinal():
            string += "_Final"
        return string

    def __repr__(self):
        return self.__str__()
