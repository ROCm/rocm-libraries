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

from copy import deepcopy
import itertools

from Tensile.Common.ValidParameters import checkParametersAreValid
from Tensile.Common import print1, print2, hasParam, printExit
from Tensile.Common.GlobalParameters import defaultBenchmarkCommonParameters, globalParameters, \
                                            defaultBatchedBenchmarkFinalProblemSizes, \
                                            defaultBenchmarkFinalProblemSizes
from Tensile.Common.ValidParameters import validParameters
from Tensile.SolutionStructs.Problem import ProblemType

from .CustomKernels import getAllCustomKernelNames
from .SolutionStructs import ProblemSizes, ActivationArgs, BiasTypeArgs, \
        FactorDimArgs


def getDefaultsForMissingParameters(paramList, defaultParams):
    """Returns all parameters (with values) in defaultParams not present in paramList"""
    benchmarkParams = {}
    for paramDict in defaultParams:
        for name, value in paramDict.items():
            if not hasParam(name, paramList) \
                    or name == "ProblemSizes":
                benchmarkParams[name] = value
    return benchmarkParams


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

    def __init__(self, problemTypeConfig, problemSizeGroupConfig, printIndexAssignmentInfo: bool):
        """Create from the two sections of a config for a BenchmarkProblem"""
        self.problemType = ProblemType(problemTypeConfig, printIndexAssignmentInfo)
        self.isBatched = "Batched" in problemTypeConfig and problemTypeConfig["Batched"]
        print2("# BenchmarkProcess beginning {}".format(self.problemType))

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
        print2("")
        print2("####################################################################")
        print1("# Filling in Parameters With Defaults")
        print2("####################################################################")
        print2("")

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
        self.internalSupportParams = getNonNoneFromConfig("InternalSupportParams", {})
        if self.customKernels == [] and self.internalSupportParams != {}:
            printExit("InternalSupportParams only supports Custom Kernels")

        activationConf = ""
        biasTypesConf  = ""
        factorDimConf  = ""
        icacheFlush = None
        if "BenchmarkFinalParameters" in config:
            sizes          = config["BenchmarkFinalParameters"][0]["ProblemSizes"]
            for bfp in config["BenchmarkFinalParameters"][1:]:
                if "ActivationArgs" in bfp:
                  if activationConf:
                    printExit("Duplicated ActivationArgs.")
                  activationConf = bfp["ActivationArgs"]
                if "BiasTypeArgs" in bfp:
                  if biasTypesConf:
                    printExit("Duplicated BiasTypeArgs.")
                  biasTypesConf = bfp["BiasTypeArgs"]
                if "FactorDimArgs" in bfp:
                  if factorDimConf:
                    printExit("Duplicated FactorDimArgs.")
                  factorDimConf = bfp["FactorDimArgs"]
                if "ICacheFlush" in bfp:
                  if icacheFlush is not None:
                    printExit("Duplicated ICacheFlush.")
                  icacheFlush = bfp["ICacheFlush"]
        else:
            sizes = defaultBatchedBenchmarkFinalProblemSizes if isbatched \
                else defaultBenchmarkFinalProblemSizes

        if icacheFlush is None:
          icacheFlush = [False,]

        self.problemSizes = ProblemSizes(self.problemType, sizes)
        checkCDBufferAndStrides(self.problemType, self.problemSizes, globalParameters["CEqualD"])

        self.biasTypesArgs  = BiasTypeArgs(self.problemType, biasTypesConf)
        self.activationArgs = ActivationArgs(self.problemType, activationConf)
        self.factorDimArgs  = FactorDimArgs(self.problemType, factorDimConf)
        self.icacheFlushArgs = icacheFlush

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
        print2("Single Value Parameters:")
        for k, v in self.singleValueParams.items():
            print2("    {}: {}".format(k, v))

        print2("Multi-Value Parameters:")
        for k, v in self.multiValueParams.items():
            print2("    {}: {}".format(k, v))

        if len(self.paramGroups) > 0:
            print2("{} Parameter Group(s):".format(len(self.paramGroups)))
            for i, group in enumerate(self.paramGroups):
                print2("    {} entries is group {}".format(len(group), i + 1))

    def convertParametersToSteps(self):
        """Create benchmark steps based on parsed parameters"""
        print2("")
        print2("####################################################################")
        print1("# Convert Parameters to Benchmark Step(s)")
        print2("####################################################################")
        print2("")

        # currently only a single step is supported
        print2("")
        print2("####################################################################")
        print1("# Benchmark Final")
        benchmarkStep = BenchmarkStep( \
                self.multiValueParams, \
                self.singleValueParams, \
                self.paramGroups, \
                self.customKernels, \
                self.internalSupportParams, \
                self.problemSizes, \
                self.biasTypesArgs, \
                self.factorDimArgs, \
                self.activationArgs, \
                self.icacheFlushArgs, \
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

    def __init__(self, forkParams, constantParams, paramGroups, customKernels, internalSupportParams, problemSizes, biasTypeArgs, factorDimArgs, activationArgs, icacheFlushArgs, idx):
        """Basic constructor storing each argument"""
        self.forkParams = forkParams
        self.constantParams = constantParams
        self.paramGroups = paramGroups
        self.customKernels = customKernels
        self.internalSupportParams = internalSupportParams
        self.problemSizes = problemSizes
        self.biasTypeArgs = biasTypeArgs
        self.factorDimArgs = factorDimArgs
        self.activationArgs = activationArgs
        self.icacheFlushArgs = icacheFlushArgs
        self.stepIdx = idx

        self.customKernelWildcard = False
        if self.customKernels == ["*"]:
            self.customKernels = getAllCustomKernelNames()
            self.customKernelWildcard = True

        print2("# Creating BenchmarkStep: {} fork params and {} sizes" \
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
