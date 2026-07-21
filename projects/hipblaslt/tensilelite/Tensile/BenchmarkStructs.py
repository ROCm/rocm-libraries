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

from Tensile.Common.ValidParameters import checkParametersAreValid, validateInternalSupportParams
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


def _isValidParameterValue(name, value):
    """Return True if value is a scalar valid value for name."""
    if name not in validParameters:
        return False
    validValues = validParameters[name]
    return validValues == -1 or value in validValues


def _groupedParameterValueOptions(name, value):
    """Return scalar choices for one parameter inside a group entry."""
    if not isinstance(value, list):
        return [value]
    if len(value) == 0:
        printExit("You must specify value(s) for parameter \"{}\" in Groups".format(name))
    if name == "MatrixInstruction" and all(not isinstance(v, list) for v in value):
        return [value]
    if name in validParameters and validParameters[name] != -1 and _isValidParameterValue(name, value):
        return [value]
    return value


def _expandGroupedParameters(paramGroups):
    """Expand list-valued entries inside Groups into scalar group entries.

    Groups historically contained scalar dictionaries.  Keep list-valued scalar
    parameters such as MatrixInstruction intact, but allow list-valued entries
    like NonTemporalA: [0, 1] to expand within that group entry.
    """
    expandedParamGroups = []
    for paramGroup in paramGroups:
        expandedParamGroup = []
        for groupEntry in paramGroup:
            names = []
            valueOptions = []
            for name, value in groupEntry.items():
                names.append(name)
                valueOptions.append(_groupedParameterValueOptions(name, value))

            for values in itertools.product(*valueOptions):
                expandedParamGroup.append(dict(zip(names, values)))
        expandedParamGroups.append(expandedParamGroup)

    return expandedParamGroups


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

    def __init__(self, problemTypeConfig, problemSizeGroupConfig, printIndexAssignmentInfo: bool,
                 keyPathPrefix: str = "", srcFile: str = ""):
        """Create from the two sections of a config for a BenchmarkProblem.

        ``keyPathPrefix`` (e.g. ``BenchmarkProblems[i][1+groupIdx]``) and
        ``srcFile`` are threaded through to ``checkParametersAreValid``
        so type-mismatch errors carry the YAML location of the offending
        key. Both are optional; an empty prefix produces the unqualified
        keypath used by ad-hoc callers and tests.
        """
        self.problemType = ProblemType(problemTypeConfig, printIndexAssignmentInfo)
        self.isBatched = "Batched" in problemTypeConfig and problemTypeConfig["Batched"]
        print2("# BenchmarkProcess beginning {}".format(self.problemType))

        # fill parameter values from config
        self.singleValueParams = {}
        self.multiValueParams = {}
        self.customKernels = []
        self.sizes = None
        self.getConfigParameters(self.isBatched, problemSizeGroupConfig,
                                 keyPathPrefix=keyPathPrefix, srcFile=srcFile)

        # convert parameter lists to steps
        # previously, multiple benchmark steps were possible
        # currently only 1 benchmark step is possible; more may be added back later
        self.benchmarkSteps = []
        self.benchmarkStepIdx = 0
        self.convertParametersToSteps()

    def getConfigParameters(self, isbatched, config, keyPathPrefix: str = "", srcFile: str = ""):
        """Parse and validate benchmarking parameters in config.

        ``keyPathPrefix`` is the YAML location of the surrounding
        ``BenchmarkProblems[i][1+groupIdx]`` slice; per-section keypaths
        (``.BenchmarkCommonParameters``, ``.ForkParameters``,
        ``.ForkParameters.Groups[g][e]``) are appended by this method
        when invoking ``checkParametersAreValid``. ``srcFile`` is
        forwarded so error messages can include the YAML path and a
        recovered line number.
        """
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
        self.paramGroups = _expandGroupedParameters(forkParams.pop("Groups")) if "Groups" in forkParams else []
        self.customKernels = getNonNoneFromConfig("CustomKernels", [])
        self.internalSupportParams = getNonNoneFromConfig("InternalSupportParams", {})
        if self.customKernels == [] and self.internalSupportParams != {}:
            printExit("InternalSupportParams only supports Custom Kernels")

        ispPrefix = f"{keyPathPrefix}.InternalSupportParams" if keyPathPrefix \
                    else "InternalSupportParams"
        validateInternalSupportParams(
            self.internalSupportParams,
            srcFile=srcFile, keyPathPrefix=ispPrefix,
        )

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

        commonPrefix = f"{keyPathPrefix}.BenchmarkCommonParameters" if keyPathPrefix \
                       else "BenchmarkCommonParameters"
        forkPrefix = f"{keyPathPrefix}.ForkParameters" if keyPathPrefix else "ForkParameters"

        for param in benchmarkCommonParams.items():
            checkParametersAreValid(
                param, validParameters,
                keyPathPrefix=commonPrefix, srcFile=srcFile,
            )
        for param in forkParams.items():
            checkParametersAreValid(
                param, validParameters,
                keyPathPrefix=forkPrefix, srcFile=srcFile,
            )

        # TODO other checks on groups (same params for each entry? no dups between groups?)
        for gIdx, list in enumerate(self.paramGroups):
            for eIdx, group in enumerate(list):
                groupsPrefix = f"{forkPrefix}.Groups[{gIdx}][{eIdx}]"
                for k, v in group.items():
                    checkParametersAreValid(
                        (k, [v]), validParameters,
                        keyPathPrefix=groupsPrefix, srcFile=srcFile,
                    )

        params = dict(itertools.chain(*[x.items() for x in defaultBenchmarkCommonParameters]))
        params.update({**benchmarkCommonParams, **forkParams})
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

class constructForkPermutations():
    def __init__(self, forkParams, paramGroups):
        totalPermutations = 0
        for groups in itertools.product(*paramGroups):
            group = set()
            for g in groups:
                group.update(g.keys())
            groupPermutations = 1
            for name, values in forkParams.items():
                if name not in group:
                    groupPermutations *= len(values)
            totalPermutations += groupPermutations

        self.forkParams = forkParams
        self.paramGroups = paramGroups
        self.totalPermutations = totalPermutations
        self._generator = None

    def __iter__(self):
        self._generator = constructLazyForkPermutations(self.forkParams, self.paramGroups)
        return self._generator

    def __len__(self):
        return self.totalPermutations

    def __next__(self):
        if not self._generator:
            self.__iter__()
        return next(self._generator)

def constructLazyForkPermutations(forkParams, paramGroups):
    """Constructs cartesian product of parameter values in forkParams and paramGroups"""

    for groups in itertools.product(*paramGroups):
        group = {}
        for g in groups:
            group.update(g)
        params_list = list(forkParams)
        for name in group:
            if name not in forkParams:
                params_list.append(name)
        values = []
        for name in params_list:
            values.append([group[name]] if name in group else forkParams[name])
        for combination in itertools.product(*reversed(values)):
            permutation = dict(zip(params_list, reversed(combination)))
            yield permutation


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
