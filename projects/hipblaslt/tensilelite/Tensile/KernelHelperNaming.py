################################################################################
#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
from enum import IntEnum

from Tensile.Common.GlobalParameters import internalParameters
from Tensile.KernelWriterBetaOnly import KernelWriterBetaOnly
from Tensile.KernelWriterConversion import KernelWriterConversion
from Tensile.KernelWriterActivationEnumHeader import KernelWriterActivationEnumHeader
from Tensile.KernelWriterActivationFunction import KernelWriterActivationFunction
from Tensile.KernelWriterReduction import KernelWriterReduction


class KernelHelperEnum(IntEnum):
    BetaOnly = 0
    Conversion = 1
    ActivationEnumHeader = 2
    ActivationFunction = 3
    Reduction = 4
    All = 5


def conversionKernelNames(solution):
  # need to check that the fields mutated in the the init function aren't used in the naming function
  # e.g. usebias got set to zero if gradient which changes behavior of name function.
  conversionKernelNames = []
  loadVectorWidth = [1, 2] if solution["ProblemType"]["DataType"].isDouble() else [1, 2, 4]
  gsuList = [internalParameters["GlobalSplitUPGR"]]
  if solution["GlobalSplitUAlgorithm"] == "SingleBuffer":
    gsuList = [1]
  elif solution["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel":
    return
  for vw in loadVectorWidth:
    # for _ in gsuList: I don't think this does anything
    if solution["ProblemType"]["UseBias"]:
      typeList = solution["ProblemType"]["BiasDataTypeList"]
      if solution["ProblemType"]["Gradient"]:
      #  # If gradient + bias D, generates a normal GSU kernel for bias D = nullptr case
        conversionKernelNames.append(KernelWriterConversion.kernelName(solution, vw))
      for btype in typeList:
        conversionKernelNames.append(KernelWriterConversion.kernelName(solution, vw, btype))
    else:
      conversionKernelNames.append(KernelWriterConversion.kernelName(solution, vw))
  return conversionKernelNames if conversionKernelNames else []


def activationEnumHeaderNames(solution):
  activationEnumHeaderNames = []
  if solution["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all']:
    activationEnumHeaderNames.append(KernelWriterActivationEnumHeader.kernelName(solution))
  return activationEnumHeaderNames


def activationFunctionNames(solution):
  activationFunctionNames = []
  if solution["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all']:
    activationFunctionNames.append(KernelWriterActivationFunction.kernelName(solution))
  return activationFunctionNames


def reductionKernelNames(solution):
  reductionKernelNames = []
  if solution["ProblemType"]["Gradient"] and solution["ProblemType"]["UseBias"]:
    for btype in solution["ProblemType"]["BiasDataTypeList"]:
      reductionKernelNames.append(KernelWriterReduction.kernelName(solution, btype))
  return reductionKernelNames


def betaOnlyKernelNames(solution):
  betaOnlyKernelNames = []
  if (solution["GlobalSplitU"] > 1 or solution["GlobalSplitU"] == -1) or (solution["StreamK"] > 0 and solution["StreamKAtomic"] == 1):
    if solution["ProblemType"]["UseBias"]:
      for btype in solution["ProblemType"]["BiasDataTypeList"]:
        betaOnlyKernelNames.append(KernelWriterBetaOnly.kernelName(solution, btype))
    else:
      betaOnlyKernelNames.append(KernelWriterBetaOnly.kernelName(solution))
  return betaOnlyKernelNames


def kernelObjectNameCallables():
    return [(KernelHelperEnum.Conversion, conversionKernelNames),
            (KernelHelperEnum.ActivationEnumHeader, activationEnumHeaderNames),
            (KernelHelperEnum.ActivationFunction, activationFunctionNames),
            (KernelHelperEnum.Reduction, reductionKernelNames),
            (KernelHelperEnum.BetaOnly, betaOnlyKernelNames)]


def initHelperKernelObjects(solution, kernelHelperType, cxxCompiler, isaInfoMap):
    if kernelHelperType not in KernelHelperEnum:
        raise Exception("Failed to find kernel helper type.")
    result = []
    if kernelHelperType == KernelHelperEnum.BetaOnly or kernelHelperType == KernelHelperEnum.All:
        result.extend(initBetaOnlyKernelObjects(solution))
    if kernelHelperType == KernelHelperEnum.Conversion or kernelHelperType == KernelHelperEnum.All:
        result.extend(initConversionKernelObjects(solution, isaInfoMap))
    if kernelHelperType == KernelHelperEnum.ActivationEnumHeader or kernelHelperType == KernelHelperEnum.All:
        result.extend(initActivationEnumHeaderObjects(solution))
    if kernelHelperType == KernelHelperEnum.ActivationFunction or kernelHelperType == KernelHelperEnum.All:
        result.extend(initActivationFunctionObjects(solution, cxxCompiler, isaInfoMap))
    if kernelHelperType == KernelHelperEnum.Reduction or kernelHelperType == KernelHelperEnum.All:
        result.extend(initReductionKernelObjects(solution))
    sortByEnum = lambda x: ("Enum" in x.getKernelName(), result.index(x))
    return sorted(result, key=sortByEnum, reverse=True) # Ensure that we write Enum kernel helpers are first in list


def initBetaOnlyKernelObjects(solution):
  betaOnlyKernelObjects = []
  if (solution["GlobalSplitU"] > 1 or solution["GlobalSplitU"] == -1) or (solution["StreamK"] > 0 and solution["StreamKAtomic"] == 1):
    if solution["ProblemType"]["UseBias"]:
      for btype in solution["ProblemType"]["BiasDataTypeList"]:
        state = {}
        state["ProblemType"] = deepcopy(solution["ProblemType"])
        state["ProblemType"]["GroupedGemm"] = False
        state["ProblemType"]["BiasDataTypeList"] = []
        state["ProblemType"]["BiasDataType"] = deepcopy(btype)
        state["KernelLanguage"] = "Source"
        state["_GlobalAccumulation"] = solution["_GlobalAccumulation"]
        betaOnlyKernelObjects.append(KernelWriterBetaOnly(state))
    else:
      state = {}
      state["ProblemType"] = deepcopy(solution["ProblemType"])
      state["ProblemType"]["GroupedGemm"] = False
      state["KernelLanguage"] = "Source"
      state["_GlobalAccumulation"] = solution["_GlobalAccumulation"]
      betaOnlyKernelObjects.append(KernelWriterBetaOnly(state))
  return betaOnlyKernelObjects


def initConversionKernelObjects(solution, isaInfoMap):
  conversionKernelObjects = []
  loadVectorWidth = [1, 2] if solution["ProblemType"]["DataType"].isDouble() else [1, 2, 4]
  genPGRPostKernels = True
  gsuList = [internalParameters["GlobalSplitUPGR"]]
  if solution["GlobalSplitUAlgorithm"] == "SingleBuffer":
    genPGRPostKernels = False
    gsuList = [1]
  elif solution["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel":
    return conversionKernelObjects
  for vw in loadVectorWidth:
    for globalSplitU in gsuList:
      unrollOnly = False if globalSplitU == internalParameters["GlobalSplitUPGR"] else True
      if solution["ProblemType"]["UseBias"]:
        typeList = solution["ProblemType"]["BiasDataTypeList"]
        if solution["ProblemType"]["Gradient"]:
          # If gradient + bias D, generates a normal GSU kernel for bias D = nullptr case
          state = {}
          state["ProblemType"] = deepcopy(solution["ProblemType"])
          state["ProblemType"]["GroupedGemm"] = False
          state["ProblemType"]["UseBias"] = 0
          state["GenPGRPostKernels"] = genPGRPostKernels
          state["KernelLanguage"] = "Source"
          state["GlobalSplitU"] = globalSplitU
          state["UnrollOnly"] = unrollOnly
          state["_GlobalAccumulation"] = solution["_GlobalAccumulation"]
          state["ActivationFused"] = solution["ActivationFused"]
          conversionKernelObjects.append(KernelWriterConversion(state, vw, isaInfoMap))
        for btype in typeList:
          state = {}
          state["ProblemType"] = deepcopy(solution["ProblemType"])
          state["ProblemType"]["GroupedGemm"] = False
          state["ProblemType"]["BiasDataTypeList"] = []
          state["ProblemType"]["BiasDataType"] = deepcopy(btype)
          state["GenPGRPostKernels"] = genPGRPostKernels
          state["KernelLanguage"] = "Source"
          state["GlobalSplitU"] = globalSplitU
          state["UnrollOnly"] = unrollOnly
          state["_GlobalAccumulation"] = solution["_GlobalAccumulation"]
          state["ActivationFused"] = solution["ActivationFused"]
          conversionKernelObjects.append(KernelWriterConversion(state, vw, isaInfoMap))
      else:
        state = {}
        state["ProblemType"] = deepcopy(solution["ProblemType"])
        state["ProblemType"]["GroupedGemm"] = False
        state["GenPGRPostKernels"] = genPGRPostKernels
        state["KernelLanguage"] = "Source"
        state["GlobalSplitU"] = globalSplitU
        state["UnrollOnly"] = unrollOnly
        state["_GlobalAccumulation"] = solution["_GlobalAccumulation"]
        state["ActivationFused"] = solution["ActivationFused"]
        conversionKernelObjects.append(KernelWriterConversion(state, vw, isaInfoMap))
  return conversionKernelObjects


def initActivationEnumHeaderObjects(solution):
  activationEnumHeaderObjects = []
  if solution["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all']:
    state = {}
    state["ProblemType"] = deepcopy(solution["ProblemType"])
    state["ProblemType"]["GroupedGemm"] = False
    state["KernelLanguage"] = "Source"
    activationEnumHeaderObjects.append(KernelWriterActivationEnumHeader(state))
  return activationEnumHeaderObjects


def initActivationFunctionObjects(solution, cxxCompiler, isaInfoMap):
  activationFunctionObjects = []
  if solution["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all']:
    state = {}
    state["ProblemType"] = deepcopy(solution["ProblemType"])
    state["ProblemType"]["GroupedGemm"] = False
    state["KernelLanguage"] = "Source"
    state["Kernel"] = {"WavefrontSize": solution["WavefrontSize"], "ISA": tuple(solution["ISA"])}
    activationFunctionObjects.append(KernelWriterActivationFunction(state, cxxCompiler, list(isaInfoMap.keys())))
  return activationFunctionObjects


def initReductionKernelObjects(solution):
  reductionKernelObjects = []
  if solution["ProblemType"]["Gradient"] and solution["ProblemType"]["UseBias"]:
    for btype in solution["ProblemType"]["BiasDataTypeList"]:
      state = {}
      state["ProblemType"] = deepcopy(solution["ProblemType"])
      state["ProblemType"]["GroupedGemm"] = False
      state["ProblemType"]["BiasDataTypeList"] = []
      state["ProblemType"]["BiasDataType"] = deepcopy(btype)
      reductionKernelObjects.append(KernelWriterReduction(state))
  return reductionKernelObjects
