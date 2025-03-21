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

from .TensileInstructions import DataType, roundUpToNearestMultiple
from .TensileInstructions.Base import fastdeepcopy as deepcopy

from .KernelWriterBetaOnly import KernelWriterBetaOnly
from .KernelWriterConversion import KernelWriterConversion
from .KernelWriterActivationEnumHeader import KernelWriterActivationEnumHeader
from .KernelWriterActivationFunction import KernelWriterActivationFunction
from .KernelWriterActivationOnly import KernelWriterActivationOnly
from .KernelWriterReduction import KernelWriterReduction

from .AsmStoreState import VectorDataTypes
from .Activation import ActivationType

from .CustomKernels import isCustomKernelConfig

from .Common import assignParameterWithDefault, \
                    defaultProblemType, defaultSolution, \
                    defaultInternalSupportParams, \
                    globalParameters, internalParameters, \
                    print2, printExit, printWarning, \
                    validMFMA, validSMFMA, validParameters, \
                    validGEMMTypes, HPATypes, roundUp, validWMMA, INDEX_CHARS

from collections import OrderedDict
from collections.abc import Mapping
from enum import Enum
from functools import lru_cache
from typing import List

import collections
import math
import operator
import sys

########################################
# Print a reject message :
def reject(state, *args):
  if state and "NoReject" in state and state["NoReject"]:
    return

  if globalParameters["PrintSolutionRejectionReason"]:
    sys.stdout.write("\nreject: ")
    for a in args:
      print(a)
    #traceback.print_stack(None, 2)
    solutionIndex = state["SolutionIndex"] if (state != None and "SolutionIndex" in state) else -1
    if solutionIndex != -1:
      # If we have valid solutionIndex, this means we are during TensileCreateLibrary stage
      # In this stage, all solutions in the logic should be valid
      # So if any rejection happens, print the warning for further check
      # This will be done only when --global-parameters=PrintSolutionRejectionReason=True
      solutionNameMin = state["SolutionNameMin"] if ("SolutionNameMin" in state) else None
      # if we don't have SolutionNameMin, we simply use the problemTypeName
      solutionNameMin = str(state["ProblemType"]) if (solutionNameMin == None) else solutionNameMin
      print("!! Warning: Any rejection of a LibraryLogic is not expected, please check. \
        SolutionIndex: %d (or SolutionName/ProblemType: %s)"%(solutionIndex, solutionNameMin))
  if state != None:
    state["Valid"] = False

# print a labled variable
def pvar(state, field):
  return field + "=" + str(state[field])

def roundupRatio(dividend, divisor):
  return int(math.ceil(float(dividend) / float(divisor)))

class Fbs(Enum):
  Free=0     # Expect to be free dimension
  Batch=1    # Expect to be batch dimension
  Sum=2      # Expect to be summation dimension

################################################################################
# ProblemType
# name of solution should begin with name of problemType, and arguments can be listed out explicitly
class ProblemType(Mapping):
  ########################################
  def __init__(self, config):
    self.state = {}

    for key in defaultProblemType:
      assignParameterWithDefault(self.state, key, config, defaultProblemType)

    # adjusting all data types
    if "DataType" in config:
      self["DataType"]  = DataType(config["DataType"])
      self["DataTypeA"] = self["DataType"]
      self["DataTypeB"] = self["DataType"]
    else:
      printExit("NO data type specified")
      self["DataType"]  = DataType(0)
      self["DataTypeA"] = DataType(0)
      self["DataTypeB"] = DataType(0)

    if "DataTypeA" in config:
      self["DataTypeA"] = DataType(config["DataTypeA"])

    if "DataTypeB" in config:
      self["DataTypeB"] = DataType(config["DataTypeB"])

    if "DestDataType" in config:
      self["DestDataType"] = DataType(config["DestDataType"])
    else:
      if "DataType" in config:
        self["DestDataType"] = DataType(config["DataType"])
      else:
        printExit("NO dest data type or data type specified")
        self["DataType"] = DataType(0)

    self["DataTypeE"] = self["DestDataType"]
    if "DataTypeE" in config:
      self["DataTypeE"] = DataType(config["DataTypeE"])

    if "ComputeDataType" in config:
      self["ComputeDataType"] = DataType(config["ComputeDataType"])
    else:
      if "DestDataType" in config:
        self["ComputeDataType"] = DataType(config["DestDataType"])
      else:
        if "DataType" in config:
          self["ComputeDataType"] = DataType(config["DataType"])
        else:
          printExit("NO compute data type, or dest data type, or data type specified")
          self["DataType"] = DataType(0)

    # Just like DataTypeE is DestDataType by default; DataTypeAmaxD if ComputeDataType by default.
    # So far we don't have to set it in config yamls
    self["DataTypeAmaxD"] = self["ComputeDataType"]
    if "DataTypeAmaxD" in config:
      self["DataTypeAmaxD"] = DataType(config["DataTypeAmaxD"])

    if self["Sparse"]:
      self["DataTypeMetadata"] = DataType("I8")

    if "F32XdlMathOp" in config:
        self["F32XdlMathOp"] = DataType(config["F32XdlMathOp"])
    else:
        self["F32XdlMathOp"] = DataType(0)

    # Modifying ComputeDataType for HHH+HPA: if (HHH+HPA), convert it to HHS_BH by setting ComputeDataType to S.
    if self["ComputeDataType"].isHalf() and self["DataType"].isHalf() and self["HighPrecisionAccumulate"]:
      printWarning("Inconsistent DataTypes: DataType == f16, DestType == f16, ComputeDataType == f16, but HPA == True (HHH+HPA, no such a type); Converting HHH+HPA to HHS_BH by setting compute data type to f32.")
      self["ComputeDataType"] = DataType('s')

    # Modifying ComputeDataType for BBB+HPA: if (BBB+HPA), convert it to BBS_BH by setting ComputeDataType to S.
    if self["ComputeDataType"].isBFloat16() and self["DataType"].isBFloat16() and self["HighPrecisionAccumulate"]:
      printWarning("Inconsistent DataTypes: DataType == bf16, DestType == bf16, ComputeDataType == bf16, but HPA == True (BBB+HPA, no such a type); Converting BBB+HPA to BBS_BH by setting compute data type to f32.")
      self["ComputeDataType"] = DataType('s')

    # Modifying ComputeDataType for I8I8I_BH: if (I8I8I8+HPA), convert it to I8I8I_BH by setting ComputeDataType to i.
    if self["ComputeDataType"].isInt8() and DataType(config["DataType"]).isInt8() and self["HighPrecisionAccumulate"]:
      print2("DataType == i8 and HPA == True; setting compute data type to int32")
      self["ComputeDataType"] = DataType('i')

    if self["OperationType"] == "GEMM":
      self.checkIfSupportedGEMMType()
      self.initGEMM()
    else:
      printExit("Unsupported OperationType = %s" % self["OperationType"])

    self.state["AssignedDerivedParameters"] = False
    ProblemType.assignDerivedParameters(self.state)

    for tc in ('A', 'B'):
      for sc in self["SetConstStride%s"%tc] :
          (anchorDim, stride) = sc[:2]
          if anchorDim not in self.state["IndexAssignments%s"%tc]:
              printExit("SetConstStride%s=%s anchorDim=%u is not in IndexAssignments%s"%(tc, sc, anchorDim, tc))

    # Bias
    # If compute data type is not equal to dest data type, tensile will run conversion kernel.
    # In this case we don't need to apply bias in beta only kernel.
    if "UseBias" in config:
      if self["ComputeDataType"] != self["DestDataType"]:
        self["BetaOnlyUseBias"] = False
      else:
        self["BetaOnlyUseBias"] = True if self["UseBias"] > 0 else False
      if "BiasDataTypeList" in config:
        self["BiasDataTypeList"] = [DataType(btype) for btype in config["BiasDataTypeList"]]
        self["BiasDataTypeList"].sort() # Make name unique
      else:
        self["BiasDataTypeList"] = getBiasDataTypeListDefault(self)
    else:
      self["BetaOnlyUseBias"] = False
      self["BiasDataTypeList"] = []

    # Activation
    # Currently, ActivationType supports only 'all' and 'hipblaslt_all', and is active only when the Activation configuration is set to True.
    # Otherwise, ActivationType will be set to 'none'.
    if "Activation" in config:
      typeStr = config.get("ActivationType", 'none')
      if typeStr not in ['all', 'hipblaslt_all']:
        typeStr = 'none'
    else:
      typeStr = 'none'
    self["ActivationType"] = ActivationType(typeStr)
    if "ActivationComputeDataType" in config:
      self["ActivationComputeDataType"] = DataType(config["ActivationComputeDataType"])
    else:
      self["ActivationComputeDataType"] = self["ComputeDataType"]

    if self["ActivationType"] != 'none':
      # This is a dummy guard in case we currently don't have a converter to convert data from compute type to activation compute type
      if self["ActivationComputeDataType"] not in [self["ComputeDataType"], self["DestDataType"]]:
        printWarning("TensileLite currently only supports ActivationComputeDataType (%s) = ComputeDataType (%s) or DestDataType (%s). \
                      ActivationComputeDataType will be set to ComputeDataType automatically."%(self["ActivationComputeDataType"].toChar(), \
                                                                                                self["ComputeDataType"], \
                                                                                                self["DestDataType"]))
        self["ActivationComputeDataType"] = self["ComputeDataType"]
      if (self["ActivationComputeDataType"].numRegisters() != self["ComputeDataType"].numRegisters()) and \
        (self["DataType"].numRegisters() < self["DestDataType"].numRegisters()):
        printWarning("TensileLite only supports ActivationComputeDataType = ComputeDataType if DestDataType > DataType. \
                      ActivationComputeDataType will be set to ComputeDataType automatically.")
        self["ActivationComputeDataType"] = self["ComputeDataType"]

    if "UseE" in config:
      if config["UseE"]:
        if self["ActivationType"] == 'none':
          printWarning("Use E is disabled cause Activation is set to False.")
          self["UseE"] = False
        else:
          self["UseE"] = config["UseE"]
      else:
        self["UseE"] = config["UseE"]

    if "Gradient" in config:
      if config["Gradient"]:
        if (not self["UseBias"]) and self["ActivationType"] == 'none':
          printWarning("Gradient is disabled cause bias and activation are both disabled.")
          self["Gradient"] = False
        if self["ActivationType"] != 'none' and self["UseE"] == False:
          printWarning("Use E is enabled cause Activation is enabled.")
          self["UseE"] = True
        elif self["ActivationType"] != 'none' and self["UseE"] == False:
          printWarning("Use E is disabled cause Activation is disabled.")
          self["UseE"] = False
        # if self["UseScaleAlphaVec"]:
        #   printWarning("Use scaleAlphaVec is disabled cause Gradient is enabled.")
        #   self["UseScaleAlphaVec"] = False
      self["Gradient"] = config["Gradient"]

    # Need gradient info
    biasSrcList = ["A", "B", "D"]
    if "BiasSrc" in config:
      if not self["Gradient"] and config["BiasSrc"] != "D":
        printWarning("BiasSrc is set to D cause Gradient is disabled.")
        self["BiasSrc"] = "D"
      elif self["Gradient"]:
        # # Currently only supports D :)
        # if config["BiasSrc"] != "D":
        #   printExit("BiasSrc currently only supports D.")
        if config["BiasSrc"] not in biasSrcList:
          printExit("BiasSrc only supports A, B, D.")

    if "ActivationNoGuard" in config:
      self["ActivationNoGuard"] = config["ActivationNoGuard"]
      if self["ActivationNoGuard"]:
        if self["ActivationType"] == 'none':
          printWarning("ActivationNoGuard is set to False cause Acivation is off.")
          self["ActivationNoGuard"] = False
        if (not self["Gradient"]):
          printWarning("ActivationNoGuard is set to False cause Gradient is off.")
          self["ActivationNoGuard"] = False

  ################################################################################
   # Function checkIfSupportedGEMMType:
  #   Assures 3 data-types are valid, supported and well-assigned
  #   See the discussion on Common.py for validGEMMTypes
  ################################################################################
  def checkIfSupportedGEMMType(self):
    inType = self["DataType"]
    outType = self["DestDataType"]
    computeType = self["ComputeDataType"]

    gemmType = ( inType.toChar(), outType.toChar(), computeType.toChar() )
    if gemmType not in validGEMMTypes:
      printExit("This typed-GEMM (Ti, To, Tc) = (%s, %s, %s) is not supported yet."%(gemmType[0],gemmType[1],gemmType[2]))

  ########################################
  def initGEMM(self):
    sumIdx = 3 if self["Batched"] else 2
    self["IndexAssignmentsA"] = [0, sumIdx] # N
    self["IndexAssignmentsB"] = [sumIdx, 1] # N
    if self.state["Sparse"] == 2:
      self["IndexAssignmentsMetadata"] = [sumIdx, 1] # N (ref B)
    else:
      self["IndexAssignmentsMetadata"] = [sumIdx, 0] # T (ref A)
    if self["TransposeA"]:
      self["IndexAssignmentsA"] = [sumIdx, 0] # T
    if self["TransposeB"]:
      self["IndexAssignmentsB"] = [1, sumIdx] # T
    if self["Batched"]:
      self["IndexAssignmentsA"].append(2)
      self["IndexAssignmentsB"].append(2)
      self["IndexAssignmentsMetadata"].append(2)
      self["NumIndicesC"] = 3
    else:
      self["NumIndicesC"] = 2

    self["NumIndicesLD"] = 4
    self["IndexAssignmentsLD"][0] = self["NumIndicesC"] + 1
    for i in range(1, len(self["IndexAssignmentsLD"])):
      self["IndexAssignmentsLD"][i] = self["IndexAssignmentsLD"][i-1] + 1

  ########################################
  def isGEMM(self):
    return self.operationType == 0

  ########################################
  # determine d0, d1, dU
  @staticmethod
  def assignDerivedParameters(state):
    if "AssignedDerivedParameters" in state:
      if state["AssignedDerivedParameters"]:
        return
    state["AssignedDerivedParameters"] = False

    state["TotalIndices"] = max(max(state["IndexAssignmentsA"])+1, \
        max(state["IndexAssignmentsB"])+1)

    # determine num free, batch
    state["IndicesFree"] = []
    state["IndicesBatch"] = []
    state["IndicesSummation"] = []

    for i in range(0, state["NumIndicesC"]):
      inA = i in state["IndexAssignmentsA"]
      inB = i in state["IndexAssignmentsB"]
      if inA and inB:
        state["IndicesBatch"].append(i)

      elif inA or inB:
        state["IndicesFree"].append(i)
      else:
        printExit("invalid index %u (inC but not (inA or inB))" % i)

    # determine num summation
    for i in range(state["NumIndicesC"], state["TotalIndices"]):
      inA = i in state["IndexAssignmentsA"]
      inB = i in state["IndexAssignmentsB"]
      if inA and inB:
        state["IndicesSummation"].append(i)
      else:
        printExit("invalid index %u (expected summation but not (inA and inB))" % i)
    # print index assignments
    if globalParameters["PrintIndexAssignments"]:
      print("IndicesFree:  %s" % state["IndicesFree"])
      print("IndicesBatch: %s" % state["IndicesBatch"])
      print("IndicesSum:   %s" % state["IndicesSummation"])
      print("IndexAssignmentsA:   %s" % state["IndexAssignmentsA"])
      print("IndexAssignmentsB:   %s" % state["IndexAssignmentsB"])
      print("NumIndicesC:  %s" % state["NumIndicesC"])

    for k in ('IndexAssignmentsA','IndexAssignmentsB'):
      if len(state[k]) != len(set(state[k])):
        printExit("duplicate index in %s=%s"% (k,state[k]))

    state["NumIndicesFree"] = len(state["IndicesFree"])
    state["NumIndicesBatch"] = len(state["IndicesBatch"])
    state["NumIndicesSummation"] = len(state["IndicesSummation"])
    if not state["AllowNoFreeDims"] and state["NumIndicesFree"] < 2 :
      printExit("Tensile requires >= 2 free indices or set AllowNoFreeDims; FreeIndices=%s."% state["IndicesFree"])

    # by default, unroll index will be the last/inner summation index
    state["IndexUnroll"] = state["IndicesSummation"][len(state["IndicesSummation"])-1]
    for i in range(0, len(state["IndexAssignmentsA"])):
      if state["IndexAssignmentsA"][i] == state["IndexUnroll"]:
        state["IndexUnrollA"] = i
        break
    for i in range(0, len(state["IndexAssignmentsB"])):
      if state["IndexAssignmentsB"][i] == state["IndexUnroll"]:
        state["IndexUnrollB"] = i
        break
    for i in range(0, len(state["IndexAssignmentsMetadata"])):
      if state["IndexAssignmentsMetadata"][i] == state["IndexUnroll"]:
        state["IndexUnrollM"] = i
        break
    #print2("IndexUnrollA: %u" % state["IndexUnrollA"])
    #print2("IndexUnrollB: %u" % state["IndexUnrollB"])

    # assign d0, d1
    if state["AllowNoFreeDims"]:
      dimList = state["IndicesFree"] + state["IndicesBatch"]
    else:
      dimList = state["IndicesFree"]
    state["Index01A"] = [i for i in state["IndexAssignmentsA"] if i in dimList][0]
    state["Index01B"] = [i for i in state["IndexAssignmentsB"] if i in dimList][0]
    #print2("Index01A: %u" % state["Index01A"])
    #print2("Index01B: %u" % state["Index01B"])
    # Store code is optimized for 0 as the fastest-moving in memory
    # whichever has lower stride in C (lower value), is 0, other is 1
    if state["Index01A"] < state["Index01B"]:
      state["Index0"]  = state["Index01A"]
      state["Index1"]  = state["Index01B"]
      state["Tensor0"] = 0
      state["Tensor1"] = 1
      state["TileA"] = 0
      state["TileB"] = 1
    else:
      state["Index0"]  = state["Index01B"]
      state["Index1"]  = state["Index01A"]
      state["Tensor0"] = 1
      state["Tensor1"] = 0
      state["TileA"] = 1
      state["TileB"] = 0

    # generalize transpose
    strideIdxA = state["IndexAssignmentsA"].index(state["Index01A"])
    strideIdxB = state["IndexAssignmentsB"].index(state["Index01B"])
    unrollIdxA = state["IndexAssignmentsA"].index(state["IndexUnroll"])
    unrollIdxB = state["IndexAssignmentsB"].index(state["IndexUnroll"])
    state["TLUA"] = strideIdxA < unrollIdxA
    state["TLUB"] = strideIdxB < unrollIdxB
    #state["TLUB"] = True # hack

    if globalParameters["PrintIndexAssignments"]:
      print("TLUA:  %s (stridePosA(%d) <? unrollIdxA(%d)" % \
			(state["TLUA"], strideIdxA, unrollIdxA))
      print("TLUB:  %s (stridePosB(%d) <? unrollIdxB(%d)" % \
	  		(state["TLUB"], strideIdxB, unrollIdxB))
      print("Index01A:  %s" % state["Index01A"])
      print("Index01B:  %s" % state["Index01B"])
    #unrollDimStrideGreaterThanTileDimStrideA = TLUA = !transA = fast
    #!unrollDimStrideLessThanTileDimStrideB   = TLUB =  transB = fast
    state["AssignedDerivedParameters"] = True

    if state["Sparse"]:
      state["Index01Metadata"] = [i for i in state["IndexAssignmentsMetadata"] if i in dimList][0]
      strideIdxM = state["IndexAssignmentsMetadata"].index(state["Index01Metadata"])
      unrollIdxM = state["IndexAssignmentsMetadata"].index(state["IndexUnroll"])
      state["TLUMetadata"] = strideIdxM < unrollIdxM
      if globalParameters["PrintIndexAssignments"]:
        print("TLUMetadata:  %s (stridePosM(%d) <? unrollIdxM(%d)" % \
          (state["TLUMetadata"], strideIdxM, unrollIdxM))
        print("Index01Metadata:  %s" % state["Index01Metadata"])

  ########################################
  def __str__(self):
    indexChars = INDEX_CHARS
    # C dimensions
    name = "C"
    for i in range(0, self["NumIndicesC"]):
      name += indexChars[i].lower()
    # A dimensions
    name += "_A"
    for i in self["IndexAssignmentsA"]:
      name += indexChars[i] if i in self["MirrorDimsA"] else indexChars[i].lower()
    if self["ComplexConjugateA"]:
      name += "C"
    # B dimensions
    name += "_B"
    for i in self["IndexAssignmentsB"]:
      name += indexChars[i] if i in self["MirrorDimsB"] else indexChars[i].lower()
    if self["ComplexConjugateB"]:
      name += "C"

    # DataTypes
    if self["DataType"] != self["DataTypeA"] or self["DataType"] != self["DataTypeB"]:
      name += "_"
      name += self["DataTypeA"].toChar() + self["DataTypeB"].toChar()
    name += "_"
    name += self["DataType"].toChar() # Type of A/B

    # Special condition for some newly supported kernels:
    #   HHS, HSS, BSS and I8II kernels, use a clearer naming _TiToTc_
    # TODO: Distinguish all kernels by _TiToTc_ to be more consistent with rocblas
    gemmType = (self["DataType"].toChar(),self["DestDataType"].toChar(),self["ComputeDataType"].toChar() )
    if gemmType in HPATypes:
      name += self["DestDataType"].toChar()    # Type of C/D
      name += self["ComputeDataType"].toChar() # Type of Alpha/Beta
      name += "_"

    if not self["F32XdlMathOp"].isSingle() and self["DataType"].isSingle():
      name += "_M"
      name += self["F32XdlMathOp"].toChar()
      name += "_"

    if self["SwizzleTensorA"]:
      name += "STA_"

    if self["SwizzleTensorB"]:
      name += "STB_"

    # Other
    if self["UseBeta"]: name += "B"
    if self["HighPrecisionAccumulate"] and not self["SilentHighPrecisionAccumulate"]: name += "H"
    if self["UseInitialStridesAB"]: name += "I"
    if self["UseInitialStridesCD"]: name += "Ic"
    if self["UseBias"]:
      name += "_Bias"
      if self["BiasDataTypeList"] != getBiasDataTypeListDefault(self):
        for i in self["BiasDataTypeList"]:
          name += i.toChar()
      if self["BiasSrc"] and self["Gradient"]: # Show bias src if gradient = True
        name += "_BiasSrc%s"%self["BiasSrc"]

    factorDim = max(self["UseScaleAlphaVec"], self["UseBias"])
    if factorDim > 1 :
        name += "_FD%s"%("N" if factorDim == 2 else "MN")

    if self["UseE"]:
      if self["Gradient"]:
        name += "_Grad%s"%self["DataTypeE"].toChar()
      else:
        name += "_Aux%s"%self["DataTypeE"].toChar() # Not showing aux types
    if self["OutputAmaxD"]:
      name += "_AmaxD"
    if self["Sparse"]:
      if self["Sparse"] == 2:
        name += "_SPB"
      else:
        name += "_SPA"

    # precision and other
    # name += "_SB" if self["StridedBatched"] else "_GB"
    if self["GroupedGemm"]:
      name += "_GG"
    else:
      name += "" if self["StridedBatched"] else "_GB" # legacy

    # Activation Naming
    if self["ActivationType"] != 'none':
      if self["ActivationType"] == 'all':
        name += "_A"
      elif self["ActivationType"] == 'hipblaslt_all':
        name += "_HA"
      else:
        name += "_%s"%str(self["ActivationType"]).upper()
      name += self["ActivationComputeDataType"].toChar()
    if self["ActivationNoGuard"]: name += "NG"

    if self["UseScaleAB"] == "Scalar": name += "_SAB"
    elif self["UseScaleAB"] == "Vector": name += "_SABV"
    if self["UseScaleCD"]: name += "_SCD"
    if self["UseScaleAlphaVec"]: name += "_SAV"

    if self["SupportUserArgs"]: name += "_UserArgs"

    return name

  def keys(self):
    return list(self.state.keys())
  def __len__(self):
    return len(self.state)
  def __iter__(self):
    return iter(self.state)
  def __getitem__(self, key):
    return self.state[key]
  def __setitem__(self, key, value):
    self.state[key] = value
  def __repr__(self):
    return self.__str__()
  def getAttributes(self):
    return self.state
  def __hash__(self):
    return hash(str(self))
  def __eq__(self, other):
    return isinstance(other, ProblemType) and self.getAttributes() == other.getAttributes()
  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result

  def get(self, key, default=None):
    try:
      return self.state[key]
    except:
      return default



################################################################################
# ProblemSizeRange
################################################################################
class ProblemSizeRange:

  ########################################
  def __init__(self, problemType, config):
    self.totalIndices = 1+max(problemType["IndexAssignmentsA"]) + problemType["NumIndicesLD"]
    if len(config) < self.totalIndices:
      for i in range(len(config), self.totalIndices):
        if i < self.totalIndices - problemType["NumIndicesLD"]:
          config.append(0)
        else:
          config.append([0])

    self.indexMax = []
    self.indexIsSized = []
    self.indicesSized = []
    self.indicesMapped = []
    for i in range(0, self.totalIndices):
      dim = deepcopy(config[i])
      if isinstance(dim, list):
        if len(dim) == 1:
          self.indicesSized.append([dim[0], 1, 0, dim[0]])
        elif len(dim) == 2:
          self.indicesSized.append([dim[0], dim[0], 0, dim[1]])
        elif len(dim) == 3:
          self.indicesSized.append([dim[0], dim[1], 0, dim[2]])
        elif len(dim) == 4:
          self.indicesSized.append([dim[0], dim[1], dim[2], dim[3]])
        else:
          printExit("dimension[%u] config (%s) has %u descriptors rather than 1-4."
              % ( i, dim, len(dim) ))
        self.indexIsSized.append(True)
        self.indexMax.append(self.indicesSized[len(self.indicesSized)-1][3])

      elif isinstance(dim, int):
        self.indicesMapped.append(dim)
        self.indexIsSized.append(False)
        self.indexMax.append(self.indicesSized[self.indicesMapped[ \
            len(self.indicesMapped)-1]][3])

    # max num elements in each tensor
    self.maxNumElements = [ 1, 1, 1 ]
    for i in range(0, problemType["NumIndicesC"]):
      self.maxNumElements[0] *= self.indexMax[i]
    for i in problemType["IndexAssignmentsA"]:
      self.maxNumElements[1] *= self.indexMax[i]
    for i in problemType["IndexAssignmentsB"]:
      self.maxNumElements[2] *= self.indexMax[i]

    self.totalProblemSizes = 1
    self.numProblemSizes = [] # per index
    self.problemSizeToIndex = []
    self.problemIndexToSize = []
    sizedIdx = 0
    for i in range(0, len(self.indexIsSized)):
      self.problemSizeToIndex.append({})
      self.problemIndexToSize.append({})
      if self.indexIsSized[i]:
        self.numProblemSizes.append(0)
        index = self.indicesSized[sizedIdx]
        sizedIdx += 1
        currentSize = index[0]
        currentIncrement = index[1]
        while currentSize <= index[3]:
          currentSize += currentIncrement
          currentIncrement += index[2]
          self.numProblemSizes[i] += 1
      else:
        self.numProblemSizes.append(1)
      self.totalProblemSizes *= self.numProblemSizes[i]

    ########################################
    # enumerate problem sizes
    currentSizedIndexSizes = []
    currentSizedIndexIncrements = []
    for i in range(0, len(self.indicesSized)):
      currentSizedIndexSizes.append(self.indicesSized[i][0])
      currentSizedIndexIncrements.append(self.indicesSized[i][1])

    # iterate over all problem sizes
    self.problemSizes = []
    moreProblemSizes = True
    problemIdx = 0
    problemSize = [0]*self.totalIndices
    while moreProblemSizes:
      #/ convert current sized and mapped indices to full sizes
      currentSizedIdx = 0
      currentMappedIdx = 0
      for i in range(0, self.totalIndices):
        if self.indexIsSized[i]:
          problemSize[i] = currentSizedIndexSizes[currentSizedIdx]
          currentSizedIdx+=1
        else:
          problemSize[i] = problemSize[self.indicesMapped[currentMappedIdx]]
          currentMappedIdx+=1
      self.problemSizes.append(tuple(problemSize))

      #/ increment sizes for next benchmark
      currentSizedIndexSizes[0] += currentSizedIndexIncrements[0]
      currentSizedIndexIncrements[0] += self.indicesSized[0][2]
      for i in range(1, len(self.indicesSized)+1):
        # if prior index past max, reset to min and increment next index
        if currentSizedIndexSizes[i-1] > self.indicesSized[i-1][3]:
          #/ reset prior index
          currentSizedIndexSizes[i-1] = self.indicesSized[i-1][0]
          currentSizedIndexIncrements[i-1] = self.indicesSized[i-1][1]
          # increment next index
          if i >= len(self.indicesSized):
            moreProblemSizes = False
          else:
            currentSizedIndexSizes[i] += currentSizedIndexIncrements[i]
            currentSizedIndexIncrements[i] += self.indicesSized[i][2]

      problemIdx+=1

  ########################################
  # YAML format
  def __str__(self):
    state = "[ "
    sizedIdx = 0
    mappedIdx = 0
    for i in range(0, len(self.indexIsSized)):
      if self.indexIsSized[i]:
        indices = self.indicesSized[sizedIdx]
        state += "[ %u, %u, %u, %u ]" \
            % (indices[0], indices[1], indices[2], indices[3])
        sizedIdx += 1
      else:
        indices = self.indicesSized[self.indicesMapped[mappedIdx]]
        state += str(self.indicesMapped[mappedIdx])
        mappedIdx += 1
      if i < len(self.indexIsSized)-1:
        state += ", "
    state += " ]"
    return state

class Problem:
  """ Problem sizes, strides, padding and other info"""
  def __init__(self, sizes=None, stridesA=None, stridesB=None, stridesC=None, stridesD=None, count=None):
    self.sizes = tuple(sizes) if sizes else None
    self.stridesA = tuple(stridesA) if stridesA else None
    self.stridesB = tuple(stridesB) if stridesB else None
    self.stridesC = tuple(stridesC) if stridesC else None
    self.stridesD = tuple(stridesD) if stridesD else None

    self.count = count

  def __str__(self):
    rv= "{ sizes:" + str(list(self.sizes))
    if self.stridesA:
      rv += ", stridesA:" + str(list(self.stridesA))
    if self.stridesB:
      rv += ", stridesB:" + str(list(self.stridesB))
    if self.stridesC:
      rv += ", stridesC:" + str(list(self.stridesC))
    if self.stridesD:
      rv += ", stridesD:" + str(list(self.stridesD))
    rv += " }"
    return rv

class ExactList(Problem):
  def __init__(self, e, problemType):
    if len(e) == problemType["TotalIndices"]:
      if -1 in e:
        printExit("ExactSize %s contains -1" % (e))
      if problemType["OperationType"] == "GEMM":
        e += [-1, -1, -1, -1]
        e = ExactList.convertLeadingDims(problemType, tuple(e))
      sizes=e

    elif len(e) == (problemType["TotalIndices"] + problemType["NumIndicesLD"]):
      sizes = ExactList.convertLeadingDims(problemType, tuple(e))
    else:
      printExit("ExactSize %s doesn't match indices of ProblemType %s, totalIndices=%d, len e=%d, NumIndicesLD = %d" \
          % (e, problemType, problemType["TotalIndices"], len(e), problemType["NumIndicesLD"]) )

    # TODO- pass strides here, remove calls to convertLeadingDims
    Problem.__init__(self, sizes=sizes)

  def __str__(self):
    return str(list(self.sizes))

  @staticmethod
  def convertLeadingDims(problemType, problemSize, stridesA = None, stridesB = None, stridesC = None, stridesD = None):
    # FIXME-problem: refactor to eliminate max, pass strides in strideB parm rather than hacked
    # onto the end of the sizes list
    predStridesD = stridesD is not None and stridesD[1] != -1
    predStridesC = stridesC is not None and stridesC[1] != -1
    predStridesA = stridesA is not None and stridesA[1] != -1
    predStridesB = stridesB is not None and stridesB[1] != -1
    return problemSize[:problemType["NumIndicesC"]+1] + \
           (max(problemSize[0], problemSize[problemType["IndexAssignmentsLD"][0]]) if not predStridesD else stridesD[1], ) + \
           (max(problemSize[0], problemSize[problemType["IndexAssignmentsLD"][1]]) if not predStridesC else stridesC[1], ) + \
           (max(problemSize[problemType["IndexAssignmentsLD"][2]],
                problemSize[problemType["IndexAssignmentsA"][0]]) if not predStridesA else stridesA[1], ) + \
           (max(problemSize[problemType["IndexAssignmentsLD"][3]],
                problemSize[problemType["IndexAssignmentsB"][0]]) if not predStridesB else stridesB[1], )


class ExactDict(Problem):
  AllowedFields = [ 'count', 'sizes', 'stridesA', 'stridesB', 'stridesC', 'stridesD' ]

  def __init__(self, e, problemType):
    Problem.__init__(self)

    for f in e:
      if f in ExactDict.AllowedFields:
        setattr(self, f, e[f])
      else:
        raise RuntimeError ("specified field '%s' is not a valid Exact dict field"%f)

    if problemType:
      if "OperationType" in problemType and problemType["OperationType"] == "GEMM":
        sizesTuple = tuple(self.sizes + [-1, -1, -1, -1])
        self.sizes = ExactList.convertLeadingDims(problemType, sizesTuple, self.stridesA, self.stridesB, self.stridesC, self.stridesD)

    if problemType:
      if "OperationType" in problemType and problemType["OperationType"] == "GEMM":
        if len(self.sizes) != (problemType["TotalIndices"] + problemType["NumIndicesLD"]):
        # FIXME-ExactDict size descriptor still (but preferrably not so) uses 8-tuple for GEMM problems
          raise RuntimeError ("specified size=%s does not have enough indices for problem (expected %d, got %d)" \
                % (self.sizes, problemType["TotalIndices"]+problemType["NumIndicesLD"], len(self.sizes)))
      elif len(self.sizes) != problemType["TotalIndices"]:
        raise RuntimeError ("specified size=%s does not have enough indices for problem (expected %d, got %d)" \
                % (self.sizes, problemType["TotalIndices"], len(self.sizes)))


################################################################################
# ProblemSizes
################################################################################
"""
Adapter class for class `ProblemSizes`. It satisfies the implicit usage requirement
of ClientWriter.writeClientConfig() by converting ExactLogic to list of `Problem` objects
"""
class ProblemSizesMock:
  def __init__(self, exactLogic):
    self.problems = [Problem(problem) for problem, solution in exactLogic]

class ProblemSizesMockDummy:
  def __init__(self):
    self.problems = [Problem(sizes=[128, 128, 1, 512])]

class ProblemSizes:

  ########################################
  def __init__(self, problemType, config):
    self.problemType = problemType
    self.ranges = []
    self.exacts = []
    self.minStrides = None
    if config:
      for dictionary in config:
        for sizeTypeKey in dictionary:
          #print ("PROBLEM parsed:", sizeTypeKey, dictionary[sizeTypeKey])
          if sizeTypeKey == "Range":
            psr = ProblemSizeRange(problemType, dictionary[sizeTypeKey])
            self.ranges.append( psr )
          elif sizeTypeKey == "Exact":
            e= dictionary[sizeTypeKey]
            if isinstance(e,list):
              self.exacts.append(ExactList(e, problemType))
            elif isinstance(e,dict):
              self.exacts.append(ExactDict(e, problemType))
            else:
              printExit("Unsupported Exact type==%s"%type(e))
          elif sizeTypeKey == "MinStride":
            e = dictionary[sizeTypeKey]
            if len(e) != problemType["TotalIndices"]:
              printExit("MinStride %s doesn't match indices of ProblemType %s" \
                  % (e, problemType) )
            if self.minStrides:
              printExit("Only one MinStride command is allowed in a ProblemsSizes definition.  Previous minStrides:%s, New minstride:%s" \
                  % (self.minStrides, e) )

            self.minStrides=(tuple(e))
          else:
            printExit("ProblemSize Type %s not supported"%sizeTypeKey)

    if not self.minStrides:
      # set harmless default mins of 0
      self.minStrides = ([0]* problemType["TotalIndices"])

    # not the ideal spot, but convert leading dims that are below the minimum size
    if problemType["OperationType"] == "GEMM":
      for i in range(0, len(self.ranges)):
        self.ranges[i].problemSizes[:] = \
          [ExactList.convertLeadingDims(self.problemType, problemSize) for problemSize in self.ranges[i].problemSizes]

    self.problems = OrderedDict()
    for sizeRange in self.ranges:
      for rangeSize in sizeRange.problemSizes:
        self.problems.update({Problem(rangeSize) : 1})
    for e in self.exacts:
        self.problems.update({e : 1})
    self.problems =  list(self.problems.keys())
    self.totalProblemSizes = len(self.problems)

    # max sizes
    self.maxD = 0
    self.maxC = 0
    self.maxA = 0
    self.maxB = 0
    for problem in self.problems:
      problemSize = problem.sizes # FIXME-problem.   This should use problem.strides*

      sizeLdd = problemSize[self.problemType["IndexAssignmentsLD"][0]] if problemType["OperationType"] == "GEMM" else problemSize[0]
      sizeD = max(self.minStrides[0], sizeLdd)
      for i in range(1, problemType["NumIndicesC"]):
        sizeD *= max(self.minStrides[i], problemSize[i])

      sizeLdc = problemSize[self.problemType["IndexAssignmentsLD"][1]] if problemType["OperationType"] == "GEMM" else problemSize[0]
      sizeC = max(self.minStrides[0], sizeLdc)
      for i in range(1, problemType["NumIndicesC"]):
        sizeC *= max(self.minStrides[i], problemSize[i])

      sizeLda = problemSize[self.problemType["IndexAssignmentsLD"][2]] \
                if problemType["OperationType"] == "GEMM" \
                else problemSize[self.problemType["IndexAssignmentsA"][0]]
      sizeA = max(self.minStrides[self.problemType["IndexAssignmentsA"][0]], sizeLda)
      for i in self.problemType["IndexAssignmentsA"][1:]:
        sizeA *= max(self.minStrides[i], problemSize[i])

      sizeLdb = problemSize[self.problemType["IndexAssignmentsLD"][3]] \
                if problemType["OperationType"] == "GEMM" \
                else problemSize[self.problemType["IndexAssignmentsB"][0]]
      sizeB = max(self.minStrides[self.problemType["IndexAssignmentsB"][0]], sizeLdb)
      for i in self.problemType["IndexAssignmentsB"][1:]:
        sizeB *= max(self.minStrides[i], problemSize[i])

      self.maxD = max(self.maxD, sizeD)
      self.maxC = max(self.maxC, sizeC)
      self.maxA = max(self.maxA, sizeA)
      self.maxB = max(self.maxB, sizeB)

  def __str__(self):
    s = "ProblemSizes\n"
    for sizeRange in self.ranges:
      s += "  %s" % sizeRange
    return s

################################################################################
# Factor Type
################################################################################

class FactorDimArgs:

  ########################################
  def __init__(self, problemType, config):
    self.factorDims = []
    self.totalProblemSizes = 0
    if problemType["UseScaleAlphaVec"] or problemType["UseBias"]:
      for fdim in config:
        dim = int(fdim)
        if dim not in [0, 1]:
          printWarning("Factor Dim: must be 0 or 1, current is %s."%(dim))
        self.factorDims.append(dim)
      self.totalProblemSizes = len(self.factorDims)

  def __str__(self):
    s = "FactorDimArgs\n"
    return s

################################################################################
# Bias Type
################################################################################

def getBiasDataTypeListDefault(problem: ProblemType) -> List[DataType]:
  bList = []
  for d in ["DataType", "ComputeDataType", "DestDataType"]:
    dtype = DataType(problem[d])
    # filter out int8/f8/b8, because it is not supported by bias datatype
    # TODO
    if not dtype.isInt8() and not dtype.is8bitFloat():
      bList.append(dtype)

  biasDataTypeList = list(set(bList))
  biasDataTypeList.sort() # Make name unique
  return biasDataTypeList

class BiasTypeArgs:

  ########################################
  def __init__(self, problemType, config):
    self.biasTypes = []
    self.totalProblemSizes = 0
    if problemType["UseBias"]:
      for btype in config:
        datatype = DataType(btype)
        if datatype not in problemType["BiasDataTypeList"]:
          printWarning("Datatype: %s not support in this kernel (%s)"%(datatype, str(problemType["BiasDataTypeList"])))
        self.biasTypes.append(datatype)

      if not self.biasTypes:
        printExit("Must provide a bias type in benchmark parameters if UseBias is set to True.")

      self.totalProblemSizes = len(self.biasTypes)

  def __str__(self):
    s = "BiasTypesArgs\n"
    return s

################################################################################
# Activation
################################################################################

class activationSetting:
  def __init__(self):
    self.activationEnum = ""
class ActivationArgs:

  ########################################
  def __init__(self, problemType, config):
    self.settingList = []
    self.totalProblemSizes = 0
    if problemType["ActivationType"] == 'none':
      return
    if config:
      for settings in config:
        actSetting = activationSetting()
        for dictionary in settings:
          for sizeTypeKey in dictionary:
            if sizeTypeKey == "Enum":
              actSetting.activationEnum = ActivationType(dictionary[sizeTypeKey])
        if problemType["ActivationType"] in ['all', 'hipblaslt_all']:
          if (not actSetting.activationEnum):
            printExit("Must provide an activation enum if Activation is set to True.")
        else:
          actSetting.activationEnum = problemType["ActivationType"]
        self.settingList.append(actSetting)
        self.totalProblemSizes += 1
    if (problemType["ActivationType"] == 'all') and (not self.settingList):
        printExit("Must provide an activation enum in benchmark parameters if Activation is set to True.")
  def __str__(self):
    s = "ActivationArgs\n"
    return s

# kds is class Solution or class Kernel
# All free dims are packed
def isPackedIndex(ks, index):
  problemType = ks["ProblemType"]
  return index in problemType["IndicesFree"]

def isExtractableIndex(ks, index, tc='x'):
  xA = index in ks['PackedC0IndicesX'][:-1]
  xB = index in ks['PackedC1IndicesX'][:-1]
  if tc=='A':
    return xA
  elif tc=='B':
    return xB
  else:
    return xA or xB

################################################################################
# Solution
################################################################################
class Solution(collections.abc.Mapping):

  ########################################
  def __init__(self, config, cxxCompiler: str, srcName: str = ""):
    self._name = None
    self.cxxCompiler = cxxCompiler
    self.srcName = srcName
    config = config

    self._state = {}
    # problem type
    if "ProblemType" in config:
      self["ProblemType"] = ProblemType(config["ProblemType"])
    else:
      self["ProblemType"] = ProblemType(defaultProblemType)

    if "InternalSupportParams" in config:
      self["InternalSupportParams"] = {}
      for key in defaultInternalSupportParams:
        assignParameterWithDefault(self["InternalSupportParams"], key, config["InternalSupportParams"], defaultInternalSupportParams)
    else:
      self["InternalSupportParams"] = defaultInternalSupportParams


    # assign parameters with defaults
    for key in defaultSolution:
      assignParameterWithDefault(self._state, key, config, defaultSolution)
    if 'ISA' not in self._state:
      if 'ISA' in config:
        if not globalParameters["AsmCaps"][tuple(config['ISA'])]["SupportedISA"]:
          defaultIsa = [9,0,0]
          print("warning: ISA:", config['ISA'], " is not supported; overriding with ", defaultIsa)
          self._state['ISA'] = defaultIsa
        else:
          self._state['ISA'] = config['ISA']
      else:
        # Assembly by default
        self._state['ISA'] = list(globalParameters["CurrentISA"])
        if 'KernelLanguage' in config:
          if config['KernelLanguage'] != 'Assembly':
            self._state['ISA'] = [0,0,0]

    if "CodeObjectVersion" not in self._state:
      if "CodeObjectVersion" in config:
        self._state["CodeObjectVersion"] = str(config["CodeObjectVersion"])
      else:
        self._state["CodeObjectVersion"] = str(globalParameters["CodeObjectVersion"])
    # assign parameters without defaults
    for key in config:
      if (key != "ProblemType" or key != "InternalSupportParams") and key not in self._state:
        self._state[key] = config[key]
    self["Valid"] = True
    # this could prevent OriginalSolution from re-assigning the parameters, save lots of time
    if "AssignedProblemIndependentDerivedParameters" not in self._state:
      self["AssignedProblemIndependentDerivedParameters"] = False
    if "AssignedDerivedParameters" not in self._state:
      self["AssignedDerivedParameters"] = False
    Solution.assignDerivedParameters(self._state)
    self._name = config["CustomKernelName"] if isCustomKernelConfig(config) else None
    self.initHelperKernelObjects()

  # these keys are copied from ProblemType to internal that may be overridden
  InternalKeys = ["UseSgprForGRO","VectorStore"]


  ########################################
  # get a list of kernel parameters for this solution
  def getKernels(self):
    kernel = self
    kernel._state.update({"Kernel": True})
    kernels = []
    kernels.append(kernel)
    return kernels


  ########################################
  # create Helper Kernels
  def initHelperKernelObjects(self):
    self.initBetaOnlyKernelObjects()
    self.initConversionKernelObjects()
    self.initActivationEnumHeaderObjects()
    self.initActivationFunctionObjects()
    self.initActivationOnlyKernelObjects()
    self.initReductionKernelObjects()

  ########################################
  # create BetaOnly Kernels
  def initBetaOnlyKernelObjects(self):
    self.betaOnlyKernelObjects = []
    if self["GlobalSplitU"] > 1 or (self["StreamK"] > 0 and self["StreamKAtomic"] == 1):
      if self["ProblemType"]["UseBias"]:
        for btype in self["ProblemType"]["BiasDataTypeList"]:
          state = {}
          state["ProblemType"] = deepcopy(self["ProblemType"])
          state["ProblemType"]["GroupedGemm"] = False
          state["ProblemType"]["BiasDataTypeList"] = []
          state["ProblemType"]["BiasDataType"] = deepcopy(btype)
          state["KernelLanguage"] = "Source"
          state["_GlobalAccumulation"] = self["_GlobalAccumulation"]
          self.betaOnlyKernelObjects.append(KernelWriterBetaOnly(state))
      else:
        state = {}
        state["ProblemType"] = deepcopy(self["ProblemType"])
        state["ProblemType"]["GroupedGemm"] = False
        state["KernelLanguage"] = "Source"
        state["_GlobalAccumulation"] = self["_GlobalAccumulation"]
        self.betaOnlyKernelObjects.append(KernelWriterBetaOnly(state))


  ########################################
  # create Conversion Kernels
  def initConversionKernelObjects(self):
    self.conversionKernelObjects = []
    load_vector_width = [1, 2] if self["ProblemType"]["DataType"].isDouble() else [1, 2, 4]
    genPGRPostKernels = True
    gsuList = [internalParameters["GlobalSplitUPGR"]]
    if self["GlobalSplitUAlgorithm"] == "SingleBuffer":
      genPGRPostKernels = False
      gsuList = [1]
    elif self["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel":
      return
    for vw in load_vector_width:
      for globalSplitU in gsuList:
        unrollOnly = False if globalSplitU == internalParameters["GlobalSplitUPGR"] else True
        if self["ProblemType"]["UseBias"]:
          typeList = self["ProblemType"]["BiasDataTypeList"]
          if self["ProblemType"]["Gradient"]:
            # If gradient + bias D, generates a normal GSU kernel for bias D = nullptr case
            state = {}
            state["ProblemType"] = deepcopy(self["ProblemType"])
            state["ProblemType"]["GroupedGemm"] = False
            state["ProblemType"]["UseBias"] = 0
            state["GenPGRPostKernels"] = genPGRPostKernels
            state["KernelLanguage"] = "Source"
            state["GlobalSplitU"] = globalSplitU
            state["UnrollOnly"] = unrollOnly
            state["_GlobalAccumulation"] = self["_GlobalAccumulation"]
            state["ActivationFused"] = self["ActivationFused"]
            self.conversionKernelObjects.append(KernelWriterConversion(state, vw))
          for btype in typeList:
            state = {}
            state["ProblemType"] = deepcopy(self["ProblemType"])
            state["ProblemType"]["GroupedGemm"] = False
            state["ProblemType"]["BiasDataTypeList"] = []
            state["ProblemType"]["BiasDataType"] = deepcopy(btype)
            state["GenPGRPostKernels"] = genPGRPostKernels
            state["KernelLanguage"] = "Source"
            state["GlobalSplitU"] = globalSplitU
            state["UnrollOnly"] = unrollOnly
            state["_GlobalAccumulation"] = self["_GlobalAccumulation"]
            state["ActivationFused"] = self["ActivationFused"]
            self.conversionKernelObjects.append(KernelWriterConversion(state, vw))
        else:
          state = {}
          state["ProblemType"] = deepcopy(self["ProblemType"])
          state["ProblemType"]["GroupedGemm"] = False
          state["GenPGRPostKernels"] = genPGRPostKernels
          state["KernelLanguage"] = "Source"
          state["GlobalSplitU"] = globalSplitU
          state["UnrollOnly"] = unrollOnly
          state["_GlobalAccumulation"] = self["_GlobalAccumulation"]
          state["ActivationFused"] = self["ActivationFused"]
          self.conversionKernelObjects.append(KernelWriterConversion(state, vw))

  def initActivationEnumHeaderObjects(self):
    self.activationEnumHeaderObjects = []
    if self["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all']:
      state = {}
      state["ProblemType"] = deepcopy(self["ProblemType"])
      state["ProblemType"]["GroupedGemm"] = False
      state["KernelLanguage"] = "Source"
      self.activationEnumHeaderObjects.append(KernelWriterActivationEnumHeader(state))

  def initActivationFunctionObjects(self):
    self.activationFunctionObjects = []
    if self["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all']:
      state = {}
      state["ProblemType"] = deepcopy(self["ProblemType"])
      state["ProblemType"]["GroupedGemm"] = False
      state["KernelLanguage"] = "Source"
      state["Kernel"] = {"WavefrontSize": self["WavefrontSize"], "ISA": tuple(self["ISA"])}
      self.activationFunctionObjects.append(KernelWriterActivationFunction(state, self.cxxCompiler))

  def initActivationOnlyKernelObjects(self):
    self.activationOnlyKernelObjects = []
    if (self["ActivationFused"] == False) and (self["ProblemType"]["ActivationType"] != 'none') :
      state = {}
      state["ProblemType"] = deepcopy(self["ProblemType"])
      state["ProblemType"]["GroupedGemm"] = False
      state["ProblemType"]["UseBias"] = 0
      state["ProblemType"]["BiasDataTypeList"] = []
      state["KernelLanguage"] = "Source"
      state["_GlobalAccumulation"] = self["_GlobalAccumulation"]
      state["ActivationFused"] = self["ActivationFused"]
      self.activationOnlyKernelObjects.append(KernelWriterActivationOnly(state))

  def initReductionKernelObjects(self):
    self.reductionKernelObjects = []
    if self["ProblemType"]["Gradient"] and self["ProblemType"]["UseBias"]:
      for btype in self["ProblemType"]["BiasDataTypeList"]:
        state = {}
        state["ProblemType"] = deepcopy(self["ProblemType"])
        state["ProblemType"]["GroupedGemm"] = False
        state["ProblemType"]["BiasDataTypeList"] = []
        state["ProblemType"]["BiasDataType"] = deepcopy(btype)
        self.reductionKernelObjects.append(KernelWriterReduction(state))

  ########################################
  # get Helper Kernels
  def getHelperKernelObjects(self):
    return self.activationEnumHeaderObjects + self.activationFunctionObjects + \
           self.betaOnlyKernelObjects + self.conversionKernelObjects + \
           self.activationOnlyKernelObjects + self.reductionKernelObjects


  ########################################
  # get Helper Kernels
  def getKernelBetaOlnyObjects(self):
    return self.betaOnlyKernelObjects


  ########################################
  # get Helper Kernels
  def getKernelConversionObjects(self):
    return self.conversionKernelObjects

  @staticmethod
  def getMIOutputInfo(state):
    outputVectorWidth = 4
    RegsPerOut = 1

    isa = tuple(state["ISA"])
    if globalParameters["AsmCaps"][isa]['HasMFMA']:
      if state["ProblemType"]["DataType"].MIOutputTypeNameAbbrev() == 'f64':
        outputVectorWidth, RegsPerOut = 1, 2
      else:
        outputVectorWidth, RegsPerOut = 4, 1
    elif globalParameters["AsmCaps"][isa]['HasWMMA_V1']:
        outputVectorWidth, RegsPerOut = 1, 1
    elif globalParameters["AsmCaps"][isa]['HasWMMA_V2']:
        outputVectorWidth, RegsPerOut = 8, 1
    else:
      print("WARNING: unexpect code flow")

    return outputVectorWidth, RegsPerOut

  ########################################
  # assign tile sizes
  @staticmethod
  def assignProblemIndependentDerivedParameters(state):

    if globalParameters["NewClient"] != 2:
      print("WARNING: Old client deprecated, NewClient parameter being set to 2.")
      globalParameters["NewClient"] = 2

    if "AssignedProblemIndependentDerivedParameters" in state:
      if state["AssignedProblemIndependentDerivedParameters"]:
        return
    state["AssignedProblemIndependentDerivedParameters"] = False
    if "Valid" not in state:
      state["Valid"] = True

    if (not state["ProblemType"]["StridedBatched"]) and (not state["ProblemType"]['Batched']):
      reject(state, "General Batched GEMM only support Batched Problem")

    if (not state["ProblemType"]["StridedBatched"]) and (state["ProblemType"]["OperationType"] != 'GEMM'):
      reject(state, "General Batched GEMM only support GEMM OperationType")

    Solution.MatrixInstructionToMIParameters(state)
    EnableMatrixInstruction = state["EnableMatrixInstruction"] if "EnableMatrixInstruction" in state else None
    if EnableMatrixInstruction == None:
      if  ("MIBlock" in state and len(state["MIBlock"]) == 6) \
          and ("MIWaveGroup" in state and len(state["MIWaveGroup"]) == 2) \
          and ("MIWaveTile" in state and len(state["MIWaveTile"]) == 2):
        EnableMatrixInstruction = True
      elif ("WorkGroup" in state and len(state["WorkGroup"]) == 3) \
          and ("ThreadTile" in state and len(state["ThreadTile"]) == 2) :
        EnableMatrixInstruction = False
      else:
        reject(state, "EnableMatrixInstruction undetermined")

    if EnableMatrixInstruction == True:
      state["MatrixInstM"]         = state["MIBlock"][0]
      state["MatrixInstN"]         = state["MIBlock"][1]
      state["MatrixInstK"]         = state["MIBlock"][2]
      state["MatrixInstB"]         = state["MIBlock"][3]
      state["MatrixInstBM"]        = state["MIBlock"][4]
      state["MatrixInstBN"]        = state["MIBlock"][5]

      state["LocalSplitU"] = state["WorkGroup"][2]
      state["NumWaveSplitK"] = 1
      
      state["MIOutputVectorWidth"], state["MIRegPerOut"] = Solution.getMIOutputInfo(state)

      if state["MatrixInstM"] == 4:
        state["ThreadTile0"] = state["MIWaveTile"][0] * state["MIOutputVectorWidth"]
        state["ThreadTile1"] = state["MIWaveTile"][1]
        state["SubGroup0"]   = state["MIWaveGroup"][0] * state["MatrixInstM"] * state["MatrixInstBM"] // state["MIOutputVectorWidth"]
        state["SubGroup1"]   = state["MIWaveGroup"][1] * state["MatrixInstN"] * state["MatrixInstBN"]
      else:
        state["ThreadTile0"] = state["MatrixInstBM"] * state["MIWaveTile"][0] * (state["MatrixInstM"] * state["MatrixInstN"] // state["WavefrontSize"])
        state["ThreadTile1"] = state["MatrixInstBN"] * state["MIWaveTile"][1]
        state["SubGroup0"]   = state["MIWaveGroup"][0] * (state["WavefrontSize"] // state["MatrixInstN"])
        state["SubGroup1"]   = state["MIWaveGroup"][1] * state["MatrixInstN"]

      #for the old Logic yaml file which does not contain keys: MIInputPerThreadA/B
      if not "MIInputPerThreadA" in state:
        state["MIInputPerThreadA"] = state["MIInputPerThread"]
        state["MIInputPerThreadB"] = state["MIInputPerThread"]

    elif EnableMatrixInstruction == False:
      state["ThreadTile0"] = state["ThreadTile"][0]
      state["ThreadTile1"] = state["ThreadTile"][1]

      state["SubGroup0"]   = state["WorkGroup"][0]
      state["SubGroup1"]   = state["WorkGroup"][1]
      # choose between 2 split modes
      state["LocalSplitU"] = 1 if state["WaveSplitK"] else state["WorkGroup"][2]
      state["NumWaveSplitK"]  = state["WorkGroup"][2] if state["WaveSplitK"] else 1

    if "SubGroup0" in state and "SubGroup1" in state and "LocalSplitU" in state and "NumWaveSplitK" in state:
      state["NumThreads"] = state["SubGroup0"] * state["SubGroup1"] * state["LocalSplitU"] * state["NumWaveSplitK"]
      if (state["NumThreads"] % state['WavefrontSize']) != 0:
        reject(state, f"size of WorkGroup {state['NumThreads']} should be multiple of WavefrontSize {state['WavefrontSize']}")

    # macro tile sizes
    if "SubGroup0" in state and "ThreadTile0" in state:
      state["MacroTile0"] = state["SubGroup0"]*state["ThreadTile0"]
    if "SubGroup1" in state and "ThreadTile1" in state:
      state["MacroTile1"] = state["SubGroup1"]*state["ThreadTile1"]
    if "MacroTile" in state:
      if state["MacroTile0"] != state["MacroTile"][0] \
          or state["MacroTile1"] != state["MacroTile"][1]:
        reject(state, "MacroTile mismatch")

    # dot2: currently only support fp16 with HPA on gfx942
    state["UseDotInstruction"] = (not state["EnableMatrixInstruction"]) and state["ProblemType"]["DataType"].isHalf() \
      and state["ProblemType"]["HighPrecisionAccumulate"] and (globalParameters["CurrentISA"] == (9,4,2))
    if state["UseDotInstruction"]:
      # need modification for dot4 or dot8
      state["NumDotElements"] = 2
    if not state["UseDotInstruction"] and state["WaveSplitK"]:
      reject(state, "WaveSplitK currently only support dot2 kernel.")
      return

    # tail loop optimization
    state["tailLoopOptA"] = True
    state["tailLoopOptB"] = True

    if (tuple(state["ISA"]) != (9, 4, 2)) or \
       (state["ProblemType"]["Sparse"]) or \
       (state["UseDotInstruction"]):
      state["tailLoopOptA"] = False
      state["tailLoopOptB"] = False
    if (state["DirectToVgprA"]):
      state["tailLoopOptA"] = False
    if (state["DirectToVgprB"]):
      state["tailLoopOptB"] = False

    # reorder globalread instructions if dtv and TN cases. (along coalesced dim)
    if state["ScheduleIterAlg"] == 3:
      state["reorderGRInstForDTVA"] = True if state["ProblemType"]["TransposeA"] and \
                                              state["DirectToVgprA"] and \
                                              not state["ProblemType"]["SwizzleTensorA"] else False
      state["reorderGRInstForDTVB"] = True if not state["ProblemType"]["TransposeB"] and \
                                              state["DirectToVgprB"] and \
                                              not state["ProblemType"]["SwizzleTensorB"] else False
    else:
      state["reorderGRInstForDTVA"] = False
      state["reorderGRInstForDTVB"] = False

    # done
    state["AssignedProblemIndependentDerivedParameters"] = True

  ########################################
  # This is the "classic" algorithm which requires that each threads load the same number of bytes
  # Called with tc=A and then with tc=B
  # totalVectors is totalElements/GRVW, this is #vectors loaded by the LoadTile
  # Reduces the GlobalReadVectorWidth if necessary if each thread has a small amount of work to do.
  # Output from this function:
  #  state[GlobalReadVectorWidth*]
  #  state[NumLoads*] # only used in SolutionStructs, with classic alg
  @staticmethod
  def setGlobalReadVectorWidth(state, tc, totalVectors, grvw):
    validDepthU = True
    if grvw not in [1,2,4,8,16,32]:
      validDepthU = False
    if totalVectors % state["NumThreads"] != 0:
      reject(None, "totalVectors%s %u %% NumThreads %u != 0" \
          % (tc, totalVectors, state["NumThreads"]))
      validDepthU = False

    state["GlobalReadVectorWidth%s"%tc] = grvw

    # NumLoads is NOT used on the fractional path
    # NumLoads is number of vector loads per-thread
    state["NumLoads%s"%tc] = totalVectors // state["NumThreads"]
    #print "result: ", pvar(state, "GlobalReadVectorWidth%s"%tc), \
    #        pvar(state, "NumLoads%s"%tc)

    return validDepthU

  ########################################
  # Sets the Global Read Tile dims (para, perp)
  # This information controls which threads read which addresses from global mem)
  # Output from this function:
  #   state[NumLoadsCoalescedA]
  #   state[NumLoadsPerpendicularA]
  #   state[LSCA]
  #   state[LSPA]
  @staticmethod
  def setGlobalLoadTileDimClassic(state, tc, numLoads, totalVectorsCoalesced, totalElementsPerp, depthU):

    if state["WaveSeparateGlobalRead%s"%tc]:
      totalElementsPerp = roundupRatio(totalElementsPerp, state["NumThreads"] // state["WavefrontSize"])

    # nlc = 1
    if state["NumLoadsCoalesced%s"%tc] == 1 :
      foundValid = False
      nlcStart = 1
      if (tc == "A" or tc == "B") and state["DirectToVgpr%s"%tc]:
        # adjust nlc for DirectToVgprA/B
        if state["ProblemType"]["TLU%s"%tc]:
          nlcStart = roundupRatio(state["MIWaveTile%s"%tc], state["GlobalReadVectorWidth%s"%tc])
        else:
          nlcStart = roundupRatio(depthU, state["MatrixInstK"] * state["GlobalReadVectorWidth%s"%tc] * state["LocalSplitU"] // state["MIInputPerThread"])
      for nlc in range(nlcStart, int(state["NumLoads%s"%tc]+1)):
        nlp = state["NumLoads%s"%tc] // nlc
        if state["NumLoads%s"%tc] % nlc == 0 \
            and totalVectorsCoalesced % nlc == 0 \
            and totalElementsPerp % nlp == 0:
          state["NumLoadsCoalesced%s"%tc] = nlc
          state["NumLoadsPerpendicular%s"%tc] = nlp
          # print("NumLoads%s:"%tc,state["NumLoads%s"%tc])
          # print("NumLoadsCoalesced%s:"%tc,state["NumLoadsCoalesced%s"%tc])
          # print("NumLoadsPerpendicular%s:"%tc,state["NumLoadsPerpendicular%s"%tc])
          # print("\n")
          foundValid = True
          break
      if not foundValid:
        reject(state, "%s: No NumLoadsCoalesced=1 found"%tc)
        return False

    # nlc = -1
    elif state["NumLoadsCoalesced%s"%tc] == -1:
      foundValid = False
      for nlc in range(state["NumLoads%s"%tc], 0, -1):
        nlp = state["NumLoads%s"%tc] // nlc
        if state["NumLoads%s"%tc] % nlc == 0 \
            and totalVectorsCoalesced % nlc == 0 \
            and totalElementsPerp % nlp == 0:
          state["NumLoadsCoalesced%s"%tc] = nlc
          state["NumLoadsPerpendicular%s"%tc] = nlp
          foundValid = True
          break
      if not foundValid:
        reject(state, "%s: No NumLoadsCoalesced=-1 found"%tc)
        return False

    # nlc = other
    else:
      if state["NumLoadsCoalesced%s"%tc] > state["NumLoads%s"%tc]:
        reject(state, "%s nlc > numLoads"%tc)
        return False

      state["NumLoadsPerpendicular%s"%tc] = state["NumLoads%s"%tc] \
          // state["NumLoadsCoalesced%s"%tc]

      if state["NumLoads%s"%tc] % state["NumLoadsCoalesced%s"%tc] != 0:
        reject(state, "%s: numLoads %u %% numLoadsCoalesced %u != 0" \
            % (tc, state["NumLoads%s"%tc], state["NumLoadsCoalesced%s"%tc]))
        return False

      if totalVectorsCoalesced % state["NumLoadsCoalesced%s"%tc] != 0 :
        reject(state, "%s: totalVectorsCoalesced %u %% numLoadsPara %u != 0" \
              % (tc, totalVectorsCoalesced, state["NumLoadsCoalesced%s"%tc]))
        return False
      if totalElementsPerp % state["NumLoadsPerpendicular%s"%tc] != 0:
        reject(state, "%s: totalElementsPerp %u %% numLoadsPerp %u != 0" \
              % (tc, totalElementsPerp, state["NumLoadsPerpendicular%s"%tc]))
        return False

    if state["ProblemType"]["TLU%s"%tc]:
      state["LSC%s"%tc] = state["MacroTile%s"%tc] // state["NumLoadsCoalesced%s"%tc]
      state["LSP%s"%tc] = int(math.ceil(float(depthU) / state["NumLoadsPerpendicular%s"%tc]))
    else:
      state["LSC%s"%tc] = int(math.ceil(float(depthU) / state["NumLoadsCoalesced%s"%tc]))
      state["LSP%s"%tc] = state["MacroTile%s"%tc] // state["NumLoadsPerpendicular%s"%tc]

    if state["WaveSeparateGlobalRead%s"%tc] == 1:
      state["LSP%s"%tc] = roundupRatio(state["LSP%s"%tc], state["NumThreads"] // state["WavefrontSize"])
    elif state["WaveSeparateGlobalRead%s"%tc] == 2:
      state["LSP%s"%tc] = state["NumThreads"] // state["WavefrontSize"]

    return True


  ########################################
  # Sets the Global Read Tile dims (para, perp)
  # This information controls which threads read which addresses from global mem)
  # Output from this function:
  #   state[NumLoadsCoalesced*]
  #   state[NumLoadsPerpendicular*]
  #   state[LSC*]
  #   state[LSP*]
  #   state[GlobalReadVectorWidth]
  #
  # LSC and LSP define the shape of the PerLoadTile, measured in elements.
  #   LSC*LSP is the elements loaded by a single instruction across all
  #   threads in the group.
  #   LSC is the number of elements loaded in the para(coalesced) dimension
  #   LSP is the number of elements loaded in the perp(noncoalesced) dimension
  #   PerLoadTile is always rectangular.
  #   When BufferLoad=1, the area (LSC*LSP) can be larger than NumThreads.
  #   In this case, some threads will generate a dummy OOB GRO.
  #   Related fields:
  #     LVC = LSC/GRVW  (LVCA = LSCA/GLVWA)
  #     LVP = LSP/GRVW  (LVPA = LSPA/GLVWA)
  #
  # NumLoadsCoalesced and NumLoadsPerpendicular define the number of times the
  #   PerLoadTile is loaded in each dimension to fetch the LoadTile
  # LoadTile = (LSC * NumLoadsCoalesced) * (LSP * NumLoadsPerpendicular).
  #   For Fractional, the LoadTile can be larger than the MacroTile. Buffer
  #   loads will clip any OOB references to 0 and will also avoid writing these
  #   into LDS.

  # Fractional load algorithm:
  #  - Each load instruction loads one or more (complete) rows of the load tile.
  #     - Each row is LSC elements wide
  #     - Rows are complete and do not wrap. This allows a single base GRO VGPR
  #       to be used for all loads in the tile.
  #     - Some work-items in the load may not perform useful work. These WI will
  #       set their GRO to a large OOB number so as to do no harm
  #     - Some G2L registers space may be unused as well.
  #     - The 'used' message at the bottom of this routine computes and prints the
  #       wasted register space.
  #     - The wasted space is removed when the data is written to LDS- the LWO
  #       for work-items beyond the valid ones are set to safely write to OOB locations.

  #     - In cases where each load is loading multiple rows (multiple lines of lsc
  #       elements), the last load is allowed to load fewer lines than the others.
  #       The KernelWriterAssembly will modify the LWO for the last load.  This allows
  #       flexibility in the unroll factors for example.
  @staticmethod
  def setGlobalLoadTileDimFractional(state, tc, depthU):

    assert(depthU > 0)
    dbFract = 0

    # parDim, perpDim define the LoadTile and are measured in elements
    if state["ProblemType"]["TLU%s"%tc]:
      parDim  = state["MacroTile%s"%tc]
      perpDim = depthU
    else:
      parDim  = depthU
      perpDim = state["MacroTile%s"%tc]

    if dbFract:
        print("\ninfo: %s Fractional MT%u_%u_%u Par=%u Perp=%u WG%02u_%02u_%02u NumThreads=%u GRWV%s=%u" \
          % (tc, state["MacroTile0"], state["MacroTile1"], depthU, \
            parDim, perpDim, \
            state["WorkGroup"][0], state["WorkGroup"][1], state["LocalSplitU"], \
            state["NumThreads"], tc, state["GlobalReadVectorWidth%s"%tc]))

    # Try to find a GRVW which is smaller than the LSC and also does not force
    # the LSC to wrap - both of these conditions can be tested with lsc % grvw ==0.
    # Each iteration divides GRWV by 2 which provides finer granularity
    # and a possible opportunity to handle the lsc
    grvw = state["GlobalReadVectorWidth%s"%tc]
    minGrvw = 2 if state["ProblemType"]["DataType"].isHalf() and \
                globalParameters["ArchCaps"][globalParameters["CurrentISA"]]["HasEccHalf"] else 1
    # TODO- check this for int8 and fractional load
    # minGrvw = 4 if state["ProblemType"]["DataType"].isInt8() and \
    #             globalParameters["ArchCaps"][globalParameters["CurrentISA"]]["HasEccHalf"] else 1
    bestVw = -1
    while grvw >= minGrvw:
      # Per instruction across the entire group:
      elementsLoadedPerInst = state["NumThreads"]*grvw
      mik = 1
      if (state["DirectToVgpr%s"%tc] and state["ProblemType"]["TLU%s"%tc]):
        mik = state["MatrixInstK"] * state["LocalSplitU"] // state["MIInputPerThread"]
        elementsLoadedPerInst //= mik
      # LSC, LSP - #elements loaded along specified dim with each load
      if parDim >= elementsLoadedPerInst:
        # entire work-group can work on (part) of the same row
        state["LSC%s"%tc] = elementsLoadedPerInst
        state["LSP%s"%tc] = mik if state["ProblemType"]["TLU%s"%tc] else state["MatrixInstK"]
        state["NumLoadsCoalesced%s"%tc] = roundupRatio(parDim , state["LSC%s"%tc])
        state["NumLoadsPerpendicular%s"%tc] = 1
      else:
        # work-group exceeds read dimension so wraps to multiple rows
        state["LSC%s"%tc] = parDim
        state["LSP%s"%tc] = min(perpDim, elementsLoadedPerInst // parDim)
        state["NumLoadsCoalesced%s"%tc] = 1
        state["NumLoadsPerpendicular%s"%tc] = roundupRatio(perpDim , state["LSP%s"%tc])

      # Vector loads can't wrap to next P dim, so LSC must be divisible by vector elements;
      if dbFract:
        print("  lsc search : lsc(%u) %% grvw(%u) = %u (?0)" % (state["LSC%s"%tc], grvw, state["LSC%s"%tc] % grvw))
      if state["LSC%s"%tc] % grvw == 0:
        bestVw = grvw
        # Try to shrink GRVW if possible while keeping same LSC and LSP:
        # For example, avoid cases where we use a GRVW=4 with many empty addresses
        # when a GRVW=1 will do instead.
        validElementsLoadedPerInst = state["LSC%s"%tc] * state["LSP%s"%tc]
        grvw //= 2
        while grvw >= minGrvw:
          elementsLoadedPerInst = state["NumThreads"]*grvw//mik
          if elementsLoadedPerInst < validElementsLoadedPerInst:
            break # Went too far, not enough load elements at this VW
          if state["LSC%s"%tc] % grvw == 0:
            if dbFract:
              print("  stepdown success (valid)elementsLoadedPerInst=", validElementsLoadedPerInst, "/", elementsLoadedPerInst, "grvw=", grvw, "lsc=", state["LSC%s"%tc])
            bestVw = grvw
          grvw //= 2
        break

      # TODO - could have this generate dwordx3 loads in addition, step down by 1 instead of div2
      # Would need to change asm code gen to generate x3
      grvw //= 2
      # end-- while loop

    if bestVw == -1:
      if dbFract:
        print ("reject fractional - no acceptable tile dim? GlobalReadVectorWidth%s"%tc, \
         state["GlobalReadVectorWidth%s"%tc])
      return False  # could not find a solution, perhaps only possible for half ?

    state["GlobalReadVectorWidth%s"%tc] = bestVw
    if bestVw != state["GlobalReadVectorWidth%s"%tc]:
      if dbFract:
        print("  reducing GlobalReadVectorWidth%s from %u to %u" \
            % (tc, state["GlobalReadVectorWidth%s"%tc], bestVw))

    # How many loads per threads in each dimension.
    # threads which are outside the global read tile bounds will be clipped
    # in the assembly code generator.
    # Multiply the LSC*GRVW
    state["NumLoadsCoalesced%s"%tc] = roundupRatio(parDim, state["LSC%s"%tc])
    state["NumLoadsPerpendicular%s"%tc] = roundupRatio(perpDim , state["LSP%s"%tc])

    nlc = state["NumLoadsCoalesced%s"%tc]
    nlp = state["NumLoadsPerpendicular%s"%tc]

    # LoadTile must at least cover the MacroTile:
    assert(nlc*state["LSC%s"%tc] >= parDim)
    assert(nlp*state["LSP%s"%tc] >= perpDim)

    perpOverhang = perpDim % state["LSP%s"%tc]
    state["fractionalPerpOverhang%s"%tc] = perpOverhang
    if dbFract:
      # how many threads compute Global Read Offsets (GRO) that are not used
      print("  PerLoadTile=%ux%u elements Loads/WI=%ux%u LoadTile/WI=%ux%u (MT=%ux%u), %u/%u = %.1f%% WI GRO used %s" \
          % (state["LSC%s"%tc], state["LSP%s"%tc], \
             nlc, nlp, \
             nlc*state["LSC%s"%tc], nlp*state["LSP%s"%tc], \
             parDim, perpDim, \
             parDim*perpDim, \
             nlc*nlp*state["NumThreads"]*state["GlobalReadVectorWidth%s"%tc], \
             float(parDim*perpDim), \
             float(nlc*nlp*state["NumThreads"]*state["GlobalReadVectorWidth%s"%tc]) * 100.0) \
             )

      for p in range(0,nlp):
        elementWidth = 4
        if p != nlp-1:
          perp = state["LSP%s"%tc]
        else:
          perp = perpOverhang if perpOverhang else state["LSP%s"%tc]

        validElements = state["LSC%s"%tc] * perp
        print("  buffer_load_element_x%u %ux%ux%u bytes,  %u/%u valid GRO" %\
              (state["GlobalReadVectorWidth%s"%tc], \
              state["LSC%s"%tc], perp, \
              elementWidth, \
              validElements//state["GlobalReadVectorWidth%s"%tc],
              state["NumThreads"]))

    return True


  @staticmethod
  def MatrixInstructionToMIParameters(state):
    isa = tuple(state["ISA"])
    if len(state["MatrixInstruction"]) == 9:
      mi                          = state["MatrixInstruction"]
      state["MatrixInstruction"]  = [state["MatrixInstruction"][0],state["MatrixInstruction"][1],state["MatrixInstruction"][2],state["MatrixInstruction"][3]]

      waves                       = mi[7]* mi[8]
      miwg0                       = mi[4] * mi[0] * mi[7]
      state["WorkGroup"][0]       = miwg0
      state["WorkGroup"][1]       = waves*state["WavefrontSize"] // state["WorkGroup"][0]
      state["ThreadTile"][0]      = 1  # dummy
      state["ThreadTile"][1]      = 1  # dummy

      state["MFMA_BF16_1K"] = False
      if not state["ProblemType"]["Sparse"]:
        miDataType = state["ProblemType"]["DataType"] if (not state["EnableF32XdlMathOp"]) else state["ProblemType"]["F32XdlMathOp"]
        if globalParameters["AsmCaps"][isa]["HasMFMA"]:
          if not (miDataType.toChar() in validMFMA and \
            state["MatrixInstruction"] in validMFMA[miDataType.toChar()]):
            if miDataType.isBFloat16() and \
              state["MatrixInstruction"] in validMFMA["B1k"]:
              state["MFMA_BF16_1K"] = True
            else:
              reject(state, "MatrixInstruction %s not valid for DataType %s" % (state["MatrixInstruction"], miDataType))
        elif globalParameters["AsmCaps"][isa]["HasWMMA"]:
          if state["MatrixInstruction"] not in validWMMA:
            reject(state, "MatrixInstruction %s not valid for DataType %s" % (state["MatrixInstruction"], state["ProblemType"]["DataType"]))
      else:
        if not (state["ProblemType"]["DataType"].toChar() in validSMFMA and \
          state["MatrixInstruction"] in validSMFMA[state["ProblemType"]["DataType"].toChar()]):
          reject(state, "Sparse MatrixInstruction %s not valid for DataType %s" % (state["MatrixInstruction"], state["ProblemType"]["DataType"]))

      # set EnableMatrixInstruction
      state["EnableMatrixInstruction"] = True

      # set MIBlock
      MIBlock_BM = miwg0 // mi[0]
      MIBlock_BM = min(MIBlock_BM, mi[3])
      MIBlock_BN = mi[3] // MIBlock_BM

      state["MIBlock"]    = [32, 32, 2, 1, 1, 1]
      state["MIBlock"][0] = mi[0]
      state["MIBlock"][1] = mi[1]
      state["MIBlock"][2] = mi[2]
      state["MIBlock"][3] = mi[3]
      state["MIBlock"][4] = MIBlock_BM
      state["MIBlock"][5] = MIBlock_BN

      # set MIWaveGroup
      state['MIWaveGroup']     = [1, 1]
      state['MIWaveGroup'][0]  = min((miwg0 // mi[0]) // MIBlock_BM, waves)
      state['MIWaveGroup'][1]  = waves // state['MIWaveGroup'][0]

      # set MIWaveTile
      state['MIWaveTile']      = [1, 1]
      state['MIWaveTile'][0]   = mi[5]
      state['MIWaveTile'][1]   = mi[6]
      # set MIInputPerThread
      isa = tuple(state["ISA"])
      state['MIInputPerThread'] = state["MatrixInstruction"][0] * state["MatrixInstruction"][2] * state["MatrixInstruction"][3] // state["WavefrontSize"]
      if (not globalParameters["AsmCaps"][isa]['HasMFMA']) and globalParameters["AsmCaps"][isa]['HasWMMA']:
        if state['ISA'][0] == 10 or state['ISA'][0] == 11:
          state['MIInputPerThread'] = state["MatrixInstruction"][2]
      sparseA = False if not state["ProblemType"]["Sparse"] else False if state["ProblemType"]["Sparse"] == 2 else True
      sparseB = False if not state["ProblemType"]["Sparse"] else True if state["ProblemType"]["Sparse"] == 2 else False
      state['MIInputPerThreadA'] = state['MIInputPerThread'] if not sparseA else state['MIInputPerThread']//2
      state['MIInputPerThreadB'] = state['MIInputPerThread'] if not sparseB else state['MIInputPerThread']//2
      state['MIInputPerThreadMetadata'] = state['MIInputPerThread'] if not state["ProblemType"]["Sparse"] else state['MIInputPerThread']//8
    elif state["MatrixInstruction"] != [] and len(state["MatrixInstruction"]) == 4:
      state["EnableMatrixInstruction"] = True
    else:
      state["EnableMatrixInstruction"] = False


  ##############################################
  # check and calculate Wave Separate Global Read
  @staticmethod
  def checkAndAssignWaveSeparateGlobalRead(state, tc):
    # check can we use WaveSeparateGlobalRead
    numOfWaves = state["NumThreads"] // state["WavefrontSize"]
    if state["WaveSeparateGlobalRead%s"%tc]:
      if state["ProblemType"]["TLU%s"%tc] and (state["_DepthU%s"%tc] > 0) and (state["_DepthU%s"%tc] % numOfWaves != 0):
        reject(state, "didn't support WaveSeparateGlobalRead when DepthU is not multiple of wave %u in TLU%s" % (state["_DepthU%s"%tc], tc))
      if not state["ProblemType"]["TLU%s"%tc] and (state["MacroTile%s" % tc] % numOfWaves != 0):
        reject(state, "didn't support WaveSeparateGlobalRead when MacroTile is not multiple of wave %u in TLU%s" % (state["MacroTile%s"%tc], tc))


  ########################################
  # determine can we use VgprForLocalReadPacking
  @staticmethod
  def isVgprForLocalReadPackingDoable(state):
    isa = tuple(state["ISA"])
    doable = True
    # MatrixInstruction only
    if not state["EnableMatrixInstruction"]:
      doable = False
    # only for HasEccHalf
    if not globalParameters["ArchCaps"][isa]["HasEccHalf"]:
      doable = False
    # only for PLR>=1 (except for DTVA+B)
    if state["PrefetchLocalRead"] < 1 and not (state["DirectToVgprA"] and state["DirectToVgprB"]):
      doable = False
    # only for 1 or 2 byte input (numRegister < 1)
    if state["ProblemType"]["DataType"].numRegisters() >= 1:
      doable = False
    return doable

  ########################################
  # determine if current datatype can support DirectToVgpr
  @staticmethod
  def isDirectToVgprSupportDataType(state):
    return (state["ProblemType"]["DataType"].isSingle() or state["ProblemType"]["DataType"].isDouble() or state["ProblemType"]["DataType"].isComplex() or \
            state["ProblemType"]["DataType"].isHalf() or state["ProblemType"]["DataType"].isBFloat16() or state["ProblemType"]["DataType"].isInt8()) or \
            state["ProblemType"]["DataType"].is8bitFloat()

  ########################################
  # determine can we use DirectToVgpr
  @staticmethod
  def isDirectToVgprDoable(state, tc):
    MIindex = 0 if tc == 'A' else 1
    numBytes = state["ProblemType"]["DataType"].numBytes()
    numBytesGR = state["ProblemType"]["DataType%s"%tc].numBytes()
    # With MatrixInstruction only
    if not state["EnableMatrixInstruction"] :
      reject(state, "DirectToVgpr is for MatrixInstruction only")
      return False

    # disable the following combinations for initial implementation
    # TODO: enable them
    if state["LocalSplitU"] != 1 and (not state["ProblemType"]["TLU%c"%tc]):
      reject(state, "DirectToVgpr + LSU + TLU=False has not been enabled yet(tentative)")
      return False

    if state["DirectToVgprA"] and state["DirectToVgprB"]:
      # change the following parameter values
      state["PrefetchGlobalRead"] = 1
      state["ExpandPointerSwap"] = 0
      state["1LDSBuffer"] = 0
      state["PrefetchLocalRead"] = 0
      # So far, DTVA + DTVB does not perform well (waitcnt is not ideal).
      # Disable it for now (TODO: improve waitcnt and re-enable)
      reject(state, "DirectToVgprA + DirectToVgprB disabled")
      return False

    # DTV + input type conversion
    if state["ProblemType"]["DataType%s"%tc] != state["ProblemType"]["DataType"]:
      if not state["ConvertAfterDS"]:
        reject(state, "DirectToVgpr%s + input conversion + ConvertAfterDS=False not supported"%(tc))
        return False

    # check if the DataType can support DirectToVgpr
    if not Solution.isDirectToVgprSupportDataType(state):
      reject(state, "no DirectToVgpr support for this input data type")
      return False

    # Does not work with TLU = False and PrefetchLocalRead = 0
    if (not state["ProblemType"]["TLU%c"%tc]) and state["PrefetchLocalRead"] == 0:
      reject(state, "DirectToVgpr%c does not supports TLU%c = False and PrefetchLocalRead = 0"%(tc, tc))
      return False

    # Does not work with TLU = False and CGEMM/DGEMM/DGEMM (not supported)
    if (not state["ProblemType"]["TLU%c"%tc]) and (state["ProblemType"]["DataType"].isDouble() or \
        state["ProblemType"]["DataType"].isComplex()):
      reject(state, "DirectToVgpr%c does not supports TLU%c = False + S/C/D/ZGEMM"%(tc, tc))
      return False

    if numBytesGR * state["GlobalReadVectorWidth%c"%tc] < 4:
      # no support for DTV + numBytesGR * GlobalReadVectorWidth< 4
      reject(state, "DirectToVgpr%c does not support TLU%c + numByte * GlobalReadVectorWidth%c < 4"%(tc, tc, tc))
      return False

    if numBytes < 4:
      # numBytes < 4 case
      if state["ProblemType"]["TLU%c"%tc]:
        # use pack logic (with v_perm) same as local read (only if VgprForLocalReadPacking is doable)
        if not Solution.isVgprForLocalReadPackingDoable(state):
          reject(state, "Does not meet the requirement for DirectToVgpr%c + TLU%c + numByte < 4"%(tc, tc))
          return False
        # force ClusterLocalRead=1 for DTV + pack
        state["ClusterLocalRead"] = 1
    else:
      # numBytes >= 4 case
      if state["ProblemType"]["TLU%c"%tc] and state["MIInputPerThread"] > 1:
        # no support for numBytes >= 4 + MIInputPerThread > 1
        reject(state, "DirectToVgpr%c does not support TLU%c+ numByte >= 4 + MIInputPerThread > 1"%(tc, tc))
        return False

    # MatrixInstBM,BN check
    #  for A, MatrixInstBN should be 1
    #  for B, MatrixInstBM should be 1
    # This is to limit the number of Vgpr
    if tc == 'A' and not (state['MatrixInstBN'] == 1):
      reject(state, "MatrixInstBN should be 1 for DirectToVgprA. Current value is %d"%(state['MatrixInstBN']))
      return False
    if tc == 'B' and not (state['MatrixInstBM'] == 1):
      reject(state, "MatrixInstBM should be 1 for DirectToVgprB. Current value is %d"%(state['MatrixInstBM']))
      return False

    # Does not work with WaveSeparateGlobalRead
    if state["WaveSeparateGlobalRead%c"%tc]:
      reject(state, "DirectToVgpr%c does not supports WaveSeparateGlobalRead%c"%(tc, tc))
      return False

    # Does not work with TLU + VectorWidth != GlobalReadVectorWidth (VW = 2 + GRVW = 1 or VW = 1 + GRVW = 2 does not work)
    if state["ProblemType"]["TLU%c"%tc] and state["VectorWidth%s"%tc] != state["GlobalReadVectorWidth%c"%tc]:
      reject(state, "DirectToVgpr%c does not supports TLU + VectorWidth%s(=%u) != GlobalReadVectorWidth%c(%u)"%(tc, tc, state["VectorWidth%s"%tc], tc, state["GlobalReadVectorWidth%c"%tc]))
      return False

    # Does not work with TLU=False and NumLoadsCoalesced != DepthU//(MatrixInstK*GRVW*LSU//MIInputPerThread)
    if (not state["ProblemType"]["TLU%c"%tc]) and \
        state["NumLoadsCoalesced%c"%tc] != state["DepthU"] // (state["MatrixInstK"] * state["GlobalReadVectorWidth%c"%tc] * state["LocalSplitU"] // state["MIInputPerThread"]):
      reject(state, "DirectToVgpr%c does not supports TLU=False and NumLoadsCoalesced%c != DepthU//(MatrixInstK*GlobalReadVectorWidth*LocalSplitU//MIInputPerThread(=%u))"%(tc, tc, state["MIInputPerThread"]))
      return False

    # TLU=False case, need GlobalReadVectorWidth == LocalReadVectorWidth
    if (not state["ProblemType"]["TLU%c"%tc]) and \
       state["GlobalReadVectorWidth%c"%tc] != state["LocalReadVectorWidth"]:
      reject(state, "DirectToVgpr%c does not supports TLU=False GlobalReadVectorWidth%c(%u) != LocalReadVectorWidth(%u)"%(tc, tc, state["GlobalReadVectorWidth%c"%tc], state["LocalReadVectorWidth"]))
      return False

    # Does not work with SIA<3
    if state["ScheduleIterAlg"] < 3:
      reject(state, "DirectToVgpr%c does not supports ScheduleIterAlg < 3"%(tc))
      return False

    # Does not work with InnerUnroll>1
    if state["InnerUnroll"]>1:
      reject(state, "DirectToVgpr%c does not supports InnerUnroll>1"%(tc))
      return False

    # Reject TLU = UnrollMajorLDS
    if state["ProblemType"]["TLU%c"%tc] == state["UnrollMajorLDS%c"%tc]:
      reject(state, "DirectToVgpr%c does not supports TLU%c = UnrollMajorLDS%c"%(tc, tc, tc))
      return False

    # does not work with UnrollLoopSwapGlobalReadOrder
    if state["UnrollLoopSwapGlobalReadOrder"]:
      reject(state, "DirectToVgpr%c does not supports UnrollLoopSwapGlobalReadOrder"%(tc))
      return False

    # does not work with PGR2 + EPS
    if state["PrefetchGlobalRead"] == 2 and state["ExpandPointerSwap"]:
      # force EPS=0 and continue
      state["ExpandPointerSwap"] = 0

    # does not work with Sparse
    if state["ProblemType"]["Sparse"]:
      reject(state, "DirectToVgpr%c does not supports Sparse"%(tc))
      return False

    # for DTVA/DTVB, does not work with PGR0
    if state["PrefetchGlobalRead"] == 0:
      reject(state, "DirectToVgpr%c does not supports PrefetchGlobalRead == 0."%(tc))
      return False

    # for DTVA, does not work with NN and TLDS0
    if tc == 'A' and state["TransposeLDS"] == 0 and (not state["ProblemType"]["TransposeA"] and not state["ProblemType"]["TransposeB"]):
      reject(state, "DirectToVgpr%c does not supports NN case with TransposeLDS == 0."%(tc))
      return False

    # for DTVA, does not work with TT and Tail-loop
    if tc == 'A' and (state["ProblemType"]["TransposeA"] and state["ProblemType"]["TransposeB"]):
        # Use AssertSummationElementMultiple (BoundSizeMultiple in predicates) to exclude failed tail-loop cases
        state["AssertSummationElementMultiple"] = max(state["AssertSummationElementMultiple"], state["DepthU"])

    # for DTVB, does not work with NN and Tail-loop
    if  tc == 'B' and (not state["ProblemType"]["TransposeA"] and not state["ProblemType"]["TransposeB"]):
        # Use AssertSummationElementMultiple (BoundSizeMultiple in predicates) to exclude failed tail-loop cases
        state["AssertSummationElementMultiple"] = max(state["AssertSummationElementMultiple"], state["DepthU"])

    # Does not work with DirectToLDS
    # -> this will be checked after DirectToLDS doable check is done

    return True

  ########################################
  # determine can we use DirectToLds
  @staticmethod
  def isDirectToLdsDoable(state, tc):

    numBytes = state["ProblemType"]["DataType"].numBytes()
    isa = tuple(state["ISA"])

    # x4 support for directToLds
    canDTLx4 = globalParameters["AsmCaps"][isa]["HasDirectToLdsx4"]

    # numelements_perlane = 4/numBytes
    # TN with transposeLDS feature should work as long as state["AssertSummationElementMultiple"] % (numelements_perlane*2) = 0
    #                                                     state["AssertSummationElementMultiple"] % (numelements_perlane*4) = 0

    #NT
    # use only for f32 & DGEMM and TLU = 1
    #TN
    # use for all precisions with TransposeLDS=1

    numBytesAB = state["ProblemType"]["DataType%s"%tc].numBytes()
    numBytesPerLoad = state["GlobalReadVectorWidth%s"%tc] * numBytesAB

    # so far, numBytesAB<4 case, TLU=False only (continue with False)
    if numBytesAB < 4 and state["ProblemType"]["TLU%c"%tc]:
      return False

    # x2 DTL is not supported
    if numBytesPerLoad == 8:
      reject(state, "can't use DirectToLds with b64 buffer load")
      return False

    if numBytesPerLoad == 16 and not canDTLx4:
      reject(state, "b128 DirectToLds not supported")
      return False

    # so far MFMA only (TODO: enable non MFMA case)
    if not state["EnableMatrixInstruction"]:
      reject(state, "DirectToLds is for MatrixInstruction only for now (tentative)")
      return False

    # so far, DirectToLds does not work with StreamK (TODO: enable StreamK case)
    if state["StreamK"]:
      reject(state, "DirectToLds does not support StreamK (tentative)")
      return False

    # ToDo: Review def of lrvw and this check
    # DTL + LocalReadVectorWidth > MIInputPerThread does not work
    # Need support for TailLoop
    if not state["ProblemType"]["Sparse"] and (not globalParameters["AsmCaps"][isa]["HasMFMA_f8f6f4"] or state["MatrixInstK"] <= 32):
      if state["LocalReadVectorWidth"] > state["MIInputPerThread"]:
        reject(state, "DirectToLds does not work with LocalReadVectorWidth > MIInputPerThread")
        return False

    if not state["ProblemType"]["Sparse"] and (not globalParameters["AsmCaps"][isa]["HasMFMA_f8f6f4"] or state["MatrixInstK"] <= 32):
      if state["ProblemType"]["DataType"].isBFloat16() and state["AssertSummationElementMultiple"] % (2 * state["GlobalReadVectorWidth%c"%tc]) != 0:
        reject(state, "can't use DirectToLds for BF16 with AssertSummationElementMultiple %u" % state["AssertSummationElementMultiple"])
        return False

    if state["NumThreads"] % state["WavefrontSize"] != 0:
      reject(state, "can't use DirectToLds for NumThreads % WavefrontSize != 0")
      return False

    if state["ProblemType"]["TLU%c"%tc] == state["UnrollMajorLDS%c" % tc]:
      reject(state, "can't use DirectToLds for TLU%c == UnrollMajorLDS%c"%(tc, tc))
      return False

    # avoid picking x2&x4 for precisions < f32/f64 in [ProblemType][TLU] == TRUE
    if not state["EnableMatrixInstruction"]:
      if state["GlobalReadVectorWidth%c"%tc] * numBytesAB * state["WavefrontSize"] > 256:
        reject(state, "can't use DirectToLds for not EnableMatrixInstruction and GlobalReadVectorWidth%c * bpe%c * WavefrontSize > 256"%(tc,tc))
        return False

    if state["WaveSeparateGlobalRead%c" % tc]:
      if state["LSC%c"%tc] * state["LSP%c"%tc] * numBytesAB != state["WavefrontSize"] * state["GlobalReadVectorWidth%c"%tc] * numBytesAB:
        reject(state, "can't use DirectToLds for LSC%c and LSP%c * bpe!= WavefrontSize * GlobalReadVectorWidth%c * bpe%c > 4"%(tc, tc, tc, tc))
        return False
    else:
      if state["LSC%c"%tc] * state["LSP%c"%tc] * numBytesAB != state["NumThreads"] * state["GlobalReadVectorWidth%c"%tc] * numBytesAB:
        reject(state, "can't use DirectToLds for LSC%c and LSP%c * bpe != NumThreads * GlobalReadVectorWidth%c * bpe%c > 4"%(tc, tc, tc, tc))
        return False

    # so far, DirectToLds does not work well with PGR=2
    # performance is not good and a lot of ds_read for DTL can cause scheduling issue(need fix)
    if state["PrefetchGlobalRead"] == 2:
      reject(state, "can't use DirectToLds for PrefetchGlobalRead == 2")
      return False

    # so far, DirectToLds does not work with LRVW=2
    if state["LocalReadVectorWidth"] == 2:
      reject(state, "can't use DirectToLds for LocalReadVectorWidth == 2")
      return False

    # Does not work with (NumLoadsCoalesced>1 and UseInstOffsetForGRO) + DGEMM
    if state["ProblemType"]["DataType"].isDouble() and \
      (state["NumLoadsCoalesced%c"%tc] > 1 and state["UseInstOffsetForGRO"]):
      reject(state, "DirectToLds%c does not supports NumLoadsCoalesced%c > 1 and UseInstOffsetForGRO for dgemm"%(tc, tc))
      return False

    # Does not work with NumLoadsCoalesced>1 + ZGEMM
    if state["ProblemType"]["DataType"].isDoubleComplex() and state["NumLoadsCoalesced%c"%tc] > 1:
      reject(state, "DirectToLds%c does not supports NumLoadsCoalesced%c > 1 for zgemm"%(tc, tc))
      return False

    # Does not work with PrefetchGlobalRead=2 and PrefetchLocalRead=1 (cannot schedule DTL global read after local read)
    if state["PrefetchGlobalRead"] == 2 and state["PrefetchLocalRead"] == 1:
      reject(state, "DirectToLds%c does not work with PrefetchGlobalRead=2 and PrefetchLocalRead=1"%(tc))
      return False

    # DirectToLds does not work if MacroTile is not power of 2
    # LDS offset swap/rotate logic works only when MacroTile is power of 2
    mt = state["MacroTile%c"%tc]
    if mt & (mt - 1) != 0:
      reject(state, "can't use DirectToLds if MacroTile%s is not power of 2"%tc)
      return False

    # DirectToLds does not work with TLU=False and bpe > bpr and DepthU//NumLoadsCoalesced < 8
    # bpe > bpr case, Lower and upper 4 bytes elements are stored separately.
    # if TLU=False and DepthU//NumLoadsCoalesced is smaller than lower block size (8 elements),
    # current offset swap logic does not work
    if (not state["ProblemType"]["TLU%c"%tc]) and state["ProblemType"]["DataType"].numRegisters() > 1 and \
       state["_DepthU%s"%tc] // state["NumLoadsCoalesced%c"%tc] < 8:
      reject(state, "DirectToLds%c does not work with TLU=False and bpe > bpr and DepthU//NumLoadsCoalesced%c < 8"%(tc, tc))
      return False

    return True

  @staticmethod
  def getDivisorName(state, tC):
    if state["GlobalReadCoalesceGroup{}".format(tC)]:
      if state["GlobalReadCoalesceVector{}".format(tC)]:
        divisorName = "LVC{}".format(tC)
      else:
        # Fractional load use the more accurate lsc, multiply by VW later
        divisorName = "LSC{}".format(tC)
    else:
      if state["GlobalReadCoalesceVector{}".format(tC)]:
        divisorName = "LSP{}".format(tC)
      else:
        divisorName = "LVP{}".format(tC)
    return divisorName

  ########################################
  # assign all derived parameters
  @staticmethod
  def assignDerivedParameters(state):
    state["EnableF32XdlMathOp"] = False #ignore the F32 xDL MathOp by default.
    #enable F32 xDL MathOp only when the input type is f32.
    if "F32XdlMathOp" in state["ProblemType"] \
       and (not state["ProblemType"]["F32XdlMathOp"].isSingle()) \
       and (state["ProblemType"]["DataType"].isSingle()):
      state["EnableF32XdlMathOp"] = True

    Solution.assignProblemIndependentDerivedParameters(state)

    if "AssignedDerivedParameters" in state:
      if state["AssignedDerivedParameters"]:
        return
    state["AssignedDerivedParameters"] = False

    for s in Solution.InternalKeys:
        state['_'+s] = state[s]
        #del state[s]

    # Force update _GlobalAccumulation
    computeBytes = state["ProblemType"]["ComputeDataType"].numBytes()
    state["_GlobalAccumulation"] = None
    computeName  = state["ProblemType"]["ComputeDataType"].toName()
    if state["StreamK"] > 0 and state["StreamKAtomic"] == 0:
      # StreamK Workspace size
      state["_GlobalAccumulation"] = 'PartialsBuffer'
    elif state["GlobalSplitUAlgorithm"] == 'SingleBuffer':
      if computeName != state["ProblemType"]["DestDataType"].toName():
        state["_GlobalAccumulation"] = 'SingleBuffer'
    elif state["GlobalSplitUAlgorithm"] == 'MultipleBuffer':
      state["_GlobalAccumulation"] = 'MultipleBuffer'
    elif state["GlobalSplitUAlgorithm"] == 'MultipleBufferSingleKernel':
      if (not globalParameters["SplitGSU"]):
        state["_GlobalAccumulation"] = 'MultipleBufferSingleKernel'
      else:
        if state["GlobalSplitU"] > 1:
          state["_GlobalAccumulation"] = 'MultipleBufferSingleKernel'

    if state["_GlobalAccumulation"] == 'MultipleBufferSingleKernel':
      state["SynchronizerSizeCheck"] = 1
    #   state["BatchSizeEqual"] = 1

    isa = tuple(state["ISA"])

    if state["StreamK"] != 0:
      state["GlobalSplitU"] = 0 # Cannot enable both Stream-K and GSU
      state["GlobalSplitUAlgorithm"] = "MultipleBuffer" # Set default Algorithm
      if state["ProblemType"]["DataType"].isDouble():
        reject(state, "Type {} for DataType not yet supported with StreamK".format(state["ProblemType"]["DataType"].toChar()))
      if state["MIWaveGroup"][0] * state["MIWaveGroup"][1] != 4:
        reject(state, "Stream-K requries MIWaveGroup0*MIWaveGroup1=4")
      if not state["EnableMatrixInstruction"]:
        reject(state, "Stream-K requires MatrixInstruction")
      if globalParameters["AsmCaps"][isa]["HasWMMA"]:
        reject(state, "Stream-K untested with WMMA")
      # if state["PersistentKernel"]:
      #   reject(state, "Cannot enable both Stream-K and PersistentKernel")
      if not state["ProblemType"]["StridedBatched"]:
        reject(state, "General batch not supported with Stream-K")
      if state["ProblemType"]["GroupedGemm"]:
        reject(state, "Grouped gemm not yet supported with Stream-K")
      if state["ScheduleGlobalRead"] != 1:
        reject(state, "ScheduleGlobalRead not supported with Stream-K")
      if state["ScheduleLocalWrite"] != 1:
        reject(state, "ScheduleLocalWrite not supported with Stream-K")
      if state["ScheduleIterAlg"] != 2 and state["ScheduleIterAlg"] != 3:
        reject(state, "ScheduleIterAlg not supported with Stream-K")
      if state["StreamKAtomic"] == 1:
        if not state["ProblemType"]["DataType"].isSingle():
          reject(state, "Atomic Stream-K currently only tested for SGEMM")
        if not state["BufferStore"]:
          reject(state, "Atomic Stream-K requires BufferStore")
        if state["LocalSplitU"] > 1:
          reject(state, "Atomic Stream-K not working with LocalSplitU")
      if not state["Valid"]:
        return
    else:
      # If not using StreamK, clear other stream-k settings to avoid duplicate kernels
      state["StreamKAtomic"] = 0
      state["StreamKXCCMapping"] = 0
      state["DebugStreamK"] = 0

    computeBytes = state["ProblemType"]["ComputeDataType"].numBytes()
    state["_WorkspaceSizePerElemC"] = computeBytes
    state["_WorkspaceSizePerElemBias"] = 0
    if state["ProblemType"]["UseBias"] and state["ProblemType"]["Gradient"]:
      state["_WorkspaceSizePerElemBias"] = computeBytes

    state["WorkspaceCheck"] = [state["_WorkspaceSizePerElemC"], state["_WorkspaceSizePerElemBias"], state["GlobalSplitU"] if (state["GlobalSplitUAlgorithm"] == 'MultipleBuffer' or state["_GlobalAccumulation"] == 'MultipleBufferSingleKernel') else 1]

    if state["VectorStore"] == -1:
        state["_VectorStore"] = 1 # default, may be changed if needed to generate a valid kernel

    ProblemType.assignDerivedParameters(state["ProblemType"])
    if not state["Valid"]:
      print2("in assignDerivedParameters, state['Valid'] = False")
      return

    if not globalParameters["AsmCaps"][isa]["HasNTModifier"]:
      # force to disable nt flag if it is not supported by arch
      for ch in ["", "A", "B", "C", "D", "E", "WS", "Metadata"]:
        if state["NonTemporal%s"%ch] >= 4:
          state["NonTemporal%s"%ch] -= 4

    if state["WavefrontSize"] == 32 and not globalParameters["ArchCaps"][isa]["HasWave32"]:
      reject(state, "WavefrontSize=32 not supported for ISA {}".format(isa))
      return

    if state["WavefrontSize"] == 32 and state["KernelLanguage"] == "Source":
      reject(state, "WavefrontSize=32 not yet supported for source kernels.")
      return

    if state["EnableMatrixInstruction"]:
      if not (globalParameters["AsmCaps"][isa]["HasMFMA"] or globalParameters["AsmCaps"][isa]["HasWMMA"]):
        reject(state, f"isa {isa} doesn't support matrix instruction")
        return
      if not (state["ProblemType"]["DataType"].isSingle() \
              or state["ProblemType"]["DataType"].isDouble() \
              or state["ProblemType"]["DataType"].isBFloat16() \
              or state["ProblemType"]["DataType"].isHalf() \
              or state["ProblemType"]["DataType"].isComplex() \
              or state["ProblemType"]["DataType"].is8bitFloat() \
              or state["ProblemType"]["DataType"].isInt8()):
        reject(state, "didn't support Matrix Instruction with type %s" % str(state["ProblemType"]["DataType"]))
        return
      if (not globalParameters["AsmCaps"][isa]["HasMFMA"] and globalParameters["AsmCaps"][isa]["HasWMMA"] and (state["WavefrontSize"] == 64)):
         print2("!! Warning: WMMA only well tune on WGP mode, wave size = 32")
      #  reject(state, "WMMA only suppport on WGP mode, wave size = 32")
      #  return
      if not state["MIBlock"] or len(state["MIBlock"]) != 6:
        reject(state, "invalid MIBlock")
        return
      if not state["MIWaveGroup"] or len(state["MIWaveGroup"]) != 2:
        reject(state, "invalid MIWaveGroup")
        return
      if not state["MIWaveTile"] or len(state["MIWaveTile"]) != 2:
        reject(state, "invalid MIWaveTile")
        return
      if globalParameters["AsmCaps"][isa]["HasMFMA"]:
        if not state["ProblemType"]["HighPrecisionAccumulate"] \
           and state["ProblemType"]["DataType"].numRegisters() < 1 :
          reject(state, "Matrix instructions for half, bf16 (or i8) types are natively accumulated" + \
           " in fp32 (or i32) precision. Please add the following config:" + \
           "\n - HighPrecisionAccumulate: True")
          return
      if globalParameters["AsmCaps"][isa]["HasWMMA"]:
        if state["ProblemType"]["DataType"].numRegisters() >=1:
          reject(state, "WMMA only support half, bf16 and i8 type")
          return
      if state["InterleaveAlpha"]:
        reject(state, "Matrix instruction doesn't support InterleaveAlpha")
        return
      if state["ProblemType"]["DataType"].isInt8():
        if isa[:2] == (9, 4):
          if tuple(state["MatrixInstruction"])[:3] in ((32, 32, 8), (16, 16, 16)):
            reject(state, "v_mfma_i32_32x32x8 and v_mfma_i32_16x16x16 have been deprecated in gfx94x")
            return
      if state["ProblemType"]["ComputeDataType"].isDouble():
        # See [4,4,4,4] snop for more info
        if state["MatrixInstruction"] == [4,4,4,4] and (not state['ISA'] == [9,0,10]) and state["ScheduleIterAlg"] == 3:
          reject(state, "Currently Matrix instructions [4,4,4,4] is disabled.")
          return
    else:
      if not state["ProblemType"]["HighPrecisionAccumulate"] \
         and state["ProblemType"]["ComputeDataType"].numRegisters() > state["ProblemType"]["DataType"].numRegisters() :
        reject(state, "For non-MI Kernel, if sizeof(ComputeDataType) > sizeof(DataType), " + \
         "Please add the following config:" + \
         "\n - HighPrecisionAccumulate: True")
        return
      if state["ProblemType"]["Sparse"]:
        reject(state, "Sparse A problem is only supported by SMFMA MI kernel.")
        return

      if state["ThreadTile0"] > 16 or state["ThreadTile1"] > 16:
        reject(state, "Invalid value for ThreadTile")
        return

      if state["ScheduleIterAlg"] == 2 or state["ScheduleIterAlg"] == 3:
        reject(state, "SIA2 and SIA3 only support MatrixInstruction")
        return

    if state["ProblemType"]["Tensor0"]==0:
      state["ThreadTileA"] = state["ThreadTile0"]
      state["ThreadTileB"] = state["ThreadTile1"]
      state["SubGroupA"] = state["SubGroup0"]
      state["SubGroupB"] = state["SubGroup1"]
      state["MacroTileA"] = state["MacroTile0"]
      state["MacroTileB"] = state["MacroTile1"]
      if state["EnableMatrixInstruction"]:
        state["MIWaveTileA"] = state["MIWaveTile"][0]
        state["MIWaveTileB"] = state["MIWaveTile"][1]
    else:
      state["ThreadTileB"] = state["ThreadTile0"]
      state["ThreadTileA"] = state["ThreadTile1"]
      state["SubGroupB"] = state["SubGroup0"]
      state["SubGroupA"] = state["SubGroup1"]
      state["MacroTileB"] = state["MacroTile0"]
      state["MacroTileA"] = state["MacroTile1"]
      if state["EnableMatrixInstruction"]:
        state["MIWaveTileA"] = state["MIWaveTile"][1]
        state["MIWaveTileB"] = state["MIWaveTile"][0]

    if state["ProblemType"]["Sparse"] == 2 and state["DirectToVgprSparseMetadata"]:
      reject(state, "Sparse B does not supprot DirectToVgprSparseMetadata")
      return


    if state["ProblemType"]["Sparse"]:
      if state["ProblemType"]["Sparse"] == 2:
        if not state["DirectToVgprSparseMetadata"]:
          state["ThreadTileMetadata"] = state["ThreadTileB"]
          state["SubGroupMetadata"] = state["SubGroupB"]
          state["MacroTileMetadata"] = state["MacroTileB"]
          state["WaveSeparateGlobalReadMetadata"] = state["WaveSeparateGlobalReadB"]
          state["DirectToLdsMetadata"] = False
          state["LocalWriteUseSgprMetadat"] = False
          state["ProblemType"]["MirrorDimsMetadata"]  = list(state["ProblemType"]["MirrorDimsB"])
          state["VectorWidthMetadata"] = state["VectorWidthB"]
        if state["EnableMatrixInstruction"]:
          state["MIWaveTileMetadata"] = state["MIWaveTileB"]
      else:
        if not state["DirectToVgprSparseMetadata"]:
          state["ThreadTileMetadata"] = state["ThreadTileA"]
          state["SubGroupMetadata"] = state["SubGroupA"]
          state["MacroTileMetadata"] = state["MacroTileA"]
          state["WaveSeparateGlobalReadMetadata"] = state["WaveSeparateGlobalReadA"]
          state["DirectToLdsMetadata"] = False
          state["LocalWriteUseSgprMetadat"] = False
          state["ProblemType"]["MirrorDimsMetadata"]  = list(state["ProblemType"]["MirrorDimsA"])
          state["VectorWidthMetadata"] = state["VectorWidthA"]
        if state["EnableMatrixInstruction"]:
          state["MIWaveTileMetadata"] = state["MIWaveTileA"]
    elif not state["ProblemType"]["Sparse"]:
      state["DirectToVgprSparseMetadata"] = False
      state["MIWaveTileMetadata"] = 0

    if state["NonTemporal"] != -1:
      state["NonTemporalA"] = state["NonTemporal"]
      state["NonTemporalB"] = state["NonTemporal"]
      state["NonTemporalC"] = state["NonTemporal"]
      state["NonTemporalD"] = state["NonTemporal"]
      state["NonTemporalMetadata"] = state["NonTemporal"]

    # Init vars early since there are early-exit return statements below
    state["DirectToLdsA"] = False
    state["DirectToLdsB"] = False
    state["LocalWriteUseSgprA"] = False
    state["LocalWriteUseSgprB"] = False

    state["WorkGroupMappingXCC"] = abs(state["WorkGroupMappingXCC"])

    problemType = state["ProblemType"]

    for (tc,batchMask) in (('A', 0x1), ('B', 0x2)):
      freeDims = [i for i in problemType["IndexAssignments%s"%tc] if i in problemType["IndicesFree"]]
      if not freeDims:
        reject(state, "tensor%s contains no free indices.")
        return False

    # Determine which indices will be packed together as this impacts several different parms (sizes, magic numbers, etc)
    # The order in PackedC*Indices also determines the order that dimensions are packed - the first elements in
    # the list are the fastest-moving elements.
    # The store code optimizes for C0 being the coalesced dimension and C1 the perp dimension.
    # C0/C1 indices can come from IndexAssignmentsA or IndexAssignmentsB
    # grid size [0,1]
    state["PackedC0IdxChars"] = []
    state["PackedC0IndicesX"] = []
    indexChars = INDEX_CHARS
    # Pack all the dimensions (free) of A into grid[0]

    if problemType["Index0"] in problemType["IndexAssignmentsA"]:
      tc0 = 'A'
      tc1 = 'B'
    else:
      tc0 = 'B'
      tc1 = 'A'
    assert(isPackedIndex(state, problemType["Index01A"]))
    assert(isPackedIndex(state, problemType["Index01B"]))

    # Pack all the dimensions (batch and free) of A into grid[0]
    for idx in problemType["IndexAssignments%s"%tc0]:
      if isPackedIndex(state, idx):
        assert (idx < problemType["NumIndicesC"])
        state["PackedC0IdxChars"].append("%s" % indexChars[idx])
        state["PackedC0IndicesX"].append(idx)

    state["PackedC1IdxChars"] = []
    state["PackedC1IndicesX"] = []
    for idx in problemType["IndexAssignments%s"%tc1]:
      if isPackedIndex(state, idx):
        assert (idx < problemType["NumIndicesC"])
        state["PackedC1IdxChars"].append("%s" % indexChars[idx])
        state["PackedC1IndicesX"].append(idx)

    # If dims are packed, then need to ensure a global vector load isn't split by a tensor dim
    # (since this could result in non-contiguous addresses)
    # Current implementation ensures that the vector load is not partial across the Free* boundary:
    # GlobalReadVectorWidth=1 will always meet this requirement.
    # (TODO - could make this more sophisticated if dims use default strides and are thus contiguous)
    packedC0 = len(state["PackedC0IdxChars"])>1
    packedC1 = len(state["PackedC1IdxChars"])>1

    bufferLoad = state["BufferLoad"] and state["KernelLanguage"] == "Assembly"
    if not bufferLoad:
      state["DirectToLds"] = False
      state["_UseSgprForGRO"] = False
      if state["PrefetchGlobalRead"] == 2:
        reject(state, "BufferLoad=0 does not support PrefetchGlobalRead=2")
        return

      if problemType["UseBias"]:
        reject(state, "BufferLoad=0 does not support UseBias due to no suppress no load.")
        return

    #These modes only work under certain conditions, apply them here:
    #  - The "NoLoad" loop is only generated if PrefetchGlobalRead>0
    #  - And Suppress does not work if GSU>1 for some reason
    if state["SuppressNoLoadLoop"] == 1:
      if not (bufferLoad and state["PrefetchGlobalRead"] == 1 and (state["GlobalSplitU"]==1)):
        state["SuppressNoLoadLoop"] = 0

    #print("PackedC0IdxChars", state["PackedC0IdxChars"])
    #print("PackedC1IdxChars", state["PackedC1IdxChars"])

    if state["EnableMatrixInstruction"]:
      if state["TransposeLDS"] == -1:
        if state["ProblemType"]["TLUA"] and state["ProblemType"]["TLUB"]:
          state["TransposeLDS"] = 0
        else:
          state["TransposeLDS"] = 1
      if state["TransposeLDS"] == 0:
        state["UnrollMajorLDSA"] = 0
        state["UnrollMajorLDSB"] = 0
      elif state["TransposeLDS"] == 1:
        state["UnrollMajorLDSA"] = not state["ProblemType"]["TLUA"]
        state["UnrollMajorLDSB"] = not state["ProblemType"]["TLUB"]
      elif state["TransposeLDS"] == 2:
        state["UnrollMajorLDSA"] = 1
        state["UnrollMajorLDSB"] = 1
    else: # mac instruction
      if state["UseDotInstruction"]:
        # dot2: force TLDS = 2
        if state["TransposeLDS"] not in [-1,2]:
          reject(state, "dot2 kernel does not support TransposeLDS != 2")
        state["TransposeLDS"] =  2
        state["UnrollMajorLDSA"] = True
        state["UnrollMajorLDSB"] = True
      else:
        state["TransposeLDS"] =  0
        state["UnrollMajorLDSA"] = False
        state["UnrollMajorLDSB"] = False

    if state["VectorWidthA"] == -1:
      if state["EnableMatrixInstruction"]:
        regPerElem = state["ProblemType"]["DataType"].numRegisters()
        optVW = int(4 // regPerElem)
        while 1:
          if state["MIWaveTile"][0] % optVW == 0:
            state["VectorWidthA"] = optVW
            break
          else:
            optVW //= 2
        if state["ProblemType"]["Sparse"]:
          state["VectorWidthA"] = 1
      else:
        state["VectorWidthA"] = 1

    if state["VectorWidthB"] == -1:
      if state["EnableMatrixInstruction"]:
        regPerElem = state["ProblemType"]["DataType"].numRegisters()
        optVW = int(4 // regPerElem)
        while 1:
          if state["MIWaveTile"][1] % optVW == 0:
            state["VectorWidthB"] = optVW
            break
          else:
            optVW //= 2
        if state["ProblemType"]["Sparse"]:
          state["VectorWidthB"] = 1
      else:
        state["VectorWidthB"] = 1

    if state["ProblemType"]["Sparse"] and not state["DirectToVgprSparseMetadata"]:
      state["VectorWidthMetadata"] = state["VectorWidthA"] if state["ProblemType"]["Sparse"] == 1 else state["VectorWidthB"]

    numBytes = state["ProblemType"]["DataType"].numBytes()
    isa = tuple(state["ISA"])
    state["enableLDSTrA"] = state["LDSTrInst"] and globalParameters["AsmCaps"][isa]["HasLDSTr"] and numBytes == 2 \
            and not state["UnrollMajorLDSA"] and not state["DirectToVgprA"]
    state["enableLDSTrB"] = state["LDSTrInst"] and globalParameters["AsmCaps"][isa]["HasLDSTr"] and numBytes == 2 \
            and not state["UnrollMajorLDSB"] and not state["DirectToVgprB"]

    if state["enableLDSTrA"]:
      state["VectorWidthA"] = 1

    if state["enableLDSTrB"]:
      state["VectorWidthB"] = 1

    if state["LDSTrInst"] and state["1LDSBuffer"] == 0:
      reject(state, "Current LDS Transpose implementation does not support two LDS buffers")
      return

    # if state["EnableMatrixInstruction"] and not state["SourceSwap"] and (state["VectorWidthA"] > 1 or state["VectorWidthB"] > 1):
    #   reject(state, "not implement VectorWidth without SourceSwap")

    # TT0,1 both must be multiples of VW, b/c of rC, rA, rB
    if state["EnableMatrixInstruction"]:
      if (state["MIWaveTile"][0] % state["VectorWidthA"]) != 0:
        reject(state, "MIWaveTile0(%u) should be multiple of VectorWidthA(%u)" % (state["MIWaveTile"][0], state["VectorWidthA"]))
        return
      if (state["MIWaveTile"][1] % state["VectorWidthB"]) != 0:
        reject(state, "MIWaveTile0(%u) should be multiple of VectorWidthB(%u)" % (state["MIWaveTile"][1], state["VectorWidthB"]))
        return

    if len(problemType["IndicesSummation"]) > 1:
      # not supported with multiple summations, bug is maybe something with
      # how stagger iteration is wrapped when unroll loop exits
      state["StaggerU"] = 0

    # Some restrictions for half:
    if state["KernelLanguage"] == "Assembly" \
      and state["ProblemType"]["DataType"].isHalf():

      if globalParameters["ArchCaps"][globalParameters["CurrentISA"]]["HasEccHalf"]:
        if not state["ProblemType"]["HighPrecisionAccumulate"] and state["AssertFree0ElementMultiple"] % 2 != 0:
          # beta-on-edge has AF0EM requirement except for HPA kernels
          reject(state, "Archs with HasEccHalf require AF0EM%2==0 except for HPA kernels")
          return

    if state["ConvertAfterDS"]:
        if (state["ProblemType"]["DataType"].isHalf() == False):
            reject(state, "ConvertAfterDS only support DataType half")
            return
        if (state["ProblemType"]["DataTypeA"].isAnyFloat8() == False) and (state["ProblemType"]["DataTypeB"].isAnyFloat8() == False):
            reject(state, "one of DataTypeA or DataTypeB need to be float8/float8_fnuz")
            return

    # DepthU == -1?
    if state["DepthU"] == -1:
      if state["ProblemType"]["ComputeDataType"].numBytes() < 4:
        depthuList = [256, 128, 64, 32]
      else:
        depthuList = [128, 64, 32, 16]
    else:
      depthuList = [state["DepthU"]]
    index = [0]
    backupValues = []
    for key, value in state.items():
      if isinstance(value, int) and value < 0:
        backupValues.append([key, value])
    while True:
      for backup in backupValues:
        state[backup[0]] = backup[1]
      state["ValidDepthU"] = True
      state["DepthU"]      = depthuList[index[0]]
      Solution.depthUIteration(state, index, depthuList, problemType, isa, bufferLoad, packedC0, packedC1)
      if state["Valid"] or (state["ValidDepthU"] and (not state["Valid"])):
        break
      index[0] += 1
      if index[0] >= len(depthuList):
        break
    if "ValidDepthU" in state:
      del state["ValidDepthU"]
 
  def depthUIteration(state, index, depthuList, problemType, isa, bufferLoad, packedC0, packedC1):
    ########################################
    # Auto search for DepthU starts here
    # Activates when DepthU == -1
    ########################################
    resetStaggerUStride = state["StaggerUStride"]
    resetLocalReadVectorWidth = state["LocalReadVectorWidth"]
    resetGlobalReadVectorWidthA = state["GlobalReadVectorWidthA"]
    resetGlobalReadVectorWidthB = state["GlobalReadVectorWidthB"]

    while True:
      userDepthU = depthuList[index[0]]
      state["StaggerUStride"] = resetStaggerUStride
      state["LocalReadVectorWidth"] = resetLocalReadVectorWidth
      state["GlobalReadVectorWidthA"] = resetGlobalReadVectorWidthA
      state["GlobalReadVectorWidthB"] =  resetGlobalReadVectorWidthB
      state["Valid"] = True
      ########################################
      # Initial DepthU
      ########################################
      depthU = userDepthU
      depthUA = depthUB = depthUM = depthU
      if state["ProblemType"]["Sparse"]:
        if state["ProblemType"]["Sparse"] == 2:
          depthUB = depthUB // 2
          depthUM = depthUB if state["DirectToVgprSparseMetadata"] else depthUB // 4
        else:
          depthUA = depthUA // 2
          depthUM = depthUA if state["DirectToVgprSparseMetadata"] else depthUA // 4
      state["_DepthU"] = state["DepthU"]# internal
      state["_DepthUA"] = depthUA# internal
      state["_DepthUB"] = depthUB# internal
      state["_DepthUMetadata"] = depthUM# internal

      Solution.checkAndAssignWaveSeparateGlobalRead(state, 'A')
      Solution.checkAndAssignWaveSeparateGlobalRead(state, 'B')
      if state["ProblemType"]["Sparse"]:
        if state["ProblemType"]["Sparse"] == 2:
          if not state["DirectToVgprSparseMetadata"]:
            Solution.checkAndAssignWaveSeparateGlobalRead(state, 'Metadata')
        else:
          if not state["DirectToVgprSparseMetadata"]:
            Solution.checkAndAssignWaveSeparateGlobalRead(state, 'Metadata')

      # Set up stagger shift:
      bpeAB = int(4*state["ProblemType"]["DataType"].numRegisters())
      # (1<<staggerStrideShift) is number of loop iterations to traverse the stride
      if state["StaggerU"] == 0:
        state["StaggerUMapping"] = 0
        state["StaggerUStride"] = 0

      if state["StaggerUStride"] == -1 or state["StaggerUStride"] < (state["DepthU"] * bpeAB):
        # (StaggerUStride) shoud be greater than or equal to (DepthU * bpeAB)
        state["StaggerUStride"] = state["DepthU"] * bpeAB

      state["_staggerStrideShift"] = (int)(math.ceil(math.log(state["StaggerUStride"] / (state["DepthU"] * bpeAB), 2)))

      def calcLdsPad(lrvw: int) -> int:
        ldsPadA = state["LdsPadA"]
        ldsPadB = state["LdsPadB"]
        optPadA = optPadB = lrvw
        readRegsA = readRegsB = lrvw * state["ProblemType"]["DataType"].numBytes() // 4
        if state["ProblemType"]["Sparse"]:
          if state["ProblemType"]["Sparse"] == 2:
            optPadB //= 2
            readRegsB //= 2
          else:
            optPadA //= 2
            readRegsA //= 2
        if (not globalParameters["AsmCaps"][isa]['HasWMMA']) and (readRegsA > 4 or readRegsB > 4):
          reject(state, "LocalReadVectorWidth results in attemping to read LDS larger than b128, reject")
          return
        if state["EnableMatrixInstruction"]:
          # for readRegs = 1 or 4, we need to double pad for MI16x16xNx1 to avoid bank conflict.
          if state["MatrixInstB"] == 1 and state["MatrixInstM"] == 16:
            if readRegsA == 4 or readRegsA == 1:
              optPadA *= 2
            if readRegsB == 4 or readRegsB == 1:
              optPadB *= 2
        if ldsPadA == -1:
          if not state["UnrollMajorLDSA"]:
            if state["EnableMatrixInstruction"]:
              ldsPadA = 0
              if state["MatrixInstB"] == 1 and state["MatrixInstM"] == 16:
                ldsPadA = ((16 * state["VectorWidthA"] * state["ProblemType"]["DataType"].numBytes() + state["MacroTile0"] * state["ProblemType"]["DataType"].numBytes() * state["LocalReadVectorWidth"]) % 128) // state["ProblemType"]["DataType"].numBytes()
              if state["GlobalReadVectorWidthA"] * state["ProblemType"]["DataType"].numBytes() == 32 and ldsPadA == 0:
                ldsPadA = 16 // state["ProblemType"]["DataType"].numBytes()
            else: # mac instruction
              if state["ProblemType"]["TLUA"]:
                ldsPadA = 0
              else:
                ldsPadA = state["VectorWidthA"]
          else:
            ldsPadA = max(state["GlobalReadVectorWidthA"],optPadA)
            ## turn-off padding for directToLds
            if state["DirectToLdsA"]:
              ldsPadA = 0
          assert(ldsPadA >= 0)

        if ldsPadB == -1:
          if not state["UnrollMajorLDSB"]:
            if state["EnableMatrixInstruction"]:
              ldsPadB = 0
              if state["MatrixInstB"] == 1 and state["MatrixInstM"] == 16:
                ldsPadB = ((16 * state["VectorWidthB"] * state["ProblemType"]["DataType"].numBytes() + state["MacroTile1"] * state["ProblemType"]["DataType"].numBytes() * state["LocalReadVectorWidth"]) % 128) // state["ProblemType"]["DataType"].numBytes()
              if state["GlobalReadVectorWidthB"] * state["ProblemType"]["DataType"].numBytes() == 32 and ldsPadB == 0:
                ldsPadB = 16 // state["ProblemType"]["DataType"].numBytes()
            else:
              if state["ProblemType"]["TLUB"]:
                ldsPadB = 0
              else:
                ldsPadB = state["VectorWidthB"]
          else:
            ldsPadB = max(state["GlobalReadVectorWidthB"],optPadB)
            if state["DirectToLdsB"]:
              ldsPadB = 0
          assert(ldsPadB >= 0)

        ldsPadM = state["LdsPadMetadata"]
        if state["ProblemType"]["Sparse"] and not state["DirectToVgprSparseMetadata"]:
          optPadM = (optPadB if state["ProblemType"]["Sparse"] == 2 else optPadA) // 4
          grvwM = (state["GlobalReadVectorWidthB"] if state["ProblemType"]["Sparse"] == 2 else state["GlobalReadVectorWidthA"])  // 4
          vwM = (state["VectorWidthB"] if state["ProblemType"]["Sparse"] == 2 else state["VectorWidthA"]) // 4

          if ldsPadM == -1:
            ldsPadM = 0
            if not state["ProblemType"]["TLUMetadata"]:
              if state["EnableMatrixInstruction"] and state["TransposeLDSMetadata"]:
                ldsPadM = max(grvwM, optPadM)
              else:
                ldsPadM = vwM
              ## turn-off padding for directToLds
              if state["EnableMatrixInstruction"] and state["TransposeLDSMetadata"] and state["DirectToLdsMetadata"]:
                ldsPadM = 0
          assert(ldsPadM >= 0)

        # set ldsPadA,B=0 for DirectToLds or DirectToVgpr
        # TODO: enable ldsPad for DirectToLds (if needed)
        if state["DirectToLds"] or state["DirectToVgprA"]:
          ldsPadA = 0
        if state["DirectToLds"] or state["DirectToVgprB"]:
          ldsPadB = 0

        return ldsPadA, ldsPadB, ldsPadM

      def calcLdsBlockSizePerPad(lrvw: int) -> int:
        LdsBlockSizePerPadA = state["LdsBlockSizePerPadA"]
        LdsBlockSizePerPadB = state["LdsBlockSizePerPadB"]
        tmpBpe = state["ProblemType"]["DataTypeA"].numBytes() if state["ConvertAfterDS"] else state["ProblemType"]["DataType"].numBytes()
        if LdsBlockSizePerPadA == -1:
          if state["EnableMatrixInstruction"]:
            if state["UnrollMajorLDSA"]:
              LdsBlockSizePerPadA = roundUpToNearestMultiple(state["_DepthUA"] * tmpBpe, 128)
              if state["_DepthUA"] * tmpBpe * state["VectorWidthA"] > 128:
                LdsBlockSizePerPadA = roundUpToNearestMultiple(state["_DepthUA"] * tmpBpe * state["VectorWidthA"], 128)
            else:
              if state["MatrixInstB"] == 1 and state["MatrixInstM"] == 16:
                LdsBlockSizePerPadA = state["MacroTile0"] * tmpBpe * lrvw
              else:
                LdsBlockSizePerPadA = 0
          else:
            LdsBlockSizePerPadA = 0
        tmpBpe = state["ProblemType"]["DataTypeB"].numBytes() if state["ConvertAfterDS"] else state["ProblemType"]["DataType"].numBytes()
        if LdsBlockSizePerPadB == -1:
          if state["EnableMatrixInstruction"]:
            if state["UnrollMajorLDSB"]:
              LdsBlockSizePerPadB = roundUpToNearestMultiple(state["_DepthUB"] * tmpBpe, 128)
              if state["_DepthUB"] * tmpBpe * state["VectorWidthB"] > 128:
                LdsBlockSizePerPadB = roundUpToNearestMultiple(state["_DepthUB"] * tmpBpe * state["VectorWidthB"], 128)
            else:
              if state["MatrixInstB"] == 1 and state["MatrixInstM"] == 16:
                LdsBlockSizePerPadB = state["MacroTile1"] * tmpBpe * lrvw
              else:
                LdsBlockSizePerPadB = 0
          else:
            LdsBlockSizePerPadB = 0

        # set LdsBlockSizePerPadA,B=0 for DirectToLds or DirectToVgpr
        if state["DirectToLds"] or state["DirectToVgprA"]:
          LdsBlockSizePerPadA = 0
        if state["DirectToLds"] or state["DirectToVgprB"]:
          LdsBlockSizePerPadB = 0

        return LdsBlockSizePerPadA, LdsBlockSizePerPadB

      def calcLdsNumBytes(ldsPadA: int, LdsBlockSizePerPadA: int, ldsPadB: int, LdsBlockSizePerPadB: int) -> int:
        bpeA = state["ProblemType"]["DataTypeA"].numBytes() if state["ConvertAfterDS"] else state["ProblemType"]["DataType"].numBytes()
        bpeB = state["ProblemType"]["DataTypeB"].numBytes() if state["ConvertAfterDS"] else state["ProblemType"]["DataType"].numBytes()
        ldsAlign = int(64 / state["ProblemType"]["DataType"].numRegisters())

        if state["UnrollMajorLDSA"]:
          ldsNumBytesA = (state["_DepthUA"] + ldsPadA) * state["MacroTileA"] * bpeA
        else:
          ldsNumBytesA = state["_DepthUA"] * (state["MacroTileA"] + ldsPadA) * bpeA
        padInterval = LdsBlockSizePerPadA
        if padInterval != 0:
          ldsNumBytesA = int((state["_DepthUA"] * state["MacroTileA"] * bpeA) / padInterval * (padInterval + ldsPadA * bpeA))
        ldsNumBytesAlignedA = roundUpToNearestMultiple(ldsNumBytesA, ldsAlign)

        # DirectToVgpr case, set 0 to lds related variables
        if state["DirectToVgprA"]:
          ldsNumBytesA = 0
          ldsNumBytesAlignedA = 0

        if state["UnrollMajorLDSB"]:
          ldsNumBytesB = (state["_DepthUB"] + ldsPadB) * state["MacroTileB"] * bpeB
        else:
          ldsNumBytesB = state["_DepthUB"] * (state["MacroTileB"] + ldsPadB) * bpeB
        padInterval = LdsBlockSizePerPadB
        if padInterval != 0:
          ldsNumBytesB = int((state["_DepthUB"] * state["MacroTileB"] * bpeB) / padInterval * (padInterval + ldsPadB * bpeB))
        ldsNumBytesAlignedB = roundUpToNearestMultiple(ldsNumBytesB, ldsAlign)

        # DirectToVgpr case, set 0 to lds related variables
        if state["DirectToVgprB"]:
          ldsNumBytesB = 0
          ldsNumBytesAlignedB = 0

        if state["ProblemType"]["Sparse"] and not state["DirectToVgprSparseMetadata"]:
          bpeAB = state["ProblemType"]["DataType"].numBytes()
          if state["UnrollMajorLDSMetadata"]:
            ldsNumBytesMetadata = (state["_DepthUMetadata"] + state["LdsPadMetadata"]) * state["MacroTileMetadata"]
          else:
            ldsNumBytesMetadata = state["_DepthUMetadata"] * (state["MacroTileMetadata"] + state["LdsPadMetadata"])
          ldsNumBytesMetadata = roundUp(ldsNumBytesMetadata / bpeAB) # metadata is in byte type. so divide ldsNumBytesMetadata by A,B's bpe
          padInterval = state["LdsBlockSizePerPadMetadata"]
          if padInterval != 0:
            ldsNumBytesMetadata = int(roundUp(state["_DepthUMetadata"] * state["MacroTileMetadata"] / bpeAB) / padInterval * (padInterval + state["LdsPadMetadata"]))
          ldsNumBytesAlignedMetadata = roundUpToNearestMultiple(ldsNumBytesMetadata, ldsAlign) * bpeAB
          ldsNumBytesMetadata = ldsNumBytesMetadata * bpeAB
        else:
          ldsNumBytesMetadata = 0
          ldsNumBytesAlignedMetadata = 0

        return ldsNumBytesA, ldsNumBytesAlignedA, ldsNumBytesB, ldsNumBytesAlignedB, ldsNumBytesMetadata, ldsNumBytesAlignedMetadata

      # Default LocalReadVectorWidth
      if state["EnableMatrixInstruction"]:
        autoLRVW = 0
        if state["LocalReadVectorWidth"] == -1:
          autoLRVW = 1
          if state["TransposeLDS"] and (not state["DirectToLds"]):
            state["LocalReadVectorWidth"] = 16 // state["ProblemType"]["DataType"].numBytes()
          else:
            if state["ProblemType"]["Sparse"] and state["MIInputPerThread"] * state["ProblemType"]["DataType"].numBytes() > 16:
              state["LocalReadVectorWidth"] = 16 // state["ProblemType"]["DataType"].numBytes()
            else:
              state["LocalReadVectorWidth"] = state["MIInputPerThread"]
        else:
          if state["ProblemType"]["Sparse"] and state["MIInputPerThread"] * state["ProblemType"]["DataType"].numBytes() > 16:
            if state["LocalReadVectorWidth"] < state["MIInputPerThread"] // 2:
              reject(state, "LocalReadVectorWidth < %u" %(state["MIInputPerThread"] // 2))
          elif not state["ProblemType"]["Sparse"] and (not globalParameters["AsmCaps"][isa]["HasMFMA_f8f6f4"] or state["MatrixInstK"] <= 32):
            if state["LocalReadVectorWidth"] < state["MIInputPerThread"]:
              reject(state, "LocalReadVectorWidth < %u" %(state["MIInputPerThread"]))
          if state["LocalReadVectorWidth"] > state["MIInputPerThread"] and not state["TransposeLDS"]:
            reject(state, "LocalReadVectorWidth require Transpose LDS")

        if autoLRVW:
          if state["LocalReadVectorWidth"] // state["MIInputPerThread"] > 1:
            if (state["DepthU"] // state["MatrixInstK"] <= state["LocalReadVectorWidth"] // state["MIInputPerThread"]):
              # if only have 1 iteration with wider local read, reduce LRVW to have better scheduling (at least 2 iterations)
              state["LocalReadVectorWidth"] //= 2
          if state["LocalReadVectorWidth"] // state["MIInputPerThread"] > 1:
            padA, padB, padM = calcLdsPad(state["LocalReadVectorWidth"])
            ldsBlockSizePerPadA, ldsBlockSizePerPadB = calcLdsBlockSizePerPad(state["LocalReadVectorWidth"])
            ldsNumBytesA, ldsNumBytesAlignedA, ldsNumBytesB, ldsNumBytesAlignedB, ldsNumBytesMetadata, ldsNumBytesAlignedMetadata = calcLdsNumBytes(padA, ldsBlockSizePerPadA, padB, ldsBlockSizePerPadB)
            if (ldsNumBytesAlignedA + ldsNumBytesAlignedB) > globalParameters["MaxLDS"]:
              state["LocalReadVectorWidth"] //= 2
      else:
        if state["UseDotInstruction"]:
          # dot2: LRVW should be equal to NumDotElements * InnerUnroll
          if state["LocalReadVectorWidth"] not in [-1, state["NumDotElements"] * state["InnerUnroll"]]:
            reject(state, "dot kernel requires LocalReadVectorWidth = NumDotElements(%u) * InnerUnroll(%u)" \
              % (state["NumDotElements"], state["InnerUnroll"]))
            return
          state["LocalReadVectorWidth"] = state["NumDotElements"] * state["InnerUnroll"]
        else:
          if state["LocalReadVectorWidth"] == -1:
            state["LocalReadVectorWidth"] = 1

      def calcOptGRVW(lrvw: int, unrollMajorLDS: bool, datatype: DataType) -> int:
        # with UnrollMajorLDS, GRVW need to less or equal than LRVW to have conflict free LDS read with padding.
        optGRVW = lrvw if unrollMajorLDS else 4 / datatype.numRegisters()
        if optGRVW * datatype.numBytes() > 16:
          optGRVW = 16 // datatype.numBytes()
        return optGRVW

      def calSwizzleK(state, tc):
        return 16 // state[f"MIInputPerThread{tc}"] // state["ProblemType"][f"DataType{tc}"].numBytes()

      genGRVWA = False
      genGRVWB = False
      # Default GlobalReadVectorWidthA
      if state["EnableMatrixInstruction"]:
        if state["GlobalReadVectorWidthA"] < 0:
          genGRVWA = True
          if state["GlobalReadVectorWidthA"] == -2:
            if state["MatrixInstBM"] == 1 and state["MIWaveTile"][0] == 1 and state["MIWaveGroup"][0] == 1 and state["ProblemType"]["TLUA"]:
              state["GlobalReadVectorWidthA"] = 1
            else:
              reject(state, "GRVWA=-2 is set for skinny MT")
          elif state["GlobalReadVectorWidthA"] == -1:
            if state["ProblemType"]["SwizzleTensorA"]:
              state["GlobalReadVectorWidthA"] = state["MIInputPerThreadA"] * calSwizzleK(state, "A")
            else:
              optGRVW = calcOptGRVW(state["LocalReadVectorWidth"], state["UnrollMajorLDSA"], state["ProblemType"]["DataTypeA"])
              curGRVW = 1
              state["GlobalReadVectorWidthA"] = int(curGRVW)
              while (curGRVW <= optGRVW):
                if (state["MacroTile0"]*state["_DepthUA"]//state["NumThreads"]) % curGRVW == 0:
                  state["GlobalReadVectorWidthA"] = int(curGRVW)
                curGRVW *= 2
      else:
        # dot2
        if state["UseDotInstruction"]:
          coalescedA = state["MacroTile0"] if state["ProblemType"]["TLUA"] else state["DepthU"]
          elementPerThread = state["MacroTile0"] * state["DepthU"] // state["NumThreads"]
          if state["GlobalReadVectorWidthA"] == -1:
            for tmpGRVW in [8,4,2,1]:
              if coalescedA % tmpGRVW == 0 and elementPerThread % tmpGRVW == 0:
                state["GlobalReadVectorWidthA"] = tmpGRVW
                break
          if coalescedA % state["GlobalReadVectorWidthA"] != 0 or elementPerThread % state["GlobalReadVectorWidthA"] != 0:
            reject(state, "dot2: non-valid GRVWA")
            return
          # TODO: support edge shiftptr to release this constraint.
          if state["ProblemType"]["TLUA"]:
            state["AssertFree0ElementMultiple"] = max(state["AssertFree0ElementMultiple"], state["GlobalReadVectorWidthA"])
        

      # Default GlobalReadVectorWidthB
      if state["EnableMatrixInstruction"]:
        if state["GlobalReadVectorWidthB"] < 0:
          genGRVWB = True
          if state["GlobalReadVectorWidthB"] == -2:
            if state["MatrixInstBN"] == 1 and state["MIWaveTile"][1] == 1 and state["MIWaveGroup"][1] == 1 and state["ProblemType"]["TLUB"]:
              state["GlobalReadVectorWidthB"] = 1
            else:
              reject(state, "GRVWB=-2 is set for skinny MT")
          elif state["GlobalReadVectorWidthB"] == -1:
            if state["ProblemType"]["SwizzleTensorB"]:
              state["GlobalReadVectorWidthB"] = state["MIInputPerThreadB"] * calSwizzleK(state, "B")
            else:
              optGRVW = calcOptGRVW(state["LocalReadVectorWidth"], state["UnrollMajorLDSB"], state["ProblemType"]["DataTypeB"])
              curGRVW = 1
              state["GlobalReadVectorWidthB"] = int(curGRVW)
              while (curGRVW <= optGRVW):
                if (state["MacroTile1"]*state["_DepthUB"]//state["NumThreads"]) % curGRVW == 0:
                  state["GlobalReadVectorWidthB"] = int(curGRVW)
                curGRVW *= 2
      else:
        # dot2
        if state["UseDotInstruction"]:
          coalescedB = state["MacroTile1"] if state["ProblemType"]["TLUB"] else state["DepthU"]
          elementPerThread = state["MacroTile1"] * state["DepthU"] // state["NumThreads"]
          if state["GlobalReadVectorWidthB"] == -1:
            for tmpGRVW in [8,4,2,1]:
              if coalescedB % tmpGRVW == 0 and elementPerThread % tmpGRVW == 0:
                state["GlobalReadVectorWidthB"] = tmpGRVW
                break
          if coalescedB % state["GlobalReadVectorWidthB"] != 0 or elementPerThread % state["GlobalReadVectorWidthB"] != 0:
            reject(state, "dot2: non-valid GRVWB")
            return
          # TODO: support edge shiftptr to release this constraint.
          if state["ProblemType"]["TLUB"]:
            state["AssertFree1ElementMultiple"] = max(state["AssertFree1ElementMultiple"], state["GlobalReadVectorWidthB"])

      #for tensor swizzling, we calculate pack-k to achieve buffer_load_dwordx4
      for tc in ("A", "B",):
        if state["ProblemType"][f"SwizzleTensor{tc}"]:
          if not state["EnableMatrixInstruction"]:
            reject(state, f"Tensor {tc} swizzling supports MI only")
          # Print rejection reason instead of force set
          # 16 means bytes of buffer_load_dwordx4
          SwizzlePackK = calSwizzleK(state, tc)
          if state[f"GlobalReadVectorWidth{tc}"] != state[f"MIInputPerThread{tc}"] * SwizzlePackK:
            GRVW_TC = state[f"GlobalReadVectorWidth{tc}"]
            MIInPerThread = state[f"MIInputPerThread{tc}"]
            reject(state, f"SwizzleTensor{tc} doesn't support GRVW{tc} ({GRVW_TC}) != MIInputPerThread{tc} ({MIInPerThread}) * {SwizzlePackK}")

      if state["ProblemType"]["SwizzleTensorA"]:
        if not state["DirectToVgprA"]:
          reject(state, f"Tensor A swizzling requires DirectToVgprA")
        if not state["ProblemType"]["TransposeA"]:
          reject(state, f"Tensor A swizzling supports TN or TT only")

      if state["ProblemType"]["SwizzleTensorB"]:
        if not state["DirectToVgprB"]:
          reject(state, f"Tensor B swizzling requires DirectToVgprB")
        if state["ProblemType"]["TransposeB"]:
          reject(state, f"Tensor B swizzling supports TN or NN only")

        # TODO- NN fails validation due to DTVB + Tail-Loop is not working correctly
        if not (state["ProblemType"]["TransposeA"] and not state["ProblemType"]["TransposeB"]):
          reject(state, f"Tensor B swizzling supports TN only")

      # Force GRVW the same when UnrollLoopSwapGlobalReadOrder = 1.
      if genGRVWA and state["UnrollLoopSwapGlobalReadOrder"] == 1:
        state["GlobalReadVectorWidthA"] = min(state["GlobalReadVectorWidthA"], state["GlobalReadVectorWidthB"])
      if genGRVWB and state["UnrollLoopSwapGlobalReadOrder"] == 1:
        state["GlobalReadVectorWidthB"] = min(state["GlobalReadVectorWidthA"], state["GlobalReadVectorWidthB"])

      # reject - VW too big
      if (state["VectorWidthA"] * state["ProblemType"]["DataType"].numBytes()) > 16:
        reject(state, "VWA * DataType.numBytes() > 16")
      if (state["VectorWidthB"] * state["ProblemType"]["DataType"].numBytes()) > 16:
        reject(state, "VWB * DataType.numBytes() > 16")

      # reject - GRVW too big
      if (state["GlobalReadVectorWidthA"] * state["ProblemType"]["DataTypeA"].numBytes()) > 16:
        reject(state, "GRVWA * DataTypeA.numBytes() > 16")
      if (state["GlobalReadVectorWidthB"] * state["ProblemType"]["DataTypeB"].numBytes()) > 16:
        reject(state, "GRVWB * DataTypeB.numBytes() > 16")

      ########################################
      # Search DepthU
      # Inputs:
      #  - depthU, userDepthU, state["LocalSplitU"], state["InnerUnroll"], state["MacroTile0/1"], state["GlobalReadVectorWidth"]
      #  - state["MatrixInstK"], ...
      # Outputs:
      #  - totalVectorsCoalescedA, totalVectorsCoalescedB, totalElementsPerpA, totalElementsPerpB, state["DepthU"]
      #######################################
      while True: # exit criteria at end
        validDepthU = True

        # how many elements to load
        if state["ProblemType"]["TLUA"]: # NT/NN
          totalElementsCoalescedA = state["MacroTileA"]
          totalElementsPerpA = depthUA
          if state["DirectToVgprA"]:
            totalElementsCoalescedA *= state["MIWaveGroup"][1]
        else: # TN/TT
          totalElementsCoalescedA = depthUA
          totalElementsPerpA = state["MacroTileA"]
          if state["DirectToVgprA"]:
            totalElementsPerpA *= state["MIWaveGroup"][1]

        if state["ProblemType"]["TLUB"]: # NT/TT
          totalElementsCoalescedB = state["MacroTileB"]
          totalElementsPerpB = depthUB
          if state["DirectToVgprB"]:
            totalElementsCoalescedB *= state["MIWaveGroup"][0]
        else: # TN/NN
          totalElementsCoalescedB = depthUB
          totalElementsPerpB = state["MacroTileB"]
          if state["DirectToVgprB"]:
            totalElementsPerpB *= state["MIWaveGroup"][0]

        totalElementsA = totalElementsCoalescedA * totalElementsPerpA
        totalElementsB = totalElementsCoalescedB * totalElementsPerpB
        if state["ProblemType"]["Sparse"] and not state["DirectToVgprSparseMetadata"]:
          if state["ProblemType"]["TLUMetadata"]:
            totalElementsCoalescedM = state["MacroTileMetadata"]
            totalElementsPerpM = depthUM
          else:
            totalElementsCoalescedM = depthUM
            totalElementsPerpM = state["MacroTileMetadata"]
          totalElementsM = totalElementsCoalescedM * totalElementsPerpM

        tva = totalElementsA // state["GlobalReadVectorWidthA"]
        if not Solution.setGlobalReadVectorWidth(state, "A", tva, state["GlobalReadVectorWidthA"]):
          validDepthU = False
        tvb = totalElementsB // state["GlobalReadVectorWidthB"]
        if not Solution.setGlobalReadVectorWidth(state, "B", tvb, state["GlobalReadVectorWidthB"]):
          validDepthU = False

        if state["EnableMatrixInstruction"] and state["GlobalReadVectorWidthA"]:
          partialA = state["ProblemType"]["TLUA"] and (state["AssertFree0ElementMultiple"] % state["GlobalReadVectorWidthA"] != 0)
          if partialA:
            glvwAlimit = 16 // state["ProblemType"]["DataType"].numBytes()
            if state["SourceSwap"]:
              matrixInstM = (state["MatrixInstM"] * state["MatrixInstBM"]) if (state["MatrixInstM"] == 4) else state["MatrixInstM"]
              glvwAlimit = matrixInstM * state["VectorWidthA"]
            else:
              matrixInstN = (state["MatrixInstN"] * state["MatrixInstBN"]) if (state["MatrixInstN"] == 4) else state["MatrixInstN"]
              glvwAlimit  = state["MIOutputVectorWidth"] * (state["WavefrontSize"] // matrixInstN)
            if state["ProblemType"]["DataType"].numRegisters() == 0.25:
              glvwAlimit = max(glvwAlimit, 4)

            # reduce GLVA if GLVA larger than MIOVW
            if state["GlobalReadVectorWidthA"] > glvwAlimit:
              tva = totalElementsA // glvwAlimit
              if not Solution.setGlobalReadVectorWidth(state, "A", tva, glvwAlimit):
                validDepthU = False

        if state["EnableMatrixInstruction"] and state["GlobalReadVectorWidthB"]:
          partialB = state["ProblemType"]["TLUB"] and (state["AssertFree1ElementMultiple"] % state["GlobalReadVectorWidthB"] != 0)
          if partialB:
            glvwBlimit = 16 // state["ProblemType"]["DataType"].numBytes()
            if state["SourceSwap"]:
              matrixInstM = (state["MatrixInstM"] * state["MatrixInstBM"]) if (state["MatrixInstM"] == 4) else state["MatrixInstM"]
              glvwBlimit  = state["MIOutputVectorWidth"] * (state["WavefrontSize"] // matrixInstM)
            # else:  # use origin shiftptr for B
            #   matrixInstN = (state["MatrixInstN"] * state["MatrixInstBN"]) if (state["MatrixInstN"] == 4) else state["MatrixInstN"]
            if state["ProblemType"]["DataType"].numRegisters() == 0.25:
              glvwBlimit = max(glvwBlimit, 4)

            # reduce GLVB if GLVB larger than MIOVW
            if state["GlobalReadVectorWidthB"] > glvwBlimit:
              tvb = totalElementsB // glvwBlimit
              if not Solution.setGlobalReadVectorWidth(state, "B", tvb, glvwBlimit):
                validDepthU = False

        if validDepthU and state["KernelLanguage"] == "Assembly":
          if globalParameters["ArchCaps"][globalParameters["CurrentISA"]]["HasEccHalf"]:
            if state["ProblemType"]["DataType"].numRegisters() == 0.5 and (not state["ProblemType"]["HighPrecisionAccumulate"]):
                if state["GlobalReadVectorWidthA"] == 1 or state["GlobalReadVectorWidthB"] == 1:
                  reject(state, "HalfEcc requires HPA if glvw = 1")
                  break

        if state["ProblemType"]["Sparse"] and not state["DirectToVgprSparseMetadata"]:
          grvw = 1
          vw = 1
          if state["ProblemType"]["Sparse"] == 2:
            grvw = state["GlobalReadVectorWidthB"] // 4
            vw = state["VectorWidthB"] // 4
            if state["GlobalReadVectorWidthB"] % 4 != 0:
              reject(state, "Sparse B requires GRVWB %% 4 == 0, current GRVWB is %u"%state["GlobalReadVectorWidthB"])
              break
          else:
            grvw = state["GlobalReadVectorWidthA"] // 4
            vw = state["VectorWidthA"] // 4
            if state["GlobalReadVectorWidthA"] % 4 != 0:
              reject(state, "Sparse A requires GRVWA %% 4 == 0, current GRVWA is %u"%state["GlobalReadVectorWidthA"])
              break


          tvm = totalElementsM // grvw

          if not Solution.setGlobalReadVectorWidth(state, "Metadata", tvm, grvw):
            validDepthU = False

          if state["EnableMatrixInstruction"] and state["GlobalReadVectorWidthMetadata"]:
            partialM = True
            if state["ProblemType"]["Sparse"] == 2:
              partialM = state["ProblemType"]["TLUMetadata"] and (state["AssertFree1ElementMultiple"] % state["GlobalReadVectorWidthB"] != 0)
            else:
              partialM = state["ProblemType"]["TLUMetadata"] and (state["AssertFree0ElementMultiple"] % state["GlobalReadVectorWidthA"] != 0)

            if partialM:
              glvwMlimit = 16
              if state["SourceSwap"]:
                matrixInstM = (state["MatrixInstM"] * state["MatrixInstBM"]) if (state["MatrixInstM"] == 4) else state["MatrixInstM"]
                glvwMlimit = matrixInstM * vw
              else:
                if state["ProblemType"]["Sparse"] == 1:
                  matrixInstN = (state["MatrixInstN"] * state["MatrixInstBN"]) if (state["MatrixInstN"] == 4) else state["MatrixInstN"]
                  glvwMlimit  = state["MIOutputVectorWidth"] * (state["WavefrontSize"] // matrixInstN)

              # reduce GLVMetadata if GLVMetadata larger than MIOVW
              if state["GlobalReadVectorWidthMetadata"] > glvwMlimit:
                tvm = totalElementsM // glvwMlimit
                if not Solution.setGlobalReadVectorWidth(state, "Metadata", tvm, glvwMlimit):
                  validDepthU = False

        if state["ProblemType"]["Sparse"] and state["DirectToVgprSparseMetadata"]:
          if state["VectorWidthA"] > 1 or state["VectorWidthB"] > 1 :
            reject(state, "Not implement DTVSM with VW>1")
            break

        # Now convert elements to vectors based on GlobalReadVectorWidth
        GlobalReadVectorWidthA = state["GlobalReadVectorWidthA"]
        GlobalReadVectorWidthB = state["GlobalReadVectorWidthB"]
        totalVectorsCoalescedA = totalElementsCoalescedA // GlobalReadVectorWidthA
        totalVectorsCoalescedB = totalElementsCoalescedB // GlobalReadVectorWidthB

        if validDepthU:
          if not state["ProblemType"]["TLUA"]:
            if depthUA < state["GlobalReadVectorWidthA"]:
              validDepthU = False

          if not state["ProblemType"]["TLUB"]:
            if depthUB < state["GlobalReadVectorWidthB"]:
              validDepthU = False

          if state["ProblemType"]["Sparse"] and not state["DirectToVgprSparseMetadata"]:
            if not state["ProblemType"]["TLUMetadata"]:
              if depthUM < state["GlobalReadVectorWidthMetadata"]:
                validDepthU = False
        # this depthU is valid, done unless user wants to double (for TN)
        if validDepthU:
          state["DepthU"] = depthU
          break

        # this depthU not valid
        else:
          reject(state, "No valid DepthU found")
          state["ValidDepthU"] = False
          break
      ########################################
      # end DepthU loop
      ########################################
      if state["Valid"]:
        break
      index[0] += 1
      if index[0] >= len(depthuList):
        break
    if not state["Valid"]:
      return
    assert(state["DepthU"]> 0)

    if state["ScheduleIterAlg"] == 2:
      state["InnerUnroll"] = state["DepthU"] // state["MatrixInstK"]
      state["PrefetchLocalRead"] = 1
      state["ExpandPointerSwap"] = 1
      state["1LDSBuffer"] = 1
      print2("\nSet SIA=2, force PrefetchLocalRead=1, ExpandPointerSwap=1, 1LDSBuffer=1")

    if state["ExpandPointerSwap"] == 1:
      # Pointer swap only used if PGR==1 or (PGR>1 and double/double complex) - so set ExpandPointerSwap=0 here
      # So far, EPS=1 and PGR>1 works only with double/double complex.
      #if not (bufferLoad and state["PrefetchGlobalRead"] == 1):
      if not (bufferLoad and ( state["PrefetchGlobalRead"] == 1 \
              or (state["PrefetchGlobalRead"] > 1 and \
                  (state["ProblemType"]["DataType"].isDouble() or state["ProblemType"]["DataType"].isDoubleComplex()))
              or (state["ProblemType"]["Sparse"] and state["PrefetchGlobalRead"] > 0))):
        state["ExpandPointerSwap"] = 0

    # Default GlobalStoreVectorWidth
    if state["StoreVectorWidth"] == -1:
      if state["SourceSwap"]:
        state["StoreVectorWidth"] = state["VectorWidthA"]
      else:
        if state["EnableMatrixInstruction"]:
          state["StoreVectorWidth"] = state["MIOutputVectorWidth"]
          if state["VectorWidthA"] * state["MIOutputVectorWidth"] <= 4 / state["ProblemType"]["DestDataType"].numRegisters():
            state["StoreVectorWidth"] = state["VectorWidthA"] * state["MIOutputVectorWidth"]
          if state["LocalSplitU"] > 1:
            state["StoreVectorWidth"] = state["VectorWidthA"]
        else:
          state["StoreVectorWidth"] = state["VectorWidthA"]

    if state["EnableMatrixInstruction"]:
      if state["SourceSwap"]:
        if ((state["VectorWidthA"] % state["StoreVectorWidth"]) != 0):
          reject(state, "MFMA SourceSwap mode doesn't support vwA(%u) with svw(%u)" % (state["VectorWidthA"], state["StoreVectorWidth"]))
          return
      else:
        if (((state["VectorWidthA"] * state["MIOutputVectorWidth"]) % state["StoreVectorWidth"]) != 0):
          reject(state, "MFMA non-SourceSwap mode doesn't support miovw(%u) with svw(%u)" % (state["VectorWidthA"]*state["MIOutputVectorWidth"], state["StoreVectorWidth"]))
          return

    # LocalSplitU too large?
    # dot2: every NumWaveSplitK threads compute the same element.
    numElementsPerWorkGroup = state["MacroTile0"]*state["MacroTile1"]*state["NumWaveSplitK"]

    if numElementsPerWorkGroup < state["NumThreads"]:
      reject(state, "NumElementsPerWorkGroup %u < NumThreads %u; reduce LocalSplitU" \
          % (numElementsPerWorkGroup, state["NumThreads"]))
      return

    state["NumElementsPerThread"] = numElementsPerWorkGroup // state["NumThreads"]
    state["GlobalWriteVectorWidth"] = min(state["VectorWidthA"], state["NumElementsPerThread"] )
    if state["NumElementsPerThread"] % state["GlobalWriteVectorWidth"] != 0:
      reject(state, "LSU NumElementsPerThread %u not divisible into GWVW %u" \
          % (state["NumElementsPerThread"], state["GlobalWriteVectorWidth"]))
      return
    state["NumGlobalWriteVectorsPerThread"] = state["NumElementsPerThread"] \
        // state["GlobalWriteVectorWidth"]


    # LocalSplitU but can't NumThreads%MacroTile doesn't support sideways store
    if state["LocalSplitU"] > 1:
      if not state["SourceSwap"] and state["StoreVectorWidth"] > state["VectorWidthA"]:
        reject(state, "LSU and non-SourceSwap doesn't support StoreVectorWidth(%u)>VWA(%u)." \
            % (state["StoreVectorWidth"], state["VectorWidthA"]))
        return
      if not (state["ProblemType"]["ComputeDataType"].isSingle() or state["ProblemType"]["ComputeDataType"].isInt32()):
        reject(state, "TODO: LSU doesn't support ComputeDataType!=(single or Int32).")
        return
      if state["StoreRemapVectorWidth"] > 0:
        reject(state, "TODO: LSU doesn't support StoreRemapVectorWidth>0.")
        return
      if state["NumThreads"] % state["MacroTile0"] != 0:
        reject(state, "LocalSplitU but NumThreads=%u not divisible by MT0=%u for sideways store" \
            % (state["NumThreads"], state["MacroTile0"]))
        return
      if state["MacroTile0"]*state["MacroTile1"] % state["NumThreads"] != 0:
        reject(state, "LocalSplitU but MT0*MT1=%u elements doesn't divide into NumThreads=%u" \
            % (state["MacroTile0"]*state["MacroTile1"], state["NumThreads"]))
        return

    # GlobalSplitU doesn't work with some other things:
    if state["GlobalSplitU"] > 1:
      # added GSU support for DGEMM
      supported = \
        (state["ProblemType"]["DataType"].isSingle()) or \
        (state["ProblemType"]["DataType"].isDouble() and state["BufferStore"]) or \
        (state["ProblemType"]["DestDataType"].isInt32()) or \
        (state["KernelLanguage"] == "Assembly" and
            (state["ProblemType"]["DataType"].isHalf() and not state["ProblemType"]["HighPrecisionAccumulate"]) or
            (state["_GlobalAccumulation"])
        )
      if not supported:
        reject(state, "GlobalSplitU only compatible with single or asm and (half or mixed) precision")
        return

    if state["ProblemType"]["DataType"].isHalf() and state["KernelLanguage"] == "Assembly":
      if state["GlobalSplitU"] > 1 and (not state["_GlobalAccumulation"]):
        if state["AssertFree0ElementMultiple"] < 2:
          reject(state, "Assembly GSU half requires AF0EM>=2 (for atomics on edge tiles)")

        if state["EnableMatrixInstruction"] and globalParameters["AsmCaps"][isa]['HasWMMA']:
          reject(state, "Half WMMA doesn't support single buffer GSU")
          return
    # dot2 limitation
    if state["UseDotInstruction"]:
      if state["EnableMatrixInstruction"]:
        reject(state, "dot inst is for mac kernel!")
      if not bufferLoad:
        reject(state, "dot2 kernel only support bufferLoad!")
      if not (state["ProblemType"]["DataType"].isHalf() and state["ProblemType"]["HighPrecisionAccumulate"]):
        reject(state, "dot2 kernel only support DataType fp16 with HPA")
      if state["InnerUnroll"] not in [1,2,4]:
        reject(state, "dot2 kernel requires InnerUnroll = 1,2 or 4")
      if state["NumWaveSplitK"] not in [1,2,4,8,16,32,64]:
        reject(state, "Unsupported NumWaveSplitK value. Need to be power of 2 and does not exceed 64.")
      if state["DepthU"] % (state["LocalReadVectorWidth"] * state["NumWaveSplitK"]) != 0:
        reject(state, "Non-valid DepthU for dot2 kernel, need to be multiple of (LocalReadVectorWidth * InnerUnroll * NumWaveSplitK) for atomics")
      # TODO: Need to consider VectorWidth in LraTileAssignmentVALU
      if state["VectorWidthA"] != 1 or state["VectorWidthB"] != 1:
        reject(state, "dot2 kernel requires VectorWidth = 1")
      # TODO: Need to remap VGPR index
      if (state["ThreadTile0"] != 1 or state["ThreadTile1"] != 1) and state["InnerUnroll"] > 1: 
        reject(state, "dot2 kernel does not support wider local read with ThreadTile > 1")
      if state["ScheduleLocalWrite"] != 1:
        reject(state, "dot2 kernel requires ScheduleLocalWrite = 1")
      if state["LocalSplitU"] != 1:
        reject(state, "dot2 kernel requires LocalSplitU = 1")
      if state["ProblemType"]["Sparse"]:
        reject(state, "dot2 kernel does not support sparse gemm")

    if state["ProblemType"]["Sparse"] and not state["DirectToVgprSparseMetadata"]:
      state["NumLoadsCoalescedMetadata"] = 1

    if not Solution.setGlobalLoadTileDimClassic(state, "A", state["NumLoadsA"], \
        totalVectorsCoalescedA, totalElementsPerpA, depthUA):
      return
    if not Solution.setGlobalLoadTileDimClassic(state, "B", state["NumLoadsB"], \
        totalVectorsCoalescedB, totalElementsPerpB, depthUB):
      return

    if state["ProblemType"]["Sparse"] and not state["DirectToVgprSparseMetadata"]:
      if state["ProblemType"]["TLUMetadata"]:
        totalElementsCoalescedM = state["MacroTileMetadata"]
        totalElementsPerpM = depthUM
      else:
        totalElementsCoalescedM = depthUM
        totalElementsPerpM = state["MacroTileMetadata"]
      totalElementsM = totalElementsCoalescedM * totalElementsPerpM
      GlobalReadVectorWidthMetadata = state["GlobalReadVectorWidthMetadata"]
      totalVectorsCoalescedM = totalElementsCoalescedM // GlobalReadVectorWidthMetadata

      # Try to enlarge GLVW for metadata
      bGlobalReadVectorWidthMetadata = state["GlobalReadVectorWidthMetadata"]
      if state["ProblemType"]["Sparse"] == 2:
        GlobalReadVectorWidth = min(state["GlobalReadVectorWidthMetadata"] * state["NumLoadsPerpendicularB"], depthUM) #sum all need read
        tvm = totalElementsM // GlobalReadVectorWidth
        if not Solution.setGlobalReadVectorWidth(state, "Metadata", tvm, GlobalReadVectorWidth):
          #fallback
          tvm = totalElementsM // bGlobalReadVectorWidthMetadata
          Solution.setGlobalReadVectorWidth(state, "Metadata", tvm, bGlobalReadVectorWidthMetadata)

        GlobalReadVectorWidthMetadata = state["GlobalReadVectorWidthMetadata"]
        if GlobalReadVectorWidthMetadata == 0:
          GlobalReadVectorWidthMetadata = 1
        totalVectorsCoalescedM = totalElementsCoalescedM // GlobalReadVectorWidthMetadata
        totalVectorsM = totalElementsM // GlobalReadVectorWidthMetadata
      else:
        GlobalReadVectorWidth = min(state["GlobalReadVectorWidthMetadata"] * state["NumLoadsPerpendicularA"], depthUM) #sum all need read
        tvm = totalElementsM // GlobalReadVectorWidth
        if not Solution.setGlobalReadVectorWidth(state, "Metadata", tvm, GlobalReadVectorWidth):
          #fallback
          tvm = totalElementsM // bGlobalReadVectorWidthMetadata
          Solution.setGlobalReadVectorWidth(state, "Metadata", tvm, bGlobalReadVectorWidthMetadata)

        GlobalReadVectorWidthMetadata = state["GlobalReadVectorWidthMetadata"]
        if GlobalReadVectorWidthMetadata == 0:
          GlobalReadVectorWidthMetadata = 1
        totalVectorsCoalescedM = totalElementsCoalescedM // GlobalReadVectorWidthMetadata
        totalVectorsM = totalElementsM // GlobalReadVectorWidthMetadata

      if not Solution.setGlobalLoadTileDimClassic(state, "Metadata", state["NumLoadsMetadata"], \
          totalVectorsCoalescedM, totalElementsPerpM, depthUM):
        return

    # TODO
    if (0 and state["LSCA"] % state["GlobalReadVectorWidthA"] != 0):
      reject(state, "lsca % grvw != 0")
      return
    if (0 and state["LSPA"] % state["GlobalReadVectorWidthA"] != 0):
      reject(state, "lspa % grvw != 0")
      return
    if (0 and state["LSCB"] % state["GlobalReadVectorWidthB"] != 0):
      reject(state, "lscb % grvw != 0")
      return
    if (0 and state["LSPB"] % state["GlobalReadVectorWidthB"] != 0):
      reject(state, "lspb % grvw != 0")
      return

    state["LVCA"] = roundupRatio(state["LSCA"] , state["GlobalReadVectorWidthA"])
    state["LVPA"] = roundupRatio(state["LSPA"] , state["GlobalReadVectorWidthA"])
    state["LVCB"] = roundupRatio(state["LSCB"] , state["GlobalReadVectorWidthB"])
    state["LVPB"] = roundupRatio(state["LSPB"] , state["GlobalReadVectorWidthB"])

    if state["ProblemType"]["Sparse"] and not state["DirectToVgprSparseMetadata"]:
      state["LVCMetadata"] = roundupRatio(state["LSCMetadata"] , state["GlobalReadVectorWidthMetadata"])
      state["LVPMetadata"] = roundupRatio(state["LSPMetadata"] , state["GlobalReadVectorWidthMetadata"])

    for tc in ('A','B'):
      if problemType["TLU%s"%tc]:
        pos = problemType["IndexAssignments%s"%tc].index(problemType["Index01%s"%tc])
      else:
        pos = problemType["IndexAssignments%s"%tc].index(problemType["IndexUnroll"])

    # Some of these might become 0?
    if 0:
      print("info: ", pvar(state, "LVCA"), pvar(state, "LVPA"), \
            pvar(state, "LVCB"), pvar(state, "LVPB"))

    # lds buffer size for A, B
    if state["KernelLanguage"] == "Source" and \
       state["LdsPadA"] != state["LdsPadB"]:
      reject(state, "Source KernelLanguage only supports LdsPadA == LdsPadB")
      return

    # NoTailLoop parameter initialization.
    # If ASEM is multiple of DepthU TailLoop will not be used.
    # Unless kernel is Stream-K; Stream-K always requires TailLoop to handle work division.
    state["NoTailLoop"] = False
    if state["AssertSummationElementMultiple"] % state["DepthU"] == 0 and state["StreamK"] == 0:
      state["NoTailLoop"] = True

    # Determine if we can load directly-to-Vgpr
    # need to check after state["LocalReadVectorWidth"] = -1 is resolved
    if state["DirectToVgprA"]:
      if not Solution.isDirectToVgprDoable(state, 'A'):
        return  # rejected
    if state["DirectToVgprB"]:
      if not Solution.isDirectToVgprDoable(state, 'B'):
        return  # rejected

    ########################################
    # LDS
    ########################################

    state["TransposeLDSMetadata"] = False if state["ProblemType"]["Sparse"] == 2 else True
    state["UnrollMajorLDSMetadata"] = False if state["ProblemType"]["Sparse"] == 2 else True

    if state["ProblemType"]["Sparse"] and not state["DirectToVgprSparseMetadata"]:
      state["UnrollMajorLDSMetadata"] = state["TransposeLDSMetadata"] and (not state["ProblemType"]["TLUMetadata"])


    auto_LdsBlockSizePerPadA_for_mix = 0
    if state["LdsBlockSizePerPadA"] == -1:
      auto_LdsBlockSizePerPadA_for_mix = 1
    auto_LdsBlockSizePerPadB_for_mix = 0
    if state["LdsBlockSizePerPadB"] == -1:
      auto_LdsBlockSizePerPadB_for_mix = 1
    state["LdsBlockSizePerPadA"], state["LdsBlockSizePerPadB"] = calcLdsBlockSizePerPad(state["LocalReadVectorWidth"])

    if state["LdsBlockSizePerPadMetadata"] == -1:
      state["LdsBlockSizePerPadMetadata"] = state["LdsBlockSizePerPadA"]

    if state["EnableMatrixInstruction"]:
      if state["LdsBlockSizePerPadA"]:
        if state["UnrollMajorLDSA"]:
          if state["LdsBlockSizePerPadA"] % (state["_DepthUA"] * state["ProblemType"]["DataTypeA"].numBytes()) != 0:
            reject(state, "reject: LdsBlockSizePerPadA %u %% depthU %u x bpeA != 0" % (state["LdsBlockSizePerPadA"],state["_DepthUA"]))
          if (state["LdsBlockSizePerPadA"] // (state["_DepthUA"] * state["ProblemType"]["DataType"].numBytes())) % state["LSPA"] != 0 and \
              state["LSPA"] % (state["LdsBlockSizePerPadA"] // (state["_DepthUA"] * state["ProblemType"]["DataType"].numBytes())) != 0:
            reject(state, "can't pad by addrVgpr or instOffset")

      if state["LdsBlockSizePerPadB"]:
        if state["UnrollMajorLDSB"]:
          if state["LdsBlockSizePerPadB"] % state["_DepthUB"] * state["ProblemType"]["DataTypeB"].numBytes() != 0:
            reject(state, "reject: LdsBlockSizePerPadB %u %% depthU %u x bpeB != 0" % (state["LdsBlockSizePerPadB"],state["_DepthUB"]))
          if (state["LdsBlockSizePerPadB"] // (state["_DepthUB"] * state["ProblemType"]["DataType"].numBytes())) % state["LSPB"] != 0 and \
              state["LSPB"] % (state["LdsBlockSizePerPadB"] // (state["_DepthUB"] * state["ProblemType"]["DataType"].numBytes())) != 0:
            reject(state, "can't pad by addrVgpr or instOffset")
    else:
      if not state["UseDotInstruction"] and (state["UnrollMajorLDSA"] or state["UnrollMajorLDSB"]):
        reject(state, "didn't support UnrollMajorLDS in VALU mode yet (except for dot2 kernel)")
      if state["LdsBlockSizePerPadA"] != 0 or state["LdsBlockSizePerPadB"] != 0:
        reject(state, "didn't support LdsBlockSizePerPad in VALU mode yet")

    def checkLdsBlockSizePerPad(tc):
      """
        Simulated to calculate the local write address and local write offset, to check the pad will be added correctly or not.

          Expected: (address + offset) // lbspp == (address // lbspp + offset // lbspp)

          # offset is an array, the amount is 'NLP x NLC x (grvw / width of ds write instruction)'
          # Assumed Wave ID is zero

        refer KernelWriterAssembly.lwaFirstOffset and KernelWriterAssembly.lwaTileAssignment  to come out the address (local write address).
        refer KereelWriterAssembly.localWriteDo and KereelWriteAssembly.calculateLdsWriteOffset to come out the offset (local write offset).
      """

      def caculateLdsWriteOffset(perp, para, sPerp, sPara, tlu, uMLds, bpe, tc, idx):
        mask = 0
        #print "tc ", tc, " perp ", perp, " para ", para, " sPerp ", sPerp, " sPara ", sPara
        lscaOffset = para * state["LSC%s"%tc]
        perp_masked = perp
        perp_rem = 0
        lspaOffset = perp_masked * state["LSP%s"%tc]
        rem = 0

        # Add component offset to interleave from different regs
        # and compute mysterious "i"
        assert(sPerp==0 or sPara==0)

        if tlu != uMLds:
          lspaOffset += sPerp & mask
          lscaOffset += sPara
          rem = (sPerp & ~mask)
        else:
          lscaOffset += sPara
          lspaOffset += sPerp
          rem = 0

        lds_stride = state["_DepthU%s"%tc] if uMLds else state["MacroTile%d"%idx]
        if tlu != uMLds:
          lspaOffset *= lds_stride
          lspaOffset += rem + perp_rem
        else:
          lscaOffset *= lds_stride
          lscaOffset += rem

        offsetElements = (lspaOffset + lscaOffset)
        offsetBytes = offsetElements * bpe
        return offsetBytes

      def caculateLdsWriteAddress(tc, idx, serial, tlu, uMLds, grvw, bpe):
        id = serial
        if state["WaveSeparateGlobalRead%s"%tc]:
          id = id % state["WavefrontSize"]

        q = id // state["LVC%s"%tc]
        r = id % state["LVC%s"%tc]

        #assumed wave id = 0
        if state["WaveSeparateGlobalRead%s"%tc] == 2:
          q *= state["NumLoadsPerpendicular%s"%tc]*state["NumThreads"]//state["WavefrontSize"]

        if tlu:
          t = r
          u = q
          t *= grvw
        else:
          t = q
          u = r
          u *= grvw

        address = 0
        if uMLds:
          address = (state["_DepthU%s"%tc] * t + u) * bpe
        else:
          address = (state["MacroTile%s"%tc] * u + t) * bpe

        return address

      def findValidWriteBlockWidth(nwcv, bpe, bpr):
        localWriteWidth = nwcv * bpe // bpr
        if localWriteWidth < 1:
          localWriteWidth = (1.0* nwcv * bpe )/bpr
        blockWidth = 0
        for bw in [8, 4, 2, 1, 0.5, 0.25]:
          if localWriteWidth >= bw:
            blockWidth = bw
            break
        if blockWidth == 0:
          reject(state, "invalid local write block width")

        return blockWidth

      def subCheckLdsBlockSizePerPad(tc, idx):
        lbspp = state["LdsBlockSizePerPad%s"%tc]
        bpe = state["ProblemType"]["DataType"].numBytes()
        bpr = 4
        vw = state["GlobalReadVectorWidth%s"%tc]
        tlu = state["ProblemType"]["TLU%s"%tc]
        uMLds = state["UnrollMajorLDS%s"%tc]
        if tlu != uMLds: # NT no transpose
          wtc = False # Vector
          # writeCoal indicates writes should be done in the coal dim or else perp
          nwcv = vw
          nwpv = 1
        else: # TN yes transpose
          wtc = True
          nwcv = 1
          nwpv = vw

        blockWidth = findValidWriteBlockWidth(nwcv, bpe, bpr)
        nwcvpi = int(blockWidth * bpr / bpe)

        serials = []
        if tlu != uMLds:
          serials = range(0, state["LVC%s"%tc])
        else:
          serials = [state["LVC%s"%tc] * q for q in range(0, max(1, state["NumThreads"] // state["LVC%s"%tc]))]

        for serial in serials:
          address = caculateLdsWriteAddress(tc, idx, serial, tlu, uMLds, vw, bpe)
          for perp in range(0, state["NumLoadsPerpendicular%s"%tc]):
            for para in range(0, state["NumLoadsCoalesced%s"%tc]):
              sPerp = 0
              sPara = 0
              for s in range(0, vw // nwcvpi):
                if tlu != uMLds:
                  if wtc:
                    sPerp = s
                else:
                  if wtc:
                    sPara = s
                offset = caculateLdsWriteOffset(perp, para, sPerp, sPara, tlu, uMLds, bpe, tc, idx)
                lLdsBlocks = (address + offset) // lbspp
                rLdsBlocks = address // lbspp + offset // lbspp
                if 0: #Debug
                  pad = state["LdsPad%s"%tc]
                  print(tc, serial, state["UnrollMajorLDS%s"%tc], perp, para, bpe, lbspp, address, offset, address // lbspp * pad * bpe, offset // lbspp * pad * bpe, lLdsBlocks, rLdsBlocks, address + offset + lLdsBlocks * pad * bpe, address + offset + rLdsBlocks * pad * bpe)
                if lLdsBlocks != rLdsBlocks:
                  return False
        return True

      if state["LdsBlockSizePerPad%s"%tc] != 0 and state["LdsPad%s"%tc] != 0:
        idx = 0 if tc == "A" else 1
        auto_LdsBlockSizePerPad_for_mix = auto_LdsBlockSizePerPadA_for_mix if tc == "A" else auto_LdsBlockSizePerPadB_for_mix

        if not subCheckLdsBlockSizePerPad(tc, idx):
          if auto_LdsBlockSizePerPad_for_mix:
            printWarning("Padded address is inconisstent, set LdsBlockSizePerPad%s=0."%tc)
            state["LdsBlockSizePerPad%s"%tc] = 0
          else:
            reject(state, "%s's padded address is inconisstent"%tc)

    if(not (state["CustomKernelName"] and state["CustomKernelName"] != "")): #don't check the custom kernel.
      checkLdsBlockSizePerPad("A")
      checkLdsBlockSizePerPad("B")

    # Determine if we can load directly-to-LDS.
    # Transpose requires a trip through registers to perform the transpose so can't use DirectToLdsA
    # LDS loads always write 4 bytes apart so can use only 4-byte operations
    #   TODO - for doubles we need to add something special here?
    # The matrix must not require transposing since that is done by reading to VGPR and writing in different order
    # The LSC (load size coalesced) must load some multiple of 256 bytes since that is what each DirectToLds load provides
    # Note for these matrices LSC is same as MacroTile dim
    # MatrixInstruction rules:
    # DirectToLDS is supported for TLU=0  (make sure transposeLDS=1)
    # LDS (load size coalesced) * LSPA must load some multiple of 256 bytes.
    # No longer support loadX2/loadx4 .
    if state["DirectToLds"]:
      if (not state["DirectToVgprA"]) and Solution.isDirectToLdsDoable(state, 'A'):
        state["DirectToLdsA"] = True
        state["LocalWriteUseSgprA"] = True
        state["LdsPadA"] = 0
        printWarning("DirectToLdsA enabled, set LdsPadA=0.")
        #print("DirectToLdsA", state["DirectToLdsA"])

      if (not state["DirectToVgprB"]) and Solution.isDirectToLdsDoable(state, 'B'):
        state["DirectToLdsB"] = True
        state["LocalWriteUseSgprB"] = True
        state["LdsPadB"] = 0
        printWarning("DirectToLdsB enabled, set LdsPadB=0.")
        #print("DirectToLdsB", state["DirectToLdsB"])

      # Update parent variable so kernel display is accurate
      state["DirectToLds"] = state["DirectToLdsA"] or state["DirectToLdsB"]
      if state["1LDSBuffer"] == -1 and state["DirectToLds"]:
        #1LDS buffer must be 0 for DirectToLdsA
        state["1LDSBuffer"] = 0

      # Re-check DTV + WaveGroup after DTL is confirmed
      if state["DirectToLds"]:
        if state["DirectToVgprA"] and state['MIWaveGroup'][1] > 1:
          reject(state, "DirectToLds + (DirectToVgprA + WaveGroups along N-Dim) is not supported yet")
          return False
        if state["DirectToVgprB"] and state['MIWaveGroup'][0] > 1:
          reject(state, "DirectToLds + (DirectToVgprB + WaveGroups along M-Dim) is not supported yet")
          return False

    # set NoLdsWriteCode if (DirectToVgpr or DirectToLds)A+B is enabled
    state["NoLdsWriteCode"] = False
    if (state["DirectToVgprA"] or state["DirectToLdsA"]) and (state["DirectToVgprB"] or state["DirectToLdsB"]):
      state["NoLdsWriteCode"] = True

    # calculate ldsPad
    state["LdsPadA"], state["LdsPadB"], state["LdsPadMetadata"] = calcLdsPad(state["LocalReadVectorWidth"])

    if state["GlobalReadVectorWidthA"] * state["ProblemType"]["DataType"].numBytes() == 32 and state["LdsPadA"] == 16 // state["ProblemType"]["DataType"].numBytes():
      if auto_LdsBlockSizePerPadA_for_mix:
        state["LdsBlockSizePerPadA"] = 128
    assert(state["LdsPadA"] >= 0)

    if state["GlobalReadVectorWidthB"] * state["ProblemType"]["DataType"].numBytes() == 32 and state["LdsPadB"] == 16 // state["ProblemType"]["DataType"].numBytes():
      if auto_LdsBlockSizePerPadB_for_mix:
        state["LdsBlockSizePerPadB"] = 128
    assert(state["LdsPadB"] >= 0)

    if (state["UnrollMajorLDSA"] or state["UnrollMajorLDSB"]) and (not state["EnableMatrixInstruction"]) and (not state["UseDotInstruction"]):
        reject(state, "UnrollMajorLDS Supports only in EnableMatrixInstruction=1 or dot2 kernel")

    ldsNumBytesA, ldsNumBytesAlignedA, ldsNumBytesB, ldsNumBytesAlignedB, ldsNumBytesMetadata, ldsNumBytesAlignedMetadata = calcLdsNumBytes(state["LdsPadA"], state["LdsBlockSizePerPadA"], state["LdsPadB"], state["LdsBlockSizePerPadB"])
    state["LdsOffsetA_Blk"]=0
    state["LdsOffsetB_Blk"]=0
    # todo, can the alignment be a power of 2?
    state["LdsOffsetA"] = 0
    if state["PrefetchGlobalRead"]:
      state["LdsNumElementsAlignedA"] = ldsNumBytesAlignedA
      state["LdsNumElementsAlignedB"] = ldsNumBytesAlignedB
      state["LdsNumElementsAlignedMetadata"] = ldsNumBytesAlignedMetadata
      state["LdsOffsetMetadata"] = state["LdsOffsetA"] + state["LdsNumElementsAlignedA"]
      state["LdsOffsetB"] = state["LdsOffsetMetadata"] + state["LdsNumElementsAlignedMetadata"]

      offsetBlk = state["LdsOffsetB"] +  ldsNumBytesAlignedB
      if offsetBlk > 0:
        offsetBlk = int(2**(math.ceil(math.log(offsetBlk, 2))))

      state["LdsOffsetA_Blk"] = offsetBlk
      state["LdsOffsetMetadata_Blk"] = state["LdsOffsetA_Blk"] + state["LdsNumElementsAlignedA"]
      state["LdsOffsetB_Blk"] = state["LdsOffsetMetadata_Blk"] + state["LdsNumElementsAlignedMetadata"]
      ldsNumBytesAB = state["LdsOffsetB_Blk"] + ldsNumBytesB
    else:
      state["LdsOffsetMetadata"] = ldsNumBytesAlignedA
      state["LdsOffsetB"] = state["LdsOffsetMetadata"] + ldsNumBytesAlignedMetadata
      ldsNumBytesAB = state["LdsOffsetB"] + ldsNumBytesB

    # lds buffer size for reduction
    # if User want to control the LDS usage, we may open this para in the future
    ldsNumBytesReduction = state["LocalSplitU"] * state["MacroTile0"] * state["MacroTile1"] * state["ProblemType"]["ComputeDataType"].numBytes() if state["LocalSplitU"] > 1 else 0
    state["LocalSplitUReuseLDS"] = 1
    if ldsNumBytesReduction > globalParameters["MaxLDS"]:
      state["LocalSplitUReuseLDS"] = math.ceil(ldsNumBytesReduction / globalParameters["MaxLDS"])
      # reserve all the LDS to LSU.
      ldsNumBytesReduction = globalParameters["MaxLDS"]

    # lds max occupancy
    ldsSizeOccupancy = globalParameters["DeviceLDS"] // state["MaxOccupancy"]
    ldsNumBytesOccupancy = ldsSizeOccupancy

    #print("LdsOffsetB", state["LdsOffsetB"])
    #print("LdsOffsetMetadata", state["LdsOffsetMetadata"])
    #if state["PrefetchGlobalRead"]:
    #  print("LdsOffsetA_BLK", state["LdsOffsetA_Blk"])
    #  print("LdsOffsetB_BLK", state["LdsOffsetB_Blk"])
    #  print("LdsOffsetMetadata_BLK", state["LdsOffsetMetadata_Blk"])

    if state["EnableMatrixInstruction"]:
      if state["DirectToLds"] and state["1LDSBuffer"]:
        reject(state, "1LDSBuffer must be 0 for directToLds")

    if state["1LDSBuffer"] == -1:
      if ldsNumBytesAB  <= max(ldsSizeOccupancy,32768) or \
          (state["ProblemType"]["ComputeDataType"].numBytes() * state["MacroTile0"] * state["MacroTile1"] > 32768*4 and \
            not (ldsNumBytesAB > globalParameters["DeviceLDS"])):
        state["1LDSBuffer"] = 0
      else:
        state["1LDSBuffer"] = 1

    if state["1LDSBuffer"]:
      if not state["PrefetchGlobalRead"]:
        reject(state, "PGR=0 already use 1 LDS buffer only")
      # Should be able to support as long as NO scheduleLocalWrite
      if (not state["ScheduleIterAlg"] == 2) and (not state["ScheduleIterAlg"] == 3) and (state["ScheduleLocalWrite"]):
        reject(state, "1LDSBuffer only support SIA2 or SIA3, or SIA1 without SLW")
      state["LdsOffsetB"] = ldsNumBytesAlignedA
      state["LdsOffsetMetadata"] = state["LdsOffsetB"] + ldsNumBytesAlignedB
      ldsNumBytesAB = ldsNumBytesAlignedA + ldsNumBytesAlignedB + ldsNumBytesMetadata

    # lds size is the greater of the two
    ldsNumBytes = max(ldsNumBytesAB, ldsNumBytesReduction, ldsNumBytesOccupancy)

    if state["NumElementsPerBatchStore"] == -1:
      if ldsNumBytes > 32768 or \
          state["ProblemType"]["ComputeDataType"].numBytes() * state["MacroTile0"] * state["MacroTile1"] > 32768*4:
        state["NumElementsPerBatchStore"] = 0
        state["StorePriorityOpt"] = 0
        state["StoreSyncOpt"] = 0
        state["GroupLoadStore"] = 0
      else:
        state["NumElementsPerBatchStore"] = 16 if not state["ProblemType"]["DataType"].numBytes() == 8 else 1

    # Mbsk prefetch optimization
    if state["_GlobalAccumulation"] != 'MultipleBufferSingleKernel':
        state["MbskPrefetchOpt"] = 0
    elif state["MbskPrefetchOpt"] == -1:
      numStoreElements = state["NumElementsPerThread"] // state["StoreVectorWidth"]
      state["MbskPrefetchOpt"] = 1 if numStoreElements >= 4 else 0
    if state["MbskPrefetchOpt"] == 1:
      state["NumMbskPrefetchElements"] = 16
      storeRegs = state["StoreVectorWidth"] * state["ProblemType"]["ComputeDataType"].numRegisters()
      # exceed 16*4 = 64 VPGRs
      if storeRegs > 4:
        state["NumMbskPrefetchElements"] //= storeRegs // 4
      if state["NumElementsPerBatchStore"] == 0 or state["NumElementsPerBatchStore"] > state["NumMbskPrefetchElements"]:
          state["NumElementsPerBatchStore"] = state["NumMbskPrefetchElements"]

    if state["StoreRemapVectorWidth"] == -1:
      # use de_read_b64 as default in storeRemap to avoid bank conflict
      defaultRemap = 8 // state["ProblemType"]["DestDataType"].numBytes()
      defaultRemap = max(defaultRemap, state["MacroTile0"]//state["WavefrontSize"])
      if state["EnableMatrixInstruction"]:
        ldsRemapPad = max(defaultRemap, state["MIOutputVectorWidth"])
        ldsNumElementsRemapC = (state["MacroTile0"]+ldsRemapPad)* state["MatrixInstN"] * state["MIWaveGroup"][1]
        computeBytes = state["ProblemType"]["ComputeDataType"].numBytes()
        # max(GlobalAccumulation case, non-GlobalAccumulation)
        ldsNumElementsRemapC = max(ldsNumElementsRemapC, ldsNumElementsRemapC * (computeBytes / state["ProblemType"]["DestDataType"].numBytes()))
        ldsSize = ldsNumElementsRemapC * state["ProblemType"]["DestDataType"].numBytes()
        if not math.log(state["MacroTile0"],2).is_integer() or \
            ldsSize > globalParameters["MaxLDS"] or \
            state["SourceSwap"] or \
            (state["GlobalSplitU"] > 1) and (state["_GlobalAccumulation"] != 'MultipleBuffer') or \
            state["MatrixInstBN"] > 1 and state["MatrixInstN"] == 4 :
          state["StoreRemapVectorWidth"] = 0
        else:
          state["StoreRemapVectorWidth"] = defaultRemap
      else:
        state["StoreRemapVectorWidth"] = defaultRemap

      if not state["SourceSwap"]:
        if not state["StoreRemapVectorWidth"]:
          reject(state, "reject to reduce number of kernels")
        elif state["VectorWidthA"] > 1:
          reject(state, "reject to reduce number of kernels")

    # GuaranteeNoPartial
    if state["ProblemType"]["TLUA"]:
      state["GuaranteeNoPartialA"] = state["AssertFree0ElementMultiple"]%state["GlobalReadVectorWidthA"]==0
    else:
      state["GuaranteeNoPartialA"] = True

    if state["ProblemType"]["TLUB"]:
      state["GuaranteeNoPartialB"] = state["AssertFree1ElementMultiple"]%state["GlobalReadVectorWidthB"]==0
    else:
      state["GuaranteeNoPartialB"] = True

    state["GuaranteeNoPartialMetadata"] = False if state["ProblemType"]["Sparse"] else True

    # SourceSwap
    if state["StoreRemapVectorWidth"]:
      if state["SourceSwap"]:
        reject(state, "SourceSwap not compatible with StoreRemap")
        return
      if state["VectorWidthA"] > 1 or state["VectorWidthB"] > 1:
        reject(state, "VW>1 not compatible with StoreRemap")
        return

    # Sparse problem
    if state["ProblemType"]["Sparse"]:
      if state["PrefetchGlobalRead"] and not state["ExpandPointerSwap"]:
        reject(state, "Sparse A kernel only support PGR with EPS=1.")
        return
      if state["EnableMatrixInstruction"] and state["MIArchVgpr"]:
        reject(state, "Sparse A kernel does not support MIArchVgpr yet.")
        return
      # Not Support Feature
      if state["ProblemType"]["Sparse"] == 1 and state["SourceSwap"] :
        reject(state, "Sparse A kernel cannot support SourceSwap.")
        return
      else:
        if state["ProblemType"]["Sparse"] == 2 and not state["SourceSwap"]:
          reject(state, "Sparse B kernel must enable SourceSwap.")
          return
      state["AssertSummationElementMultiple"] = 8

    # check if need to use lds init Acc vgprs
    state["LdsInitCVgprs"] = False
    if globalParameters["ArchCaps"][isa]["HasAccCD"] and \
         state["EnableMatrixInstruction"] and state["StorePriorityOpt"] and \
         state["ProblemType"]["DataType"].isDouble():
      state["LdsInitCVgprs"] = True

    # force MIArchVgpr when using WMMA
    if state["EnableMatrixInstruction"] and globalParameters["AsmCaps"][isa]["HasWMMA"]:
      state["MIArchVgpr"] = True

    if state["MIArchVgpr"]:
      if not state["EnableMatrixInstruction"]:
        reject(state, "MIArchVgpr only support for MatrixInstruction")
        return

      if globalParameters["AsmCaps"][isa]["HasMFMA"]:
        if not (state["ProblemType"]["ComputeDataType"].isDouble() or \
                state["ProblemType"]["ComputeDataType"].isSingle() or \
                (state["ProblemType"]["ComputeDataType"].isHalf() and state["ProblemType"]["HighPrecisionAccumulate"]) or \
                state["ProblemType"]["ComputeDataType"].isInt32() or \
                state["ProblemType"]["ComputeDataType"].isComplex()):
          reject(state, "MIArchVgpr now only support fp64, fp64c, fp32, fp32c, fp16, int8 MatrixInstruction.")
          return

    #check not support cases and calculate lds resources
    ldsNumBytesRemapC = 0
    if state["StoreRemapVectorWidth"]:
      if not state["EnableMatrixInstruction"]:
        reject(state, "storeRemap only support MatrixInstruction kernel")
        return
      if ((state["GlobalSplitU"] > 1) and (state["_GlobalAccumulation"] != 'MultipleBuffer' or state["_GlobalAccumulation"] == 'MultipleBufferSingleKernel')) or \
        (state["GlobalSplitU"] == 1 and state["_GlobalAccumulation"] == 'SingleBuffer'):
        reject(state, "storeRemap doesn't support GlobalSplitU yet, except GSU algorithm 2")
        return
      if packedC0 or packedC1:
        reject(state, "storeRemap doesn't support packedC0 and packedC1 yet")
        return
      if state["MatrixInstBN"] > 1 and state["MatrixInstN"] == 4:
        reject(state, "storeRemap doesn't support MI4x4 multi blocks in N direction yet")
        return
      if not math.log(state["MacroTile0"],2).is_integer():
        reject(state, "storeRemap only supports power-of-2 MT0")
        # TODO - this return should be here, but this is a hotfix,
        # Somehow we have a "Validation Failed" kernel in rocBLAS now (SRVW=4 and MT0=96) and this will stop the whole building process
        # Actions: 1. Hotfix, comment out this "return" temporarily for that invalidated kernel
        #          2. Remove / replace that invalidated kernel
        #          3. Put back this return
        #          4. How to design a better way to prevent from invalid kernel in rocBLAS?
        # return

      storeInstMinWidth = 1 # minimum dwordx1
      storeInstMaxWidth = 4 # maximum dwordx4
      srMinVw = max(storeInstMinWidth, int(storeInstMinWidth/state["ProblemType"]["DestDataType"].numRegisters()))
      numReg  = max(state["ProblemType"]["DestDataType"].numRegisters(), state["ProblemType"]["ComputeDataType"].numRegisters())

      srMaxVw = int(storeInstMaxWidth/numReg)
      # FIXME: Add StoreRemapVectorWidthGSU and StoreRemapVectorWidthNonGSU
      while srMaxVw < state["StoreRemapVectorWidth"]:
        state["StoreRemapVectorWidth"] = state["StoreRemapVectorWidth"] // 2
      if srMinVw > state["StoreRemapVectorWidth"] or srMaxVw < state["StoreRemapVectorWidth"]:
        reject(state, "StoreRemapVectorWidth %u is not allowed for this data type" % state["StoreRemapVectorWidth"])
        return

      if state["StoreRemapVectorWidth"] * state["WavefrontSize"] < state["MacroTile0"]:
        reject(state, "storeRemap: Per wave single global write instruction doesn't enough to write one M column." + \
               " Please use larger StoreRemapVectorWidth.")
        return
      if (state["MacroTile0"]*state["MatrixInstN"])//state["MIWaveGroup"][0] < state["StoreRemapVectorWidth"]*state["WavefrontSize"]:
        reject(state, "storeRemap: number elements of lds less than per wave per local read elements." + \
               " Please use smaller StoreRemapVectorWidth.")
        return
      ldsRemapPad = max(state["StoreRemapVectorWidth"],state["MIOutputVectorWidth"])
      ldsNumBytesRemapC = (state["MacroTile0"]+ldsRemapPad)* state["MatrixInstN"] * state["MIWaveGroup"][1]


      computeBytes = state["ProblemType"]["ComputeDataType"].numBytes()
      multiplierGSU = computeBytes
      if state["ProblemType"]["DestDataType"].numBytes() > state["ProblemType"]["DataType"].numBytes():
        # Determine ratio of output to input element size.
        # SRVW remaps output so we need to scale up resources.
        multiplier = state["ProblemType"]["DestDataType"].numBytes()
      else:
        multiplier = state["ProblemType"]["DataType"].numBytes()

      ldsNumBytesRemapCNonGSU = ldsNumBytesRemapC * multiplier
      ldsNumBytesRemapCGSU    = ldsNumBytesRemapC * multiplierGSU
      ldsNumBytesRemapC *= max(multiplier, multiplierGSU)


      #print("ldsNumBytesRemapC=%u" % ldsNumBytesRemapC)

      # if LDS is bound by RemapC (SRVW), then 1LDSBuffer actually doesn't help in SIA3
      # since LDS usage couldn't be reduced
      if state["1LDSBuffer"] and (state["ScheduleIterAlg"] == 3) and (ldsNumBytes < ldsNumBytesRemapC):
        # TODO- Remove this DataType test condition,
        # Currently we do this test is just because we don't want to affect existing logic in rocBLAS
        if state["ProblemType"]["DataType"].isInt8():
          reject(state, "LDS usage is bound be StoreRemap, thus 1LDSBuffer wouldn't have any help. Skip.")
          return

      ldsNumBytes = max(ldsNumBytes, ldsNumBytesRemapC)

    state["LdsOffsetBias"] = 0  # TODO: ldsBiasOffset = ldsNumBytesAB
    state["LdsOffsetBiasNonGSU"] = 0
    state["LdsOffsetBiasGSU"] = 0

    # TODO: Should change name to LdsOffsetEpilogue or something.
    if state["StoreRemapVectorWidth"]:
      state["LdsOffsetBiasNonGSU"] = ldsNumBytesRemapCNonGSU
      state["LdsOffsetBiasGSU"] = ldsNumBytesRemapCGSU
      state["LdsOffsetBias"] = ldsNumBytesRemapC

    # Calcualte the correct LDS usages
    def calcEpilogueTurns(factorDims: List) -> int:
      divisor = state["SubGroup0"] * state["SubGroup1"]
      # d will be a list containing 0 or 1
      maxTurn = 0
      for d in range(len(factorDims)):
        turn = math.ceil(state["MacroTile%d"%d] / divisor)
        maxTurn = max(maxTurn, turn)
      return maxTurn

    # Calc the required LDS
    vecDT = VectorDataTypes()
    biasDim = state["ProblemType"]["UseBias"]
    savDim = state["ProblemType"]["UseScaleAlphaVec"]
    sAB = state["ProblemType"]["UseScaleAB"] == "Vector"
    # Calc LDS for Bias
    maxTurn = 0
    if biasDim == 1:
      maxTurn = calcEpilogueTurns([0])
    elif biasDim == 2:
      maxTurn = calcEpilogueTurns([1])
    elif biasDim == 3:
      maxTurn = calcEpilogueTurns([0, 1])
    vecDT.bias(0).turn = maxTurn
    vecDT.bias(1).turn = maxTurn
  
    # Calc LDS for SAV
    maxTurn = 0
    if savDim == 1:
      maxTurn = calcEpilogueTurns([0])
    elif savDim == 2:
      maxTurn = calcEpilogueTurns([1])
    elif savDim == 3:
      maxTurn = calcEpilogueTurns([0, 1])
    vecDT.scaleAlpha(0).turn = maxTurn
    vecDT.scaleAlpha(1).turn = maxTurn

    # Calc LDS for ScaleA, ScaleB
    if sAB:
      vecDT.scaleA.turn = calcEpilogueTurns([0])
      vecDT.scaleB.turn = calcEpilogueTurns([1])

    epilogueSize = 0
    # Bias
    if state["ProblemType"]["UseBias"]:
      # Currently all offsets starts from 0
      if state["ProblemType"]["Gradient"]:
        if state["ProblemType"]["BiasSrc"] == "A":
          tile01 = state["ProblemType"]["Index01A"]
        elif state["ProblemType"]["BiasSrc"] == "B":
          tile01 = state["ProblemType"]["Index01B"]
        elif state["ProblemType"]["BiasSrc"] == "D":
          tile01 = -1
        else:
          assert 0 and "Unsupported tile01 for bias lds calculation."
        # Don't need to calculate lds bias
        if tile01 > -1:
          maxKId = state["WavefrontSize"] // ((state["MatrixInstM"] if (tile01 == 0) else state["MatrixInstN"]) * state["MatrixInstB"])
          for dataType in state["ProblemType"]["BiasDataTypeList"]:
            epilogueSize = max(epilogueSize, state["MacroTile%d"%tile01] * maxKId * dataType.numBytes()) # TODO- GetTurn ?
      else:
        epilogueSize = state["NumThreads"] * state["ProblemType"]["ComputeDataType"].numBytes() * vecDT.bias(0).turn
    # Calculate max ldsNumBytes for other epilogues
    if state["ProblemType"]["UseScaleAlphaVec"]:
      epilogueSize += state["NumThreads"] * state["ProblemType"]["ComputeDataType"].numBytes() * vecDT.scaleAlpha(0).turn
    if state["ProblemType"]["UseScaleAB"] == "Vector":
      epilogueSize += state["NumThreads"] * state["ProblemType"]["ComputeDataType"].numBytes() * (vecDT.scaleA.turn + vecDT.scaleB.turn)
    ldsNumBytes = max(ldsNumBytes, state["LdsOffsetBias"] + epilogueSize)

    state["LdsBytesNoAmax"] = ldsNumBytes
    if state["ProblemType"]["OutputAmaxD"]:
      # used in reduce inter wave
      # 4 data * half_wave_num * amax bytePerE
      num_workItems = state["NumThreads"]
      half_wave_size = state["WavefrontSize"] // 2
      amaxBPE = state["ProblemType"]["DataTypeAmaxD"].numBytes()
      ldsAmaxDBytes = 4 * (num_workItems // half_wave_size) * amaxBPE
      ldsNumBytes += ldsAmaxDBytes

    state["LdsNumBytes"] = ldsNumBytes
    ldsSize = ldsNumBytes
    if ldsSize > globalParameters["MaxLDS"]:
      reject(state, "Kernel Uses %u > %u bytes of LDS" % ( ldsSize, globalParameters["MaxLDS"]))
      state["ValidDepthU"] = False
      return

    # LoopUnroll  = DepthU / LocalSplitU
    if "LocalSplitU" in state:
      state["LoopUnroll"] = state["DepthU"] // state["LocalSplitU"]
    if state["LoopUnroll"] * state["LocalSplitU"] != state["DepthU"]:
      state["Valid"] = False
    if state["KernelLanguage"] != "Assembly" and state["InnerUnroll"] != 1:
      reject(state, "InnerUnroll only supported on assembly")
    state["LoopUnroll"] //= state["InnerUnroll"]

    if 0:
      print("info: ", pvar(state, "LoopUnroll"), " LDS Stats:", pvar(state, "LdsOffsetA"), pvar(state, "LdsOffsetB"))
      print("info: ", pvar(state["ProblemType"], "TLUA"), \
          pvar(state, "NumLoadsCoalescedA"), pvar(state, "NumLoadsPerpendicularA"), \
          pvar(state, "LSCA"), pvar(state, "LSPA"))
      print("info:", pvar(state["ProblemType"], "TLUB"), \
          pvar(state, "NumLoadsCoalescedB"), pvar(state, "NumLoadsPerpendicularB"), \
          pvar(state, "LSCB"), pvar(state, "LSPB"))

    state["LoopIters"] = state["LoopUnroll"]
    if state["EnableMatrixInstruction"]:
      state["LoopIters"] //= state["MatrixInstK"]
    elif state["UseDotInstruction"]:
      # dot2
      state["LoopIters"] //= (state["NumDotElements"] * state["NumWaveSplitK"])

    if state["LoopIters"] < 1:
      reject(state, "LoopIters need to greater than 0")
      return

    # Since we use PLR >= LoopIters for allocating numberOfIters vgprBuffer for a while
    # we need to support both PLR >= LoopIters and CLR parameter for solutions in rocBLAS
    if state["ClusterLocalRead"] and state["PrefetchLocalRead"] >= state["LoopIters"] and not state["ScheduleIterAlg"] == 2:
      # 1 or 2 Byte input + DTVA or DTVB case, does not work with PLR=0. Reject it here.
      if state["ProblemType"]["DataType"].numBytes() < 4 and \
         (state["ProblemType"]["TLUA"] and state["DirectToVgprA"] or state["ProblemType"]["TLUB"] and state["DirectToVgprB"]):
        reject(state, "DirectToVgpr does not work with 1 or 2 Byte input + TLU + PrefetchLocalRead(%u) >= LoopIters(%u)"%(state["PrefetchLocalRead"], state["LoopIters"]))
        return
      state["ClusterLocalRead"] = 0
      state["PrefetchLocalRead"] = 0
    if not state["EnableMatrixInstruction"]:
      state["ClusterLocalRead"] = 0
      # dot2: allow PLR=1
      if state["UseDotInstruction"]:
        state["PrefetchLocalRead"] = min(state["PrefetchLocalRead"], state["LoopIters"]-1)
        if state["LoopIters"] % (state["PrefetchLocalRead"]+1) != 0:
          reject(state, "dot2 kernel does not support LoopIters(%u) %% (PLR+1)(%u) != 0" % (state["LoopIters"], state["PrefetchLocalRead"]+1))

    # reject iterations are not enough to use wider local read
    if state["EnableMatrixInstruction"] and state["PrefetchLocalRead"] > 0:
      # Multiple = WLR-size / input-size = how many iters could be covered by one WLR ?
      wlrMultiple = state["LocalReadVectorWidth"]//state["MIInputPerThread"]
      # NOTE: wlrmultiple can be 0 for new MFMA
      if not state["ProblemType"]["Sparse"] and (not globalParameters["AsmCaps"][isa]["HasMFMA_f8f6f4"] or state["MatrixInstK"] <= 32):
        if wlrMultiple == 0:
          reject(state, "LocalReadVectorWidth %u is less than MIInput" % (state["LocalReadVectorWidth"]))
          return
      # for example, if the original ds_read is b32...
      #   1. if LoopIters = 5 (b32 x 5 times), WLR-Multiple = 2 (b64), then we can fit the WLR
      #   2. if LoopIters = 2 (b32 x 2 times), WLR-Multiple = 4 (b128), this is not allowed
      #   3. if LoopIters = 2 (b32 x 2 times), WLR-Multiple = 2 (b64), this is allowed
      if wlrMultiple and state["LoopIters"] % wlrMultiple != 0:
        reject(state, "LocalReadVectorWidth %u cannot be distributed evenly, LoopIters %u should be divisible by WLR-Multiple %u" \
          % (state["LocalReadVectorWidth"], state["LoopIters"], wlrMultiple))

      if state["LoopIters"] - (state["PrefetchLocalRead"] * wlrMultiple) < 0 :
        reject(state, "with PrefetchLocalRead %u LoopIters %u LocalReadVectorWidth %u, not enough LoopIters to prefetch %ux%u iterations, " \
          % (state["PrefetchLocalRead"],state["LoopIters"],state["LocalReadVectorWidth"], state["PrefetchLocalRead"] , wlrMultiple) )

    # # reject conditions with lower performance
    # if state["ScheduleIterAlg"] == 2 and \
    # (state["ExpandPointerSwap"] != 1 or state["LoopIters"] != 1 or state["ScheduleGlobalRead"] != 1):
    #   reject(state, "ScheduleIterAlg 2 only work with EPS1_SGR1, LoopIter=1")

    if state["TransposeLDS"] == 1:
      if not state["EnableMatrixInstruction"]:
        reject(state, "TransposeLds Supports only in MatrixInstruction=1")
      if state["ProblemType"]["TLUA"] and state["ProblemType"]["TLUB"]:
          # TODO: Now in rocBLAS, lot of logic yamls are Type=NT and TLDS=1? Why aren't they rejected and how to get rid of them?
          reject(state, "TransposeLds requires TLUA=0 or TLUB=0")
    if state["EnableMatrixInstruction"]:
      # enable widerLocalRead
      if state["LocalReadVectorWidth"] > state["MIInputPerThread"]:
        # wider localRead support 2 types
        # 1. prefetch all lds to register
        # 2. using larger InnerUnroll
        if not (state["PrefetchLocalRead"] >= state["LoopIters"] and state["InnerUnroll"] == 1) and \
            not state["ClusterLocalRead"] and \
            not state["InnerUnroll"] >= state["LocalReadVectorWidth"] // state["MIInputPerThread"]:
          reject(state, "wider localRead only support ClusterLocalRead or (InnerUnroll > WiderLocalReadxN)")

    if state["GlobalReadPerMfma"] > 1 and state["PrefetchGlobalRead"] == 2:
      reject(state, "GlobalReadPerMfma need to be 1 if PGR2")

    if state["UseInstOffsetForGRO"] == -1:
      state["UseInstOffsetForGRO"] = 1 if state["DirectToLds"] else 0

    state["ULSGRODoubleG2L"] = 0
    if state["UnrollLoopSwapGlobalReadOrder"] == 1:
      bpeAB       = state["ProblemType"]["DataType"].numBytes()
      bpr         = 4
      numVgprG2LA = roundUp((state["NumLoadsCoalescedA"] * state["NumLoadsPerpendicularA"] * \
        state["GlobalReadVectorWidthA"] * bpeAB) / (float)(bpr))
      numVgprG2LB = roundUp((state["NumLoadsCoalescedB"] * state["NumLoadsPerpendicularB"] * \
        state["GlobalReadVectorWidthB"] * bpeAB) / (float)(bpr))
      if numVgprG2LA % 2 == 1 or numVgprG2LB % 2 == 1:
        reject(state, "G2LA/B vgpr has bubble inside. Cannot use UnrollLoopSwapGlobalReadOrder=1.")
      if state["GlobalReadVectorWidthA"] != state["GlobalReadVectorWidthB"]:
        # TODO: Add a configuration to schedule better.
        state["ULSGRODoubleG2L"] = 1
      minGRVW = min(state["GlobalReadVectorWidthA"], state["GlobalReadVectorWidthB"])
      if minGRVW * bpeAB < 4:
        # G2LA/B vgpr index will jump.
        state["ULSGRODoubleG2L"] = 1
      if state["ExpandPointerSwap"] == 1:
        reject(state, "ExpandPointerSwap need to be 0 if UnrollLoopSwapGlobalReadOrder")
      if state["PrefetchGlobalRead"] != 2:
        reject(state, "PrefetchGlobalRead need to be 2 if UnrollLoopSwapGlobalReadOrder")
      if state["ProblemType"]["DataTypeA"].numBytes() != state["ProblemType"]["DataTypeB"].numBytes():
        reject(state, "UnrollLoopSwapGlobalReadOrder doesn't support mixed precision.")

    # guard against out of bounds reads
    # None: don't guard against ou
    # ShiftPtr: shift read pointers to be in bounds, then unshift registers (source & assembly),
    # ShiftPtr does not support very small problem dims < global load vector width since the shift
    # would move outside the array bounds.
    # If GLVW==1 or Assert*ElementMultiple for the coalesced dim is > GRVW, then shifting is not
    # necessary and the shift/unshift code will not be generated
    state["EdgeType"] = "ShiftPtr" # Use ShiftPtr by default

    # Precise bounds check uses the "num_records" field in the buffer to
    # precisely detect when we are inbounds or not.  Only a one-dimensional
    # check is used since this is faster and also for computation we only
    # need to ensure that none of the loads fault.  threads which are
    # computing bogus sections of the C tile will later be ignored.
    # precise checking only works when all elements of the load are in-bounds
    # since if the vload crosses boundary we ignore all components not just the
    # ones that are OOB. See comments for groOffsetInMacroTile in KernelWriterAssembly.py
    #
    # So check for the cases where the unroll loop can
    # generate partial loads here and reject PBC solutions:
    # For non-TLU the free dim is in perp dim - should always be TRUE?  TODO

    #--
    # ShiftPtr can't use UseSgprForGRO since it needs to modify the VGPR pointers
    if bufferLoad and state["_UseSgprForGRO"] and state["EdgeType"]=="ShiftPtr":
      if not state["GuaranteeNoPartialA"] or not state["GuaranteeNoPartialB"] or not state["GuaranteeNoPartialMetadata"]:
        state["_UseSgprForGRO"] = False
        #reject(state, "PBC with wide load has insufficient overlap guarantees- try GRVW=1 or adding appropriate Assert*ElementMultiple")




    if state["EnableMatrixInstruction"]:
      cont1 = not state["GuaranteeNoPartialB"]
      cont2 = ((state["MatrixInstN"] % state["GlobalReadVectorWidthB"]) != 0)
      if cont1 and cont2:
        reject(state, "MatrixInstN %u %% GlobalReadVectorWidthB %u must be 0" % \
          (state["MatrixInstN"], state["GlobalReadVectorWidthB"]))

    # Use SGPR to store an offset from GlobalReadOffsetA+0.
    # (as opposed to using dedicated VGPR for each GRO
    # Requires preciseBounds check since we rely on the buffer bounds check, not
    # individual vector registers doing bounds compares.

    if state["_UseSgprForGRO"] == 1 and (state["ProblemType"]["SwizzleTensorA"] or state["ProblemType"]["SwizzleTensorB"]):
      reject(state, "UseSgprForGRO for Swizzle is not supported")

    if state["_UseSgprForGRO"] == -1:
      # Don't use SGPR if it looks like we might not have enough - better to leave PBC enabled even if we have to use VGPR
      # 40 is based on current SGPR usage, this may need to be tuned in the future:
      numLoadsA = state["NumLoadsCoalescedA"]*state["NumLoadsPerpendicularA"]
      numLoadsB = state["NumLoadsCoalescedB"]*state["NumLoadsPerpendicularB"]
      numLoadsM = 0
      if state["ProblemType"]["Sparse"] and not state["DirectToVgprSparseMetadata"]:
        numLoadsM = state["NumLoadsCoalescedMetadata"]*state["NumLoadsPerpendicularMetadata"]
      if numLoadsA + numLoadsB + numLoadsM > 35 or state["ProblemType"]["SwizzleTensorA"] or state["ProblemType"]["SwizzleTensorB"]:
        #print "info: Disabling UseSgprForGRO since predicting too many SGPR will be used"
        state["_UseSgprForGRO"] = 0
      else:
        state["_UseSgprForGRO"] = 1

    if packedC0 and not state["GuaranteeNoPartialA"]:
      reject(state, "packedC0 requires GuaranteeNoPartialA")
    if packedC1 and not state["GuaranteeNoPartialB"]:
      reject(state, "packedC1 requires GuaranteeNoPartialB")

    if packedC0 or packedC1:
      state["_UseSgprForGRO"] = 0

      if state["EdgeType"] != "ShiftPtr":
        reject(state, "Packed dims requires EdgeType==ShiftPtr")
      if state["KernelLanguage"] == "Assembly":
        if not bufferLoad:
          reject(state, "Packed dims for Assembly requires BufferLoad")

    if packedC0: # VectorWidth must not span tensor dim
      if state["KernelLanguage"] == "Source":
        if state["AssertFree0ElementMultiple"]<state["VectorWidthA"]:
          reject(state, "packedC0 Source requires AF0EM>=VectorWidth (for loads and stores)")
      else:
        if state["AssertFree0ElementMultiple"]<state["VectorWidthA"]\
          or state["AssertFree0ElementMultiple"] == 1:
            if state["VectorStore"] <= 0:
              state["_VectorStore"] = 0
            else:
              reject(state, "packedC0 Assembly requires AF0EM>=VectorWidth or not VectorStore (for stores)")

    state["AssignedDerivedParameters"] = True

    # UnrollLoopEfficiencyEnable does not work with f16/bf16/int8x4
    if globalParameters["UnrollLoopEfficiencyEnable"] and (state["ProblemType"]["DataType"].isHalf() or \
       state["ProblemType"]["DataType"].isBFloat16() or state["ProblemType"]["DataType"].isInt8x4()):
      reject(state, "UnrollLoopEfficiencyEnable does not support f16/bf16/int8x4")

    # UnrollLoopEfficiencyEnable supports only ThreadTile0,1=[6,4] or [4,6] or [4,4] or [6.6] or [8,4] or [4,8]
    if globalParameters["UnrollLoopEfficiencyEnable"] and \
      not ((state["ThreadTile0"] == 6 and state["ThreadTile1"] == 4) or \
           (state["ThreadTile0"] == 4 and state["ThreadTile1"] == 6) or \
           (state["ThreadTile0"] == 4 and state["ThreadTile1"] == 4) or \
           (state["ThreadTile0"] == 6 and state["ThreadTile1"] == 6) or \
           (state["ThreadTile0"] == 8 and state["ThreadTile1"] == 4) or \
           (state["ThreadTile0"] == 4 and state["ThreadTile1"] == 8)):
      reject(state, "UnrollLoopEfficiencyEnable does not support ThreadTile0,1 = [%u,%u]"%(state["ThreadTile0"], state["ThreadTile1"]))

    # Set E
    if state["ProblemType"]["UseE"]:
      if (state["_GlobalAccumulation"] == 'SingleBuffer') and state["GlobalSplitU"] > 1:
        reject(state, "GlobalSplitU > 1 only compatible with MultipleBuffer")
      if len(state["PackedC1IndicesX"]) > 1:
        reject(state, "Use E does not support len(PackedC1IndicesX) > 1.")
      if not state["BufferStore"]:
        reject(state, "Use E only supports BufferStore due to no suppress no store.")
      if state["StoreRemapVectorWidth"] and (state["GlobalSplitU"] == 1):
        reject(state, "Use E does not support StoreRemapVectorWidth if GSU == 1.")
      if state["GroupLoadStore"]:
        reject(state, "Use E does not support GroupLoadStore.")

    # Activation
    # Function call is set to false if GSU != 1 or Activation is not fused or ActivationType is not All.
    if not (state["ActivationFused"] and state["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all']) \
      and state["ActivationFuncCall"]:
      state["ActivationFuncCall"] = False

    if state["ActivationAlt"]:
      reject(state, "Currently does not accept ActivationAlt.")

    # Bias reduction
    if state["ProblemType"]["UseBias"] and state["ProblemType"]["Gradient"]:
      if (state["_GlobalAccumulation"] == 'SingleBuffer') and state["GlobalSplitU"] > 1:
        reject(state, "GlobalSplitU > 1 only compatible with MultipleBuffer for bias reduction")
      if len(state["PackedC1IndicesX"]) > 1:
        reject(state, "Bias reduction does not support len(PackedC1IndicesX) > 1.")
      if not state["BufferStore"]:
        reject(state, "Bias reduction only supports BufferStore due to no suppress no store.")
      if state["StoreRemapVectorWidth"] and (state["GlobalSplitU"] == 1):
        reject(state, "Bias reduction does not support StoreRemapVectorWidth if GSU == 1.")
      if state["GroupLoadStore"]:
        reject(state, "Bias reduction does not support GroupLoadStore.")

    # Bias and ScaleAlphaVec
    if state["ProblemType"]["UseBias"] != 0 and state["ProblemType"]["UseScaleAlphaVec"] != 0 and state["ProblemType"]["UseBias"] != state["ProblemType"]["UseScaleAlphaVec"]:
      reject(state, "When both UseBias and UseScaleAlphaVec are enabled then UseBias and UseScaleAlphaVec must have same settings.")

    # ScaleAB or ScaleABVec
    if state["ProblemType"]["DataTypeA"] != state["ProblemType"]["DataTypeB"] and \
      state["ProblemType"]["DataTypeA"] != state["ProblemType"]["DataType"] and \
      state["ProblemType"]["UseScaleAB"] == "Vector":
      reject(state, "Currently does not support using scaleABVec if DataTypeA != DataTypeB != DataType.")

    if state["ProblemType"]["UseScaleAB"] and state["OptNoLoadLoop"]:
      # Hard to check alpha == 1.0 directly
      # Turn off ONLL for now
      # TODO: support ONLL if necessary
      state["OptNoLoadLoop"] = 0

    # if state["GlobalSplitU"] > 1:
    #   if state["ProblemType"]["SupportUserArgs"] and state["_GlobalAccumulation"] != 'MultipleBufferSingleKernel':
    #     reject(state, "Currently SupportUserArgs does not support GSU > 1.")

    if state["_GlobalAccumulation"] == 'MultipleBufferSingleKernel':
      if (state["NumElementsPerBatchStore"] == 1):
        reject(state, "too many store at MultipleBufferSingleKernel direct reject")
      if state["ProblemType"]["UseScaleCD"]:
        reject(state, "MultipleBufferSingleKernel not support UseScaleCD yet")
      if state["ProblemType"]["UseE"]:
        reject(state, "MultipleBufferSingleKernel not support UseE yet")
      if state["ProblemType"]["BiasSrc"] != "D":
        reject(state, "MultipleBufferSingleKernel not support BiasSrc not D yet")
      if state["ProblemType"]["DataType"].isDouble():
        reject(state, "MultipleBufferSingleKernel not support " + str(state["ProblemType"]["DataType"])  + " yet")
      if state["ProblemType"]["Sparse"] != 0:
        reject(state, "MultipleBufferSingleKernel not support sparse yet")

    #Need to force disabling PreloadKernArgs if compiler does not support
    #Can not just reject the solution since the user library may find any solutions
    if state["PreloadKernArgs"]:
      hipccver = globalParameters['HipClangVersion'].split(".")
      hipccMaj = int(hipccver[0])
      hipccPatch = int(hipccver[2].split("-")[0])
      if not (hipccMaj >= 6 and hipccPatch >= 32650 and (isa == (9, 0, 10) or isa[:2] == (9, 4))):
        #print("Force to Disable PreloadKernArgs since this hipcc version doesn't support",)
        state["PreloadKernArgs"] = 0

  ########################################
  # create a dictionary with booleans on whether to include parameter in name
  @staticmethod
  def getMinNaming(objs):
    nonCKObjs = [obj for obj in objs if not isCustomKernelConfig(obj)]

    # early return
    if len(nonCKObjs) == 0:
      return {}

    # determine keys
    requiredParameters = {}
    if isinstance(nonCKObjs[0], Solution):
      keys = list(nonCKObjs[0]._state.keys())
    else:
      keys = list(nonCKObjs[0].keys())
    # only 1, rather than name being nothing, it'll be everything
    if len(nonCKObjs) == 1:
      for key in keys:
        if key in list(validParameters.keys()):
          requiredParameters[key] = False
    else:
      for key in keys:
        required = False
        if key in list(validParameters.keys()):
          for i in range(1, len(nonCKObjs)):
            if nonCKObjs[0][key] != nonCKObjs[i][key]:
              required = True
              break
        if required:
          requiredParameters[key] = True
        else:
          requiredParameters[key] = False

    requiredParameters["GlobalSplitU"] = True
    requiredParameters["WorkGroupMapping"] = True

    if "MatrixInstM" in nonCKObjs[0]._state:
      # Use MIWaveGroup and MIWaveTile instead of WG and MT
      requiredParameters["MIWaveTile"]  = True
      requiredParameters["ThreadTile"]  = False

    requiredParameters["ProblemType"]       = False # always prepended
    requiredParameters["MacroTile0"]        = False # always prepended
    requiredParameters["MacroTile1"]        = False # always prepended
    requiredParameters["DepthU"]            = False # always prepended
    requiredParameters["MatrixInstruction"] = False # always prepended
    requiredParameters["MatrixInstM"]       = False # always prepended
    requiredParameters["MatrixInstN"]       = False # always prepended
    requiredParameters["MatrixInstK"]       = False # always prepended
    requiredParameters["MatrixInstB"]       = False # always prepended
    requiredParameters["MatrixInstBM"]      = False # always prepended
    requiredParameters["MatrixInstBN"]      = False # always prepended
    requiredParameters["CustomKernelName"]  = False # Will not affect naming

    requiredParameters["Kernel"]            = True  # distinguish kernels from solutions
                                                    # for single-source compilation
    return requiredParameters

  ########################################
  @ staticmethod
  def getKeyNoInternalArgs(state):
    state_copy = deepcopy(state)

    state_copy["ProblemType"]["GroupedGemm"] = False

    if globalParameters["SplitGSU"]:
      state_copy["GlobalSplitU"] = "M" if (state_copy["GlobalSplitU"] > 1) else state_copy["GlobalSplitU"]
    elif state["GlobalSplitU"] > 0:
      state_copy["GlobalSplitU"] = "M"
    state_copy["WorkGroupMapping"] = "M"
    state_copy["WorkGroupMappingXCC"] = "M"
    state_copy["WorkGroupMappingXCCGroup"] = "M"
    state_copy["StaggerU"] = "M"
    state_copy["StaggerUStride"] = "M"
    state_copy["StaggerUMapping"] = "M"
    state_copy["GlobalSplitUCoalesced"] = "M"
    state_copy["GlobalSplitUWorkGroupMappingRoundRobin"] = "M"

    return state_copy

  @ staticmethod
  def getNameFull(state):
    requiredParameters = {}
    for key in state:
      if key in list(validParameters.keys()):
        requiredParameters[key] = True
    if "MatrixInstM" in state:
      # Use MIWaveGroup and MIWaveTile instead of WG and MT
      requiredParameters["MIWaveTile"]  = True
      requiredParameters["ThreadTile"]  = False
    return Solution.getNameMin(state, requiredParameters)

  ########################################
  # Get Name Min
  @ staticmethod
  def getNameMin(state, requiredParameters, ignoreInternalArgs = False):
    if isCustomKernelConfig(state):
      return state["CustomKernelName"]

    components = []

    backup = state["ProblemType"]["GroupedGemm"]
    if ignoreInternalArgs:
      state["ProblemType"]["GroupedGemm"] = False

    if "ProblemType" in state:
      components.append(f'{str(state["ProblemType"])}')
      # name += str(state["ProblemType"]) + "_"

    if ignoreInternalArgs:
      state["ProblemType"]["GroupedGemm"] = backup

    if "MacroTile0" in state \
        and "MacroTile1" in state \
        and "DepthU" in state:
      components.append(f'{Solution.getParameterNameAbbreviation("MacroTile")}{state["MacroTile0"]}x{state["MacroTile1"]}x{state["DepthU"]}')

    if "MatrixInstM" in state:
      components.append(f'{Solution.getParameterNameAbbreviation("MatrixInstruction")}{state["MatrixInstM"]}x{state["MatrixInstN"]}x{state["MatrixInstB"]}')

    backup = state["GlobalSplitU"]

    if ignoreInternalArgs:
      if globalParameters["SplitGSU"]:
        state["GlobalSplitU"] = "M" if (state["GlobalSplitU"] > 1) else state["GlobalSplitU"]
      elif state["GlobalSplitU"] > 0:
        requiredParameters["GlobalSplitU"] = False
      requiredParameters["WorkGroupMapping"] = False
      requiredParameters["WorkGroupMappingXCC"] = False
      requiredParameters["WorkGroupMappingXCCGroup"] = False
      requiredParameters["StaggerU"] = False
      requiredParameters["StaggerUStride"] = False
      requiredParameters["StaggerUMapping"] = False
      requiredParameters["GlobalSplitUCoalesced"] = False
      requiredParameters["GlobalSplitUWorkGroupMappingRoundRobin"] = False

    useWaveTile, useThreadTile = requiredParameters.get("MIWaveTile", False), requiredParameters.get("ThreadTile", False)

    if 'MatrixInstM' in state:
      requiredParameters["MIWaveTile"] = True
      requiredParameters["ThreadTile"] = False
    else:
      requiredParameters["MIWaveTile"] = False
      requiredParameters["ThreadTile"] = True

    components.append('SN')
    for key in sorted(state.keys()):
      if key in requiredParameters and key[0] != '_':
        if requiredParameters[key] and key != "CustomKernelName":
          components.append(f'{Solution.getParameterNameAbbreviation(key)}{Solution.getParameterValueAbbreviation(key, state[key])}')

    state["GlobalSplitU"] = backup
    requiredParameters["GlobalSplitU"] = True
    requiredParameters["WorkGroupMapping"] = True
    requiredParameters["WorkGroupMappingXCC"] = True
    requiredParameters["WorkGroupMappingXCCGroup"] = True
    requiredParameters["StaggerU"] = True
    requiredParameters["StaggerUStride"] = True
    requiredParameters["StaggerUMapping"] = True
    requiredParameters["GlobalSplitUCoalesced"] = True
    requiredParameters["GlobalSplitUWorkGroupMappingRoundRobin"] = True
    requiredParameters["MIWaveTile"] = useWaveTile
    requiredParameters["ThreadTile"] = useThreadTile

    return '_'.join(components)

  ########################################
  # create a dictionary of lists of parameter values
  @staticmethod
  def getSerialNaming(objs):
    data = {}
    for obj in objs:
      for paramName in sorted(obj.keys()):
        if paramName in validParameters.keys():
          paramValue = obj[paramName]
          if paramName in data:
            if paramValue not in data[paramName]:
              data[paramName].append(paramValue)
          else:
            data[paramName] = [ paramValue ]
    maxObjs = 1
    for paramName in data:
      if not isinstance(data[paramName][0], dict):
        data[paramName] = sorted(data[paramName])
      maxObjs *= len(data[paramName])
    numDigits = len(str(maxObjs))
    return [ data, numDigits ]

  ########################################
  # Get Name Serial
  @ staticmethod
  def getNameSerial(state, serialNaming):
    data = serialNaming[0]
    numDigits = serialNaming[1]

    serial = 0
    multiplier = 1
    for paramName in sorted(state.keys()):
      if paramName in list(validParameters.keys()):
        paramValue = state[paramName]
        paramData = data[paramName]
        paramNameMultiplier = len(paramData)
        if paramValue in paramData:
          paramValueIdx = paramData.index(paramValue)
        serial += paramValueIdx * multiplier
        multiplier *= paramNameMultiplier
    name = "%s%0*u" % ("S" if isinstance(state, Solution) else "K", \
        numDigits, serial)
    return name


  ########################################
  @ staticmethod
  def getParametersIndented(state, indent):
    s = ""
    s += "%sProblemType: %s\n" % (indent, str(state["ProblemType"]))
    for key in sorted(state):
      s += "%s%s: %s\n" % (indent, str(key), str(state[key]))
    return s

  ########################################
  @ staticmethod
  @ lru_cache(maxsize=None)
  def getParameterNameAbbreviation( name: str ):
    return ''.join(c for c in name if c.isupper())

  ########################################

  class NonprimitiveParameterValueException(Exception):
    pass

  @ staticmethod
  @ lru_cache(maxsize=None)
  def getPrimitiveParameterValueAbbreviation(key, value):
    if isinstance(value, str):
      return Solution.getParameterNameAbbreviation(value)
    elif isinstance(value, bool):
      return "1" if value else "0"
    elif isinstance(value, int):
      if value >= 0:
        return "%u" % value
      else: # -1 -> n1
        return "n%01u" % abs(value)
    elif isinstance(value, ProblemType):
      return str(value)
    elif isinstance(value, float):
      val1 = int(value)
      val2 = int(round(value*100)) - int(value)*100
      if val2 > 0:
        s =  "%dp%s" % (val1,str(val2).zfill(2))
      else:
        s = "%d" % (val1)
      return s

  ########################################

  @ staticmethod
  def getParameterValueAbbreviation( key, value ):
    if key == "ISA":
      return f"{value[0]}{value[1]}{value[2]:x}"

    compositieTypes = (dict, list, tuple,)

    if not isinstance(value, compositieTypes):
      return Solution.getPrimitiveParameterValueAbbreviation(key, value)
    elif isinstance(value, tuple):
      return ''.join(str(v) for v in value)
    elif isinstance(value, list):
      return '_'.join(Solution.getParameterValueAbbreviation(key, v) for v in value)
    elif isinstance(value, dict):
      return "_".join(f"{pos:d}{k:d}" for pos,k in value.items())
    else:
      printExit('Parameter {key}={value} is new object type ({t})'.format(key=key, value=value, t=type(value)))
      return str(value)

  ##########################
  # make class look like dict
  def keys(self):
    return list(self._state.keys())

  def __len__(self):
    return len(self._state)

  def __iter__(self):
    return iter(self._state)

  def __getitem__(self, key):
    return self._state[key]

  def __setitem__(self, key, value):
    self._name = None
    self._state[key] = value

  def __str__(self):
    if self._name is None:
      self._name = Solution.getNameFull(self._state)
    return self._name

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return self._state

  def __hash__(self):
    return hash(str(self) + self._state.get("codeObjectFile", ""))
    #return hash(self.getAttributes())

  def __eq__(self, other):
    #return isinstance(other, Solution) and self.getAttributes() == other.getAttributes()
    return isinstance(other, Solution) and str(self) == str(other)

  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result
