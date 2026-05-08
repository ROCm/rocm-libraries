################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
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

import collections
import math
import sys

from enum import Enum
from typing import List, Dict, Literal

from Tensile.AsmStoreState import VectorDataTypes
from Tensile.Activation import ActivationType
from Tensile.Activation import ActivationType
from Tensile.AsmStoreState import VectorDataTypes
from Tensile.Common import assignParameterWithDefault, IsaInfo, \
                    print2, printExit, printWarning, \
                    roundUp, INDEX_CHARS, IsaVersion, SemanticVersion, \
                    roundUpToNearestMultiple
from Tensile.Common.DataType import DataType
from Tensile.Common.GlobalParameters import defaultSolution, \
                                            defaultInternalSupportParams
from Tensile.Common.ValidParameters import validParameters
from Tensile.SolutionStructs.Naming import getSolutionNameFull
from Tensile.SolutionStructs.Problem import ProblemType
from Tensile.Toolchain.Component import Assembler
from Tensile.Components.CustomSchedule import hasCustomSchedule

from ..Component import TensorDataMover
from ..Components.TensorDataMover import TensorDataMoverLoad
from .Utilities import reject, roundupRatio, pvar

def _getExpectedTypes(validParams):
  """Build a map from parameter name to the set of allowed Python types.

  Uses the validParameters registry as the source of truth.  For each
  parameter whose allowed-value list is not the sentinel ``-1``, we
  collect the concrete ``type()`` of every allowed value.  Because
  Python ``bool`` is a subclass of ``int``, we use ``type()`` (not
  ``isinstance``) so that ``bool`` and ``int`` are kept distinct.

  Returns:
      dict[str, set[type]]: e.g. {"UseCustomMainLoopSchedule": {int},
                                   "BufferLoad": {bool}, ...}
  """
  typeMap = {}
  for name, allowedValues in validParams.items():
    if allowedValues == -1:
      continue
    if isinstance(allowedValues, list) and len(allowedValues) > 0:
      typeMap[name] = set(type(v) for v in allowedValues)
  return typeMap

# Pre-compute once at import time so the per-Solution cost is a dict lookup.
_expectedParamTypes = _getExpectedTypes(validParameters)

# Parameters to skip during type validation because YAML serialization
# inherently produces a different type (e.g. [9, 0, 10] -> list) and the
# conversion to the canonical type happens downstream in the pipeline.
_skipTypeCheck = {"ISA"}


# Module-level collector that accumulates type mismatches across all Solution
# instances during a build.  Key is (param_name, actual_type_name,
# expected_type_str); value is a dict with 'count', 'values' (set of unique
# repr(value)), and 'files' (set of source file paths).
_typeMismatchCollector = {}


def resetTypeMismatchCollector():
  """Clear the module-level type mismatch collector.

  Call this before a new build pass or in test setUp to ensure a clean
  slate.  Safe to call even when the collector is already empty.

  Uses dict.clear() so that any existing references to
  ``_typeMismatchCollector`` (e.g. in tests) see the same empty dict.
  """
  _typeMismatchCollector.clear()


def getTypeMismatchCollector():
  """Return a copy of the current collector state.

  Used by worker processes to return their collected mismatch data to the
  main process after parallel YAML parsing.
  """
  return {k: {"count": v["count"], "values": set(v["values"]), "files": set(v["files"])}
          for k, v in _typeMismatchCollector.items()}


def mergeTypeMismatchCollector(data):
  """Merge mismatch data returned from a worker process into the main collector.

  Args:
      data: A dict in the same format as ``_typeMismatchCollector``, as
            returned by ``getTypeMismatchCollector()`` in a worker process.
  """
  for key, entry in data.items():
    if key not in _typeMismatchCollector:
      _typeMismatchCollector[key] = {"count": 0, "values": set(), "files": set()}
    _typeMismatchCollector[key]["count"] += entry["count"]
    _typeMismatchCollector[key]["values"] |= entry["values"]
    _typeMismatchCollector[key]["files"] |= entry["files"]


def validateParameterTypes(state, srcFile=""):
  """Validate that every solution parameter has the correct Python type.

  Compares each value in *state* against the types implied by
  ``validParameters``.  A ``bool`` where ``int`` is expected (or vice
  versa) is the most common error -- YAML ``false``/``true`` vs ``0``/``1``
  are different Python types and produce different msgpack wire types,
  which causes ``std::bad_cast`` at C++ deserialization time.

  Instead of raising on the first mismatch, mismatches are collected into
  the module-level ``_typeMismatchCollector`` dict.  Call
  ``printTypeMismatchSummary()`` at the end of the build to emit a
  consolidated warning.

  Args:
      state: The solution state dict (parameter name -> value).
      srcFile: The YAML source file path, included in warning messages.
  """
  for key, value in state.items():
    if key not in _expectedParamTypes or key in _skipTypeCheck:
      continue
    expectedTypes = _expectedParamTypes[key]
    actualType = type(value)
    # Use type() not isinstance() so that bool and int are distinguished
    if actualType not in expectedTypes:
      expectedStr = " or ".join(sorted(t.__name__ for t in expectedTypes))
      collectorKey = (key, actualType.__name__, expectedStr)
      if collectorKey not in _typeMismatchCollector:
        _typeMismatchCollector[collectorKey] = {
          "count": 0,
          "values": set(),
          "files": set(),
        }
      entry = _typeMismatchCollector[collectorKey]
      entry["count"] += 1
      entry["values"].add(repr(value))
      if srcFile:
        entry["files"].add(srcFile)


def printTypeMismatchSummary(numFiles=0):
  """Print a summary of all collected type mismatches.

  If no mismatches have been collected, prints a confirmation message
  showing how many files were checked cleanly, and returns 0.  Otherwise
  it emits a WARNING block with one line per unique (parameter,
  actual_type) combination showing the count, observed values, and
  expected type.

  Args:
      numFiles: Total number of YAML logic files that were checked.

  Returns:
      int: The total number of individual mismatches (0 if clean).
  """
  if not _typeMismatchCollector:
    if numFiles > 0:
      print(f"Checked {numFiles} YAML logic files - no type mismatches found.", flush=True)
    return 0

  totalCount = sum(e["count"] for e in _typeMismatchCollector.values())
  allFiles = set()
  for e in _typeMismatchCollector.values():
    allFiles |= e["files"]

  lines = []
  lines.append("")
  lines.append("===========================================================")
  lines.append(
    f"WARNING: YAML parameter type mismatches detected "
    f"({totalCount} total across {len(allFiles)} files):"
  )
  lines.append("===========================================================")

  # Sort by parameter name for stable output
  for (paramName, actualType, expectedStr), entry in sorted(
    _typeMismatchCollector.items(), key=lambda kv: kv[0][0]
  ):
    valuesStr = ", ".join(sorted(entry["values"]))
    lines.append(
      f"  {paramName}: found {actualType} in {entry['count']} solutions "
      f"(values: {valuesStr}) - expected {expectedStr}"
    )

  lines.append("-----------------------------------------------------------")
  lines.append("  This will cause std::bad_cast at runtime because msgpack")
  lines.append("  serializes bool and int as different wire types.")
  lines.append("  Fix these to prevent future build failures.")
  lines.append("===========================================================")

  print("\n".join(lines), flush=True)
  return totalCount


class Fbs(Enum):
  Free=0     # Expect to be free dimension
  Batch=1    # Expect to be batch dimension
  Sum=2      # Expect to be summation dimension

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
  MAX_NUM_DS_LOAD_VGPRS: int = 4
  MAX_NUM_DS_LOAD_BYTES: int = 4 * MAX_NUM_DS_LOAD_VGPRS

  ########################################   # need to be sure PSRR is passing to all fxns
  def __init__(
    self,
    config,
    splitGSU: bool,
    printSolutionRejectionReason: bool,
    printIndexAssignmentInfo: bool,
    assembler: Assembler,
    isaInfoMap: Dict[IsaVersion, IsaInfo],
    srcName: str = ""
  ):

    self._name = None
    self.assembler = assembler
    self.isaInfoMap = isaInfoMap
    self.srcName = srcName
    self.splitGSU = splitGSU
    config = config
    targetIsas = list(isaInfoMap.keys())

    self._state = {}
    # problem type
    if "ProblemType" in config:
      self["ProblemType"] = ProblemType(config["ProblemType"], printIndexAssignmentInfo)
    else:
      self["ProblemType"] = ProblemType.FromDefaultConfig(printIndexAssignmentInfo)

    if "InternalSupportParams" in config:
      self["InternalSupportParams"] = {}
      for key in defaultInternalSupportParams:
        assignParameterWithDefault(self["InternalSupportParams"], key, config["InternalSupportParams"], defaultInternalSupportParams)
    else:
      self["InternalSupportParams"] = defaultInternalSupportParams

    # Assign solution state from config, filling missing from the defaultSolution
    for key in defaultSolution:
      assignParameterWithDefault(self._state, key, config, defaultSolution)

    # Validate parameter types against the validParameters registry.
    # Catches bool-vs-int mismatches (YAML false vs 0) that would cause
    # std::bad_cast at C++ msgpack deserialization time.
    validateParameterTypes(self._state, srcFile=srcName)

    if 'ISA' not in self._state:
      if 'ISA' in config:
        # The ISA is expected to be defined when calling from TensileCreateLibrary
        isa = config['ISA']
        isa = IsaVersion(isa[0], isa[1], isa[2])
        assert self.isaInfoMap[isa].asmCaps["SupportedISA"]
        self._state['ISA'] = IsaVersion(isa[0], isa[1], isa[2])
      else:
        # When calling from Tensile, the ISA is typically not defined.
        printWarning(f"ISA not set on config using {targetIsas[0]}.")
        self._state['ISA'] = targetIsas[0]

    if "CodeObjectVersion" not in self._state:
      if "CodeObjectVersion" in config:
        self._state["CodeObjectVersion"] = str(config["CodeObjectVersion"])
      else:
        self._state["CodeObjectVersion"] = self.assembler.code_object_version
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
    Solution.assignDerivedParameters(
      self._state,
      splitGSU,
      printSolutionRejectionReason,
      printIndexAssignmentInfo,
      isaInfoMap,
      assembler.rocm_version
    )
    self._name = config["CustomKernelName"] if "CustomKernelName" in config and config["CustomKernelName"] else None

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
  # get Helper Kernels
  def getKernelBetaOlnyObjects(self):
    return self.betaOnlyKernelObjects


  ########################################
  # get Helper Kernels
  def getKernelConversionObjects(self):
    return self.conversionKernelObjects

  @staticmethod
  def getMIOutputInfo(state, isaInfoMap: Dict[str, IsaInfo]):
    outputVectorWidth = 4
    RegsPerOut = 1

    isa = tuple(state["ISA"])
    if isaInfoMap[isa].asmCaps['HasMFMA']:
      if state["ProblemType"]["DataType"].MIOutputTypeNameAbbrev() == 'f64':
        outputVectorWidth, RegsPerOut = 1, 2
      else:
        outputVectorWidth, RegsPerOut = 4, 1
    elif isaInfoMap[isa].asmCaps['HasWMMA_V1']:
        outputVectorWidth, RegsPerOut = 1, 1
    elif isaInfoMap[isa].asmCaps['HasWMMA_V2'] or isaInfoMap[isa].asmCaps['HasWMMA_V3']:
        outputVectorWidth, RegsPerOut = 8, 1
    else:
      print("WARNING: unexpect code flow")

    return outputVectorWidth, RegsPerOut

  ########################################
  # assign tile sizes
  @staticmethod
  def assignProblemIndependentDerivedParameters(state, printRejectionReason: bool, isaInfoMap: Dict[str, IsaInfo]):

    if "AssignedProblemIndependentDerivedParameters" in state:
      if state["AssignedProblemIndependentDerivedParameters"]:
        return
    state["AssignedProblemIndependentDerivedParameters"] = False
    if "Valid" not in state:
      state["Valid"] = True

    if (not state["ProblemType"]["StridedBatched"]) and (not state["ProblemType"]['Batched']):
      reject(state, printRejectionReason, "General Batched GEMM only support Batched Problem")

    if (not state["ProblemType"]["StridedBatched"]) and (state["ProblemType"]["OperationType"] != 'GEMM'):
      reject(state, printRejectionReason, "General Batched GEMM only support GEMM OperationType")

    ### ---> This is where we previously called matrixInstructionToMIParameters

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
        reject(state, printRejectionReason, "EnableMatrixInstruction undetermined")

    if EnableMatrixInstruction == True:
      state["MatrixInstM"]         = state["MIBlock"][0]
      state["MatrixInstN"]         = state["MIBlock"][1]
      state["MatrixInstK"]         = state["MIBlock"][2]
      state["MatrixInstB"]         = state["MIBlock"][3]
      state["MatrixInstBM"]        = state["MIBlock"][4]
      state["MatrixInstBN"]        = state["MIBlock"][5]

      state["LocalSplitU"] = state["WorkGroup"][2]
      state["NumWaveSplitK"] = 1

      state["MIOutputVectorWidth"], state["MIRegPerOut"] = Solution.getMIOutputInfo(state, isaInfoMap)

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

      state["MIWaveGroup"] = [state["SubGroup0"] // state["WavefrontSize"],  state["SubGroup1"]]


    if "SubGroup0" in state and "SubGroup1" in state and "LocalSplitU" in state and "NumWaveSplitK" in state:
      state["NumThreads"] = state["SubGroup0"] * state["SubGroup1"] * state["LocalSplitU"] * state["NumWaveSplitK"]
      if (state["NumThreads"] % state['WavefrontSize']) != 0:
        reject(state, printRejectionReason, f"size of WorkGroup {state['NumThreads']} should be multiple of WavefrontSize {state['WavefrontSize']}")

    # macro tile sizes
    if "SubGroup0" in state and "ThreadTile0" in state:
      state["MacroTile0"] = state["SubGroup0"]*state["ThreadTile0"]
    if "SubGroup1" in state and "ThreadTile1" in state:
      state["MacroTile1"] = state["SubGroup1"]*state["ThreadTile1"]
    if "MacroTile" in state:
      if state["MacroTile0"] != state["MacroTile"][0] \
          or state["MacroTile1"] != state["MacroTile"][1]:
        reject(state, printRejectionReason, "MacroTile mismatch")

    # dot2: currently only support fp16 with HPA on gfx942 or fp16 &bf16 on gfx950
    state["UseDotInstruction"] = (not state["EnableMatrixInstruction"]) \
      and state["ProblemType"]["HighPrecisionAccumulate"] \
      and ((state["ISA"] == IsaVersion(9,4,2) and state["ProblemType"]["DataType"].isHalf()) \
      or (state["ISA"] == IsaVersion(9,5,0) and (state["ProblemType"]["DataType"].isBFloat16() or state["ProblemType"]["DataType"].isHalf())))
    if state["UseDotInstruction"]:
      # need modification for dot4 or dot8
      state["NumDotElements"] = 2
    if not state["UseDotInstruction"] and state["WaveSplitK"]:
      reject(state, "WaveSplitK currently only support dot2 kernel.")
      return

    # workaround for MX
    # set ASEM=minASEMforMX for not TLUA or not TLUB
    # so far, kernel code can support 16, but host code cannot hanlde it
    # TODO: enable 16 (or less)
    # TODO: enable less than 256 for Subtile
    minASEMforMX = 32 if not state["UseSubtileImpl"] else 256
    if (state["ProblemType"]["MXBlockA"] or state["ProblemType"]["MXBlockB"]) and \
       ((not state["ProblemType"]["TLUA"]) or (not state["ProblemType"]["TLUB"])):
      if state["AssertSummationElementMultiple"] % minASEMforMX != 0:
        state["AssertSummationElementMultiple"] = minASEMforMX

    # tail loop optimization
    state["tailLoopOptA"] = True
    state["tailLoopOptB"] = True
    if state["ProblemType"]["MXBlockA"]:
      state["tailLoopOptMXSA"] = True
    if state["ProblemType"]["MXBlockB"]:
      state["tailLoopOptMXSB"] = True

    # Use nonDTL loads in DTL tail loop
    state["NonDTLTailLoopA"] = False
    state["NonDTLTailLoopB"] = False
    if state["ProblemType"]["MXBlockA"]:
      state["NonDTLTailLoopMXSA"] = False
    if state["ProblemType"]["MXBlockB"]:
      state["NonDTLTailLoopMXSB"] = False

    # Initialize DTLA, DTLB for tailLoopOpt/NonDTLTailLoop and initial calcLdsBlockSizePerPad() call
    state["DirectToLdsA"] = state["DirectToLds"] == 1 or state["DirectToLds"] == 2
    state["DirectToLdsB"] = state["DirectToLds"] == 1 or state["DirectToLds"] == 3

    bpeA = state["ProblemType"]["DataTypeA"].numBytes()
    bpeB = state["ProblemType"]["DataTypeB"].numBytes()
    aemA = state["AssertSummationElementMultiple"] if not state["ProblemType"]["TLUA"] else state["AssertFree0ElementMultiple"]
    aemB = state["AssertSummationElementMultiple"] if not state["ProblemType"]["TLUB"] else state["AssertFree1ElementMultiple"]
    # For DTL, we use nonDTL loads in tail loop only if
    # a partial 32b read is required to read the last few elements of a row/col of A/B
    # i.e. ASEM * BPE % 4 != 0. In this case dword/dwordx4 DTL load will
    # zero out the entire partial 32b read and cause accuracy issues.
    # TLU case, we check AF{0,1}EM instead of ASEM.
    # TLU + ShiftPtr case, we can use wider global read for TailLoop. Enable Tailloop opt for DTL
    # (ShiftPtr is default and not set to state["EdgeType"] yet here)
    # MX case, force enable NonDTLTailLoopA/B
    if ((aemA * bpeA) % 4 != 0 or not state["BufferLoad"]) or state["ProblemType"]["MXBlockA"]:
      if (not state["ProblemType"]["TLUA"]) and state["DirectToLdsA"] and not state["DirectToVgprA"]:
        state["NonDTLTailLoopA"] = True
        state["tailLoopOptA"] = False
        if state["ProblemType"]["MXBlockA"]:
          state["NonDTLTailLoopMXSA"] = True
          state["tailLoopOptMXSA"] = False
    if ((aemB * bpeB) % 4 != 0 or not state["BufferLoad"]) or state["ProblemType"]["MXBlockB"]:
      if (not state["ProblemType"]["TLUB"]) and state["DirectToLdsB"] and not state["DirectToVgprB"]:
        state["NonDTLTailLoopB"] = True
        state["tailLoopOptB"] = False
        if state["ProblemType"]["MXBlockB"]:
          state["NonDTLTailLoopMXSB"] = True
          state["tailLoopOptMXSB"] = False

    if (state["ISA"] != (9, 4, 2) and state["ISA"] != (9, 5, 0)) or \
       (state["ProblemType"]["Sparse"]) or \
       (state["UseDotInstruction"]):
      state["tailLoopOptA"] = False
      state["tailLoopOptB"] = False
      if state["ProblemType"]["MXBlockA"]:
        state["tailLoopOptMXSA"] = False
      if state["ProblemType"]["MXBlockB"]:
        state["tailLoopOptMXSB"] = False
    if (not state["ProblemType"]["TLUA"]) and (state["DirectToVgprA"]):
      state["tailLoopOptA"] = False
    if (not state["ProblemType"]["TLUB"]) and (state["DirectToVgprB"]):
      state["tailLoopOptB"] = False
    if state["ProblemType"]["MXBlockA"] and state["DirectToVgprMXSA"]:
      state["tailLoopOptMXSA"] = False
    if state["ProblemType"]["MXBlockB"] and state["DirectToVgprMXSB"]:
      state["tailLoopOptMXSB"] = False

    # so far, disable tailLoopOpt in MX case
    # TODO: enable tailLoopOpt for MX
    if state["ProblemType"]["MXBlockA"] or state["ProblemType"]["MXBlockB"]:
      state["tailLoopOptA"] = False
      state["tailLoopOptB"] = False
      state["tailLoopOptMXSA"] = False
      state["tailLoopOptMXSB"] = False

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

    # F32XEmulation related initialization
    state["UseDot2F32XEmulation"] = False
    state["UseMFMAF32XEmulation"] = False
    state["UseDirect32XEmulation"] = False # directly local read into temporary vgpr
    #ignore the F32 xDL MathOp by default.
    #enable F32 xDL MathOp only when the input type is f32.
    if state["UseF32XEmulation"]:
      state["UseF32XEmulation"] = True
      state["UseDirect32XEmulation"] = True
      state["UseDirect32XEmulationInterleaveTreg"] = False # True: enable conventional T reg allocation
      # select conversion logic for X3
      # (1) UseMFMAF32XEmulation = True
      # (2) UseDot2F32XEmulation = True (set (1) to False)
      # (3) cvt + sub  (set both (1) and (2) False)
      isa = state["ISA"]
      if isaInfoMap[isa].asmCaps.get("HasMFMA", False):
        state["UseMFMAF32XEmulation"] = True # MFMA version for gfx950 etc.

    state["MfmaInitCVgprs"] = False
    # Only enable UseSubtileImpl on gfx950; ignore user request on other ISAs.
    isgfx950 = state["ISA"] == IsaVersion(9,5,0)
    state["UseSubtileImpl"] = state["UseSubtileImpl"] and isgfx950

    if isgfx950 and (state["ProblemType"]["MXBlockA"] or state["ProblemType"]["MXBlockB"]) and not state["UseSubtileImpl"]:
        reject(state, printRejectionReason, "gfx950 MX requires UseSubtileImpl")

    if state["UseSubtileImpl"]:
      state["VectorWidthA"] = 1
      state["VectorWidthB"] = 1
      state["SourceSwap"] = False
      # Force BufferStore=1: UseSubtileImpl optimized storeD path is only implemented
      # for buffer stores for now.
      state["BufferStore"] = 1
      # Not currently implemented in subtile implementation
      state["Use64bShadowLimit"] = False
      state["Use64bShadowLimitMX"] = False

      # DepthU should be multiple of 2 * MIK. DepthU=-1 case, set DepthU=2*MIK*LSU
      duUnit = 2 * state["MatrixInstK"] * state["LocalSplitU"]
      if state["DepthU"] == -1:
        state["DepthU"] = duUnit
      if state["DepthU"] % duUnit != 0:
        reject(state, printRejectionReason, "UseSubtileImpl=1 support only DepthU multiple of 2 * MatrixInstK * LocalSplitU")

      for tc in ('A', 'B'):
        dtype = state["ProblemType"][f"DataType{tc}"]
        tlu = state["ProblemType"].get(f"TLU{tc}", False)
        if tlu:
          if dtype.isBFloat16() or dtype.isHalf():
            state[f"_ABTilePair{tc}"] = "AB_B16_TLU1"
          else:
            reject(state, printRejectionReason, f"No TLU=1 subtile geometry for dtype {dtype}")
            return
        elif dtype.isBFloat16() or dtype.isHalf():
          state[f"_ABTilePair{tc}"] = "AB_B16"
        elif dtype.is8bitFloat():
          state[f"_ABTilePair{tc}"] = "AB_B8"
        elif dtype.is6bitFloat() or dtype.isFloat4():
          state[f"_ABTilePair{tc}"] = "AB_B4"
        else:
          reject(state, printRejectionReason, f"No subtile geometry for dtype {dtype}")
          return

      bytesLoaded = state["NumThreads"] * 16
      if state["ProblemType"]["MXBlockA"]:
        numBytesMXSA = (state["DepthU"] // state["ProblemType"]["MXBlockA"]) * state["MacroTile0"]
        if bytesLoaded < numBytesMXSA:
          reject(state, printRejectionReason, "Unable to load MXSA scales using one load per wave")
      if state["ProblemType"]["MXBlockB"]:
        numBytesMXSB = (state["DepthU"] // state["ProblemType"]["MXBlockB"]) * state["MacroTile1"]
        if bytesLoaded < numBytesMXSB:
          reject(state, printRejectionReason, "Unable to load MXSB scales using one load per wave")

      for tc in ("MXSA", "MXSB"):
        if state["ProblemType"]["MXBlock" + tc[-1]]:
          for field in ("LSC", "LSP", "NumLoadsCoalesced", "NumLoadsPerpendicular", "NumTotalPackedLoads"):
            state["%s%s" % (field, tc)] = 0

      if state["PrefetchGlobalRead"] not in [0, 2]:
        reject(state, printRejectionReason,
               "UseSubtileImpl=1 requires PrefetchGlobalRead 0 or 2, got %d" % state["PrefetchGlobalRead"])
      if not (state["MatrixInstM"] == 16 and state["MatrixInstN"] == 16):
        reject(state, printRejectionReason, "UseSubtileImpl=1 requires MatrixInst 16x16")
      if state["ScheduleIterAlg"] == 1 or state["ScheduleIterAlg"] == 2:
        reject(state, printRejectionReason, "UseSubtileImpl=1 does not support ScheduleIterAlg")
      if state["StreamK"] == 0:
        reject(state, printRejectionReason, "UseSubtileImpl=1 supports StreamK only (no support for GSU)")

    # TODO: Support other LdsBlockSizePerPadMXSA/B for gfx1250.
    if state["ISA"] == (12, 5, 0):
      if ((state["LdsBlockSizePerPadMXSA"] > 0) or (state["LdsBlockSizePerPadMXSB"] > 0 )):
        reject(state, "LdsBlockSizePerPadMXSA/LdsBlockSizePerPadMXSB support -1 and 0 for gfx1250")
        return

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
  def setGlobalReadVectorWidth(state, tc, totalVectors, grvw, printRejectionReason: bool):
    validDepthU = True
    # Skip GRVW range check for subtile impl: scale uses serial*loadWidth DTL addressing, not standard GRVW chunks
    if grvw not in [1,2,4,8,16,32] and not state["UseSubtileImpl"]:
      validDepthU = False
    if totalVectors % state["NumThreads"] != 0:
      reject(None, printRejectionReason, "totalVectors%s %u %% NumThreads %u != 0" \
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
  def setGlobalLoadTileDimClassic(state, tc, numLoads, totalVectorsCoalesced, totalElementsPerp, depthU, printRejectionReason: bool):

    if state["WaveSeparateGlobalRead%s"%tc]:
      totalElementsPerp = roundupRatio(totalElementsPerp, state["NumThreads"] // state["WavefrontSize"])

    if (tc == "A" or tc == "B") and state["enableGLTr%s"%tc]:
      totalVectorsCoalesced, totalElementsPerp = totalElementsPerp, totalVectorsCoalesced

    # nlc = 1
    if state["NumLoadsCoalesced%s"%tc] == 1:
      foundValid = False
      nlcStart = 1
      if (tc == "A" or tc == "B") and state["DirectToVgpr%s"%tc]:
        # adjust nlc for DirectToVgprA/B
        if state["ProblemType"]["TLU%s"%tc] and not state["enableGLTr%s"%tc]:
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
        reject(state, printRejectionReason, "%s: No NumLoadsCoalesced=1 found"%tc)
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
        reject(state, printRejectionReason, "%s: No NumLoadsCoalesced=-1 found"%tc)
        return False

    # nlc = other
    else:
      if state["NumLoadsCoalesced%s"%tc] > state["NumLoads%s"%tc]:
        reject(state, printRejectionReason, "%s nlc > numLoads"%tc)
        return False

      state["NumLoadsPerpendicular%s"%tc] = state["NumLoads%s"%tc] \
          // state["NumLoadsCoalesced%s"%tc]

      if state["NumLoads%s"%tc] % state["NumLoadsCoalesced%s"%tc] != 0:
        reject(state, printRejectionReason, "%s: numLoads %u %% numLoadsCoalesced %u != 0" \
            % (tc, state["NumLoads%s"%tc], state["NumLoadsCoalesced%s"%tc]))
        return False

      if totalVectorsCoalesced % state["NumLoadsCoalesced%s"%tc] != 0 :
        reject(state, printRejectionReason, "%s: totalVectorsCoalesced %u %% numLoadsPara %u != 0" \
              % (tc, totalVectorsCoalesced, state["NumLoadsCoalesced%s"%tc]))
        return False
      if totalElementsPerp % state["NumLoadsPerpendicular%s"%tc] != 0:
        reject(state, printRejectionReason, "%s: totalElementsPerp %u %% numLoadsPerp %u != 0" \
              % (tc, totalElementsPerp, state["NumLoadsPerpendicular%s"%tc]))
        return False

    if (tc == "A" or tc == "B") and state["enableGLTr%s"%tc]:
      state["NumLoadsCoalesced%s"%tc], state["NumLoadsPerpendicular%s"%tc] = state["NumLoadsPerpendicular%s"%tc], state["NumLoadsCoalesced%s"%tc]

    # Generalized nlc = 1 case
    idxWG = 0 if tc == 'A' else 1
    totalLoadsNeeded = (state["MacroTile%s"%tc] * depthU) // (state["GlobalReadVectorWidth%s"%tc] * state["WavefrontSize"])
    MT = state["MacroTile%s"%tc]
    numWaves = state["NumThreads"] // state["WavefrontSize"]
    if state["UseGeneralizedNLCOne%s"%tc]:
      if (totalLoadsNeeded % numWaves) != 0:
        reject(state, printRejectionReason, "GNLC%s: totalLoadsNeeded (%u) %% numWaves (%u) != 0"%(tc, totalLoadsNeeded, numWaves))
      state["NumTotalPackedLoads%s"%tc] = totalLoadsNeeded // numWaves
      state["NumLoadsPerpendicular%s"%tc] = state["NumTotalPackedLoads%s"%tc]
      state["NumLoadsCoalesced%s"%tc] = 1
    else:
      state["NumTotalPackedLoads%s"%tc] = -1

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




  ##############################################
  # check and calculate Wave Separate Global Read
  @staticmethod
  def checkAndAssignWaveSeparateGlobalRead(state, tc, printRejectionReason: bool):
    # check can we use WaveSeparateGlobalRead
    numOfWaves = state["NumThreads"] // state["WavefrontSize"]
    if state["WaveSeparateGlobalRead%s"%tc]:
      if state["ProblemType"]["TLU%s"%tc] and (state["_DepthU%s"%tc] > 0) and (state["_DepthU%s"%tc] % numOfWaves != 0):
        reject(state, printRejectionReason, "didn't support WaveSeparateGlobalRead when DepthU is not multiple of wave %u in TLU%s" % (state["_DepthU%s"%tc], tc))
      if not state["ProblemType"]["TLU%s"%tc] and (state["MacroTile%s" % tc] % numOfWaves != 0):
        reject(state, printRejectionReason, "didn't support WaveSeparateGlobalRead when MacroTile is not multiple of wave %u in TLU%s" % (state["MacroTile%s"%tc], tc))


  ########################################
  # determine can we use VgprForLocalReadPacking
  @staticmethod
  def isVgprForLocalReadPackingDoable(state, isaInfoMap: Dict[str, IsaInfo]):
    isa = tuple(state["ISA"])
    doable = True
    # MatrixInstruction only
    if not state["EnableMatrixInstruction"]:
      doable = False
    # only for HasEccHalf
    if not isaInfoMap[isa].archCaps["HasEccHalf"]:
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
  def isDirectToVgprDoable(state, tc, printRejectionReason: bool, isaInfoMap: Dict[str, IsaInfo]):
    numBytes = state["ProblemType"]["MacDataType%s"%tc].numBytes()
    numBytesGR = state["ProblemType"]["DataType%s"%tc].numBytes()
    isSwizzle = state["ProblemType"]["SwizzleTensor%s"%tc]

    # With MatrixInstruction only
    if not state["EnableMatrixInstruction"] :
      reject(state, printRejectionReason, "DirectToVgpr is for MatrixInstruction only")
      return False

    if state["LocalReadVectorWidth%s" % tc] < state["MIInputPerThread"]:
      reject(state, "LocalReadVectorWidth < MIInputPerThread %d" % state["MIInputPerThread"])
      return False

    # disable the following combinations for initial implementation
    # TODO: enable them (for pure DTV)
    if (state["LocalSplitU"] != 1) and (not state["ProblemType"]["TLU%c"%tc]) and (not isSwizzle):
      reject(state, printRejectionReason, "Non-Swizzled DirectToVgpr + LSU + TLU=False has not been enabled yet(tentative)")
      return False

    if state["DirectToVgprA"] and state["DirectToVgprB"]:
      # change the following parameter values
      state["PrefetchGlobalRead"] = 1
      state["ExpandPointerSwap"] = False
      state["1LDSBuffer"] = 0
      state["PrefetchLocalRead"] = 0
      # So far, DTVA + DTVB does not perform well (waitcnt is not ideal).
      # Disable it for now (TODO: improve waitcnt and re-enable)
      # Exception: allow if enableGLTrA or enableGLTrB are set; this still provides better performance than enabling only one side
      if not (state["enableGLTrA"] or state["enableGLTrB"]):
        reject(state, printRejectionReason, "DirectToVgprA + DirectToVgprB disabled")
        return False

    # DTV + input type conversion
    if state["ProblemType"]["DataType%s"%tc] != state["ProblemType"]["MacDataType%s"%tc]:
      if not state["ConvertAfterDS"]:
        reject(state, printRejectionReason, "DirectToVgpr%s + input conversion + ConvertAfterDS=False not supported"%(tc))
        return False

    # check if the DataType can support DirectToVgpr
    if not Solution.isDirectToVgprSupportDataType(state):
      reject(state, printRejectionReason, "no DirectToVgpr support for this input data type")
      return False

    # Does not work with TLU = False and PrefetchLocalRead = 0
    if (not state["ProblemType"]["TLU%c"%tc]) and state["PrefetchLocalRead"] == 0 and not (state["DirectToVgprA"] and state["DirectToVgprB"]):
      reject(state, printRejectionReason, "DirectToVgpr%c does not supports TLU%c = False and PrefetchLocalRead = 0"%(tc, tc))
      return False

    # Does not work with TLU = False and CGEMM/DGEMM/DGEMM (not supported)
    if (not state["ProblemType"]["TLU%c"%tc]) and (state["ProblemType"]["MacDataType%s"%tc].isDouble() or \
        state["ProblemType"]["MacDataType%s"%tc].isComplex()):
      reject(state, printRejectionReason, "DirectToVgpr%c does not supports TLU%c = False + S/C/D/ZGEMM"%(tc, tc))
      return False

    if numBytesGR * state["GlobalReadVectorWidth%c"%tc] < 4:
      # no support for DTV + numBytesGR * GlobalReadVectorWidth< 4
      reject(state, printRejectionReason, "DirectToVgpr%c does not support TLU%c + numByte * GlobalReadVectorWidth%c < 4"%(tc, tc, tc))
      return False

    if numBytes < 4:
      # numBytes < 4 case
      if state["ProblemType"]["TLU%c"%tc] and not state["enableGLTr%c"%tc]:
        # use pack logic (with v_perm) same as local read (only if VgprForLocalReadPacking is doable)
        if not Solution.isVgprForLocalReadPackingDoable(state, isaInfoMap):
          reject(state, printRejectionReason, "Does not meet the requirement for DirectToVgpr%c + TLU%c + numByte < 4"%(tc, tc))
          return False
        # force ClusterLocalRead=1 for DTV + pack
        state["ClusterLocalRead"] = 1
    else:
      # numBytes >= 4 case
      if state["ProblemType"]["TLU%c"%tc] and state["MIInputPerThread"] > 1:
        # no support for numBytes >= 4 + MIInputPerThread > 1
        reject(state, printRejectionReason, "DirectToVgpr%c does not support TLU%c+ numByte >= 4 + MIInputPerThread > 1"%(tc, tc))
        return False

    # MatrixInstBM,BN check
    #  for A, MatrixInstBN should be 1
    #  for B, MatrixInstBM should be 1
    # This is to limit the number of Vgpr
    if tc == 'A' and not (state['MatrixInstBN'] == 1):
      reject(state, printRejectionReason, "MatrixInstBN should be 1 for DirectToVgprA. Current value is %d"%(state['MatrixInstBN']))
      return False
    if tc == 'B' and not (state['MatrixInstBM'] == 1):
      reject(state, printRejectionReason, "MatrixInstBM should be 1 for DirectToVgprB. Current value is %d"%(state['MatrixInstBM']))
      return False

    # Does not work with WaveSeparateGlobalRead
    if state["WaveSeparateGlobalRead%c"%tc]:
      reject(state, printRejectionReason, "DirectToVgpr%c does not supports WaveSeparateGlobalRead%c"%(tc, tc))
      return False

    # Does not work with TLU + VectorWidth != GlobalReadVectorWidth (VW = 2 + GRVW = 1 or VW = 1 + GRVW = 2 does not work)
    if state["ProblemType"]["TLU%c"%tc] and (state["VectorWidth%s"%tc] != state["GlobalReadVectorWidth%c"%tc]) and (not state["enableGLTr%s"%tc]):
      reject(state, printRejectionReason, "DirectToVgpr%c does not supports TLU + VectorWidth%s(=%u) != GlobalReadVectorWidth%c(%u)"%(tc, tc, state["VectorWidth%s"%tc], tc, state["GlobalReadVectorWidth%c"%tc]))
      return False

    # Does not work with TLU=False and NumLoadsCoalesced != DepthU//(MatrixInstK*GRVW*LSU//MIInputPerThread)
    if (not state["ProblemType"]["TLU%c"%tc]) and \
        state["NumLoadsCoalesced%c"%tc] != state["DepthU"] // (state["MatrixInstK"] * state["GlobalReadVectorWidth%c"%tc] * state["LocalSplitU"] // state["MIInputPerThread"]):
      reject(state, printRejectionReason, "DirectToVgpr%c does not supports TLU=False and NumLoadsCoalesced%c != DepthU//(MatrixInstK*GlobalReadVectorWidth*LocalSplitU//MIInputPerThread(=%u))"%(tc, tc, state["MIInputPerThread"]))
      return False

    # Does not work with enableGLTr and GRVW != 8
    if state["enableGLTr%c"%tc] and state["GlobalReadVectorWidth%c"%tc] != 8:
      reject(state, printRejectionReason, "DirectToVgpr%c does not supports enableGLTr%c and GlobalReadVectorWidth != 8"%(tc, tc))
      return False

    # TLU=False or enableGLTr case, need GlobalReadVectorWidth == LocalReadVectorWidth
    if ((not state["ProblemType"]["TLU%c"%tc]) or state["enableGLTr%c"%tc]) and \
       state["GlobalReadVectorWidth%c"%tc] != state["LocalReadVectorWidth%s"%tc]:
      reject(state, printRejectionReason, "DirectToVgpr%c does not supports TLU=False GlobalReadVectorWidth%c(%u) != LocalReadVectorWidth(%u)"%(tc, tc, state["GlobalReadVectorWidth%c"%tc], state["LocalReadVectorWidth%s"%tc]))
      return False

    # Does not work with SIA<3
    if state["ScheduleIterAlg"] < 3:
      reject(state, printRejectionReason, "DirectToVgpr%c does not supports ScheduleIterAlg < 3"%(tc))
      return False

    # Does not work with InnerUnroll>1
    if state["InnerUnroll"]>1:
      reject(state, printRejectionReason, "DirectToVgpr%c does not supports InnerUnroll>1"%(tc))
      return False

    # Reject TLU = UnrollMajorLDS
    if state["ProblemType"]["TLU%c"%tc] == state["UnrollMajorLDS%c"%tc]:
      reject(state, printRejectionReason, "DirectToVgpr%c does not supports TLU%c = UnrollMajorLDS%c"%(tc, tc, tc))
      return False

    # does not work with UnrollLoopSwapGlobalReadOrder
    if state["UnrollLoopSwapGlobalReadOrder"]:
      reject(state, printRejectionReason, "DirectToVgpr%c does not supports UnrollLoopSwapGlobalReadOrder"%(tc))
      return False

    # does not work with PGR>=2 + EPS
    if state["PrefetchGlobalRead"] >= 2 and state["ExpandPointerSwap"]:
      # force EPS=0 and continue
      state["ExpandPointerSwap"] = False

    # does not work with Sparse
    if state["ProblemType"]["Sparse"]:
      reject(state, printRejectionReason, "DirectToVgpr%c does not supports Sparse"%(tc))
      return False

    # for DTVA/DTVB, does not work with PGR0
    if state["PrefetchGlobalRead"] == 0:
      reject(state, printRejectionReason, "DirectToVgpr%c does not supports PrefetchGlobalRead == 0."%(tc))
      return False

    # for DTVA, does not work with NN and TLDS0
    if tc == 'A' and state["TransposeLDS"] == 0 and (not state["ProblemType"]["TransposeA"] and not state["ProblemType"]["TransposeB"]):
      reject(state, printRejectionReason, "DirectToVgpr%c does not supports NN case with TransposeLDS == 0."%(tc))
      return False

    # for DTVA, does not work with TT and Tail-loop
    if tc == 'A' and (state["ProblemType"]["TransposeA"] and state["ProblemType"]["TransposeB"]):
        # Use AssertSummationElementMultiple (BoundSizeMultiple in predicates) to exclude failed tail-loop cases
        state["AssertSummationElementMultiple"] = max(state["AssertSummationElementMultiple"], state["DepthU"])

    # for DTVB, does not work with NN and Tail-loop
    if tc == 'B' and (not state["ProblemType"]["TransposeA"] and not state["ProblemType"]["TransposeB"]):
        # Use AssertSummationElementMultiple (BoundSizeMultiple in predicates) to exclude failed tail-loop cases
        state["AssertSummationElementMultiple"] = max(state["AssertSummationElementMultiple"], state["DepthU"])

    # Does not work with DirectToLDS
    # -> this will be checked after DirectToLDS doable check is done

    return True

  ########################################
  # determine can we use DirectToLds
  @staticmethod
  def isDirectToLdsDoable(state, tc, isaInfoMap, printRejectionReason: bool):
    isa = state["ISA"]

    if state["UseSubtileImpl"]:
      return True

    # x4 support for directToLds
    canDTLx4 = isaInfoMap[isa].asmCaps["HasDirectToLdsx4"]

    # numelements_perlane = 4/numBytes
    # TN with transposeLDS feature should work as long as state["AssertSummationElementMultiple"] % (numelements_perlane*2) = 0
    #                                                     state["AssertSummationElementMultiple"] % (numelements_perlane*4) = 0

    #NT
    # use only for f32 & DGEMM and TLU = 1
    #TN
    # use for all precisions with TransposeLDS=1

    if tc in ["MXSA", "MXSB"]:
      numBytesAB = 1
    else:
      numBytesAB = state["ProblemType"]["DataType%s"%tc].numBytes()
    numBytesPerLoad = state["GlobalReadVectorWidth%s"%tc] * numBytesAB

    # x2 DTL is not supported
    if numBytesPerLoad == 8:
      printWarning("can't use DirectToLds with b64 buffer load, using non DirectToLds version instead")
      return False

    if numBytesPerLoad == 16 and not canDTLx4:
      reject(state, printRejectionReason, "b128 DirectToLds not supported")
      return False

    if numBytesPerLoad < 4:
      reject(state, printRejectionReason, "DirectToLds not supported for loads less than 32bits")
      return False

    if tc in ["MXSA", "MXSB"]:
      # MXSA/B case, check numBytesPerLoad only
      return True

    MT = state["MacroTile0"] if tc == 'A' else state["MacroTile1"]

    if (MT & (MT-1)) != 0 and not state["UseGeneralizedNLCOne%s"%tc]: # Check of MT not power of 2
      # so far, numBytesAB<=4 case, TLU=False only (continue with False)
      if (numBytesAB <= 4) and state["ProblemType"]["TLU%c"%tc]:
        return False

    # so far MFMA only (TODO: enable non MFMA case)
    if not state["EnableMatrixInstruction"]:
      reject(state, printRejectionReason, "DirectToLds is for MatrixInstruction only for now (tentative)")
      return False

    # ToDo: Review def of lrvw and this check
    # DTL + LocalReadVectorWidth > MIInputPerThread does not work
    # Need support for TailLoop
    if not state["ProblemType"]["Sparse"] and not(state["ProblemType"]["MacDataType%s"%tc].is8bitFloat() and (state["MatrixInstK"] == 64 or state["MatrixInstK"] == 128)):
      if state["LocalReadVectorWidth%s"%tc] > state["MIInputPerThread"]:
        reject(state, printRejectionReason, "DirectToLds does not work with LocalReadVectorWidth > MIInputPerThread")
        return False

    if state["NumThreads"] % state["WavefrontSize"] != 0:
      reject(state, printRejectionReason, "can't use DirectToLds for NumThreads % WavefrontSize != 0")
      return False

    if state["ProblemType"]["TLU%c"%tc] == state["UnrollMajorLDS%c" % tc]:
      printWarning("can't use DirectToLds for TLU%c == UnrollMajorLDS%c, using nonDirectToLds version instead"%(tc, tc))
      return False

    # avoid picking x2&x4 for precisions < f32/f64 in [ProblemType][TLU] == TRUE
    if not state["EnableMatrixInstruction"]:
      if state["GlobalReadVectorWidth%c"%tc] * numBytesAB * state["WavefrontSize"] > 256:
        reject(state, printRejectionReason, "can't use DirectToLds for not EnableMatrixInstruction and GlobalReadVectorWidth%c * bpe%c * WavefrontSize > 256"%(tc,tc))
        return False

    if state["WaveSeparateGlobalRead%c" % tc]:
      if state["LSC%c"%tc] * state["LSP%c"%tc] * numBytesAB != state["WavefrontSize"] * state["GlobalReadVectorWidth%c"%tc] * numBytesAB:
        reject(state, printRejectionReason, "can't use DirectToLds for LSC%c and LSP%c * bpe!= WavefrontSize * GlobalReadVectorWidth%c * bpe%c > 4"%(tc, tc, tc, tc))
        return False
      if state["WaveSeparateGlobalRead%c" % tc] == 2:
        reject(state, printRejectionReason, "can't use DirectToLds for WSGR%s = 2"%(tc))
        return False
    elif not state["UseGeneralizedNLCOne%s"%tc]:
      if state["LSC%c"%tc] * state["LSP%c"%tc] * numBytesAB != state["NumThreads"] * state["GlobalReadVectorWidth%c"%tc] * numBytesAB:
        reject(state, printRejectionReason, "can't use DirectToLds for LSC%c and LSP%c * bpe != NumThreads * GlobalReadVectorWidth%c * bpe%c > 4"%(tc, tc, tc, tc))
        return False

    # so far, DirectToLds does not work with LRVW=2
    if state["LocalReadVectorWidth%s"%tc] == 2:
      reject(state, printRejectionReason, "can't use DirectToLds for LocalReadVectorWidth == 2")
      return False

    # Does not work with (NumLoadsCoalesced>1 and UseInstOffsetForGRO) + DGEMM
    if state["ProblemType"]["MacDataType%s"%tc].isDouble() and \
      (state["NumLoadsCoalesced%c"%tc] > 1 and state["UseInstOffsetForGRO"]):
      reject(state, printRejectionReason, "DirectToLds%c does not supports NumLoadsCoalesced%c > 1 and UseInstOffsetForGRO for dgemm"%(tc, tc))
      return False

    # Does not work with NumLoadsCoalesced>1 + ZGEMM
    if state["ProblemType"]["MacDataType%s"%tc].isDoubleComplex() and state["NumLoadsCoalesced%c"%tc] > 1:
      reject(state, printRejectionReason, "DirectToLds%c does not supports NumLoadsCoalesced%c > 1 for zgemm"%(tc, tc))
      return False

    # DirectToLds does not work with TLU=False and bpe > bpr and DepthU//NumLoadsCoalesced < 8
    # bpe > bpr case, Lower and upper 4 bytes elements are stored separately.
    # if TLU=False and DepthU//NumLoadsCoalesced is smaller than lower block size (8 elements),
    # current offset swap logic does not work
    if (not state["ProblemType"]["TLU%c"%tc]) and state["ProblemType"]["MacDataType%s"%tc].numRegisters() > 1 and \
       state["_DepthU%s"%tc] // state["NumLoadsCoalesced%c"%tc] < 8:
      reject(state, printRejectionReason, "DirectToLds%c does not work with TLU=False and bpe > bpr and DepthU//NumLoadsCoalesced%c < 8"%(tc, tc))
      return False

    # TODO: Currently DTL with input types of different size is not support. There are functional issues
    # This needs to be fixed.
    if state["ProblemType"]["DataType%s"%tc].numBytes() != state["ProblemType"]["MacDataType%s"%tc].numBytes():
      reject(state, printRejectionReason, "DirectToLds%s with conversion to different sized data types is not supported"%tc)
      return False

    # DTL + input type conversion
    if state["ProblemType"]["DataType%s"%tc] != state["ProblemType"]["MacDataType%s"%tc]:
      if not state["ConvertAfterDS"]:
        reject(state, printRejectionReason, "DirectToLds%s + input conversion + ConvertAfterDS=False not supported"%(tc))
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
  def assignDerivedParameters(
    state,
    splitGSU: bool,
    printRejectionReason: bool,
    printIndexAssignmentInfo: bool,
    isaInfoMap,
    rocmVersion: SemanticVersion
  ):
    isa = tuple(state["ISA"])

    if state["WavefrontSize"] == -1:
      state["WavefrontSize"] = 32 if isaInfoMap[isa].archCaps["HasWave32"] else 64

    if state["MaxLDS"] == -1:
      state["MaxLDS"] = isaInfoMap[isa].archCaps["DeviceLDS"]

    # NOTE: This entry should instead should already be set on the solution within the logic
    # files. This code will be removed once all logic files are updated to contain both
    # the keys "EnableF32XdlMathOp" and "F32XdlMathOp".
    state["EnableF32XdlMathOp"] = False
    state["UseF32XEmulation"] = False # enable emulation for missing hardware support
    #ignore the F32 xDL MathOp by default.
    #enable F32 xDL MathOp only when the input type is f32.
    if "F32XdlMathOp" in state["ProblemType"] \
       and (not state["ProblemType"]["F32XdlMathOp"].isSingle()) \
       and (state["ProblemType"]["DataType"].isSingle()):
      state["EnableF32XdlMathOp"] = True
      if isaInfoMap[isa].archCaps["HasF32XEmulation"]:
        state["UseF32XEmulation"] = True


    # initial info to be exported for solution prediction
    state["CUOccupancy"]            = -1
    state["MathClocksUnrolledLoop"] = 0

    Solution.assignProblemIndependentDerivedParameters(state, printRejectionReason, isaInfoMap)
    # KRingShift currently only supported for TN (A transposed, B not transposed).
    # Disallow enabling KRingShift on NN/NT/TT until those paths are validated.
    if state["KRingShift"]:
      ta = int(state["ProblemType"]["TransposeA"])
      tb = int(state["ProblemType"]["TransposeB"])
      if not (ta == 1 and tb == 0):
        reject(state, printRejectionReason, f"KRingShift requires TN (TransposeA=1, TransposeB=0); got TransposeA={ta}, TransposeB={tb}")
        return

    # KRingShift is defined to operate only in conjunction with BAddrInterleave (BInterleaveG).
    # If BAddrInterleave is not enabled, do not allow KRingShift to be enabled.
    if state["KRingShift"] and (not state["BAddrInterleave"]):
      reject(state, printRejectionReason, "KRingShift requires BAddrInterleave (BInterleaveG)")
      return

    # BAddrInterleave runtime restriction (host-side predicate, not codegen):
    # Match the kernel's initBInterleaveG enable conditions:
    #   - require tiles1 = SizeJ / MT1 to be an integer (SizeJ % MT1 == 0)
    #   - require lowbit(tiles1) > 1 so that G=min(lowbit(tiles1), LVCB) is > 1 (enabled)
    # Note: if lowbit(tiles1) == 1, then G==1 and the kernel disables BAddrInterleave.
    if state["BAddrInterleave"]:
      state["AssertFree1DivByMT1LowbitGT1"] = state["MacroTile1"]

    if state["UseDirect32XEmulation"] == True:
      #   Turn off Direct32X for the following kernels:
      #   Cijk_Ailk_Bjlk_S_MX_B_Bias_HA_S_SAV_UserArgs_MT16x16x512_MI16x16x1
      if (state["MacroTile0"] == 16 and state["MacroTile1"] == 16 and state["DepthU"] == 512):
        state["UseDirect32XEmulation"] = False
    if "AssignedDerivedParameters" in state:
      if state["AssignedDerivedParameters"]:
        return
    state["AssignedDerivedParameters"] = False

    for s in Solution.InternalKeys:
      if '_'+s not in state:
        state['_'+s] = state[s]
        #del state[s]

    # Force update _GlobalAccumulation
    computeBytes = int(state["ProblemType"]["ComputeDataType"].numBytes())
    state["_GlobalAccumulation"] = None
    computeName  = state["ProblemType"]["ComputeDataType"].toName()
    if state["UseDotInstruction"] and state["GlobalSplitUAlgorithm"] == 'MultipleBufferSingleKernel':
      # dot2 kernel does not support MBSK
      state["GlobalSplitUAlgorithm"] = 'MultipleBuffer'
      state["MbskPrefetchMethod"] = 0
    if state["StreamK"] > 0 and state["StreamKAtomic"] == 0:
      # StreamK Workspace size
      state["_GlobalAccumulation"] = 'PartialsBuffer'
    elif state["GlobalSplitUAlgorithm"] == 'SingleBuffer':
      if computeName != state["ProblemType"]["DestDataType"].toName():
        state["_GlobalAccumulation"] = 'SingleBuffer'
    elif state["GlobalSplitUAlgorithm"] == 'MultipleBuffer':
      state["_GlobalAccumulation"] = 'MultipleBuffer'
    elif state["GlobalSplitUAlgorithm"] == 'MultipleBufferSingleKernel':
      if (not splitGSU):
        state["_GlobalAccumulation"] = 'MultipleBufferSingleKernel'
      else:
        if state["GlobalSplitU"] > 1 or state["GlobalSplitU"] == -1:
          state["_GlobalAccumulation"] = 'MultipleBufferSingleKernel'

    if state["_GlobalAccumulation"] == 'MultipleBufferSingleKernel':
      state["SynchronizerSizeCheck"] = 1
    #   state["BatchSizeEqual"] = 1

    if state["StreamK"] == 0 and state["GlobalSplitU"] == 0:
      reject(state, printRejectionReason, "Either GSU or StreamK must be enabled")
      return

    if state["StreamK"] != 0:
      #state["AssertSummationElementMultiple"] = 1 # Cannot keep ASEM with Stream-K
      state["GlobalSplitU"] = 0 # Cannot enable both Stream-K and GSU
      state["InternalSupportParams"]["SupportUserGSU"] = False # Disable UserGSU for Stream-K
      state["GlobalSplitUAlgorithm"] = "MultipleBuffer" # Set default Algorithm
      state["AdaptiveGemmGSUA"] = 0 # Disable AdaptiveGemmGSUA for Stream-K
      if not state["EnableMatrixInstruction"]:
        reject(state, printRejectionReason, "Stream-K requires MatrixInstruction")
      # if state["PersistentKernel"]:
      #   reject(state, printRejectionReason, "Cannot enable both Stream-K and PersistentKernel")
      if not state["ProblemType"]["StridedBatched"]:
        reject(state, printRejectionReason, "General batch not supported with Stream-K")
      if state["ProblemType"]["GroupedGemm"]:
        reject(state, printRejectionReason, "Grouped gemm not yet supported with Stream-K")
      if state["ScheduleGlobalRead"] != 1:
        reject(state, printRejectionReason, "ScheduleGlobalRead not supported with Stream-K")
      if state["ScheduleLocalWrite"] != 1:
        reject(state, printRejectionReason, "ScheduleLocalWrite not supported with Stream-K")
      if state["ScheduleIterAlg"] != 2 and state["ScheduleIterAlg"] != 3:
        reject(state, printRejectionReason, "ScheduleIterAlg not supported with Stream-K")
      if state["StreamKAtomic"] == 1:
        if not state["ProblemType"]["DataType"].isSingle():
          reject(state, printRejectionReason, "Atomic Stream-K currently only tested for SGEMM")
        if not state["BufferStore"]:
          reject(state, printRejectionReason, "Atomic Stream-K requires BufferStore")
        if state["LocalSplitU"] > 1:
          reject(state, printRejectionReason, "Atomic Stream-K not working with LocalSplitU")
      if not state["Valid"]:
        print2("in assignDerivedParameters, state['Valid'] = False")
        return
    else:
      # If not using StreamK, clear other stream-k settings to avoid duplicate kernels
      state["StreamKAtomic"] = 0
      state["StreamKXCCMapping"] = 0
      state["StreamKFixupTreeReduction"] = 0
      state["DebugStreamK"] = 0

    computeBytes = int(state["ProblemType"]["ComputeDataType"].numBytes())
    state["_WorkspaceSizePerElemC"] = computeBytes
    state["_WorkspaceSizePerElemBias"] = 0
    if state["ProblemType"]["UseBias"] and state["ProblemType"]["Gradient"]:
      state["_WorkspaceSizePerElemBias"] = computeBytes

    state["WorkspaceCheck"] = [state["_WorkspaceSizePerElemC"], state["_WorkspaceSizePerElemBias"], state["GlobalSplitU"] if (state["GlobalSplitUAlgorithm"] == 'MultipleBuffer' or state["_GlobalAccumulation"] == 'MultipleBufferSingleKernel') else 1]

    if state["VectorStore"] == -1:
        state["_VectorStore"] = 1 # default, may be changed if needed to generate a valid kernel

    ProblemType.assignDerivedParameters(state["ProblemType"], printIndexAssignmentInfo)
    if not state["Valid"]:
      print2("in assignDerivedParameters, state['Valid'] = False")
      return

    if not isaInfoMap[isa].asmCaps["HasNTModifier"]:
      # force to disable nt flag if it is not supported by arch
      for ch in ["", "A", "B", "C", "D", "E", "WS", "Metadata"]:
        if state["NonTemporal%s"%ch] >= 4:
          state["NonTemporal%s"%ch] -= 4

    if state["WavefrontSize"] == 32 and not isaInfoMap[isa].archCaps["HasWave32"]:
      reject(state, printRejectionReason, "WavefrontSize=32 not supported for ISA {}".format(isa))
      return

    if state["WavefrontSize"] == 32 and state["KernelLanguage"] == "Source":
      reject(state, printRejectionReason, "WavefrontSize=32 not yet supported for source kernels.")
      return

    if state["EnableMatrixInstruction"]:
      if not (isaInfoMap[isa].asmCaps["HasMFMA"] or isaInfoMap[isa].asmCaps["HasWMMA"]):
        reject(state, printRejectionReason, f"isa {isa} doesn't support matrix instruction")
        return
      key = state["ProblemType"]["MacDataTypeA"].toChar() + state["ProblemType"]["MacDataTypeB"].toChar()
      if key not in ("SS", "XX", "DD", "BB", "HH", "CC", "ZZ", "I8I8", \
                     "F8F8", "F8B8", "B8B8", "B8F8", \
                     "F8NF8N", "F8NB8N", "B8NB8N", "B8NF8N", \
                     "F8F6", "F6F8", "F8F4", "F4F8","B8F6", "F6B8", "B8F4", "F4B8", "F8B6", "B6F8","B8B6", "B6B8",\
                     "F6F6", "B6B6", "F6B6", "B6F6","B6F4", "F4B6", "F6F4", "F4F6", "F4F4"):
        reject(state, printRejectionReason, "didn't support Matrix Instruction with type %s" % key)
        return
      if (not isaInfoMap[isa].asmCaps["HasMFMA"] and isaInfoMap[isa].asmCaps["HasWMMA"] and (state["WavefrontSize"] == 64)):
         print2("!! Warning: WMMA only well tune on WGP mode, wave size = 32")
      #  reject(state, printRejectionReason, "WMMA only suppport on WGP mode, wave size = 32")
      #  return
      if not state["MIBlock"] or len(state["MIBlock"]) != 6:
        reject(state, printRejectionReason, "invalid MIBlock")
        return
      if not state["MIWaveGroup"] or len(state["MIWaveGroup"]) != 2:
        reject(state, printRejectionReason, "invalid MIWaveGroup")
        return
      if not state["MIWaveTile"] or len(state["MIWaveTile"]) != 2:
        reject(state, printRejectionReason, "invalid MIWaveTile")
        return
      if state["UseSubtileImpl"] and (state["ProblemType"]["MXBlockA"] or state["ProblemType"]["MXBlockB"]):
        if state["MIWaveTile"][0] % 2 != 0 or state["MIWaveTile"][1] % 2 != 0:
          reject(state, printRejectionReason,
                 "UseSubtileImpl=1 with MX datatype requires even MIWaveTile, got [%d, %d]"
                 % (state["MIWaveTile"][0], state["MIWaveTile"][1]))
          return
      if isaInfoMap[isa].asmCaps["HasMFMA"]:
        if not state["ProblemType"]["HighPrecisionAccumulate"] \
           and state["ProblemType"]["DataType"].numRegisters() < 1 \
           and state["ProblemType"]["DataTypeA"].numRegisters() <= state["ProblemType"]["MacDataTypeA"].numRegisters() \
           and state["ProblemType"]["DataTypeB"].numRegisters() <= state["ProblemType"]["MacDataTypeB"].numRegisters():
          reject(state, printRejectionReason, "Matrix instructions for half, bf16 (or i8) types are natively accumulated" + \
           " in fp32 (or i32) precision. Please add the following config:" + \
           "\n - HighPrecisionAccumulate: True")
          return
      #FIXME: remove this section?
      if isaInfoMap[isa].asmCaps["HasWMMA"]:
        if state["ProblemType"]["DataType"].numRegisters() > 2:
          reject(state, printRejectionReason, "WMMA only support f32, half, bf16 and i8 type")
          return
      if state["ProblemType"]["DataType"].isDouble() and not (isaInfoMap[isa].asmCaps["HasMFMA_f64"] or isaInfoMap[isa].asmCaps["HasWMMA_V3_f64"]):
        reject(state, printRejectionReason, f"isa {isa} doesn't support matrix instruction with type f64")
        return
      if state["InterleaveAlpha"]:
        reject(state, printRejectionReason, "Matrix instruction doesn't support InterleaveAlpha")
        return
      if state["ProblemType"]["DataType"].isInt8():
        if isa[:2] == (9, 4):
          if tuple(state["MatrixInstruction"])[:3] in ((32, 32, 8), (16, 16, 16)):
            reject(state, printRejectionReason, "v_mfma_i32_32x32x8 and v_mfma_i32_16x16x16 have been deprecated in gfx94x")
            return
      if state["ProblemType"]["ComputeDataType"].isDouble():
        # See [4,4,4,4] snop for more info
        if state["MatrixInstruction"] == [4,4,4,4] and (not state['ISA'] == IsaVersion(9,0,10)) and state["ScheduleIterAlg"] == 3:
          reject(state, printRejectionReason, "Currently Matrix instructions [4,4,4,4] is disabled.")
          return
      if state["UseF32XEmulation"]:
        if not isaInfoMap[isa].archCaps["HasF32XEmulation"]:
          reject(state, printRejectionReason, "Missing emulation for F32X")
          return
        if state["ScheduleIterAlg"] not in (1, 3):
          reject(state, printRejectionReason, "F32X Emulation only supported with Schedule Iter Alg == (1 or 3)")
          return
        if tuple(state["MatrixInstruction"])[:3] in ((16, 16, 8), (16, 16, 16), (32, 32, 4)):
          reject(state, printRejectionReason, "tf32 emulation currently only supports mfma MI 16x16x32 and 32x32x16")
          return

    else:
      if not state["ProblemType"]["HighPrecisionAccumulate"] \
         and state["ProblemType"]["ComputeDataType"].numRegisters() > state["ProblemType"]["DataType"].numRegisters() :
        reject(state, printRejectionReason, "For non-MI Kernel, if sizeof(ComputeDataType) > sizeof(DataType), " + \
         "Please add the following config:" + \
         "\n - HighPrecisionAccumulate: True")
        return
      if state["ProblemType"]["Sparse"]:
        reject(state, printRejectionReason, "Sparse A problem is only supported by SMFMA MI kernel.")
        return

      if state["ThreadTile0"] > 16 or state["ThreadTile1"] > 16:
        reject(state, printRejectionReason, "Invalid value for ThreadTile")
        return

      if state["ScheduleIterAlg"] == 2 or state["ScheduleIterAlg"] == 3:
        reject(state, printRejectionReason, "SIA2 and SIA3 only support MatrixInstruction")
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
      reject(state, printRejectionReason, "Sparse B does not supprot DirectToVgprSparseMetadata")
      return


    if state["ProblemType"]["Sparse"]:
      state["LocalWriteUseSgprMetadata"] = False
      if state["ProblemType"]["Sparse"] == 2:
        if not state["DirectToVgprSparseMetadata"]:
          state["ThreadTileMetadata"] = state["ThreadTileB"]
          state["SubGroupMetadata"] = state["SubGroupB"]
          state["MacroTileMetadata"] = state["MacroTileB"]
          state["WaveSeparateGlobalReadMetadata"] = state["WaveSeparateGlobalReadB"]
          state["DirectToLdsMetadata"] = False
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
    # tentative init for UseGeneralizedNLCOneA/B
    # set True for DTL
    state["UseGeneralizedNLCOneA"] = state["DirectToLdsA"]
    state["UseGeneralizedNLCOneB"] = state["DirectToLdsB"]

    state["UseGeneralizedNLCOneMXSA"] = False
    state["UseGeneralizedNLCOneMXSB"] = False

    state["LocalWriteUseSgprA"] = False
    state["LocalWriteUseSgprB"] = False
    state["StoreSwapAddr"] = False

    if state["WorkGroupMappingXCC"] == -1:
      if state["StreamK"] == 0:
        reject(state, printRejectionReason, "Can only use auto WGMXCC with StreamK.")
        return False
      if state["StreamKXCCMapping"] != 0:
        reject(state, printRejectionReason, "Cannot use auto WGMXCC with SKXCC.")
        return False

    if state["WorkGroupMapping"] == 0:
      if state["WorkGroupMappingXCC"] == -1:
        if state["StreamK"] == 0:
          reject(state, printRejectionReason, "Can only use auto WGM with StreamK.")
          return False

    problemType = state["ProblemType"]

    for (tc,batchMask) in (('A', 0x1), ('B', 0x2)):
      freeDims = [i for i in problemType["IndexAssignments%s"%tc] if i in problemType["IndicesFree"]]
      if not freeDims:
        reject(state, printRejectionReason, "tensor%s contains no free indices.")
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
      state["DirectToLds"] = 0
      state["DirectToLdsA"] = False
      state["DirectToLdsB"] = False
      state["_UseSgprForGRO"] = False
      if state["PrefetchGlobalRead"] >= 2:
        reject(state, printRejectionReason, "BufferLoad=0 does not support PrefetchGlobalRead>=2")
        return

      if problemType["UseBias"]:
        reject(state, printRejectionReason, "BufferLoad=0 does not support UseBias due to no suppress no load.")
        return

    #These modes only work under certain conditions, apply them here:
    #  - The "NoLoad" loop is only generated if PrefetchGlobalRead>0
    #  - And Suppress does not work if GSU>1 for some reason
    if state["SuppressNoLoadLoop"]:
      if not (bufferLoad and state["PrefetchGlobalRead"] == 1 and (state["GlobalSplitU"]==1 or state["GlobalSplitU"]==-1)):
        state["SuppressNoLoadLoop"] = False

    #print("PackedC0IdxChars", state["PackedC0IdxChars"])
    #print("PackedC1IdxChars", state["PackedC1IdxChars"])

    #
    # UnrollMajorLDS{A,B}
    #   0: means M/N is contiguous in LDS
    #   1: means K is contiguous in LDS
    #
    # TLU{A,B}
    #   1: means M/N is contiguous in global memory
    #   0: means K is contiguous in global memory
    #
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

    def findSparseVectorWidth(steps, optVW):
      if steps == 0:
        return optVW
      optVW_new = max(optVW // 2, 1)
      if optVW_new == 1:
        return optVW
      else:
        return findSparseVectorWidth(steps-1, optVW_new)

    if state["VectorWidthA"] == -1:
      if state["EnableMatrixInstruction"]:
        regPerElem = state["ProblemType"]["MacDataTypeA"].numRegisters()
        optVW = int(4 // regPerElem)
        while 1:
          if state["MIWaveTile"][0] % optVW == 0:
            state["VectorWidthA"] = optVW
            break
          else:
            optVW //= 2
        if state["ProblemType"]["Sparse"]:
          # sparse gfx1250: Currently disable autoVectorWidthA. VectorWidthA still can be set manually in yaml file.
          # if isaInfoMap[isa].asmCaps["HasSWMMAC_gfx1250"] and state["ProblemType"]["Sparse"] == 1:
          #   state["VectorWidthA"] = max(findSparseVectorWidth(2, state["VectorWidthA"]), 1)
          # else:
          state["VectorWidthA"] = 1
      else:
        state["VectorWidthA"] = 1

    if state["VectorWidthB"] == -1:
      if state["EnableMatrixInstruction"]:
        regPerElem = state["ProblemType"]["MacDataTypeB"].numRegisters()
        optVW = int(4 // regPerElem)
        while 1:
          if state["MIWaveTile"][1] % optVW == 0:
            state["VectorWidthB"] = optVW
            break
          else:
            optVW //= 2
        if state["ProblemType"]["Sparse"]:
          # sparse gfx1250: Currently disable autoVectorWidthB. VectorWidthB still can be set manually in yaml file.
          # if isaInfoMap[isa].asmCaps["HasSWMMAC_gfx1250"] and state["ProblemType"]["Sparse"] == 2:
          #   state["VectorWidthB"] = max(findSparseVectorWidth(2, state["VectorWidthB"]), 1)
          # else:
          state["VectorWidthB"] = 1
      else:
        state["VectorWidthB"] = 1

    def isLDSTrEnabled(asmCaps: Dict, hasLDSTrans: bool, unrollMajorLDS: bool, dtv: bool, numBytes: int):
      if unrollMajorLDS:
        return False

      if not hasLDSTrans:
        return False

      if numBytes == 0.5:
        return asmCaps["HasLDSTrB64B4"]
      elif numBytes == 0.75:
        return asmCaps["HasLDSTrB96B6"]
      elif numBytes == 1:
        return asmCaps["HasLDSTrB64B8"]
      elif numBytes == 2:
        return asmCaps["HasLDSTrB64B16"] or asmCaps["HasLDSTrB128B16"]
      return False

    numBytesA = state["ProblemType"]["MacDataTypeA"].numBytes()
    numBytesB = state["ProblemType"]["MacDataTypeB"].numBytes()
    isa = tuple(state["ISA"])
    # state["enableLDSTrA"] = state["LDSTrInst"] and isaInfoMap[isa].asmCaps["HasLDSTr"] and numBytes == 2 \
    #         and not state["UnrollMajorLDSA"] and not state["DirectToVgprA"]
    # state["enableLDSTrB"] = state["LDSTrInst"] and isaInfoMap[isa].asmCaps["HasLDSTr"] and numBytes == 2 \
    #         and not state["UnrollMajorLDSB"] and not state["DirectToVgprB"]
    # TODO- Is it possible for devices with asmCaps["HasLDSTr"], we automatically use it when UnrollMajorLDS=0
    #       Supporting manually transpose load when having "HasLDSTr" is not worthy.
    state["enableLDSTrA"] = isLDSTrEnabled(isaInfoMap[isa].asmCaps, state["LDSTrInst"], state["UnrollMajorLDSA"], state["DirectToVgprA"], numBytesA)
    state["enableLDSTrMXSA"] = False
    state["enableLDSTrB"] = isLDSTrEnabled(isaInfoMap[isa].asmCaps, state["LDSTrInst"], state["UnrollMajorLDSB"], state["DirectToVgprB"], numBytesB)
    state["enableLDSTrMXSB"] = False

    # This reject kernels in 950 logic yaml, temporarily comment it out.
    # finalLDSTrInst = state["enableLDSTrA"] or state["enableLDSTrB"]
    # if state["LDSTrInst"] != finalLDSTrInst:
    #   # This means LDSTrInst=True, but none of A/B can be enabled (False)
    #   reject(state, printRejectionReason, "LDSTrInst is True but none of A/B can be enabled")
    #   return

    state["enableGLTrA"] = state["DirectToVgprA"] and state["ProblemType"]["TLUA"] \
      and ((numBytesA == 1 and isaInfoMap[isa].asmCaps["HasGLTr8B64"]) \
        or (numBytesA == 2 and isaInfoMap[isa].asmCaps["HasGLTr16B128"]) \
      )
    state["enableGLTrB"] = state["DirectToVgprB"] and state["ProblemType"]["TLUB"] \
      and ((numBytesB == 1 and isaInfoMap[isa].asmCaps["HasGLTr8B64"]) \
        or (numBytesB == 2 and isaInfoMap[isa].asmCaps["HasGLTr16B128"]) \
      )
	  
    if state["enableLDSTrA"] or state["enableGLTrA"]:
      state["VectorWidthA"] = 1

    if state["enableLDSTrB"] or state["enableGLTrB"]:
      state["VectorWidthB"] = 1

    if state["ScheduleIterAlg"] == 2:
      state["ExpandPointerSwap"] = True
      print2("\nSet SIA=2, force ExpandPointerSwap=1")

    if state["ExpandPointerSwap"]:
      # Pointer swap only used if PGR==1 or (PGR>1 and double/double complex) - so set ExpandPointerSwap=0 here
      # So far, EPS=1 and PGR>1 works only with double/double complex.
      #if not (bufferLoad and state["PrefetchGlobalRead"] == 1):
      if not (bufferLoad and ( state["PrefetchGlobalRead"] == 1 \
              or (state["PrefetchGlobalRead"] > 1 and \
                  (state["ProblemType"]["DataType"].isDouble() or state["ProblemType"]["DataType"].isDoubleComplex()))
              or (state["ProblemType"]["Sparse"] and state["PrefetchGlobalRead"] > 0))):
        state["ExpandPointerSwap"] = False

    #################################################################
    # ForceUnrollSubIter requirements
    # - Needs PGR > 0, double buffer
    # - MIWaveTile must be even and larger than 2
    # - TLU{A,B} cases only supported if using LdsTR or if VPerm not needed (size{A,B} >= 4)
    # - Not supported for mixed precision cases currently
    # - For non-CMS F32X kernels, SubIter is disabled after CMS check because
    #   the default scheduler's sub-tiling conflicts with F32X pack code.
    TLUA = state["ProblemType"]["TLUA"]
    TLUB = state["ProblemType"]["TLUB"]
    def _canEnableSubIter():
      sizeDataTypeA = state["ProblemType"]["DataTypeA"].numBytes()
      sizeDataTypeB = state["ProblemType"]["DataTypeB"].numBytes()
      sizeDataType = state["ProblemType"]["DataType"].numBytes()
      return (
        state["EnableMatrixInstruction"] and not state["ExpandPointerSwap"]
        and state["DepthU"] == state["MatrixInstK"] and state["PrefetchGlobalRead"] and not state["1LDSBuffer"]
        and (state["MIWaveTile"][0] > 2  and state["MIWaveTile"][1] > 2)
        and (state["MIWaveTile"][0] % 2 == 0 and state["MIWaveTile"][1] % 2 == 0)
        and (sizeDataTypeA == sizeDataType) and (sizeDataTypeB == sizeDataType)
        and ((TLUA == False or state["enableLDSTrA"] or sizeDataTypeA >= 4) and (TLUB == False or state["enableLDSTrB"] or sizeDataTypeB >= 4) )
        and (not state["ProblemType"]["Sparse"] or state["TransposeLDSMetadata"])
        and not state["TDMInst"]
      )

    def _applySubIterSetting(enable):
      if enable:
        state["ForceUnrollSubIter"] = True
        state["numSubTiles"] = 2
      else:
        state["ForceUnrollSubIter"] = False
        state["numSubTiles"] = 1

    _applySubIterSetting(_canEnableSubIter())
    if state["ForceUnrollSubIter"]:
      state["ClusterLocalRead"] = 1
      state["TailloopInNll"] = False

    if state["VectorWidthA"] == -1:
      if state["EnableMatrixInstruction"]:
        regPerElem = state["ProblemType"]["MacDataTypeA"].numRegisters()
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
    # VW adjustment for F32XEmu + TLU + SubIter
    # -> this logic does not work. Reject the kernel so far.
    if state["UseF32XEmulation"]:
      if TLUA and state["numSubTiles"] > 1 and state["MIWaveTile"][0] % (state["VectorWidthA"] * state["numSubTiles"]) != 0:
        # reject(state, printRejectionReason, "F32XEmulation + SubTiles requires MIWaveTile[0](%u) is multiple of VectorWidthA(%u) * 2"%(state["MIWaveTile"][0], state["VectorWidthA"]))
        # return
        # Many existing kernels are rejected with this condition.
        # So far, continue with VectorWidthA //=state["numSubTiles"]
        while state["MIWaveTile"][0] % (state["VectorWidthA"] * state["numSubTiles"]) != 0:
          state["VectorWidthA"] //= state["numSubTiles"]
        if state["SourceSwap"] and state["StoreVectorWidth"] > state["VectorWidthA"]:
          # need to adjust StoreVectorWidth in SourceSwap case
          state["StoreVectorWidth"] = state["VectorWidthA"]

    if state["VectorWidthB"] == -1:
      if state["EnableMatrixInstruction"]:
        regPerElem = state["ProblemType"]["MacDataTypeB"].numRegisters()
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
    # VW adjustment for F32XEmu + TLU + SubIter
    # -> this logic does not work. Reject the kernel so far.
    if state["UseF32XEmulation"]:
      if TLUB and state["numSubTiles"] > 1 and state["MIWaveTile"][1] % (state["VectorWidthB"] * state["numSubTiles"]) != 0:
        # reject(state, printRejectionReason, "F32XEmulation + SubTiles requires MIWaveTile[1](%u) is multiple of VectorWidthB(%u) * 2"%(state["MIWaveTile"][1],state["VectorWidthB"]))
        # return
        # Many existing kernels are rejected with this condition.
        # So far, continue with VectorWidthB //=state["numSubTiles"]
        while state["MIWaveTile"][1] % (state["VectorWidthB"] * state["numSubTiles"]) != 0:
          state["VectorWidthB"] //= state["numSubTiles"]

    if state["ProblemType"]["Sparse"]:
      if not state["DirectToVgprSparseMetadata"]:
        state["VectorWidthMetadata"] = state["VectorWidthA"] if state["ProblemType"]["Sparse"] == 1 else state["VectorWidthB"]
      # ON/OFF the sourceswap according to the sparse type automatically
      state["SourceSwap"] = False if state["ProblemType"]["Sparse"] == 1 else True

    # The real value of "1LDSBuffer" will be determined later (when it is -1), not here

    # if state["EnableMatrixInstruction"] and not state["SourceSwap"] and (state["VectorWidthA"] > 1 or state["VectorWidthB"] > 1):
    #   reject(state, printRejectionReason, "not implement VectorWidth without SourceSwap")

    # TT0,1 both must be multiples of VW, b/c of rC, rA, rB
    if state["EnableMatrixInstruction"]:
      if (state["MIWaveTile"][0] % state["VectorWidthA"]) != 0:
        reject(state, printRejectionReason, "MIWaveTile0(%u) should be multiple of VectorWidthA(%u)" % (state["MIWaveTile"][0], state["VectorWidthA"]))
        return
      if (state["MIWaveTile"][1] % state["VectorWidthB"]) != 0:
        reject(state, printRejectionReason, "MIWaveTile0(%u) should be multiple of VectorWidthB(%u)" % (state["MIWaveTile"][1], state["VectorWidthB"]))
        return

    if len(problemType["IndicesSummation"]) > 1:
      # not supported with multiple summations, bug is maybe something with
      # how stagger iteration is wrapped when unroll loop exits
      state["StaggerU"] = 0

    if state["ProblemType"]["MXBlockA"]:
      state["DirectToVgprMXSA"] = state["DirectToVgprA"]
      state["ThreadTileMXSA"] = state["ThreadTileA"]
      state["SubGroupMXSA"] = state["SubGroupA"]
      state["MacroTileMXSA"] = state["MacroTileA"]
      state["WaveSeparateGlobalReadMXSA"] = state["WaveSeparateGlobalReadA"]
      state["NumLoadsCoalescedMXSA"] = state["NumLoadsCoalescedA"]
      Solution.checkAndAssignWaveSeparateGlobalRead(state, 'MXSA', printRejectionReason)
      state["DirectToLdsMXSA"] = state["DirectToLdsA"]
      state["LocalWriteUseSgprMXSA"] = state["DirectToLdsMXSA"]
      state["ProblemType"]["MirrorDimsMXSA"] = list(state["ProblemType"]["MirrorDimsA"])
      state["VectorWidthMXSA"] = state["VectorWidthA"]
      state["MIWaveTileMXSA"] = state["MIWaveTileA"]
      state["UnrollMajorLDSMXSA"] = state["UnrollMajorLDSA"]

    if state["ProblemType"]["MXBlockB"]:
      state["DirectToVgprMXSB"] = state["DirectToVgprB"]
      state["ThreadTileMXSB"] = state["ThreadTileB"]
      state["SubGroupMXSB"] = state["SubGroupB"]
      state["MacroTileMXSB"] = state["MacroTileB"]
      state["WaveSeparateGlobalReadMXSB"] = state["WaveSeparateGlobalReadB"]
      state["NumLoadsCoalescedMXSB"] = state["NumLoadsCoalescedB"]
      Solution.checkAndAssignWaveSeparateGlobalRead(state, 'MXSB', printRejectionReason)
      state["DirectToLdsMXSB"] = state["DirectToLdsB"]
      state["LocalWriteUseSgprMXSB"] = state["DirectToLdsMXSB"]
      state["ProblemType"]["MirrorDimsMXSB"]  = list(state["ProblemType"]["MirrorDimsB"])
      state["VectorWidthMXSB"] = state["VectorWidthB"]
      state["MIWaveTileMXSB"] = state["MIWaveTileB"]
      state["UnrollMajorLDSMXSB"] = state["UnrollMajorLDSB"]

    if state["ProblemType"]["MXBlockA"] or state["ProblemType"]["MXBlockB"]:
      if state["ProblemType"]["MXBlockA"]:
        state["LocalReadVectorWidthMXS"] = state["MIInputPerThreadMXSA"]
      elif state["ProblemType"]["MXBlockB"]:
        state["LocalReadVectorWidthMXS"] = state["MIInputPerThreadMXSB"]

    # Some restrictions for half:
    if state["KernelLanguage"] == "Assembly" \
      and state["ProblemType"]["DataType"].isHalf():

      if isaInfoMap[state["ISA"]].archCaps["HasEccHalf"]:
        if not state["ProblemType"]["HighPrecisionAccumulate"] and state["AssertFree0ElementMultiple"] % 2 != 0:
          # beta-on-edge has AF0EM requirement except for HPA kernels
          reject(state, printRejectionReason, "Archs with HasEccHalf require AF0EM%2==0 except for HPA kernels")
          return

    if state["ConvertAfterDS"]:
      if (state["ProblemType"]["DataType"].isHalf() == False) and (state["ProblemType"]["DataType"].isBFloat16() == False):
          reject(state, printRejectionReason, "ConvertAfterDS only support DataType half")
          return
      if (state["ProblemType"]["DataTypeA"].isAnyFloat8() == False) and (state["ProblemType"]["DataTypeB"].isAnyFloat8() == False) \
          and not (state["ProblemType"]["DataTypeA"].isSingle() and state["ProblemType"]["DataTypeB"].isSingle()):
          reject(state, printRejectionReason, "one of DataTypeA or DataTypeB need to be float8/float8_fnuz or both are fp32")
          return
      if state["ProblemType"]["DataType"].isBFloat16() \
          and (state["ProblemType"]["DataTypeA"].isSingle() and state["ProblemType"]["DataTypeB"].isSingle()):
          reject(state, printRejectionReason, "ConvertAfterDS doesn't support SS_BSS type")
          return

    # Complex datatype restrictions.
    if state["ProblemType"]["DataType"].isComplex():
      if state["MIArchVgpr"] and state["StreamK"] != 0:
        reject(state, printRejectionReason, "Complex datatype kernel does not support StreamK with MIArchVgpr yet.")
        return


    if state["TDMInst"]:
      if not isaInfoMap[isa].asmCaps["HasTDM"]:
        reject(state, printRejectionReason, "This arch does not support TDM")
        return

    tdmInst: int = state["TDMInst"]
    state["enableTDMA"] = bool(tdmInst & 0x01)
    state["enableTDMB"] = bool(tdmInst & 0x02)

    if tdmInst not in (0, 3):
      reject(state, printRejectionReason, "Currently TDMA and TDMB must be enabled simultaneously")
      return

    # DepthU == -1?
    if state["DepthU"] == -1:
      depthuList = [1024,512,256,128,64,32,16]
    else:
      depthuList = [state["DepthU"]]
    index = [0]
    backupValues = []
    for key, value in state.items():
      if isinstance(value, int) and value < 0:
        backupValues.append([key, value])
    # Skip this check for subtile impl?
    # TODO: Add this check back
    while True:
      for backup in backupValues:
        state[backup[0]] = backup[1]
      state["ValidDepthU"] = True
      state["DepthU"]      = depthuList[index[0]]
      Solution.depthUIteration(
        state,
        index,
        depthuList,
        problemType,
        isa,
        bufferLoad,
        packedC0,
        packedC1,
        printRejectionReason,
        isaInfoMap,
        rocmVersion
      )
      if state["Valid"] or (state["ValidDepthU"] and (not state["Valid"])):
        break
      index[0] += 1
      if index[0] >= len(depthuList):
        break
    if "ValidDepthU" in state:
      del state["ValidDepthU"]

    if state["UseDirect32XEmulation"] == True:
      #   Turn off Direct32X for the following kernels:
      #   Cijk_Ailk_Bjlk_S_MX_B_Bias_HA_S_SAV_UserArgs_MT16x16x512_MI16x16x1
      if (state["MacroTile0"] == 16 and state["MacroTile1"] == 16 and state["DepthU"] == 512):
        state["UseDirect32XEmulation"] = False

    # backup UsePLRPack from yaml before calling hasCustomSchedule
    backup_UsePLRPack = state["UsePLRPack"]
    # Check if CMS is available for this solution
    if state["UseCustomMainLoopSchedule"] in [-1, 1]:
      # initialize CMS related config parameters (for CMS only)
      state["SwapGlobalReadOrder"] = 0
      state["UsePLRPack"] = 0

      hasCMS,_ = hasCustomSchedule(state)
      if state["UseCustomMainLoopSchedule"] == 1 and not hasCMS:
        reject(state, printRejectionReason, "UseCustomMainLoopSchedule=1 but CMS is not supported")
      state["UseCustomMainLoopSchedule"] = 1 if hasCMS else 0
      # reject CMS + TailloopInNll
      if state["TailloopInNll"] and state["UseCustomMainLoopSchedule"] == 1:
        reject(state, printRejectionReason, "UseCustomMainLoopSchedule=1 is incompatible with TailloopInNll=True")
        return
    # UseSubtileImpl has its own main loop scheduler; CMS is not compatible.
    if state["UseSubtileImpl"] and state["UseCustomMainLoopSchedule"] == 1:
        reject(state, printRejectionReason, "UseCustomMainLoopSchedule=1 is incompatible with UseSubtileImpl")

    # additional setting for non CMS
    if state["UseCustomMainLoopSchedule"] == 0:
      # Non-CMS, non-MFMA F32X (e.g. gfx1250 WMMA): disable SubIter because
      # the default scheduler's sub-tiling conflicts with F32X pack code's
      # destructive in-place VGPR conversion.
      # MFMA-based XF32 (gfx950) and CMS kernels are exempt.
      if state["UseF32XEmulation"] and not state["UseMFMAF32XEmulation"] and state["ForceUnrollSubIter"]:
        _applySubIterSetting(False)
      if state["UseMFMAF32XEmulation"]:
        state["MfmaInitCVgprs"] = True
      # usePLRPack check
      # adjust setting only for non CMS (keep original setting for CMS)
      if backup_UsePLRPack:
        state["UsePLRPack"] = 1
        # MatrixInstruction only
        if not state["EnableMatrixInstruction"]:
          state["UsePLRPack"] = 0
        # F32X emulation only
        if not state["UseF32XEmulation"]:
          state["UsePLRPack"] = 0
        # SIA3 only
        if state["ScheduleIterAlg"] != 3:
          state["UsePLRPack"] = 0
        # enable UsePLRPack for SubIter only
        if not state["ForceUnrollSubIter"]:
          state["UsePLRPack"] = 0
        # DirectToLds (both A and B) only
        if state["DirectToLds"] != 1:
          state["UsePLRPack"] = 0
        # PGR and PLR should be non 0
        if state["PrefetchGlobalRead"] == 0 or state["PrefetchLocalRead"] == 0:
          state["UsePLRPack"] = 0

    # disable SwapGlobalReadOrder if grmode(normal/DTL/DTV) is different between A and B
    # GRA and GRB need to be equivalent to swap the order
    # keep original setting in CMS case
    def getGrMode(tc):
      grmode = 0
      if state["DirectToLds%s"%tc]:
        grmode = 1
      elif state["DirectToVgpr%s"%tc]:
        grmode = 2
      return grmode
    if state["UseCustomMainLoopSchedule"] == 0:
      if getGrMode("A") != getGrMode("B"):
        state["SwapGlobalReadOrder"] = 0
      # SwapGlobalReadOrder does not work with UnrollLoopSwapGlobalReadOrder
      if state["UnrollLoopSwapGlobalReadOrder"]:
        state["SwapGlobalReadOrder"] = 0

    # 0: Normal mode. Hardware applies all of the normal data dependency checks
    # 1: Full expert mode (not suppoeted yet). Disable hardware checks against: VA_VDST, VA_SDST, VA_SSRC, VA_VCC, VM_VSRC and SA_SDST.
    # 2: Disable only VA_VDST and VM_VSRC checks.
    def evaluateExpertSchedulingMode() -> Literal[0, 1, 2]:
      # Check if the current parameters is supported by the ExpertSchedulingMode
      if not isaInfoMap[isa].archCaps["HasSchedMode"]: return 0
      if state["ProblemType"]["Sparse"]: return 0
      if state["ProblemType"]["DataType"].isSingle(): return 0

      # parameters not tested yet:
      supportedParameters = {
        "EnableMatrixInstruction": state["EnableMatrixInstruction"],
        "ScheduleIterAlg": state["ScheduleIterAlg"] == 3,
        "WavefrontSize": state["WavefrontSize"] == 32,
        "UnrollLoopSwapGlobalReadOrder": not state["UnrollLoopSwapGlobalReadOrder"],
        "SuppressNoLoadLoop": not state["SuppressNoLoadLoop"],
        "ScheduleGlobalRead": state["ScheduleGlobalRead"] == 1,
        "ScheduleLocalWrite": state["ScheduleLocalWrite"] == 1,
        "GlobalReadPerMfma": state["GlobalReadPerMfma"] == 1,
        "InterleaveAlpha": not state["InterleaveAlpha"],
        "DirectToLds": not state["DirectToLds"],
        "UseSgprForGRO": state["UseSgprForGRO"] == -1,
        "UseInstOffsetForGRO": not state["UseInstOffsetForGRO"],
        "Use64bShadowLimit": state["Use64bShadowLimit"] == 1,
        "StorePriorityOpt": not state["StorePriorityOpt"],
        "StoreSyncOpt": not state["StoreSyncOpt"],
        "GroupLoadStore": not state["GroupLoadStore"],
        "StreamK": not state["StreamK"],
        "StreamKAtomic": not state["StreamKAtomic"],
        "StreamKXCCMapping": not state["StreamKXCCMapping"],
        "StreamKFixupTreeReduction": not state["StreamKFixupTreeReduction"],
        "DebugStreamK": not state["DebugStreamK"],
        "WorkGroupReduction": not state["WorkGroupReduction"],
        "ConvertAfterDS": not state["ConvertAfterDS"],
        "ForceDisableShadowInit": not state["ForceDisableShadowInit"],
      }

      for key, supported in supportedParameters.items():
        if supported:
          continue

        print2(f"ExpertSchedulingMode not supported with {key}={state[key]}")
        return 0

      # Currently, only the mode that disables VA_VDST and VM_VSRC checks is supported.
      return 2

    state["ExpertSchedulingMode"] = evaluateExpertSchedulingMode()

    # Some restrictions for float4 and 6bitFloat:
    # TODO: remove this if edge and tail are supported for fp4/fp6/bf6
    isFloat4 = state["ProblemType"]["DataTypeA"].isFloat4() or state["ProblemType"]["DataTypeB"].isFloat4()
    isFloat6 = state["ProblemType"]["DataTypeA"].is6bitFloat() or state["ProblemType"]["DataTypeB"].is6bitFloat()
    if isa[:2] == (12, 5) and state["KernelLanguage"] == "Assembly" and (isFloat4 or isFloat6):
      if state["ProblemType"]["MacDataTypeA"].isFloat4() or state["ProblemType"]["MacDataTypeB"].isFloat4():
        if not state["enableLDSTrA"] and not state["UnrollMajorLDSA"]:
          reject(state, printRejectionReason, "Currently FP4 requires LDSTrInst == True for UnrolledMajorLDSA == False")
          return

        if not state["enableLDSTrB"] and not state["UnrollMajorLDSB"]:
          reject(state, printRejectionReason, "Currently FP4 requires LDSTrInst == True for UnrolledMajorLDSB == False")
          return

        # Currently we only support fp4 edge with AssertFree0(1)ElementMultiple = 2.
        # TODO: Enalbe edge with arbitrary number
        state["AssertFree0ElementMultiple"] = 2
        state["AssertFree1ElementMultiple"] = 2

        # Currently we only support fp4 tail-loop with AssertSummationElementMultiple = 8.
        # TODO: Enalbe tail-loop with arbitrary number
        state["AssertSummationElementMultiple"] = 8

      if state["ProblemType"]["MacDataTypeA"].is6bitFloat() or state["ProblemType"]["MacDataTypeB"].is6bitFloat():
        state["AssertFree0ElementMultiple"] = 1
        state["AssertFree1ElementMultiple"] = 1
        state["AssertSummationElementMultiple"] = 16
        if not state["ProblemType"]["TransposeA"]:
          state["AssertFree0ElementMultiple"] = 16
        if state["ProblemType"]["TransposeB"]:
          state["AssertFree1ElementMultiple"] = 16

    state["AssertSummationElementMultiple"] = max(state["ProblemType"]["MXBlockA"], state["AssertSummationElementMultiple"])
    state["AssertSummationElementMultiple"] = max(state["ProblemType"]["MXBlockB"], state["AssertSummationElementMultiple"])

    # We have the real "1LDSBuffer" value now, so we have to test the rejection condition here
    # TODO-
    #  On gfx1250, i8, f8, it seem working for 1LDSBuffer=0 "BUT EPS=0", haven't checked for other archs/types, so we still reject by 1LDSBuffer only
    # if (state["enableLDSTrA"] or state["enableLDSTrB"]) and (not state["1LDSBuffer"] and state["ExpandPointerSwap"]):
    if (state["enableLDSTrA"] or state["enableLDSTrB"]) and (not state["1LDSBuffer"] and state["ExpandPointerSwap"]):
      reject(state, printRejectionReason, "Current LDSTrInst implementation does not support 1LDSBuffer=0")
      return

  @staticmethod
  def depthUIteration(
      state,
      index,
      depthuList,
      problemType,
      isa,
      bufferLoad,
      packedC0,
      packedC1,
      printRejectionReason: bool,
      isaInfoMap: Dict[IsaVersion, IsaInfo],
      rocmVersion: SemanticVersion
    ):
    ########################################
    # Auto search for DepthU starts here
    # Activates when DepthU == -1
    ########################################
    resetStaggerUStride = state["StaggerUStride"]
    resetLocalReadVectorWidth = state["LocalReadVectorWidth"]
    resetLocalReadVectorWidthA = state["LocalReadVectorWidthA"]
    resetLocalReadVectorWidthB = state["LocalReadVectorWidthB"]
    resetGlobalReadVectorWidthA = state["GlobalReadVectorWidthA"]
    resetGlobalReadVectorWidthB = state["GlobalReadVectorWidthB"]

    while True:
      userDepthU = depthuList[index[0]]
      state["StaggerUStride"] = resetStaggerUStride
      state["LocalReadVectorWidth"] = resetLocalReadVectorWidth
      state["LocalReadVectorWidthA"] = resetLocalReadVectorWidthA
      state["LocalReadVectorWidthB"] = resetLocalReadVectorWidthB
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
      if state["ProblemType"]["MXBlockA"]:
        state["_DepthUMXSA"] = depthUA // state["ProblemType"]["MXBlockA"]
      state["_DepthUB"] = depthUB# internal
      if state["ProblemType"]["MXBlockB"]:
        state["_DepthUMXSB"] = depthUB // state["ProblemType"]["MXBlockB"]
      state["_DepthUMetadata"] = depthUM# internal

      Solution.checkAndAssignWaveSeparateGlobalRead(state, 'A', printRejectionReason)
      Solution.checkAndAssignWaveSeparateGlobalRead(state, 'B', printRejectionReason)
      if state["ProblemType"]["Sparse"]:
        if state["ProblemType"]["Sparse"] == 2:
          if not state["DirectToVgprSparseMetadata"]:
            Solution.checkAndAssignWaveSeparateGlobalRead(state, 'Metadata', printRejectionReason)
        else:
          if not state["DirectToVgprSparseMetadata"]:
            Solution.checkAndAssignWaveSeparateGlobalRead(state, 'Metadata', printRejectionReason)

      # Set up stagger shift:
      bpeAB = state["ProblemType"]["DataType"].numBytes()
      # (1<<staggerStrideShift) is number of loop iterations to traverse the stride
      if state["StaggerU"] == 0:
        state["StaggerUMapping"] = 0
        state["StaggerUStride"] = 0

      if state["StaggerUStride"] == -1 or state["StaggerUStride"] < (state["DepthU"] * bpeAB):
        # (StaggerUStride) shoud be greater than or equal to (DepthU * bpeAB)
        state["StaggerUStride"] = state["DepthU"] * bpeAB

      state["_staggerStrideShift"] = (int)(math.ceil(math.log(state["StaggerUStride"] / (state["DepthU"] * bpeAB), 2)))

      def calcLdsPad(isaInfoMap: Dict[str, IsaInfo]) -> int:
        isMX = state["ProblemType"].get("MXBlockA", 0) != 0 or state["ProblemType"].get("MXBlockB", 0) != 0
        numBytesA = state["ProblemType"]["MacDataTypeA"].numBytes()
        numBytesB = state["ProblemType"]["MacDataTypeB"].numBytes()
        lrvwA = state["LocalReadVectorWidthA"]
        lrvwB = state["LocalReadVectorWidthB"]
        ldsPadA = state["LdsPadA"]
        ldsPadB = state["LdsPadB"]
        ldsPadM = state["LdsPadMetadata"]
        optPadA = lrvwA
        optPadB = lrvwB
        readRegsA = int(lrvwA * state["ProblemType"]["MacDataTypeA"].numBytes() // 4)
        readRegsB = int(lrvwB * state["ProblemType"]["MacDataTypeB"].numBytes() // 4)
        if state["ProblemType"]["Sparse"]:
          if state["ProblemType"]["Sparse"] == 2:
            optPadB //= 2
            readRegsB //= 2
          else:
            optPadA //= 2
            readRegsA //= 2
        if (not isaInfoMap[isa].asmCaps['HasWMMA']) and (readRegsA > 6 or readRegsB > 6):
          reject(state, "LocalReadVectorWidth results in attemping to read LDS larger than b192, reject")
          return ldsPadA, ldsPadB, ldsPadM, 0, 0
        # SubtileImpl does not need LDS padding.
        if state["UseSubtileImpl"]:
          return 0, 0, 0, 0, 0
        if state["EnableMatrixInstruction"]:
          # for readRegs = 1 or 4, we need to double pad for MI16x16xNx1 to avoid bank conflict.
          if state["MatrixInstB"] == 1 and state["MatrixInstM"] == 16:
            if readRegsA == 4 or readRegsA == 1:
              optPadA *= 2
            if readRegsB == 4 or readRegsB == 1:
              optPadB *= 2
        if ldsPadA == -1:
          if isMX and (state["ProblemType"]["DataTypeA"].is6bitFloat() or state["ProblemType"]["DataTypeA"].isFloat4()):
            ldsPadA = 0
          else:
            if not state["UnrollMajorLDSA"]:
              if state["EnableMatrixInstruction"]:
                ldsPadA = 0
                if state["MatrixInstB"] == 1 and state["MatrixInstM"] == 16:
                  ldsPadA = int(((16 * state["VectorWidthA"] * numBytesA + state["MacroTile0"] * numBytesA * lrvwA) % 128) // numBytesA)
                if state["GlobalReadVectorWidthA"] * numBytesA == 32 and ldsPadA == 0:
                  ldsPadA = int(16 // numBytesA)
                if state["DirectToLdsA"]:
                  # TODO: Check if there are cases which benefit from padding, currently set to zero by default
                  ldsPadA = state["MatrixInstM"] if state["enableLDSTrA"] else 0
              else: # mac instruction
                if state["ProblemType"]["TLUA"]:
                  ldsPadA = 0
                else:
                  ldsPadA = state["VectorWidthA"]
            else:
              if state["DirectToLdsA"]:
                if not state["ProblemType"]["TLUA"]:
                  bpeA = state["ProblemType"]["DataTypeA"].numBytes()
                  LdsStride = state["VectorWidthA"] * bpeA * state["DepthU"]
                  MinLdsBlockSizePerPadA = (state[f"GlobalReadVectorWidthA"] * bpeA) * state["WavefrontSize"]
                  isM0PadEnough = LdsStride >= MinLdsBlockSizePerPadA
                  ldsPadA = state["MatrixInstK"] if bpeA == 2 and not isM0PadEnough else 2 * lrvwA
                else:
                  ldsPadA = 0
              else:
                ldsPadA = max(state["GlobalReadVectorWidthA"],optPadA)
          assert(ldsPadA >= 0)

        if ldsPadB == -1:
          if isMX and (state["ProblemType"]["DataTypeB"].is6bitFloat() or state["ProblemType"]["DataTypeB"].isFloat4()):
            ldsPadB = 0
          else:
            if not state["UnrollMajorLDSB"]:
              if state["EnableMatrixInstruction"]:
                ldsPadB = 0
                if state["MatrixInstB"] == 1 and state["MatrixInstM"] == 16:
                  ldsPadB = int(((16 * state["VectorWidthB"] * numBytesB + state["MacroTile1"] * numBytesB * lrvwB) % 128) // numBytesB)
                if state["GlobalReadVectorWidthB"] * numBytesB == 32 and ldsPadB == 0:
                  ldsPadB = int(16 // numBytesB)
                if state["DirectToLdsB"]:
                  # TODO: Check if there are cases which benefit from padding, currently set to zero by default
                  ldsPadB = state["MatrixInstM"] if state["enableLDSTrB"] else 0
              else: # mac instruction
                if state["ProblemType"]["TLUB"]:
                  ldsPadB = 0
                else:
                  ldsPadB = state["VectorWidthB"]
            else:
              if state["DirectToLdsB"]:
                if not state["ProblemType"]["TLUB"]:
                  bpeB = state["ProblemType"]["DataTypeB"].numBytes()
                  LdsStride = state["VectorWidthB"] * bpeB * state["DepthU"]
                  MinLdsBlockSizePerPadB = (state[f"GlobalReadVectorWidthB"] * bpeB) * state["WavefrontSize"]
                  isM0PadEnough = LdsStride >= MinLdsBlockSizePerPadB
                  ldsPadB = state["MatrixInstK"] if bpeB == 2 and not isM0PadEnough else 2 * lrvwB
                else:
                  ldsPadB = 0
              else:
                ldsPadB = max(state["GlobalReadVectorWidthB"],optPadB)
          assert(ldsPadB >= 0)

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

        def removeLdsPadLogicForDTL(tc, ldsPad):
          ret = ldsPad
          miwt = state["MIWaveTile%s"%tc]
          # If TLU = 1 and not using LDSTR, lds read is contiguous so no padding needed
          if state["ProblemType"]["TLU%s"%tc] and (not state["enableLDSTr%s"%tc]):
            ret = 0
          if state["ProblemType"]["TLU%s"%tc] and state["enableLDSTr%s"%tc] \
             and (miwt & (miwt-1)) != 0 and state["UseGeneralizedNLCOne%s"%tc]:
            ret = 0
          return ret

        if state["DirectToLdsA"]:
          ldsPadA = removeLdsPadLogicForDTL('A', ldsPadA)
        if state["DirectToLdsB"]:
          ldsPadB = removeLdsPadLogicForDTL('B', ldsPadB)
        # set ldsPadA,B=0 for DirectToVgpr
        if state["DirectToVgprA"]:
          ldsPadA = 0
        if state["DirectToVgprB"]:
          ldsPadB = 0

        ldsPadMXSA = 4 * 2 if (state["LdsPadMXSA"] == -1) else state["LdsPadMXSA"]
        ldsPadMXSB = 4 * 2 if (state["LdsPadMXSB"] == -1) else state["LdsPadMXSB"]

        if state["TDMInst"]:
          pads = {"A": ldsPadA * state["ProblemType"]["MacDataTypeA"].numBytes(), "B": ldsPadB * state["ProblemType"]["MacDataTypeB"].numBytes(), "MXSA": ldsPadMXSA, "MXSB": ldsPadMXSB}
          for tc, val in pads.items():
            if val == 0: continue
            if TensorDataMoverLoad.calPadAmount(val) > 127:
              reject(state, printRejectionReason, f"pad_amount=(ldsPad//4-1)={pad_amount} should be smaller than or equal to 127 for ldsPad{tc}={val}")

        return ldsPadA, ldsPadB, ldsPadM, ldsPadMXSA, ldsPadMXSB

      def checkLdsBlockSizePerPadForTDM(ldsBlockSizePerPadA: int, ldsBlockSizePerPadB: int, ldsBlockSizePerPadMXSA: int, ldsBlockSizePerPadMXSB: int):
        if state["TDMInst"]:
          pads = {"A": ldsBlockSizePerPadA, "B": ldsBlockSizePerPadB, "MXSA": ldsBlockSizePerPadMXSA, "MXSB": ldsBlockSizePerPadMXSB}
          for tc, val in pads.items():
            if val == 0: continue
            pad_interval = TensorDataMoverLoad.calPadInterval(val)
            if pad_interval > 7:
              reject(state, printRejectionReason, f"pad_interval=(log2(LdsBlockSizePerPad//4)-1)={pad_interval} should be smaller than or equal to 7 for ldsBlockSizePerPad{tc}={val}")

      def calcMXSLdsBlockSizePerPad(tc: str, lrvw: int) -> int:
        LdsBlockSizePerPad = state["LdsBlockSizePerPad%s"%tc]
        bpe = 1 # MX scale size
        multiple = 256 if isa[:2] == (12, 5) else 128
        if LdsBlockSizePerPad == -1:
          vw = state["VectorWidthA"] if "A" in tc else state["VectorWidthB"]
          if state["ISA"] == (12, 5, 0):
            LdsBlockSizePerPad = 0
          else:
            LdsBlockSizePerPad = roundUpToNearestMultiple(int(state["_DepthU%s"%tc] * bpe * vw), multiple)
        return LdsBlockSizePerPad

      def getLdsBpe(tc: str) -> float:
        return state["ProblemType"]["DataType%s"%tc].numBytes() if state["ConvertAfterDS"] else state["ProblemType"]["MacDataType%s"%tc].numBytes()

      def calcLdsBlockSizePerPad(tc: str, lrvw: int) -> int:
        if "MXS" in tc:
          return calcMXSLdsBlockSizePerPad(tc, lrvw)
        mt = state["MacroTile0"] if ("A" in tc) else state["MacroTile1"]
        LdsBlockSizePerPad = state["LdsBlockSizePerPad%s"%tc]
        tmpBpe = getLdsBpe(tc)
        multiple = 256 if isa[:2] == (12, 5) else 128
        if LdsBlockSizePerPad == -1:
          if state["EnableMatrixInstruction"] and tmpBpe != 0.75:
            if state["UnrollMajorLDS%s"%tc]:
              LdsBlockSizePerPad = roundUpToNearestMultiple(int(state["_DepthU%s"%tc] * tmpBpe), multiple)
              if state["_DepthU%s"%tc] * tmpBpe * state["VectorWidth%s"%tc] > multiple:
                LdsBlockSizePerPad = roundUpToNearestMultiple(int(state["_DepthU%s"%tc] * tmpBpe * state["VectorWidth%s"%tc]), multiple)
            else:
              if state["MatrixInstB"] == 1 and state["MatrixInstM"] == 16:
                LdsBlockSizePerPad = int(mt * tmpBpe * lrvw)
              else:
                LdsBlockSizePerPad = 0
          else:
            LdsBlockSizePerPad = 0

        # set LdsBlockSizePerPad,B=0 for DirectToVgpr
        if state["DirectToVgpr%s"%tc]:
          LdsBlockSizePerPad = 0

        if state["DirectToLds%s"%tc]:
          bpeA = state["ProblemType"]["DataType%s"%tc].numBytes()
          # For DTL lds padding must be a multiple of the instruction load size (in bytes)
          # Also, LBSPP must be less than or equal to GRVW*bpe*NumThreads (cannot handle LBSPP larger than m0 inc value)
          MinLdsBlockSizePerPad = int((state[f"GlobalReadVectorWidth%s"%tc] * bpeA) * state["WavefrontSize"])
          MaxLdsBlockSizePerPad = int((state[f"GlobalReadVectorWidth%s"%tc] * bpeA) * state["NumThreads"])
          if state["UseGeneralizedNLCOne%s"%tc]:
            LdsBlockSizePerPad = MinLdsBlockSizePerPad
          else:
            LdsBlockSizePerPad = max(LdsBlockSizePerPad, MinLdsBlockSizePerPad)
            LdsBlockSizePerPad = min(LdsBlockSizePerPad, MaxLdsBlockSizePerPad)
            LdsBlockSizePerPad = roundUpToNearestMultiple(LdsBlockSizePerPad, MinLdsBlockSizePerPad)

        return int(LdsBlockSizePerPad)


      def calcLdsNumBytesAB(mxTc: str, ldsPad: int, LdsBlockSizePerPad: int):
        tc = mxTc.replace("MXS", "")
        if ("MXS" in mxTc) and (state["ProblemType"]["MXBlock%s"%tc] == 0):
          return (0, 0)
        if state["DirectToVgpr%s"%mxTc]:
          return (0, 0)

        bpe = state["ProblemType"]["DataType%s"%tc].numBytes() if state["ConvertAfterDS"] else state["ProblemType"]["MacDataType%s"%tc].numBytes()
        bpe = 1 if ("MXS" in mxTc) else bpe
        ldsAlign = int(64 / state["ProblemType"]["MacDataType%s"%tc].numRegisters())
        ldsAlign = 64 if state["ProblemType"]["MacDataType%s"%tc].is6bitFloat() else ldsAlign

        if state["UnrollMajorLDS%s"%mxTc]:
          ldsNumBytes = int((state["_DepthU%s"%mxTc] + ldsPad) * state["MacroTile%s"%mxTc] * bpe)
        else:
          ldsNumBytes = int(state["_DepthU%s"%mxTc] * (state["MacroTile%s"%mxTc] + ldsPad) * bpe)
        padInterval = LdsBlockSizePerPad

        if padInterval != 0:
          ldsNumBytes = int((state["_DepthU%s"%mxTc] * state["MacroTile%s"%mxTc] * bpe) / padInterval * (padInterval + ldsPad * bpe))
        ldsNumBytesAligned = roundUpToNearestMultiple(ldsNumBytes, ldsAlign)

        if state["DirectToVgpr%s"%mxTc]:
          ldsNumBytes = 0
          ldsNumBytesAligned = 0

        return (ldsNumBytes, ldsNumBytesAligned)


      def calcLdsNumBytesM():
        if state["ProblemType"]["Sparse"] and not state["DirectToVgprSparseMetadata"]:
          ldsAlign = int(64 / state["ProblemType"]["DataType"].numRegisters())
          ldsAlign = 64 if state["ProblemType"]["DataType"].is6bitFloat() else ldsAlign
          bpeAB = state["ProblemType"]["DataType"].numBytes()
          if state["UnrollMajorLDSMetadata"]:
            ldsNumBytesMetadata = (state["_DepthUMetadata"] + state["LdsPadMetadata"]) * state["MacroTileMetadata"]
          else:
            ldsNumBytesMetadata = state["_DepthUMetadata"] * (state["MacroTileMetadata"] + state["LdsPadMetadata"])
          ldsNumBytesMetadata = roundUp(ldsNumBytesMetadata / bpeAB) # metadata is in byte type. so divide ldsNumBytesMetadata by A,B's bpe
          padInterval = state["LdsBlockSizePerPadMetadata"]
          if padInterval != 0:
            ldsNumBytesMetadata = int(roundUp(state["_DepthUMetadata"] * state["MacroTileMetadata"] / bpeAB) / padInterval * (padInterval + state["LdsPadMetadata"]))
          ldsNumBytesAlignedMetadata = int(roundUpToNearestMultiple(ldsNumBytesMetadata, ldsAlign) * bpeAB)
          ldsNumBytesMetadata = int(ldsNumBytesMetadata * bpeAB)
        else:
          ldsNumBytesMetadata = 0
          ldsNumBytesAlignedMetadata = 0

        return ldsNumBytesMetadata, ldsNumBytesAlignedMetadata

      if state["LocalReadVectorWidthA"] == -1:
        state["LocalReadVectorWidthA"] = state["LocalReadVectorWidth"]
      if state["LocalReadVectorWidthB"] == -1:
        state["LocalReadVectorWidthB"] = state["LocalReadVectorWidth"]

      def calLRVW():
        # Default LocalReadVectorWidth
        if state["EnableMatrixInstruction"]:
          # Default LocalReadVectorWidth
          autoLRVWA = False
          maxLRVWA = int(Solution.MAX_NUM_DS_LOAD_BYTES // state["ProblemType"]["MacDataTypeA"].numBytes())
          # Set maxLRVW to 32 for 6 bits float: use two load instructions b128(4 vgpr) and b64(2 vgpr) to mimic b192
          if isaInfoMap[isa].asmCaps["HasWMMA_f8f6f4"] and state["ProblemType"]["MacDataTypeA"].numBytes() == 0.75:
            maxLRVWA = 32
          if state["LocalReadVectorWidthA"] == -1:
            autoLRVWA = True
            if state["TransposeLDS"] or (state["MIInputPerThread"] * state["ProblemType"]["MacDataTypeA"].numBytes() > Solution.MAX_NUM_DS_LOAD_BYTES):
              state["LocalReadVectorWidthA"] = maxLRVWA
            else:
              state["LocalReadVectorWidthA"] = min(state["MIInputPerThread"], maxLRVWA)
            if state["LocalReadVectorWidthA"] > maxLRVWA:
              raise RuntimeError("LocalReadVectorWidthA (%d) exceeds max %d (# bytes of lrvw > 32)" \
                                 % (state["LocalReadVectorWidthA"], maxLRVWA))
          else:
            if isaInfoMap[isa].asmCaps["HasWMMA_V3"]:
              if state["LocalReadVectorWidthA"] != maxLRVWA and state["TransposeLDS"]:
                reject(state, printRejectionReason, f"gfx1250 requires lrvwA == {maxLRVWA} for datatype {state['ProblemType']['MacDataTypeA']}, actual value: {state['LocalReadVectorWidthA']}")
            if state["ProblemType"]["Sparse"]:
              if isaInfoMap[isa].asmCaps["HasSMFMA"]: # gfx942, gfx950
                if state["MIInputPerThread"] * state["ProblemType"]["MacDataTypeA"].numBytes() > Solution.MAX_NUM_DS_LOAD_BYTES:
                  if state["LocalReadVectorWidthA"] < state["MIInputPerThread"] // 2:
                    reject(state, printRejectionReason, "LocalReadVectorWidthA < %u" %(state["MIInputPerThread"] // 2))
              elif isaInfoMap[isa].asmCaps["HasSWMMAC_gfx1250"]: # gfx1250
                if state["ProblemType"]["Sparse"] and state["LocalReadVectorWidthA"] * state["ProblemType"]["MacDataTypeA"].numBytes() > Solution.MAX_NUM_DS_LOAD_BYTES:
                  reject(state, printRejectionReason, "LocalReadVectorWidthA * BytePerMacDataTypeA(%s) > %d bytes." % (state["ProblemType"]["MacDataTypeA"].numBytes(), Solution.MAX_NUM_DS_LOAD_BYTES))
            elif not state["ProblemType"]["Sparse"] and not state["UseF32XEmulation"] and not(state["ProblemType"]["MacDataTypeA"].is8bitFloat() and (state["MatrixInstK"] in [64, 128,])):
              if state["LocalReadVectorWidthA"] < state["MIInputPerThread"] and not state["LDSTrInst"] and not isaInfoMap[isa].asmCaps["HasWMMA_V3"]:
                reject(state, printRejectionReason, "LocalReadVectorWidthA < %u" %(state["MIInputPerThread"])) # << Rejected here
            if state["LocalReadVectorWidthA"] > state["MIInputPerThread"] and not state["TransposeLDS"]:
              reject(state, printRejectionReason, "LocalReadVectorWidth require Transpose LDS")

          if autoLRVWA:
            if state["LocalReadVectorWidthA"] // state["MIInputPerThread"] > 1:
              if (state["DepthU"] // state["MatrixInstK"] <= state["LocalReadVectorWidthA"] // state["MIInputPerThread"]):
                # if only have 1 iteration with wider local read, reduce LRVW to have better scheduling (at least 2 iterations)
                state["LocalReadVectorWidthA"] //= 2

          # Default LocalReadVectorWidth
          autoLRVWB = False
          maxLRVWB = int(Solution.MAX_NUM_DS_LOAD_BYTES // state["ProblemType"]["MacDataTypeB"].numBytes())
          # Set maxLRVW to 32 for 6 bits float: use two load instructions b128(4 vgpr) and b64(2 vgpr) to mimic b192
          if isaInfoMap[isa].asmCaps["HasWMMA_f8f6f4"] and state["ProblemType"]["MacDataTypeB"].numBytes() == 0.75:
            maxLRVWB = 32
          if state["LocalReadVectorWidthB"] == -1:
            autoLRVWB = True
            if state["TransposeLDS"] or (state["MIInputPerThread"] * state["ProblemType"]["MacDataTypeB"].numBytes() > Solution.MAX_NUM_DS_LOAD_BYTES):
              state["LocalReadVectorWidthB"] = maxLRVWB
            else:
              state["LocalReadVectorWidthB"] = min(state["MIInputPerThread"], maxLRVWB)
            if state["LocalReadVectorWidthB"] > maxLRVWB:
              raise RuntimeError("LocalReadVectorWidthB (%d) exceeds max %d (# bytes of lrvw > 32)" \
                                 % (state["LocalReadVectorWidthB"], maxLRVWB))
          else:
            if isaInfoMap[isa].asmCaps["HasWMMA_V3"]:
              if state["LocalReadVectorWidthB"] != maxLRVWB and state["TransposeLDS"]:
                reject(state, printRejectionReason, f"gfx1250 requires lrvwB == {maxLRVWB} for MacDataTypeB {state['ProblemType']['MacDataTypeB']}, actual value: {state['LocalReadVectorWidthB']}")
            # TODO: Find better conditons to filter gfx1250 solutions
            if state["ProblemType"]["Sparse"]:
              if isaInfoMap[isa].asmCaps["HasSMFMA"]: # gfx942, gfx950
                if state["MIInputPerThread"] * state["ProblemType"]["MacDataTypeB"].numBytes() > Solution.MAX_NUM_DS_LOAD_BYTES:
                  if state["LocalReadVectorWidthB"] < state["MIInputPerThread"] // 2:
                    reject(state, printRejectionReason, "LocalReadVectorWidthB < %u" %(state["MIInputPerThread"] // 2))
              elif isaInfoMap[isa].asmCaps["HasSWMMAC_gfx1250"]: # gfx1250
                if state["ProblemType"]["Sparse"] and state["LocalReadVectorWidthB"] * state["ProblemType"]["MacDataTypeB"].numBytes() > Solution.MAX_NUM_DS_LOAD_BYTES:
                  reject(state, printRejectionReason, "LocalReadVectorWidthB * BytePerMacDataTypeB(%s) > %d bytes." % (state["ProblemType"]["MacDataTypeB"].numBytes(), Solution.MAX_NUM_DS_LOAD_BYTES))
            elif not state["ProblemType"]["Sparse"] and not state["UseF32XEmulation"] and not(state["ProblemType"]["MacDataTypeB"].is8bitFloat() and (state["MatrixInstK"] in [64, 128,])):
              if state["LocalReadVectorWidthB"] < state["MIInputPerThread"] and not state["LDSTrInst"] and not isaInfoMap[isa].asmCaps["HasWMMA_V3"]:
                reject(state, printRejectionReason, "LocalReadVectorWidthB < %u" %(state["MIInputPerThread"]))
            if state["LocalReadVectorWidthB"] > state["MIInputPerThread"] and not state["TransposeLDS"]:
              reject(state, printRejectionReason, "LocalReadVectorWidthB require Transpose LDS")

          if autoLRVWB:
            if state["LocalReadVectorWidthB"] // state["MIInputPerThread"] > 1:
              if (state["DepthU"] // state["MatrixInstK"] <= state["LocalReadVectorWidthB"] // state["MIInputPerThread"]):
                # if only have 1 iteration with wider local read, reduce LRVW to have better scheduling (at least 2 iterations)
                state["LocalReadVectorWidthB"] //= 2

          if autoLRVWA or autoLRVWB:
            wlrA = max(state["LocalReadVectorWidthA"] // state["MIInputPerThread"], 1)
            wlrB = max(state["LocalReadVectorWidthB"] // state["MIInputPerThread"], 1)
            if (wlrA > 1) or (wlrB > 1):
              padA, padB, padM, padMXSA, padMXSB = calcLdsPad(isaInfoMap)
              ldsBlockSizePerPadA = calcLdsBlockSizePerPad("A", state["LocalReadVectorWidthA"])
              ldsBlockSizePerPadB = calcLdsBlockSizePerPad("B", state["LocalReadVectorWidthB"])
              ldsBlockSizePerPadMXSA = calcLdsBlockSizePerPad("MXSA", state["LocalReadVectorWidthMXS"]) if state["ProblemType"]["MXBlockA"] else 0
              ldsBlockSizePerPadMXSB = calcLdsBlockSizePerPad("MXSB", state["LocalReadVectorWidthMXS"]) if state["ProblemType"]["MXBlockB"] else 0
              ldsBlockSizePerPadA = 0 if padA == 0 else ldsBlockSizePerPadA
              ldsBlockSizePerPadB = 0 if padB == 0 else ldsBlockSizePerPadB
              ldsBlockSizePerPadMXSA = 0 if padMXSA == 0 else ldsBlockSizePerPadMXSA
              ldsBlockSizePerPadMXSB = 0 if padMXSB == 0 else ldsBlockSizePerPadMXSB
              checkLdsBlockSizePerPadForTDM(ldsBlockSizePerPadA, ldsBlockSizePerPadB, ldsBlockSizePerPadMXSA, ldsBlockSizePerPadMXSB)
              (ldsNumBytesA, ldsNumBytesAlignedA) = calcLdsNumBytesAB("A", padA, ldsBlockSizePerPadA)
              (ldsNumBytesB, ldsNumBytesAlignedB) = calcLdsNumBytesAB("B", padB, ldsBlockSizePerPadB)
              (ldsNumBytesMXSA, ldsNumBytesAlignedMXSA) = calcLdsNumBytesAB("MXSA", padMXSA, ldsBlockSizePerPadMXSA) if state["ProblemType"]["MXBlockA"] else (0, 0)
              (ldsNumBytesMXSB, ldsNumBytesAlignedMXSB) = calcLdsNumBytesAB("MXSB", padMXSB, ldsBlockSizePerPadMXSB) if state["ProblemType"]["MXBlockB"] else (0, 0)
              (ldsNumBytesMetadata, ldsNumBytesAlignedMetadata) = calcLdsNumBytesM()
              if (ldsNumBytesAlignedA + ldsNumBytesAlignedB) > state["MaxLDS"]:
                if wlrA > 1:
                  state["LocalReadVectorWidthA"] //= 2
                if wlrB > 1:
                  state["LocalReadVectorWidthB"] //= 2
            wlrA = max(state["LocalReadVectorWidthA"] // state["MIInputPerThread"], 1)
            wlrB = max(state["LocalReadVectorWidthB"] // state["MIInputPerThread"], 1)
            if wlrA > wlrB:
              state["LocalReadVectorWidthA"] = wlrB * state["MIInputPerThread"]
            if wlrA < wlrB:
              state["LocalReadVectorWidthB"] = wlrA * state["MIInputPerThread"]

          if state["ProblemType"]["Sparse"] == 1:
            state["LocalReadVectorWidthMetadata"] = state["LocalReadVectorWidthA"]
          elif state["ProblemType"]["Sparse"] == 2:
            state["LocalReadVectorWidthMetadata"] = state["LocalReadVectorWidthB"]

          if state["enableGLTrA"] or state["enableGLTrB"]:
            state["LocalReadVectorWidth"] = 8
            state["LocalReadVectorWidthA"] = 8
            state["LocalReadVectorWidthB"] = 8
        else:
          if state["UseDotInstruction"]:
            # dot2: LRVW should be equal to NumDotElements * InnerUnroll
            if state["LocalReadVectorWidth"] not in [-1, state["NumDotElements"] * state["InnerUnroll"]]:
              reject(state, printRejectionReason, "dot kernel requires LocalReadVectorWidth = NumDotElements(%u) * InnerUnroll(%u)" \
                % (state["NumDotElements"], state["InnerUnroll"]))
              return
            state["LocalReadVectorWidth"] = state["NumDotElements"] * state["InnerUnroll"]
            state["LocalReadVectorWidthA"] = state["NumDotElements"] * state["InnerUnroll"]
            state["LocalReadVectorWidthB"] = state["NumDotElements"] * state["InnerUnroll"]
          else: # mac
            if state["LocalReadVectorWidth"] == -1:
              state["LocalReadVectorWidth"] = state["VectorWidthA"]
            if state["LocalReadVectorWidth"] != state["VectorWidthA"] or \
               state["LocalReadVectorWidth"] != state["VectorWidthB"]:
              reject(state, printRejectionReason, "LocalReadVectorWidth must equal VectorWidthA/B for MAC kernels")
            if state["LocalReadVectorWidthA"] == -1:
              state["LocalReadVectorWidthA"] = state["VectorWidthA"]
            if state["LocalReadVectorWidthB"] == -1:
              state["LocalReadVectorWidthB"] = state["VectorWidthB"]


      def calLRVWFor950MX():
        # Determine if we need to infer LRVW. Returns True if,
        #   - state["LocalReadVectorWidth{tc}"] is -1, and,
        #   - state["ProblemType"]["MacDataType{tc}"] is not 6-bit float
        # If the LRVW is set by the user, validate the configuration and rejects if,
        #   - state["LocalReadVectorWidth{tc}"] * state["ProblemType"]["MacDataType{tc}"].numRegisters() < 1 if not sparse
        #   - state["LocalReadVectorWidth{tc}"] // 2 * state["ProblemType"]["MacDataType{tc}"].numRegisters() < 1 is sparse
        #   - state["LocalReadVectorWidth{tc}"] > state["MIInputPerThread"] and LDS is not transposed 
        def isAutoLRVW(tc) -> bool:
          autoLRVW = False
          if state[f"LocalReadVectorWidth{tc}"] != -1:
            tmplrvw = (state[f"LocalReadVectorWidth{tc}"] // 2) if state["ProblemType"]["Sparse"] else state[f"LocalReadVectorWidth{tc}"]
            if tmplrvw * state["ProblemType"][f"MacDataType{tc}"].numRegisters() < 1:
              reject(state, "LocalReadVectorWidth * dataRegister < 1")
            if state[f"LocalReadVectorWidth{tc}"] > state["MIInputPerThread"] and not state["TransposeLDS"]:
              reject(state, "LocalReadVectorWidth require Transpose LDS")
          else:
            if state["ProblemType"][f"MacDataType{tc}"].is6bitFloat():
              state[f"LocalReadVectorWidth{tc}"] = 32 if state[f"UnrollMajorLDS{tc}"] else 16
            else:
              autoLRVW = True
              if state["TransposeLDS"] and (not state[f"DirectToLds{tc}"]):
                state[f"LocalReadVectorWidth{tc}"] = int(16 // state["ProblemType"][f"MacDataType{tc}"].numBytes())
              else:
                if state["ProblemType"]["Sparse"] and state["MIInputPerThread"] * state["ProblemType"][f"MacDataType{tc}"].numBytes() > 16:
                  state[f"LocalReadVectorWidth{tc}"] = int(16 // state["ProblemType"][f"MacDataType{tc}"].numBytes())
                else:
                  state[f"LocalReadVectorWidth{tc}"] = state["MIInputPerThread"]
              if state[f"LocalReadVectorWidth{tc}"] // state["MIInputPerThread"] > 1:
                if (state["DepthU"] // state["MatrixInstK"] <= state[f"LocalReadVectorWidth{tc}"] // state["MIInputPerThread"]):
                  # if only have 1 iteration with wider local read, reduce LRVW to have better scheduling (at least 2 iterations)
                  state[f"LocalReadVectorWidth{tc}"] //= 2
          return autoLRVW

        if state["EnableMatrixInstruction"]:
          autoLRVWA = isAutoLRVW("A")
          autoLRVWB = isAutoLRVW("B")
          if autoLRVWA or autoLRVWB:
            wlrA = max(state["LocalReadVectorWidthA"] // state["MIInputPerThread"], 1)
            wlrB = max(state["LocalReadVectorWidthB"] // state["MIInputPerThread"], 1)

            if (wlrA > 1) or (wlrB > 1):
              padA, padB, padM = calcLdsPad(state["LocalReadVectorWidth"], isaInfoMap)
              ldsBlockSizePerPadA, ldsBlockSizePerPadB = calcLdsBlockSizePerPad(state["LocalReadVectorWidth"])
              ldsNumBytesA, ldsNumBytesAlignedA, ldsNumBytesB, ldsNumBytesAlignedB, ldsNumBytesMetadata, ldsNumBytesAlignedMetadata, \
              ldsNumBytesMXSA, ldsNumBytesAlignedMXSA, ldsNumBytesMXSB, ldsNumBytesAlignedMXSB \
                = calcLdsNumBytes(padA, ldsBlockSizePerPadA, padB, ldsBlockSizePerPadB)
              ldsNumBytes = ldsNumBytesAlignedA + ldsNumBytesAlignedB + \
                            ldsNumBytesAlignedMXSA + ldsNumBytesAlignedMXSB + \
                            ldsNumBytesAlignedMetadata 
              if ldsNumBytes > state["MaxLDS"]:
                if wlrA > 1:
                  state["LocalReadVectorWidthA"] //= 2
                if wlrB > 1:
                  state["LocalReadVectorWidthB"] //= 2

            wlrA = max(state["LocalReadVectorWidthA"] // state["MIInputPerThread"], 1)
            wlrB = max(state["LocalReadVectorWidthB"] // state["MIInputPerThread"], 1)
            if wlrA > wlrB:
              state["LocalReadVectorWidthA"] = wlrB * state["MIInputPerThread"]
            if wlrA < wlrB:
              state["LocalReadVectorWidthB"] = wlrA * state["MIInputPerThread"]

          if state["ProblemType"]["Sparse"] == 1:
            state["LocalReadVectorWidthMetadata"] = state["LocalReadVectorWidthA"]
          elif state["ProblemType"]["Sparse"] == 2:
            state["LocalReadVectorWidthMetadata"] = state["LocalReadVectorWidthB"]

          if state["ProblemType"]["MXBlockA"]:
            state["LocalReadVectorWidthMXSA"] = 1 # TODO: check if need to fomulization
          if state["ProblemType"]["MXBlockB"]:
            state["LocalReadVectorWidthMXSB"] = 1 # TODO: check if need to fomulization
        else:
          reject(state, printRejectionReason, "expecting MFMA for MX datatypes")
          return

      # Default LocalReadVectorWidth
      if state["ISA"] == IsaVersion(9,5,0) and (state["ProblemType"]["MXBlockA"] or state["ProblemType"]["MXBlockB"]):
        calLRVWFor950MX()
      else:
        calLRVW()

      def calcOptGRVW(lrvw: int, unrollMajorLDS: bool, datatype: DataType) -> int:
        # with UnrollMajorLDS, GRVW need to less or equal than LRVW to have conflict free LDS read with padding.
        optGRVW = lrvw if unrollMajorLDS else 4 / datatype.numRegisters()
        if optGRVW * datatype.numBytes() > 16 and not(isaInfoMap[isa].asmCaps["HasWMMA_f8f6f4"] and datatype.numBytes() == 0.75):
          optGRVW = int(16 // datatype.numBytes())
        return optGRVW

      def calSwizzlePackK(state, tc):
        return 16 // state[f"MIInputPerThread{tc}"] // int(state["ProblemType"][f"DataType{tc}"].numBytes())

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
              reject(state, printRejectionReason, "GRVWA=-2 is set for skinny MT")
          elif state["GlobalReadVectorWidthA"] == -1:
            if state["ProblemType"]["SwizzleTensorA"]:
              state["GlobalReadVectorWidthA"] = state["MIInputPerThreadA"] * calSwizzlePackK(state, "A")
            elif state["ProblemType"]["DataTypeA"].is6bitFloat():
              state["GlobalReadVectorWidthA"] = 32	  
            elif state["enableGLTrA"]:
              state["GlobalReadVectorWidthA"] = 8
            else:
              optGRVW = calcOptGRVW(state["LocalReadVectorWidthA"], state["UnrollMajorLDSA"], state["ProblemType"]["DataTypeA"])
              curGRVW = 1
              state["GlobalReadVectorWidthA"] = int(curGRVW)
              while (curGRVW <= optGRVW):
                if (state["MacroTile0"]*state["_DepthUA"]//state["NumThreads"]) % curGRVW == 0:
                  state["GlobalReadVectorWidthA"] = int(curGRVW)
                curGRVW *= 2
        if state["ProblemType"]["MXBlockA"]:
          state["GlobalReadVectorWidthMXSA"] = max(state["MacroTile0"] * state["_DepthUMXSA"] // state["NumThreads"], 1)
          # workaround for DTL
          # use 32bit load for DTL if GlobalReadVectorWidthMXS is 8
          if state["DirectToLdsMXSA"] and state["GlobalReadVectorWidthMXSA"] == 8:
            state["GlobalReadVectorWidthMXSA"] = 4
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
              reject(state, printRejectionReason, "GRVWB=-2 is set for skinny MT")
          elif state["GlobalReadVectorWidthB"] == -1:
            if state["ProblemType"]["SwizzleTensorB"]:
              state["GlobalReadVectorWidthB"] = state["MIInputPerThreadB"] * calSwizzlePackK(state, "B")
            elif state["ProblemType"]["DataTypeB"].is6bitFloat():
              state["GlobalReadVectorWidthB"] = 32
            elif state["enableGLTrB"]:
              state["GlobalReadVectorWidthB"] = 8
            else:
              optGRVW = calcOptGRVW(state["LocalReadVectorWidthB"], state["UnrollMajorLDSB"], state["ProblemType"]["DataTypeB"])
              curGRVW = 1
              state["GlobalReadVectorWidthB"] = int(curGRVW)
              while (curGRVW <= optGRVW):
                if (state["MacroTile1"]*state["_DepthUB"]//state["NumThreads"]) % curGRVW == 0:
                  state["GlobalReadVectorWidthB"] = int(curGRVW)
                curGRVW *= 2
        if state["ProblemType"]["MXBlockB"]:
          state["GlobalReadVectorWidthMXSB"] = max(state["MacroTile1"] * state["_DepthUMXSB"] // state["NumThreads"], 1)
          # workaround for DTL
          # use 32bit load for DTL if GlobalReadVectorWidthMXS is 8
          if state["DirectToLdsMXSB"] and state["GlobalReadVectorWidthMXSB"] == 8:
            state["GlobalReadVectorWidthMXSB"] = 4
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

      if state["EnableMatrixInstruction"]:
        if state["ProblemType"]["MXBlockA"]:
          state["GlobalReadVectorWidthMXSA"] = max(state["MacroTile0"] * state["_DepthUMXSA"] // state["NumThreads"], 1)
          if state["ProblemType"]["TLUMXSA"]:
            state["GlobalReadVectorWidthMXSA"] = min(state["GlobalReadVectorWidthMXSA"], state["MacroTile0"])
          else:
            state["GlobalReadVectorWidthMXSA"] = min(state["GlobalReadVectorWidthMXSA"], state["_DepthUMXSA"])
          state["GlobalReadVectorWidthMXSA"] = min(state["GlobalReadVectorWidthMXSA"], 16)
        if state["ProblemType"]["MXBlockB"]:
          state["GlobalReadVectorWidthMXSB"] = max(state["MacroTile1"] * state["_DepthUMXSB"] // state["NumThreads"], 1)
          if state["ProblemType"]["TLUMXSB"]:
            state["GlobalReadVectorWidthMXSB"] = min(state["GlobalReadVectorWidthMXSB"], state["MacroTile1"])
          else:
            state["GlobalReadVectorWidthMXSB"] = min(state["GlobalReadVectorWidthMXSB"], state["_DepthUMXSB"])
          state["GlobalReadVectorWidthMXSB"] = min(state["GlobalReadVectorWidthMXSB"], 16)

      #for tensor swizzling, we calculate pack-k to achieve buffer_load_dwordx4
      for tc in ("A", "B",):
        if state["ProblemType"][f"SwizzleTensor{tc}"]:
          if not state["EnableMatrixInstruction"]:
            reject(state, printRejectionReason, f"Tensor {tc} swizzling supports MI only")
          # Print rejection reason instead of force set
          # 16 means bytes of buffer_load_dwordx4
          SwizzlePackK = calSwizzlePackK(state, tc)
          if state[f"GlobalReadVectorWidth{tc}"] != state[f"MIInputPerThread{tc}"] * SwizzlePackK:
            GRVW_TC = state[f"GlobalReadVectorWidth{tc}"]
            MIInPerThread = state[f"MIInputPerThread{tc}"]
            reject(state, printRejectionReason, f"SwizzleTensor{tc} doesn't support GRVW{tc} ({GRVW_TC}) != MIInputPerThread{tc} ({MIInPerThread}) * {SwizzlePackK}")

      if state["ProblemType"]["SwizzleTensorA"]:
        if not state["DirectToVgprA"]:
          reject(state, printRejectionReason, f"Tensor A swizzling requires DirectToVgprA")
        if not state["ProblemType"]["TransposeA"]:
          reject(state, printRejectionReason, f"Tensor A swizzling supports TN or TT only")

      if state["ProblemType"]["SwizzleTensorB"]:
        if not state["DirectToVgprB"]:
          reject(state, printRejectionReason, f"Tensor B swizzling requires DirectToVgprB")
        if state["ProblemType"]["TransposeB"]:
          reject(state, printRejectionReason, f"Tensor B swizzling supports TN or NN only")

        # TODO- NN fails validation due to DTVB + Tail-Loop is not working correctly
        if not (state["ProblemType"]["TransposeA"] and not state["ProblemType"]["TransposeB"]):
          reject(state, printRejectionReason, f"Tensor B swizzling supports TN only")

      # Force GRVW the same when UnrollLoopSwapGlobalReadOrder = 1.
      if genGRVWA and state["UnrollLoopSwapGlobalReadOrder"] == 1:
        state["GlobalReadVectorWidthA"] = min(state["GlobalReadVectorWidthA"], state["GlobalReadVectorWidthB"])
      if genGRVWB and state["UnrollLoopSwapGlobalReadOrder"] == 1:
        state["GlobalReadVectorWidthB"] = min(state["GlobalReadVectorWidthA"], state["GlobalReadVectorWidthB"])

      # reject - VW too big
      if (state["VectorWidthA"] * state["ProblemType"]["MacDataTypeA"].numBytes()) > 16:
        reject(state, printRejectionReason, "VWA * MacDataTypeA.numBytes() > 16")
      if (state["VectorWidthB"] * state["ProblemType"]["MacDataTypeB"].numBytes()) > 16:
        reject(state, printRejectionReason, "VWB * MacDataTypeB.numBytes() > 16")

      # reject - GRVW too big
      if (state["GlobalReadVectorWidthA"] * state["ProblemType"]["DataTypeA"].numBytes()) > 24:
        reject(state, printRejectionReason, "GRVWA * DataTypeA.numBytes() > 24")
      if (state["GlobalReadVectorWidthB"] * state["ProblemType"]["DataTypeB"].numBytes()) > 24:
        reject(state, printRejectionReason, "GRVWB * DataTypeB.numBytes() > 24")

      disableGNLC = False # Set to true to disable GNLC if needed
      if state["UseSgprForGRO"] == 1:
        disableGNLC = True
      isMixedPrec = (state["ProblemType"]["DataTypeA"].numBytes() != state["ProblemType"]["DataTypeB"].numBytes())
      if state["DirectToLds"] and state["LocalSplitU"] == 1 \
         and not isMixedPrec and not state["ProblemType"]["Sparse"] \
         and state["MatrixInstB"] == 1 \
         and not disableGNLC:

        for tc in ['A', 'B']:
          if state["DirectToLds%s"%tc]:
            # Check if we are requesting b64 loads for A/B - these are not compatible with DTL
            grwidth = state["GlobalReadVectorWidth%s"%tc] * state["ProblemType"]["DataType%s"%tc].numBytes()
            # Check that GR layout is the same as LDS layout for A/B
            sameLayout = state["ProblemType"]["TLU%s"%tc] != state["UnrollMajorLDS%s"%tc]
            state["UseGeneralizedNLCOne%s"%tc] = grwidth != 8 and sameLayout \
              and state["WaveSeparateGlobalRead%s"%tc] == 0 and not state["DirectToVgpr%s"%tc]
            # workaround: disable UseGeneralizedNLCOne for MXBlock (TODO: enable it for MXBlock)
            if state["ProblemType"]["MXBlock%s"%tc]:
              state["UseGeneralizedNLCOne%s"%tc] = False
          else:
            state["UseGeneralizedNLCOne%s"%tc] = False
        state["UseGeneralizedNLCOneMetadata"] = False
        state["_UseSgprForGRO"] = 0
      else:
        state["UseGeneralizedNLCOneA"] = False
        state["UseGeneralizedNLCOneB"] = False
        state["UseGeneralizedNLCOneMetadata"] = False

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

        # handle global read vector width A
        if state["ProblemType"]["TLUA"]: # NT/NN
          totalElementsCoalescedA = state["MacroTileA"]
          totalElementsPerpA = state["_DepthUA"]
          if state["DirectToVgprA"]:
            totalElementsCoalescedA *= state["MIWaveGroup"][1]
        else: # TN/TT
          totalElementsCoalescedA = state["_DepthUA"]
          totalElementsPerpA = state["MacroTileA"]
          if state["DirectToVgprA"]:
            totalElementsPerpA *= state["MIWaveGroup"][1]

        if state["GlobalReadVectorWidthA"] > totalElementsCoalescedA:
          reject(state, printRejectionReason, f"GRVWA({state['GlobalReadVectorWidthA']}) > Coaleased({totalElementsCoalescedA})")

        totalElementsA = totalElementsCoalescedA * totalElementsPerpA

        tva = totalElementsA // state["GlobalReadVectorWidthA"]
        if not Solution.setGlobalReadVectorWidth(state, "A", tva, state["GlobalReadVectorWidthA"], printRejectionReason):
          validDepthU = False

        if state["EnableMatrixInstruction"] and state["GlobalReadVectorWidthA"]:
          partialA = state["ProblemType"]["TLUA"] and (state["AssertFree0ElementMultiple"] % state["GlobalReadVectorWidthA"] != 0)
          if partialA and not state["UseGeneralizedNLCOneA"]:
            limitBytes = 24 if state["ProblemType"]["DataType"].is6bitFloat else 16
            glvwAlimit = int(limitBytes / state["ProblemType"]["MacDataTypeA"].numBytes())
            if state["SourceSwap"]:
              matrixInstM = (state["MatrixInstM"] * state["MatrixInstBM"]) if (state["MatrixInstM"] == 4) else state["MatrixInstM"]
              glvwAlimit = matrixInstM * state["VectorWidthA"]
            else:
              matrixInstN = (state["MatrixInstN"] * state["MatrixInstBN"]) if (state["MatrixInstN"] == 4) else state["MatrixInstN"]
              glvwAlimit  = state["MIOutputVectorWidth"] * (state["WavefrontSize"] // matrixInstN)
            if state["ProblemType"]["MacDataTypeA"].numRegisters() == 0.25:
              glvwAlimit = max(glvwAlimit, 4)

            # reduce GLVA if GLVA larger than MIOVW
            if state["GlobalReadVectorWidthA"] > glvwAlimit:
              tva = totalElementsA // glvwAlimit
              if not Solution.setGlobalReadVectorWidth(state, "A", tva, glvwAlimit, printRejectionReason):
                validDepthU = False

        GlobalReadVectorWidthA = state["GlobalReadVectorWidthA"]
        totalVectorsCoalescedA = totalElementsCoalescedA // GlobalReadVectorWidthA

        # handle global read vector width MXSA
        if state["ProblemType"]["MXBlockA"] and not state["UseSubtileImpl"]:
          if state["ProblemType"]["TLUMXSA"]: # NT/NN
            totalElementsCoalescedMXSA = state["MacroTileMXSA"]
            totalElementsPerpMXSA = state["_DepthUMXSA"]
            if state["DirectToVgprMXSA"]:
              totalElementsCoalescedMXSA *= state["MIWaveGroup"][1]
          else: # TN/TT
            totalElementsCoalescedMXSA = state["_DepthUMXSA"]
            totalElementsPerpMXSA = state["MacroTileMXSA"]
            if state["DirectToVgprMXSA"]:
              totalElementsPerpMXSA *= state["MIWaveGroup"][1]

          totalElementsMXSA = totalElementsCoalescedMXSA * totalElementsPerpMXSA

          tvmxsa = totalElementsMXSA // state["GlobalReadVectorWidthMXSA"]
          if not Solution.setGlobalReadVectorWidth(state, "MXSA", tvmxsa, state["GlobalReadVectorWidthMXSA"], printRejectionReason):
            validDepthU = False

          GlobalReadVectorWidthMXSA = state["GlobalReadVectorWidthMXSA"]
          totalVectorsCoalescedMXSA = totalElementsCoalescedMXSA // GlobalReadVectorWidthMXSA

        # handle global read vector width B
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

        if state["GlobalReadVectorWidthB"] > totalElementsCoalescedB:
          reject(state, printRejectionReason, f"GRVWB({state['GlobalReadVectorWidthB']}) > Coaleased({totalElementsCoalescedB})")

        totalElementsB = totalElementsCoalescedB * totalElementsPerpB

        # handle global read vector width M
        if state["ProblemType"]["Sparse"] and not state["DirectToVgprSparseMetadata"]:
          if state["ProblemType"]["TLUMetadata"]:
            totalElementsCoalescedM = state["MacroTileMetadata"]
            totalElementsPerpM = depthUM
          else:
            totalElementsCoalescedM = depthUM
            totalElementsPerpM = state["MacroTileMetadata"]
          totalElementsM = totalElementsCoalescedM * totalElementsPerpM

        tvb = totalElementsB // state["GlobalReadVectorWidthB"]
        if not Solution.setGlobalReadVectorWidth(state, "B", tvb, state["GlobalReadVectorWidthB"], printRejectionReason):
          validDepthU = False

        if state["EnableMatrixInstruction"] and state["GlobalReadVectorWidthB"]:
          partialB = state["ProblemType"]["TLUB"] and (state["AssertFree1ElementMultiple"] % state["GlobalReadVectorWidthB"] != 0)
          if partialB and not state["UseGeneralizedNLCOneB"]:
            limitBytes = 24 if state["ProblemType"]["DataType"].is6bitFloat else 16
            glvwBlimit = int(limitBytes / state["ProblemType"]["MacDataTypeB"].numBytes())
            if state["SourceSwap"]:
              matrixInstM = (state["MatrixInstM"] * state["MatrixInstBM"]) if (state["MatrixInstM"] == 4) else state["MatrixInstM"]
              glvwBlimit  = state["MIOutputVectorWidth"] * (state["WavefrontSize"] // matrixInstM)
            else:  # use origin shiftptr for B
              matrixInstN = (state["MatrixInstN"] * state["MatrixInstBN"]) if (state["MatrixInstN"] == 4) else state["MatrixInstN"]
              glvwBlimit = matrixInstN * state["VectorWidthB"]
            if state["ProblemType"]["MacDataTypeB"].numRegisters() == 0.25:
              glvwBlimit = max(glvwBlimit, 4)

            # reduce GLVB if GLVB larger than MIOVW
            if state["GlobalReadVectorWidthB"] > glvwBlimit:
              tvb = totalElementsB // glvwBlimit
              if not Solution.setGlobalReadVectorWidth(state, "B", tvb, glvwBlimit, printRejectionReason):
                validDepthU = False

        GlobalReadVectorWidthB = state["GlobalReadVectorWidthB"]
        totalVectorsCoalescedB = totalElementsCoalescedB // GlobalReadVectorWidthB

        # handle global read vector width MXSB
        if state["ProblemType"]["MXBlockB"] and not state["UseSubtileImpl"]:
          if state["ProblemType"]["TLUMXSB"]: # NT/NN
            totalElementsCoalescedMXSB = state["MacroTileMXSB"]
            totalElementsPerpMXSB = state["_DepthUMXSB"]
            if state["DirectToVgprMXSB"]:
              totalElementsCoalescedMXSB *= state["MIWaveGroup"][1]
          else: # TN/TT
            totalElementsCoalescedMXSB = state["_DepthUMXSB"]
            totalElementsPerpMXSB = state["MacroTileMXSB"]
            if state["DirectToVgprMXSB"]:
              totalElementsPerpMXSB *= state["MIWaveGroup"][1]

          totalElementsMXSB = totalElementsCoalescedMXSB * totalElementsPerpMXSB

          tvmxsa = totalElementsMXSB // state["GlobalReadVectorWidthMXSB"]
          if not Solution.setGlobalReadVectorWidth(state, "MXSB", tvmxsa, state["GlobalReadVectorWidthMXSB"], printRejectionReason):
            validDepthU = False

          GlobalReadVectorWidthMXSB = state["GlobalReadVectorWidthMXSB"]
          totalVectorsCoalescedMXSB = totalElementsCoalescedMXSB // GlobalReadVectorWidthMXSB

         # handle global read vector width Metadata
        if state["ProblemType"]["Sparse"] and not state["DirectToVgprSparseMetadata"]:
          grvw = 1
          vw = 1
          if state["ProblemType"]["Sparse"] == 2:
            grvw = math.ceil(state["GlobalReadVectorWidthB"] / 4)
            vw = math.ceil(state["VectorWidthB"] / 4)
            if state["GlobalReadVectorWidthB"] % 4 != 0:
              reject(state, printRejectionReason, "Sparse B requires GRVWB %% 4 == 0, current GRVWB is %u"%state["GlobalReadVectorWidthB"])
              break
          else:
            grvw = math.ceil(state["GlobalReadVectorWidthA"] / 4)
            vw = math.ceil(state["VectorWidthA"] / 4)
            if state["GlobalReadVectorWidthA"] % 4 != 0:
              reject(state, printRejectionReason, "Sparse A requires GRVWA %% 4 == 0, current GRVWA is %u"%state["GlobalReadVectorWidthA"])
              break

          tvm = totalElementsM // grvw

          if not Solution.setGlobalReadVectorWidth(state, "Metadata", tvm, grvw, printRejectionReason):
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
                if not Solution.setGlobalReadVectorWidth(state, "Metadata", tvm, glvwMlimit, printRejectionReason):
                  validDepthU = False

        if validDepthU and state["KernelLanguage"] == "Assembly":
          if isaInfoMap[state["ISA"]].archCaps["HasEccHalf"]:
            if state["ProblemType"]["DataType"].numRegisters() == 0.5 and (not state["ProblemType"]["HighPrecisionAccumulate"]) \
               and (not state["ProblemType"]["DataTypeA"].isSingle()) and (not state["ProblemType"]["DataTypeB"].isSingle()):
                if state["GlobalReadVectorWidthA"] == 1 or state["GlobalReadVectorWidthB"] == 1:
                  reject(state, printRejectionReason, "HalfEcc requires HPA if glvw = 1")
                  break

        if state["ProblemType"]["Sparse"] and state["DirectToVgprSparseMetadata"]:
          if state["VectorWidthA"] > 1 or state["VectorWidthB"] > 1 :
            reject(state, printRejectionReason, "Not implement DTVSM with VW>1")
            break

        extraComment = ""
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

          # swizzle
          if state["LocalSplitU"] > 1:
            if state["ProblemType"]["SwizzleTensorA"]:
              SwizzlePackK = calSwizzlePackK(state, "A")
              if depthUA < state["MatrixInstK"] * SwizzlePackK * state["LocalSplitU"]:
                validDepthU = False
                extraComment = ": DepthU(%u) < Min-DU for swizzleA + LSU(%u)"%(depthUA, state["LocalSplitU"])

            if state["ProblemType"]["SwizzleTensorB"]:
              SwizzlePackK = calSwizzlePackK(state, "B")
              if depthUB < state["MatrixInstK"] * SwizzlePackK * state["LocalSplitU"]:
                validDepthU = False
                extraComment = ": DepthU(%u) < Min-DU for swizzleB + LSU(%u)"%(depthUB, state["LocalSplitU"])
        # this depthU is valid, done unless user wants to double (for TN)
        if validDepthU:
          state["DepthU"] = depthU
          break

        # this depthU not valid
        else:
          reject(state, printRejectionReason, "No valid DepthU found%s"%(extraComment))
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
      #state["ExpandPointerSwap"] = 1 # EPS is adjusted in advance
      state["1LDSBuffer"] = 1
      print2("\nSet SIA=2, force PrefetchLocalRead=1, 1LDSBuffer=1")

    # Default GlobalStoreVectorWidth
    if state["StoreVectorWidth"] == -1:
      if state["SourceSwap"]:
        state["StoreVectorWidth"] = state["VectorWidthA"]
      else:
        if state["EnableMatrixInstruction"]:
          # Adjusting StoreVectorWidth for larger CGEMM register count
          if state["ProblemType"]["DestDataType"].isSingleComplex():
            state["StoreVectorWidth"] = 4 // state["ProblemType"]["DestDataType"].numRegisters()
          else:
            state["StoreVectorWidth"] = state["MIOutputVectorWidth"]
            if state["VectorWidthA"] * state["MIOutputVectorWidth"] <= 4 / state["ProblemType"]["DestDataType"].numRegisters():
              state["StoreVectorWidth"] = state["VectorWidthA"] * state["MIOutputVectorWidth"]
          if state["LocalSplitU"] > 1:
            state["StoreVectorWidth"] = state["VectorWidthA"]
        else:
          state["StoreVectorWidth"] = state["VectorWidthA"]

    if state["EnableMatrixInstruction"] and not state["UseSubtileImpl"]:
      if state["SourceSwap"]:
        if ((state["VectorWidthA"] % state["StoreVectorWidth"]) != 0):
          reject(state, printRejectionReason, "MFMA SourceSwap mode doesn't support vwA(%u) with svw(%u)" % (state["VectorWidthA"], state["StoreVectorWidth"]))
          return
      else:
        if (((state["VectorWidthA"] * state["MIOutputVectorWidth"]) % state["StoreVectorWidth"]) != 0):
          reject(state, printRejectionReason, "MFMA non-SourceSwap mode doesn't support miovw(%u) with svw(%u)" % (state["VectorWidthA"]*state["MIOutputVectorWidth"], state["StoreVectorWidth"]))
          return

    # LocalSplitU too large?
    # dot2: every NumWaveSplitK threads compute the same element.
    numElementsPerWorkGroup = state["MacroTile0"]*state["MacroTile1"]*state["NumWaveSplitK"]

    if numElementsPerWorkGroup < state["NumThreads"]:
      reject(state, printRejectionReason, "NumElementsPerWorkGroup %u < NumThreads %u; reduce LocalSplitU" \
          % (numElementsPerWorkGroup, state["NumThreads"]))
      return

    state["NumElementsPerThread"] = numElementsPerWorkGroup // state["NumThreads"]
    state["GlobalWriteVectorWidth"] = min(state["VectorWidthA"], state["NumElementsPerThread"] )
    if state["NumElementsPerThread"] % state["GlobalWriteVectorWidth"] != 0:
      reject(state, printRejectionReason, "LSU NumElementsPerThread %u not divisible into GWVW %u" \
          % (state["NumElementsPerThread"], state["GlobalWriteVectorWidth"]))
      return
    state["NumGlobalWriteVectorsPerThread"] = state["NumElementsPerThread"] \
        // state["GlobalWriteVectorWidth"]


    # LocalSplitU but can't NumThreads%MacroTile doesn't support sideways store
    if state["LocalSplitU"] > 1:
      if not state["SourceSwap"] and state["StoreVectorWidth"] > state["VectorWidthA"]:
        reject(state, printRejectionReason, "LSU and non-SourceSwap doesn't support StoreVectorWidth(%u)>VWA(%u)." \
            % (state["StoreVectorWidth"], state["VectorWidthA"]))
        return
      if not (state["ProblemType"]["ComputeDataType"].isSingle() or state["ProblemType"]["ComputeDataType"].isInt32() or state["ProblemType"]["ComputeDataType"].isComplex()):
        reject(state, printRejectionReason, "TODO: LSU doesn't support ComputeDataType!=(single, Int32 or complex).")
        return
      if state["StoreRemapVectorWidth"] > 0:
        reject(state, printRejectionReason, "TODO: LSU doesn't support StoreRemapVectorWidth>0.")
        return
      if state["NumThreads"] % state["MacroTile0"] != 0:
        reject(state, printRejectionReason, "LocalSplitU but NumThreads=%u not divisible by MT0=%u for sideways store" \
            % (state["NumThreads"], state["MacroTile0"]))
        return
      if state["MacroTile0"]*state["MacroTile1"] % state["NumThreads"] != 0:
        reject(state, printRejectionReason, "LocalSplitU but MT0*MT1=%u elements doesn't divide into NumThreads=%u" \
            % (state["MacroTile0"]*state["MacroTile1"], state["NumThreads"]))
        return

    # GlobalSplitU doesn't work with some other things:
    if state["GlobalSplitU"] > 1 or state["GlobalSplitU"] == -1:
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
        reject(state, printRejectionReason, "GlobalSplitU only compatible with single or asm and (half or mixed) precision")
        return

    if state["ProblemType"]["DataType"].isHalf() and state["KernelLanguage"] == "Assembly":
      if (state["GlobalSplitU"] > 1 or state["GlobalSplitU"] == -1) and (not state["_GlobalAccumulation"]):
        if state["AssertFree0ElementMultiple"] < 2:
          reject(state, printRejectionReason, "Assembly GSU half requires AF0EM>=2 (for atomics on edge tiles)")

        if state["EnableMatrixInstruction"] and isaInfoMap[isa].asmCaps['HasWMMA']:
          reject(state, printRejectionReason, "Half WMMA doesn't support single buffer GSU")
          return
    # dot2 limitation
    if state["UseDotInstruction"]:
      if state["EnableMatrixInstruction"]:
        reject(state, "dot inst is for mac kernel!")
      if not bufferLoad:
        reject(state, "dot2 kernel only support bufferLoad!")
      if not ((isaInfoMap[isa].asmCaps['v_dot2_f32_f16'] and state["ProblemType"]["DataType"].isHalf()) \
      or (isaInfoMap[isa].asmCaps['v_dot2_f32_bf16'] and state["ProblemType"]["DataType"].isBFloat16())) \
      and state["ProblemType"]["HighPrecisionAccumulate"]:
        reject(state, "dot2 kernel only support DataType fp16 or bf16 with HPA")
      if state["InnerUnroll"] not in [1,2,4]:
        reject(state, "dot2 kernel requires InnerUnroll = 1,2 or 4")
      if state["NumWaveSplitK"] not in [1,2,4,8,16,32,64]:
        reject(state, "Unsupported NumWaveSplitK value. Need to be power of 2 and does not exceed 64.")
      if state["_DepthUA"] % (state["LocalReadVectorWidthA"] * state["NumWaveSplitK"]) != 0:
        reject(state, "Non-valid DepthU for dot2 kernel, need to be multiple of (LocalReadVectorWidth * InnerUnroll * NumWaveSplitK) for atomics")
      if state["_DepthUB"] % (state["LocalReadVectorWidthB"] * state["NumWaveSplitK"]) != 0:
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
      # TODO: Need to fix WS address calculation of MT<16x16 cases
      if state["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel":
        reject(state, "dot2 kernel does not support MBSK")

    if state["ProblemType"]["Sparse"] and not state["DirectToVgprSparseMetadata"]:
      state["NumLoadsCoalescedMetadata"] = 1

    if not Solution.setGlobalLoadTileDimClassic(state, "A", state["NumLoadsA"], \
        totalVectorsCoalescedA, totalElementsPerpA, state["_DepthUA"], printRejectionReason):
      return
    if state["ProblemType"]["MXBlockA"] and not state["UseSubtileImpl"]:
      if not Solution.setGlobalLoadTileDimClassic(state, "MXSA", state["NumLoadsMXSA"], \
          totalVectorsCoalescedMXSA, totalElementsPerpMXSA, state["_DepthUMXSA"], printRejectionReason):
        return
    if not Solution.setGlobalLoadTileDimClassic(state, "B", state["NumLoadsB"], \
        totalVectorsCoalescedB, totalElementsPerpB, state["_DepthUB"], printRejectionReason):
      return
    if state["ProblemType"]["MXBlockB"] and not state["UseSubtileImpl"]:
      if not Solution.setGlobalLoadTileDimClassic(state, "MXSB", state["NumLoadsMXSB"], \
          totalVectorsCoalescedMXSB, totalElementsPerpMXSB, state["_DepthUMXSB"], printRejectionReason):
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
      glvwMlimit = 16
      if state["GlobalReadVectorWidthMetadata"] < glvwMlimit:
        if state["ProblemType"]["Sparse"] == 2:
          GlobalReadVectorWidth = min(state["GlobalReadVectorWidthMetadata"] * state["NumLoadsPerpendicularB"], depthUM, glvwMlimit) #sum all need read
          tvm = totalElementsM // GlobalReadVectorWidth
          if not Solution.setGlobalReadVectorWidth(state, "Metadata", tvm, GlobalReadVectorWidth, printRejectionReason):
            #fallback
            tvm = totalElementsM // bGlobalReadVectorWidthMetadata
            Solution.setGlobalReadVectorWidth(state, "Metadata", tvm, bGlobalReadVectorWidthMetadata, printRejectionReason)

          GlobalReadVectorWidthMetadata = state["GlobalReadVectorWidthMetadata"]
          if GlobalReadVectorWidthMetadata == 0:
            GlobalReadVectorWidthMetadata = 1
          totalVectorsCoalescedM = totalElementsCoalescedM // GlobalReadVectorWidthMetadata
          totalVectorsM = totalElementsM // GlobalReadVectorWidthMetadata
        else:
          GlobalReadVectorWidth = min(state["GlobalReadVectorWidthMetadata"] * state["NumLoadsPerpendicularA"], depthUM, glvwMlimit) #sum all need read
          tvm = totalElementsM // GlobalReadVectorWidth
          if not Solution.setGlobalReadVectorWidth(state, "Metadata", tvm, GlobalReadVectorWidth, printRejectionReason):
            #fallback
            tvm = totalElementsM // bGlobalReadVectorWidthMetadata
            Solution.setGlobalReadVectorWidth(state, "Metadata", tvm, bGlobalReadVectorWidthMetadata, printRejectionReason)

          GlobalReadVectorWidthMetadata = state["GlobalReadVectorWidthMetadata"]
          if GlobalReadVectorWidthMetadata == 0:
            GlobalReadVectorWidthMetadata = 1
          totalVectorsCoalescedM = totalElementsCoalescedM // GlobalReadVectorWidthMetadata
          totalVectorsM = totalElementsM // GlobalReadVectorWidthMetadata

      if not Solution.setGlobalLoadTileDimClassic(state, "Metadata", state["NumLoadsMetadata"], \
          totalVectorsCoalescedM, totalElementsPerpM, depthUM, printRejectionReason):
        return

    # TODO
    if (0 and state["LSCA"] % state["GlobalReadVectorWidthA"] != 0):
      reject(state, printRejectionReason, "lsca % grvw != 0")
      return
    if (0 and state["LSPA"] % state["GlobalReadVectorWidthA"] != 0):
      reject(state, printRejectionReason, "lspa % grvw != 0")
      return
    if (0 and state["LSCB"] % state["GlobalReadVectorWidthB"] != 0):
      reject(state, printRejectionReason, "lscb % grvw != 0")
      return
    if (0 and state["LSPB"] % state["GlobalReadVectorWidthB"] != 0):
      reject(state, printRejectionReason, "lspb % grvw != 0")
      return

    state["LVCA"] = roundupRatio(state["LSCA"] , state["GlobalReadVectorWidthA"])
    state["LVPA"] = roundupRatio(state["LSPA"] , state["GlobalReadVectorWidthA"])

    if state["ProblemType"]["MXBlockA"]:
      state["LVCMXSA"] = roundupRatio(state["LSCMXSA"] , state["GlobalReadVectorWidthMXSA"])
      state["LVPMXSA"] = roundupRatio(state["LSPMXSA"] , state["GlobalReadVectorWidthMXSA"])

    state["LVCB"] = roundupRatio(state["LSCB"] , state["GlobalReadVectorWidthB"])
    state["LVPB"] = roundupRatio(state["LSPB"] , state["GlobalReadVectorWidthB"])

    if state["ProblemType"]["MXBlockB"]:
      state["LVCMXSB"] = roundupRatio(state["LSCMXSB"] , state["GlobalReadVectorWidthMXSB"])
      state["LVPMXSB"] = roundupRatio(state["LSPMXSB"] , state["GlobalReadVectorWidthMXSB"])

    # KRingShift wrap handling exists only in the tail loop.
    # If (k + KRingShift) would wrap inside the main loop, the kernel will be incorrect (no main-loop wrap fix).
    # Enforce a host-side runtime predicate which guarantees any KRS wrap happens only in tail.
    #
    # NOTE: This must be encoded after LVCB/LSCB are computed above.
    if state["KRingShift"]:
      # Pack predicate args (see ContractionProblemPredicates.hpp::KRingShiftTailWrapOnly):
      #   [63:48]=cacheLineBytes, [47:32]=depthU, [31:16]=mt1, [15:8]=lvcb, [7:0]=bpeB
      cacheLineBytes = int(isaInfoMap[isa].archCaps.get("vL1DCacheLineBytes", 0))
      depthU         = int(state["DepthU"])
      mt1            = int(state["MacroTile1"])
      bpeB           = int(state["ProblemType"]["DataTypeB"].numBytes())
      lvcb           = int(state["LVCB"])

      if (0 < cacheLineBytes < (1<<16)) and (0 < depthU < (1<<16)) and (0 < mt1 < (1<<16)) and (0 < lvcb < (1<<8)) and (0 < bpeB < (1<<8)):
        state["AssertKRingShiftTailWrapOnly"] = (cacheLineBytes << 48) | (depthU << 32) | (mt1 << 16) | (lvcb << 8) | bpeB
      else:
        reject(state, printRejectionReason,
               f"KRingShift requires encodable AssertKRingShiftTailWrapOnly predicate "
               f"(cacheLineBytes={cacheLineBytes}, depthU={depthU}, mt1={mt1}, "
               f"lvcb={lvcb}, lscb={state.get('LSCB', None)}, grvwB={state.get('GlobalReadVectorWidthB', None)}, "
               f"bpeB={bpeB})")
        return

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
      reject(state, printRejectionReason, "Source KernelLanguage only supports LdsPadA == LdsPadB")
      return

    # NoTailLoop parameter initialization.
    # If ASEM is multiple of DepthU TailLoop will not be used.
    state["NoTailLoop"] = False
    if state["AssertSummationElementMultiple"] % state["DepthU"] == 0:
      state["NoTailLoop"] = True

    # TailloopInNll optimization check
    if state["TailloopInNll"]:
      # Disable TailloopInNll
      # - (not MFMA) or WMMA
      # - PrefetchGlobalRead is 0
      # - NoTailLoop
      # - DepthU is not power of 2
      # - LocalSplitU > 1
      # - MX case
      if ((not state["EnableMatrixInstruction"]) or isaInfoMap[isa].asmCaps["HasWMMA"]) or \
         (state["PrefetchGlobalRead"] == 0) or \
         state["NoTailLoop"] or \
         (state["DepthU"] <=1 or (state["DepthU"] & (state["DepthU"] - 1) != 0)) or \
         state["LocalSplitU"] > 1 or \
         state["ProblemType"]["MXBlockA"] or state["ProblemType"]["MXBlockB"]:
        state["TailloopInNll"] = False

      # need restrictions for TailloopInNll
      if state["TailloopInNll"]:
        # need to disable SuppressNoLoadLoop
        state["SuppressNoLoadLoop"] = False

      # disable StaggerUcode
      # - TailloopInNll
      # - MX + StreamK (not enough sgpr)
      if state["TailloopInNll"] or \
         (state["StreamK"] and (state["ProblemType"]["MXBlockA"] or state["ProblemType"]["MXBlockB"])):
        # need to disable StaggerU
        state["StaggerU"] = 0
        state["StaggerUMapping"] = 0
        state["StaggerUStride"] = 0
        state["InternalSupportParams"]["SupportCustomStaggerU"] = False # Disable CustomStaggerU for no StagggerU code

    # Determine if we can load directly-to-Vgpr
    # need to check after state["LocalReadVectorWidth"] = -1 is resolved
    if state["DirectToVgprA"]:
      if not Solution.isDirectToVgprDoable(state, 'A', printRejectionReason, isaInfoMap):
        return  # rejected
    if state["DirectToVgprB"]:
      if not Solution.isDirectToVgprDoable(state, 'B', printRejectionReason, isaInfoMap):
        return  # rejected

    ########################################
    # LDS
    ########################################

    if state["ProblemType"]["Sparse"]:
      transposeLDSMetadata = int(state["TransposeLDSMetadata"])
      if transposeLDSMetadata == -1:
        state["TransposeLDSMetadata"] = int(not state["ProblemType"]["TLUMetadata"])
      else:
        state["TransposeLDSMetadata"] = int(transposeLDSMetadata)

      state["UnrollMajorLDSMetadata"] = False if state["ProblemType"]["Sparse"] == 2 else True

      if not state["DirectToVgprSparseMetadata"]:
        state["UnrollMajorLDSMetadata"] = state["TransposeLDSMetadata"]

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
    for tc in ['A', 'B']:
      tcmx = "MXS%s"%tc
      if state["UseSubtileImpl"] and state["ProblemType"]["MXBlock%s"%tc]:
        state["DirectToLds%s"%tcmx] = False
      if state["DirectToLds%s"%tc]:
        isDtlDoable = Solution.isDirectToLdsDoable(state, tc, isaInfoMap, printRejectionReason)
        if (not state["DirectToVgpr%s"%tc]) and isDtlDoable:
          state["DirectToLds%s"%tc] = True
          state["LocalWriteUseSgpr%s"%tc] = True
          # MX case
          if state["ProblemType"]["MXBlock%s"%tc]:
            isDtlMxDoable = Solution.isDirectToLdsDoable(state, tcmx, isaInfoMap, printRejectionReason)
            state["DirectToLds%s"%tcmx] = isDtlMxDoable
        else:
          state["DirectToLds%s"%tc] = False
          state["LocalWriteUseSgpr%s"%tc] = False
          # MX case
          if state["ProblemType"]["MXBlock%s"%tc]:
            state["DirectToLds%s"%tcmx] = False
          if not isDtlDoable:
            if state["UseGeneralizedNLCOne%s"%tc]:
              reject(state, printRejectionReason, "DirectToLds%s not doable, but GNLC%s enabled, rejecting"%(tc, tc))

    # Update parent variable so kernel display is accurate
    if state["DirectToLdsA"] and state["DirectToLdsB"]:
      state["DirectToLds"] = 1
    elif state["DirectToLdsA"]:
      state["DirectToLds"] = 2
    elif state["DirectToLdsB"]:
      state["DirectToLds"] = 3
    else:
      state["DirectToLds"] = 0
    if state["1LDSBuffer"] == -1 and state["DirectToLds"]:
      #1LDS buffer must be 0 for DirectToLdsA
      state["1LDSBuffer"] = 0
    # MX case
    if (state["ProblemType"]["MXBlockA"] or state["ProblemType"]["MXBlockB"]):
      if state["DirectToLdsA"] != state["DirectToLdsMXSA"] or state["DirectToLdsB"] != state["DirectToLdsMXSB"]:
          reject(state, printRejectionReason, "DirectToLdsA/B and DirectToLdsMXSA/B should match")
      if state["DirectToLdsA"] != state["DirectToLdsB"]:
          reject(state, printRejectionReason, "DirectToLdsA and DirectToLdsB should match")

    # does not work with UnrollLoopSwapGlobalReadOrder
    if (state["DirectToLds"] == 2 or state["DirectToLds"] == 3) and state["UnrollLoopSwapGlobalReadOrder"]:
      reject(state, printRejectionReason, "DirectToLdsA or B only does not supports UnrollLoopSwapGlobalReadOrder")
      return False

    # Re-check DTV + WaveGroup after DTL is confirmed
    if state["DirectToLds"]:
      if state["DirectToVgprA"] and state['MIWaveGroup'][1] > 1:
        reject(state, printRejectionReason, "DirectToLds + (DirectToVgprA + WaveGroups along N-Dim) is not supported yet")
        return False
      if state["DirectToVgprB"] and state['MIWaveGroup'][0] > 1:
        reject(state, printRejectionReason, "DirectToLds + (DirectToVgprB + WaveGroups along M-Dim) is not supported yet")
        return False

    state["enableLDSTrMetadata"] = isaInfoMap[isa].asmCaps["HasLDSTrB64B8"] and state["ProblemType"]["MetadataLayout"]
    if state["enableLDSTrMetadata"]:
      state["VectorWidthMetadata"] = 1

      # the VetorWidth of the sparse matrix and metadta need to be the same.
      if state["ProblemType"]["Sparse"] == 1:
        state["VectorWidthA"] = 1
      else:
        state["VectorWidthB"] = 1

    auto_LdsBlockSizePerPadA_for_mix = 0
    if state["LdsBlockSizePerPadA"] == -1:
      auto_LdsBlockSizePerPadA_for_mix = 1
    auto_LdsBlockSizePerPadB_for_mix = 0
    if state["LdsBlockSizePerPadB"] == -1:
      auto_LdsBlockSizePerPadB_for_mix = 1
    state["LdsBlockSizePerPadA"] = calcLdsBlockSizePerPad("A", state["LocalReadVectorWidthA"])
    state["LdsBlockSizePerPadB"] = calcLdsBlockSizePerPad("B", state["LocalReadVectorWidthB"])
    state["LdsBlockSizePerPadMXSA"] = calcLdsBlockSizePerPad("MXSA", state["LocalReadVectorWidthMXS"]) if state["ProblemType"]["MXBlockA"] else 0
    state["LdsBlockSizePerPadMXSB"] = calcLdsBlockSizePerPad("MXSB", state["LocalReadVectorWidthMXS"]) if state["ProblemType"]["MXBlockB"] else 0
    checkLdsBlockSizePerPadForTDM(state["LdsBlockSizePerPadA"], state["LdsBlockSizePerPadB"], state["LdsBlockSizePerPadMXSA"], state["LdsBlockSizePerPadMXSB"])

    if state["LdsBlockSizePerPadMetadata"] == -1:
      state["LdsBlockSizePerPadMetadata"] = state["LdsBlockSizePerPadA"]

    if state["EnableMatrixInstruction"]:
      if state["LdsBlockSizePerPadA"] and not state["UseGeneralizedNLCOneA"]:
        if state["UnrollMajorLDSA"]:
          if state["LdsBlockSizePerPadA"] % int(state["_DepthUA"] * getLdsBpe("A")) != 0:
            reject(state, printRejectionReason, "reject: LdsBlockSizePerPadA %u %% depthU %u x bpeA != 0" % (state["LdsBlockSizePerPadA"],state["_DepthUA"]))
          if (state["LdsBlockSizePerPadA"] // (state["_DepthUA"] * state["ProblemType"]["MacDataTypeA"].numBytes())) % state["LSPA"] != 0 and \
              state["LSPA"] % (state["LdsBlockSizePerPadA"] // (state["_DepthUA"] * state["ProblemType"]["MacDataTypeA"].numBytes())) != 0:
            reject(state, printRejectionReason, "can't pad by addrVgpr or instOffset")

      if state["LdsBlockSizePerPadB"] and not state["UseGeneralizedNLCOneB"]:
        if state["UnrollMajorLDSB"]:
          if state["LdsBlockSizePerPadB"] % int(state["_DepthUB"] * getLdsBpe("B")) != 0:
            reject(state, printRejectionReason, "reject: LdsBlockSizePerPadB %u %% depthU %u x bpeB != 0" % (state["LdsBlockSizePerPadB"],state["_DepthUB"]))
          if (state["LdsBlockSizePerPadB"] // (state["_DepthUB"] * state["ProblemType"]["MacDataTypeB"].numBytes())) % state["LSPB"] != 0 and \
              state["LSPB"] % (state["LdsBlockSizePerPadB"] // (state["_DepthUB"] * state["ProblemType"]["MacDataTypeB"].numBytes())) != 0:
            reject(state, printRejectionReason, "can't pad by addrVgpr or instOffset")
    else:
      if not state["UseDotInstruction"] and (state["UnrollMajorLDSA"] or state["UnrollMajorLDSB"]):
        reject(state, printRejectionReason, "didn't support UnrollMajorLDS in VALU mode yet (except for dot2 kernel)")
      if state["LdsBlockSizePerPadA"] != 0 or state["LdsBlockSizePerPadB"] != 0:
        reject(state, printRejectionReason, "didn't support LdsBlockSizePerPad in VALU mode yet")

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
          reject(state, printRejectionReason, "invalid local write block width")

        return blockWidth

      def subCheckLdsBlockSizePerPad(tc, idx):
        lbspp = state["LdsBlockSizePerPad%s"%tc]
        bpe = state["ProblemType"]["MacDataType%s"%tc].numBytes()
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

        if not subCheckLdsBlockSizePerPad(tc, idx) and not state["UseGeneralizedNLCOne%s"%tc]:
          if auto_LdsBlockSizePerPad_for_mix:
            printWarning("Padded address is inconisstent, set LdsBlockSizePerPad%s=0."%tc)
            state["LdsBlockSizePerPad%s"%tc] = 0
          else:
            reject(state, printRejectionReason, "%s's padded address is inconisstent"%tc)

    if(not (state["CustomKernelName"] and state["CustomKernelName"] != "")): #don't check the custom kernel.
      checkLdsBlockSizePerPad("A")
      checkLdsBlockSizePerPad("B")

    # set NoLdsWriteCode if (DirectToVgpr or DirectToLds)A+B is enabled
    state["NoLdsWriteCode"] = False
    if (state["DirectToVgprA"] or state["DirectToLdsA"]) and (state["DirectToVgprB"] or state["DirectToLdsB"]):
      state["NoLdsWriteCode"] = True
      # MX case
      if state["ProblemType"]["MXBlockA"]:
        if not (state["DirectToVgprMXSA"] or state["DirectToLdsMXSA"]):
          state["NoLdsWriteCode"] = False
      if state["ProblemType"]["MXBlockB"]:
        if not (state["DirectToVgprMXSB"] or state["DirectToLdsMXSB"]):
          state["NoLdsWriteCode"] = False
    elif state["enableTDMA"] and state["enableTDMB"]:
      state["NoLdsWriteCode"] = True

    # Use64bShadowLimitMX for MX case only
    if not (state["ProblemType"]["MXBlockA"] or state["ProblemType"]["MXBlockB"]):
      state["Use64bShadowLimitMX"] = False

    # enable scheduling GR (in LWcode for PGR2) over barrier sync
    if not state["EnableMatrixInstruction"]:
      # not valid for non MFMA case
      state["ScheduleGROverBarrier"] = 0
    else:
      numMFMA = state["MIWaveTile"][0] * state["MIWaveTile"][1] * (state["DepthU"] // (state["MatrixInstK"] * state["LocalSplitU"]))
      if state["ScheduleGROverBarrier"] == -1:
        # auto enable ScheduleGROverBarrier for PGR>=3
        # PGR2+tripleLDS case, manual enablement only
        state["ScheduleGROverBarrier"] = 0
        if state["PrefetchGlobalRead"] >= 3:
          # better to avoid applying this logic for smaller MT sizes
          # Set threshold as
          #  PGR3: MT128x64x64 with MT16x16x32x1 (with 4 waves)
          # threshold //= 2 for MI32
          thresholdMFMA = 128*64*64 / (16*16*32*4) # MI16x16x32x1 4waves
          if state["MatrixInstM"] == 32:
            thresholdMFMA //= 2
          if numMFMA >= thresholdMFMA:
            state["ScheduleGROverBarrier"] = 1
      if state["ScheduleGROverBarrier"]:
        # enable ScheduleGROverBarrier only for
        # - NoLdsWriteCode and not DTV, and
        # - PGR>=2, and
        # - PLR>=1
        # - small MT (detail below)
        if (not state["NoLdsWriteCode"]) or \
           (state["DirectToVgprA"] or state["DirectToVgprB"]) or \
           state["PrefetchGlobalRead"] < 2:
          state["ScheduleGROverBarrier"] = 0
        # disable this logic for very small MT sizes
        # Set threshold as MT64x64x64 with MT16x16x32x1 (with 4 waves)
        # threshold //= 2 for MI32
        thresholdMFMA = 64*64*64 / (16*16*32*4) # MI16x16x32x1 4waves
        if state["MatrixInstM"] == 32:
          thresholdMFMA //= 2
        if numMFMA < thresholdMFMA:
          state["ScheduleGROverBarrier"] = 0

    # number of minimum GR inc inst per MFMA
    # default 1
    # Set at least 2 for gfx950 + MI16 + smaller MT case
    # Set 3 for MX (number of GRInc is doubled)
    if state["MinGRIncPerMfma"] == -1:
      state["MinGRIncPerMfma"] = 1
      if isa == (9, 5, 0):
        if state["EnableMatrixInstruction"] and state["MatrixInstM"] == 16 and state["MatrixInstB"] == 1:
          if numMFMA<=16 or (state["ProblemType"]["MXBlockA"] or state["ProblemType"]["MXBlockB"]):
            state["MinGRIncPerMfma"] = 3
          elif numMFMA<=32:
            state["MinGRIncPerMfma"] = 2

    # calculate ldsPad
    state["LdsPadA"], state["LdsPadB"], state["LdsPadMetadata"], state["LdsPadMXSA"], state["LdsPadMXSB"] = calcLdsPad(isaInfoMap)

    if state["GlobalReadVectorWidthA"] * state["ProblemType"]["MacDataTypeA"].numBytes() == 32 and state["LdsPadA"] == 16 // state["ProblemType"]["MacDataTypeA"].numBytes():
      if auto_LdsBlockSizePerPadA_for_mix:
        state["LdsBlockSizePerPadA"] = 128
    assert(state["LdsPadA"] >= 0)

    if state["GlobalReadVectorWidthB"] * state["ProblemType"]["MacDataTypeB"].numBytes() == 32 and state["LdsPadB"] == 16 // state["ProblemType"]["MacDataTypeB"].numBytes():
      if auto_LdsBlockSizePerPadB_for_mix:
        state["LdsBlockSizePerPadB"] = 128
    assert(state["LdsPadB"] >= 0)

    # set ldsbspp = 0 for ldspad = 0
    for tc in ['A', 'B']:
      if state["LdsPad%s"%tc] == 0:
        state["LdsBlockSizePerPad%s"%tc] = 0

    # Normalize lds block-size-per-pad fields to native Python int.
    assert(int(state["LdsBlockSizePerPadA"]) == state["LdsBlockSizePerPadA"])
    assert(int(state["LdsBlockSizePerPadB"]) == state["LdsBlockSizePerPadB"])
    assert(int(state["LdsBlockSizePerPadMetadata"]) == state["LdsBlockSizePerPadMetadata"])
    state["LdsBlockSizePerPadA"] = int(state["LdsBlockSizePerPadA"])
    state["LdsBlockSizePerPadB"] = int(state["LdsBlockSizePerPadB"])
    state["LdsBlockSizePerPadMetadata"] = int(state["LdsBlockSizePerPadMetadata"])

    if (state["UnrollMajorLDSA"] or state["UnrollMajorLDSB"]) and (not state["EnableMatrixInstruction"]) and (not state["UseDotInstruction"]):
        reject(state, printRejectionReason, "UnrollMajorLDS Supports only in EnableMatrixInstruction=1 or dot2 kernel")

    (ldsNumBytesA, ldsNumBytesAlignedA) = calcLdsNumBytesAB("A", state["LdsPadA"], state["LdsBlockSizePerPadA"])
    (ldsNumBytesB, ldsNumBytesAlignedB) = calcLdsNumBytesAB("B", state["LdsPadB"], state["LdsBlockSizePerPadB"])
    (ldsNumBytesMXSA, ldsNumBytesAlignedMXSA) = calcLdsNumBytesAB("MXSA", state["LdsPadMXSA"], state["LdsBlockSizePerPadMXSA"]) if state["ProblemType"]["MXBlockA"] else (0, 0)
    (ldsNumBytesMXSB, ldsNumBytesAlignedMXSB) = calcLdsNumBytesAB("MXSB", state["LdsPadMXSB"], state["LdsBlockSizePerPadMXSB"]) if state["ProblemType"]["MXBlockB"] else (0, 0)
    (ldsNumBytesMetadata, ldsNumBytesAlignedMetadata) = calcLdsNumBytesM()

    state["LdsOffsetA_Blk"] = 0
    state["LdsOffsetB_Blk"] = 0
    state["LdsOffsetMXSA_Blk"] = 0
    state["LdsOffsetMXSB_Blk"] = 0
    state["LdsOffsetMetadata_Blk"] = 0

    # todo, can the alignment be a power of 2?
    state["LdsOffsetA"] = 0
    state["LdsNumElementsAlignedA"] = int(ldsNumBytesAlignedA)
    state["LdsNumElementsAlignedMXSA"] = int(ldsNumBytesAlignedMXSA)
    state["LdsNumElementsAlignedB"] = int(ldsNumBytesAlignedB)
    state["LdsNumElementsAlignedMXSB"] = int(ldsNumBytesAlignedMXSB)
    state["LdsNumElementsAlignedMetadata"] = int(ldsNumBytesAlignedMetadata)
    # check for auto DtlPlusLdsBuf
    if state["DtlPlusLdsBuf"] == -1:
      if state["PrefetchGlobalRead"] > 2:
        state["DtlPlusLdsBuf"] = 1
      else:
        state["DtlPlusLdsBuf"] = 0
    # disable DtlPlusLdsBuf if not applicable
    if state["DtlPlusLdsBuf"]:
      # DTLA + DTLB only
      if not(state["DirectToLdsA"] and state["DirectToLdsB"]):
        state["DtlPlusLdsBuf"] = 0
      # PGR>=2 only
      if state["PrefetchGlobalRead"] < 2:
        state["DtlPlusLdsBuf"] = 0
      # SIA==3 only
      if state["ScheduleIterAlg"] != 3:
        state["DtlPlusLdsBuf"] = 0
      # restrict feature combinations
      if state["DtlPlusLdsBuf"]:
        # force 1LDSBuffer = 0
        state["1LDSBuffer"] = 0

    # Here, 1LDSBuffer == -1 is not resolved yet.
    # (cannot move 1LDSBuffer==-1 resolution code above because of referring ldsNumBytesAB)
    # Assuming larger buffer here
    numLdsBlk = 1 if state["1LDSBuffer"] == 1 else 2
    if state["PrefetchGlobalRead"] >= 3:
      # PGR>=3, number of LDS Block is set to PGR
      numLdsBlk = state["PrefetchGlobalRead"]
      # force 1LDSBuffer = 0
      state["1LDSBuffer"] = 0
    if state["PrefetchGlobalRead"] >= 2 and state["DtlPlusLdsBuf"]:
      # PGR>=2 + DtlPlusLdsBuf case, try to allocate PGR+1 LDSBlk to schedule GR over barrier
      numLdsBlk = state["PrefetchGlobalRead"] + 1

    def setLdsOffsets(offsetBlk, numLdsBlk, ldsNumBytesB):
      if numLdsBlk <= 1:
        # 1LDSBuffer case, use 2 to calculate ldsNumBytesAB
        numLdsBlk = 2
      state["LdsOffsetA_Blk"] = offsetBlk
      state["LdsOffsetMXSA_Blk"] = state["LdsOffsetA_Blk"] + state["LdsNumElementsAlignedA"]
      state["LdsOffsetMXSB_Blk"] = state["LdsOffsetMXSA_Blk"] + state["LdsNumElementsAlignedMXSA"]
      state["LdsOffsetMetadata_Blk"] = state["LdsOffsetMXSB_Blk"] + state["LdsNumElementsAlignedMXSB"]
      state["LdsOffsetB_Blk"] = state["LdsOffsetMetadata_Blk"] + state["LdsNumElementsAlignedMetadata"]
      ldsNumBytesAB = (numLdsBlk - 2) * offsetBlk + state["LdsOffsetB_Blk"] + ldsNumBytesB
      return ldsNumBytesAB

    state["ldsNumBytesA"] = ldsNumBytesA
    state["ldsNumBytesB"] = ldsNumBytesB
    state["ldsNumBytesMXSA"] = ldsNumBytesMXSA
    state["ldsNumBytesMXSB"] = ldsNumBytesMXSB
    state["ldsNumBytesMetadata"] = ldsNumBytesMetadata

    state["LdsOffsetA"] = 0
    state["LdsOffsetMXSA"] = state["LdsOffsetA"] + state["LdsNumElementsAlignedA"]
    state["LdsOffsetMXSB"] = state["LdsOffsetMXSA"] + state["LdsNumElementsAlignedMXSA"]
    state["LdsOffsetMetadata"] = state["LdsOffsetMXSB"] + state["LdsNumElementsAlignedMXSB"]
    state["LdsOffsetB"] = state["LdsOffsetMetadata"] + state["LdsNumElementsAlignedMetadata"]
    if state["PrefetchGlobalRead"]:
      offsetBlk = state["LdsOffsetB"] + ldsNumBytesAlignedB
      roundupOffsetBlk = int(2**(math.ceil(math.log(offsetBlk, 2)))) if offsetBlk > 0 else 0

      if not isaInfoMap[isa].asmCaps["HasWMMA"]:
        state["StoreSwapAddr"] = (state["PrefetchGlobalRead"] == 2) and \
          (state["1LDSBuffer"] == 0) and numLdsBlk == 2 and \
          (offsetBlk + roundupOffsetBlk) > state["MaxLDS"]
      else:
        state["StoreSwapAddr"] = offsetBlk > 0 and (state["1LDSBuffer"] != 1) and \
          (offsetBlk + int(2**(math.ceil(math.log(offsetBlk, 2)))) > state["MaxLDS"])

      if offsetBlk > 0 and not state["StoreSwapAddr"] and numLdsBlk == 2:
        # Rounds offsetBlk to a power of two to enable inlining {s,v}_xor constants for swapping offsets
        # skip ceiling to nearest power of two for numLdsBlk>=3
        offsetBlk = roundupOffsetBlk

      ldsNumBytesAB = setLdsOffsets(offsetBlk, numLdsBlk, ldsNumBytesB)
      # decrement numLdsBlk for DtlPlusLdsBuf if it exceeds MaxLDS
      # PGR 2 case, reject kernel (need to use StoreSwapAddr in that case)
      if state["DtlPlusLdsBuf"] and ldsNumBytesAB > state["MaxLDS"]:
        numLdsBlk -= 1
        # continue with original logic for PGR2 + numLdsBlk==2
        if state["PrefetchGlobalRead"] == 2:
          if  (offsetBlk + roundupOffsetBlk) > state["MaxLDS"]:
            state["StoreSwapAddr"] = True
          else:
            offsetBlk = roundupOffsetBlk
        # re-calculate LDS size with numLdsBlk==2
        ldsNumBytesAB = setLdsOffsets(offsetBlk, numLdsBlk, ldsNumBytesB)
        # unset DtlPlusLdsBuf
        state["DtlPlusLdsBuf"] = 0
    else:
      ldsNumBytesAB = state["LdsOffsetB"] + ldsNumBytesB
    state["NumLdsBlk"] = numLdsBlk

    # lds buffer size for reduction
    # if User want to control the LDS usage, we may open this para in the future
    ldsNumBytesReduction = state["LocalSplitU"] * state["MacroTile0"] * state["MacroTile1"] * int(state["ProblemType"]["ComputeDataType"].numBytes()) if state["LocalSplitU"] > 1 else 0
    state["LocalSplitUReuseLDS"] = 1
    if ldsNumBytesReduction > state["MaxLDS"]:
      state["LocalSplitUReuseLDS"] = math.ceil(ldsNumBytesReduction / state["MaxLDS"])
      # reserve all the LDS to LSU.
      ldsNumBytesReduction = state["MaxLDS"]

    # lds max occupancy
    ldsSizeOccupancy = isaInfoMap[isa].archCaps["DeviceLDS"] // state["MaxOccupancy"]
    ldsNumBytesOccupancy = ldsSizeOccupancy

    #print("LdsOffsetB", state["LdsOffsetB"])
    #print("LdsOffsetMetadata", state["LdsOffsetMetadata"])
    #if state["PrefetchGlobalRead"]:
    #  print("LdsOffsetA_BLK", state["LdsOffsetA_Blk"])
    #  print("LdsOffsetB_BLK", state["LdsOffsetB_Blk"])
    #  print("LdsOffsetMetadata_BLK", state["LdsOffsetMetadata_Blk"])

    if state["EnableMatrixInstruction"]:
      if state["DirectToLds"] and state["1LDSBuffer"]:
        reject(state, printRejectionReason, "1LDSBuffer must be 0 for directToLds")

    if state["1LDSBuffer"] == -1:
      if ldsNumBytesAB  <= max(ldsSizeOccupancy,32768) or \
          (state["ProblemType"]["ComputeDataType"].numBytes() * state["MacroTile0"] * state["MacroTile1"] > 32768*4 and \
            not (ldsNumBytesAB > isaInfoMap[isa].archCaps["DeviceLDS"])):
        state["1LDSBuffer"] = 0
      else:
        state["1LDSBuffer"] = 1

    if state["1LDSBuffer"]:
      if not state["PrefetchGlobalRead"]:
        reject(state, printRejectionReason, "PGR=0 already use 1 LDS buffer only")
      # Should be able to support as long as NO scheduleLocalWrite
      if (not state["ScheduleIterAlg"] == 2) and (not state["ScheduleIterAlg"] == 3) and (state["ScheduleLocalWrite"]):
        reject(state, printRejectionReason, "1LDSBuffer only support SIA2 or SIA3, or SIA1 without SLW")
      state["LdsOffsetA"] = 0
      state["LdsOffsetMXSA"] = state["LdsOffsetA"] + state["LdsNumElementsAlignedA"]
      state["LdsOffsetB"] = state["LdsOffsetMXSA"] + state["LdsNumElementsAlignedMXSA"]
      state["LdsOffsetMXSB"] = state["LdsOffsetB"] + state["LdsNumElementsAlignedB"]
      state["LdsOffsetMetadata"] = state["LdsOffsetMXSB"] + state["LdsNumElementsAlignedMXSB"]
      ldsNumBytesAB = state["LdsOffsetMetadata"] + ldsNumBytesMetadata
      state["StoreSwapAddr"] = False

    # lds size is the greater of the two
    ldsNumBytes = max(ldsNumBytesAB, ldsNumBytesReduction, ldsNumBytesOccupancy)

    if state["NumElementsPerBatchStore"] == -1:
      if ldsNumBytes > 32768 or \
          state["ProblemType"]["ComputeDataType"].numBytes() * state["MacroTile0"] * state["MacroTile1"] > 32768*4:
        state["NumElementsPerBatchStore"] = 0
        state["StorePriorityOpt"] = False
        state["StoreSyncOpt"] = 0
        state["GroupLoadStore"] = False
      else:
        state["NumElementsPerBatchStore"] = 16 if not state["ProblemType"]["DataType"].numBytes() == 8 else 1

    # Mbsk prefetch optimization
    if state["_GlobalAccumulation"] != 'MultipleBufferSingleKernel' and state["AdaptiveGemmGSUA"] == 0:
        state["MbskPrefetchMethod"] = 0
    elif state["MbskPrefetchMethod"] == -1:
      numStoreElements = state["NumElementsPerThread"] // state["StoreVectorWidth"]
      state["MbskPrefetchMethod"] = 1 if numStoreElements >= 4 else 0
    if state["MbskPrefetchMethod"] == 1:
      state["NumMbskPrefetchElements"] = 16
      storeRegs = state["StoreVectorWidth"] * state["ProblemType"]["ComputeDataType"].numRegisters()
      # exceed 16*4 = 64 VPGRs
      if storeRegs > 4:
        state["NumMbskPrefetchElements"] //= storeRegs // 4
      if state["NumElementsPerBatchStore"] == 0 or state["NumElementsPerBatchStore"] > state["NumMbskPrefetchElements"]:
          state["NumElementsPerBatchStore"] = state["NumMbskPrefetchElements"]

    if state["StoreRemapVectorWidth"] == -1:
      # use de_read_b64 as default in storeRemap to avoid bank conflict
      defaultRemap = 8 // int(state["ProblemType"]["DestDataType"].numBytes())
      defaultRemap = max(defaultRemap, state["MacroTile0"]//state["WavefrontSize"])
      if state["EnableMatrixInstruction"]:
        ldsRemapPad = max(defaultRemap, state["MIOutputVectorWidth"])
        ldsNumElementsRemapC = (state["MacroTile0"]+ldsRemapPad)* state["MatrixInstN"] * state["MIWaveGroup"][1]
        computeBytes = int(state["ProblemType"]["ComputeDataType"].numBytes())
        # max(GlobalAccumulation case, non-GlobalAccumulation)
        ldsNumElementsRemapC = max(ldsNumElementsRemapC, ldsNumElementsRemapC * (computeBytes / state["ProblemType"]["DestDataType"].numBytes()))
        ldsSize = ldsNumElementsRemapC * state["ProblemType"]["DestDataType"].numBytes()
        if not math.log(state["MacroTile0"],2).is_integer() or \
            ldsSize > state["MaxLDS"] or \
            state["SourceSwap"] or \
            (state["GlobalSplitU"] > 1 or state["GlobalSplitU"] == -1) and (state["_GlobalAccumulation"] != 'MultipleBuffer') or \
            state["MatrixInstBN"] > 1 and state["MatrixInstN"] == 4 :
          state["StoreRemapVectorWidth"] = 0
        else:
          state["StoreRemapVectorWidth"] = defaultRemap
      else:
        state["StoreRemapVectorWidth"] = defaultRemap

      if not state["SourceSwap"]:
        if not state["StoreRemapVectorWidth"]:
          reject(state, printRejectionReason, "reject to reduce number of kernels")
        elif state["VectorWidthA"] > 1:
          reject(state, printRejectionReason, "reject to reduce number of kernels")

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
        reject(state, printRejectionReason, "SourceSwap not compatible with StoreRemap")
        return
      if state["VectorWidthA"] > 1 or state["VectorWidthB"] > 1:
        reject(state, printRejectionReason, "VW>1 not compatible with StoreRemap")
        return

    # Sparse problem
    if state["ProblemType"]["Sparse"]:
      # if state["PrefetchGlobalRead"] and not state["ExpandPointerSwap"]:
      #   reject(state, printRejectionReason, "Sparse A kernel only support PGR with EPS=1.")
      #   return
      if not isaInfoMap[isa].asmCaps["HasSWMMAC"] and state["EnableMatrixInstruction"] and state["MIArchVgpr"]:
        reject(state, printRejectionReason, "Current ISA does not support MIArchVgpr in Sparse kernels.")
        return
      state["AssertSummationElementMultiple"] = 8

    # check if need to use lds init Acc vgprs
    state["LdsInitCVgprs"] = False
    if isaInfoMap[isa].archCaps["HasAccCD"] and \
         state["EnableMatrixInstruction"] and state["StorePriorityOpt"] and \
         state["ProblemType"]["DataType"].isDouble():
      state["LdsInitCVgprs"] = True

    # force MIArchVgpr when using WMMA
    if state["EnableMatrixInstruction"] and isaInfoMap[isa].asmCaps["HasWMMA"]:
      state["MIArchVgpr"] = True

    if state["MIArchVgpr"]:
      if not state["EnableMatrixInstruction"]:
        reject(state, printRejectionReason, "MIArchVgpr only support for MatrixInstruction")
        return

      if isaInfoMap[isa].asmCaps["HasMFMA"]:
        if not (state["ProblemType"]["ComputeDataType"].isDouble() or \
                state["ProblemType"]["ComputeDataType"].isSingle() or \
                (state["ProblemType"]["ComputeDataType"].isHalf() and state["ProblemType"]["HighPrecisionAccumulate"]) or \
                state["ProblemType"]["ComputeDataType"].isInt32() or \
                state["ProblemType"]["ComputeDataType"].isComplex()):
          reject(state, printRejectionReason, "MIArchVgpr now only support fp64, fp64c, fp32, fp32c, fp16, int8 MatrixInstruction.")
          return

    #check not support cases and calculate lds resources
    ldsNumBytesRemapC = 0
    if state["StoreRemapVectorWidth"]:
      if not state["EnableMatrixInstruction"]:
        reject(state, printRejectionReason, "storeRemap only support MatrixInstruction kernel")
        return
      if ((state["GlobalSplitU"] > 1 or state["GlobalSplitU"] == -1) and (state["_GlobalAccumulation"] != 'MultipleBuffer' or state["_GlobalAccumulation"] == 'MultipleBufferSingleKernel')) or \
        ((state["GlobalSplitU"] == 1 or state["GlobalSplitU"] == -1) and state["_GlobalAccumulation"] == 'SingleBuffer'):
        reject(state, printRejectionReason, "storeRemap doesn't support GlobalSplitU yet, except GSU algorithm 2")
        return
      if packedC0 or packedC1:
        reject(state, printRejectionReason, "storeRemap doesn't support packedC0 and packedC1 yet")
        return
      if state["MatrixInstBN"] > 1 and state["MatrixInstN"] == 4:
        reject(state, printRejectionReason, "storeRemap doesn't support MI4x4 multi blocks in N direction yet")
        return
      if not math.log(state["MacroTile0"],2).is_integer():
        reject(state, printRejectionReason, "storeRemap only supports power-of-2 MT0")
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
        reject(state, printRejectionReason, "StoreRemapVectorWidth %u is not allowed for this data type" % state["StoreRemapVectorWidth"])
        return

      if state["StoreRemapVectorWidth"] * state["WavefrontSize"] < state["MacroTile0"]:
        reject(state, printRejectionReason, "storeRemap: Per wave single global write instruction doesn't enough to write one M column." + \
               " Please use larger StoreRemapVectorWidth.")
        return
      if (state["MacroTile0"]*state["MatrixInstN"])//state["MIWaveGroup"][0] < state["StoreRemapVectorWidth"]*state["WavefrontSize"]:
        reject(state, printRejectionReason, "storeRemap: number elements of lds less than per wave per local read elements." + \
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

      ldsNumBytesRemapCNonGSU = int(ldsNumBytesRemapC * multiplier)
      ldsNumBytesRemapCGSU    = int(ldsNumBytesRemapC * multiplierGSU)
      ldsNumBytesRemapC       = int(ldsNumBytesRemapC * max(multiplier, multiplierGSU))


      #print("ldsNumBytesRemapC=%u" % ldsNumBytesRemapC)

      # if LDS is bound by RemapC (SRVW), then 1LDSBuffer actually doesn't help in SIA3
      # since LDS usage couldn't be reduced
      if state["1LDSBuffer"] and (state["ScheduleIterAlg"] == 3) and (ldsNumBytes < ldsNumBytesRemapC):
        # TODO- Remove this DataType test condition,
        # Currently we do this test is just because we don't want to affect existing logic in rocBLAS
        if state["ProblemType"]["DataType"].isInt8():
          reject(state, printRejectionReason, "LDS usage is bound be StoreRemap, thus 1LDSBuffer wouldn't have any help. Skip.")
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
            epilogueSize = max(epilogueSize, int(state["MacroTile%d"%tile01] * maxKId * dataType.numBytes())) # TODO- GetTurn ?
      else:
        epilogueSize = int(state["NumThreads"] * state["ProblemType"]["ComputeDataType"].numBytes() * vecDT.bias(0).turn)
    # Calculate max ldsNumBytes for other epilogues
    if state["ProblemType"]["UseScaleAlphaVec"]:
      epilogueSize += int(state["NumThreads"] * state["ProblemType"]["ComputeDataType"].numBytes() * vecDT.scaleAlpha(0).turn)
    if state["ProblemType"]["UseScaleAB"] == "Vector":
      epilogueSize += int(state["NumThreads"] * state["ProblemType"]["ComputeDataType"].numBytes() * (vecDT.scaleA.turn + vecDT.scaleB.turn))
    ldsNumBytes = max(ldsNumBytes, state["LdsOffsetBias"] + epilogueSize)

    state["LdsBytesNoAmax"] = ldsNumBytes
    if state["ProblemType"]["OutputAmaxD"]:
      # used in reduce inter wave
      # 4 data * half_wave_num * amax bytePerE
      num_workItems = state["NumThreads"]
      half_wave_size = state["WavefrontSize"] // 2
      amaxBPE = int(state["ProblemType"]["DataTypeAmaxD"].numBytes())
      ldsAmaxDBytes = 4 * (num_workItems // half_wave_size) * amaxBPE
      ldsNumBytes += ldsAmaxDBytes

    state["LdsNumBytes"] = ldsNumBytes
    ldsSize = ldsNumBytes
    if ldsSize > state["MaxLDS"]:
      reject(state, printRejectionReason, "Kernel Uses %u > %u bytes of LDS" % ( ldsSize, state["MaxLDS"]))
      state["ValidDepthU"] = False
      return

    # LoopUnroll  = DepthU / LocalSplitU
    if "LocalSplitU" in state:
      state["LoopUnroll"] = state["DepthU"] // state["LocalSplitU"]
    if state["LoopUnroll"] * state["LocalSplitU"] != state["DepthU"]:
      state["Valid"] = False
    if state["KernelLanguage"] != "Assembly" and state["InnerUnroll"] != 1:
      reject(state, printRejectionReason, "InnerUnroll only supported on assembly")
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
      reject(state, printRejectionReason, "LoopIters need to greater than 0")
      return

    # Since we use PLR >= LoopIters for allocating numberOfIters vgprBuffer for a while
    # we need to support both PLR >= LoopIters and CLR parameter for solutions in rocBLAS
    if state["ClusterLocalRead"] and state["PrefetchLocalRead"] >= state["LoopIters"] and not state["ScheduleIterAlg"] == 2 and not state["ForceUnrollSubIter"]:
      # Reject configuration: DTV enabled on one side is incompatible with PLR = 0
      if state["DirectToVgprA"] ^ state["DirectToVgprB"]:
        reject(state, printRejectionReason, "DirectToVgpr does not work with PrefetchLocalRead(%u) >= LoopIters(%u)"%(state["PrefetchLocalRead"], state["LoopIters"]))
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
      wlrMultiple = state["LocalReadVectorWidthA"]//(state["MIInputPerThreadA"] * (2 if state["ProblemType"]["Sparse"] == 1 else 1))
      # NOTE: wlrmultiple can be 0 for new MFMA
      if not state["ProblemType"]["Sparse"] and not state["UseF32XEmulation"] and not(state["ProblemType"]["DataType"].is8bitFloat() and (state["MatrixInstK"] in [64, 128,])) and (not isaInfoMap[isa].asmCaps["HasWMMA_V3"]):
        if wlrMultiple == 0:
          reject(state, printRejectionReason, "LocalReadVectorWidthA %u is less than MIInputA" % (state["LocalReadVectorWidthA"]))
          return
      # for example, if the original ds_read is b32...
      #   1. if LoopIters = 5 (b32 x 5 times), WLR-Multiple = 2 (b64), then we can fit the WLR
      #   2. if LoopIters = 2 (b32 x 2 times), WLR-Multiple = 4 (b128), this is not allowed
      #   3. if LoopIters = 2 (b32 x 2 times), WLR-Multiple = 2 (b64), this is allowed
      if wlrMultiple and state["LoopIters"] % wlrMultiple != 0:
        reject(state, printRejectionReason, "LocalReadVectorWidthA %u cannot be distributed evenly, LoopIters %u should be divisible by WLR-Multiple %u" \
          % (state["LocalReadVectorWidthA"], state["LoopIters"], wlrMultiple))

      if state["LoopIters"] - (state["PrefetchLocalRead"] * wlrMultiple) < 0 :
        reject(state, printRejectionReason, "with PrefetchLocalRead %u LoopIters %u LocalReadVectorWidthA %u, not enough LoopIters to prefetch %ux%u iterations, " \
          % (state["PrefetchLocalRead"],state["LoopIters"],state["LocalReadVectorWidthA"], state["PrefetchLocalRead"] , wlrMultiple) )

      # Multiple = WLR-size / input-size = how many iters could be covered by one WLR ?
      wlrMultiple = state["LocalReadVectorWidthB"]//(state["MIInputPerThreadB"] * (2 if state["ProblemType"]["Sparse"] == 2 else 1))
      # NOTE: wlrmultiple can be 0 for new MFMA
      if not state["ProblemType"]["Sparse"] and not state["UseF32XEmulation"] and not(state["ProblemType"]["DataType"].is8bitFloat() and (state["MatrixInstK"] in [64, 128,])) and (not isaInfoMap[isa].asmCaps["HasWMMA_V3"]):
        if wlrMultiple == 0:
          reject(state, printRejectionReason, "LocalReadVectorWidthB %u is less than MIInputB" % (state["LocalReadVectorWidthB"]))
          return
      # for example, if the original ds_read is b32...
      #   1. if LoopIters = 5 (b32 x 5 times), WLR-Multiple = 2 (b64), then we can fit the WLR
      #   2. if LoopIters = 2 (b32 x 2 times), WLR-Multiple = 4 (b128), this is not allowed
      #   3. if LoopIters = 2 (b32 x 2 times), WLR-Multiple = 2 (b64), this is allowed
      if wlrMultiple and state["LoopIters"] % wlrMultiple != 0:
        reject(state, printRejectionReason, "LocalReadVectorWidthB %u cannot be distributed evenly, LoopIters %u should be divisible by WLR-Multiple %u" \
          % (state["LocalReadVectorWidthB"], state["LoopIters"], wlrMultiple))

      if state["LoopIters"] - (state["PrefetchLocalRead"] * wlrMultiple) < 0 :
        reject(state, printRejectionReason, "with PrefetchLocalRead %u LoopIters %u LocalReadVectorWidthB %u, not enough LoopIters to prefetch %ux%u iterations, " \
          % (state["PrefetchLocalRead"],state["LoopIters"],state["LocalReadVectorWidthB"], state["PrefetchLocalRead"] , wlrMultiple) )

    # # reject conditions with lower performance
    # if state["ScheduleIterAlg"] == 2 and \
    # (state["ExpandPointerSwap"] != 1 or state["LoopIters"] != 1 or state["ScheduleGlobalRead"] != 1):
    #   reject(state, printRejectionReason, "ScheduleIterAlg 2 only work with EPS1_SGR1, LoopIter=1")

    # reject for PGR>=3
    if state["PrefetchGlobalRead"] >= 3:
      # DTLA + DTLB only
      if not(state["DirectToLdsA"] and state["DirectToLdsB"]):
        reject(state, printRejectionReason, "PrefetchGlobalRead>=3 Supports only DirectToLdsA and DirectToLdsB")
      # does not work with PLR=0
      if state["PrefetchLocalRead"] == 0:
        reject(state, printRejectionReason, "PrefetchGlobalRead>=3 Supports only PrefetchLocalRead >= 1")
      # support SIA==3 only
      if state["ScheduleIterAlg"] != 3:
        reject(state, printRejectionReason, "PrefetchGlobalRead>=3 Supports only ScheduleIterAlg == 3")
      # TODO: enable PGR>=3 for Sparse
      if state["ProblemType"]["Sparse"]:
        reject(state, printRejectionReason, "PrefetchGlobalRead>=3 + Sparse is not supported yet")
    if state["TransposeLDS"] == 1:
      if not state["EnableMatrixInstruction"]:
        reject(state, printRejectionReason, "TransposeLds Supports only in MatrixInstruction=1")
      if state["ProblemType"]["TLUA"] and state["ProblemType"]["TLUB"]:
          # TODO: Now in rocBLAS, lot of logic yamls are Type=NT and TLDS=1? Why aren't they rejected and how to get rid of them?
          reject(state, printRejectionReason, "TransposeLds requires TLUA=0 or TLUB=0")
    if state["EnableMatrixInstruction"]:
      # enable widerLocalRead
      if not state["ProblemType"]["Sparse"]:
        if state["LocalReadVectorWidthA"] > state["MIInputPerThreadA"]:
          # wider localRead support 2 types
          # 1. prefetch all lds to register
          # 2. using larger InnerUnroll
          if not (state["PrefetchLocalRead"] >= state["LoopIters"] and state["InnerUnroll"] == 1) and \
              not state["ClusterLocalRead"] and \
              not state["InnerUnroll"] >= state["LocalReadVectorWidthA"] // state["MIInputPerThreadA"]:
            reject(state, printRejectionReason, "wider localRead only support ClusterLocalRead or (InnerUnroll > WiderLocalReadxN)")
        if state["LocalReadVectorWidthB"] > state["MIInputPerThreadB"]:
          # wider localRead support 2 types
          # 1. prefetch all lds to register
          # 2. using larger InnerUnroll
          if not (state["PrefetchLocalRead"] >= state["LoopIters"] and state["InnerUnroll"] == 1) and \
              not state["ClusterLocalRead"] and \
              not state["InnerUnroll"] >= state["LocalReadVectorWidthB"] // state["MIInputPerThreadB"]:
            reject(state, printRejectionReason, "wider localRead only support ClusterLocalRead or (InnerUnroll > WiderLocalReadxN)")
      else:
        if state["LocalReadVectorWidth"] > state["MIInputPerThread"]:
          # wider localRead support 2 types
          # 1. prefetch all lds to register
          # 2. using larger InnerUnroll
          if not (state["PrefetchLocalRead"] >= state["LoopIters"] and state["InnerUnroll"] == 1) and \
              not state["ClusterLocalRead"] and \
              not state["InnerUnroll"] >= state["LocalReadVectorWidth"] // state["MIInputPerThread"]:
            reject(state, printRejectionReason, "wider localRead only support ClusterLocalRead or (InnerUnroll > WiderLocalReadxN)")

    if state["GlobalReadPerMfma"] > 1 and state["PrefetchGlobalRead"] >= 2:
      reject(state, printRejectionReason, "GlobalReadPerMfma need to be 1 if PGR>=2")

    if state["UseInstOffsetForGRO"] == -1:
      state["UseInstOffsetForGRO"] = 1 if state["DirectToLds"] else 0

    state["ULSGRODoubleG2L"] = 0
    if state["UnrollLoopSwapGlobalReadOrder"] == 1:
      bpeA       = state["ProblemType"]["DataType"].numBytes()
      bpr         = 4
      numVgprG2LA = roundUp((state["NumLoadsCoalescedA"] * state["NumLoadsPerpendicularA"] * \
        state["GlobalReadVectorWidthA"] * bpeA) / (float)(bpr))
      numVgprG2LB = roundUp((state["NumLoadsCoalescedB"] * state["NumLoadsPerpendicularB"] * \
        state["GlobalReadVectorWidthB"] * bpeA) / (float)(bpr))
      if numVgprG2LA % 2 == 1 or numVgprG2LB % 2 == 1:
        reject(state, printRejectionReason, "G2LA/B vgpr has bubble inside. Cannot use UnrollLoopSwapGlobalReadOrder=1.")
      if state["GlobalReadVectorWidthA"] != state["GlobalReadVectorWidthB"]:
        # TODO: Add a configuration to schedule better.
        state["ULSGRODoubleG2L"] = 1
      minGRVW = min(state["GlobalReadVectorWidthA"], state["GlobalReadVectorWidthB"])
      if minGRVW * bpeA < 4:
        # G2LA/B vgpr index will jump.
        state["ULSGRODoubleG2L"] = 1
      if state["ExpandPointerSwap"] == 1:
        reject(state, printRejectionReason, "ExpandPointerSwap need to be 0 if UnrollLoopSwapGlobalReadOrder")
      if state["PrefetchGlobalRead"] < 2:
        reject(state, printRejectionReason, "PrefetchGlobalRead need to be >=2 if UnrollLoopSwapGlobalReadOrder")
      if state["ProblemType"]["DataTypeA"].numBytes() != state["ProblemType"]["DataTypeB"].numBytes():
        reject(state, printRejectionReason, "UnrollLoopSwapGlobalReadOrder doesn't support mixed precision.")
      if state["ProblemType"]["MXBlockA"] and (not state["DirectToLdsMXSA"]) or state["ProblemType"]["MXBlockB"] and (not state["DirectToLdsMXSB"]):
        reject(state, printRejectionReason, "UnrollLoopSwapGlobalReadOrder doesn't support MX + non DTL")

    if state["ExpandPointerSwap"] == 1 and state["LDSTrInst"]:
      reject(state, printRejectionReason, "LDSTrInst + ExpandPointerSwap not supported")

    # guard against out of bounds reads
    # None: don't guard against ou
    # ShiftPtr: shift read pointers to be in bounds, then unshift registers (source & assembly),
    # ShiftPtr does not support very small problem dims < global load vector width since the shift
    # would move outside the array bounds.
    # If GLVW==1 or Assert*ElementMultiple for the coalesced dim is > GRVW, then shifting is not
    # necessary and the shift/unshift code will not be generated
    state["EdgeType"] = "ShiftPtr" # Use ShiftPtr by default
    if state["enableTDMA"] and state["enableTDMB"]:
      state["EdgeType"] = "None"

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
        #reject(state, printRejectionReason, "PBC with wide load has insufficient overlap guarantees- try GRVW=1 or adding appropriate Assert*ElementMultiple")




    if state["EnableMatrixInstruction"]:
      cont1 = not state["GuaranteeNoPartialB"]
      cont2 = ((state["MatrixInstN"] % state["GlobalReadVectorWidthB"]) != 0)
      if cont1 and cont2:
        reject(state, printRejectionReason, "MatrixInstN %u %% GlobalReadVectorWidthB %u must be 0" % \
          (state["MatrixInstN"], state["GlobalReadVectorWidthB"]))
    else: # mac
      # if not bufferLoad or not state["GuaranteeNoPartialA"]:
      # Restrict GRVW/VW combos so shift-ptr logic will work
      if state["GlobalReadVectorWidthA"] > 1 \
          and state["GlobalReadVectorWidthA"] != state["VectorWidthA"]:
          reject(state, printRejectionReason, "GlobalReadVectorWidthA %u must be == VectorWidthA %u or == 1" % \
                  (state["GlobalReadVectorWidthA"], state["VectorWidthA"]))
      if state["GlobalReadVectorWidthB"] > 1 \
          and state["GlobalReadVectorWidthB"] != state["VectorWidthB"]:
          reject(state, printRejectionReason, "GlobalReadVectorWidthB %u must be == VectorWidthB %u or == 1" % \
                  (state["GlobalReadVectorWidthB"], state["VectorWidthB"]))

    # Use SGPR to store an offset from GlobalReadOffsetA+0.
    # (as opposed to using dedicated VGPR for each GRO
    # Requires preciseBounds check since we rely on the buffer bounds check, not
    # individual vector registers doing bounds compares.

    if state["_UseSgprForGRO"] == 1 and (state["ProblemType"]["SwizzleTensorA"] or state["ProblemType"]["SwizzleTensorB"]):
      reject(state, printRejectionReason, "UseSgprForGRO for Swizzle is not supported")

    if state["_UseSgprForGRO"] == -1:
      # Don't use SGPR if it looks like we might not have enough - better to leave PBC enabled even if we have to use VGPR
      # 40 is based on current SGPR usage, this may need to be tuned in the future:
      numLoadsA = state["NumLoadsCoalescedA"]*state["NumLoadsPerpendicularA"]
      numLoadsB = state["NumLoadsCoalescedB"]*state["NumLoadsPerpendicularB"]
      numLoadMXSA = state["NumLoadsCoalescedMXSA"]*state["NumLoadsPerpendicularMXSA"] if state["ProblemType"]["MXBlockA"] else 0
      numLoadMXSB = state["NumLoadsCoalescedMXSB"]*state["NumLoadsPerpendicularMXSB"] if state["ProblemType"]["MXBlockB"] else 0
      numLoadsM = 0
      if state["ProblemType"]["Sparse"] and not state["DirectToVgprSparseMetadata"]:
        numLoadsM = state["NumLoadsCoalescedMetadata"]*state["NumLoadsPerpendicularMetadata"]
      if (numLoadsA + numLoadsB + numLoadsM + numLoadMXSA + numLoadMXSB  > 35) or state["ProblemType"]["SwizzleTensorA"] or state["ProblemType"]["SwizzleTensorB"]:
        #print "info: Disabling UseSgprForGRO since predicting too many SGPR will be used"
        state["_UseSgprForGRO"] = 0
      else:
        state["_UseSgprForGRO"] = 1

    if packedC0 and not state["GuaranteeNoPartialA"]:
      reject(state, printRejectionReason, "packedC0 requires GuaranteeNoPartialA")
    if packedC1 and not state["GuaranteeNoPartialB"]:
      reject(state, printRejectionReason, "packedC1 requires GuaranteeNoPartialB")

    if packedC0 or packedC1:
      state["_UseSgprForGRO"] = 0

      if state["EdgeType"] != "ShiftPtr":
        reject(state, printRejectionReason, "Packed dims requires EdgeType==ShiftPtr")
      if state["KernelLanguage"] == "Assembly":
        if not bufferLoad:
          reject(state, printRejectionReason, "Packed dims for Assembly requires BufferLoad")

    if packedC0: # VectorWidth must not span tensor dim
      if state["KernelLanguage"] == "Source":
        if state["AssertFree0ElementMultiple"]<state["VectorWidthA"]:
          reject(state, printRejectionReason, "packedC0 Source requires AF0EM>=VectorWidth (for loads and stores)")
      else:
        if state["AssertFree0ElementMultiple"]<state["VectorWidthA"]\
          or state["AssertFree0ElementMultiple"] == 1:
            if state["VectorStore"] <= 0:
              state["_VectorStore"] = 0
            else:
              reject(state, printRejectionReason, "packedC0 Assembly requires AF0EM>=VectorWidth or not VectorStore (for stores)")

    state["AssignedDerivedParameters"] = True

    # Set E
    if state["ProblemType"]["UseE"]:
      if (state["_GlobalAccumulation"] == 'SingleBuffer') and (state["GlobalSplitU"] > 1 or state["GlobalSplitU"] == -1):
        reject(state, printRejectionReason, "GlobalSplitU > 1 only compatible with MultipleBuffer")
      if len(state["PackedC1IndicesX"]) > 1:
        reject(state, printRejectionReason, "Use E does not support len(PackedC1IndicesX) > 1.")
      if not state["BufferStore"]:
        reject(state, printRejectionReason, "Use E only supports BufferStore due to no suppress no store.")
      if state["StoreRemapVectorWidth"] and (state["GlobalSplitU"] == 1 or state["GlobalSplitU"] == -1):
        reject(state, printRejectionReason, "Use E does not support StoreRemapVectorWidth if GSU == 1.")
      if state["GroupLoadStore"]:
        reject(state, printRejectionReason, "Use E does not support GroupLoadStore.")

    # Activation
    # Function call is set to false if GSU != 1 or Activation is not fused or ActivationType is not All.
    if not (state["ActivationFused"] and state["ProblemType"]["ActivationType"] in ['all', 'hipblaslt_all']) \
      and state["ActivationFuncCall"]:
      state["ActivationFuncCall"] = False

    if state["ActivationAlt"]:
      reject(state, printRejectionReason, "Currently does not accept ActivationAlt.")

    # Bias reduction
    if state["ProblemType"]["UseBias"] and state["ProblemType"]["Gradient"]:
      if (state["_GlobalAccumulation"] == 'SingleBuffer') and (state["GlobalSplitU"] > 1 or state["GlobalSplitU"] == -1):
        reject(state, printRejectionReason, "GlobalSplitU > 1 only compatible with MultipleBuffer for bias reduction")
      if len(state["PackedC1IndicesX"]) > 1:
        reject(state, printRejectionReason, "Bias reduction does not support len(PackedC1IndicesX) > 1.")
      if not state["BufferStore"]:
        reject(state, printRejectionReason, "Bias reduction only supports BufferStore due to no suppress no store.")
      if state["StoreRemapVectorWidth"] and (state["GlobalSplitU"] == 1 or state["GlobalSplitU"] == -1):
        reject(state, printRejectionReason, "Bias reduction does not support StoreRemapVectorWidth if GSU == 1.")
      if state["GroupLoadStore"]:
        reject(state, printRejectionReason, "Bias reduction does not support GroupLoadStore.")

    # Bias and ScaleAlphaVec
    if state["ProblemType"]["UseBias"] != 0 and state["ProblemType"]["UseScaleAlphaVec"] != 0 and state["ProblemType"]["UseBias"] != state["ProblemType"]["UseScaleAlphaVec"]:
      reject(state, printRejectionReason, "When both UseBias and UseScaleAlphaVec are enabled then UseBias and UseScaleAlphaVec must have same settings.")

    # ScaleAB or ScaleABVec
    if state["ProblemType"]["DataTypeA"] != state["ProblemType"]["MacDataTypeA"] and \
      state["ProblemType"]["DataTypeB"] != state["ProblemType"]["MacDataTypeB"] and \
      state["ProblemType"]["UseScaleAB"] == "Vector":
      reject(state, "Currently does not support using scaleABVec if DataTypeA != MacDataTypeA and DataTypeB != MacDataTypeB.")

    if state["ProblemType"]["UseScaleAB"] and state["OptNoLoadLoop"]:
      # Hard to check alpha == 1.0 directly
      # Turn off ONLL for now
      # TODO: support ONLL if necessary
      state["OptNoLoadLoop"] = 0

    # if state["GlobalSplitU"] > 1 or state["GlobalSplitU"] == -1:
    #   if state["ProblemType"]["SupportUserArgs"] and state["_GlobalAccumulation"] != 'MultipleBufferSingleKernel':
    #     reject(state, printRejectionReason, "Currently SupportUserArgs does not support GSU > 1.")

    if state["_GlobalAccumulation"] == 'MultipleBufferSingleKernel':
      if state["NumElementsPerBatchStore"] == 1:
        reject(state, printRejectionReason, "too many store at MultipleBufferSingleKernel direct reject")
      if state["ProblemType"]["UseScaleCD"]:
        reject(state, printRejectionReason, "MultipleBufferSingleKernel not support UseScaleCD yet")
      if state["ProblemType"]["UseE"]:
        reject(state, printRejectionReason, "MultipleBufferSingleKernel not support UseE yet")
      if state["ProblemType"]["BiasSrc"] != "D":
        reject(state, printRejectionReason, "MultipleBufferSingleKernel not support BiasSrc not D yet")
      if state["ProblemType"]["DataType"].isDouble():
        reject(state, printRejectionReason, "MultipleBufferSingleKernel not support " + str(state["ProblemType"]["DataType"])  + " yet")

    #Need to force disabling PreloadKernArgs if compiler does not support
    #Can not just reject the solution since the user library may find any solutions
    if state["PreloadKernArgs"]:
      if (rocmVersion.major < 6 or (rocmVersion.major == 6 and rocmVersion.patch < 32650)) or \
          not (isa == (9, 0, 10) or isa[:2] == (9, 4) or isa == (9, 5, 0)):
        #print("Force to Disable PreloadKernArgs since this hipcc version doesn't support",)
        state["PreloadKernArgs"] = False

    # change negative ExtraLatencyForLR to 0 for non DirectToVgpr
    if state["ExtraLatencyForLR"] < 0 and not (state["DirectToVgprA"] or state["DirectToVgprB"]):
      state["ExtraLatencyForLR"] = 0

  ########################################
  @ staticmethod
  def getParametersIndented(state, indent):
    s = ""
    s += "%sProblemType: %s\n" % (indent, str(state["ProblemType"]))
    for key in sorted(state):
      s += "%s%s: %s\n" % (indent, str(key), str(state[key]))
    return s


  class NonprimitiveParameterValueException(Exception):
    pass


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
      self._name = getSolutionNameFull(self._state, self.splitGSU)
    return self._name

  def __repr__(self):
    return self.__str__()

  def getAttributes(self):
    return self._state

  def __hash__(self):
    deviceNames = str(self._state.get("DeviceNames", ""))
    return hash(str(self) + self._state.get("codeObjectFile", "") + deviceNames)

  def __eq__(self, other):
    if not isinstance(other, Solution):
      return False
    if str(self) != str(other):
      return False
    # Check DeviceNames to prevent deduplication of solutions for different devices
    selfDeviceNames = self._state.get("DeviceNames", None)
    otherDeviceNames = other._state.get("DeviceNames", None)
    return selfDeviceNames == otherDeviceNames

  def __ne__(self, other):
    result = self.__eq__(other)
    if result is NotImplemented:
      return result
    return not result
