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
from functools import lru_cache
from typing import List

from Tensile.Common.Constants import MAX_FILENAME_LENGTH
from Tensile.Common.ValidParameters import validParameters
from Tensile.Common.RequiredParameters import getRequiredParametersMin, getRequiredParametersFull

from .Problem import ProblemType


def getKeyNoInternalArgs(state, splitGSU: bool):
  state_copy = deepcopy(state)
  state_copy["ProblemType"]["GroupedGemm"] = False
  if splitGSU:
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


@lru_cache(maxsize=None)
def getParameterNameAbbreviation( name: str ):
  return ''.join(c for c in name if c.isupper())


@ lru_cache(maxsize=None)
def getPrimitiveParameterValueAbbreviation(key, value):
  if isinstance(value, str):
    return getParameterNameAbbreviation(value)
  elif isinstance(value, bool):
    return "1" if value else "0"
  elif isinstance(value, int):
    if value >= 0:
      return "%u" % value
    else: # -1 -> n1
      return "n%01u" % abs(value)
  elif isinstance(value, ProblemType): # will need to deal with this
    return str(value)
  elif isinstance(value, float):
    val1 = int(value)
    val2 = int(round(value*100)) - int(value)*100
    if val2 > 0:
      s =  "%dp%s" % (val1,str(val2).zfill(2))
    else:
      s = "%d" % (val1)
    return s


def getParameterValueAbbreviation(key, value):
  if key == "ISA":
    return f"{value[0]}{value[1]}{value[2]:x}"
  compositieTypes = (dict, list, tuple,)
  if not isinstance(value, compositieTypes):
    return getPrimitiveParameterValueAbbreviation(key, value)
  elif isinstance(value, tuple):
    return ''.join(str(v) for v in value)
  elif isinstance(value, list):
    return '_'.join(getParameterValueAbbreviation(key, v) for v in value)
  elif isinstance(value, dict):
    return "_".join(f"{pos:d}{k:d}" for pos,k in value.items())
  else:
    raise Exception(f"Parameter {key}={value} is new object type ({type(value)})")


def _getName(state, requiredParameters: frozenset, splitGSU: bool, ignoreInternalArgs):

  if "CustomKernelName" in state and state["CustomKernelName"]:
    return state["CustomKernelName"]

  gsuBackup = state["GlobalSplitU"]
  ggBackup = state["ProblemType"]["GroupedGemm"]

  if ignoreInternalArgs:
    state["ProblemType"]["GroupedGemm"] = False
    if splitGSU:
      state["GlobalSplitU"] = "M" if (state["GlobalSplitU"] > 1) else state["GlobalSplitU"]


  requiredParametersTemp = set(requiredParameters.union(["GlobalSplitU"]))

  if ignoreInternalArgs:
    if state["GlobalSplitU"] > 0:
      requiredParametersTemp.discard("GlobalSplitU")
  else:
    requiredParametersTemp = requiredParametersTemp.union(["WorkGroupMapping",
                                                           "WorkGroupMappingXCC",
                                                           "WorkGroupMappingXCCGroup",
                                                           "StaggerU",
                                                           "StaggerUStride",
                                                           "StaggerUMapping",
                                                           "GlobalSplitUCoalesced",
                                                           "GlobalSplitUWorkGroupMappingRoundRobin"])
  components = [f'{str(state["ProblemType"])}']

  if "MacroTile0" in state \
      and "MacroTile1" in state \
      and "DepthU" in state:
    components.append(f'{getParameterNameAbbreviation("MacroTile")}{state["MacroTile0"]}x{state["MacroTile1"]}x{state["DepthU"]}')

  if "MatrixInstM" in state:
    components.append(f'{getParameterNameAbbreviation("MatrixInstruction")}{state["MatrixInstM"]}x{state["MatrixInstN"]}x{state["MatrixInstB"]}')
    requiredParametersTemp.add("MIWaveTile")
  else:
    requiredParametersTemp.add("ThreadTile")

  components.append('SN')
  for key in sorted(state.keys()):
    if key[0] != '_' and key != "CustomKernelName" and key in requiredParametersTemp:
        components.append(f'{getParameterNameAbbreviation(key)}{getParameterValueAbbreviation(key, state[key])}')

  state["GlobalSplitU"] = gsuBackup
  state["ProblemType"]["GroupedGemm"] = ggBackup

  return '_'.join(components)


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
  name = "%s%0*u" % ("S" if hasattr(state, "_state") else "K", \
      numDigits, serial)
  return name


def shortenFileBase(splitGSU, kernel):
  base = getKernelNameMin(kernel, splitGSU)
  if len(base) <= MAX_FILENAME_LENGTH:
    return base
  import hashlib
  import base64
  pivot = MAX_FILENAME_LENGTH * 3 // 4
  firstPart = base[:pivot]
  secondPart = base[pivot:]
  secondHash = hashlib.sha256(secondPart.encode()).digest()
  secondPart = base64.b64encode(secondHash, b'_-').decode()
  return firstPart + secondPart


def getKernelFileBase(useShortNames: bool, splitGSU: bool, kernelSerialNaming, kernel):
  if "CustomKernelName" in kernel and kernel["CustomKernelName"]:
    fileBase = kernel["CustomKernelName"]
  elif useShortNames:
    fileBase = getNameSerial(kernel, kernelSerialNaming)
  else:
    fileBase = shortenFileBase(splitGSU, kernel)
  return fileBase


def getKernelNameMin(kernel, splitGSU: bool):
  return _getName(kernel, getRequiredParametersMin(), splitGSU, True)


def getSolutionNameMin(solution, splitGSU: bool):
  return _getName(solution, getRequiredParametersMin(), splitGSU, False)


def getSolutionNameFull(state, splitGSU: bool):
  return _getName(state, getRequiredParametersFull(), splitGSU, False)

