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

from .Problem import ProblemType

########################################
# create a dictionary with booleans on whether to include parameter in name
def getMinNaming(objs: list):
  nonCKObjs = [obj for obj in objs if not ("CustomKernelName" in obj and obj["CustomKernelName"])]
  # early return
  if len(nonCKObjs) == 0:
    return {}
  # determine keys
  requiredParameters = {}
  if hasattr(nonCKObjs[0], "_state"):
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


def getNameFull(state, splitGSU: bool):
  requiredParameters = {}
  for key in state:
    if key in list(validParameters.keys()):
      requiredParameters[key] = True
  if "MatrixInstM" in state:
    # Use MIWaveGroup and MIWaveTile instead of WG and MT
    requiredParameters["MIWaveTile"]  = True
    requiredParameters["ThreadTile"]  = False
  return getNameMin(state, requiredParameters, splitGSU)


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


def getNameMin(state, requiredParameters, splitGSU: bool, ignoreInternalArgs = False):
  if "CustomKernelName" in state and state["CustomKernelName"]:
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
    components.append(f'{getParameterNameAbbreviation("MacroTile")}{state["MacroTile0"]}x{state["MacroTile1"]}x{state["DepthU"]}')
  if "MatrixInstM" in state:
    components.append(f'{getParameterNameAbbreviation("MatrixInstruction")}{state["MatrixInstM"]}x{state["MatrixInstN"]}x{state["MatrixInstB"]}')
  backup = state["GlobalSplitU"]
  if ignoreInternalArgs:
    if splitGSU:
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
        components.append(f'{getParameterNameAbbreviation(key)}{getParameterValueAbbreviation(key, state[key])}')
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


def shortenFileBase(kernelMinNaming, splitGSU, kernel):
  base = getKernelName(kernelMinNaming, splitGSU, kernel)
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


def getKernelFileBase(useShortNames: bool, splitGSU: bool, kernelMinNaming, kernelSerialNaming, kernel):
  if "CustomKernelName" in kernel and kernel["CustomKernelName"]:
    fileBase = kernel["CustomKernelName"]
  elif useShortNames:
    fileBase = getNameSerial(kernel, kernelSerialNaming)
  else:
    fileBase = shortenFileBase(kernelMinNaming, splitGSU, kernel)
  return fileBase


def getKernelName(kernelMinNaming, splitGSU, kernel):
  kernelName = getNameMin(kernel, kernelMinNaming, splitGSU, True)
  return kernelName
