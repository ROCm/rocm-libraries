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
from functools import lru_cache

from Tensile.Common.Constants import MAX_FILENAME_LENGTH
from Tensile.Common.RequiredParameters import getRequiredParametersMin, getRequiredParametersFull

from .Problem import ProblemType


# Parameters that are "internal args" — runtime dispatch parameters that don't
# affect the generated kernel assembly. Two kernels differing only in these
# fields compile to identical code objects.
_INTERNAL_ARGS = (
    "WorkGroupMapping",
    "WorkGroupMappingXCC",
    "WorkGroupMappingXCCGroup",
    "StaggerU",
    "StaggerUStride",
    "StaggerUMapping",
    "GlobalSplitUCoalesced",
    "GlobalSplitUWorkGroupMappingRoundRobin",
    "SFCWGM",
)

def getKeyNoInternalArgs(state, splitGSU: bool) -> str:
  """Return a string key that identifies a kernel ignoring internal args.

  Internal args (WorkGroupMapping, StaggerU, etc.) are runtime dispatch
  parameters — they don't change the generated assembly. This function
  produces a canonical key where those parameters are masked to "M" and
  GroupedGemm is forced to False, so that kernels differing only in
  internal args map to the same key.

  Used to:
    - Deduplicate kernels before code generation (BenchmarkProblems.py,
      Run.py:getUniqueKernels) — avoids compiling the same assembly twice.
    - Identify invalid kernels after compilation and propagate failures
      to all solutions sharing that kernel (Run.py:removeInvalidSolutionsAndKernels).
    - Build kernel-to-solution mappings for post-processing
      (Run.py:passPostKernelInfoToSolution).
  """
  # Work on the raw dict to avoid Solution.__setitem__ invalidating _name cache
  s = state._state if hasattr(state, '_state') else state
  pt = s["ProblemType"]

  # Save originals
  backups = {k: s[k] for k in _INTERNAL_ARGS}
  gsu_backup = s["GlobalSplitU"]
  gg_backup = pt["GroupedGemm"]

  # Mask internal args
  pt["GroupedGemm"] = False
  if splitGSU:
    s["GlobalSplitU"] = "M" if (gsu_backup > 1 or gsu_backup == -1) else gsu_backup
  elif gsu_backup != 0:
    s["GlobalSplitU"] = "M"
  for k in _INTERNAL_ARGS:
    s[k] = "M"

  # Compute string key (same as what str(deep_copied_solution) would produce)
  key = _getName(s, getRequiredParametersFull(), splitGSU, False)

  # Restore
  pt["GroupedGemm"] = gg_backup
  s["GlobalSplitU"] = gsu_backup
  for k in _INTERNAL_ARGS:
    s[k] = backups[k]

  # Include codeObjectFile and DeviceNames in the key to prevent
  # over-deduplication across different code object files / devices.
  # The old code returned a Solution object whose __hash__ included these
  # fields, so kernels targeting different .co files were kept separate.
  cof = s.get("codeObjectFile", "")
  dn = str(s.get("DeviceNames", ""))
  return key + cof + dn


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

  if "CustomKernel" in state and state["CustomKernel"]["name"]:
    return state["CustomKernel"]["name"]
  if state.get("CustomKernelName", ""):
    return state["CustomKernelName"]

  gsuBackup = state["GlobalSplitU"]
  ggBackup = state["ProblemType"]["GroupedGemm"]

  if ignoreInternalArgs:
    state["ProblemType"]["GroupedGemm"] = False
    if splitGSU:
      state["GlobalSplitU"] = "M" if (state["GlobalSplitU"] > 1 or state["GlobalSplitU"] == -1) else state["GlobalSplitU"]

  requiredParametersTemp = set(requiredParameters.union(["GlobalSplitU"]))

  if ignoreInternalArgs:
    if state["GlobalSplitU"] > 0 or state["GlobalSplitU"] == -1:
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
  pt = state["ProblemType"]
  if isinstance(pt, ProblemType):
    components = [str(pt)]
  else:
    components = [str(ProblemType(pt, printIndexAssignmentInfo=False))]

  if "MacroTile0" in state \
      and "MacroTile1" in state \
      and "DepthU" in state:
    components.append(f'{getParameterNameAbbreviation("MacroTile")}{state["MacroTile0"]}x{state["MacroTile1"]}x{state["DepthU"]}')

  if "MatrixInstM" in state:
    components.append(f'{getParameterNameAbbreviation("MatrixInstruction")}{state["MatrixInstM"]}x{state["MatrixInstN"]}x{state["MatrixInstB"]}')
    requiredParametersTemp.add("MIWaveTile")
  else:
    requiredParametersTemp.add("ThreadTile")

  if state["UseCustomMainLoopSchedule"]:
    components.append('CMS')

  components.append('SN')

  # Skip SFA tag if using default wgm algo
  if "SpaceFillingAlgo" in requiredParametersTemp and len(state["SpaceFillingAlgo"]) == 0:
    requiredParametersTemp.discard("SpaceFillingAlgo")

  for key in sorted(requiredParametersTemp):
    if key not in state or key == "CustomKernel":
      continue
    components.append(f'{getParameterNameAbbreviation(key)}{getParameterValueAbbreviation(key, state[key])}')

  state["GlobalSplitU"] = gsuBackup
  state["ProblemType"]["GroupedGemm"] = ggBackup

  return '_'.join(components)


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


def getKernelFileBase(splitGSU: bool, kernel):
  if "CustomKernel" in kernel and kernel["CustomKernel"]["name"] \
      and not kernel["CustomKernel"].get("generated", False):
    fileBase = kernel["CustomKernel"]["name"]
  elif kernel.get("CustomKernelName", ""):
    fileBase = kernel["CustomKernelName"]
  else:
    fileBase = shortenFileBase(splitGSU, kernel)
  return fileBase


def getKernelNameMin(kernel, splitGSU: bool):
  return _getName(kernel, getRequiredParametersMin(), splitGSU, True)


def getSolutionNameMin(solution, splitGSU: bool):
  return _getName(solution, getRequiredParametersMin(), splitGSU, False)


def getSolutionNameFull(state, splitGSU: bool):
  return _getName(state, getRequiredParametersFull(), splitGSU, False)
