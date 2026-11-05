################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

from ...ExecutionPolicy import hasStaticAssignment, isPersistent, isStreamKSpatialCluster, \
  requiresPartialReduction
from ..Utilities import reject


def validateTDMStore(state, printRejectionReason: bool) -> bool:
  """Reject TDMStore solutions outside the configurations its epilogue is built for.

  The TDM store block stages the converted D tile in LDS over the main-loop buffers and
  stores it once, after the last store batch, with one tensor_store_from_lds per wave.
  That needs a 16-bit HPA output written directly to D, by a GlobalSplitU=1 kernel or as the
  final store of a non-atomic StreamK=3 tile, no LDS data kept from one persistent tile to the
  next, and MacroTile1 split evenly across the waves. The block also computes the elements
  outside D and bounds only the D store, so besides D the epilogue may only read the bias and
  ScaleAlphaVec vectors along M and the ScaleAB vectors, which go after the staged tile in LDS.
  """
  if not state["TDMStore"]:
    return True

  pt = state["ProblemType"]
  if tuple(state["ISA"])[:2] != (12, 5):
    reason = "TDMStore requires gfx1250"
  elif not (pt["DestDataType"].isBFloat16() or pt["DestDataType"].isHalf()):
    reason = "TDMStore supports only bf16 and fp16 DestDataType"
  elif not pt["HighPrecisionAccumulate"] or not pt["ComputeDataType"].isSingle():
    reason = "TDMStore requires HighPrecisionAccumulate with a single-precision ComputeDataType"
  elif not state["BufferStore"]:
    reason = "TDMStore requires BufferStore"
  elif isPersistent(state) and not (requiresPartialReduction(state) and hasStaticAssignment(state)):
    reason = "TDMStore supports persistent kernels only as non-atomic StreamK=3"
  elif state["PrefetchAcrossPersistent"] or state["ReuseAcrossPersistent"] or isStreamKSpatialCluster(state):
    reason = "TDMStore does not support PrefetchAcrossPersistent, ReuseAcrossPersistent or StreamK spatial clusters"
  elif not isPersistent(state) and state["GlobalSplitU"] != 1:
    reason = "TDMStore requires GlobalSplitU=1"
  elif state["UseSubtileImpl"]:
    reason = "TDMStore does not support UseSubtileImpl"
  elif state["StoreRemapVectorWidth"]:
    reason = "TDMStore does not support StoreRemapVectorWidth"
  elif state["CompactLoopStore"]:
    reason = "TDMStore does not support CompactLoopStore"
  elif pt["FusedGemmA2A"]:
    reason = "TDMStore does not support FusedGemmA2A"
  elif pt["UseE"] or pt["Gradient"] or pt["UseGateResidual"] or pt["UseScaleCD"] or pt["OutputAmaxD"] \
      or pt["GroupedGemm"]:
    reason = "TDMStore does not support E, Gradient, gate residual, ScaleCD, AmaxD or GroupedGemm"
  elif pt["UseBias"] > 1 or pt["UseScaleAlphaVec"] > 1:
    reason = "TDMStore supports bias and ScaleAlphaVec only along M"
  elif state["MacroTile1"] % (state["NumThreads"] // state["WavefrontSize"]) != 0:
    reason = "TDMStore requires MacroTile1 to be a multiple of the number of waves"
  else:
    return True

  reject(state, printRejectionReason, reason)
  return False
