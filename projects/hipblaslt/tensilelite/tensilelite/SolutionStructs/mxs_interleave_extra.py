# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Does MXScaleVgprInterleaveExtraVgpr change a gfx1250 MX kernel?

The parameter only matters when neither the stacked nor the cross-group shared conflict-free
layout of the MXSA/MXSB scale VGPRs fits the default span (step 3 of
KernelWriter._mxsInterleaveRegion). Elsewhere True and False give the same assembly under two
different kernel names, so Solution folds True into False there.

This module answers the question from the Solution state alone: it rebuilds the scale geometry
KernelWriter derives (block sizes, buffer count, local-read width) and runs the same planner
(KernelWriter._mxsInterleavePlan) on it. Anything it cannot reproduce exactly returns None and the
caller keeps True.
"""

from math import ceil, floor
from types import SimpleNamespace
from typing import Optional

from ..Common.MxScaleLayout import mxFreeTile, mxReadVectorWidth

# (blockWidth, numOffsets, numAddresses) of the LocalRead MemoryInstruction table in
# KernelWriter.initKernel, in selection order. KernelWriter asserts that its own table matches.
LOCAL_READ_SELECTION_TABLE = (
  (6, 1, 1), (4, 1, 1), (2, 2, 1), (2, 1, 1), (1, 2, 1), (1, 1, 1), (0.5, 1, 1), (0.25, 1, 1),
)

# Plain Solution-state conditions under which the interleave plan does not exist or the register
# layout differs from the one rebuilt here: the caller keeps True.
_UNDECIDABLE_IF_SET = (
  "ReuseAcrossPersistent", "DirectToVgprA", "DirectToVgprB", "DirectToVgprMXSA", "DirectToVgprMXSB",
  "UseSubtileImpl", "ForceUnrollSubIter",
)


def _selectBlockWidth(width: float) -> Optional[float]:
  """KernelWriterAssembly.findMemoryInstructionForWidthStride with combine=False."""
  for blockWidth, numOffsets, numAddresses in LOCAL_READ_SELECTION_TABLE:
    if width < blockWidth or width / blockWidth != floor(width / blockWidth):
      continue
    if numOffsets > 1 or numAddresses > 1:
      continue
    return blockWidth
  return None


def extraVgprMatters(state, isaInfo, bpr: int = 4) -> Optional[bool]:
  """True: ExtraVgpr changes the layout (step 3). False: it does not. None: cannot tell."""
  # Imported here: KernelWriter imports Solution.
  from ..KernelWriter import KernelWriter
  from ..Components.LocalRead import LocalReadMFMA

  pt = state["ProblemType"]
  asmCaps, archCaps = isaInfo.asmCaps, isaInfo.archCaps
  if not (state["ISA"][:2] == (12, 5) and pt["MXBlockA"] and pt["MXBlockB"]):
    return None
  if any(state[k] for k in _UNDECIDABLE_IF_SET):
    return None
  if state["MXScaleFormat"] != "InMemorySwizzle" or not asmCaps["HasWMMA_V3"]:
    return None

  numBuffer = state["LoopIters"] if state["ClusterLocalRead"] else state["PrefetchLocalRead"] + 1
  perBlock, numValu, lrWidth, lrvwTile = {}, {}, {}, {}
  for tc in "AB":
    mxs = "MXS" + tc
    miWaveTile, miInput = state["MIWaveTile" + mxs], state["MIInputPerThread" + mxs]
    n = miWaveTile * ceil(miInput / bpr)
    if mxFreeTile(state, mxs) > 1:
      n = max(1, miWaveTile // max(1, int(state["VectorWidth" + mxs]))) * ceil(miInput / bpr)
    if n == 0:
      n = miWaveTile
    tile01 = 1 if pt["Index01" + tc] else 0
    if n > 0 and LocalReadMFMA.getMxsTileSpanInfo(state, mxs, tile01, asmCaps) is not None:
      n //= 2
    perBlock[tc] = n
    numValu[tc] = n * numBuffer * state["InnerUnroll"]
    lrvwTile[tc] = state["VectorWidth" + tc]

    # Local-read width of an MX scale (one byte per element), as initLocalReadMemoryInstruction.
    if state["UnrollMajorLDS" + mxs]:
      vectorWidth = state["VectorWidthB"]
      if tc == "A":
        vectorWidth = mxReadVectorWidth(state, mxs) if state.get("WaveTileContiguousA") else state["VectorWidthA"]
      width = state["LocalReadVectorWidthMXS"] * vectorWidth / bpr
    else:
      width = lrvwTile[tc] / bpr
    blockWidth = _selectBlockWidth(width)
    if blockWidth is None:
      return None
    lrWidth[tc] = max(1, int(ceil(blockWidth)))

  states = SimpleNamespace(
    numVgprBuffer=numBuffer, lrvwTileMXSA=lrvwTile["A"], lrvwTileMXSB=lrvwTile["B"], archCaps=archCaps,
    mxsa=SimpleNamespace(numVgprValu=numValu["A"], numVgprValuPerBlock=perBlock["A"]),
    mxsb=SimpleNamespace(numVgprValu=numValu["B"], numVgprValuPerBlock=perBlock["B"]))
  writer = SimpleNamespace(states=states)
  writer._mxsDefaultBlocks = lambda kernel, tc: KernelWriter._mxsDefaultBlocks(writer, kernel, tc)
  writer._alignMXSB = lambda kernel, idx: KernelWriter._alignMXSB(writer, kernel, idx)
  tensor = lambda width: {"localReadInstruction": SimpleNamespace(blockWidth=width)}
  tPA, tPB = tensor(lrWidth["A"]), tensor(lrWidth["B"])

  plan, _ = KernelWriter._mxsInterleavePlan(writer, state, tPA, tPB)
  if plan is None:
    return None
  # The MXS region is the first static allocation (vgprIdx = 0 with MX scales on both sides).
  defaultLen = writer._alignMXSB(state, numValu["A"]) + numValu["B"]
  needed = sum(g["span"] for g in plan["groups"])
  if needed > defaultLen:
    shared, _ = KernelWriter._mxsInterleavePlan(writer, state, tPA, tPB, shareGroups=True)
    if shared is not None:
      needed = min(needed, sum(g["span"] for g in shared["groups"]))

  # Conditions under which _mxsInterleaveRegion drops the interleave altogether (the MXS region is the
  # first static allocation, so it starts at v0): with PGR=0 and no PAP the shared region cannot hold the
  # MXSA G2L, and a region past v255 cannot be addressed by the MXSA scale operand.
  prefetchAcross = bool(state.get("_PrefetchAcrossPersistentEnabled", state.get("PrefetchAcrossPersistent", 0)))
  keepsDefault = (not state["PrefetchGlobalRead"]) and not prefetchAcross
  appliesFalse = not keepsDefault and defaultLen <= 256
  appliesTrue = not keepsDefault and max(needed, defaultLen) <= 256
  if appliesFalse != appliesTrue:
    return True   # one of them keeps the default layout, the other interleaves
  if not appliesFalse:
    return False  # both keep the default layout
  return needed > defaultLen
