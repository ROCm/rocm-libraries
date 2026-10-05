# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build the LoopModel theta and target facts once per kernel."""

from ...Component import Component
from ...LoopModel.adapter import kernel_to_params, params_to_theta
from ...LoopModel.schedule import build_S


def loopModelTheta(writer, kernel):
  """`(theta, S)` for this kernel, built once and cached on the writer.

  The two consumers run at different times -- the vgpr sizing from `_initKernel` long before the
  Program is built -- so neither owns θ.  `build_S` is pure in θ, so an S derived from the same θ
  is equal by construction.
  """
  key = getattr(writer.states, "kernelName", None) or id(kernel)
  cache = getattr(writer, "_loopModelThetaCache", None)
  if cache is not None and cache[0] == key:
    return cache[1]
  theta = params_to_theta(kernel_to_params(kernel, target=loopModelTarget(writer, kernel)))
  S, _floor = build_S(theta)
  writer._loopModelThetaCache = (key, (theta, S))
  return theta, S


def mxScaleBlockWidth(writer, kernel, tc, lrvw):
  """VGPRs one `ds_read` of `tc` writes, or None when the shape is not covered.

  The authority is the `localReadInstruction` that `initLocalReadMemoryInstruction` selects, and
  neither it nor `tP["MX"]` exists yet: both are assigned in `_initKernel` after θ is cached.
  The width is therefore reconstructed from the Solution keys that decide it, every one of which
  is itself a pure function of the Solution.
  """
  if not lrvw:
    return None
  parent = tc[-1]                                   # "MXSA" -> "A"
  pt = kernel["ProblemType"]
  try:
    bpeDS = int(pt["DataTypeMXS%s" % parent].numBytes())
  except Exception:
    return None                                     # unknown scale type -> no map, not a guess
  if bpeDS != 1:
    # Only E8M0 maps directly to a representable block width. A wider element
    # would require the real pool lookup.
    return None
  caps = getattr(writer.states, "asmCaps", {}) or {}
  if kernel.get("UnrollMajorLDS%s" % tc):
    width = lrvw * bpeDS / 4.0
    if caps.get("HasWMMA_V3") and kernel.get("MXScaleFormat") == "InMemorySwizzle":
      width *= int(kernel["VectorWidth%s" % parent])
  else:
    width = int(kernel["VectorWidth%s" % parent]) * bpeDS / 4.0
  if width <= 0:
    return None
  return min(width, 4.0)                            # the pool's widest LocalRead is b128


def mxScaleTilesPerRead(writer, kernel, tc, lrvw, mxUnit) -> int:
  """Scale TILES one `ds_read` of `tc` covers -- `blockWidth * 4 // mxUnit`.

  A shape not covered returns 1, supplying no map, and `leaves.emitLdsReadTile` keeps its own
  coverage.
  """
  blockWidth = mxScaleBlockWidth(writer, kernel, tc, lrvw)
  if mxUnit <= 0 or blockWidth is None:
    return 1
  return max(1, int(blockWidth * 4) // mxUnit)


def loopModelTarget(writer, kernel):
  """Target facts θ cannot see -- the domain its fields are chosen from.

    ReadVectorElems    elements one instruction moves per lane, per operand.
    ReadPhi / ReadRho  the carrier group as the two numbers naming it: Φ, the movement instances
                       one cooperative instruction merges, and ρ, the sub-agent span they are
                       distributed over.  θ derives the carrier/slot map from them.
    RegisterBudget     the hardware ceiling less RESERVE, θ's footprint being A/B/C/MXS only.

  Every input is read at the FIRST θ build, `loopModelRegBuffers` from `_initKernel`, so it comes
  from the Solution or from pure geometry rather than from later `writer.states` fields.
  """
  _regCaps = getattr(writer.states, "regCaps", None) or {}
  _maxVgpr = int(_regCaps.get("MaxVgpr", 0) or 0)

  elems, phi, rho = {}, {}, {}
  pt = kernel["ProblemType"]
  for tc, tile01, mx in (("MXSA", 0, "MXBlockA"), ("MXSB", 1, "MXBlockB")):
    if not int(pt.get(mx, 0) or 0):
      continue
    mxUnit = int(kernel["MatrixInstK"]) // int(pt[mx])
    lrvw = int(kernel.get("LocalReadVectorWidthMXS", 0) or 0)
    if mxUnit <= 0 or lrvw <= 0:
      continue
    # Use the instruction width rather than lrvw. An InMemorySwizzle read is
    # VectorWidth times wider.
    blockWidth = mxScaleBlockWidth(writer, kernel, tc, lrvw)
    elems[tc] = int(blockWidth * 4) if blockWidth is not None else lrvw
    tiles = mxScaleTilesPerRead(writer, kernel, tc, lrvw, mxUnit)
    span = Component.LocalRead.find(writer).getMxsTileSpanInfo(
        kernel, tc, tile01, writer.states.asmCaps) if hasattr(writer, "states") else None
    # TileSpan partitions the half-wave: one ds_load holds blocks 2g and
    # 2g+1, so the carrier group spans two sub-agents.
    if span:
      tiles *= 2
    partner = 2 if span else 0
    phi[tc] = int(tiles)
    rho[tc] = int(partner)
  _RESERVE = 32                                     # G2L staging, addresses, temporaries
  return {"ReadVectorElems": elems, "ReadPhi": phi, "ReadRho": rho,
          "RegisterBudget": (_maxVgpr - _RESERVE) if _maxVgpr else None}
