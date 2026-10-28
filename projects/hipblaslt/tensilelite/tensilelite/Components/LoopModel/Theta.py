# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Build the LoopModel theta and target facts once per kernel."""

from contextlib import contextmanager

from ...Component import Component
from ...LoopModel.adapter import kernel_to_params, params_to_theta
from ...LoopModel.schedule import build_S


def loopModelSolutionName(writer) -> str:
  """The solution this writer is building, for a failure to name itself with."""
  return getattr(writer.states, "kernelName", None) or "<unnamed solution>"


@contextmanager
def namingSolution(writer, kernel=None):
  """Re-raise anything the model refuses with the SOLUTION it came from.

  θ and GIR are pure in the Solution and never learn its name, so a bare refusal costs a bisect
  to place.  Naming it at the two writer-facing entry points keeps the model ignorant and the
  message useful.  The flag makes it idempotent: the inner builder names it, the outer passes it
  through rather than saying it twice.

  NAMING ONLY -- THE CLASS IS THE RAISER'S.  A `ValueError` means the decoder and the traversal
  disagree, and it stops the run so the disagreement gets repaired; stating VgprGroup/VgprAlloc
  does not make it any less of a defect, so this must not downgrade one to a skip.  A site that
  genuinely refuses a request raises `RuntimeError` itself, where the reason is known.
  """
  try:
    yield
  except (ValueError, RuntimeError) as error:
    if getattr(error, "_ulmNamedSolution", False):
      raise
    # No parameter list appended: the refusals already name the operand and the value that could not
    # be met, and a kernel-wide roll-call next to a per-operand message reads as a contradiction.
    named = type(error)("%s\n  solution: %s" % (error, loopModelSolutionName(writer)))
    named._ulmNamedSolution = True
    raise named from error


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
  with namingSolution(writer, kernel):
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

  Every input is read at the FIRST θ build, `loopModelRegBuffers` from `_initKernel`, so it comes
  from the Solution or from pure geometry rather than from later `writer.states` fields.
  """
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
  return {"ReadVectorElems": elems, "ReadPhi": phi, "ReadRho": rho}
