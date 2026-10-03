# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Derive LoopModel Valu-ring register allocation from cached theta and S."""

from ...LoopModel import traversal as _geometry

from .Theta import loopModelTheta


def loopModelRegBuffers(writer, kernel):
  """Each operand's maximum ring depth -- the real VGPR buffer index, regions included.

  GIR and the `Valu*_X*` table read the same cached θ/S, so every slot GIR names has a matching
  symbol.  Register groups share the X axis, so they take `max` rather than `sum`.
  """
  if not kernel["UseLoopModel"]:
    return {}
  theta, S = loopModelTheta(writer, kernel)
  widths = {}
  for op in theta.operands:
    if op.trajectory.fragment_fill is None:
      continue
    w = max((_geometry.group_ring_depth(theta, op, group, S)
             for group in op.fragment.groups()), default=1)
    widths[op.name] = max(1, int(w))
  return widths


def loopModelValuRegs(writer, kernel):
  """Physical VGPRs per tensorChar for the Valu rings GIR names.

  Each register group owns a compact block for every one of its W rotation slots.
  """
  if not kernel["UseLoopModel"]:
    return {}
  theta, S = loopModelTheta(writer, kernel)
  out = {}
  for op in theta.operands:
    if op.trajectory.fragment_fill is None:
      continue
    out[op.name] = int(_geometry.operand_emitted_regs(theta, op, S))
  return out


def loopModelRegisterLayout(writer, kernel):
  """Compact per-group physical register offsets from the cached θ/S."""
  if not kernel["UseLoopModel"]:
    return {}
  theta, S = loopModelTheta(writer, kernel)
  return _geometry.register_layout(theta, S)


def applyLoopModelValuRegs(writer, kernel):
  """Overwrite the scaffold's Valu-ring sizing with θ/S's, after it has run."""
  if not kernel["UseLoopModel"]:
    return
  required = writer.states.loopModelValuVgprs
  byOperand = {"A": writer.states.a, "B": writer.states.b}
  if kernel["ProblemType"]["MXBlockA"]:
    byOperand["MXSA"] = writer.states.mxsa
  if kernel["ProblemType"]["MXBlockB"]:
    byOperand["MXSB"] = writer.states.mxsb
  missing = [name for name in byOperand if name not in required]
  if missing:
    raise RuntimeError("UseLoopModel has no Valu VGPR allocation for %s" % ", ".join(missing))
  for name, state in byOperand.items():
    count = int(required[name])
    if count < 1:
      raise RuntimeError("UseLoopModel derived invalid Valu VGPR count %d for %s" % (count, name))
    state.numVgprValu = count
