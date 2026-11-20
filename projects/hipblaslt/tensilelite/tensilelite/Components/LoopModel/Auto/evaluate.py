# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Judge one LoopModel candidate: is it buildable, does it fit, and what does it cost.

Tier 1 runs the checks Solution and the writer would, in their order, on a COPY of the state,
then builds θ and counts its registers -- about a millisecond.  A refusal names the stage that
refused, so the search can report why nothing fit.  Tier 2 (`tier2`) lowers the survivor to GIR
and simulates its steady body; it costs about a second and is only asked of the best few.
"""

from dataclasses import dataclass, field
from typing import Optional

from ....LoopModel.schedule import prefetch_refusals
from ....SolutionStructs.Validators.LocalReadPrefetch import localReadPrefetchRejections
from .. import Registers
from ..Theta import buildTheta
from .resources import gprCount


#: Stages in the order tier 1 runs them.
STAGES = ("order", "registers", "local-read-prefetch", "theta", "prefetch", "budget")


@dataclass
class Evaluation:
  order: object
  registers: object
  stage: Optional[str] = None       #: the stage that refused it, None when feasible
  reason: Optional[str] = None
  kernel: Optional[dict] = None     #: the candidate's resolved state
  theta: object = None
  S: object = None
  gpr: object = None
  perf: object = None
  tier2: object = None
  extra: dict = field(default_factory=dict)

  @property
  def feasible(self) -> bool:
    return self.stage is None

  def params(self):
    out = dict(self.order.params())
    out.update(self.registers.params())
    return out


def applyOrder(state, order):
  """A shallow copy of `state` under `order`, or `(None, reason)`.

  Order and Registers only ever replace top-level keys, so a shallow copy isolates them.
  """
  kernel = dict(state)
  try:
    order.apply(kernel)
  except (ValueError, RuntimeError) as error:
    return None, str(error)
  return kernel, None


def tier1(state, order, registers, costs, asmCaps, orderedKernel=None):
  """Evaluate `(order, registers)` on `state`.  `orderedKernel` reuses an `applyOrder` result."""
  ev = Evaluation(order, registers)
  if orderedKernel is None:
    orderedKernel, reason = applyOrder(state, order)
    if orderedKernel is None:
      ev.stage, ev.reason = "order", reason
      return ev
  kernel = dict(orderedKernel)
  # The winner is built with its values STATED, and a stated VgprGroup/VgprAlloc is what drops
  # the no-load loop and marks the allocation explicit -- so every candidate is judged stated.
  kernel.update(registers.params())
  kernel["SuppressNoLoadLoop"] = True
  try:
    Registers.derive(kernel)
  except (ValueError, RuntimeError) as error:
    ev.stage, ev.reason = "registers", str(error)
    return ev
  for reason, _stop in localReadPrefetchRejections(kernel, bool(asmCaps.get("HasWMMA_V3"))):
    ev.stage, ev.reason = "local-read-prefetch", reason
    return ev
  try:
    theta, S = buildTheta(kernel, asmCaps)
  except (ValueError, RuntimeError, KeyError, ZeroDivisionError) as error:
    ev.stage, ev.reason = "theta", "%s: %s" % (type(error).__name__, error)
    return ev
  refusals = prefetch_refusals(theta, S)
  if refusals:
    ev.stage, ev.reason = "prefetch", "; ".join(str(r) for r in refusals)
    return ev
  try:
    gpr = gprCount(kernel, theta, S, costs)
  except RuntimeError as error:
    ev.stage, ev.reason = "registers", str(error)
    return ev
  ev.kernel, ev.theta, ev.S, ev.gpr = kernel, theta, S, gpr
  if not gpr.fits:
    ev.stage, ev.reason = "budget", gpr.reason()
  return ev
