# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Search the free LoopModel parameters for the cheapest candidate that fits and builds.

Three tiers, each asked of fewer candidates than the one before:

  tier 0  every (order, register) pair, from the nest alone (`perf.NestWalk`, `perf.sideCost`):
          VGPRs, whether each ring is long enough to be correct, and the ds_load latency each
          unit leaves exposed.  The two operands' register choices are independent here, so
          each order keeps only its best few joint choices under budget.  An MX kernel's
          scale rings are the one count asked of θ (`_ScaleRings`), once per distinct ring.
  tier 1  the best tier-0 candidates through Solution's and the writer's checks and θ
          (`evaluate.tier1`), with the exact register count.
  tier 2  the tier-1 survivors lowered to GIR, the last feasibility gate, and simulated
          (`perf.simulate`).  The simulated cycles are reported, not ranked on: measured on
          hardware they order candidates worse than tier 0 does (see `hw.calibration`).

The winner is the best tier-0 candidate that lowers.  Ties go to fewer units loaded per trip
(each pays the latency), then fewer steady-body copies, then fewer VGPRs, then the canonical
order.

Tier 3 is not here: the Expand CLI (`Utilities/loopmodel_autoselect.py --tier3 N`) emits the
first N of this ranking and re-ranks them by the loop StinkyTofu scheduled (`asmsim`).  It
ranks measured kernels far better than tier 0 (`hw.calibration["asmDsLatency"]`), but needs a
full kernel emit per candidate, so Solution derivation does not run it.
"""

import time
from collections import Counter
from dataclasses import replace
from typing import NamedTuple

from . import evaluate, hw, perf, resources, space
from .report import AutoResult, Ranked


#: Joint register choices each loop order keeps after tier 0.
PER_ORDER = 16
#: Tier-1 survivors lowered for tier 2.
TIER2_COUNT = 16
#: Tier-2 candidates per loop order.
TIER2_PER_ORDER = 2
#: Lowering attempts before the search settles for what it has.
MAX_LOWERINGS = 48
#: Steady-body copies beyond which a candidate is not lowered (GIR time grows with them).
MAX_COPIES = 6
#: Tier-0 candidates passed to tier 1 at most.
MAX_TIER1 = 400


def _validChoices(kernel, walk, side, free, pinned, costs, rho, mx):
  """`[(SideCost, SideChoice)]` for every choice whose ring is long enough to be correct."""
  options = []
  for choice in space.sideChoices(walk.shapes[side], side, free, pinned):
    cost = perf.sideCost(kernel, walk, side, choice, costs, rho, mx)
    if cost is not None:
      options.append((cost, choice))
  return options


class _ScaleRings:
  """θ's MX scale ring for one loop order's side choices, asked once per `(side, VgprGroup,
  VgprAlloc, ClusterLocalRead)`: θ sizes it from those and the order alone, not from the side's
  PLR or the other side's choice, so a fixed ring-valid partner stands in for the other side."""

  #: Partners tried before a choice keeps tier 0's estimate.
  PARTNERS = 4

  def __init__(self, state, order, kernel, valid, costs, asmCaps):
    self.state, self.order, self.kernel = state, order, kernel
    self.costs, self.asmCaps = costs, asmCaps
    self.partners = {side: [choice for _cost, choice in valid[side][:self.PARTNERS]]
                     for side in ("A", "B")}
    self.known = {}

  def __call__(self, side, choice):
    """VGPRs of `side`'s scale ring under `choice`, or None when θ could not size it."""
    key = (side, choice.vg, choice.va, choice.clr)
    if key not in self.known:
      self.known[key] = None
      for partner in self.partners["B" if side == "A" else "A"]:
        registers = (space.RegisterChoice(choice, partner) if side == "A"
                     else space.RegisterChoice(partner, choice))
        ev = evaluate.tier1(self.state, self.order, registers, self.costs, self.asmCaps,
                            orderedKernel=self.kernel)
        if ev.gpr is not None:
          self.known[key] = int(ev.gpr.operands.get("MXS" + side, 0))
          break
    return self.known[key]


def _sideOptions(kernel, walk, side, costs, trip, options, scale=None):
  if scale is not None:
    exact = []
    for cost, choice in options:
      regs = scale(side, choice)
      exact.append((cost if regs is None else replace(cost, regs=cost.ring + regs), choice))
    options = exact
  # A window's shortfall counts only beyond the least the copies rule charges any body.
  floor = max(0.0, costs.tdmLatency - trip)
  landing = perf.tdmLandingLatency(kernel, costs)
  def short(cost):
    alone = perf.tdmWindowExposed([cost.tdm], landing, trip / float(walk.wmmas))
    return alone if alone > floor else 0.0
  # A choice another matches or beats on every count is never needed; of equals, the first in
  # canonical order stays.
  def counts(cost):
    return (cost.regs, cost.exposed, cost.units, cost.copies, short(cost))

  options.sort(key=lambda item: (counts(item[0]), item[1]))
  front = []
  for cost, choice in options:
    mine = counts(cost)
    if any(all(o <= m for o, m in zip(counts(other), mine)) for other, _choice in front):
      continue
    front.append((cost, choice))
  return front


class Tier0(NamedTuple):
  """One tier-0 candidate; the field order is the ranking."""
  score: float      #: cycles per trip
  units: int        #: units both operands load per trip
  copies: int       #: steady-body copies
  vgpr: int         #: VGPRs besides the accumulator and `resources.fixedVgpr` of the state
  order: space.OrderChoice
  registers: space.RegisterChoice


def _lcm(a, b):
  from math import gcd
  return a * b // gcd(a, b)


def _walks(state, free, rejected, examples):
  """One `(order, kernel, walk)` per distinct nest, in canonical order.

  Two spellings of one nest (a factor whose other part is 1) are one candidate.  A spelling
  with an extent-1 axis gives θ a degenerate unit, which it may widen into more registers
  than the walk counts, so a spelling without one represents the nest when it has one.
  """
  nests = {}
  for order in space.orderChoices(state, free, state):
    kernel, reason = evaluate.applyOrder(state, order)
    if kernel is None:
      rejected["order"] += 1
      examples.setdefault("order", reason)
      continue
    try:
      walk = perf.NestWalk(kernel, space.registerShapes(kernel))
    except (ValueError, RuntimeError) as error:
      rejected["order"] += 1
      examples.setdefault("order", str(error))
      continue
    nest = tuple((name[0], extent) for name, extent in walk.axes if extent > 1)
    degenerate = any(extent == 1 for _name, extent in walk.axes)
    held = nests.get(nest)
    if held is None or (held[0] and not degenerate):
      nests[nest] = (degenerate, order, kernel, walk)
  return sorted(((order, kernel, walk) for _degenerate, order, kernel, walk in nests.values()),
                key=lambda item: item[0])


def _tier0(state, free, costs, budget, keep=PER_ORDER, asmCaps=None):
  """Every order's best joint register choices, best first: `[(score, vgpr, order, regs)]`.

  With `asmCaps` an MX kernel's scale rings are θ's (`_ScaleRings`); without, tier 0's
  estimate."""
  pt = state["ProblemType"]
  mx = bool(pt.get("MXBlockA") or pt.get("MXBlockB"))
  rejected, examples, out = Counter(), {}, []
  for order, kernel, walk in _walks(state, free, rejected, examples):
    cWmma, cLds, rho, _loads = perf.orderCore(kernel, walk, costs)
    perPosition = max(cWmma, cLds) / float(walk.wmmas)
    tdm = perf.tensorLoads(kernel) * costs.tdmIssueCycles
    landing = perf.tdmLandingLatency(kernel, costs)
    slabs = perf.tailSlabs(kernel)
    valid = {side: _validChoices(kernel, walk, side, free, state, costs, rho, mx)
             for side in ("A", "B")}
    scale = (_ScaleRings(state, order, kernel, valid, costs, asmCaps)
             if mx and asmCaps is not None else None)
    sideA = _sideOptions(kernel, walk, "A", costs, max(cWmma, cLds), valid["A"], scale)
    sideB = _sideOptions(kernel, walk, "B", costs, max(cWmma, cLds), valid["B"], scale)
    joint = []
    for costA, choiceA in sideA:
      for costB, choiceB in sideB:
        # The tail loop's slabs reuse the A and B rings and reserve only what they outgrow; that
        # total grows with each side's registers, so the sorted B list can stop on it.  The
        # writer's padding (`resources.gprCount`) does not, so it only skips a pair.
        regs = costA.regs + costB.regs + max(0, slabs - costA.ring - costB.ring)
        if regs > budget:
          break
        scaleA, scaleB = costA.regs - costA.ring, costB.regs - costB.ring
        regs += (resources.bankPadB(kernel, costs, costA.ring) - scaleA - scaleB
                 + resources.mxScaleRegion(kernel, costs, {"MXSA": scaleA, "MXSB": scaleB}))
        if regs > budget:
          continue
        copies = _lcm(costA.copies, costB.copies)
        score = (max(cWmma, cLds) + costA.exposed + costB.exposed + tdm
                 + perf.tdmExposed(cWmma, cLds, copies, costs, (costA.tdm, costB.tdm), perPosition,
                                  landing)
                 + costs.tripOverhead)
        joint.append(Tier0(score, costA.units + costB.units, copies, regs, order,
                           space.RegisterChoice(choiceA, choiceB)))
    if not joint:
      rejected["budget"] += 1
      examples.setdefault("budget", "no register choice under %s fits %d VGPRs"
                          % (kernel["LoopOrder"], budget))
      continue
    joint.sort()
    out.extend(joint[:keep])
  out.sort()
  return out, rejected, examples


def search(state, free, asmCaps, costs=None, tier2=True, topK=TIER2_COUNT,
           perOrder=TIER2_PER_ORDER, band=None, maxLowerings=MAX_LOWERINGS, tier0PerOrder=PER_ORDER,
           accept=None):
  """Pick the free LoopModel parameters of `state` (a resolved Solution state).

  `free` names the parameters the user left at -1; every other searched parameter keeps the
  value `state` already has.  Returns an `AutoResult`; its `ranked[0]` is the winner.

  `perOrder` caps the candidates one loop order sends on and `tier0PerOrder` the register
  choices it scores (None: no cap).  With `band` set the
  search keeps every candidate whose tier-0 score is within that fraction of the best, rather
  than stopping at `topK`; `maxLowerings` still bounds it.

  `accept(params)` is a last check on each candidate that passes the tiers: it returns the stage
  that refuses it, or None.  A refused candidate is counted under that stage and takes no place
  in the ranking, so the next one is tried.
  """
  from ..Theta import loopModelTargetFor
  costs = costs or hw.costsFor(state.get("ISA")) or hw.GFX1250
  started = time.time()
  pt = state["ProblemType"]
  accumulator = int(state["MIWaveTile"][0]) * int(state["MIWaveTile"][1]) * max(
      1, int(state["MatrixInstM"]) * int(state["MatrixInstN"]) // int(state["WavefrontSize"]))
  budget = costs.maxVgpr - resources.fixedVgpr(state, costs) - accumulator
  candidates, rejected, examples = _tier0(state, free, costs, budget, tier0PerOrder, asmCaps)
  result = AutoResult(None, rejected=rejected, examples=examples)
  result.searched = len(candidates) + sum(rejected.values())

  target = loopModelTargetFor(state, asmCaps)
  lowered, ranked, perOrderCount = 0, [], Counter()
  ceiling = candidates[0].score * (1.0 + band) if band is not None and candidates else None
  for candidate in candidates[:MAX_TIER1]:
    order, registers = candidate.order, candidate.registers
    if lowered >= maxLowerings:
      break
    if ceiling is not None:
      if candidate.score > ceiling:
        break
    # Ties with the last place go on: the ranking breaks them on the exact VGPR count, which
    # only tier 1 knows, so a smaller `topK` keeps the same leaders.
    elif len(ranked) >= topK and candidate[:3] > max((r.score, r.units, r.copies) for r in ranked):
      break
    # The slots are spread over loop orders rather than spent on one order's register
    # variants; an order's first candidates are its best, so the winner is unaffected.
    if perOrder is not None and perOrderCount[order] >= perOrder:
      continue
    ev = evaluate.tier1(state, order, registers, costs, asmCaps)
    if not ev.feasible:
      result.rejected[ev.stage] += 1
      result.examples.setdefault(ev.stage, ev.reason)
      continue
    ev.perf = perf.estimate(ev.kernel, ev.theta, ev.S, costs, target)
    item = Ranked(ev.params(), candidate.score, ev.gpr.vgpr, units=candidate.units,
                  copies=candidate.copies)
    if tier2:
      if ev.perf.extra.get("copies", 1) > MAX_COPIES:
        result.rejected["copies"] += 1
        result.examples.setdefault("copies", "%d ring copies of the steady body"
                                   % ev.perf.extra["copies"])
        continue
      lowered += 1
      perOrderCount[order] += 1
      try:
        program = perf.lower(ev.theta)
        timeline = perf.simulate(ev.kernel, program, costs)
        item.tier2, item.lowered = timeline.cycles, True
        item.note = timeline.summary()
      except Exception as error:  # noqa: BLE001 -- GIR's refusal is the verdict
        item.lowered = False
        item.note = "GIR: %s" % (str(error).splitlines() or [type(error).__name__])[0][:120]
        result.refused.append(item)
        result.rejected["gir"] += 1
        result.examples.setdefault("gir", item.note)
        continue
    else:
      perOrderCount[order] += 1
    if accept is not None:
      stage = accept(item.params)
      if stage is not None:
        result.rejected[stage] += 1
        result.examples.setdefault(stage, " ".join("%s=%s" % kv
                                                   for kv in sorted(item.params.items())))
        continue
    ranked.append(item)

  ranked.sort(key=Ranked.key)
  result.ranked = ranked
  result.winner = ranked[0] if ranked else None
  result.seconds = time.time() - started
  return result
