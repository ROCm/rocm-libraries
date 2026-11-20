# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The values each searched LoopModel parameter may take for one Solution.

A parameter the user left at -1 is FREE and ranges over every value its constraints allow; a
stated one is a single-value domain, so a user fork is never overridden.  The domains here are
the cheap, nest-only bounds -- `Registers.registerShapes` -- and the evaluator still judges
every candidate, so a value listed here may yet be refused.
"""

from dataclasses import dataclass
from typing import Tuple

from .. import Order, Registers


ORDER_PARAMS = ("WmmaInnerOrder", "WmmaOuterOrder", "WmmaFactorM", "WmmaFactorN", "WmmaFactorK")
REGISTER_PARAMS = ("PrefetchLocalRead", "ClusterLocalRead", "VgprGroup", "VgprAlloc")
SIDE_REGISTER_PARAMS = tuple(p + side for side in "AB" for p in REGISTER_PARAMS)
SPLIT_PARAMS = ("TDMSplitA", "TDMSplitB")
SEARCHED_PARAMS = ORDER_PARAMS + SIDE_REGISTER_PARAMS + SPLIT_PARAMS

#: Upper bounds on a free value, so a large tile cannot make the space explode.
MAX_FACTOR = 16
MAX_PLR = 8
MAX_VA = 16
#: `ValidParameters` accepts VgprGroup 1 to 8.
MAX_VG = 8


@dataclass(frozen=True, order=True)
class OrderChoice:
  inner: int
  outer: int
  factorM: int
  factorN: int
  factorK: int

  def apply(self, kernel):
    kernel["WmmaInnerOrder"] = self.inner
    kernel["WmmaOuterOrder"] = self.outer
    kernel["WmmaFactorM"] = self.factorM
    kernel["WmmaFactorN"] = self.factorN
    kernel["WmmaFactorK"] = self.factorK
    kernel.pop("LoopOrder", None)
    Order.calculate(kernel)

  def params(self):
    return dict(zip(ORDER_PARAMS, (self.inner, self.outer, self.factorM, self.factorN,
                                   self.factorK)))


@dataclass(frozen=True, order=True)
class SideChoice:
  plr: int
  clr: int
  vg: int
  va: int

  def choice(self):
    return {"PrefetchLocalRead": self.plr, "ClusterLocalRead": self.clr,
            "VgprGroup": self.vg, "VgprAlloc": self.va}


@dataclass(frozen=True, order=True)
class RegisterChoice:
  a: SideChoice
  b: SideChoice

  def choice(self):
    return {"A": self.a.choice(), "B": self.b.choice()}

  def params(self):
    out = {}
    for side, value in (("A", self.a), ("B", self.b)):
      for name, v in value.choice().items():
        out[name + side] = v
    return out


def freeParams(userState) -> Tuple[str, ...]:
  """The searched parameters the user left free (-1) in `userState`."""
  free = []
  for name in SEARCHED_PARAMS:
    value = userState.get(name, -1)
    if value is None or int(value) == -1:
      free.append(name)
  return tuple(free)


def _divisors(n, limit):
  n = max(1, int(n))
  return [d for d in range(1, min(n, limit) + 1) if n % d == 0]


def _domain(name, free, pinned, values):
  if name in free:
    return list(values)
  return [int(pinned[name])]


def orderChoices(kernel, free, pinned):
  """Every distinct loop order the free order parameters allow, in canonical order.

  Two parameter sets that lower to the same `LoopOrder` word and factors are one order; the
  first in canonical order represents it.
  """
  forced = Order.forcedFactors(kernel)
  tiles = kernel["MIWaveTile"]
  loopIters = max(1, int(kernel["LoopIters"]))
  domains = {
      "WmmaInnerOrder": _domain("WmmaInnerOrder", free, pinned, range(1, 7)),
      "WmmaOuterOrder": _domain("WmmaOuterOrder", free, pinned, range(0, 4)),
  }
  for role, extent in (("M", tiles[0]), ("N", tiles[1]), ("K", loopIters)):
    name = Order.FACTOR_PARAM[role]
    values = [forced[role]] if forced[role] > 1 else _divisors(extent, MAX_FACTOR)
    domains[name] = _domain(name, free, pinned, values)
  seen, out = set(), []
  for inner in domains["WmmaInnerOrder"]:
    for outer in domains["WmmaOuterOrder"]:
      for fM in domains["WmmaFactorM"]:
        for fN in domains["WmmaFactorN"]:
          for fK in domains["WmmaFactorK"]:
            choice = OrderChoice(inner, outer, fM, fN, fK)
            try:
              word = Order.loopOrder(inner, outer, {"M": fM, "N": fN, "K": fK})
            except ValueError:
              continue
            key = (word, fM, fN, fK)
            if key in seen:
              continue
            seen.add(key)
            out.append(choice)
  return out


def sideChoices(shape, side, free, pinned, maxPlr=MAX_PLR, maxVa=MAX_VA):
  """Register tuples one operand may take under one loop order (`Registers.RegisterShape`)."""
  vgs = _domain("VgprGroup" + side, free, pinned, _divisors(shape.unit, MAX_VG))
  clrs = _domain("ClusterLocalRead" + side, free, pinned, (0, 1))
  out = []
  for vg in vgs:
    if vg < 1 or shape.unit % vg:
      continue
    cap = max(1, shape.capBase * vg)
    plrs = _domain("PrefetchLocalRead" + side, free, pinned, range(0, min(cap, maxPlr) + 1))
    for plr in plrs:
      if plr > cap:
        continue
      for clr in clrs:
        if clr == 1:
          if shape.residentTiles % shape.unit:
            continue
          resident = max(1, shape.residentTiles // shape.unit * vg)
          vas = _domain("VgprAlloc" + side, free, pinned, (resident,))
        else:
          # A ring must hold its read-ahead plus the unit in use: PLR >= VA lowers to a
          # different loop than tier 0 scores, and GIR refuses some as WRONG SOURCE.  One
          # longer than the positions it walks (times the groups) only repeats slots.
          low = plr + 1
          high = max(low, max(shape.positions, plr + 1) * vg)
          vas = _domain("VgprAlloc" + side, free, pinned, range(low, min(high, maxVa) + 1))
          if "PrefetchLocalRead" + side in free or "VgprAlloc" + side in free:
            vas = [va for va in vas if int(va) > plr]
        for va in vas:
          out.append(SideChoice(plr, clr, vg, int(va)))
  return out


def registerShapes(kernel):
  return Registers.registerShapes(kernel)
