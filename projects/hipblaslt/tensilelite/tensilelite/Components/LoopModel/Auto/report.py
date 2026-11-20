# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What the LoopModel parameter search chose, and why."""

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Ranked:
  params: dict          #: the searched parameters, resolved
  score: float          #: tier-0 cycles per trip, what the ranking is on
  vgpr: int = 0
  units: int = 0        #: units both operands load per trip
  copies: int = 1       #: steady-body copies
  tier2: Optional[float] = None   #: simulated cycles per trip, reported only
  tier3: Optional[float] = None   #: emitted-loop cycles per trip (`asmsim`); ranks when set
  lowered: Optional[bool] = None  #: None: not tried; False: GIR refused it
  note: str = ""

  def key(self):
    return (self.score, self.units, self.copies, self.vgpr, sorted(self.params.items()))

  def tier3Key(self):
    return (self.tier3,) + self.key()


@dataclass
class AutoResult:
  winner: Optional[Ranked]
  ranked: List[Ranked] = field(default_factory=list)     #: best first
  refused: List[Ranked] = field(default_factory=list)    #: tier-1 survivors GIR or tier 3 refused
  rejected: Counter = field(default_factory=Counter)      #: {stage: count}
  examples: dict = field(default_factory=dict)            #: {stage: one reason}
  searched: int = 0
  seconds: float = 0.0

  def reason(self) -> str:
    """Why nothing was chosen -- the most common refusal, with one example of it."""
    if not self.rejected:
      return "LoopModelAutoSelect found no candidate"
    stage, count = self.rejected.most_common(1)[0]
    return ("LoopModelAutoSelect: none of %d candidate(s) is buildable; most were refused at %s "
            "(%d), e.g. %s" % (self.searched, stage, count, self.examples.get(stage, "")))

  def summary(self, top=5) -> str:
    lines = ["LoopModelAutoSelect: %d candidate(s) in %.1fs, refused %s"
             % (self.searched, self.seconds, dict(self.rejected))]
    for index, item in enumerate(self.ranked[:top]):
      lines.append("  %s%d. %s  vgpr=%d units=%d copies=%d  tier0=%.0f%s%s%s"
                   % ("*" if item is self.winner else " ", index + 1,
                      " ".join("%s=%s" % kv for kv in sorted(item.params.items())), item.vgpr,
                      item.units, item.copies, item.score,
                      "" if item.tier2 is None else "  tier2=%.0f" % item.tier2,
                      "" if item.tier3 is None else "  tier3=%.0f" % item.tier3,
                      "  (%s)" % item.note if item.note else ""))
    return "\n".join(lines)
