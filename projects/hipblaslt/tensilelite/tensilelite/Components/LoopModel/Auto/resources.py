# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""How many GPRs a LoopModel candidate needs, and whether that fits.

COUNTED, NOT ALIGNED: the operand rings as the writer sizes them (`operandValu`), the tail
loop's A/B slabs where they outgrow those rings, and the accumulator.

UNDER STINKYTOFU ALLOCATION THE COUNT IS A LOWER BOUND: nothing is added for the registers
outside the rings.  The allocator folds some of them (the LDS swap and GL2 prefetch addresses,
Serial) into ring slots idle at the time and keeps the rest (the local-read addresses, the
alignment of wide operands), so the declared count sits 0..~10 above this one on the gfx1250
MXF8 family.

WITHOUT IT THE COUNT IS A LOWER BOUND TOO: it adds the writer blocks outside the rings that every
such kernel holds (`nonRingVgprs`).  The writer's layout sits a few registers above that, and the
declared count can sit higher still, since the store code grows the register pool.
"""

import re
from dataclasses import dataclass

from ....Common.MxScaleLayout import mxFreePhase
from ....Common.ValidParameters import parseStinkyTofuRegisterAllocation
from ....LoopModel import traversal as _geometry
from ...TDMSplit import split_factors
from ..Registers import operandValu

#: The operands whose tail-loop slabs the writer reserves together (KernelWriter's tailEnd).
TAIL_OPERANDS = ("A", "B")


@dataclass(frozen=True)
class GprCount:
  operands: dict      #: {operand: VGPRs its main-loop ring holds}
  tailReserve: int    #: VGPRs the tail loop's A/B slabs need beyond the A/B rings
  accumulator: int
  fixedVgpr: int
  sgpr: int
  maxVgpr: int
  maxSgpr: int

  @property
  def vgpr(self) -> int:
    return sum(self.operands.values()) + self.tailReserve + self.accumulator + self.fixedVgpr

  @property
  def fits(self) -> bool:
    return self.vgpr <= self.maxVgpr and self.sgpr <= self.maxSgpr

  def reason(self):
    """Why it does not fit, or None."""
    if self.vgpr > self.maxVgpr:
      return ("needs %d VGPRs (%s, tail reserve %d, accumulator %d, fixed %d) but the target has %d"
              % (self.vgpr, ", ".join("%s %d" % kv for kv in self.operands.items()),
                 self.tailReserve, self.accumulator, self.fixedVgpr, self.maxVgpr))
    if self.sgpr > self.maxSgpr:
      return "needs %d SGPRs but the target has %d" % (self.sgpr, self.maxSgpr)
    return None


def allocatorPacksVgprs(kernel) -> bool:
  """Whether StinkyTofu register allocation rewrites this kernel's VGPRs (mode 2 or 3)."""
  try:
    canonical, mode = parseStinkyTofuRegisterAllocation(kernel.get("StinkyTofuRegisterAllocation", 0))
  except (TypeError, ValueError):
    return False
  if mode < 2:
    return False
  classes = re.search(r"classes=(\w+)", str(canonical))
  return classes is None or "v" in classes.group(1)


def _tensors(kernel):
  pt = kernel["ProblemType"]
  return ["A", "B"] + ["MXS" + side for side in ("A", "B") if pt.get("MXBlock" + side)]


def _side(tc):
  return tc[-1]


def _ldsAddressMultiplier(kernel, tc, costs) -> int:
  """Address registers one LDS address needs to reach the tensor's LDS extent (KernelWriter's
  numVgprMultiplier, without the segment-interleave and read-window cases, which only add)."""
  maxOffset = int(kernel.get("LdsNumElementsAligned" + tc, 0) or 0)
  if kernel.get("ExpandPointerSwap") and not kernel.get("1LDSBuffer") and not kernel.get("StoreSwapAddr"):
    maxOffset += int(kernel.get("LdsOffsetA_Blk", 0) or 0)
  reach = costs.ldsConstOffsetReach
  if tc == "B" and kernel.get("LDSSegmentBRotate"):
    return max(1, (maxOffset + reach - 1) // reach)
  return maxOffset // reach + 1


def _packsA(kernel) -> bool:
  return kernel["ProblemType"]["DataTypeA"].numBytes() < 4 and not kernel.get("UnrollMajorLDSA") \
      and not kernel.get("enableLDSTrA")


def nonRingVgprs(kernel, costs, operands=None) -> dict:
  """The writer's VGPR blocks outside the rings and the accumulator that every such kernel
  allocates, by block: a LOWER bound -- the other alignment, global-to-LDS staging, pack
  temporaries and the registers the store code adds to the pool are left out.  `operands` (the
  ring sizes) adds the bank padding between the A and B rings."""
  blocks = {"Serial": 1}
  if costs.vgprBankPadsB and operands is not None and not _packsA(kernel):
    blocks["BankPadB"] = 2 - int(operands.get("A", 0)) % 2
  for tc in _tensors(kernel):
    side = _side(tc)
    tdm = bool(kernel.get("enableTDM" + side))
    direct = bool(kernel.get("DirectToVgpr" + tc))
    multiplier = _ldsAddressMultiplier(kernel, tc, costs)
    if tc.startswith("MXS") and mxFreePhase(kernel, tc):
      multiplier = 1
    localRead = 0 if direct else multiplier
    localWrite = 0 if (direct or tdm or kernel.get("DirectToLds" + tc) or kernel.get("LocalWriteUseSgpr" + tc)) \
        else multiplier
    swap = 0
    if kernel.get("StoreSwapAddr"):
      swap = int(localRead > 0) + int(localWrite > 0)
    globalRead = 0
    if kernel.get("BufferLoad"):
      globalRead = 1 if kernel.get("_UseSgprForGRO") or not tdm else 0
    for name, count in (("LocalReadAddr", localRead), ("LocalWriteAddr", localWrite),
                        ("SwapAddr", swap), ("GlobalReadOffset", globalRead)):
      if count:
        blocks["%s%s" % (name, tc)] = count
  if kernel.get("PrefetchGL2"):
    perTensor = 1 if kernel.get("PrefetchGL2SAddr") else 2
    tensors = len(_tensors(kernel)) + int(bool(kernel.get("enableTDMMetadata")))
    blocks["GL2PrefetchAddr"] = perTensor * tensors
  return blocks


def fixedVgpr(kernel, costs, operands=None) -> int:
  """VGPRs counted for everything outside the rings and the accumulator: none when StinkyTofu
  allocates VGPRs, else the writer blocks every such kernel holds (`nonRingVgprs`)."""
  return 0 if allocatorPacksVgprs(kernel) else sum(nonRingVgprs(kernel, costs, operands).values())


def gprCount(kernel, theta, S, costs) -> GprCount:
  """Raises RuntimeError when the writer could not size a ring (`operandValu`)."""
  pt = kernel["ProblemType"]
  operands, accumulator, tailSlabs = {}, 0, 0
  for operand in theta.operands:
    if operand.is_output:
      accumulator += _geometry.free_tiles(theta, operand) * _geometry.frag_regs(theta, operand)
    elif operand.trajectory.fragment_fill is not None:
      # The writer's own sizing, not θ's footprint: a degenerate unit widens θ's ring
      # (`register_unit_modes`) past what the writer allocates and the kernel addresses.
      bpe = pt["MacDataType" + operand.name].numBytes() if operand.name in TAIL_OPERANDS else None
      operands[operand.name] = int(operandValu(theta, S, kernel, operand.name, bpe).total)
      if operand.name in TAIL_OPERANDS:
        tailSlabs += max(1, _geometry.free_tiles(theta, operand) * _geometry.frag_regs(theta, operand))
  rings = sum(operands.get(name, 0) for name in TAIL_OPERANDS)
  splits = split_factors(kernel) or (1, 1, 1, 1)
  splitOperands = int(splits[0] > 1 or splits[2] > 1) + int(splits[1] > 1 or splits[3] > 1)
  return GprCount(
      operands=operands,
      tailReserve=max(0, tailSlabs - rings),
      accumulator=accumulator,
      fixedVgpr=fixedVgpr(kernel, costs, operands),
      sgpr=costs.fixedSgpr + costs.sgprPerSplitOperand * splitOperands,
      maxVgpr=costs.maxVgpr,
      maxSgpr=costs.maxSgpr,
  )
