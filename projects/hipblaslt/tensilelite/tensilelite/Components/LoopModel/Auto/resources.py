# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""How many GPRs a LoopModel candidate needs, and whether that fits.

COUNTED, NOT ALIGNED: the operand rings as the writer sizes them (`operandValu`), the tail
loop's A/B slabs where they outgrow those rings, and the accumulator.  The one alignment counted
is the MX scale region's (`mxScaleRegion`), which moves the accumulator and everything above it.

UNDER STINKYTOFU ALLOCATION THE COUNT IS THE TOP OF THE B RING: the allocator keeps every WMMA
operand (the accumulator, A, B) where the writer put them (`ProducerPinMatrix`), so what the
writer lays out below B stays counted -- the address blocks between the accumulator and A and
the bank pad before B (`belowRingVgprs`).  What it lays out above B (the LDS swap and GL2
prefetch addresses, Serial) the allocator folds into idle registers; the declared count sits
0..4 above this one on the gfx1250 MXF8 family, and a kernel whose B ring ends past the limit
is refused however little else it holds.

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
#: The `nonRingVgprs` blocks the writer lays out between the accumulator and the A ring.
_BELOW_A = ("GlobalReadOffset", "LocalWriteAddr", "LocalReadAddr")


@dataclass(frozen=True)
class GprCount:
  operands: dict      #: {operand: VGPRs its main-loop ring holds}
  tailReserve: int    #: VGPRs the tail loop's A/B slabs need beyond the A/B rings
  accumulator: int
  fixedVgpr: int
  sgpr: int
  maxVgpr: int
  maxSgpr: int
  scaleAlignment: int = 0  #: VGPRs `mxScaleRegion` adds to the MX scale rings

  @property
  def vgpr(self) -> int:
    return (sum(self.operands.values()) + self.tailReserve + self.accumulator + self.fixedVgpr
            + self.scaleAlignment)

  @property
  def fits(self) -> bool:
    return self.vgpr <= self.maxVgpr and self.sgpr <= self.maxSgpr

  def reason(self):
    """Why it does not fit, or None."""
    if self.vgpr > self.maxVgpr:
      return ("needs %d VGPRs (%s, scale alignment %d, tail reserve %d, accumulator %d, fixed %d) "
              "but the target has %d"
              % (self.vgpr, ", ".join("%s %d" % kv for kv in self.operands.items()),
                 self.scaleAlignment, self.tailReserve, self.accumulator, self.fixedVgpr,
                 self.maxVgpr))
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


def bankPadB(kernel, costs, ringA) -> int:
  """VGPRs between an A ring of `ringA` and the B ring (none counted after a packed A ring)."""
  return 2 - ringA % 2 if costs.vgprBankPadsB and not _packsA(kernel) else 0


def nonRingVgprs(kernel, costs, operands=None) -> dict:
  """The writer's VGPR blocks outside the rings and the accumulator that every such kernel
  allocates, by block: a LOWER bound -- the other alignment, global-to-LDS staging, pack
  temporaries and the registers the store code adds to the pool are left out.  `operands` (the
  ring sizes) adds the bank padding between the A and B rings."""
  blocks = {"Serial": 1}
  if operands is not None and bankPadB(kernel, costs, int(operands.get("A", 0))):
    blocks["BankPadB"] = bankPadB(kernel, costs, int(operands.get("A", 0)))
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


def _offBank(costs, end) -> int:
  """VGPRs the writer skips after a block ending at `end` before the next one: on to the other
  bank pair under `vgprBankPadsB`, else to an even VGPR (KernelWriter's `_alignMXSB`, and the
  same rule between the A and B rings)."""
  return 2 - end % 2 if costs.vgprBankPadsB else end % 2


def mxScaleRegion(kernel, costs, scales) -> int:
  """VGPRs below the accumulator, for MX scale rings of `scales` (`{"MXSA": n, "MXSB": n}`) laid
  out as the writer does (`_layoutMXScaleVgprs`): a dummy pair when only one side has scales,
  MXSB moved off MXSA's bank, and the accumulator on an even VGPR.  MXScaleVgprInterleave never
  moves the accumulator, and a TDM-fed scale has no pack or G2L block, so neither is counted."""
  pt = kernel["ProblemType"]
  hasA, hasB = bool(pt.get("MXBlockA")), bool(pt.get("MXBlockB"))
  end = 2 if hasA != hasB else 0
  if hasA:
    end += int(scales.get("MXSA", 0))
  if hasB:
    end += _offBank(costs, end) + int(scales.get("MXSB", 0))
  return end + end % 2


def belowRingVgprs(kernel, costs, operands=None) -> int:
  """The writer blocks laid out below the B ring outside the rings and the scale region: the
  addresses between the accumulator and A, rounded to an even VGPR, and with `operands` the
  bank pad between the A and B rings."""
  blocks = nonRingVgprs(kernel, costs, operands)
  below = sum(count for name, count in blocks.items() if name.startswith(_BELOW_A))
  return below + below % 2 + blocks.get("BankPadB", 0)


def fixedVgpr(kernel, costs, operands=None) -> int:
  """VGPRs counted for everything outside the rings and the accumulator: when StinkyTofu
  allocates VGPRs, what the writer lays out below the pinned B ring (`belowRingVgprs`), else the
  writer blocks every such kernel holds (`nonRingVgprs`)."""
  if allocatorPacksVgprs(kernel):
    return belowRingVgprs(kernel, costs, operands)
  return sum(nonRingVgprs(kernel, costs, operands).values())


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
  scales = sum(count for name, count in operands.items() if name.startswith("MXS"))
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
      scaleAlignment=mxScaleRegion(kernel, costs, operands) - scales,
  )
