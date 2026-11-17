# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""How many GPRs a LoopModel candidate needs, and whether that fits.

COUNTED, NOT ALIGNED: StinkyTofuRegisterAllocation packs the register file, so the count is
what the kernel holds live -- the operand rings `operandValu` sizes for the writer, the
accumulator, and a calibrated remainder -- not what the writer's own aligned allocator reserves
before the allocator compacts it.
"""

from dataclasses import dataclass

from ....LoopModel import traversal as _geometry
from ...TDMSplit import split_factors


@dataclass(frozen=True)
class GprCount:
  operands: dict      #: {operand: VGPRs its ring holds}
  accumulator: int
  fixedVgpr: int
  sgpr: int
  maxVgpr: int
  maxSgpr: int

  @property
  def vgpr(self) -> int:
    return sum(self.operands.values()) + self.accumulator + self.fixedVgpr

  @property
  def fits(self) -> bool:
    return self.vgpr <= self.maxVgpr and self.sgpr <= self.maxSgpr

  def reason(self):
    """Why it does not fit, or None."""
    if self.vgpr > self.maxVgpr:
      return ("needs %d VGPRs (%s, accumulator %d, fixed %d) but the target has %d"
              % (self.vgpr, ", ".join("%s %d" % kv for kv in self.operands.items()),
                 self.accumulator, self.fixedVgpr, self.maxVgpr))
    if self.sgpr > self.maxSgpr:
      return "needs %d SGPRs but the target has %d" % (self.sgpr, self.maxSgpr)
    return None


def gprCount(kernel, theta, S, costs) -> GprCount:
  # The rings as `Registers.loopModelValuRegs` hands them to the writer.
  operands, accumulator = {}, 0
  for operand in theta.operands:
    if operand.is_output:
      accumulator += _geometry.free_tiles(theta, operand) * _geometry.frag_regs(theta, operand)
    elif operand.trajectory.fragment_fill is not None:
      operands[operand.name] = int(_geometry.operand_emitted_regs(theta, operand, S))
  splits = split_factors(kernel) or (1, 1, 1, 1)
  splitOperands = int(splits[0] > 1 or splits[2] > 1) + int(splits[1] > 1 or splits[3] > 1)
  return GprCount(
      operands=operands,
      accumulator=accumulator,
      fixedVgpr=costs.fixedVgpr,
      sgpr=costs.fixedSgpr + costs.sgprPerSplitOperand * splitOperands,
      maxVgpr=costs.maxVgpr,
      maxSgpr=costs.maxSgpr,
  )
