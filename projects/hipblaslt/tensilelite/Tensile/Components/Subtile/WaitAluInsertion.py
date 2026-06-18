# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

################################################################################
# Post-schedule s_wait_alu insertion for the LR offset-swap RAW hazard.
#
# In SCHED_MODE 2 the LR swap's v_xor -> ds_read RAW on the offset VGPR is no
# longer guarded by hardware.  A wait is only needed when the read is too close
# to the swap: if MIN_MMA_BEFORE_LR_READ WMMAs separate them the v_xor has
# already retired, otherwise we insert one SWaitAlu(va_vdst=0).
#
#
################################################################################

from rocisa.code import Module
from rocisa.container import RegisterContainer
from rocisa.instruction import (
    LocalReadInstruction, MFMAInstruction, MXMFMAInstruction, SWaitAlu, VXorB32,
)

# Number of WMMAs that must issue between a swap XOR and its dependent ds_read
# for the v_xor latency to be fully hidden (no s_wait_alu required).
MIN_MMA_BEFORE_LR_READ = 4

_isMMA = lambda x: isinstance(x, (MFMAInstruction, MXMFMAInstruction))


def _vgprIndices(container):
  """Concrete VGPR indices a register operand covers, or empty for non-VGPRs."""
  if not isinstance(container, RegisterContainer) or container.regType != 'v':
    return ()
  return range(container.regIdx, container.regIdx + container.regNum)


def insertLRSwapWaitAlu(module, writer, kernel):
  """Guard the LR offset-swap -> ds_read RAW hazard.

  For each swap window, if fewer than MIN_MMA_BEFORE_LR_READ WMMAs separate the
  swap XORs from the first dependent ds_read, insert one SWaitAlu(va_vdst=0)
  before that read (which validates all pending swaps).  No-op unless SCHED_MODE
  2 (HasWmmaArbStallBit) is active.  Returns a rebuilt Module; the input is left
  untouched.
  """
  # gfx1250 only
  if not writer.states.archCaps.get("HasWmmaArbStallBit", False):
    return module

  dirty = set()       # offset VGPR indices written by a swap-XOR, not yet drained
  mmaSinceSwap = 0    # WMMAs issued since the first un-drained swap XOR
  result = Module(module.name)

  for inst in module.flatitems():
    # Producer: a swap-XOR writes an LR offset VGPR.
    if isinstance(inst, VXorB32):
      if not dirty:
        mmaSinceSwap = 0
      dirty.update(_vgprIndices(inst.dst))
      result.add(inst)
      continue

    if dirty and _isMMA(inst):
      mmaSinceSwap += 1

    # Consumer: a ds_read whose address source is a swapped offset VGPR.
    if dirty and isinstance(inst, LocalReadInstruction):
      # DSLoad.getParams() -> [dst, addrSrc]; the address source is index 1.
      params = inst.getParams()
      addr = params[1] if len(params) > 1 else None
      if any(idx in dirty for idx in _vgprIndices(addr)):
        if mmaSinceSwap < MIN_MMA_BEFORE_LR_READ:
          result.add(SWaitAlu(va_vdst=0, comment="wait for LR offset swap to complete"))
        dirty.clear()

    result.add(inst)

  return result
