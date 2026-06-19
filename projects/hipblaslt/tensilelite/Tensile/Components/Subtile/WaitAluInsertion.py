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
    LocalReadInstruction, MFMAInstruction, MXMFMAInstruction, SWaitAlu,
    SWaitCnt, VXorB32,
)

# Number of WMMAs that must issue between a swap XOR and its dependent ds_read
# for the v_xor latency to be fully hidden (no s_wait_alu required).
# Conversative value based on s_wait_alu latency.
MIN_MMA_BEFORE_LR_READ = 4

_isMMA = lambda x: isinstance(x, (MFMAInstruction, MXMFMAInstruction))


def _vgprIndices(container):
  """Concrete VGPR indices a register operand covers, or empty for non-VGPRs."""
  if not isinstance(container, RegisterContainer) or container.regType != 'v':
    return ()
  return range(container.regIdx, container.regIdx + container.regNum)


def setMatrixAReuse(module, writer, kernel):
  """Enable the gfx1250 WMMA matrix-A reuse hint where it is safe.

  No-op unless gfx1250 (HasWmmaArbStallBit).  Mutates instructions in place.
  """
  if not writer.states.archCaps.get("HasWmmaArbStallBit", False):
    return module

  mmas = [inst for inst in module.flatitems() if _isMMA(inst)]
  for prev, cur in zip(mmas, mmas[1:]):
    if tuple(_vgprIndices(cur.a)) and _vgprIndices(cur.a) == _vgprIndices(prev.a):
      prev.reuseA = True
  return module


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


def insertLRSwapWarWaitAlu(module, writer, kernel):
  """Guard the ds_read -> LR offset-swap WAR hazard.

  A ds_read uses an LR offset VGPR as its address source; the swap v_xor then
  overwrites that same VGPR.  The address read is tracked by VM_VSRC, so the
  xor must not overwrite the offset until the outstanding reads have drained.
  In SCHED_MODE 2 the hardware no longer enforces this, and with PGR=0 (one
  offset set, reloaded every iteration) the xor can outrun the reads.  One
  SWaitAlu(vm_vsrc=0) drains every pending address read at once.

  We place that wait at the exact instruction that needs it: the first swap
  v_xor whose destination offset VGPR still has an in-flight ds_read address
  use.  This mirrors insertLRSwapWaitAlu's dependency walk -- a ds_read marks
  its address VGPR pending, an s_wait_dscnt 0 (the end-of-subIterK LR-complete
  wait) drains all pending DS reads, and a swap v_xor that writes a pending
  VGPR is the WAR consumer.  Because vm_vsrc(0) is a full drain, one wait
  clears the whole pending set, so later xors in the same block need none.
  No-op unless SCHED_MODE 2 (HasWmmaArbStallBit) is active.  Returns a rebuilt
  Module; the input is left untouched.
  """
  if not writer.states.archCaps.get("HasWmmaArbStallBit", False):
    return module

  # VGPR indices with an in-flight ds_read address use, not yet drained.
  pending = set()
  result = Module(module.name)
  for inst in module.flatitems():
    # Producer: a ds_read puts its address VGPR's read in flight on VM_VSRC.
    if isinstance(inst, LocalReadInstruction):
      params = inst.getParams()
      addr = params[1] if len(params) > 1 else None
      pending.update(_vgprIndices(addr))
      result.add(inst)
      continue

    # Drain: s_wait_dscnt 0 retires all in-flight DS (LDS) source reads, so the
    # offset reads are guaranteed complete and can no longer be clobbered.
    if isinstance(inst, SWaitCnt) and inst.dscnt == 0:
      pending.clear()
      result.add(inst)
      continue

    # Consumer: a swap v_xor that overwrites an offset VGPR with a pending read
    # is the WAR.  Drain once (covers every pending offset), then clear.
    if isinstance(inst, VXorB32) and any(idx in pending for idx in _vgprIndices(inst.dst)):
      result.add(SWaitAlu(vm_vsrc=0,
                          comment="wait for LR offset read before swap (WAR)"))
      pending.clear()

    result.add(inst)
  return result
