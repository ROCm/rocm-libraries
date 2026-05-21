# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

################################################################################
# Scale GR/LR emit for MX scale factor operands (MXSA/MXSB).
#
# Scale factors use a simpler access pattern than data tiles:
#   GR: DTL with linear offset (serial * loadWidth), one buffer_load per wave
#   LR: ds_read_b32 per scale group (2 M-adjacent subtiles per group)
#
# Each function operates on a single tensor component (MXSA or MXSB),
# called once per scale operand.
#
# Uses ti.sharedVgprGROffset / ti.sharedVgprLROffset (compat properties)
# since MXScaleTilePair has gr=None, lr=None.
################################################################################

import math

from rocisa.code import Module
from rocisa.container import DSModifiers, MUBUFModifiers, vgpr, sgpr, mgpr
from rocisa.instruction import (
    BufferLoadB128,
    DSLoadB32,
    SAddCU32, SAddU32, SLShiftLeftB32, SMovB32, SMulI32, SNop, SXorB32,
    VAddU32, VAndB32, VMulLOU32, VReadfirstlaneB32, VXorB32,
    VLShiftLeftB32, VLShiftRightB32,
)


# ---------------------------------------------------------------------------
# Scale GR offset
# ---------------------------------------------------------------------------
#
# NOTE: this is currently a STUB (returns immediately on the first line).
# The active implementation lives in `_graTileAssignmentScaleSwizzledCommon`
# below; the unreachable body here is kept as a sketch. Any R-period audit
# of the GR offset math should be applied to that function, not this one.
#
# Caveat: if the STUB return is ever removed, the body below uses
# `ti.localSubtileGrid[1]` instead of `ti.lrGlobalSubtileGrid[1]`. For
# MXScaleTilePair the former is always 1 (see Kernel.py: TileInfo for
# MXScale sets `localSubtileGrid=[1,1]`), so the formula would not pick up
# the R-period growth in per-scale-fetch K subtile count and would
# silently miscompile for any scaleDepthU > one LR K subtile.

def emitScaleGROffset(ti, writer, kernel):
  """Compute per-thread DTL vaddr for scale GR load."""
  return Module(f"Scale GR Offset ({ti.tc})")  # STUB
  module = Module(f"Scale GR Offset ({ti.tc})")
  tc = ti.tc
  loadWidth = ti.loadWidthGR
  loadWidthShift = loadWidth.bit_length() - 1

  scaleGroupSize = ti.lrSubtileSize
  numThreadsPerGroup = (scaleGroupSize * int(ti.localSubtileGrid[1])) // loadWidth

  vtmp = writer.vgprPool.checkOut(1)
  stmp = writer.sgprPool.checkOut(1)

  module.add(VLShiftRightB32(dst=vgpr(vtmp),
             shiftHex=hex(numThreadsPerGroup.bit_length()-1), src=vgpr("Serial"),
             comment=f"scale{tc}: groupId"))
  module.add(SMulI32(dst=sgpr(stmp), src0=int(ti.bpe), src1=sgpr("Strides" + tc),
             comment=f"scale{tc}: stride * bpe"))
  module.add(VMulLOU32(dst=vgpr(vtmp), src1=vgpr(vtmp), src0=sgpr(stmp),
             comment=f"scale{tc}: groupId * stride"))
  module.add(VAndB32(dst=vgpr(ti.sharedVgprGROffset[0]),
             src0=hex(numThreadsPerGroup - 1), src1=vgpr("Serial"),
             comment=f"scale{tc}: threadId"))
  module.add(VLShiftLeftB32(dst=vgpr(ti.sharedVgprGROffset[0]),
             shiftHex=hex(loadWidthShift), src=vgpr(ti.sharedVgprGROffset[0]),
             comment=f"scale{tc}: threadId * loadWidth"))
  module.add(VAddU32(dst=vgpr(ti.sharedVgprGROffset[0]),
             src0=vgpr(ti.sharedVgprGROffset[0]), src1=vgpr(vtmp),
             comment=f"scale{tc}: final offset"))

  writer.vgprPool.checkIn(vtmp)
  writer.sgprPool.checkIn(stmp)
  return module


# ---------------------------------------------------------------------------
# Scale GR load (DTL)
# ---------------------------------------------------------------------------

def emitScaleGRLoad(ti, writer, kernel):
  """Emit buffer_load_b128 DTL for scale data (global -> LDS)."""
  module = Module(f"Scale GR Load ({ti.tc})")
  tc = ti.tc

  isGlc = bool(kernel.get(f"NonTemporal{tc}", 0) & 0x1)
  isSlc = bool(kernel.get(f"NonTemporal{tc}", 0) & 0x2)
  isNT  = bool(kernel.get(f"NonTemporal{tc}", 0) & 0x4)

  module.add(SMovB32(dst=mgpr(0), src=sgpr(f"LocalWriteBaseAddr{tc}"),
             comment=f"scale{tc}: M0 = scaleLdsBase"))

  mubuf = MUBUFModifiers(offen=True, offset12=0, glc=isGlc, slc=isSlc, nt=isNT, lds=True)
  module.add(BufferLoadB128(dst=None, vaddr=vgpr(ti.sharedVgprGROffset[0]),
             saddr=sgpr(f"Srd{tc}", 4), soffset=0, mubuf=mubuf,
             comment=f"scale{tc}: DTL b128 load"))

  return module


# ---------------------------------------------------------------------------
# Scale LR offset
# ---------------------------------------------------------------------------

def emitScaleLROffset(ti, writer, kernel):
  """Compute per-lane LDS read offset for scale LR."""
  return Module(f"Scale LR Offset ({ti.tc})")  # STUB
  module = Module(f"Scale LR Offset ({ti.tc})")
  tc = ti.tc
  wavesize = kernel["WavefrontSize"]

  mi = kernel["MIWaveGroup"]
  totalScaleBytes = (ti.macroTile // ti.waveGroupSize) * ti.scaleDepthU * int(ti.bpe)

  waveIdVgpr = writer.vgprPool.checkOut(1)
  module.add(VLShiftRightB32(dst=vgpr(waveIdVgpr), shiftHex=hex(wavesize.bit_length()-1),
             src=vgpr("Serial"), comment=f"scale{tc}: waveId"))

  vtmp = writer.vgprPool.checkOut(1)
  stmp = writer.sgprPool.checkOut(1)

  if tc in ('A', 'MXSA'):
    module.add(VAndB32(dst=vgpr(vtmp), src0=mi[0]-1, src1=vgpr(waveIdVgpr),
               comment=f"scale{tc}: waveId %% {mi[0]}"))
  else:
    module.add(VLShiftRightB32(dst=vgpr(vtmp),
               shiftHex=int(math.log2(mi[0])), src=vgpr(waveIdVgpr),
               comment=f"scale{tc}: waveId / {mi[0]}"))

  module.add(SMovB32(dst=sgpr(stmp), src=totalScaleBytes,
             comment=f"scale{tc}: partition stride"))
  module.add(VMulLOU32(dst=vgpr(ti.sharedVgprLROffset[0]),
             src0=sgpr(stmp), src1=vgpr(vtmp),
             comment=f"scale{tc}: partition offset"))

  writer.vgprPool.checkIn(vtmp)
  writer.vgprPool.checkIn(waveIdVgpr)

  # Per-lane offset: laneId * 4
  laneOffset = writer.vgprPool.checkOut(1)
  module.add(VAndB32(dst=vgpr(laneOffset), src0=vgpr("Serial"), src1=wavesize-1,
             comment=f"scale{tc}: laneId"))
  module.add(VLShiftLeftB32(dst=vgpr(laneOffset), shiftHex=hex(2), src=vgpr(laneOffset),
             comment=f"scale{tc}: laneId * 4"))
  module.add(VAddU32(dst=vgpr(ti.sharedVgprLROffset[0]),
             src0=vgpr(laneOffset), src1=vgpr(ti.sharedVgprLROffset[0]),
             comment=f"scale{tc}: + laneOffset"))
  writer.vgprPool.checkIn(laneOffset)

  # Add global LDS offset
  ldsStartOffset = getattr(writer, f'ldsStartOffset{tc}', 0)
  if ldsStartOffset:
    module.add(SMovB32(dst=sgpr(stmp), src=hex(ldsStartOffset),
               comment=f"scale{tc}: LDS base offset"))
    module.add(VAddU32(dst=vgpr(ti.sharedVgprLROffset[0]),
               src0=vgpr(ti.sharedVgprLROffset[0]), src1=sgpr(stmp),
               comment=f"scale{tc}: + LDS offset"))

  # Init swap VGPRs
  module.add(SMovB32(dst=sgpr(stmp), src=writer.ldsTotalSize,
             comment=f"scale{tc}: ldsTotalSize"))
  for i in range(len(ti.sharedVgprLROffset)):
    vOff  = ti.sharedVgprLROffset[i]
    vSwap = ti.sharedVgprLROffsetSwap[i]
    module.add(VAddU32(dst=vgpr(vSwap), src0=vgpr(vOff), src1=sgpr(stmp),
               comment=f"scale{tc}: swap init"))
    module.add(VXorB32(dst=vgpr(vSwap), src0=vgpr(vOff), src1=vgpr(vSwap),
               comment=f"scale{tc}: swap mask"))

  writer.sgprPool.checkIn(stmp)
  return module


# ---------------------------------------------------------------------------
# Scale LR load
# ---------------------------------------------------------------------------
#
# R-period invariants (decouple MXFP8 scale DepthU):
#   numScaleGroups = (lrGlobalSubtileGrid[0] / waveGroupSize) * lrGlobalSubtileGrid[1]
#   is the count of ds_read_b32 instructions a wave issues per *scale fetch*.
#   With ScaleDepthURatio{tc} = R, lrGlobalSubtileGrid[1] scales linearly
#   with R, so numScaleGroups scales linearly with R as well: a single scale
#   fetch covers R times more K-bytes per wave, requiring R times more
#   ds_reads to drain the wider per-wave LDS region into VGPRs.
#
#   groupStride = lrSubtileSize is R-INVARIANT. LR subtiles remain
#   contiguous in LDS along K because UnrollMajorLDSMXS{A,B}=True in
#   Solution.py (calcLdsNumBytesAB), so consecutive group ids simply step
#   one lrSubtileSize down the K axis of the wave's partition. The total
#   range covered, numScaleGroups * lrSubtileSize, equals the per-wave
#   partition size computed in _applyScaleWavePartitionLROffset, which
#   itself grows by R.
#
#   This function is not currently on the active call path — the legacy
#   `localReadDoScaleSubtile` / `emitSubtileScaleDsRead` pair below is
#   what's wired up. Its loop bound uses `localSubtileGrid` instead of
#   `lrGlobalSubtileGrid`; see the comment on `emitSubtileScaleDsRead`.

def emitScaleLRLoad(ti, writer, kernel):
  """Emit ds_read_b32 for all scale groups."""
  module = Module(f"Scale LR Load ({ti.tc})")
  tc = ti.tc

  if ti.mxBlock == 0:
    return module

  # numScaleGroups ∝ R via lrGlobalSubtileGrid[1]; groupStride is R-invariant.
  numScaleGroups = (int(ti.lrGlobalSubtileGrid[0]) // ti.waveGroupSize) * int(ti.lrGlobalSubtileGrid[1])
  groupStride = int(ti.lrSubtileSize)

  for gid in range(numScaleGroups):
    dsOffset = groupStride * gid
    vdst = ti.vgprTiles[4 * gid].regList.indices[0]
    module.add(DSLoadB32(dst=vgpr(vdst),
               src=vgpr(ti.sharedVgprLROffset[0]),
               ds=DSModifiers(offset=dsOffset),
               comment=f"scale{tc}[group{gid}]: 4B from LDS"))

  return module


# ---------------------------------------------------------------------------
# Scale GR ptr update
# ---------------------------------------------------------------------------
#
# R-period invariants (decouple MXFP8 scale DepthU):
#   `inc` is the byte stride for advancing the scale SRD from one scale
#   fetch to the next. It equals (per-wave LR coverage along K) *
#   (bytes per LR subtile) for a single M-column slice:
#       inc = lrSubtileSize * lrGlobalSubtileGrid[1]
#           = bytes per scale K-span fetched per call
#           = R * dataDU * bpe / mxBlock     (per scale tensor)
#   The K-loop driver fires this update once per scale fetch (= once per
#   R data-DepthU body iterations under the R-period scheduler), so the
#   pointer naturally advances by R * dataDU worth of scale bytes per
#   call without any per-iteration multiplier here. R=1 reduces to the
#   legacy "advance by dataDU/mxBlock scale bytes per call" cadence.

def emitScaleGRPtrUpdate(ti, writer, kernel):
  """Advance scale SRD base pointer by one depthU iteration."""
  module = Module()
  tc = ti.tc

  # inc scales linearly with ScaleDepthURatio{tc} via lrGlobalSubtileGrid[1].
  inc = int(ti.lrSubtileSize * ti.lrGlobalSubtileGrid[1])
  module.addComment0("Scale SRD update: %s += %u" % (tc, inc))
  module.add(SAddU32(dst=sgpr(f"Srd{tc}"), src0=sgpr(f"Srd{tc}"), src1=inc))
  module.add(SAddCU32(dst=sgpr(f"Srd{tc}+1"), src0=sgpr(f"Srd{tc}+1"), src1=0))
  return module


# ---------------------------------------------------------------------------
# Scale LDS buffer swaps
# ---------------------------------------------------------------------------

def emitScaleGRLDSSwap(ti, writer, kernel):
  """Toggle scale GR DTL write target between double-buffer halves."""
  module = Module()
  tc = ti.tc
  module.addComment0("Emit code to swap %s GR m0 offsets"%tc)
  module.add(SXorB32(dst=sgpr(f"LocalWriteBaseAddr{tc}"),
             src0=sgpr(f"LocalWriteBaseAddr{tc}"), src1=sgpr(f"Swap{tc}"),
             comment=""))
  return module


def emitScaleLRLDSSwap(ti, writer, kernel):
  """Toggle scale LR read offsets between double-buffer halves."""
  module = Module()
  module.addComment0("Emit code to swap %s LR vgpr offsets"%ti.tc)
  for i in range(len(ti.sharedVgprLROffset)):
    vOff  = ti.sharedVgprLROffset[i]
    vSwap = ti.sharedVgprLROffsetSwap[i]
    module.add(VXorB32(dst=vgpr(vOff), src0=vgpr(vOff), src1=vgpr(vSwap), comment=""))
  return module


# =========================================================================
# Legacy Scale emit functions (moved from SubtileBasedKernel.py)
# =========================================================================

##################################################
# Compute the per-thread global-read (DTL) vaddr for scale tensor tc.
#
# With DTL (buffer_load lds=True) the same vaddr serves as:
#   - global byte offset from the SRD base  (where to read from global memory)
#   - LDS byte offset from M0               (where to write in LDS)
#
# Threads within a wave are split into groups of numThreadsPerGroup.
# Each group loads one contiguous subtile-column worth of scale bytes:
#
#   groupId  = serial / numThreadsPerGroup          (which scale column)
#   threadId = serial % numThreadsPerGroup           (position within group)
#
#   grOffset = groupId  * stride_bpe                (column byte offset via tensor stride)
#            + threadId * loadWidth                  (byte offset within column)
#
# Output: sharedVgprGROffset[0] = grOffset (used as vaddr in DTL load)
#
# ---------------------------------------------------------------------------
# R-period invariant (decouple MXFP8 scale DepthU):
#   With ScaleDepthURatio{A,B} = R (>=1; default 1 = legacy behavior), one
#   scale fetch covers R data-DepthU iterations of K, and
#   MXScaleTilePair.depthU is sized to R * dataDU. Consequently
#       lrGlobalSubtileGrid[1]  ∝  R
#   (K-dim LR subtile count per scale fetch grows linearly with R), and
#   therefore
#       numThreadsPerGroup = lrSubtileSize * lrGlobalSubtileGrid[1] / loadWidth
#                          ∝ R
#   i.e. R times more threads cooperate per scale "column" on each fetch.
#   This matches the larger per-fetch LDS scale region allocated by
#   Solution.py (calcLdsNumBytesAB uses _DepthUMXS{A,B} = R*dataDU/mxBlock).
#
#   numThreadsPerGroup must:
#     (a) remain a power of two — the formula uses VLShiftRightB32 with
#         shiftHex = log2(numThreadsPerGroup), and we apply bit-length /
#         math.log2 below;
#     (b) fit inside a single wavefront — DTL is per-wave, and groupId
#         derived from a single `serial` value cannot reach beyond the
#         wave's lane count without aliasing.
#   Both already held at R=1 for every supported config; the asserts
#   below catch a misconfigured R that would silently break the math.
# ---------------------------------------------------------------------------
def _graTileAssignmentScaleSwizzledCommon(tc, writer, kernel):
  module = Module()

  module.addComment("Computing GR Offset for %s"%tc)

  # TODO: revisit property mappings below (lrSubtileSize,
  # lrGlobalSubtileGrid); add helpers on TileInfo if they recur across emit functions.
  ti_ = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
  loadWidth = ti_.loadWidthGR
  loadWidthShift = loadWidth.bit_length() - 1

  # lrSubtileSize = LR subtile bytes (2x2 MMA tiles = 256B for FP4 scale).
  # This equals the old "2 * subtileSize" (2 M-adjacent [1,2] subtiles).
  # lrGlobalSubtileGrid[1] = K-dim subtile count = old localSubtileGrid[1].
  # Under R-period scheduling, lrGlobalSubtileGrid[1] (and therefore
  # numThreadsPerGroup) scales linearly with ScaleDepthURatio{tc}.
  scaleGroupSize = ti_.lrSubtileSize
  numThreadsPerGroup = (scaleGroupSize * int(ti_.lrGlobalSubtileGrid[1])) // loadWidth

  # R-period invariants: must be pow2 (we use log2 / bit_length below) and
  # must fit within a single wavefront (one buffer_load is per-wave).
  assert numThreadsPerGroup > 0 and (numThreadsPerGroup & (numThreadsPerGroup - 1)) == 0, \
    "scale%s: numThreadsPerGroup=%d must be a power of two (uses bit-shift cadence; " \
    "check lrSubtileSize=%d * lrGlobalSubtileGrid[1]=%s / loadWidth=%d)" % \
    (tc, numThreadsPerGroup, int(scaleGroupSize), str(ti_.lrGlobalSubtileGrid[1]), int(loadWidth))
  assert numThreadsPerGroup <= kernel["WavefrontSize"], \
    "scale%s: numThreadsPerGroup=%d exceeds WavefrontSize=%d — ScaleDepthURatio*lrGlobalSubtileGrid[1] " \
    "produced a per-scale-fetch K span that does not fit in one wave's lane budget" % \
    (tc, numThreadsPerGroup, kernel["WavefrontSize"])

  vtmp = writer.vgprPool.checkOut(1)

  stmp = writer.sgprPool.checkOut(1)

  module.add(VLShiftRightB32(dst=vgpr(vtmp),
                            shiftHex=hex(int(math.log2(numThreadsPerGroup))), src=vgpr("Serial"),
                            comment="%s: grOffset = serial / %d" % (tc, loadWidth)))
  module.add(SLShiftLeftB32(sgpr(stmp), int(math.log2(ti_.bpe)), sgpr("Strides%s"%tc), comment="*= bpe (%d)"%(ti_.bpe)))

  module.add(VMulLOU32(dst=vgpr(vtmp), src1=vgpr(vtmp), src0=sgpr(stmp), comment="Apply scale%s stride to each group"%tc))
  module.add(VAndB32(dst=vgpr(ti_.sharedVgprGROffset[0]),
                     src0=hex(numThreadsPerGroup - 1), src1=vgpr("Serial"),
                     comment="%s: grOffset = serial %% %d" % (tc, loadWidth)))
  module.add(VLShiftLeftB32(dst=vgpr(ti_.sharedVgprGROffset[0]),
                            shiftHex=hex(loadWidthShift), src=vgpr(ti_.sharedVgprGROffset[0]),
                            comment="Scale by load width for each thread in group"))
  module.add(VAddU32(dst=vgpr(ti_.sharedVgprGROffset[0]), src0=vgpr(ti_.sharedVgprGROffset[0]), src1=vgpr(vtmp), comment="Final offset calc"))
  writer.vgprPool.checkIn(vtmp)
  writer.sgprPool.checkIn(stmp)

  return module

##################################################
# Generate GR offset calculation for scaleA/B (DTL).
#
# With DTL, vaddr serves as both the global read offset (from SRD)
# and the LDS write offset (from M0). Simple linear access:
#   grOffset = serial * scaleLoadWidth
#
def graTileAssignmentScaleSwizzled(writer, kernel):
  module = Module()
  if not kernel["ProblemType"].get("MXBlockA", 0) and not kernel["ProblemType"].get("MXBlockB", 0):
    return module
  module.add(_graTileAssignmentScaleSwizzledCommon('MXSA', writer, kernel))
  module.add(_graTileAssignmentScaleSwizzledCommon('MXSB', writer, kernel))
  return module


##################################################
# Apply wave partition offset for scale LR.
#
# Each wave reads from its assigned LDS partition for scale A or B.
#
#   MXSA: partition index = waveId % MIWaveGroup[0]  (M-direction wave index)
#   MXSB: partition index = waveId / MIWaveGroup[0]  (N-direction wave index)
#         Using MIWaveGroup[0] (not [1]) correctly handles asymmetric configs
#         (e.g. 4x1: all 4 M-waves share the same N partition -> index = 0).
#
# Output: sharedVgprLROffset[0] = partitionIndex * totalScaleBytes
#
# ---------------------------------------------------------------------------
# R-period invariant (decouple MXFP8 scale DepthU):
#   totalScaleBytes is the size of ONE wave's scale LDS partition per scale
#   fetch. With ScaleDepthURatio{tc} = R, MXScaleTilePair.depthU is sized to
#   R * dataDU, so lrGlobalSubtileGrid[1] grows by R and
#       totalScaleBytes  ∝  R.
#
#   This is consistent with the LDS allocation in Solution.py
#   (calcLdsNumBytesAB, UnrollMajorLDSMXS{A,B}=True branch):
#       ldsNumBytes = (_DepthUMXS{tc} + ldsPad) * MacroTile{tc} * bpe
#   where _DepthUMXS{tc} = R * dataDU / mxBlock{tc} (Solution.py line
#   ~2373/2376 under the R-period plan). Because the LDS allocation also
#   grows by R, the per-wave partitions still tile the scale region exactly:
#       partitions_per_dim    = MIWaveGroup[index]
#       totalScaleBytes_wave  = ldsNumBytes / MIWaveGroup[index]
#   (modulo ldsPad, which is added outside this function).
#
#   The assert below catches the easy-to-miss case where an R chosen at
#   the kernel level makes lrGlobalSubtileGrid[0] no longer divisible by
#   MIWaveGroup[index]; under R=1 this divisibility is already enforced by
#   TileInfo._check_dim (waveGroupSize argument), so this never fires for
#   legacy configs.
# ---------------------------------------------------------------------------
def _applyScaleWavePartitionLROffset(module, writer, kernel, ti_, waveId):
  tc = ti_.tc

  # totalScaleBytes = bytes per wave partition in LDS for this scale tensor.
  # lrGlobalSubtileGrid[0] = M-dim LR subtile count (globalMMATileGrid[0] / lrSubtileShape[0])
  # lrGlobalSubtileGrid[1] = K-dim LR subtile count — scales linearly with
  #                          ScaleDepthURatio{tc} (= R), so totalScaleBytes ∝ R.
  # lrSubtileSize = bytes per LR subtile (2x2 MMA tiles for FP4 scale)
  index = 0 if tc == 'MXSA' else 1
  assert int(ti_.lrGlobalSubtileGrid[0]) % kernel["MIWaveGroup"][index] == 0, \
    "scale%s: lrGlobalSubtileGrid[0]=%s not divisible by MIWaveGroup[%d]=%d — " \
    "wave partitions cannot tile the M-dim scale region evenly" % \
    (tc, str(ti_.lrGlobalSubtileGrid[0]), index, kernel["MIWaveGroup"][index])
  totalScaleBytes = (int(ti_.lrGlobalSubtileGrid[0]) // kernel["MIWaveGroup"][index]) * int(ti_.lrGlobalSubtileGrid[1]) * int(ti_.lrSubtileSize)

  tmpSgpr = writer.sgprPool.checkOut(1)
  tmp = writer.vgprPool.checkOut(2)

  if tc == 'MXSA':
    module.add(VAndB32(dst=vgpr(tmp), src0=kernel["MIWaveGroup"][0]-1, src1=vgpr(waveId), comment="scale%s: waveId %% 2"%tc))
  else:
    module.add(VLShiftRightB32(dst=vgpr(tmp), shiftHex=int(math.log2(kernel["MIWaveGroup"][0])), src=vgpr(waveId), comment="scale%s: waveId / numWavesM"%tc))

  module.add(SMovB32(dst=sgpr(tmpSgpr), src=totalScaleBytes, comment="scale%s: scale region"%tc))
  module.add(VMulLOU32(dst=vgpr(ti_.sharedVgprLROffset[0]), src0=sgpr(tmpSgpr), src1=vgpr(tmp), comment="scale%s: partition offset"%tc))

  writer.vgprPool.checkIn(tmp)
  writer.sgprPool.checkIn(tmpSgpr)


##################################################
# Generate LR offset calculation for scaleA/B.
#
# Computes the per-lane LDS read offset for scale tensors. Called once
# during kernel setup; the resulting VGPRs are used every loop iteration.
#
# Final LR offset per lane:
#   lrOffset[lane] = wavePartitionOffset + laneId * 4 + ldsStartOffset
#
# where:
#   wavePartitionOffset  = partitionIndex * totalScaleBytes
#     MXSA partitionIndex = waveId % MIWaveGroup[0]   (M-direction)
#     MXSB partitionIndex = waveId / MIWaveGroup[0]   (N-direction)
#   laneId               = serial & (wavesize - 1)
#   ldsStartOffset       = writer.ldsStartOffsetMXSA/B
#
# LDS layout (double-buffered, one buffer shown):
#   [ DataA | DataB | ScaleA | ScaleB ]
#   ScaleA starts at ldsStartOffsetMXSA, ScaleB at ldsStartOffsetMXSB.
#
# After the LR offset is fully computed, the double-buffer swap VGPR is
# initialised here (not in localReadDTLInitCommonSwapVgpr, which runs
# before this function and would use uninitialised values):
#   swapVgpr = lrOffset XOR (lrOffset + ldsTotalSize)
# This lets localReadLDSBufferSwap toggle between buffer 0 and buffer 1.
#
def lraTileAssignmentScaleSwizzled(writer, kernel):
  return _lraTileAssignmentScaleSwizzled_legacy(writer, kernel)

def _lraTileAssignmentScaleSwizzled_legacy(writer, kernel):
  module = Module()
  if not kernel["ProblemType"].get("MXBlockA", 0) and not kernel["ProblemType"].get("MXBlockB", 0):
    return module
  tiA_ = writer.states.mxsa.tileInfo
  tiB_ = writer.states.mxsb.tileInfo
  module.addComment0("LR Offset Calculation for Scale Tensors")
  wavesize = kernel["WavefrontSize"]
  waveIdVgpr = writer.vgprPool.checkOut(1)
  module.add(VLShiftRightB32(dst=vgpr(waveIdVgpr), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="scale: waveId"))
  _applyScaleWavePartitionLROffset(module, writer, kernel, tiA_, waveIdVgpr)
  _applyScaleWavePartitionLROffset(module, writer, kernel, tiB_, waveIdVgpr)
  writer.vgprPool.checkIn(waveIdVgpr)
  laneOffset = writer.vgprPool.checkOut(1)
  module.add(VAndB32(dst=vgpr(laneOffset), src0=vgpr("Serial"), src1=wavesize-1, comment="scale: laneId"))
  module.add(VLShiftLeftB32(dst=vgpr(laneOffset), shiftHex=hex(2), src=vgpr(laneOffset), comment="scale: laneId * 4"))
  module.add(VAddU32(dst=vgpr(tiA_.sharedVgprLROffset[0]), src0=vgpr(laneOffset), src1=vgpr(tiA_.sharedVgprLROffset[0]), comment="scaleA: lrOffset = laneId * 4"))
  module.add(VAddU32(dst=vgpr(tiB_.sharedVgprLROffset[0]), src0=vgpr(laneOffset), src1=vgpr(tiB_.sharedVgprLROffset[0]), comment="scaleB: lrOffset = laneId * 4"))
  writer.vgprPool.checkIn(laneOffset)
  tmpSgpr = writer.sgprPool.checkOut(1)
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(writer.ldsStartOffsetMXSA), comment="scale: LDS offset for A scale"))
  module.add(VAddU32(dst=vgpr(tiA_.sharedVgprLROffset[0]), src0=vgpr(tiA_.sharedVgprLROffset[0]), src1=sgpr(tmpSgpr), comment="scaleA: +=LDS offset"))
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(writer.ldsStartOffsetMXSB), comment="scale: LDS offset for B scale"))
  module.add(VAddU32(dst=vgpr(tiB_.sharedVgprLROffset[0]), src0=vgpr(tiB_.sharedVgprLROffset[0]), src1=sgpr(tmpSgpr), comment="scaleB: +=LDS offset"))
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=writer.ldsTotalSize, comment="scale: total LDS size for swap"))
  for ti_ in [tiA_, tiB_]:
    for i in range(len(ti_.sharedVgprLROffset)):
      vgprId     = ti_.sharedVgprLROffset[i]
      vgprSwapId = ti_.sharedVgprLROffsetSwap[i]
      module.add(VAddU32(dst=vgpr(vgprSwapId), src0=vgpr(vgprId), src1=sgpr(tmpSgpr), comment="scale%s: LR swap"%ti_.tc))
      module.add(VXorB32(dst=vgpr(vgprSwapId), src0=vgpr(vgprId), src1=vgpr(vgprSwapId), comment="scale%s: LR swap"%ti_.tc))
  writer.sgprPool.checkIn(tmpSgpr)
  return module

##################################################
# Scale GR: Load scale bytes from global memory directly to LDS (DTL).
#
# Uses BufferLoadB128 with lds=True. M0 is set to scaleLdsBase, and
# sharedVgprGROffset[0] = serial * scaleLoadWidth serves as both the
# global read offset (from SRD) and the LDS write offset (from M0).
def globalReadDoScaleSubtile(tc, writer, kernel):
  module = Module()

  if not kernel["ProblemType"].get("MXBlockA", 0) and not kernel["ProblemType"].get("MXBlockB", 0):
    return module

  tileInfo = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo

  isGlc = bool(kernel["NonTemporal%s"%tc] & 0x1)
  isSlc = bool(kernel["NonTemporal%s"%tc] & 0x2)
  isNT  = bool(kernel["NonTemporal%s"%tc] & 0x4)

  assert len(tileInfo.sharedVgprGROffset) > 0, "Scale GR requires at least 1 GR offset VGPR"

  module.addComment0("Scale GR: %s (DTL: BufferLoadB128 -> LDS)" % tc)

  # Set M0 to scale LDS base address for DTL write destination
  module.add(SMovB32(dst=mgpr(0), src=sgpr("LocalWriteBaseAddr%s"%tc),
                     comment="scale%s: M0 = scaleLdsBase" % tc))

  # DTL load: data goes directly from global memory to LDS (no intermediate VGPR)
  mubuf = MUBUFModifiers(offen=True, offset12=0, glc=isGlc, slc=isSlc, nt=isNT, lds=True)
  module.add(BufferLoadB128(dst=None, vaddr=vgpr(tileInfo.sharedVgprGROffset[0]),
                            saddr=sgpr("Srd%s" % tc, 4), soffset=0, mubuf=mubuf,
                            comment="scale%s: DTL b128 load" % tc))

  return module

##################################################
# Scale LR: Read scale data from LDS into scale VGPRs (DSLoadB32).
#
# Each lane reads 4 bytes from LDS using ds_read_b32. The base address
# is sharedVgprLROffset[0] (computed by lraTileAssignmentScaleSwizzled).
# MMA tile and subtile selection is done via constant ds_offset at emit time.
#
# Each 32-bit VGPR holds 4 E8M0 scale bytes; opsel/opsel_hi selects
# the correct byte per MFMA invocation.
#
# ---------------------------------------------------------------------------
# R-period invariants (decouple MXFP8 scale DepthU):
#   `groupStride = lrSubtileSize` is R-INVARIANT — the byte size of one
#   LR subtile (2x2 MMA scale tiles) does not depend on ScaleDepthURatio.
#   `dsOffset = groupStride * scaleGroupIdx` simply walks consecutive LR
#   subtiles along K in the wave's LDS partition, which is contiguous along
#   K because Solution.py uses UnrollMajorLDSMXS{A,B}=True (see LDS layout
#   comment around Solution.py:2625-2638).
#
#   The number of groups (= number of times this is called per scale fetch)
#   is owned by the caller. Under R-period it should scale linearly with R
#   so that R * dataDU worth of K is drained from LDS per scale fetch
#   (matching the wider per-wave partition; see _applyScaleWavePartitionLROffset).
#
#   Caveat (pre-existing, NOT introduced by R-period): the current caller
#   `localReadDoScaleSubtile` derives `numScaleGroups` from
#   `tileInfo.localSubtileGrid`, which for MXScale is always [1, 1] (see
#   Kernel.py TileInfo MXScaleTilePair branch). That value is invariant
#   under R, so this PGR=0 call site would NOT pick up the R-period growth.
#   The active scheduler path (`emitScaleLRLoad` above, used via the
#   LogicalScheduler) uses `lrGlobalSubtileGrid` and DOES scale with R.
# ---------------------------------------------------------------------------
def emitSubtileScaleDsRead(tc, writer, kernel, scaleGroupIdx):
  """Emit a single DSLoadB32 for a scale group (2 M-adjacent [1,2] subtiles).
  Each ds_read_b32 loads 4 bytes = 4 E8M0 scale values into one VGPR."""
  module = Module()
  tileInfo = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo

  if tileInfo.mxBlock == 0:
    return module

  # TileInfo LR subtile (2,2) already spans 2 M-adjacent tiles -> stride = lrSubtileSize.
  # Legacy TileInfo subtile (1,2) spans 1 M-tile -> stride = 2 * subtileSize.
  if hasattr(tileInfo, 'lrSubtileSize'):
    groupStride = int(tileInfo.lrSubtileSize)
  else:
    groupStride = 2 * tileInfo.subtileSize
  dsOffset = groupStride * scaleGroupIdx
  vdst = tileInfo.vgprTiles[4 * scaleGroupIdx].regList.indices[0]
  module.add(DSLoadB32(dst=vgpr(vdst),
                       src=vgpr(tileInfo.sharedVgprLROffset[0]),
                       ds=DSModifiers(offset=dsOffset),
                       comment="scale%s[group%u]: load 4B from LDS" % (tc, scaleGroupIdx)))
  return module

def localReadDoScaleSubtile(tc, writer, kernel):
  """Emit scale ds_reads for all scale groups (PGR=0 path)."""
  module = Module()

  if not kernel["ProblemType"].get("MXBlockA", 0) and not kernel["ProblemType"].get("MXBlockB", 0):
    return module

  tileInfo = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo

  # Iterate over scale groups: one ds_read per 2 M-adjacent subtiles
  numScaleGroups = math.ceil(tileInfo.localSubtileGrid[0] / 2) * tileInfo.localSubtileGrid[1]
  for gid in range(numScaleGroups):
    module.add(emitSubtileScaleDsRead(tc, writer, kernel, gid))

  return module

##################################################
# Scale SRD pointer update: advance scale SRD by scaleDepthU * scaleBpe bytes.
#
# R-period invariant (decouple MXFP8 scale DepthU):
#   Thin wrapper around `emitScaleGRPtrUpdate`. The increment encoded by
#   that function already equals R * dataDU worth of scale bytes per call,
#   so this wrapper must be invoked exactly ONCE per scale fetch
#   (= once per R data-DepthU body iterations under the R-period
#   LogicalScheduler). Calling it more often would over-advance the SRD
#   by a factor of R. For R=1 this collapses to the legacy
#   "once per body iter" cadence and the call-frequency invariant is
#   trivially satisfied.
#
def globalReadScalePtrUpdates(tc, writer, kernel):
  ti_ = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
  return emitScaleGRPtrUpdate(ti_, writer, kernel)

##################################################
# Subroutine to generate DTL M0 LDS buffer swap
#
# For Swizzled Scales each wave will collectively stream
# the scale values
#
# ---------------------------------------------------------------------------
# R-period invariant (decouple MXFP8 scale DepthU):
#   `bytesPerLoad = loadWidth * wavesize` is the per-wave byte footprint of
#   ONE buffer_load_b128 (the only DTL load shape used for scales). This
#   shape — and therefore the per-wave M0 base offset spacing
#   (`waveId * bytesPerLoad`) and the Swap{MXSA,MXSB} XOR mask derivation
#   — is R-INVARIANT. ScaleDepthURatio{tc} only affects how MANY of these
#   loads happen per scale fetch and the size of the LDS scale region
#   (`ldsTotalSize` already accounts for that via Solution.py's
#   `_DepthUMXS{tc} = R * dataDU / mxBlock` allocation), not the per-load
#   layout the SGPRs initialized here depend on.
# ---------------------------------------------------------------------------
def globalReadScaleSwizzledDTLInitCommonSgpr(writer, kernel):
  module = Module()

  wavesize = kernel["WavefrontSize"]
  vgprWaveId = writer.vgprPool.checkOut(1)
  module.addComment0("Compute shared offsets used by m0 in DTL loads")
  module.add(VLShiftRightB32(dst=vgpr(vgprWaveId), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="Wave Id"))

  tiMXSA_ = writer.states.mxsa.tileInfo
  tiMXSB_ = writer.states.mxsb.tileInfo

  loadWidth = tiMXSA_.loadWidthGR

  bytesPerLoad = loadWidth * wavesize
  module.add(VLShiftLeftB32(dst=vgpr(vgprWaveId), shiftHex=hex((bytesPerLoad).bit_length()-1), src=vgpr(vgprWaveId), comment="Apply wave-specific common offset (%u) for A/B"%bytesPerLoad))

  module.add(SNop(waitState=0, comment="Wait for VGPR to be ready"))
  module.add(VReadfirstlaneB32(dst=sgpr("LocalWriteBaseAddrMXSA"), src=vgpr(vgprWaveId), comment="Store base LDS offset, will be modified"))
  module.add(VReadfirstlaneB32(dst=sgpr("LocalWriteBaseAddrMXSB"), src=vgpr(vgprWaveId), comment="Store base LDS offset, will be modified"))
  module.add(SAddU32(dst=sgpr("LocalWriteBaseAddrMXSA"), src0=sgpr("LocalWriteBaseAddrMXSA"), src1=hex(writer.ldsStartOffsetMXSA), comment=""))
  module.add(SAddU32(dst=sgpr("LocalWriteBaseAddrMXSB"), src0=sgpr("LocalWriteBaseAddrMXSB"), src1=hex(writer.ldsStartOffsetMXSB), comment=""))

  module.add(SAddU32(dst=sgpr("SwapMXSA"), src0=sgpr("LocalWriteBaseAddrMXSA"), src1=writer.ldsTotalSize, comment=""))
  module.add(SXorB32(dst=sgpr("SwapMXSA"), src0=sgpr("LocalWriteBaseAddrMXSA"), src1=sgpr("SwapMXSA"), comment=""))
  module.add(SAddU32(dst=sgpr("SwapMXSB"), src0=sgpr("LocalWriteBaseAddrMXSB"), src1=writer.ldsTotalSize, comment=""))
  module.add(SXorB32(dst=sgpr("SwapMXSB"), src0=sgpr("LocalWriteBaseAddrMXSB"), src1=sgpr("SwapMXSB"), comment=""))

  writer.vgprPool.checkIn(vgprWaveId)
  return module
