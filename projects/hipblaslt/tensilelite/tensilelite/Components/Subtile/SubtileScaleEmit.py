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
    SAddCU32, SAddU32, SAddU64, SAndB32, SSubU32, SLShiftLeftB32, SMovB32, SMovB64, SMulI32, SNop, SOrB32, SXorB32,
    TensorLoadToLds,
    VAddU32, VAndB32, VMulLOU32, VReadfirstlaneB32, VXorB32,
    VLShiftLeftB32, VLShiftRightB32,
    VSubU32, VCmpGEU32, VCndMaskB32,
)


# MXScaleLRB128 global scale layout (DGen::preSwizzleScalesGFX1250B128): each K
# group's M/N rows are padded to this multiple and interleaved within it.
SCALE_B128_ROW_GROUP = 128


def scaleLRB128Layout(kernel, tc):
  """LDS layout of one MXScaleLRB128 scale K group (A and B alike).

  M/N = group * 128 + block * 32 + lane, where a block is one A tile (32 rows)
  or one B tile pair (32 columns). The host stores each K group's 128-row group
  as [lane][block] dwords, so the TDM copies the MT's K group to LDS unchanged
  and lane L of wave partition p reads its four blocks as one 16 B load at
  p * 512 + L * 16.
  """
  wave = kernel["WavefrontSize"]
  blocks = SCALE_B128_ROW_GROUP // wave
  return {"blocks": blocks, "laneBytes": blocks * 4, "groupBytes": SCALE_B128_ROW_GROUP * 4,
          "kGroupBytes": kernel["MacroTile0" if tc == "MXSA" else "MacroTile1"] * 4}


def scaleRowPitch(mod, kernel, tc, tmpIdx):
  """Operand holding the scale tensor's M/N row count per K group.

  That is Size, except under MXScaleLRB128, whose layout pads it to
  SCALE_B128_ROW_GROUP (computed into sgpr tmpIdx).
  """
  size = sgpr("SizeI" if tc == "MXSA" else "SizeJ")
  if not kernel.get("MXScaleLRB128"):
    return size
  g = SCALE_B128_ROW_GROUP
  mod.add(SAddU32(dst=sgpr(tmpIdx), src0=size, src1=g - 1, comment=f"scale rows: Size + {g - 1}"))
  mod.add(SAndB32(dst=sgpr(tmpIdx), src0=sgpr(tmpIdx), src1=hex(~(g - 1) & 0xFFFFFFFF),
                  comment=f"scale rows: padded to {g}"))
  return sgpr(tmpIdx)


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
# Scale LR load
# ---------------------------------------------------------------------------

def emitScaleLRLoad(ti, writer, kernel):
  """Emit ds_read_b32 for all scale groups."""
  module = Module(f"Scale LR Load ({ti.tc})")
  tc = ti.tc

  if ti.mxBlock == 0:
    return module

  numScaleGroups = int(ti.lrLocalSubtileGrid[0]) * int(ti.lrLocalSubtileGrid[1])
  groupStride = int(ti.lrSubtileSize)
  tilesPerReg = max(1, int(round(1 / ti.mmaTileRegCount))) if ti.mmaTileRegCount else 1

  for gid in range(numScaleGroups):
    dsOffset = groupStride * gid
    vdst = ti.vgprTiles[tilesPerReg * gid].regList.indices[0]
    module.add(DSLoadB32(dst=vgpr(vdst),
               src=vgpr(ti.sharedVgprLROffset[0]),
               ds=DSModifiers(offset=dsOffset),
               comment=f"scale{tc}[group{gid}]: 4B from LDS"))

  return module


# ---------------------------------------------------------------------------
# Scale GR ptr update
# ---------------------------------------------------------------------------

def emitScaleTdmStepBytes(ti, kernel, dst, src, comment):
  """dst = src * (bytes of one DepthU of scales per M/N index).

  The TDM scale layout is {K/(mxBlock*mxUnit), M/N, mxUnit}, so one DepthU is
  DepthU / mxBlock bytes per M/N index (times bpe), and the per-iteration step is
  that times Size, not times MT (ti.depthUBytes).
  """
  mult = int(ti.scaleDepthU * ti.bpe)
  if mult & (mult - 1) == 0:
    return SLShiftLeftB32(dst=dst, shiftHex=int(math.log2(mult)), src=src, comment=f"{comment} ({mult})")
  return SMulI32(dst=dst, src0=src, src1=mult, comment=f"{comment} ({mult})")


def emitScaleGRPtrUpdate(ti, writer, kernel, holdOnLastIter=False):
  """Advance scale SRD / TDM address by one depthU iteration."""
  # Deferred: SubtileGREmit imports this module, so it cannot be imported here
  # at module scope.
  from .SubtileGREmit import emitSrdAdvance

  module = Module()
  tc = ti.tc
  if kernel.get("enableTDMA", False) and kernel.get("enableTDMB", False):
    sizeName = "SizeI" if tc == "MXSA" else "SizeJ"
    module.addComment0("TDM addr update: %s += %s * %u" % (tc, sizeName, ti.scaleDepthU * ti.bpe))
    with writer.allocTmpSgpr(1, tag="scaleTdmStep") as tmpSgprRes:
      inc = tmpSgprRes.idx
      rows = scaleRowPitch(module, kernel, tc, inc)
      module.add(emitScaleTdmStepBytes(ti, kernel, sgpr(inc), rows, "scale step"))
      module.add(SAddU32(dst=sgpr("Address%s+0" % tc), src0=sgpr("Address%s+0" % tc), src1=sgpr(inc)))
      module.add(SAddCU32(dst=sgpr("Address%s+1" % tc), src0=sgpr("Address%s+1" % tc), src1=0))
    # TDMSingleIssuer patches Address into the descriptor after each load.
    return module

  inc = int(ti.lrSubtileSize * ti.lrGlobalSubtileGrid[1])
  module.addComment0("Scale SRD update: %s += %u" % (tc, inc))
  emitSrdAdvance(module, tc, inc, writer, kernel, holdOnLastIter)
  # Dual-period MXFP8 only: shrink num_records in lock-step with the base advance
  # so base+num_records stays pinned to the tile's scale-buffer end.  computeLoadSrd
  # sized Srd+2 one full ScaleGroupSpan high ((numLine+1)*span) precisely to absorb
  # this shrink: across the K-loop the base slides one span forward, so the window
  # stays wide enough for every real top-group read and tightens to exactly the tile
  # end at the last real epoch -- clamping the trailing phantom prefetch (which would
  # otherwise read one group past the scale allocation and fault at deep K).  Other
  # swizzled-scale kernels (e.g. MXFP4) keep the legacy fixed num_records.
  if kernel.get("_DualPeriodMXFP8"):
    module.add(SSubU32(dst=sgpr(f"Srd{tc}+2"), src0=sgpr(f"Srd{tc}+2"), src1=inc,
                       comment="shrink num_records in lock-step with base advance"))
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


def _lrSwapParams(writer, tcName=""):
  """(nBuffers, strideBytes) for LR buffer advance of this tensor."""
  lay = getattr(writer, "ldsV7", None)
  tc = (tcName or "").replace("scale", "")
  if lay is not None and tc in ("MXSA", "MXSB"):
    return lay.sc_nbuf, lay.sc_buf
  if lay is not None and tc in ("A", "B"):
    return 2, lay.buf
  return writer.states.numLDSBlk, writer.ldsTotalSize


def emitLRBufferSwapOffsets(module, writer, kernel, offsets, swaps, tcName=""):
  """Advance LR read offsets to the next LDS buffer.

  With double buffering (numLDSBlk == 2) this is the classic xor toggle:
  ``offset ^= swapMask`` flips between buffer 0 and buffer 1.

  With triple+ buffering (numLDSBlk >= 3, TDMPlusLdsBuf) the xor trick cannot
  express a 3-way rotation, so we rotate explicitly::

      offset += stride                       # step to the next buffer
      if offset >= swaps[i]:                 # swaps[i] = origOffset + N*stride
          offset -= n * stride               # wrap back to buffer 0

  The compare is per-lane (each lane carries its own ceiling in ``swaps[i]``),
  so the rotation is self-contained: no shared counter and no assumption about
  the order in which the per-tensor swaps are emitted.
  """
  n, stride = _lrSwapParams(writer, tcName)
  if n < 3:
    for vOff, vSwap in zip(offsets, swaps):
      module.add(VXorB32(dst=vgpr(vOff), src0=vgpr(vOff), src1=vgpr(vSwap),
                         comment="%s LR buffer toggle" % tcName))
    return
  # Imm form matches ldsv7-w1: drop the two s_mov per rotate so the READY
  # swap block is VALU-only and shorter on the critical path before ds_load.
  span = n * stride
  laneCnt = 1 if kernel["WavefrontSize"] == 32 else 2
  with writer.allocTmpSgpr(laneCnt) as tmpRes:
    maskSgpr = tmpRes.idx
    vTmp = writer.vgprPool.checkOut(1, tag="emitLRBufferSwapOffsets_wrap")
    for vOff, vSwap in zip(offsets, swaps):
      module.add(VAddU32(dst=vgpr(vOff), src0=vgpr(vOff), src1=hex(stride),
                         comment="%s LR -> next LDS buffer" % tcName))
      module.add(VSubU32(dst=vgpr(vTmp), src0=vgpr(vOff), src1=hex(span),
                         comment="wrapped-to-buffer0 candidate"))
      module.add(VCmpGEU32(sgpr(maskSgpr, laneCnt), vgpr(vOff), vgpr(vSwap),
                           comment="past the last LDS buffer?"))
      module.add(VCndMaskB32(dst=vgpr(vOff), src0=vgpr(vOff), src1=vgpr(vTmp),
                             src2=sgpr(maskSgpr, laneCnt), comment="wrap LR offset to buffer 0"))
    writer.vgprPool.checkIn(vTmp)


def initLRBufferSwapCeil(module, writer, kernel, offsets, swaps, cstSgpr, tcName=""):
  """Initialise the per-lane LR swap register.

  n == 2 : swaps[i] = origOffset ^ (origOffset + stride)  (xor mask)
  n >= 3 : swaps[i] = origOffset + n*stride               (wrap ceiling)
  """
  n, stride = _lrSwapParams(writer, tcName)
  span   = (n * stride) if n >= 3 else stride
  module.add(SMovB32(dst=sgpr(cstSgpr), src=span,
                     comment="%s LR buffer span (%d)" % (tcName, span)))
  for vOff, vSwap in zip(offsets, swaps):
    module.add(VAddU32(dst=vgpr(vSwap), src0=vgpr(vOff), src1=sgpr(cstSgpr),
                       comment="%s LR swap: offset + span" % tcName))
    if n < 3:
      module.add(VXorB32(dst=vgpr(vSwap), src0=vgpr(vOff), src1=vgpr(vSwap),
                         comment="%s LR swap: xor toggle mask" % tcName))


def emitScaleLRLDSSwap(ti, writer, kernel):
  """Toggle/rotate scale LR read offsets between LDS buffers."""
  module = Module()
  module.addComment0("Emit code to swap %s LR vgpr offsets"%ti.tc)
  emitLRBufferSwapOffsets(module, writer, kernel,
                          ti.sharedVgprLROffset, ti.sharedVgprLROffsetSwap, tcName=ti.tc)
  return module


def emitScaleLRBufferAdvance(ti, writer, steps):
  """Move scale LR read offsets from buffer 0 to buffer ``steps`` (no wrap)."""
  n, stride = _lrSwapParams(writer, ti.tc)
  assert 0 < steps < n, f"{ti.tc}: LR advance by {steps} of {n} buffers"
  module = Module()
  for vOff in ti.sharedVgprLROffset:
    module.add(VAddU32(dst=vgpr(vOff), src0=vgpr(vOff), src1=hex(steps * stride),
                       comment="%s LR -> LDS buffer %d" % (ti.tc, steps)))
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
  scaleGroupSize = ti_.lrSubtileSize
  numThreadsPerGroup = (scaleGroupSize * int(ti_.lrGlobalSubtileGrid[1])) // loadWidth

  vtmp = writer.vgprPool.checkOut(1, tag="_graTileAssignmentScaleSwizzledCommon_vtmp")

  stmp = writer.sgprPool.checkOut(1, tag="_graTileAssignmentScaleSwizzledCommon_stmp")

  # numThreadsPerGroup is the scale K-subtile count times constants that are all
  # powers of two, and MX rejects a non-power-of-two DepthU, so the split is a
  # shift and a mask.
  splitComment = ("%s: groupId = serial / %u, threadId = serial %% %u"
                  % (tc, numThreadsPerGroup, numThreadsPerGroup))
  module.add(VLShiftRightB32(dst=vgpr(vtmp),
             shiftHex=hex(numThreadsPerGroup.bit_length() - 1), src=vgpr("Serial"),
             comment=splitComment))
  module.add(VAndB32(dst=vgpr(ti_.sharedVgprGROffset[0]),
             src0=hex(numThreadsPerGroup - 1), src1=vgpr("Serial"),
             comment=splitComment))
  module.add(SLShiftLeftB32(sgpr(stmp), int(math.log2(ti_.bpe)), sgpr("ScaleGroupSpan%s"%tc), comment="*= bpe (%d)"%(ti_.bpe)))

  module.add(VMulLOU32(dst=vgpr(vtmp), src1=vgpr(vtmp), src0=sgpr(stmp), comment="Apply scale%s stride to each group"%tc))
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
def _applyScaleWavePartitionLROffset(module, writer, kernel, ti_, waveId, totalScaleBytes=None):
  tc = ti_.tc

  # totalScaleBytes = bytes per wave partition in LDS for this scale tensor.
  # lrGlobalSubtileGrid[0] = M-dim LR subtile count (globalMMATileGrid[0] / lrSubtileShape[0])
  # lrGlobalSubtileGrid[1] = K-dim LR subtile count
  # lrSubtileSize = bytes per LR subtile (2x2 MMA tiles for FP4 scale)
  index = 0 if tc == 'MXSA' else 1
  if totalScaleBytes is None:
    totalScaleBytes = (int(ti_.lrGlobalSubtileGrid[0]) // kernel["MIWaveGroup"][index]) * int(ti_.lrGlobalSubtileGrid[1]) * int(ti_.lrSubtileSize)

  tmpSgpr = writer.sgprPool.checkOut(1, tag="_applyScaleWavePartitionLROffset_tmpSgpr")
  tmp = writer.vgprPool.checkOut(2, tag="_applyScaleWavePartitionLROffset_tmp")

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
  waveIdVgpr = writer.vgprPool.checkOut(1, tag="_lraTileAssignmentScaleSwizzled_legacy_waveIdVgpr")
  module.add(VLShiftRightB32(dst=vgpr(waveIdVgpr), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="scale: waveId"))
  if kernel.get("MXScaleLRB128"):
    # One 128-row group per wave partition, [lane][block] dwords (scaleLRB128Layout).
    for ti_ in [tiA_, tiB_]:
      lay = scaleLRB128Layout(kernel, ti_.tc)
      _applyScaleWavePartitionLROffset(module, writer, kernel, ti_, waveIdVgpr,
                                       totalScaleBytes=lay["groupBytes"])
    laneBytes = lay["laneBytes"]
  else:
    laneBytes = 4
    _applyScaleWavePartitionLROffset(module, writer, kernel, tiA_, waveIdVgpr)
    _applyScaleWavePartitionLROffset(module, writer, kernel, tiB_, waveIdVgpr)
  writer.vgprPool.checkIn(waveIdVgpr)
  laneOffset = writer.vgprPool.checkOut(1, tag="_lraTileAssignmentScaleSwizzled_legacy_laneOffset")
  module.add(VAndB32(dst=vgpr(laneOffset), src0=vgpr("Serial"), src1=wavesize-1, comment="scale: laneId"))
  # 4B/lane. Each gfx1250 A/B MMA scale tile uses a distinct VGPR.
  module.add(VLShiftLeftB32(dst=vgpr(laneOffset), shiftHex=hex(int(math.log2(laneBytes))), src=vgpr(laneOffset), comment="scale: laneId * %u" % laneBytes))
  module.add(VAddU32(dst=vgpr(tiA_.sharedVgprLROffset[0]), src0=vgpr(laneOffset), src1=vgpr(tiA_.sharedVgprLROffset[0]), comment="scaleA: lrOffset = laneId * %u" % laneBytes))
  module.add(VAddU32(dst=vgpr(tiB_.sharedVgprLROffset[0]), src0=vgpr(laneOffset), src1=vgpr(tiB_.sharedVgprLROffset[0]), comment="scaleB: lrOffset = laneId * %u" % laneBytes))
  writer.vgprPool.checkIn(laneOffset)
  tmpSgpr = writer.sgprPool.checkOut(1, tag="_lraTileAssignmentScaleSwizzled_legacy_tmpSgpr")
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(writer.ldsStartOffsetMXSA), comment="scale: LDS offset for A scale"))
  module.add(VAddU32(dst=vgpr(tiA_.sharedVgprLROffset[0]), src0=vgpr(tiA_.sharedVgprLROffset[0]), src1=sgpr(tmpSgpr), comment="scaleA: +=LDS offset"))
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(writer.ldsStartOffsetMXSB), comment="scale: LDS offset for B scale"))
  module.add(VAddU32(dst=vgpr(tiB_.sharedVgprLROffset[0]), src0=vgpr(tiB_.sharedVgprLROffset[0]), src1=sgpr(tmpSgpr), comment="scaleB: +=LDS offset"))
  for ti_ in [tiA_, tiB_]:
    initLRBufferSwapCeil(module, writer, kernel,
                         ti_.sharedVgprLROffset, ti_.sharedVgprLROffsetSwap,
                         tmpSgpr, tcName="scale%s" % ti_.tc)
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

  if kernel.get("enableTDMA", False) and kernel.get("enableTDMB", False):
    if kernel.get("TDMSingleIssuer"):
      from .SubtileGREmit import emitTDMSingleIssuerLoad
      module.add(emitTDMSingleIssuerLoad(writer, kernel, tc))
      return module
    module.addComment0("Scale GR: %s (TDM: tensor_load_to_lds)" % tc)
    group0 = "tdm%sGroup0" % tc
    group1 = "tdm%sGroup1" % tc
    module.add(TensorLoadToLds(sgpr(group0, 4), sgpr(group1, 8), None, None,
                               comment="TDM: global->LDS for %s" % tc))
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
def emitSubtileScaleDsRead(tc, writer, kernel, scaleGroupIdx):
  """Emit a single DSLoadB32 for a scale group (2 M-adjacent [1,2] subtiles).
  Each ds_read_b32 loads 4 bytes = 4 E8M0 scale values into one VGPR."""
  module = Module()
  tileInfo = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo

  if tileInfo.mxBlock == 0:
    return module

  # TileInfo LR subtile size is the packed group: gfx950 (2,2), gfx1250 (1,1).
  if hasattr(tileInfo, 'lrSubtileSize'):
    groupStride = int(tileInfo.lrSubtileSize)
  else:
    groupStride = 2 * tileInfo.subtileSize
  tilesPerReg = max(1, int(round(1 / tileInfo.mmaTileRegCount))) if tileInfo.mmaTileRegCount else 1
  dsOffset = groupStride * scaleGroupIdx
  vdst = tileInfo.vgprTiles[tilesPerReg * scaleGroupIdx].regList.indices[0]
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

  # One ds_read per LR scale subtile. gfx1250 A and B are both (1,1).
  numScaleGroups = int(tileInfo.lrLocalSubtileGrid[0]) * int(tileInfo.lrLocalSubtileGrid[1])
  for gid in range(numScaleGroups):
    module.add(emitSubtileScaleDsRead(tc, writer, kernel, gid))

  return module

##################################################
# Scale SRD pointer update: advance scale SRD by scaleDepthU * scaleBpe bytes.
#
def globalReadScalePtrUpdates(tc, writer, kernel, holdOnLastIter=False):
  ti_ = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
  return emitScaleGRPtrUpdate(ti_, writer, kernel, holdOnLastIter)

##################################################
# Subroutine to generate DTL M0 LDS buffer swap
#
# For Swizzled Scales each wave will collectively stream
# the scale values
#
def globalReadScaleSwizzledDTLInitCommonSgpr(writer, kernel):
  module = Module()

  wavesize = kernel["WavefrontSize"]
  vgprWaveId = writer.vgprPool.checkOut(1, tag="globalReadScaleSwizzledDTLInitCommonSgpr_vgprWaveId")
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
