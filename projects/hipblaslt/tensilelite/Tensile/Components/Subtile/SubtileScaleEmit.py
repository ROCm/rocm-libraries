# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

################################################################################
# Scale GR/LR emit for MX scale factor operands (MXSA/MXSB).
#
# HostPreSwizzle (gfx950 BLK32):
#   GR: DTL with collective wave-group offsets into HostPreSwizzle-shaped LDS
#   LR: ds_read_b32 per scale group (laneId*4 + wave partition)
#
# NoSwizzle (VEC32 / scaleA=3) — canonical gather -> HostPreSwizzle-shaped LDS:
#   Pure DTL with LDS==canonical global cannot feed the existing laneId*4 /
#   opsel packing: each lane's 4 MFMA scale bytes are non-contiguous in
#   UMLDS=1 canonical layout (m, kb), (m+16, kb), (m, kb+4), (m+16, kb+4).
#   NoSwizzle therefore keeps the proven HostPreSwizzle LDS+LR shape and
#   remaps at GR time:
#     buffer_load_u8 x4 (canonical global) -> pack -> ds_store_b32
#     (HostPreSwizzle-shaped slot).
#
# Uses ti.sharedVgprGROffset / ti.sharedVgprLROffset (compat properties)
# since MXScaleTilePair has gr=None, lr=None.
################################################################################

import math

from rocisa.code import Module
from rocisa.container import DSModifiers, MUBUFModifiers, vgpr, sgpr, mgpr
from rocisa.instruction import (
    BufferLoadB128, BufferLoadU8,
    DSLoadB32, DSStoreB32,
    SAddCU32, SAddU32, SLShiftLeftB32, SMovB32, SNop, SXorB32,
    SWaitCnt,
    VAddU32, VAndB32, VMovB32, VMulLOU32, VOrB32, VReadfirstlaneB32, VXorB32,
    VLShiftLeftB32, VLShiftRightB32,
)


def _isMxSwizzledScaleFormat(kernel):
  """True when MX scales use HostPreSwizzle or InMemorySwizzle LDS packing."""
  return kernel.get("MXScaleFormat", "NoSwizzle") in ("HostPreSwizzle", "InMemorySwizzle")


def scaleGRPtrIncBytes(ti, kernel):
  """Per-macro-iteration scale SRD advance/rewind in bytes.

  Must stay in lockstep for emitScaleGRPtrUpdate (advance) and
  Kernel._emitMultiDUTailSrdRewind (undo one PGR=1 prefetch over-advance):

    * HostPreSwizzle / InMemorySwizzle: one swizzle granule
      (lrSubtileSize * lrGlobalSubtileGrid[1]).
    * NoSwizzle: canonical K-step (scaleDepthU * bpe).
  """
  if _isMxSwizzledScaleFormat(kernel):
    return int(ti.lrSubtileSize * ti.lrGlobalSubtileGrid[1])
  return int(ti.scaleDepthU * ti.bpe)


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

def emitScaleGRPtrUpdate(ti, writer, kernel):
  """Advance scale SRD base pointer by one depthU iteration."""
  module = Module()
  tc = ti.tc

  # HostPreSwizzle / InMemorySwizzle: one swizzle granule (StreamK USO requires
  # DepthU % 256 == 0 so a K-cut lands on a granule boundary;
  # StreamK._depthUForTc applies matching *32). NoSwizzle: canonical
  # scaleDepthU*bpe (StreamK._depthUForTc unscaled).
  inc = scaleGRPtrIncBytes(ti, kernel)
  module.addComment0("Scale SRD update: %s += %u" % (tc, inc))
  module.add(SAddU32(dst=sgpr(f"Srd{tc}"), src0=sgpr(f"Srd{tc}"), src1=inc))
  module.add(SAddCU32(dst=sgpr(f"Srd{tc}+1"), src0=sgpr(f"Srd{tc}+1"), src1=0))
  return module


# ---------------------------------------------------------------------------
# Scale LDS buffer swaps
# ---------------------------------------------------------------------------

def emitScaleGRLDSSwap(ti, writer, kernel):
  """Toggle scale GR write target between double-buffer halves."""
  module = Module()
  tc = ti.tc
  if _isMxSwizzledScaleFormat(kernel):
    module.addComment0("Emit code to swap %s GR m0 offsets"%tc)
    module.add(SXorB32(dst=sgpr(f"LocalWriteBaseAddr{tc}"),
               src0=sgpr(f"LocalWriteBaseAddr{tc}"), src1=sgpr(f"Swap{tc}"),
               comment=""))
  else:
    # NoSwizzle: GR uses VGPR LDS write offsets (same layout as LR).
    module.addComment0("Emit code to swap %s GR vgpr LDS offsets"%tc)
    for i in range(len(ti.sharedVgprGROffset)):
      vOff = ti.sharedVgprGROffset[i]
      vSwap = ti.sharedVgprGROffsetSwap[i]
      module.add(VXorB32(dst=vgpr(vOff), src0=vgpr(vOff), src1=vgpr(vSwap), comment=""))
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


def graTileAssignmentScale(writer, kernel):
  """Dispatch scale GR tile assignment on MXScaleFormat."""
  if _isMxSwizzledScaleFormat(kernel):
    return graTileAssignmentScaleSwizzled(writer, kernel)
  return graTileAssignmentScaleNoSwizzle(writer, kernel)


##################################################
# NoSwizzle: init GR LDS write offsets in HostPreSwizzle-friendly layout.
# Each lane will gather 4 canonical scale bytes and ds_store them here.
#
def _graTileAssignmentScaleNoSwizzleCommon(tc, writer, kernel, waveIdVgpr, laneOffsetVgpr, tmpSgpr):
  """Init sharedVgprGROffset / GROffsetSwap for NoSwizzle (HostPreSwizzle-shaped LDS)."""
  module = Module()
  ti_ = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
  module.addComment0("NoSwizzle scale GR LDS write offset for %s (HostPreSwizzle-shaped)" % tc)

  # Reuse wave-partition math by writing into GR offset via a temporary ti_ alias:
  # Compute partition * totalScaleBytes into sharedVgprGROffset.
  index = 0 if tc == 'MXSA' else 1
  totalScaleBytes = (int(ti_.lrGlobalSubtileGrid[0]) // kernel["MIWaveGroup"][index]) * int(ti_.lrGlobalSubtileGrid[1]) * int(ti_.lrSubtileSize)

  tmp = writer.vgprPool.checkOut(2, tag="_graNoSwizzle_tmp")
  if tc == 'MXSA':
    module.add(VAndB32(dst=vgpr(tmp), src0=kernel["MIWaveGroup"][0]-1, src1=vgpr(waveIdVgpr),
               comment="scale%s: waveId %% MIWG[0]"%tc))
  else:
    module.add(VLShiftRightB32(dst=vgpr(tmp), shiftHex=int(math.log2(kernel["MIWaveGroup"][0])),
               src=vgpr(waveIdVgpr), comment="scale%s: waveId / MIWG[0]"%tc))
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=totalScaleBytes, comment="scale%s: partition stride"%tc))
  module.add(VMulLOU32(dst=vgpr(ti_.sharedVgprGROffset[0]), src0=sgpr(tmpSgpr), src1=vgpr(tmp),
             comment="scale%s: partition offset"%tc))
  writer.vgprPool.checkIn(tmp)

  module.add(VAddU32(dst=vgpr(ti_.sharedVgprGROffset[0]),
             src0=vgpr(laneOffsetVgpr), src1=vgpr(ti_.sharedVgprGROffset[0]),
             comment="scale%s: + laneId*4"%tc))

  ldsStartOffset = getattr(writer, f'ldsStartOffset{tc}', 0)
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(ldsStartOffset), comment="scale%s: LDS base"%tc))
  module.add(VAddU32(dst=vgpr(ti_.sharedVgprGROffset[0]),
             src0=vgpr(ti_.sharedVgprGROffset[0]), src1=sgpr(tmpSgpr),
             comment="scale%s: += LDS base"%tc))

  module.add(SMovB32(dst=sgpr(tmpSgpr), src=writer.ldsTotalSize, comment="scale%s: ldsTotalSize"%tc))
  for i in range(len(ti_.sharedVgprGROffset)):
    vOff = ti_.sharedVgprGROffset[i]
    vSwap = ti_.sharedVgprGROffsetSwap[i]
    module.add(VAddU32(dst=vgpr(vSwap), src0=vgpr(vOff), src1=sgpr(tmpSgpr),
               comment="scale%s: GR swap init"%tc))
    module.add(VXorB32(dst=vgpr(vSwap), src0=vgpr(vOff), src1=vgpr(vSwap),
               comment="scale%s: GR swap mask"%tc))
  return module


def graTileAssignmentScaleNoSwizzle(writer, kernel):
  """Init NoSwizzle scale GR LDS write offsets (HostPreSwizzle-shaped slots)."""
  module = Module()
  if not kernel["ProblemType"].get("MXBlockA", 0) and not kernel["ProblemType"].get("MXBlockB", 0):
    return module
  module.addComment0("NoSwizzle scale GR tile assignment (remap into HostPreSwizzle-shaped LDS)")
  wavesize = kernel["WavefrontSize"]
  waveIdVgpr = writer.vgprPool.checkOut(1, tag="graNoSwizzle_waveId")
  module.add(VLShiftRightB32(dst=vgpr(waveIdVgpr), shiftHex=hex(wavesize.bit_length()-1),
             src=vgpr("Serial"), comment="scale: waveId"))
  laneOffset = writer.vgprPool.checkOut(1, tag="graNoSwizzle_laneOffset")
  module.add(VAndB32(dst=vgpr(laneOffset), src0=vgpr("Serial"), src1=wavesize-1, comment="scale: laneId"))
  module.add(VLShiftLeftB32(dst=vgpr(laneOffset), shiftHex=hex(2), src=vgpr(laneOffset),
             comment="scale: laneId * 4"))
  tmpSgpr = writer.sgprPool.checkOut(1, tag="graNoSwizzle_tmpSgpr")
  module.add(_graTileAssignmentScaleNoSwizzleCommon('MXSA', writer, kernel, waveIdVgpr, laneOffset, tmpSgpr))
  module.add(_graTileAssignmentScaleNoSwizzleCommon('MXSB', writer, kernel, waveIdVgpr, laneOffset, tmpSgpr))
  writer.sgprPool.checkIn(tmpSgpr)
  writer.vgprPool.checkIn(laneOffset)
  writer.vgprPool.checkIn(waveIdVgpr)
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
def _applyScaleWavePartitionLROffset(module, writer, kernel, ti_, waveId):
  tc = ti_.tc

  # totalScaleBytes = bytes per wave partition in LDS for this scale tensor.
  # lrGlobalSubtileGrid[0] = M-dim LR subtile count (globalMMATileGrid[0] / lrSubtileShape[0])
  # lrGlobalSubtileGrid[1] = K-dim LR subtile count
  # lrSubtileSize = bytes per LR subtile (2x2 MMA tiles for FP4 scale)
  index = 0 if tc == 'MXSA' else 1
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
def lraTileAssignmentScale(writer, kernel):
  """Dispatch scale LR tile assignment.

  NoSwizzle remaps into the same HostPreSwizzle-shaped LDS layout, so LR
  offsets are identical for HostPreSwizzle and NoSwizzle.
  """
  return lraTileAssignmentScaleSwizzled(writer, kernel)


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
  _applyScaleWavePartitionLROffset(module, writer, kernel, tiA_, waveIdVgpr)
  _applyScaleWavePartitionLROffset(module, writer, kernel, tiB_, waveIdVgpr)
  writer.vgprPool.checkIn(waveIdVgpr)
  laneOffset = writer.vgprPool.checkOut(1, tag="_lraTileAssignmentScaleSwizzled_legacy_laneOffset")
  module.add(VAndB32(dst=vgpr(laneOffset), src0=vgpr("Serial"), src1=wavesize-1, comment="scale: laneId"))
  module.add(VLShiftLeftB32(dst=vgpr(laneOffset), shiftHex=hex(2), src=vgpr(laneOffset), comment="scale: laneId * 4"))
  module.add(VAddU32(dst=vgpr(tiA_.sharedVgprLROffset[0]), src0=vgpr(laneOffset), src1=vgpr(tiA_.sharedVgprLROffset[0]), comment="scaleA: lrOffset = laneId * 4"))
  module.add(VAddU32(dst=vgpr(tiB_.sharedVgprLROffset[0]), src0=vgpr(laneOffset), src1=vgpr(tiB_.sharedVgprLROffset[0]), comment="scaleB: lrOffset = laneId * 4"))
  writer.vgprPool.checkIn(laneOffset)
  tmpSgpr = writer.sgprPool.checkOut(1, tag="_lraTileAssignmentScaleSwizzled_legacy_tmpSgpr")
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
# Scale GR: Load scale bytes from global memory to LDS.
#
# HostPreSwizzle: BufferLoadB128 DTL (vaddr = global + LDS offset).
# NoSwizzle: per-lane gather of 4 canonical bytes + ds_store into the
# HostPreSwizzle-shaped LDS slot (sharedVgprGROffset).
#
def globalReadDoScaleSubtile(tc, writer, kernel):
  module = Module()

  if not kernel["ProblemType"].get("MXBlockA", 0) and not kernel["ProblemType"].get("MXBlockB", 0):
    return module

  if not _isMxSwizzledScaleFormat(kernel):
    return _globalReadDoScaleNoSwizzle(tc, writer, kernel)

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


def _globalReadDoScaleNoSwizzle(tc, writer, kernel):
  """NoSwizzle GR: gather 4 canonical scale bytes/lane into HostPreSwizzle-shaped LDS.

  Per scale group g and lane L in a wave partition:
    m0 = m_base + (L % 16);  k_lo = k_base + (L // 16)
    bytes = scale(m0,k_lo), scale(m0+16,k_lo), scale(m0,k_lo+4), scale(m0+16,k_lo+4)
  which matches MFMA opsel packing used by the HostPreSwizzle LR path.
  """
  module = Module()
  tileInfo = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
  assert len(tileInfo.sharedVgprGROffset) > 0, "Scale GR requires at least 1 GR offset VGPR"

  wavesize = kernel["WavefrontSize"]
  index = 0 if tc == 'MXSA' else 1
  numKGroups = max(1, int(tileInfo.lrGlobalSubtileGrid[1]))
  numMGroupsPerWave = max(1, int(tileInfo.lrGlobalSubtileGrid[0]) // kernel["MIWaveGroup"][index])
  numScaleGroups = numMGroupsPerWave * numKGroups
  # 2x2 LR subtile covers 32 M rows x 8 K-scale columns.
  mPerGroup = int(tileInfo.lrSubtileShape[0] * tileInfo.mmaTileShape[0])  # 2*16=32
  kPerGroup = int(tileInfo.lrSubtileShape[1] * tileInfo.mmaTileShape[1])  # 2*4=8

  isGlc = bool(kernel["NonTemporal%s"%tc] & 0x1)
  isSlc = bool(kernel["NonTemporal%s"%tc] & 0x2)
  isNT  = bool(kernel["NonTemporal%s"%tc] & 0x4)
  mubuf = MUBUFModifiers(offen=True, offset12=0, glc=isGlc, slc=isSlc, nt=isNT, lds=False)

  module.addComment0(
      "Scale GR: %s NoSwizzle (canonical gather -> HostPreSwizzle-shaped LDS), %u groups"
      % (tc, numScaleGroups))

  # Compact VGPR temps (peak matters for occupancy). Reuse waveId slot as partIdx.
  laneId = writer.vgprPool.checkOut(1, tag="nsScaleGR_laneId")
  partIdx = writer.vgprPool.checkOut(1, tag="nsScaleGR_part")
  module.add(VAndB32(dst=vgpr(laneId), src0=vgpr("Serial"), src1=wavesize-1, comment="scale%s: laneId"%tc))
  module.add(VLShiftRightB32(dst=vgpr(partIdx), shiftHex=hex(wavesize.bit_length()-1),
             src=vgpr("Serial"), comment="scale%s: waveId"%tc))
  if tc == 'MXSA':
    module.add(VAndB32(dst=vgpr(partIdx), src0=kernel["MIWaveGroup"][0]-1, src1=vgpr(partIdx),
               comment="scale%s: partition"%tc))
  else:
    module.add(VLShiftRightB32(dst=vgpr(partIdx),
               shiftHex=int(math.log2(kernel["MIWaveGroup"][0])), src=vgpr(partIdx),
               comment="scale%s: partition"%tc))

  # m0_lane = laneId % 16; k_lo_lane = laneId // 16 — reuse laneId as m0 after extracting k_lo
  kLoLane = writer.vgprPool.checkOut(1, tag="nsScaleGR_klo")
  module.add(VLShiftRightB32(dst=vgpr(kLoLane), shiftHex=4, src=vgpr(laneId),
             comment="scale%s: laneId//16"%tc))
  module.add(VAndB32(dst=vgpr(laneId), src0=15, src1=vgpr(laneId), comment="scale%s: laneId%%16 (=m0)"%tc))
  m0Lane = laneId

  # Temps for addresses / loaded bytes / pack (4 byte vgprs + addr + m_base)
  vAddr = writer.vgprPool.checkOut(1, tag="nsScaleGR_vaddr")
  vBytes = [writer.vgprPool.checkOut(1, tag="nsScaleGR_b%d"%i) for i in range(4)]
  vTmp = writer.vgprPool.checkOut(1, tag="nsScaleGR_vtmp")
  stmp = writer.sgprPool.checkOut(1, tag="nsScaleGR_stmp")

  for gid in range(numScaleGroups):
    mGroupLocal = gid // numKGroups
    kGroup = gid % numKGroups
    # m_base = (partition * numMGroupsPerWave + mGroupLocal) * mPerGroup
    # k_base = kGroup * kPerGroup
    module.addComment0("scale%s group %u: mGroup=%u kGroup=%u" % (tc, gid, mGroupLocal, kGroup))
    module.add(SMovB32(dst=sgpr(stmp), src=numMGroupsPerWave * mPerGroup,
               comment="scale%s: M rows per partition"%tc))
    module.add(VMulLOU32(dst=vgpr(vTmp), src0=sgpr(stmp), src1=vgpr(partIdx),
               comment="scale%s: partition * Mrows"%tc))
    module.add(SMovB32(dst=sgpr(stmp), src=mGroupLocal * mPerGroup,
               comment="scale%s: mGroup M offset"%tc))
    module.add(VAddU32(dst=vgpr(vTmp), src0=vgpr(vTmp), src1=sgpr(stmp),
               comment="scale%s: m_base"%tc))
    # m = m_base + m0Lane  (byte0/2 M); m16 = m + 16 (byte1/3 M)
    module.add(VAddU32(dst=vgpr(vTmp), src0=vgpr(vTmp), src1=vgpr(m0Lane),
               comment="scale%s: m = m_base + lane%%16"%tc))

    kBase = kGroup * kPerGroup
    # Four (m, kb) pairs → four buffer_load_u8
    # b0: (m, k_base+k_lo), b1: (m+16, k_base+k_lo), b2: (m, k_base+k_lo+4), b3: (m+16, k_base+k_lo+4)
    for bi, (mAdd, kAdd) in enumerate(((0, 0), (16, 0), (0, 4), (16, 4))):
      # vAddr = (m + mAdd) * Stride + (k_base + k_lo + kAdd)
      if mAdd:
        module.add(SMovB32(dst=sgpr(stmp), src=mAdd, comment="scale%s: +%u M"% (tc, mAdd)))
        module.add(VAddU32(dst=vgpr(vAddr), src0=vgpr(vTmp), src1=sgpr(stmp),
                   comment="scale%s: m+%u"% (tc, mAdd)))
      else:
        module.add(VMovB32(dst=vgpr(vAddr), src=vgpr(vTmp), comment="scale%s: m"%tc))
      module.add(VMulLOU32(dst=vgpr(vAddr), src0=vgpr(vAddr), src1=sgpr("Strides%s"%tc),
                 comment="scale%s: m * Stride"%tc))
      module.add(VAddU32(dst=vgpr(vAddr), src0=vgpr(vAddr), src1=vgpr(kLoLane),
                 comment="scale%s: + k_lo"%tc))
      if kBase + kAdd:
        module.add(SMovB32(dst=sgpr(stmp), src=kBase + kAdd,
                   comment="scale%s: k_base+kAdd"%tc))
        module.add(VAddU32(dst=vgpr(vAddr), src0=vgpr(vAddr), src1=sgpr(stmp),
                   comment="scale%s: + k_base+%u"% (tc, kAdd)))
      module.add(BufferLoadU8(dst=vgpr(vBytes[bi]), vaddr=vgpr(vAddr),
                 saddr=sgpr("Srd%s"%tc, 4), soffset=0, mubuf=mubuf,
                 comment="scale%s[g%u].b%u canonical load" % (tc, gid, bi)))

    # Wait for the 4 loads of this group, then pack little-endian into one VGPR.
    # Plain SWaitCnt (no adjustVmcnt): InstructionScheduler must leave vlcnt=0
    # alone. Weakening it to vmcnt(N) via the WaitGR post-pass returns before
    # these gathers complete and breaks NoSwizzle mainloop for itersPerTile>=3.
    module.add(SWaitCnt(vlcnt=0, comment="scale%s: wait gather loads group %u"% (tc, gid)))
    # packed = b0 | (b1<<8) | (b2<<16) | (b3<<24)
    module.add(VLShiftLeftB32(dst=vgpr(vBytes[1]), shiftHex=hex(8), src=vgpr(vBytes[1]),
               comment="scale%s: b1<<8"%tc))
    module.add(VLShiftLeftB32(dst=vgpr(vBytes[2]), shiftHex=hex(16), src=vgpr(vBytes[2]),
               comment="scale%s: b2<<16"%tc))
    module.add(VLShiftLeftB32(dst=vgpr(vBytes[3]), shiftHex=hex(24), src=vgpr(vBytes[3]),
               comment="scale%s: b3<<24"%tc))
    module.add(VOrB32(dst=vgpr(vBytes[0]), src0=vgpr(vBytes[0]), src1=vgpr(vBytes[1]),
               comment="scale%s: pack b0|b1"%tc))
    module.add(VOrB32(dst=vgpr(vBytes[0]), src0=vgpr(vBytes[0]), src1=vgpr(vBytes[2]),
               comment="scale%s: pack |b2"%tc))
    module.add(VOrB32(dst=vgpr(vBytes[0]), src0=vgpr(vBytes[0]), src1=vgpr(vBytes[3]),
               comment="scale%s: pack |b3"%tc))

    dsOff = int(tileInfo.lrSubtileSize) * gid
    module.add(DSStoreB32(dstAddr=vgpr(tileInfo.sharedVgprGROffset[0]),
               src=vgpr(vBytes[0]),
               ds=DSModifiers(offset=dsOff),
               comment="scale%s[g%u]: ds_store HostPreSwizzle-shaped slot"% (tc, gid)))

  module.add(SWaitCnt(dscnt=0, comment="scale%s: wait ds_store gathers"%tc))

  for v in vBytes:
    writer.vgprPool.checkIn(v)
  writer.vgprPool.checkIn(vTmp)
  writer.vgprPool.checkIn(vAddr)
  writer.vgprPool.checkIn(kLoLane)
  writer.vgprPool.checkIn(partIdx)
  writer.vgprPool.checkIn(laneId)  # m0Lane alias
  writer.sgprPool.checkIn(stmp)
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
def globalReadScalePtrUpdates(tc, writer, kernel):
  ti_ = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
  return emitScaleGRPtrUpdate(ti_, writer, kernel)

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


def globalReadScaleNoSwizzleInitCommonSgpr(writer, kernel):
  """NoSwizzle uses VGPR LDS write offsets; no M0/SGPR DTL bases."""
  module = Module()
  module.addComment0("NoSwizzle scale GR: LDS write offsets set in graTileAssignmentScaleNoSwizzle")
  return module


def globalReadScaleDTLInitCommonSgpr(writer, kernel):
  """Dispatch scale DTL/GR init on MXScaleFormat."""
  if _isMxSwizzledScaleFormat(kernel):
    return globalReadScaleSwizzledDTLInitCommonSgpr(writer, kernel)
  return globalReadScaleNoSwizzleInitCommonSgpr(writer, kernel)
