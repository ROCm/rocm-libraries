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
    DSLoadB32, DSLoadB64,
    SAddCU32, SAddU32, SLShiftLeftB32, SMovB32, SMulI32, SNop, SWaitCnt, SXorB32,
    VAddU32, VAndB32, VMovB32, VMulLOU32, VPermB32, VReadfirstlaneB32, VXorB32,
    VLShiftLeftB32, VLShiftRightB32,
)

from .SubtileGeometry import subtileInterleaveVW


# ---------------------------------------------------------------------------
# Scale GR offset
# ---------------------------------------------------------------------------

def emitScaleGROffset(ti, writer, kernel):
  """Compute per-thread DTL vaddr for scale GR load."""
  return Module(f"Scale GR Offset ({ti.tc})")  # STUB
  module = Module(f"Scale GR Offset ({ti.tc})")
  tc = ti.tc
  loadWidth = ti.loadWidthGR
  loadWidthShift = loadWidth.bit_length() - 1

  scaleGroupSize = ti.lrSubtileSize
  numThreadsPerGroup = (scaleGroupSize * int(ti.localSubtileGrid[1])) // loadWidth

  vtmp = writer.vgprPool.checkOut(1, tag="emitScaleGROffset_vtmp")
  stmp = writer.sgprPool.checkOut(1, tag="emitScaleGROffset_stmp")

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

  waveIdVgpr = writer.vgprPool.checkOut(1, tag="emitScaleLROffset_waveIdVgpr")
  module.add(VLShiftRightB32(dst=vgpr(waveIdVgpr), shiftHex=hex(wavesize.bit_length()-1),
             src=vgpr("Serial"), comment=f"scale{tc}: waveId"))

  vtmp = writer.vgprPool.checkOut(1, tag="emitScaleLROffset_vtmp")
  stmp = writer.sgprPool.checkOut(1, tag="emitScaleLROffset_stmp")

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
  laneOffset = writer.vgprPool.checkOut(1, tag="emitScaleLROffset_laneOffset")
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
##################################################
# Scale LR offset under SourceSwap.
#
# The scale bytes in LDS are in the host's blocked order: LDS byte q belongs to
# global line q//256, and within a line byte 4*L' + p + 2k carries M row
# 32*line + 16p + (L'%16) for K-block 2*(L'//16) + k.
#
# Without SourceSwap, lane L of tile j reads at 4*L with a per-group offset of
# 256*(j//2), which resolves to M row 16j + (L%16) - exactly the blocked row that
# lane computes. SourceSwap changes the data map so the same lane computes M row
# 8a + j instead (a = L%16), while this offset was left alone: every lane read a
# scale belonging to a different row. It went unnoticed because the harness
# generated scales that did not vary across M rows.
#
# Solving VW*a + j for the LDS position gives, with a = L%16, c = L//16 and
# t = VW*a (so t's bit slices are shifts of the lane id, VW being a power of two):
#   line   = t//32         (independent of j, so it moves into the lane base)
#   dword  = (t%16) + j + 16c
#   byte   = p' + 2k with p' = (t%32)//16   (independent of j, but LANE dependent)
# so the per-lane base becomes 256*(t//32) + 64*c + 4*(t%16) and the per-tile part
# is 4*j, which is why the group offset shrinks from 256 to 4*VW (see the LR emit).
# At VW=8 that is 256*(a//4) + 64*c + 32*(a%2) with p' = (a%4)//2 and a group
# stride of 8.
#
# Under the two-level map (MIWaveTile > VW) subtile row j is row
# (j%VW) + 16*VW*(j//VW); 16*VW is a whole number of 32-row lines for VW >= 2, so
# j//VW only shifts the line and the decomposition above is unchanged. The per-tile
# part becomes 4*(j%VW) + 128*VW*(j//VW).
#
# p' being lane dependent is the reason the ds_read cannot stay a plain b32: the
# MFMA selects its scale byte with a compile-time field, so the byte has to be
# moved into a fixed position first. That is what the v_perm in the LR emit does.
def scaleInterleaveVW(kernel, tc) -> int:
  """The interleave factor of the data map for the tensor `tc` scales.

  Accepts either the data name ('A'/'B') or the scale name ('MXSA'/'MXSB').
  """
  return subtileInterleaveVW(kernel, 'A' if tc in ('A', 'MXSA') else 'B')


def scaleRemapEnabled(kernel, tc=None) -> bool:
  """True when the scale rows have to follow an interleaved data local-read map.

  The data map is interleaved only when SourceSwap is on *and* that tensor's
  VectorWidth exceeds 1 (this mirrors TileInfo.lrInterleaveVW). With VectorWidth 1
  the data map stays blocked, and the scale path has to stay blocked with it --
  the two coincide, since the two-level map at VW=1 *is* the blocked map.

  `tc` selects one tensor; None asks whether any tensor is remapped, which is what
  the register-allocation sites need (they size both scale tiles alike)."""
  if not kernel.get("SourceSwap", False):
    return False
  tcs = ('A', 'B') if tc is None else (tc,)
  return any(scaleInterleaveVW(kernel, t) > 1 for t in tcs)


def _emitScaleLaneBaseInterleaved(module, acc, tmp, laneId, vw, tag):
  """Emit acc = 256*(a//(32/vw)) + 64*c + 4*vw*(a%(16/vw)), a = laneId%16, c = laneId//16.

  This is 256*(t//32) + 64*c + 4*(t%16) for t = vw*a, rewritten so every term is a
  bit slice of the lane id (vw is a power of two).
  """
  v = vw.bit_length() - 1
  lineDiv = 32 // vw   # a values per 32-row LDS line
  dwordMod = 16 // vw  # a values per 16-row dword group

  # 256 * (a // lineDiv). Zero when lineDiv > 16: every lane is on one line.
  lineMask = 15 // lineDiv
  if lineMask:
    module.add(VLShiftRightB32(dst=vgpr(acc), shiftHex=hex(5 - v), src=vgpr(laneId), comment="%s: laneId / %d" % (tag, lineDiv)))
    module.add(VAndB32(dst=vgpr(acc), src0=lineMask, src1=vgpr(acc), comment="%s: a / %d (a = laneId %%%% 16)" % (tag, lineDiv)))
    module.add(VLShiftLeftB32(dst=vgpr(acc), shiftHex=hex(8), src=vgpr(acc), comment="%s: line = 256 * (a / %d)" % (tag, lineDiv)))

  # 64 * c, c = laneId // 16  ->  K-block pair, unchanged from the blocked map
  module.add(VLShiftRightB32(dst=vgpr(tmp), shiftHex=hex(4), src=vgpr(laneId), comment="%s: c = laneId / 16" % tag))
  module.add(VLShiftLeftB32(dst=vgpr(tmp), shiftHex=hex(6), src=vgpr(tmp), comment="%s: 64 * c" % tag))
  if lineMask:
    module.add(VAddU32(dst=vgpr(acc), src0=vgpr(acc), src1=vgpr(tmp), comment="%s: += 64 * c" % tag))
  else:
    module.add(VMovB32(dst=vgpr(acc), src=vgpr(tmp), comment="%s: 64 * c" % tag))

  # 4 * vw * (a % dwordMod)
  module.add(VAndB32(dst=vgpr(tmp), src0=dwordMod - 1, src1=vgpr(laneId), comment="%s: a %%%% %d" % (tag, dwordMod)))
  module.add(VLShiftLeftB32(dst=vgpr(tmp), shiftHex=hex(v + 2), src=vgpr(tmp), comment="%s: %d * (a %%%% %d)" % (tag, 4 * vw, dwordMod)))
  module.add(VAddU32(dst=vgpr(acc), src0=vgpr(acc), src1=vgpr(tmp), comment="%s: += %d * (a %%%% %d)" % (tag, 4 * vw, dwordMod)))


def _emitScaleLaneOffsetInterleaved(module, writer, kernel, laneId, tileInfos):
  tmp = writer.vgprPool.checkOut(1, tag="_emitScaleLaneOffsetInterleaved_tmp")
  acc = writer.vgprPool.checkOut(1, tag="_emitScaleLaneOffsetInterleaved_acc")

  # A and B interleave by their own VectorWidth, so each has its own lane base. They
  # share one when the factors agree, which is the square wave-tile case.
  vws = [scaleInterleaveVW(kernel, ti_.tc) for ti_ in tileInfos]
  for vw in sorted(set(vws), key=vws.index):
    members = [ti_ for ti_, v in zip(tileInfos, vws) if v == vw]
    tag = "scale" if len(members) == len(tileInfos) else "scale%s" % members[0].tc
    _emitScaleLaneBaseInterleaved(module, acc, tmp, laneId, vw, tag)
    for ti_ in members:
      module.add(VAddU32(dst=vgpr(ti_.sharedVgprLROffset[0]), src0=vgpr(acc), src1=vgpr(ti_.sharedVgprLROffset[0]),
                         comment="scale%s: lrOffset = interleaved lane base" % ti_.tc))

  # v_perm selector. The MFMA selects its scale byte as (tile parity) + 2*k, so the
  # packed register must be k-major: byte s takes byte p' + 2*(s//2) of dword (s%2),
  # where dword 0/1 are the group's two M-adjacent tiles. With src1 = dword0 (selector
  # values 0-3) and src0 = dword1 (values 4-7) that is [0,4,2,6] = 0x06020400 at p'=0.
  # p' = (t%32)//16 = (laneId >> (4-v)) & 1 biases every selector byte equally.
  # v_mul_lo_u32 is VOP3 and takes no literal, so the byte-replication constant
  # has to come from an SGPR.
  permSgpr = writer.sgprPool.checkOut(1, tag="_emitScaleLaneOffsetInterleaved_permSgpr")
  module.add(SMovB32(dst=sgpr(permSgpr), src=hex(0x01010101), comment="scale: byte-replicate constant"))
  for ti_, vw in zip(tileInfos, vws):
    sel = ti_.sharedVgprScalePermSel
    if not sel:
      continue
    shift = 4 - (vw.bit_length() - 1)
    module.add(VLShiftRightB32(dst=vgpr(tmp), shiftHex=hex(shift), src=vgpr(laneId), comment="scale%s: laneId / %d" % (ti_.tc, 1 << shift)))
    module.add(VAndB32(dst=vgpr(tmp), src0=1, src1=vgpr(tmp), comment="scale%s: p' = (a %% %d) / %d" % (ti_.tc, 32 // vw, 16 // vw)))
    module.add(VMulLOU32(dst=vgpr(tmp), src0=sgpr(permSgpr), src1=vgpr(tmp), comment="scale%s: p' in every byte" % ti_.tc))
    module.add(VAddU32(dst=vgpr(sel[0]), src0=0x06020400, src1=vgpr(tmp), comment="scale%s: v_perm byte selector" % ti_.tc))
  writer.sgprPool.checkIn(permSgpr)

  writer.vgprPool.checkIn(acc)
  writer.vgprPool.checkIn(tmp)


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
  # Each tensor follows its own data map, so A and B can land on opposite sides of
  # this split. The interleaved bases only read laneOffset; the blocked one
  # overwrites it, so it has to go last.
  interleaved = [ti_ for ti_ in (tiA_, tiB_) if scaleRemapEnabled(kernel, ti_.tc)]
  blocked     = [ti_ for ti_ in (tiA_, tiB_) if not scaleRemapEnabled(kernel, ti_.tc)]
  if interleaved:
    _emitScaleLaneOffsetInterleaved(module, writer, kernel, laneOffset, interleaved)
  if blocked:
    module.add(VLShiftLeftB32(dst=vgpr(laneOffset), shiftHex=hex(2), src=vgpr(laneOffset), comment="scale: laneId * 4"))
    for ti_ in blocked:
      module.add(VAddU32(dst=vgpr(ti_.sharedVgprLROffset[0]), src0=vgpr(laneOffset), src1=vgpr(ti_.sharedVgprLROffset[0]),
                         comment="scale%s: lrOffset = laneId * 4" % ('A' if ti_.tc == 'MXSA' else 'B')))
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
##################################################
# Emit one scale group load.
#
# Blocked map: one ds_read_b32 at 256*group gives the 4 bytes the group needs
# (2 M-adjacent tiles x 2 K subiters), already in MFMA select order.
#
# Interleaved map (SourceSwap): the two M-adjacent tiles of a group live in
# adjacent dwords rather than in one, so the group is read as a b64 at the offset
# of its even tile. The byte each lane needs inside a dword is p' = (VW*a % 32)/16,
# which varies by lane and so cannot be reached by the MFMA's compile-time byte
# select; the v_perm gathers bytes p' and p'+2 out of both dwords into the same
# order the blocked map produced, which is what lets the select stay unchanged.
SCALE_GROUP_TILES = 2


def scaleGroupInterleavedOffset(vw, scaleGroupIdx) -> int:
  """LDS byte offset of scale group `scaleGroupIdx` under the interleaved map.

  A group is the tile pair (2g, 2g+1); tile j sits at 4*(j%vw) + 128*vw*(j//vw)
  (see the derivation above). vw is even here, so both tiles of a pair share
  j//vw and land in adjacent dwords -- which is what makes the b64 read legal.
  """
  j = SCALE_GROUP_TILES * scaleGroupIdx
  return 4 * (j % vw) + 128 * vw * (j // vw)


def emitScaleGroupLoad(writer, kernel, tileInfo, tc, vdst, scaleGroupIdx, blockedOffset, comment):
  module = Module()
  if not scaleRemapEnabled(kernel, tc):
    module.add(DSLoadB32(dst=vgpr(vdst),
                         src=vgpr(tileInfo.sharedVgprLROffset[0]),
                         ds=DSModifiers(offset=blockedOffset),
                         comment=comment))
    return module

  sel = tileInfo.sharedVgprScalePermSel
  assert sel, "scale%s: SourceSwap remap needs its v_perm selector VGPR" % tc
  # The b64 + v_perm pair reassembles exactly one tile pair, and the MFMA's byte
  # select indexes tiles within the group, so a group of any other size would need
  # a different gather.
  groupTiles = int(tileInfo.lrSubtileShape[0])
  assert groupTiles == SCALE_GROUP_TILES, (
    "scale%s: SourceSwap remap is derived for %d-tile scale groups, got %d"
    % (tc, SCALE_GROUP_TILES, groupTiles))
  module.add(DSLoadB64(dst=vgpr(vdst, 2),
                       src=vgpr(tileInfo.sharedVgprLROffset[0]),
                       ds=DSModifiers(offset=scaleGroupInterleavedOffset(scaleInterleaveVW(kernel, tc), scaleGroupIdx)),
                       comment=comment + " (interleaved: dwords for tiles 2g, 2g+1)"))
  # The pack consumes the load's own destination, so it cannot be left for the
  # scheduler's wait before the MFMA: without a wait here the v_perm reads stale
  # registers and the landing ds_read then overwrites its result, which degenerates
  # to passing the first dword through unchanged.
  module.add(SWaitCnt(dscnt=0, comment="scale%s[group%u]: pack needs the b64 landed" % (tc, scaleGroupIdx)))
  module.add(VPermB32(dst=vgpr(vdst), src0=vgpr(vdst + 1), src1=vgpr(vdst), src2=vgpr(sel[0]),
                      comment="scale%s[group%u]: pack lane bytes into MFMA select order" % (tc, scaleGroupIdx)))
  return module


def emitSubtileScaleDsRead(tc, writer, kernel, scaleGroupIdx):
  """Emit the LDS read for a scale group (2 M-adjacent [1,2] subtiles)."""
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
  module.add(emitScaleGroupLoad(writer, kernel, tileInfo, tc, vdst, scaleGroupIdx, dsOffset,
                                "scale%s[group%u]: load 4B from LDS" % (tc, scaleGroupIdx)))
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
