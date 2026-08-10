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
from ...Common import INDEX_CHARS
from rocisa.code import Module
from rocisa.container import DSModifiers, MUBUFModifiers, vgpr, sgpr, mgpr
from rocisa.instruction import (
    BufferLoadB128,
    DSLoadB32,
    SAddCU32, SAddU32, SAndB32, SLShiftLeftB32, SLShiftRightB32, SMovB32, SMovB64, SMulI32, SNop, SOrB32, SSubU32, SXorB32,
    VAddU32, VAndB32, VMulLOU32, VReadfirstlaneB32, VXorB32,
    VLShiftLeftB32, VLShiftRightB32,
)


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
  """Advance scale base pointer by one depthU iteration."""
  module = Module()
  tc = ti.tc

  useTdmForScale = getattr(writer.states, "asmCaps", {}).get("HasTDM", False)

  if useTdmForScale:
    # TDM path (gfx1250): advance Address{tc} and sync descriptor for MXSA.
    parentTc = tc[-1]  # 'A' from 'MXSA', 'B' from 'MXSB'
    mxBlock = kernel["ProblemType"][f"MXBlock{parentTc}"]
    du = kernel["DepthU"]
    scaleInc = du // mxBlock  # scale elements per depthU iteration
    module.addComment0("Scale TDM addr update: %s += %u" % (tc, scaleInc))
    module.add(SAddU32(dst=sgpr(f"Address{tc}"), src0=sgpr(f"Address{tc}"),
               src1=scaleInc, comment=f"Address{tc} += {scaleInc}"))
    module.add(SAddCU32(dst=sgpr(f"Address{tc}+1"), src0=sgpr(f"Address{tc}+1"),
               src1=0, comment=f"Address{tc}+1 carry"))
    if tc == "MXSA":
      from ...Components.TensorDataMover import TensorDataMoverLoad
      comp = TensorDataMoverLoad.find(writer)
      group0 = "tdmMXSAGroup0"
      module.add(comp.setGlobalAddr(group0, f"Address{tc}"))
    return module

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

  useTdmForScale = getattr(writer.states, "asmCaps", {}).get("HasTDM", False)

  if useTdmForScale:
    # TDM path (gfx1250): only swap for MXSA.  MXSB is aliased onto the
    # same descriptor, so a second XOR would undo the MXSA swap.  The
    # MXSB TDM load path adds ldsDelta to the already-swapped offset.
    #
    # Use the same per-address swap mask as the scale LR offsets: the
    # mask is addr XOR (addr + ldsTotalSize).  Since ldsTotalSize may
    # not be a power of 2, a plain XOR with ldsTotalSize would not
    # toggle correctly.  Instead, store a one-time swap mask in a tmp
    # SGPR and XOR with that.
    if tc == "MXSA":
      group0 = "tdmMXSAGroup0"
      ldsAddr = f"{group0}+1"
      module.addComment0("Scale TDM LDS swap: MXSA")
      module.add(SXorB32(dst=sgpr(ldsAddr),
                 src0=sgpr(ldsAddr), src1=sgpr("tdmLdsSwapMaskMXSA"),
                 comment="toggle LDS buffer"))
    return module

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
# Scale GR: Load scale bytes from global memory directly to LDS (DTL).
#
# Uses BufferLoadB128 with lds=True. M0 is set to scaleLdsBase, and
# sharedVgprGROffset[0] = serial * scaleLoadWidth serves as both the
# global read offset (from SRD) and the LDS write offset (from M0).
def globalReadDoScaleSubtile(tc, writer, kernel):
  module = Module()

  if not kernel["ProblemType"].get("MXBlockA", 0) and not kernel["ProblemType"].get("MXBlockB", 0):
    return module

  # gfx1250 does not support buffer_load with lds modifier; use TDM
  # tensor_load_to_lds instead. gfx950 continues using SRD buffer loads.
  useTdmForScale = getattr(writer.states, "asmCaps", {}).get("HasTDM", False)

  if useTdmForScale:
    return _globalReadDoScaleSubtileTDM(tc, writer, kernel)

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


def _setMxsAliasedDescriptorDim(module, comp, group1, kernel, writer, targetTi):
  """Set tile1, dim1, and stride in the aliased MXS descriptor for dimension targetTi.

  MXSB's Group1 is aliased onto MXSA's.  Called with targetTi=1 to
  switch to MXSB layout before loading, and targetTi=0 to restore
  MXSA layout afterwards.  Only emits instructions when the A and B
  dimensions actually differ.
  """
  otherTi = 1 - targetTi
  targetTc = "A" if targetTi == 0 else "B"
  otherTc  = "A" if otherTi == 0 else "B"

  mtTarget = kernel[f"MacroTile{targetTi}"]
  mtOther  = kernel[f"MacroTile{otherTi}"]
  numWaves = kernel["NumWaves"]
  perWaveRowsTarget = mtTarget // numWaves
  perWaveRowsOther  = mtOther  // numWaves
  if perWaveRowsTarget != perWaveRowsOther:
    module.add(comp.setTensorTile1(group1, perWaveRowsTarget, writer))

  sizeRefTarget = f"Size{INDEX_CHARS[targetTi]}"
  sizeRefOther  = f"Size{INDEX_CHARS[otherTi]}"
  if sizeRefTarget != sizeRefOther:
    module.add(comp.setTensorDim1(group1, sizeRefTarget, writer))

  mxBlock = kernel["ProblemType"][f"MXBlock{targetTc}"]
  sizeShifter = int(math.ceil(math.log2(mxBlock)))
  strideTarget = writer.strideRef(f"MXS{targetTc}", targetTi)
  strideOther  = writer.strideRef(f"MXS{otherTc}",  otherTi)
  if str(strideTarget) != str(strideOther):
    module.add(comp.setTensorStride0(group1, strideTarget, sizeShifter))



def _globalReadDoScaleSubtileTDM(tc, writer, kernel):
  """Emit MX scale global read via TDM tensor_load_to_lds (gfx1250).

  Uses the tdmMXS{A,B}Group0/1 descriptors (MXSB aliased onto MXSA).
  The descriptor must be initialised before the first call and advanced
  by globalReadScalePtrUpdates between iterations.
  """
  from ...Components.TensorDataMover import TensorDataMoverLoad

  module = Module()
  scaleTc = "MXSA" if tc in ("A", "MXSA") else "MXSB"
  group0 = "tdm%sGroup0" % scaleTc
  group1 = "tdm%sGroup1" % scaleTc

  comp = TensorDataMoverLoad.find(writer)

  if scaleTc == "MXSB":
    ldsBaseMXSB = writer.ldsStartOffsetMXSB
    ldsBaseMXSA = writer.ldsStartOffsetMXSA
    ldsDelta = ldsBaseMXSB - ldsBaseMXSA
    module.addComment0("Scale GR: MXSB (TDM: patch aliased descriptor)")
    module.add(comp.setGlobalAddr(group0, "AddressMXSB"))
    if ldsDelta != 0:
      module.add(SAddU32(dst=sgpr(f"{group0}+1"), src0=sgpr(f"{group0}+1"),
                 src1=ldsDelta, comment=f"LDS addr += {ldsDelta} (MXSA->MXSB)"))
    # Patch tile1, dim1, and stride for the B dimension when MT0 != MT1.
    _setMxsAliasedDescriptorDim(module, comp, group1, kernel, writer, targetTi=1)

  module.addComment0("Scale GR: %s (TDM: tensor_load_to_lds)" % scaleTc)
  comp.setMemToken([writer.states.ldsTensorTokenIdx])
  module.add(comp.issueLoad(group0, group1, None, None))

  if scaleTc == "MXSB":
    module.addComment0("Restore MXSA descriptor after MXSB load")
    module.add(comp.setGlobalAddr(group0, "AddressMXSA"))
    if ldsDelta != 0:
      module.add(SSubU32(dst=sgpr(f"{group0}+1"), src0=sgpr(f"{group0}+1"),
                 src1=ldsDelta, comment=f"LDS addr -= {ldsDelta} (MXSB->MXSA)"))
    # Restore MXSA descriptor fields
    _setMxsAliasedDescriptorDim(module, comp, group1, kernel, writer, targetTi=0)

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

  # For wave32 gfx1250 with InMemorySwizzle: the LDS layout after TDM load
  # is {numKTiles, perWaveRows, dimk} where dimk = instK/MXBlock.
  # Each ds_load_b32 reads dimk bytes per lane.  Group stride = lanes * dimk
  # advances to the next batch of 32 M-rows within the same K-tile.
  wavelen = kernel["WavefrontSize"]
  isWave32Gfx1250 = (wavelen == 32 and getattr(writer.states, "asmCaps", {}).get("HasTDM", False))
  if isWave32Gfx1250:
    parentTc = tc[-1]
    mxBlock = kernel["ProblemType"][f"MXBlock{parentTc}"]
    instK = kernel["MatrixInstK"]
    dimk = instK // mxBlock
    groupStride = wavelen * dimk  # 32 * 4 = 128 bytes between M-row groups
  elif hasattr(tileInfo, 'lrSubtileSize'):
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

  # For wave32 gfx1250: each ds_load covers wavelen M-rows.
  # numGroups = perWaveRows / wavelen (M-dimension groups).
  wavelen = kernel["WavefrontSize"]
  isWave32Gfx1250 = (wavelen == 32 and getattr(writer.states, "asmCaps", {}).get("HasTDM", False))
  if isWave32Gfx1250:
    parentTc = tc[-1]
    ti = 0 if parentTc == 'A' else 1
    mt = kernel[f"MacroTile{ti}"]
    wgAxis = kernel["MIWaveGroup"][ti]
    perWaveMRows = mt // wgAxis
    numScaleGroups = max(1, perWaveMRows // wavelen)
  else:
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

# emitWaveAxisIndex imported lazily at call site to avoid circular import


def _initTDMDescriptorMXScaleSubtile(writer, kernel, scaleTc):
  """Init TDM descriptor for one MX scale tensor (subtile gfx1250 layout).

  Sets up tdmMXS{A,B}Group0/1 to load scale data from global memory
  directly into the LDS region reserved for MX scales. The LDS offset
  matches what the subtile scale LR reads via ds_load_b32.

  Args:
    scaleTc: "MXSA" or "MXSB"
  """
  from ...Components.TensorDataMover import TensorDataMoverLoad


  comp = TensorDataMoverLoad.find(writer)
  mod = Module(f"Init TDM Descriptor Subtile {scaleTc}")

  parentTc = scaleTc[-1]  # 'A' or 'B'
  if not kernel["ProblemType"].get(f"MXBlock{parentTc}", 0):
    return mod

  group0 = f"tdm{scaleTc}Group0"
  group1 = f"tdm{scaleTc}Group1"

  tileInfo = writer.states.mxsa.tileInfo if scaleTc == "MXSA" else writer.states.mxsb.tileInfo
  tP = writer.tPA["MX"] if parentTc == "A" else writer.tPB["MX"]
  dtype = kernel["ProblemType"][f"DataType{scaleTc}"]  # E8M0
  ti = tP["idx"]  # 0 for A, 1 for B

  mxBlock = kernel["ProblemType"][f"MXBlock{parentTc}"]
  mt = kernel[f"MacroTile{ti}"]
  du = kernel["DepthU"]
  numWaves = kernel["NumWaves"]
  wavelen = kernel["WavefrontSize"]

  # Scale dimensions: each scale covers mxBlock data elements along K.
  # Total scales per tile = (mt * du) / mxBlock, arranged as mt rows x (du/mxBlock) cols.
  scaleDu = du // mxBlock     # scale columns per depthU
  scaleBpe = 1                # E8M0 = 1 byte per scale

  # dimk = instK / mxBlock: number of K-blocks consumed by one WMMA instruction.
  # In InMemorySwizzle layout, scale data is rearranged as {numKTiles, freeDim, dimk}
  # where numKTiles = scaleDu / dimk.  Each TDM load reads one K-tile worth of
  # data: freeDim * dimk contiguous bytes.  This matches the non-subtile path.
  instK = kernel["MatrixInstK"]
  dimk = instK // mxBlock     # 4 for FP4 MXBlock=32

  # LDS layout: scales start at ldsStartOffsetMXS{A,B}
  ldsBase = writer.ldsStartOffsetMXSA if scaleTc == "MXSA" else writer.ldsStartOffsetMXSB

  # Each wave loads its share of the free dimension.  For MXSA the free dim
  # is M (partitioned by MIWaveGroup[0]); for MXSB it is N (partitioned by
  # MIWaveGroup[1]).  Using NumWaves (= MIWaveGroup[0]*[1]) would under-load
  # when the tile is not square.
  wgAxis = kernel["MIWaveGroup"][ti]  # 0 for A, 1 for B
  perWaveRows = mt // wgAxis
  bytesPerWave = perWaveRows * scaleDu * scaleBpe

  sizeShifter = int(math.ceil(math.log2(mxBlock)))

  mod.add(comp.initOperands(group0, group1, None, None))
  mod.add(comp.setDataType(dtype, group1))

  # Per-workgroup + per-wave global offset for scale tensor.
  # AddressMXSA/B is rebased to the tensor base each persistent-loop
  # iteration, so apply the WG+wave offset here.
  tlu = tP["tlu"]
  sizeRefFree = f"Size{INDEX_CHARS[ti]}"
  # Allocate 4 tmp SGPRs: [0]=accumulator, [1]=spare, [2]=waveGlobalOff, [3]=waveAxisIdx.
  # waveAxisIdx is computed once and reused for both the global and LDS offset.
  with writer.allocTmpSgpr(4, tag="_initTDMScale_off") as offRes:
    tmp = offRes.idx
    waveGlobalOff = offRes.idx + 2
    waveAxisIdx = offRes.idx + 3
    # WG offset: stride_free * MT * wgId
    scaleStride = writer.strideRef(scaleTc, ti)
    mod.add(SMulI32(dst=sgpr(tmp), src0=scaleStride, src1=mt,
            comment=f"stride_free * MT({mt})"))
    mod.add(SMulI32(dst=sgpr(tmp), src0=sgpr(tmp), src1=sgpr(f"WorkGroup{ti}"),
            comment="*= wgId"))

    if wgAxis > 1:
      from .Kernel import emitWaveAxisIndex
      emitWaveAxisIndex(mod, kernel, ti, waveAxisIdx)
      strideWaveSep = writer.strideRef(scaleTc, 3) if tlu else writer.strideRef(scaleTc, ti)
      mod.add(SMulI32(dst=sgpr(waveGlobalOff), src0=sgpr(waveAxisIdx), src1=perWaveRows,
              comment=f"waveGlobalOff = waveIdx_axis * {perWaveRows}"))
      mod.add(SMulI32(dst=sgpr(waveGlobalOff), src0=sgpr(waveGlobalOff), src1=strideWaveSep,
              comment="waveGlobalOff *= stride"))
      mod.add(SAddU32(dst=sgpr(tmp), src0=sgpr(tmp), src1=sgpr(waveGlobalOff),
              comment="+= waveOff"))
    else:
      mod.add(SMovB32(dst=sgpr(waveAxisIdx), src=0, comment="single wave: waveAxisIdx=0"))

    # Undo MXBlock pre-scale to get byte offset
    mod.add(SLShiftRightB32(dst=sgpr(tmp), src=sgpr(tmp),
            shiftHex=hex(sizeShifter),
            comment=f">> {sizeShifter} (undo MXBlock pre-scale)"))
    mod.add(SAddU32(dst=sgpr(f"Address{scaleTc}"),
            src0=sgpr(f"Address{scaleTc}"), src1=sgpr(tmp),
            comment=f"Address{scaleTc} += globalOffset(lo)"))
    mod.add(SAddCU32(dst=sgpr(f"Address{scaleTc}+1"),
            src0=sgpr(f"Address{scaleTc}+1"), src1=0, comment="carry"))

    mod.add(comp.setGlobalAddr(group0, f"Address{scaleTc}"))

    # LDS offset = waveIdx_axis * bytesPerWave + ldsBase
    # Reuse waveAxisIdx computed above (avoids a second VReadfirstlaneB32).
    mod.add(SMulI32(sgpr(waveAxisIdx), sgpr(waveAxisIdx), bytesPerWave,
            f"woffset = waveIdx_axis * {bytesPerWave}"))
    mod.add(SAddU32(sgpr(waveAxisIdx), sgpr(waveAxisIdx), ldsBase,
            f"ldsOffset = woffset + {ldsBase}"))
    mod.add(comp.setLdsAddr(group0, sgpr(waveAxisIdx)))

    # Pre-compute the LDS double-buffer swap mask for MXSA.
    ldsTotalSize = writer.ldsTotalSize
    mod.add(SAddU32(dst=sgpr("tdmLdsSwapMaskMXSA"), src0=sgpr(waveAxisIdx),
            src1=ldsTotalSize,
            comment=f"addr + ldsTotalSize({ldsTotalSize})"))
    mod.add(SXorB32(dst=sgpr("tdmLdsSwapMaskMXSA"),
            src0=sgpr(waveAxisIdx), src1=sgpr("tdmLdsSwapMaskMXSA"),
            comment="swapMask = addr XOR (addr + ldsTotalSize)"))

  # Scale TDM descriptor layout for subtile:
  # Match the non-subtile InMemorySwizzle layout exactly.
  # Global data is pre-swizzled as {numKTiles, freeDim, dimk}.
  # The TDM loads one K-tile as a contiguous 1D block: tile0 = freeDim * dimk.
  # Multiple K-tiles (numKTiles = scaleDu / dimk) are loaded via tile1.
  #
  # LDS layout after TDM write (per K-tile):
  #   Byte offset = M_row * dimk + k_block
  # The subtile LR reads with laneId * 4 (= laneId * dimk), so lane L reads
  # M=L's dimk K-block bytes.  Multiple K-tiles are at offsets of
  # perWaveRows * dimk bytes apart.
  numKTiles = scaleDu // dimk
  sizeRefK = f"Size{INDEX_CHARS[3]}"

  mod.add(comp.setIterationEnabled(group1, False))
  mod.add(comp.setPadding(group1, 0, 0))

  # dim0 = freeDim (for OOB clamping along the free/M dimension).
  # Multiplied by dimk via the isMXS left-shift in setTensorDim0.
  mod.add(comp.setTensorDim0(group1, sizeRefFree, writer, int(math.ceil(math.log2(dimk))), True))
  # dim1 = K / instK (number of K-tiles for OOB clamping along K).
  mod.add(comp.setTensorDim1(group1, sizeRefK, writer, int(math.ceil(math.log2(mxBlock * dimk))), True))

  # tile0 = perWaveRows * dimk (one full K-tile's worth of data).
  mod.add(comp.setTensorTile0(group1, perWaveRows * dimk, writer, 0))

  # tile1 = numKTiles (number of K-tiles per depthU iteration).
  # For edge WGs: TDM hardware zeroes OOB bytes.
  mod.add(comp.setTensorTile1(group1, numKTiles, writer))

  # Stride = freeDimSize * dimk (bytes between K-tiles in the swizzled layout).
  mod.add(comp.setTensorStride0(group1, sizeRefFree, int(math.ceil(math.log2(dimk))), True))

  return mod



def globalReadScaleSwizzledDTLInitCommonSgpr(writer, kernel):
  module = Module()

  useTdmForScale = getattr(writer.states, "asmCaps", {}).get("HasTDM", False)

  if useTdmForScale:
    # gfx1250: init TDM descriptors for MX scales using subtile LDS layout.
    module.addComment0("Init TDM descriptor for MX scales (gfx1250 subtile, MXSA only)")
    module.add(_initTDMDescriptorMXScaleSubtile(writer, kernel, "MXSA"))
    return module

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
