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
#
# DeepseekScale (UseDeepseekScaleA/B): fp8 per-row / per-K-block E8M0 scale
# with host-pre-broadcast layout. Uses the same SA/SB scheduler path as MX
# scale (enabling PGR=0/1/2). Gated by usesScaleA/usesScaleB helpers below.
################################################################################

import math

from rocisa.code import Module
from rocisa.container import DSModifiers, MUBUFModifiers, vgpr, sgpr, mgpr
from rocisa.instruction import (
    BufferLoadB32, BufferLoadB128,
    DSLoadB32,
    SAddCU32, SAddU32, SAndB32, SLoadB64, SLShiftLeftB32, SLShiftRightB32, SMovB32,
    SMulI32, SNop, SWaitCnt, SXorB32,
    VAddCOU32, VAddCCOU32, VAddU32, VAndB32, VMulLOU32, VMovB32, VReadfirstlaneB32, VXorB32,
    VLShiftLeftB32, VLShiftRightB32,
)


def usesScaleA(kernel) -> bool:
    """True when scale A is needed (MX or DeepseekScale)."""
    return kernel["ProblemType"].get("MXBlockA", 0) > 0 or kernel.get("UseDeepseekScaleA", False)


def usesScaleB(kernel) -> bool:
    """True when scale B is needed (MX or DeepseekScale)."""
    return kernel["ProblemType"].get("MXBlockB", 0) > 0 or kernel.get("UseDeepseekScaleB", False)


def isDeepseekScale(kernel) -> bool:
    """True when this kernel uses DeepseekScale (flat fp8, no MX block-scale)."""
    return kernel.get("UseDeepseekScaleA", False) or kernel.get("UseDeepseekScaleB", False)


def deepseekScaleBNBlocksPerWave(kernel) -> int:
    """N-blocks (128 cols each) covered by one wave for DeepseekScale scaleB.

    MT1=128 -> 1; MT1=256,wg_n=1 -> 2; MT1=256,wg_n=2 -> 1.
    """
    mt1 = kernel.get("MacroTile1", 0)
    wgN = kernel["MIWaveGroup"][1]
    return max(1, mt1 // (128 * wgN))


# ---------------------------------------------------------------------------
# Scale GR offset
# ---------------------------------------------------------------------------

def emitScaleGROffset(ti, writer, kernel):
  """Compute per-thread DTL vaddr for scale GR load."""
  return Module(f"Scale GR Offset ({ti.tc})")  # STUB


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

  if isDeepseekScale(kernel):
    # DeepseekScale: advance by one K-block of wave-contiguous 4-byte groups.
    # Each wave stores waveSize * loadWidthGR bytes per K-block contiguously
    # so all waves advance by the same fixed amount per iteration.
    inc = ti.waveSize * ti.loadWidthGR
    module.addComment0("DeepseekScale SRD update: %s += %u (waveSize * loadWidthGR)" % (tc, inc))
  else:
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
  module.addComment("Computing GR Offset for %s" % tc)
  ti_ = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo

  if isDeepseekScale(kernel):
    # DeepseekScale: vaddr = laneWithinWave * loadWidthGR. Use the lane index
    # within the wave (not the workgroup-wide serial): initDeepseekScaleSrd
    # already advances SrdMXSA by the per-wave row-group offset, so adding the
    # wave part of the serial here would double-count it. This vaddr doubles as
    # both the global read offset (from SrdMXSA) and the LDS write offset (from M0).
    loadWidth = ti_.loadWidthGR  # = 4 for DeepseekScale (b32)
    loadWidthShift = loadWidth.bit_length() - 1
    waveSize = kernel["WavefrontSize"]
    module.add(VAndB32(dst=vgpr(ti_.sharedVgprGROffset[0]),
                       src0=hex(waveSize - 1), src1=vgpr("Serial"),
                       comment="%s: laneWithinWave = serial %% %d" % (tc, waveSize)))
    module.add(VLShiftLeftB32(dst=vgpr(ti_.sharedVgprGROffset[0]),
                              shiftHex=hex(loadWidthShift), src=vgpr(ti_.sharedVgprGROffset[0]),
                              comment="%s: DeepseekScale vaddr = laneWithinWave * %d" % (tc, loadWidth)))
    return module

  loadWidth = ti_.loadWidthGR
  loadWidthShift = loadWidth.bit_length() - 1
  scaleGroupSize = ti_.lrSubtileSize
  numThreadsPerGroup = (scaleGroupSize * int(ti_.lrGlobalSubtileGrid[1])) // loadWidth

  vtmp = writer.vgprPool.checkOut(1, tag="_graTileAssignmentScaleSwizzledCommon_vtmp")
  stmp = writer.sgprPool.checkOut(1, tag="_graTileAssignmentScaleSwizzledCommon_stmp")

  module.add(VLShiftRightB32(dst=vgpr(vtmp),
                             shiftHex=hex(int(math.log2(numThreadsPerGroup))), src=vgpr("Serial"),
                             comment="%s: grOffset = serial / %d" % (tc, loadWidth)))
  module.add(SLShiftLeftB32(sgpr(stmp), int(math.log2(ti_.bpe)), sgpr("Strides%s" % tc),
                            comment="*= bpe (%d)" % ti_.bpe))
  module.add(VMulLOU32(dst=vgpr(vtmp), src1=vgpr(vtmp), src0=sgpr(stmp),
                       comment="Apply scale%s stride to each group" % tc))
  module.add(VAndB32(dst=vgpr(ti_.sharedVgprGROffset[0]),
                     src0=hex(numThreadsPerGroup - 1), src1=vgpr("Serial"),
                     comment="%s: grOffset = serial %% %d" % (tc, loadWidth)))
  module.add(VLShiftLeftB32(dst=vgpr(ti_.sharedVgprGROffset[0]),
                            shiftHex=hex(loadWidthShift), src=vgpr(ti_.sharedVgprGROffset[0]),
                            comment="Scale by load width for each thread in group"))
  module.add(VAddU32(dst=vgpr(ti_.sharedVgprGROffset[0]), src0=vgpr(ti_.sharedVgprGROffset[0]),
                     src1=vgpr(vtmp), comment="Final offset calc"))
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
  if not usesScaleA(kernel) and not usesScaleB(kernel):
    return module
  if usesScaleA(kernel):
    module.add(_graTileAssignmentScaleSwizzledCommon('MXSA', writer, kernel))
  if usesScaleB(kernel):
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

  # For DeepseekScale scaleB, each wave DTL-writes waveSize * loadWidthGR * nBlocksB
  # bytes. The LR partition stride must equal this write size so each N-wave reads
  # from its own region. For MX scale the subtile-grid formula applies.
  if isDeepseekScale(kernel) and tc == 'MXSB':
    totalScaleBytes = ti_.waveSize * ti_.loadWidthGR * deepseekScaleBNBlocksPerWave(kernel)
  else:
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
  hasA = usesScaleA(kernel)
  hasB = usesScaleB(kernel)
  if not hasA and not hasB:
    return module
  tiA_ = writer.states.mxsa.tileInfo if hasA else None
  tiB_ = writer.states.mxsb.tileInfo if hasB else None
  activeTiles = [ti for ti in [tiA_, tiB_] if ti is not None]
  module.addComment0("LR Offset Calculation for Scale Tensors")
  wavesize = kernel["WavefrontSize"]
  waveIdVgpr = writer.vgprPool.checkOut(1, tag="_lraTileAssignmentScaleSwizzled_legacy_waveIdVgpr")
  module.add(VLShiftRightB32(dst=vgpr(waveIdVgpr), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="scale: waveId"))
  for ti_ in activeTiles:
    _applyScaleWavePartitionLROffset(module, writer, kernel, ti_, waveIdVgpr)
  writer.vgprPool.checkIn(waveIdVgpr)
  laneOffset = writer.vgprPool.checkOut(1, tag="_lraTileAssignmentScaleSwizzled_legacy_laneOffset")
  # DeepseekScale rows wrap at MatrixInstM (16) rather than wavesize, because
  # the scale is per-row within a 16-row MFMA tile; using the full lane id
  # would overflow the 256-byte MXSA LDS region at high M-tile indices.
  laneMask = (kernel["MatrixInstM"] - 1) if isDeepseekScale(kernel) else (wavesize - 1)
  module.add(VAndB32(dst=vgpr(laneOffset), src0=vgpr("Serial"), src1=laneMask,
                     comment="scale: laneId %% %d" % (laneMask + 1)))
  module.add(VLShiftLeftB32(dst=vgpr(laneOffset), shiftHex=hex(2), src=vgpr(laneOffset),
                             comment="scale: laneRow * 4"))
  for ti_ in activeTiles:
    label = "scaleA" if ti_ is tiA_ else "scaleB"
    module.add(VAddU32(dst=vgpr(ti_.sharedVgprLROffset[0]), src0=vgpr(laneOffset), src1=vgpr(ti_.sharedVgprLROffset[0]), comment="%s: lrOffset = laneId * 4" % label))
  writer.vgprPool.checkIn(laneOffset)
  tmpSgpr = writer.sgprPool.checkOut(1, tag="_lraTileAssignmentScaleSwizzled_legacy_tmpSgpr")
  if hasA:
    module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(writer.ldsStartOffsetMXSA), comment="scale: LDS offset for A scale"))
    module.add(VAddU32(dst=vgpr(tiA_.sharedVgprLROffset[0]), src0=vgpr(tiA_.sharedVgprLROffset[0]), src1=sgpr(tmpSgpr), comment="scaleA: +=LDS offset"))
  if hasB:
    module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(writer.ldsStartOffsetMXSB), comment="scale: LDS offset for B scale"))
    module.add(VAddU32(dst=vgpr(tiB_.sharedVgprLROffset[0]), src0=vgpr(tiB_.sharedVgprLROffset[0]), src1=sgpr(tmpSgpr), comment="scaleB: +=LDS offset"))
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=writer.ldsTotalSize, comment="scale: total LDS size for swap"))
  for ti_ in activeTiles:
    for i in range(len(ti_.sharedVgprLROffset)):
      vgprId     = ti_.sharedVgprLROffset[i]
      vgprSwapId = ti_.sharedVgprLROffsetSwap[i]
      module.add(VAddU32(dst=vgpr(vgprSwapId), src0=vgpr(vgprId), src1=sgpr(tmpSgpr), comment="scale%s: LR swap" % ti_.tc))
      module.add(VXorB32(dst=vgpr(vgprSwapId), src0=vgpr(vgprId), src1=vgpr(vgprSwapId), comment="scale%s: LR swap" % ti_.tc))
  writer.sgprPool.checkIn(tmpSgpr)
  return module

##################################################
# Scale GR: Load scale bytes from global memory directly to LDS (DTL).
#
# Uses BufferLoadB128 with lds=True. M0 is set to scaleLdsBase, and
# sharedVgprGROffset[0] = serial * scaleLoadWidth serves as both the
# global read offset (from SRD) and the LDS write offset (from M0).

def _emitDeepseekScaleBGR(writer, kernel, tileInfo, mubuf):
  """Emit one BufferLoadB32 DTL per N-block for DeepseekScale scaleB.

  Block j reads from SrdMXSB + j*(nKBlocks*wave_bytes) and writes to LDS
  at M0 + j*wave_bytes; M0 is updated before each load for j>0.
  """
  module = Module()
  nBlocksB = deepseekScaleBNBlocksPerWave(kernel)
  waveBytes = kernel["WavefrontSize"] * tileInfo.loadWidthGR
  for j in range(nBlocksB):
    if j > 0:
      with writer.allocTmpSgpr(1, tag="dsScaleBGrM0") as t:
        module.add(SAddU32(dst=sgpr(t.idx), src0=sgpr("LocalWriteBaseAddrMXSB"),
                           src1=j * waveBytes,
                           comment="scaleMXSB: LDS base for block%u" % j))
        module.add(SMovB32(dst=mgpr(0), src=sgpr(t.idx),
                           comment="scaleMXSB: M0 = LDS base block%u" % j))
      soffset = sgpr("DsScaleBBlockStride")
    else:
      soffset = 0
    module.add(BufferLoadB32(dst=None, vaddr=vgpr(tileInfo.sharedVgprGROffset[0]),
                             saddr=sgpr("SrdMXSB", 4), soffset=soffset, mubuf=mubuf,
                             comment="scaleMXSB[block%u]: DeepseekScale DTL b32 load" % j))
  return module


def globalReadDoScaleSubtile(tc, writer, kernel):
  module = Module()

  if not usesScaleA(kernel) and not usesScaleB(kernel):
    return module
  if tc == 'MXSA' and not usesScaleA(kernel):
    return module
  if tc == 'MXSB' and not usesScaleB(kernel):
    return module

  tileInfo = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo

  isGlc = bool(kernel["NonTemporal%s"%tc] & 0x1)
  isSlc = bool(kernel["NonTemporal%s"%tc] & 0x2)
  isNT  = bool(kernel["NonTemporal%s"%tc] & 0x4)

  assert len(tileInfo.sharedVgprGROffset) > 0, "scale GR requires at least 1 GR offset VGPR"

  # Set M0 to the wave's LDS base for j=0 (and for MXSA or non-DeepseekScale).
  module.add(SMovB32(dst=mgpr(0), src=sgpr("LocalWriteBaseAddr%s" % tc),
                     comment="scale%s: M0 = scaleLdsBase" % tc))

  mubuf = MUBUFModifiers(offen=True, offset12=0, glc=isGlc, slc=isSlc, nt=isNT, lds=True)
  if isDeepseekScale(kernel):
    # DeepseekScale: 4 bytes/lane (b32 DTL). For scaleB with nBlocksB>1 each
    # N-block gets its own load; scaleA is a single b32 load.
    module.addComment0("Scale GR: %s (DeepseekScale DTL: BufferLoadB32 -> LDS)" % tc)
    if tc == 'MXSB':
      module.add(_emitDeepseekScaleBGR(writer, kernel, tileInfo, mubuf))
    else:
      module.add(BufferLoadB32(dst=None, vaddr=vgpr(tileInfo.sharedVgprGROffset[0]),
                               saddr=sgpr("Srd%s" % tc, 4), soffset=0, mubuf=mubuf,
                               comment="scale%s: DeepseekScale DTL b32 load" % tc))
  else:
    module.addComment0("Scale GR: %s (DTL: BufferLoadB128 -> LDS)" % tc)
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

  if not usesScaleA(kernel) and not usesScaleB(kernel):
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


def _emitWaveBase(module, writer, tc, ldsStartOffset, swapName, bytesPerLoad, vWaveIdVgpr):
  """Emit LDS base + swap initialisation for one scale tensor.

  Derives baseAddrName as 'LocalWriteBaseAddr' + tc so callers don't need to
  pass it separately.
  """
  baseAddrName = "LocalWriteBaseAddr" + tc
  vTmp = writer.vgprPool.checkOut(1, tag="globalReadScaleSwizzledDTLInitCommonSgpr_%s" % tc)
  module.add(VLShiftLeftB32(dst=vgpr(vTmp),
                            shiftHex=hex(bytesPerLoad.bit_length() - 1),
                            src=vgpr(vWaveIdVgpr),
                            comment="%s: wave LDS offset (%u bytes/wave)" % (tc, bytesPerLoad)))
  module.add(SNop(waitState=0, comment="wait for VGPR to be ready"))
  module.add(VReadfirstlaneB32(dst=sgpr(baseAddrName), src=vgpr(vTmp),
                               comment="scale%s: wave LDS base" % tc))
  module.add(SAddU32(dst=sgpr(baseAddrName),
                     src0=sgpr(baseAddrName),
                     src1=hex(ldsStartOffset), comment=""))
  module.add(SAddU32(dst=sgpr(swapName), src0=sgpr(baseAddrName),
                     src1=writer.ldsTotalSize, comment=""))
  module.add(SXorB32(dst=sgpr(swapName), src0=sgpr(baseAddrName),
                     src1=sgpr(swapName), comment=""))
  writer.vgprPool.checkIn(vTmp)


##################################################
# Subroutine to generate DTL M0 LDS buffer swap
#
# For Swizzled Scales each wave will collectively stream
# the scale values
#
def globalReadScaleSwizzledDTLInitCommonSgpr(writer, kernel):
  module = Module()

  wavesize = kernel["WavefrontSize"]
  tiMXSA_ = writer.states.mxsa.tileInfo if usesScaleA(kernel) else None
  tiMXSB_ = writer.states.mxsb.tileInfo if usesScaleB(kernel) else None

  # Compute the plain wave index (0, 1, ..., numWaves-1); each tensor may need
  # a different per-wave LDS byte stride (scaleB with nBlocksB>1 uses a wider
  # slot), so the left-shift is applied per-tensor inside _emitWaveBase.
  vWaveId = writer.vgprPool.checkOut(1, tag="globalReadScaleSwizzledDTLInitCommonSgpr_vgprWaveId")
  module.addComment0("Compute shared offsets used by m0 in scale DTL loads")
  module.add(VLShiftRightB32(dst=vgpr(vWaveId), shiftHex=hex(wavesize.bit_length() - 1),
                             src=vgpr("Serial"), comment="Wave Id"))

  if usesScaleA(kernel):
    bytesPerLoadA = tiMXSA_.loadWidthGR * wavesize
    _emitWaveBase(module, writer, 'MXSA', writer.ldsStartOffsetMXSA,
                  "SwapMXSA", bytesPerLoadA, vWaveId)

  if usesScaleB(kernel):
    nBlocksB = deepseekScaleBNBlocksPerWave(kernel)
    bytesPerLoadB = tiMXSB_.loadWidthGR * wavesize * nBlocksB
    _emitWaveBase(module, writer, 'MXSB', writer.ldsStartOffsetMXSB,
                  "SwapMXSB", bytesPerLoadB, vWaveId)

  writer.vgprPool.checkIn(vWaveId)
  return module


# ---------------------------------------------------------------------------
# DeepseekScale SRD init
# ---------------------------------------------------------------------------

def _dsByteOffset(module, waveIdSgpr, waveOffSgpr, stmp, log2wg_m, isN, wgSgpr, wg_dim, strideSgpr, mt1=128):
  """Emit SGPR code for a per-wave byte offset into one DS scale buffer.

  A-side (isN=False): waveM = waveId & (wg_m-1); globalIdx = waveM + WG0*wg_m.
  B-side (isN=True):  globalIdx = first N-block of this wave = (WG1*MT1 + waveN*(MT1/wg_n)) >> 7.
  waveOff = globalIdx * strideSgpr, where strideSgpr = nKBlocks * wave_bytes.
  Result in waveOffSgpr; stmp used as scratch.
  """
  if isN:
    # First N-block of this wave = (WG1*MT1 + waveN_idx*(MT1/wg_n)) / 128,
    # where waveN_idx = waveId / wg_m. Handles MT1>128 (wave spans multiple
    # N-blocks) and wg_n splitting within or across N-blocks.
    colPerWaveN = mt1 // wg_dim
    if log2wg_m > 0:
      module.add(SLShiftRightB32(dst=sgpr(stmp), src=sgpr(waveIdSgpr),
                                 shiftHex=hex(log2wg_m),
                                 comment="waveN_idx = waveId / wg_m"))
    else:
      module.add(SMovB32(dst=sgpr(stmp), src=sgpr(waveIdSgpr),
                         comment="waveN_idx = waveId (wg_m=1)"))
    module.add(SMulI32(dst=sgpr(waveOffSgpr), src0=sgpr(stmp), src1=colPerWaveN,
                       comment="waveN_idx * (MT1/wg_n) columns"))
    module.add(SMulI32(dst=sgpr(stmp), src0=sgpr(wgSgpr), src1=mt1,
                       comment="WG1 * MT1 columns"))
    module.add(SAddU32(dst=sgpr(waveOffSgpr), src0=sgpr(waveOffSgpr), src1=sgpr(stmp),
                       comment="firstColumn = WG1*MT1 + waveN_idx*(MT1/wg_n)"))
    module.add(SLShiftRightB32(dst=sgpr(waveOffSgpr), src=sgpr(waveOffSgpr),
                               shiftHex=hex(7),
                               comment="globalIdx = firstColumn / 128 (N-block index)"))
    module.add(SMulI32(dst=sgpr(waveOffSgpr), src0=sgpr(waveOffSgpr), src1=sgpr(strideSgpr),
                       comment="waveOff = globalIdx * (nKBlocks * wave_bytes)"))
    return
  if log2wg_m > 0:
    module.add(SAndB32(dst=sgpr(waveOffSgpr), src0=sgpr(waveIdSgpr), src1=(1 << log2wg_m) - 1,
                       comment="waveM = waveId & (wg_m-1)"))
  else:
    module.add(SMovB32(dst=sgpr(waveOffSgpr), src=0, comment="waveM = 0 (wg_m=1)"))
  module.add(SMulI32(dst=sgpr(stmp), src0=sgpr(wgSgpr), src1=wg_dim,
                     comment="WG0 * wg_m"))
  module.add(SAddU32(dst=sgpr(waveOffSgpr), src0=sgpr(waveOffSgpr), src1=sgpr(stmp),
                     comment="globalIdx = waveM + WG0 * wg_m"))
  module.add(SMulI32(dst=sgpr(waveOffSgpr), src0=sgpr(waveOffSgpr), src1=sgpr(strideSgpr),
                     comment="waveOff = globalIdx * (nKBlocks * wave_bytes)"))


def _dsSetSrd(module, off, stmp, waveOffSgpr, srdName, srdBits):
  """Load a 64-bit pointer from KernArgs, add wave offset, and write the SRD quad."""
  module.add(SLoadB64(dst=sgpr(stmp, 2), base=sgpr("KernArgAddress", 2),
                      soffset=hex(off), comment="load %s ptr" % srdName))
  module.add(SWaitCnt(kmcnt=0, comment="wait for %s ptr" % srdName))
  module.add(SAddU32(dst=sgpr(srdName), src0=sgpr(stmp), src1=sgpr(waveOffSgpr),
                     comment="%s[0] = ptr + offset (lo)" % srdName))
  module.add(SAddCU32(dst=sgpr(srdName + "+1"), src0=sgpr(stmp + 1), src1=0,
                      comment="%s[1] = ptr (hi) + carry" % srdName))
  module.add(SMovB32(dst=sgpr(srdName + "+2"), src=hex(0x80000000),
                     comment="%s[2] = max limit" % srdName))
  module.add(SMovB32(dst=sgpr(srdName + "+3"), src=hex(srdBits),
                     comment="%s[3] = buffer flags" % srdName))


def _initDeepseekScaleSrdSide(module, writer, kernel, isN, off, tileInfo, wgSgpr, wgDim,
                               waveIdSgpr, waveOffSgpr, stmp, strideSgpr, log2wg_m, log2du,
                               srdBits):
  """Emit SRD init instructions for one DeepseekScale side (A or B).

  Computes nKBlocks*wave_bytes stride, calls _dsByteOffset for the per-wave
  global offset, persists DsScaleBBlockStride on the B-side when needed, then
  calls _dsSetSrd to load the pointer and write the SRD quad.
  """
  bufName = "scaleBBuf" if isN else "scaleABuf"
  assert off is not None, "%s not found in numStoreSgprNames" % bufName
  waveSize = kernel["WavefrontSize"]
  waveBytes = waveSize * tileInfo.loadWidthGR
  side = "B" if isN else "A"
  srdName = "SrdMXSB" if isN else "SrdMXSA"
  mt1 = kernel["MacroTile1"] if isN else 128
  module.add(SLShiftRightB32(dst=sgpr(stmp), src=sgpr("SizesSum"),
                             shiftHex=hex(log2du), comment="nKBlocks = K / DepthU"))
  module.add(SMulI32(dst=sgpr(strideSgpr), src0=sgpr(stmp), src1=waveBytes,
                     comment="stride%s = nKBlocks * wave_bytes_%s" % (side, side.lower())))
  _dsByteOffset(module, waveIdSgpr, waveOffSgpr, stmp, log2wg_m, isN,
                wgSgpr, wgDim, strideSgpr, mt1)
  if isN and deepseekScaleBNBlocksPerWave(kernel) > 1:
    module.add(SMovB32(dst=sgpr("DsScaleBBlockStride"), src=sgpr(strideSgpr),
                       comment="persist nKBlocks*wave_bytes for 2nd N-block GR soffset"))
  _dsSetSrd(module, off, stmp, waveOffSgpr, srdName, srdBits)


def _emitUniformWaveIdToSgpr(module, writer, log2ws, waveIdSgpr):
    """Read Serial >> log2ws into a uniform SGPR (VALU write -> readlane hazard guarded)."""
    vWaveId = writer.vgprPool.checkOut(1, tag="ds_srd_waveId")
    module.add(VLShiftRightB32(dst=vgpr(vWaveId), shiftHex=hex(log2ws),
                               src=vgpr("Serial"), comment="waveId = Serial >> log2(waveSize)"))
    module.add(SNop(waitState=0, comment="wait for VGPR before readfirstlane (VALU write hazard)"))
    module.add(VReadfirstlaneB32(dst=sgpr(waveIdSgpr), src=vgpr(vWaveId), comment="uniform waveId"))
    writer.vgprPool.checkIn(vWaveId)


def initDeepseekScaleSrd(writer, kernel):
  """Load ScaleABuf / ScaleBBuf pointers into SrdMXSA/B for DeepseekScale.

  Per wave: SRD base = buf_start + globalIdx * (nKBlocks * wave_bytes), where
  globalIdx = waveIdx + WG * wg_dim and nKBlocks = K / DepthU.  This keeps each
  wave's base at the start of its row-group (A) or N-group (B) slice; the DTL
  advance (emitScaleGRPtrUpdate) then steps by wave_bytes per K-block.

  For non-DP Stream-K (StreamKForceDPOnly=0), WGs with StreamKLocalStart > 0
  start mid-K-dimension. After the base SRD is set, advance it by
  StreamKLocalStart * wave_bytes so the first DTL load reads the correct K-block.
  """
  from .SubtileDeepseekScaleEmit import _scaleBufKernArgOffsets

  offA, offB = _scaleBufKernArgOffsets(writer, kernel)
  wg_m = kernel["MIWaveGroup"][0]
  wg_n = kernel["MIWaveGroup"][1]
  use_a = kernel.get("UseDeepseekScaleA", False)
  use_b = kernel.get("UseDeepseekScaleB", False)
  log2wg_m = wg_m.bit_length() - 1
  waveSize = kernel["WavefrontSize"]
  log2ws = waveSize.bit_length() - 1
  log2du = kernel["DepthU"].bit_length() - 1
  srdBits = writer.states.srdElementBits
  module = Module("DeepseekScale SRD init")

  # stmp[0:2]: scratch for nKBlocks temp and 64-bit ptr load.
  # waveIdSgpr, waveOffSgpr, strideSgpr: per-wave index, byte offset, and
  # nKBlocks*wave_bytes stride (reused for each active side).
  with writer.allocTmpSgpr(5, tag="dsInitSrd") as tmp5:
    stmp = tmp5.idx
    waveIdSgpr = tmp5.idx + 2
    waveOffSgpr = tmp5.idx + 3
    strideSgpr = tmp5.idx + 4

    _emitUniformWaveIdToSgpr(module, writer, log2ws, waveIdSgpr)

    if use_a:
      _initDeepseekScaleSrdSide(module, writer, kernel, False, offA,
                                writer.states.mxsa.tileInfo,
                                "WorkGroup0", wg_m,
                                waveIdSgpr, waveOffSgpr, stmp, strideSgpr,
                                log2wg_m, log2du, srdBits)
    if use_b:
      _initDeepseekScaleSrdSide(module, writer, kernel, True, offB,
                                writer.states.mxsb.tileInfo,
                                "WorkGroup1", wg_n,
                                waveIdSgpr, waveOffSgpr, stmp, strideSgpr,
                                log2wg_m, log2du, srdBits)

    # Non-DP Stream-K: advance SrdMXSA/B by StreamKLocalStart K-blocks so WGs
    # that start mid-K use the correct scale entry. StreamKLocalStart is not
    # allocated in DP-only mode, so skip the offset when ForceDPOnly is set.
    if kernel.get("StreamK", 0) > 0 and not kernel.get("StreamKForceDPOnly", 1):
      if use_a:
        inc_a = waveSize * writer.states.mxsa.tileInfo.loadWidthGR
        module.addComment0("SK non-DP: advance SrdMXSA by StreamKLocalStart K-blocks")
        module.add(SMulI32(dst=sgpr(stmp), src0=sgpr("StreamKLocalStart"), src1=inc_a,
                           comment="skOff = StreamKLocalStart * %d (wave_bytes_a)" % inc_a))
        module.add(SAddU32(dst=sgpr("SrdMXSA"), src0=sgpr("SrdMXSA"), src1=sgpr(stmp),
                           comment="SrdMXSA[0] += skOff (lo)"))
        module.add(SAddCU32(dst=sgpr("SrdMXSA+1"), src0=sgpr("SrdMXSA+1"), src1=0,
                            comment="SrdMXSA[1] += carry"))
      if use_b:
        inc_b = waveSize * writer.states.mxsb.tileInfo.loadWidthGR
        module.addComment0("SK non-DP: advance SrdMXSB by StreamKLocalStart K-blocks")
        module.add(SMulI32(dst=sgpr(stmp), src0=sgpr("StreamKLocalStart"), src1=inc_b,
                           comment="skOff = StreamKLocalStart * %d (wave_bytes_b)" % inc_b))
        module.add(SAddU32(dst=sgpr("SrdMXSB"), src0=sgpr("SrdMXSB"), src1=sgpr(stmp),
                           comment="SrdMXSB[0] += skOff (lo)"))
        module.add(SAddCU32(dst=sgpr("SrdMXSB+1"), src0=sgpr("SrdMXSB+1"), src1=0,
                            comment="SrdMXSB[1] += carry"))

  return module
