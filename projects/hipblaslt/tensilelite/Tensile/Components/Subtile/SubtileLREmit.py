# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

################################################################################
# LR (local read) emit and alloc dispatch.
#
# singledispatch over LR tag sentinels (LRTag_1x2, LRTag_TLU1, etc.).
# ABLRTile calls these via self.config.tag as the dispatch key.
#
# Structure:
#   1. Dispatch bases       — @singledispatch declarations
#   2. Implementations      — logic functions decorated with @register
################################################################################

from collections import namedtuple
from functools import singledispatch
from math import prod

from rocisa.code import Module
from rocisa.container import DSModifiers, EXEC, vgpr, sgpr
from rocisa.enum import RegisterType
from rocisa.instruction import (
    DSLoadB128,
    DSLoadB64TrB4, DSLoadB64TrB16,
    SMovB32, SMovB64,
    VAddU32, VAndB32, VBfeU32, VMovB32, VOrB32, VXorB32,
    VLShiftLeftB32, VLShiftRightB32,
    VMulLOU32, VPermlane16SwapB32,
)

from .SubtileGeometry import (
    LRTag_1x1, LRTag_1x2, LRTag_TLU1,
    ldsSwizzleMask, _SWZ_K_BITS_MSB_FIRST,
)
from .SubtileScaleEmit import emitScaleLRLDSSwap
from .SubtileTLUSwizzle import (selectTLUSwizzle, selectTLUColScatter, stripStrideBytes,
                                selectTLU1B16SwizzleBits)


################################################################################
# 1. Dispatch bases
################################################################################

@singledispatch
def _emitLocalReadOffset(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"emitLocalReadOffset not implemented for {type(tag).__name__}")

@singledispatch
def _emitLocalRead(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"emitLocalRead not implemented for {type(tag).__name__}")

@singledispatch
def _allocLROffsetRegisters(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"allocLROffsetRegisters not implemented for {type(tag).__name__}")

@singledispatch
def _deallocLROffsetRegisters(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"deallocLROffsetRegisters not implemented for {type(tag).__name__}")

@singledispatch
def _emitLRDTLInit(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"emitLRDTLInit not implemented for {type(tag).__name__}")

@singledispatch
def _emitLRLDSBufferSwap(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"emitLRLDSBufferSwap not implemented for {type(tag).__name__}")

# Stubs for tags not yet implemented.
_stub = lambda tag, tile, ti, writer, kernel: None
_emitLocalReadOffset.register(LRTag_TLU1)(_stub)
_emitLocalRead.register(LRTag_TLU1)(_stub)


################################################################################
# Helpers
################################################################################

def _setExecMask(module, writer, maskLo, maskHi):
  """Set EXEC mask to a 64-bit immediate value."""
  tmpSgpr = writer.sgprPool.checkOutAligned(2, 2, "setExecMask tmpSgpr", False)
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(maskLo), comment="exec mask lo"))
  module.add(SMovB32(dst=sgpr(tmpSgpr+1), src=hex(maskHi), comment="exec mask hi"))
  module.add(SMovB64(dst=EXEC(), src=sgpr(tmpSgpr, 2), comment="Set exec mask"))
  writer.sgprPool.checkIn(tmpSgpr)

setExecMask = _setExecMask


################################################################################
# 2. Implementations
################################################################################

# --- LR offset emit (TLU=0) --------------------------------------------------

@_emitLocalReadOffset.register(LRTag_1x1)
@_emitLocalReadOffset.register(LRTag_1x2)
def _emitLROffset_TLU0(tag, tile, ti, writer, kernel):
  """LR offset for row-major (TLU=0) subtile with swizzling.

  Ported from legacy lraTileAssignment + _computeLROffset + _applyWavePartitionLROffset.
  Operates on a single tensor component (A or B).

  The LDS read layout uses MFMA register mapping:
    lane16      = laneId % instM    (M row within MMA tile)
    lane16Group = laneId // instM   (K column group)

  Steps:
    1. Compute lane16 and lane16Group from Serial.
    2. Apply rotation and swizzling to colOffset (de-swizzle to match GR's LDS layout).
    3. Compute rowOffset = lane16 * subIterKBytes.
    4. For each ds_read within the subtile: offset = (colOffset + advance) % blockSize * loadWidth + rowOffset.
    5. Apply wave partition offset (shift LR offsets by wave's LDS region).
  """
  return Module(f"LR Offset 1x2 ({ti.tc})")  # STUB


_DS_IMM_LIMIT = 1 << 16   # ds_read offset field is 16-bit unsigned


def _tlu1B16MaxLROffset(ti) -> int:
  """Largest ds immediate _emitSingleDsReadTLU1B16 would ask for, in bytes.

  The read address is one swizzled per-lane base register plus an immediate that
  walks the un-swizzled strip / K-window / k_half terms, so the immediate has to
  span the operand's whole LDS region.  Past 64 KB it no longer fits the ds
  offset field and the top has to be staged through a VGPR -- see
  sharedVgprLRBigOffset.
  """
  g = _tlu1LRGeom(ti, ti.waveSize)
  stripStride = stripStrideBytes(ti)
  kWindowStride = int(ti.globalSubtileGrid[0]) * stripStride
  return ((int(ti.globalSubtileGrid[0]) - 1) * stripStride
          + (int(ti.globalSubtileGrid[1]) - 1) * kWindowStride
          + (int(g.readsPerTile) - 1) * g.kHalfStride)


# --- LR alloc/dealloc (LRTag_1x2) -------------------------------------------

@_allocLROffsetRegisters.register(LRTag_TLU1)
def _allocLROffsetRegs_tlu(tag, tile, ti, writer, kernel):
  """Allocate LR offset registers for the free-dim contiguous (TLU=1) shape.

  For fp4, one base, not one per read.  The transpose read reaches every subtile
  of the strip from a single per-lane address through its immediate offset, so
  the extra bases the row-major shape needs would never be used as an address --
  and each one also costs a swap register and its double-buffer xor, in the main
  loop as well as at setup.

  bf16 takes one register per M-TILE of the strip -- stackM of them, not the
  stackM * readsPerTile that numLRPerSubtile counts.  Its LDS bank-conflict
  swizzle XORs the m-block field of the address, and the m-block field is exactly
  what `tile_m * tileMStride` selects, so that term has to be folded in BEFORE
  the XOR: XOR does not distribute over an addition reaching into the field.

  The other per-read term does not reach it.  The paired read steps by
  kHalfStride = colsPerRead * colStride, and colStride == tileMStride * stackM is
  the top of the m-block field, so the step is a multiple of 2^(field top) and
  changes no bit at or below the mask -- carries included.  _emitTLU1LRSwizzle
  additionally asserts no swizzle bit is sourced from the k_half range, so the
  XOR term itself is identical across the paired reads.  Both halves together
  make the XOR distribute over that term, and _emitSingleDsReadTLU1B16 carries it
  in the ds immediate instead of a register.  See _lraTileAssignment_tlu_b16.
  """
  count = int(ti.lrSubtileShape[0]) if _isTLU1B16(ti) else 1
  tile.sharedVgprLROffset = [
      writer.vgprPool.checkOut(1, tag="_allocLROffsetRegs_tlu_sharedVgprLROffset")
      for _ in range(count)]
  tile.sharedVgprLROffsetSwap = [
      writer.vgprPool.checkOut(1, tag="_allocLROffsetRegs_tlu_sharedVgprLROffsetSwap")
      for _ in range(count)]
  # One scratch address, and only for the operands that outgrow the ds immediate
  # (>= 64 KB of LDS for this tensor).  It is a plain staging register, not an
  # offset, so it needs no swap partner: the add that fills it is re-emitted at
  # every read from whichever base register is live.
  tile.sharedVgprLRBigOffset = None
  if _isTLU1B16(ti) and _tlu1B16MaxLROffset(ti) >= _DS_IMM_LIMIT:
    tile.sharedVgprLRBigOffset = writer.vgprPool.checkOut(
        1, tag="_allocLROffsetRegs_tlu_sharedVgprLRBigOffset")


@_allocLROffsetRegisters.register(LRTag_1x1)
@_allocLROffsetRegisters.register(LRTag_1x2)
def _allocLROffsetRegs_1x2(tag, tile, ti, writer, kernel):
  """Allocate LR offset registers for row-major (TLU=0) 1x2 subtile shape.

  Two register groups are allocated:

  1. sharedVgprLROffset[]: one VGPR per ds_read within a subtile.
     numLRPerSubtile = ceil(lrSubtileSize / (loadWidthLR * waveSize)).
     Each VGPR holds the per-lane byte offset into LDS for one ds_read_b128.

  2. sharedVgprLROffsetSwap[]: same count, used for double-buffering.
     While one set is in use for the current iteration's LR, the other
     holds pre-computed offsets for the next iteration.
  """
  tile.sharedVgprLROffset = []
  tile.sharedVgprLROffsetSwap = []
  for i in range(ti.numLRPerSubtile):
    tile.sharedVgprLROffset.append(writer.vgprPool.checkOut(1, tag="_allocLROffsetRegs_1x2_sharedVgprLROffset"))
    tile.sharedVgprLROffsetSwap.append(writer.vgprPool.checkOut(1, tag="_allocLROffsetRegs_1x2_sharedVgprLROffsetSwap"))


@_deallocLROffsetRegisters.register(LRTag_1x1)
@_deallocLROffsetRegisters.register(LRTag_1x2)
@_deallocLROffsetRegisters.register(LRTag_TLU1)
def _deallocLROffsetRegs_1x2(tag, tile, ti, writer, kernel):
  """Deallocate LR offset registers."""
  if isinstance(tile.sharedVgprLROffset, list):
    for voff in tile.sharedVgprLROffset:
      writer.vgprPool.checkIn(voff)
    tile.sharedVgprLROffset = []
  if isinstance(tile.sharedVgprLROffsetSwap, list):
    for voff in tile.sharedVgprLROffsetSwap:
      writer.vgprPool.checkIn(voff)
    tile.sharedVgprLROffsetSwap = []
  if getattr(tile, "sharedVgprLRBigOffset", None) is not None:
    writer.vgprPool.checkIn(tile.sharedVgprLRBigOffset)
    tile.sharedVgprLRBigOffset = None


# --- LR load emit (LRTag_1x2) -----------------------------------------------

@_emitLocalRead.register(LRTag_1x1)
@_emitLocalRead.register(LRTag_1x2)
def _emitLR_1x2(tag, tile, ti, writer, kernel):
  return Module(f"LR Load 1x2 ({ti.tc})")  # STUB


# --- LR DTL init (LRTag_1x2) ------------------------------------------------

@_emitLRDTLInit.register(LRTag_1x1)
@_emitLRDTLInit.register(LRTag_1x2)
@_emitLRDTLInit.register(LRTag_TLU1)
def _emitLRDTLInit_1x2(tag, tile, ti, writer, kernel):
  return Module(f"LR DTL Init ({ti.tc})")  # STUB


# --- LR LDS buffer swap (LRTag_1x2) -----------------------------------------

@_emitLRLDSBufferSwap.register(LRTag_1x1)
@_emitLRLDSBufferSwap.register(LRTag_1x2)
@_emitLRLDSBufferSwap.register(LRTag_TLU1)
def _emitLRLDSSwap_1x2(tag, tile, ti, writer, kernel):
  """Toggle LR read offsets between double-buffer halves.

  XOR each sharedVgprLROffset with its swap mask to flip to the other buffer.
  """
  module = Module()
  module.addComment0("Emit code to swap %s LR vgpr offsets"%ti.tc)
  for i in range(len(tile.sharedVgprLROffset)):
    vOff  = tile.sharedVgprLROffset[i]
    vSwap = tile.sharedVgprLROffsetSwap[i]
    module.add(VXorB32(dst=vgpr(vOff), src0=vgpr(vOff), src1=vgpr(vSwap), comment=""))
  return module


################################################################################
# Legacy LR emit functions (moved from SubtileBasedKernel.py)
################################################################################

def _computeLROffset(module, tileInfo, colOffset, rowOffset, swizzled):
  tc = tileInfo.tc
  subIterKBytes = tileInfo.subIterKBytes
  loadWidth = tileInfo.loadWidthLR
  numMFMACols = int(tileInfo.mmaTileShape[1] * tileInfo.bpe) // loadWidth  # TN case only
  # Without LDS swizzling (e.g. TDM), the full DepthU tile is contiguous in LDS,
  # so the K-row is depthUBytes wide.  With swizzling, GR writes individual
  # subtile K-groups, so the effective K-row is subIterKBytes.
  ldsKBytes = subIterKBytes if swizzled else tileInfo.depthUBytes
  blockSize = ldsKBytes // loadWidth

  # Each ds_load_b128 fills REGS_PER_DS_READ VGPRs.  Tiles with more VGPRs
  # (e.g. 8-VGPR wave32 BF16 or wave64 FP8) need multiple reads.  Consecutive
  # LR offset entries advance by colsPerRead = numMFMACols / numReadsForTile
  # so entries within the same MMA tile cover equal K sub-portions.
  REGS_PER_DS_READ = loadWidth // 4
  numReadsForTile = tileInfo.geometry.lr.mmaLayout.vgprs // REGS_PER_DS_READ
  colsPerRead = numMFMACols // numReadsForTile

  module.add(VMovB32(dst=vgpr(tileInfo.sharedVgprLROffset[0]), src=vgpr(colOffset), comment="%s: laneId"%tc))
  for vgprId in range(1, len(tileInfo.sharedVgprLROffset)):
    module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId-1]), src1=hex(colsPerRead), comment="%s: colOffset for read %u"%(tc, vgprId)))
    module.add(VAndB32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src1=hex(blockSize-1), comment="%s: colOffset = colOffset %% block_size"%tc))

  for vgprId in range(0, len(tileInfo.sharedVgprLROffset)):
    module.add(VLShiftLeftB32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), shiftHex=hex(loadWidth.bit_length()-1), src=vgpr(tileInfo.sharedVgprLROffset[vgprId]), comment="%s: colOffset*loadWidth"%tc))
    module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src1=vgpr(rowOffset), comment="%s: row + col"%tc))

def _applyWavePartitionLROffset(module, writer, kernel, tileInfo):
  """Apply wave-based partition offset to LR offsets.

  loadRatioGR >= 2.0: no partition needed, contiguous subtiles (1x4 for A , 4x1 for B)
  loadRatioGR == 1.0: 2x2 config, each wave loads half of the subtile
  loadRatioGR == 0.5: 4x1 for A , 1x4 for B. Split in 4 subtiles groups
  """
  tc = tileInfo.tc

  # TDM handles wave partitioning via descriptors
  # For single-wave, TDM puts all data at the wave's LDS base -- no partition needed.
  # For multi-wave, each wave's TDM writes to a different LDS region, so LR
  # offsets must include a per-wave partition offset.
  if kernel.get("enableTDM%s" % tc, False):
    numWaves = prod(kernel["MIWaveGroup"])
    if numWaves == 1:
      return
    # Multi-wave TDM: add per-wave LDS offset based on axis position
    wgM, wgN = kernel["MIWaveGroup"]
    numWavesThisAxis = wgM if tc == 'A' else wgN
    if numWavesThisAxis <= 1:
      return  # this tensor's axis is not split
    wavesize = kernel["WavefrontSize"]
    du = kernel["DepthU"]
    mt = kernel["MacroTile0"] if tc == 'A' else kernel["MacroTile1"]
    bpe = tileInfo.bpe
    waveId = writer.vgprPool.checkOut(1)
    module.add(VLShiftRightB32(dst=vgpr(waveId), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="waveId"))
    # Decompose to axis component
    if tc == 'A' and wgN > 1:
      module.add(VAndB32(dst=vgpr(waveId), src0=hex(wgM - 1), src1=vgpr(waveId), comment="waveIdM = waveId %% %d" % wgM))
    elif tc == 'B' and wgM > 1:
      module.add(VLShiftRightB32(dst=vgpr(waveId), shiftHex=hex(wgM.bit_length()-1), src=vgpr(waveId), comment="waveIdN = waveId / %d" % wgM))
    # LDS offset per wave = waveId_axis * (mt / numWavesThisAxis * (du*bpe + pad))
    rowBytes = int(du * bpe) + int(tileInfo.ldsRowPadBytes)
    ldsPerWave = int(mt // numWavesThisAxis) * rowBytes
    tmpSgpr = writer.sgprPool.checkOut(1)
    module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(ldsPerWave), comment="LDS bytes per wave for %s" % tc))
    module.add(VMulLOU32(dst=vgpr(waveId), src1=vgpr(waveId), src0=sgpr(tmpSgpr), comment="waveOffset"))
    for vgprId in range(len(tileInfo.sharedVgprLROffset)):
      module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src1=vgpr(waveId), comment="%s: TDM wave partition LR offset" % tc))
    writer.vgprPool.checkIn(waveId)
    writer.sgprPool.checkIn(tmpSgpr)
    return

  if tileInfo.loadRatioGR >= 2.0:
    return

  wavesize = kernel["WavefrontSize"]
  subIterKBytes = tileInfo.subIterKBytes
  loadWidth = tileInfo.loadWidthGR

  waveId = writer.vgprPool.checkOut(1, tag="_applyWavePartitionLROffset_waveId")
  module.add(VLShiftRightB32(dst=vgpr(waveId), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="waveId"))

  partitionOffset = tileInfo.mmaTileShape[0] * tileInfo.localSubtileGrid[0]
  numRowsPerWave = wavesize // (subIterKBytes // loadWidth)

  if tileInfo.loadRatioGR == 1.0:
    mWaves = kernel["MIWaveGroup"][0]
    if tc == 'A':
      module.add(VAndB32(dst=vgpr(waveId), src0=hex(mWaves - 1), src1=vgpr(waveId), comment="%s: waveId %% %d"%(tc, mWaves)))
    else:
      module.add(VLShiftRightB32(dst=vgpr(waveId), shiftHex=hex(mWaves.bit_length()-1), src=vgpr(waveId), comment="%s: waveId / %d"%(tc, mWaves)))
    sInterval = partitionOffset * subIterKBytes
  elif tileInfo.loadRatioGR == 0.5:
    sInterval = partitionOffset * subIterKBytes
  else:
    raise NotImplementedError("Unsupported loadRatioGR for wave partition: %s"%str(tileInfo.loadRatioGR))

  if sInterval == 0:
    writer.vgprPool.checkIn(waveId)
    return

  tmpSgpr = writer.sgprPool.checkOut(1, tag="_applyWavePartitionLROffset_tmpSgpr")
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(sInterval), comment="%s: interleave stride"%tc))
  module.add(VMulLOU32(dst=vgpr(waveId), src1=vgpr(waveId), src0=sgpr(tmpSgpr), comment=""))
  for vgprId in range(len(tileInfo.sharedVgprLROffset)):
    module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src1=vgpr(waveId), comment="%s: wave partition LR offset"%tc))

  writer.vgprPool.checkIn(waveId)
  writer.sgprPool.checkIn(tmpSgpr)


##################################################
# Subroutine to generate LR offset calculation code
#
def lraTileAssignment(writer, kernel):
  return _lraTileAssignment_legacy(writer, kernel)

def _lraWavePartitioning_legacy(module, writer, kernel):
  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo
  _applyWavePartitionLROffset(module, writer, kernel, tileInfoA)
  _applyWavePartitionLROffset(module, writer, kernel, tileInfoB)

def _lraTileAssignment_fp8_legacy(writer, kernel, module):
  """FP8 LR offset: block-swap + wave de-rotation for MFMA 16x16x128.

  Two ds_read_b128 per MFMA (numLRPerSubtile=2), using complementary block
  assignments to achieve zero LDS bank conflicts:
    finalColId  = (lane16Group + 2*(lane16 >> 3)) % 4  [undo GR wave rotation]
    colOffset_0 = finalColId + swap_bit * 4
    colOffset_1 = colOffset_0 ^ 4
  where:
    swap_bit = (lane16 >> 1) & 1

  The rotation 2*(lane16>>3) undoes the GR step 2 wave K_group rotation:
  waves with waveId&1==1 (M-rows 8..15) wrote with rotation=2; lane16>=8
  reads them back with de-rotation=2. Together they achieve zero bank conflicts.
  """
  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo
  subIterKBytes = tileInfoA.subIterKBytes
  wavesize = kernel["WavefrontSize"]
  mi_m = tileInfoA.mmaTileShape[0]
  loadWidth = tileInfoA.loadWidthLR
  tmpVgpr = writer.vgprPool.checkOut(6, tag="_lraTileAssignment_fp8_legacy_tmpVgpr")
  lane16, lane16Group, scratch, rowOffset, colOffset0, colOffset1 = range(tmpVgpr, tmpVgpr + 6)
  module.add(VAndB32(dst=vgpr(lane16), src0=vgpr("Serial"), src1=mi_m-1, comment="lane16 = laneId % 16"))
  module.add(VAndB32(dst=vgpr(lane16Group), src0=vgpr("Serial"), src1=wavesize-1, comment="laneId"))
  module.add(VLShiftRightB32(dst=vgpr(lane16Group), shiftHex=hex(mi_m.bit_length()-1), src=vgpr(lane16Group), comment="lane16Group = laneId // 16"))
  module.add(VLShiftRightB32(dst=vgpr(scratch), shiftHex=hex(3), src=vgpr(lane16), comment="lane16 >> 3 (1 if M-row >= 8)"))
  module.add(VLShiftLeftB32(dst=vgpr(scratch), shiftHex=hex(1), src=vgpr(scratch), comment="rotation = 2 * (lane16 >> 3)"))
  module.add(VAddU32(dst=vgpr(colOffset0), src0=vgpr(lane16Group), src1=vgpr(scratch), comment="lane16Group + rotation"))
  module.add(VAndB32(dst=vgpr(colOffset0), src0=vgpr(colOffset0), src1=hex(3), comment="finalColId = (lane16Group + rotation) % 4"))
  module.add(VLShiftRightB32(dst=vgpr(scratch), shiftHex=hex(1), src=vgpr(lane16), comment="lane16 >> 1"))
  module.add(VAndB32(dst=vgpr(scratch), src0=vgpr(scratch), src1=hex(1), comment="swap_bit"))
  module.add(VLShiftLeftB32(dst=vgpr(scratch), shiftHex=hex(2), src=vgpr(scratch), comment="swap_val = swap_bit * 4"))
  module.add(VAddU32(dst=vgpr(colOffset0), src0=vgpr(colOffset0), src1=vgpr(scratch), comment="colOffset_0 = finalColId + swap_val"))
  module.add(VXorB32(dst=vgpr(colOffset1), src0=vgpr(colOffset0), src1=hex(4), comment="colOffset_1 = colOffset_0 ^ 4"))
  module.add(VLShiftLeftB32(dst=vgpr(rowOffset), shiftHex=hex(subIterKBytes.bit_length()-1), src=vgpr(lane16), comment=f"rowOffset = lane16 * {subIterKBytes}"))
  for tileInfo in [tileInfoA, tileInfoB]:
    module.add(VLShiftLeftB32(dst=vgpr(tileInfo.sharedVgprLROffset[0]),
               shiftHex=hex(loadWidth.bit_length()-1), src=vgpr(colOffset0),
               comment=f"{tileInfo.tc}: col0 * {loadWidth}"))
    module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[0]),
               src0=vgpr(tileInfo.sharedVgprLROffset[0]), src1=vgpr(rowOffset),
               comment=f"{tileInfo.tc}: offset[0]"))
    if len(tileInfo.sharedVgprLROffset) > 1:
      module.add(VLShiftLeftB32(dst=vgpr(tileInfo.sharedVgprLROffset[1]),
                 shiftHex=hex(loadWidth.bit_length()-1), src=vgpr(colOffset1),
                 comment=f"{tileInfo.tc}: col1 * {loadWidth}"))
      module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[1]),
                 src0=vgpr(tileInfo.sharedVgprLROffset[1]), src1=vgpr(rowOffset),
                 comment=f"{tileInfo.tc}: offset[1]"))
  writer.vgprPool.checkIn(tmpVgpr)
  _lraWavePartitioning_legacy(module, writer, kernel)
  stmp = writer.sgprPool.checkOut(1, tag="_lraTileAssignment_legacy_stmp")
  module.add(SMovB32(dst=sgpr(stmp), src=writer.ldsStartOffsetB, comment="ldsStartOffsetB"))
  for vgprId in range(len(tileInfoB.sharedVgprLROffset)):
    module.add(VAddU32(dst=vgpr(tileInfoB.sharedVgprLROffset[vgprId]),
               src0=sgpr(stmp),
               src1=vgpr(tileInfoB.sharedVgprLROffset[vgprId]),
               comment="B matrix offset in LDS"))
  writer.sgprPool.checkIn(stmp)
  return module


################################################################################
# TLU=1 bf16 transpose-read geometry
#
# fp4 and bf16 share the LRTag_TLU1 tag but not the read map: ds_read_b64_tr_b4
# hands one lane a whole 16-row MMA-M tile, while ds_read_b64_tr_b16 hands it 4
# M-rows, so four lanes span one tile and an intra-tile `mSub` coordinate exists
# that fp4 simply does not have.  Everything below derives from bpe and
# loadWidthLR rather than being assumed, so the fp4 arm falls out as
# lanesPerCol == 1.
################################################################################

_TLU1LRGeom = namedtuple("_TLU1LRGeom", [
    "instM", "instK", "kGroups", "kPerGroup", "rowsPerLaneRead", "lanesPerCol",
    "colsPerRead", "readsPerTile", "colStride", "kHalfStride", "tileMStride",
    "mSubStride", "regsPerRead",
])


def _tlu1LRGeom(ti, waveSize):
  """Derive the TLU=1 transpose-read constants for one tensor's TileInfo."""
  instM = int(ti.mmaTileShape[0])
  instK = int(ti.mmaTileShape[1])
  bpe = ti.bpe
  loadWidth = int(ti.loadWidthLR)
  rowsPerLaneRead = int(loadWidth / bpe)
  if rowsPerLaneRead <= 0 or instM % rowsPerLaneRead != 0:
    raise RuntimeError(
        f"TLU=1 LR ({ti.tc}): loadWidthLR={loadWidth} at bpe={bpe} gives "
        f"{rowsPerLaneRead} elements/lane, which does not divide instM={instM}")
  lanesPerCol = instM // rowsPerLaneRead
  colsPerRead = instM // lanesPerCol
  kGroups = waveSize // instM
  kPerGroup = instK // kGroups
  readsPerTile = kPerGroup // colsPerRead
  if readsPerTile < 1:
    raise RuntimeError(
        f"TLU=1 LR ({ti.tc}): kPerGroup={kPerGroup} < colsPerRead={colsPerRead}; "
        f"one transpose read would span more K than the MMA tile has")
  colStride = int(ti.lrSubtileShape[0] * instM * bpe)
  tileMStride = int(instM * bpe)
  mSubStride = int(rowsPerLaneRead * bpe)
  # The LDS swizzle XORs a `swizzleBits`-wide field in at bit log2(tileMStride).
  # For that XOR to be a permutation of whole m-blocks it must not collide with
  # the mSub field below it, and colStride must be a clean power-of-two column.
  if colStride & (colStride - 1):
    raise RuntimeError(
        f"TLU=1 LR ({ti.tc}): colStride={colStride} is not a power of two; "
        f"the m-block swizzle field would not be contiguous")
  if mSubStride * (lanesPerCol - 1) >= tileMStride:
    raise RuntimeError(
        f"TLU=1 LR ({ti.tc}): mSub field (max {mSubStride * (lanesPerCol - 1)} B) "
        f"overlaps the swizzle field at {tileMStride} B")
  return _TLU1LRGeom(
      instM=instM, instK=instK, kGroups=kGroups, kPerGroup=kPerGroup,
      rowsPerLaneRead=rowsPerLaneRead, lanesPerCol=lanesPerCol,
      colsPerRead=colsPerRead, readsPerTile=readsPerTile,
      colStride=colStride,
      kHalfStride=int(colsPerRead * colStride),
      tileMStride=tileMStride,
      mSubStride=mSubStride,
      regsPerRead=loadWidth // 4)


def _tlu1TrLoadInst(ti):
  """DS transpose-load class that feeds MMA from the TLU=1 column-major layout.

  Both are 64 bits per thread, which is the only width gfx950 offers for
  transposed reads of B4/B8/B16 data (DS_READ_B128_TR_B6 is the lone 128-bit
  form, and it is 6-bit only).  Do not "widen" the bf16 case to DSLoadB128TrB16:
  that renders as ds_read_tr16_b128, which the gfx950 assembler rejects.
  """
  if ti.bpe == 0.5:
    return DSLoadB64TrB4
  if ti.bpe == 2:
    return DSLoadB64TrB16     # ds_read_b64_tr_b16
  raise RuntimeError(f"No TLU=1 transpose DS load for bpe={ti.bpe} (tensor {ti.tc})")


def _isTLU1B16(tileInfo):
  """True for the bf16/fp16 TLU=1 arm (4 M-rows per lane, so lanesPerCol > 1)."""
  return _isLRTLU1(tileInfo) and float(tileInfo.bpe) == 2


def _lraColScatterBase(writer, module, tc, base, csc):
  """Build the column-scatter LR base LDS byte offset in ``base``.

  On entry ``base`` holds the logical K-column; on exit ``load*blkBytes +
  interleave(col_group)*16`` (m_chunk=0 -- the m_chunk and readIdx terms become
  ds_read immediates in emitSingleDsRead).  The bit-interleave inverts the GR
  de-interleave, so both sides address the identical chunk.
  """
  N = csc.N
  logN = N.bit_length() - 1
  kcol = writer.vgprPool.checkOut(1, tag="_lraColScatter_kcol")
  tbit = writer.vgprPool.checkOut(1, tag="_lraColScatter_tbit")
  tval = writer.vgprPool.checkOut(1, tag="_lraColScatter_tval")
  # load = k_col & (N-1)
  module.add(VAndB32(dst=vgpr(kcol), src0=vgpr(base), src1=hex(N - 1),
             comment="%s: load = k_col %% %u" % (tc, N)))
  # col_group = k_col >> logN (reuse base register to hold it)
  module.add(VLShiftRightB32(dst=vgpr(base), shiftHex=hex(logN), src=vgpr(base),
             comment="%s: col_group = k_col / %u" % (tc, N)))
  # t = interleave(col_group): place col_group bit i at thread bit cgThreadBits[i].
  module.add(VMovB32(dst=vgpr(tval), src=0, comment="%s: interleaved thread = 0" % tc))
  for i, tb in enumerate(csc.cgThreadBits):
    module.add(VLShiftRightB32(dst=vgpr(tbit), shiftHex=hex(i), src=vgpr(base),
               comment="%s: col_group bit %u" % (tc, i)))
    module.add(VAndB32(dst=vgpr(tbit), src0=vgpr(tbit), src1=hex(1),
               comment="%s: isolate" % tc))
    if tb == 0:
      module.add(VOrB32(dst=vgpr(tval), src0=vgpr(tval), src1=vgpr(tbit),
                 comment="%s: -> thread bit 0" % tc))
    else:
      module.add(VLShiftLeftB32(dst=vgpr(tbit), shiftHex=hex(tb), src=vgpr(tbit),
                 comment="%s: -> thread bit %u" % (tc, tb)))
      module.add(VOrB32(dst=vgpr(tval), src0=vgpr(tval), src1=vgpr(tbit),
                 comment="%s: accumulate" % tc))
  # base = load*blkBytes + t*16.  blkBytes = wavesize*16 + padBytes (e.g. 1032)
  # is not a power of two, and v_mul_lo_u32 forbids a literal operand, so build
  # load*blkBytes as (load << log2(wavesize*16)) + load*padBytes.
  blockBits = (csc.blkBytes - csc.padBytes).bit_length() - 1   # log2(wavesize*16)
  module.add(VLShiftLeftB32(dst=vgpr(base), shiftHex=hex(blockBits), src=vgpr(kcol),
             comment="%s: load << %u (wavesize*16)" % (tc, blockBits)))
  if csc.padBytes:
    padTmp = writer.vgprPool.checkOut(1, tag="_lraColScatter_pad")
    padBits = int(csc.padBytes).bit_length() - 1   # padBytes is a power of two (8 -> 3)
    module.add(VLShiftLeftB32(dst=vgpr(padTmp), shiftHex=hex(padBits), src=vgpr(kcol),
               comment="%s: load * %u (pad)" % (tc, csc.padBytes)))
    module.add(VAddU32(dst=vgpr(base), src0=vgpr(base), src1=vgpr(padTmp),
               comment="%s: + load pad -> load*%u" % (tc, csc.blkBytes)))
    writer.vgprPool.checkIn(padTmp)
  module.add(VLShiftLeftB32(dst=vgpr(tval), shiftHex=hex(4), src=vgpr(tval),
             comment="%s: interleaved thread * 16" % tc))
  module.add(VAddU32(dst=vgpr(base), src0=vgpr(base), src1=vgpr(tval),
             comment="%s: col_scatter LDS base" % tc))
  writer.vgprPool.checkIn(tval)
  writer.vgprPool.checkIn(tbit)
  writer.vgprPool.checkIn(kcol)


def _emitTLU1LRSwizzle(module, writer, ti, g, lane16, lane16Group, tc):
  """Emit the bf16 LR-side LDS bank-conflict XOR term into a fresh VGPR.

  Returns the VGPR holding `swizzle << log2(tileMStride)` (the caller XORs it
  into each read offset and checks it back in), or None when the geometry does
  not need a swizzle (lanesPerCol == 1, i.e. fp4).

  `lane16` must already hold colInGroup (post-shift) and `lane16Group` the
  16-lane transpose-group index.

  Mirror of the GR-side permutation in
  SubtileGREmit._emitTLU1GRSwizzleB16; the two MUST stay in step
  (a one-sided swizzle permutes A in LDS without undoing it).  Both permute the
  m-block field -- swizzleBits wide, starting at bit log2(tileMStride) -- by an
  XOR of k bits, MSB-first from _SWZ_K_BITS_MSB_FIRST = (3, 1):

    m_block ^= ((k>>3)&1) << 1 | ((k>>1)&1)          (2 bits, 64-row M-extent)
    m_block ^=  (k>>3)&1                             (1 bit,  32-row M-extent)

  k bit 3 spreads consecutive 16-lane transpose groups across banks; k bit 1
  spreads the four k-rows *within* one group, which at a 128 B column alias each
  other (wi_k 0 vs 2) and which a group-uniform gate cannot reach.

  Sourcing: k = lane16Group*kPerGroup + k_half*colsPerRead + colInGroup with
  colInGroup < colsPerRead, so it splits carry-free into three disjoint bit
  ranges -- [0, log2(colsPerRead)) from colInGroup, [log2(colsPerRead),
  log2(kPerGroup)) from k_half, and the rest from lane16Group.  Neither swizzle
  bit may land in the k_half range: that is the range the paired read walks
  (+kHalfStride), so such a bit would make the XOR differ per read and it could
  not be hoisted into one per-lane VGPR.

  The XOR (rather than an add) is safe because the m-block field belongs
  exclusively to tile_m: mSub contributes below it (mSubStride * mSub <
  tileMStride, checked in _tlu1LRGeom) and colStride/kHalfStride contribute
  above it, so base + read_const cannot carry into it.

  Caller contract: XOR does NOT distribute over an add that reaches into the
  m-block field, so every address term that lands inside that field must already
  be in the offset when this is applied -- GR permuted the *sum*, and computing
  `(tile_m ^ swz) + term` instead of `(term + tile_m) ^ swz` silently reads the
  wrong rows whenever term != 0.  The wave partition splits on which side of the
  field its stride falls:

    - a wave that owns whole strips steps by a multiple of the strip, which is
      above the field, so _lraTLUApplyWaveAndLdsStart may add it afterwards;
    - a wave that SHARES a strip steps by its own M window, a sub-strip stride
      that lands inside the field, so _emitLRWaveWindowOffset folds it in before
      this call and the tail zeroes its window term.

  Sharing is legal by design (Solution._subtileWaveStraddlesStrip permits
  stack % perWave == 0), so the second case is not exotic; it is half the shapes.
  Regression coverage: Tests/unit/test_gr_lr_roundtrip_b16_tlu1.py.
  """
  k_per_group = g.kPerGroup
  tile_m_stride = g.tileMStride
  if g.lanesPerCol <= 1:
    return None
  # Same gate as SubtileGREmit -- see selectTLU1B16SwizzleBits.  A one-sided XOR
  # silently corrupts A, so neither side may decide this locally.
  swzBits = selectTLU1B16SwizzleBits(ti)
  if not swzBits:
    return None
  colBits = g.colsPerRead.bit_length() - 1
  grpBits = k_per_group.bit_length() - 1
  kbits = _SWZ_K_BITS_MSB_FIRST[:swzBits]
  assert len(kbits) == swzBits, (
      "TLU=1 LR swizzle: need %d k bits for a %d-row M-extent but only %d are "
      "defined in _SWZ_K_BITS_MSB_FIRST"
      % (swzBits, int(ti.lrSubtileShape[0]) * g.instM, len(_SWZ_K_BITS_MSB_FIRST)))
  assert all(kbit < colBits or kbit >= grpBits for kbit in kbits), (
      "TLU=1 LR swizzle: k bits %s must come from colInGroup (<%d) or "
      "lane16Group (>=%d), never from the per-read k_half field"
      % (kbits, colBits, grpBits))
  swzVgpr = writer.vgprPool.checkOut(1, tag="_emitTLU1LRSwizzle_swz")
  bitVgpr = writer.vgprPool.checkOut(1, tag="_emitTLU1LRSwizzle_swzbit") \
            if swzBits > 1 else None
  for idx, kbit in enumerate(kbits):
    outPos = swzBits - 1 - idx
    if kbit < colBits:            # intra-group: k bit lives in colInGroup
      srcReg, srcBit, srcName = lane16, kbit, "colInGroup"
    else:                         # inter-group: k bit lives in lane16Group
      srcReg, srcBit, srcName = lane16Group, kbit - grpBits, "lane16Group"
    dst = swzVgpr if idx == 0 else bitVgpr
    module.add(VBfeU32(dst=vgpr(dst), src0=vgpr(srcReg), src1=srcBit, src2=1,
               comment=f"{tc}: swizzle k bit {kbit} = {srcName}[{srcBit}]"))
    if outPos:
      module.add(VLShiftLeftB32(dst=vgpr(dst), shiftHex=hex(outPos), src=vgpr(dst),
                 comment=f"{tc}: -> m-block bit {outPos}"))
    if idx:
      module.add(VOrB32(dst=vgpr(swzVgpr), src0=vgpr(swzVgpr), src1=vgpr(dst),
                 comment=f"{tc}: accumulate swizzle term"))
  if bitVgpr is not None:
    writer.vgprPool.checkIn(bitVgpr)
  module.add(VAndB32(dst=vgpr(swzVgpr), src0=vgpr(swzVgpr), src1=ldsSwizzleMask(swzBits),
             comment=f"{tc}: swizzle mask (0 = disabled)"))
  module.add(VLShiftLeftB32(dst=vgpr(swzVgpr), shiftHex=hex(tile_m_stride.bit_length()-1),
             src=vgpr(swzVgpr), comment=f"{tc}: * {tile_m_stride} (tileMStride)"))
  return swzVgpr


def _lraTLUSharedStripWindow(kernel, tileInfo):
  """(wavesPerStrip, perWaveBytes) when waves SHARE an LDS strip, else None.

  A wave that owns whole strips steps by a multiple of the strip; one that shares
  a strip steps by its own M-tile window, which is SMALLER than the strip and so
  lands inside the m-block field the swizzle XORs.  That distinction is what
  decides whether the window may be added after the XOR or must go before it.
  """
  tc = tileInfo.tc
  axisWaves = kernel["MIWaveGroup"][0] if tc == 'A' else kernel["MIWaveGroup"][1]
  if axisWaves <= 1:
    return None
  wavesPerStrip = int(getattr(tileInfo, "grWavesPerStrip", 1))
  if wavesPerStrip <= 1:
    return None
  perWaveMTiles = int(tileInfo.localMMATileGrid[0])
  perWaveBytes = int(perWaveMTiles * tileInfo.mmaTileShape[0] * tileInfo.bpe)
  return wavesPerStrip, perWaveBytes


def _emitLRWaveWindowOffset(writer, kernel, module, tileInfo, dst):
  """dst += (axis wave id %% wavesPerStrip) * perWaveBytes.  Caller checked shared."""
  tc = tileInfo.tc
  wavesize = kernel["WavefrontSize"]
  mWaves = kernel["MIWaveGroup"][0]
  wavesPerStrip, perWaveBytes = _lraTLUSharedStripWindow(kernel, tileInfo)
  wv = writer.vgprPool.checkOut(1, tag="_lraTileAssignment_tlu_b16_window")
  module.add(VLShiftRightB32(dst=vgpr(wv), shiftHex=hex(wavesize.bit_length() - 1),
             src=vgpr("Serial"), comment="%s: waveId" % tc))
  if tc == 'A':
    module.add(VAndB32(dst=vgpr(wv), src0=vgpr(wv), src1=hex(mWaves - 1),
               comment="%s: waveIdM = waveId %% %d" % (tc, mWaves)))
  else:
    module.add(VLShiftRightB32(dst=vgpr(wv), shiftHex=hex(mWaves.bit_length() - 1),
               src=vgpr(wv), comment="%s: waveIdN = waveId / %d" % (tc, mWaves)))
  module.add(VAndB32(dst=vgpr(wv), src0=vgpr(wv), src1=hex(wavesPerStrip - 1),
             comment="%s: window = axisId %% %u (share of the strip)" % (tc, wavesPerStrip)))
  tmpS = writer.sgprPool.checkOut(1, tag="_lraTileAssignment_tlu_b16_window_s",
                                  preventOverflow=False)
  module.add(SMovB32(dst=sgpr(tmpS), src=hex(perWaveBytes),
             comment="%s: bytes per wave window" % tc))
  module.add(VMulLOU32(dst=vgpr(wv), src0=sgpr(tmpS), src1=vgpr(wv),
             comment="%s: window * %d" % (tc, perWaveBytes)))
  module.add(VAddU32(dst=vgpr(dst), src0=vgpr(dst), src1=vgpr(wv),
             comment="%s: + wave window (pre-swizzle: it lies INSIDE the m-block field)" % tc))
  writer.sgprPool.checkIn(tmpS)
  writer.vgprPool.checkIn(wv)


def _lraTileAssignment_tlu_b16(writer, kernel, module, tileInfo):
  """LR per-lane transpose-read offsets for the bf16 TLU=1 arm.

  Separate from the fp4 arm below because ds_read_b64_tr_b16 has a different
  lane map: a lane holds 4 M-rows of one K column and four lanes span a column,
  so there is an intra-tile `mSub` coordinate.  fp4's ds_read_b64_tr_b4 gives one
  lane a whole 16-row tile and has no such coordinate.

  DirectToLds writes a dense column-major pack, LDS[(k*mExtentRows + m)*bpe], so

      base           = (lane16Group * kPerGroup + colInGroup) * colStride
                       + mSub * mSubStride
      offset[tile_m] = base + tile_m * tileMStride

  then XOR-swizzled by _emitTLU1LRSwizzle to de-conflict the LDS banks.

  Unlike the fp4 arm this needs one VGPR per M-TILE of the strip rather than a
  single base: the swizzle XORs the m-block field, which is exactly the field
  `tile_m * tileMStride` selects, and XOR does not distribute over the addition
  of an immediate that reaches into it.  It does NOT need one per read -- the
  paired-read step k_half * kHalfStride lands entirely above the field, so
  _emitSingleDsReadTLU1B16 carries it in the ds immediate.  See
  _allocLROffsetRegs_tlu for the full argument.
  """
  tc = tileInfo.tc
  wavesize = kernel["WavefrontSize"]
  g = _tlu1LRGeom(tileInfo, wavesize)
  instM = g.instM

  module.addComment0("%s: TLU=1 bf16 LR transpose-read offsets" % tc)
  tmpVgpr = writer.vgprPool.checkOut(3, tag="_lraTileAssignment_tlu_b16_tmp")
  lane16, lane16Group, baseOffset = range(tmpVgpr, tmpVgpr + 3)
  module.add(VAndB32(dst=vgpr(lane16), src0=vgpr("Serial"), src1=instM - 1,
             comment=f"{tc} TLU1: lane16 = Serial %% {instM}"))
  module.add(VAndB32(dst=vgpr(lane16Group), src0=vgpr("Serial"), src1=wavesize - 1,
             comment=f"{tc} TLU1: laneId"))
  module.add(VLShiftRightB32(dst=vgpr(lane16Group), shiftHex=hex(instM.bit_length() - 1),
             src=vgpr(lane16Group), comment=f"{tc} TLU1: lane16Group = laneId // {instM}"))

  mSubVgpr = None
  if g.lanesPerCol > 1:
    mSubVgpr = writer.vgprPool.checkOut(1, tag="_lraTileAssignment_tlu_b16_mSub")
    module.add(VAndB32(dst=vgpr(mSubVgpr), src0=vgpr(lane16), src1=g.lanesPerCol - 1,
               comment=f"{tc} TLU1: mSub = lane16 %% {g.lanesPerCol}"))
    module.add(VLShiftLeftB32(dst=vgpr(mSubVgpr), shiftHex=hex(g.mSubStride.bit_length() - 1),
               src=vgpr(mSubVgpr), comment=f"{tc} TLU1: mSub * {g.mSubStride} (mSubStride)"))
    module.add(VLShiftRightB32(dst=vgpr(lane16), shiftHex=hex(g.lanesPerCol.bit_length() - 1),
               src=vgpr(lane16), comment=f"{tc} TLU1: colInGroup = lane16 // {g.lanesPerCol}"))

  module.add(VLShiftLeftB32(dst=vgpr(baseOffset), shiftHex=hex(g.kPerGroup.bit_length() - 1),
             src=vgpr(lane16Group), comment=f"{tc} TLU1: lane16Group * {g.kPerGroup}"))
  module.add(VAddU32(dst=vgpr(baseOffset), src0=vgpr(baseOffset), src1=vgpr(lane16),
             comment=f"{tc} TLU1: + colInGroup"))
  module.add(VLShiftLeftB32(dst=vgpr(baseOffset), shiftHex=hex(g.colStride.bit_length() - 1),
             src=vgpr(baseOffset), comment=f"{tc} TLU1: * {g.colStride} (colStride)"))
  if mSubVgpr is not None:
    module.add(VAddU32(dst=vgpr(baseOffset), src0=vgpr(baseOffset), src1=vgpr(mSubVgpr),
               comment=f"{tc} TLU1: + mSub byte offset"))
    writer.vgprPool.checkIn(mSubVgpr)

  # A shared strip holds several waves' M windows side by side, so the window
  # offset is a sub-strip stride and lands INSIDE the m-block field the swizzle
  # XORs.  GR permuted whole-strip m-blocks, i.e. it swizzled (window + tile_m);
  # adding the window after the XOR would compute (tile_m ^ swz) + window, which
  # only agrees when the window misses the field -- window == 0.  So fold it in
  # here, before the XOR, and tell the shared tail to skip it.
  if _lraTLUSharedStripWindow(kernel, tileInfo) is not None:
    _emitLRWaveWindowOffset(writer, kernel, module, tileInfo, baseOffset)

  swzVgpr = _emitTLU1LRSwizzle(module, writer, tileInfo, g, lane16, lane16Group, tc)

  # offset[tile_m] = base + tile_m*tileMStride.  read_const can exceed the VOP3
  # inline-constant limit (64), so stage those via an SGPR.  The k_half term is
  # deliberately absent: it lives in the ds immediate (see the docstring).
  stmp = writer.sgprPool.checkOut(1, tag="_lraTileAssignment_tlu_b16_const")
  for tile_m in range(len(tileInfo.sharedVgprLROffset)):
    read_const = tile_m * g.tileMStride
    dst = tileInfo.sharedVgprLROffset[tile_m]
    if read_const == 0:
      module.add(VMovB32(dst=vgpr(dst), src=vgpr(baseOffset),
                 comment=f"{tc} TLU1: LR offset[{tile_m}] (tile_m={tile_m})"))
    elif read_const <= 64:
      module.add(VAddU32(dst=vgpr(dst), src0=vgpr(baseOffset), src1=read_const,
                 comment=f"{tc} TLU1: LR offset[{tile_m}] = base + {read_const}"))
    else:
      module.add(SMovB32(dst=sgpr(stmp), src=hex(read_const),
                 comment=f"{tc} TLU1: const {read_const} for offset[{tile_m}]"))
      module.add(VAddU32(dst=vgpr(dst), src0=vgpr(baseOffset), src1=sgpr(stmp),
                 comment=f"{tc} TLU1: LR offset[{tile_m}] = base + {read_const}"))
    if swzVgpr is not None:
      module.add(VXorB32(dst=vgpr(dst), src0=vgpr(dst), src1=vgpr(swzVgpr),
                 comment=f"{tc} TLU1: XOR swizzle: permute m-block (LDS bank de-conflict)"))
  if swzVgpr is not None:
    writer.vgprPool.checkIn(swzVgpr)
  writer.sgprPool.checkIn(stmp)
  writer.vgprPool.checkIn(tmpVgpr)
  return module


def _lraTileAssignment_tlu(writer, kernel, module, tileInfo):
  """LR per-lane LDS base offset for TLU=1 (NT) transpose reads.

  GR wrote this operand K-major, free-dim contiguous, so one K row is
  ``mStripBytes`` wide and the first read's base is::

      base(lane) = (kGroup * groupKStride + frow) * mStripBytes

  The second read within a tile and the M-tile selection are constant ds offsets
  applied by emitSingleDsRead.  The non-subtile s+m+k formula assumes a padded
  layout and does not match this image.
  """
  tc = tileInfo.tc
  wavesize = kernel["WavefrontSize"]
  instM = int(tileInfo.mmaTileShape[0])
  instK = int(tileInfo.mmaTileShape[1])
  bpe = tileInfo.bpe
  subtileM = int(tileInfo.subtileShape[0] * instM)
  mStripBytes = int(subtileM * bpe)          # LDS bytes per K row (free-dim strip width)
  numGroups = wavesize // instM
  groupKStride = instK // numGroups          # K rows between adjacent lane groups

  # bf16 has its own lane map (ds_read_b64_tr_b16 gives a lane 4 M-rows, not a
  # whole tile) and its own per-read offset registers.  It shares the per-wave
  # and LDS-start tail below, which is written to walk every offset register --
  # for fp4 that list is a single base, so its emission is unchanged.
  if _isTLU1B16(tileInfo):
    _lraTileAssignment_tlu_b16(writer, kernel, module, tileInfo)
    _lraTLUApplyWaveAndLdsStart(writer, kernel, module, tileInfo)
    return module

  tmp = writer.vgprPool.checkOut(2, tag="_lraTileAssignment_tlu_tmp")
  kGroup = tmp
  frow   = tmp + 1
  base   = tileInfo.sharedVgprLROffset[0]

  module.addComment0("%s: TLU=1 LR transpose-read base offset" % tc)
  module.add(VAndB32(dst=vgpr(frow), src0=vgpr("Serial"), src1=hex(instM - 1),
             comment="%s: frow = lane %% %u" % (tc, instM)))
  module.add(VAndB32(dst=vgpr(kGroup), src0=vgpr("Serial"), src1=hex(wavesize - 1),
             comment="%s: laneId" % tc))
  module.add(VLShiftRightB32(dst=vgpr(kGroup), shiftHex=hex(instM.bit_length() - 1),
             src=vgpr(kGroup), comment="%s: kGroup = lane // %u" % (tc, instM)))
  module.add(VLShiftLeftB32(dst=vgpr(kGroup), shiftHex=hex(groupKStride.bit_length() - 1),
             src=vgpr(kGroup), comment="%s: kGroup * %u (groupKStride)" % (tc, groupKStride)))
  module.add(VAddU32(dst=vgpr(base), src0=vgpr(kGroup), src1=vgpr(frow),
             comment="%s: kGroup*%u + frow" % (tc, groupKStride)))
  # ``base`` holds the logical K-column.  Column-scatter (8x1 and up) builds the
  # scattered address from it instead of the contiguous *mStripBytes + XOR path.
  csc = selectTLUColScatter(tileInfo)
  if csc is not None:
    _lraColScatterBase(writer, module, tc, base, csc)
  else:
    module.add(VLShiftLeftB32(dst=vgpr(base), shiftHex=hex(mStripBytes.bit_length() - 1),
               src=vgpr(base), comment="%s: * %u (mStripBytes)" % (tc, mStripBytes)))
  # Apply the same chunk XOR + pad the GR write used; the XOR is an involution,
  # so both sides flip identically and A round-trips.  The source bit is read
  # from ``base`` rather than kGroup because it does not always live in the
  # kGroup sub-field (4x1's chunk[4] comes from frow).  See SubtileTLUSwizzle.
  swz = selectTLUSwizzle(tileInfo)
  if swz:
    CHUNK_BYTE_BITS = 4  # log2(16 bytes per b128 chunk)
    byteFromBit = swz.xorFromBit + CHUNK_BYTE_BITS
    byteToBit   = swz.xorToBit + CHUNK_BYTE_BITS
    swzTmp = writer.vgprPool.checkOut(1, tag="_lraTileAssignment_tlu_swz")
    # swzTmp = (base >> byteFromBit) & 1  -> the chunk[xorFromBit] bit
    module.add(VLShiftRightB32(dst=vgpr(swzTmp), shiftHex=hex(byteFromBit),
               src=vgpr(base), comment="%s: base bit for chunk[%u]" % (tc, swz.xorFromBit)))
    module.add(VAndB32(dst=vgpr(swzTmp), src0=vgpr(swzTmp), src1=hex(1),
               comment="%s: isolate chunk[%u]" % (tc, swz.xorFromBit)))
    module.add(VLShiftLeftB32(dst=vgpr(swzTmp), shiftHex=hex(byteToBit),
               src=vgpr(swzTmp), comment="%s: -> LDS byte bit %u" % (tc, byteToBit)))
    module.add(VXorB32(dst=vgpr(base), src0=vgpr(base), src1=vgpr(swzTmp),
               comment="%s: swizzle chunk[%u]^=chunk[%u]" % (tc, swz.xorToBit, swz.xorFromBit)))
    # Pad: add padBytes once per 64-chunk load-block, using the post-swizzle
    # physical chunk index (byte bit blockChunkBits + log2(16)).
    module.add(VLShiftRightB32(dst=vgpr(swzTmp),
               shiftHex=hex(swz.blockChunkBits + CHUNK_BYTE_BITS),
               src=vgpr(base), comment="%s: load-block index" % tc))
    module.add(VMulLOU32(dst=vgpr(swzTmp), src0=hex(swz.padBytes), src1=vgpr(swzTmp),
               comment="%s: * padBytes" % tc))
    module.add(VAddU32(dst=vgpr(base), src0=vgpr(base), src1=vgpr(swzTmp),
               comment="%s: + load-block pad" % tc))
    writer.vgprPool.checkIn(swzTmp)
  _lraTLUApplyWaveAndLdsStart(writer, kernel, module, tileInfo)
  writer.vgprPool.checkIn(tmp)
  return module


def _lraTLUApplyWaveAndLdsStart(writer, kernel, module, tileInfo):
  """Add the per-wave strip base and the LDS start offset to every LR offset.

  Shared by both TLU=1 arms.  fp4 holds a single base register and bf16 one per
  read, so this walks ``sharedVgprLROffset`` rather than indexing [0]; for fp4
  that list has length one and the emitted code is unchanged.
  """
  tc = tileInfo.tc
  wavesize = kernel["WavefrontSize"]
  offsets = tileInfo.sharedVgprLROffset
  # Multi-wave: LDS holds the full macro tile; each axis-wave reads the strips
  # it owns, at axisId * localSub0 * stripStride bytes.  Mirrors the per-wave GR
  # write base in _globalReadDTLInitCommonSgpr_tlu.
  axisWaves = kernel["MIWaveGroup"][0] if tc == 'A' else kernel["MIWaveGroup"][1]
  if axisWaves > 1:
    mWaves = kernel["MIWaveGroup"][0]
    localSub0 = int(tileInfo.localSubtileGrid[0])
    wavesPerStrip = int(tileInfo.grWavesPerStrip)
    if wavesPerStrip > 1:
      # Shared strip: it holds every axis-wave's M tiles side by side, so a wave
      # steps WITHIN the strip by its own M-tile window rather than by whole
      # strips.  The window is the wave's M extent (localMMATileGrid[0]), not the
      # strip height -- those differ exactly by wavesPerStrip.
      #
      # That stride is SMALLER than a strip, so it overlaps the m-block field the
      # bf16 swizzle XORs.  The bf16 arm therefore folds the window in before its
      # XOR (see _emitLRWaveWindowOffset) and only the strip term is left here;
      # adding it again would double-count.  fp4 swizzles a physical chunk index
      # and is unaffected.
      perWaveMTiles = int(tileInfo.localMMATileGrid[0])
      perWaveBytes = int(perWaveMTiles * tileInfo.mmaTileShape[0] * tileInfo.bpe)
      if _isTLU1B16(tileInfo):
        perWaveBytes = 0
    else:
      perWaveBytes = int(localSub0 * stripStrideBytes(tileInfo))
    wv = writer.vgprPool.checkOut(1, tag="_lraTileAssignment_tlu_wave")
    module.add(VLShiftRightB32(dst=vgpr(wv), shiftHex=hex(wavesize.bit_length() - 1),
               src=vgpr("Serial"), comment="%s: waveId" % tc))
    if tc == 'A':
      module.add(VAndB32(dst=vgpr(wv), src0=vgpr(wv), src1=hex(mWaves - 1),
                 comment="%s: waveIdM = waveId %% %d" % (tc, mWaves)))
    else:
      module.add(VLShiftRightB32(dst=vgpr(wv), shiftHex=hex(mWaves.bit_length() - 1),
                 src=vgpr(wv), comment="%s: waveIdN = waveId / %d" % (tc, mWaves)))
    # A shared strip holds wavesPerStrip waves' M windows and no more, so the
    # axis id splits: its low bits pick the window inside the strip, its high
    # bits pick the strip.  With one strip the high part is zero and this is the
    # plain axisId*perWaveBytes.
    strips = int(tileInfo.globalSubtileGrid[0])
    stripVgpr = None
    if wavesPerStrip > 1 and strips > 1:
      stripVgpr = writer.vgprPool.checkOut(1, tag="_lraTileAssignment_tlu_strip")
      module.add(VLShiftRightB32(dst=vgpr(stripVgpr),
                 shiftHex=hex(wavesPerStrip.bit_length() - 1), src=vgpr(wv),
                 comment="%s: strip = axisId / %u" % (tc, wavesPerStrip)))
      module.add(VAndB32(dst=vgpr(wv), src0=vgpr(wv), src1=hex(wavesPerStrip - 1),
                 comment="%s: window = axisId %% %u" % (tc, wavesPerStrip)))
    tmpS = writer.sgprPool.checkOut(1, tag="_lraTileAssignment_tlu_wave_s", preventOverflow=False)
    module.add(SMovB32(dst=sgpr(tmpS), src=hex(perWaveBytes), comment="%s: LDS wave stride" % tc))
    module.add(VMulLOU32(dst=vgpr(wv), src0=sgpr(tmpS), src1=vgpr(wv),
               comment="%s: wave LDS strip base = axisId*%d" % (tc, perWaveBytes)))
    if stripVgpr is not None:
      stripBytes = int(stripStrideBytes(tileInfo))
      module.add(SMovB32(dst=sgpr(tmpS), src=hex(stripBytes),
                 comment="%s: LDS bytes per strip" % tc))
      module.add(VMulLOU32(dst=vgpr(stripVgpr), src0=sgpr(tmpS), src1=vgpr(stripVgpr),
                 comment="%s: strip * %u" % (tc, stripBytes)))
      module.add(VAddU32(dst=vgpr(wv), src0=vgpr(wv), src1=vgpr(stripVgpr),
                 comment="%s: + strip LDS offset" % tc))
      writer.vgprPool.checkIn(stripVgpr)
    writer.sgprPool.checkIn(tmpS)
    for off in offsets:
      module.add(VAddU32(dst=vgpr(off), src0=vgpr(off), src1=vgpr(wv),
                 comment="%s: + per-wave LDS strip offset" % tc))
    writer.vgprPool.checkIn(wv)
  ldsStartOffset = getattr(writer, "ldsStartOffset%s" % tc, 0)
  if ldsStartOffset:
    for off in offsets:
      module.add(VAddU32(dst=vgpr(off), src0=hex(ldsStartOffset), src1=vgpr(off),
                 comment="%s: + LDS start offset" % tc))
  return module


def _isLRTLU1(tileInfo):
  return bool(tileInfo.lr and isinstance(tileInfo.lr.config.tag, LRTag_TLU1))


def _lraTileAssignment_rowMajorSingle(writer, kernel, module, tileInfo):
  """Row-major (TLU=0) LR offsets for a single tensor.

  Same lane map as the interleaved A+B path, but every parameter comes from
  this tensor's own geometry so it can be paired with a TLU=1 operand.
  """
  tc = tileInfo.tc
  if tileInfo.bpe == 1:
    raise NotImplementedError("fp8 LR offsets are not wired for mixed TLU layouts")
  subIterKBytes = tileInfo.subIterKBytes
  wavesize = kernel["WavefrontSize"]
  mi_m = tileInfo.mmaTileShape[0]
  loadWidth = tileInfo.loadWidthLR
  ldsRowBankSize = writer.states.archCaps["LDSBankCount"] * writer.states.archCaps["LDSBankWidth"]
  ldsKBytes = subIterKBytes if writer.states.subtileLdsSwizzle else tileInfo.depthUBytes
  padBytes = int(tileInfo.ldsRowPadBytes)
  ldsRowStride = ldsKBytes + padBytes
  numRowsPerLDSBanks = ldsRowBankSize // ldsKBytes
  blockSize = ldsKBytes // loadWidth
  tmpVgpr = writer.vgprPool.checkOut(5, tag="_lraTileAssignment_rowMajorSingle_tmpVgpr")
  lane16, lane16Group, rotation, rowOffset, colOffset = range(tmpVgpr, tmpVgpr + 5)
  module.add(VAndB32(dst=vgpr(lane16Group), src0=vgpr("Serial"), src1=wavesize-1, comment="%s: laneId"%tc))
  module.add(VLShiftRightB32(dst=vgpr(lane16Group), shiftHex=hex(mi_m.bit_length()-1), src=vgpr(lane16Group), comment="%s: lane16Group"%tc))
  module.add(VAndB32(dst=vgpr(lane16), src0=vgpr("Serial"), src1=mi_m-1, comment="%s: laneId %%%% %u"%(tc, mi_m)))
  module.add(VMovB32(dst=vgpr(colOffset), src=vgpr(lane16Group), comment="%s: colOffset = lane16Group"%tc))
  if writer.states.subtileLdsSwizzle:
    module.add(VLShiftRightB32(dst=vgpr(rotation), shiftHex=hex(numRowsPerLDSBanks.bit_length()-1), src=vgpr(lane16), comment="lds_row_id"))
    module.add(VLShiftRightB32(dst=vgpr(rotation), shiftHex=hex(1), src=vgpr(rotation), comment="(lds_row_id //2 )"))
    module.add(VLShiftLeftB32(dst=vgpr(rotation), shiftHex=hex(1), src=vgpr(rotation), comment="rotation=(lds_row_id //2) * 2"))
    module.add(VAddU32(dst=vgpr(colOffset), src0=vgpr(rotation), src1=vgpr(lane16Group), comment="colOffset = rotation + lane16Group"))
    setExecMask(module, writer, 0x33333333, 0x33333333)
    module.add(VPermlane16SwapB32(dst=vgpr(colOffset), src=vgpr(colOffset), comment="apply swizzling"))
    setExecMask(module, writer, -1, -1)
  module.add(VAndB32(dst=vgpr(colOffset), src0=vgpr(colOffset), src1=hex(blockSize-1), comment="colOffset = colOffset %% blockSize"))
  if padBytes == 0:
    module.add(VLShiftLeftB32(dst=vgpr(rowOffset), shiftHex=hex(ldsRowStride.bit_length()-1), src=vgpr(lane16), comment="offsetRow = %d*lane16" % ldsRowStride))
  else:
    module.add(VMulLOU32(dst=vgpr(rowOffset), src0=hex(ldsRowStride), src1=vgpr(lane16), comment="offsetRow = %d*lane16" % ldsRowStride))
  _computeLROffset(module, tileInfo, colOffset, rowOffset, writer.states.subtileLdsSwizzle)
  writer.vgprPool.checkIn(tmpVgpr)
  _applyWavePartitionLROffset(module, writer, kernel, tileInfo)
  ldsStartOffset = getattr(writer, "ldsStartOffset%s" % tc, 0)
  if ldsStartOffset:
    for vgprId in range(len(tileInfo.sharedVgprLROffset)):
      module.add(VAddU32(dst=vgpr(tileInfo.sharedVgprLROffset[vgprId]), src0=ldsStartOffset,
                 src1=vgpr(tileInfo.sharedVgprLROffset[vgprId]), comment="%s matrix offset in LDS"%tc))


def _lraTileAssignment_legacy(writer, kernel):
  module = Module()
  module.addComment0("LR Offset Calculation for Subtile Based Tiling")
  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo
  aTLU1 = _isLRTLU1(tileInfoA)
  bTLU1 = _isLRTLU1(tileInfoB)
  # TLU=1 (NT): LDS holds each operand free-dim contiguous (K-major, one K row
  # every mmaTileShape[1]*bpe bytes). The MFMA K-layout is recovered on the read
  # with ds_read_b64_tr_b4, whose per-lane address is a pure (K-group, M-row)
  # ramp -- see _lraTileAssignment_tlu.
  if aTLU1 and bTLU1:
    _lraTileAssignment_tlu(writer, kernel, module, tileInfoA)
    _lraTileAssignment_tlu(writer, kernel, module, tileInfoB)
    return module
  # NN / TT: one operand per layout. The row-major path below shares colOffset
  # and rowOffset between A and B, so the TLU=0 operand takes the single-tensor
  # variant instead.
  if aTLU1 or bTLU1:
    for ti, isTLU1 in ((tileInfoA, aTLU1), (tileInfoB, bTLU1)):
      if isTLU1:
        _lraTileAssignment_tlu(writer, kernel, module, ti)
      else:
        _lraTileAssignment_rowMajorSingle(writer, kernel, module, ti)
    return module
  if tileInfoA.bpe == 1:  # FP8: block-swap swizzle, no VPermlane16Swap
    return _lraTileAssignment_fp8_legacy(writer, kernel, module)
  subIterKBytes = tileInfoA.subIterKBytes
  wavesize = kernel["WavefrontSize"]
  mi_m = tileInfoA.mmaTileShape[0]
  loadWidth = tileInfoA.loadWidthLR
  ldsRowBankSize = writer.states.archCaps["LDSBankCount"] * writer.states.archCaps["LDSBankWidth"]
  # With LDS swizzling (gfx950), K-row is one subtile group; without, full DepthU.
  ldsKBytes = subIterKBytes if writer.states.subtileLdsSwizzle else tileInfoA.depthUBytes
  padBytes = int(tileInfoA.ldsRowPadBytes)
  ldsRowStride = ldsKBytes + padBytes
  numRowsPerLDSBanks = ldsRowBankSize // ldsKBytes
  blockSize = ldsKBytes // loadWidth
  tmpVgpr = writer.vgprPool.checkOut(6, tag="_lraTileAssignment_legacy_tmpVgpr")
  lane16, lane16Group, rotation, rowOffset, colOffset = range(tmpVgpr, tmpVgpr + 5)
  module.add(VAndB32(dst=vgpr(lane16Group), src0=vgpr("Serial"), src1=wavesize-1, comment="laneId"))
  module.add(VLShiftRightB32(dst=vgpr(lane16Group), shiftHex=hex(mi_m.bit_length()-1), src=vgpr(lane16Group), comment="lane16Group"))
  module.add(VAndB32(dst=vgpr(lane16), src0=vgpr("Serial"), src1=mi_m-1, comment="laneId %% 16"))
  module.add(VMovB32(dst=vgpr(colOffset), src=vgpr(lane16Group), comment="colOffset = lane16Group"))
  if writer.states.subtileLdsSwizzle:
    module.add(VLShiftRightB32(dst=vgpr(rotation), shiftHex=hex(numRowsPerLDSBanks.bit_length()-1), src=vgpr(lane16), comment="lds_row_id"))
    module.add(VLShiftRightB32(dst=vgpr(rotation), shiftHex=hex(1), src=vgpr(rotation), comment="(lds_row_id //2 )"))
    module.add(VLShiftLeftB32(dst=vgpr(rotation), shiftHex=hex(1), src=vgpr(rotation), comment="rotation=(lds_row_id //2) * 2"))
    module.add(VAddU32(dst=vgpr(colOffset), src0=vgpr(rotation), src1=vgpr(lane16Group), comment="colOffset = rotation + lane16Group"))
    setExecMask(module, writer, 0x33333333, 0x33333333)
    module.add(VPermlane16SwapB32(dst=vgpr(colOffset), src=vgpr(colOffset), comment="apply swizzling"))
    setExecMask(module, writer, -1, -1)
  module.add(VAndB32(dst=vgpr(colOffset), src0=vgpr(colOffset), src1=hex(blockSize-1), comment="colOffset = colOffset %% blockSize"))
  # Without swizzling, the LDS M-row stride is depthUBytes (contiguous K row).
  # With swizzling, GR writes individual subtile K-groups, so subIterKBytes applies.
  # TDM pad adds 16B per row, breaking pow2; fall back to VMul when padded.
  if padBytes == 0:
    module.add(VLShiftLeftB32(dst=vgpr(rowOffset), shiftHex=hex(ldsRowStride.bit_length()-1), src=vgpr(lane16), comment="offsetRow = %d*lane16" % ldsRowStride))
  else:
    module.add(VMulLOU32(dst=vgpr(rowOffset), src0=hex(ldsRowStride), src1=vgpr(lane16), comment="offsetRow = %d*lane16" % ldsRowStride))
  _computeLROffset(module, tileInfoA, colOffset, rowOffset, writer.states.subtileLdsSwizzle)
  _computeLROffset(module, tileInfoB, colOffset, rowOffset, writer.states.subtileLdsSwizzle)
  writer.vgprPool.checkIn(tmpVgpr)
  _lraWavePartitioning_legacy(module, writer, kernel)
  for vgprId in range(len(tileInfoB.sharedVgprLROffset)):
    module.add(VAddU32(dst=vgpr(tileInfoB.sharedVgprLROffset[vgprId]), src0=writer.ldsStartOffsetB, src1=vgpr(tileInfoB.sharedVgprLROffset[vgprId]), comment="B matrix offset in LDS"))
  return module


def localReadResetOffsetsSubtile(writer, kernel):
  module = Module()
  module.addComment0("REMOVE WHEN IMPLEMNTED: Placeholder for subtile based LR offset reset code")
  for i in range(8):
    module.addComment("")

  return module


def _emitSingleDsReadTLU1B16(tileInfo, sId0, sId1, subIterK, dstTile):
  """ds_read_b64_tr_b16 transpose read(s) for ONE MMA-M tile (bf16 TLU=1).

  Splits the work between the address VGPR and the ds immediate differently from
  the fp4 arm.  `sId0` is a global MMA-M index; it decomposes into

      subtileRow   = sId0 // stackM     -- which LDS strip
      mTileInStrip = sId0 %  stackM     -- which 16-row M-tile inside it

  The M-tile selection is a *swizzled* term -- it sits in the m-block field the
  XOR permutes -- so it was folded into the offset register by
  _lraTileAssignment_tlu_b16, one register per M-tile of the strip, indexed by
  mTileInStrip.

  The paired-read K step is NOT swizzled, despite sitting next to one that is.
  kHalfStride = colsPerRead * colStride and colStride == tileMStride * stackM is
  the top of the m-block field, so readIdx * kHalfStride is a multiple of
  2^(field top) and cannot disturb the field or carry into it; _emitTLU1LRSwizzle
  separately asserts no swizzle bit comes from the k_half range, so the XOR term
  is the same for both reads.  The XOR therefore distributes over it and it rides
  in the ds immediate alongside the un-swizzled strip and K-window terms, which
  match the GR write base.  That is what lets one register serve readsPerTile
  reads instead of one each.

  The immediate is 16-bit, so an operand whose LDS region reaches 64 KB cannot be
  addressed this way throughout -- MT320 at DepthU 128 wants 80 KB.  Those reads
  stage the 64 KB-aligned top of the offset through sharedVgprLRBigOffset and
  keep the remainder in the immediate; the address is unchanged, only its split.
  """
  module = Module()
  trLoad = _tlu1TrLoadInst(tileInfo)
  REGS_PER_TR = int(tileInfo.loadWidthLR) // 4
  g = _tlu1LRGeom(tileInfo, tileInfo.waveSize)
  stackM = int(tileInfo.lrSubtileShape[0])
  subtileRow = sId0 // stackM
  mTileInStrip = sId0 % stackM
  stripStride = stripStrideBytes(tileInfo)
  kWindowStride = int(tileInfo.globalSubtileGrid[0]) * stripStride
  offset = subtileRow * stripStride + sId1 * kWindowStride

  # The property that licenses moving the k_half term out of the register.
  mBlockFieldTop = g.tileMStride * stackM
  assert g.kHalfStride % mBlockFieldTop == 0, (
      "TLU=1 bf16 LR (%s): kHalfStride=%d is not a multiple of the m-block field "
      "top (%d); the paired-read step would reach into the swizzled field and "
      "could not be carried in the ds immediate"
      % (tileInfo.tc, g.kHalfStride, mBlockFieldTop))

  numRegs = len(dstTile.regList.indices)
  numReads = numRegs // REGS_PER_TR
  assert len(tileInfo.sharedVgprLROffset) == stackM, (
      "TLU=1 bf16 LR (%s): expected %d offset registers (one per M-tile of the "
      "strip) but %d were allocated"
      % (tileInfo.tc, stackM, len(tileInfo.sharedVgprLROffset)))
  assert numReads <= g.readsPerTile, (
      "TLU=1 bf16 LR (%s): MMA tile needs %d transpose reads but the geometry "
      "gives only %d per M-tile"
      % (tileInfo.tc, numReads, g.readsPerTile))
  dstVgpr = dstTile.regList.indices[0]
  baseVgpr = tileInfo.sharedVgprLROffset[mTileInStrip]
  bigVgpr = getattr(tileInfo, "sharedVgprLRBigOffset", None)
  for readIdx in range(numReads):
    readOffset = offset + readIdx * g.kHalfStride
    assert readOffset >= 0, (
        "TLU=1 bf16 LR (%s): negative ds offset %d (strip=%u, kWindow=%u, "
        "k_half=%u)" % (tileInfo.tc, readOffset, subtileRow, sId1, readIdx))
    addrVgpr = baseVgpr
    if readOffset >= _DS_IMM_LIMIT:
      # Past the ds offset field.  Stage the whole 64 KB-aligned top through the
      # scratch register and leave the remainder in the immediate.  Splitting an
      # un-swizzled constant across an add and the immediate is exact -- the sum
      # is the same address either way, and the XOR already happened, on the base
      # register, before any of this is added.
      assert bigVgpr is not None, (
          "TLU=1 bf16 LR (%s): ds offset %d needs the staging register but none "
          "was allocated; _tlu1B16MaxLROffset said %d"
          % (tileInfo.tc, readOffset, _tlu1B16MaxLROffset(tileInfo)))
      bias = readOffset - (readOffset % _DS_IMM_LIMIT)
      module.add(VAddU32(dst=vgpr(bigVgpr), src0=hex(bias), src1=vgpr(baseVgpr),
                 comment="%s TLU1: stage LR offset top %d (ds imm is 16-bit)"
                         % (tileInfo.tc, bias)))
      readOffset -= bias
      addrVgpr = bigVgpr
    module.add(trLoad(
        dst=vgpr(dstVgpr + readIdx * REGS_PER_TR, REGS_PER_TR),
        src=vgpr(addrVgpr),
        ds=DSModifiers(offset=readOffset),
        comment="TrSubtile%s[%u, %u] subIterK=%u k_half=%u (strip=%u, tile_m=%u)"
                % (tileInfo.tc, sId0, sId1, subIterK, readIdx,
                   subtileRow, mTileInStrip)))
  return module


def emitSingleDsRead(tileInfo, sId0, sId1, subIterK, dstTile, swizzled=True):
  """Emit DSLoadB128 instruction(s) for one MMA tile within a subtile.

  For wave32 tiles with 8 VGPRs, emits two DSLoadB128 instructions
  (each loading 4 VGPRs) since ds_load_b256 is not available.

  Args:
      tileInfo:  TileInfo (for subtileSize, loadRatioGR, sharedVgprLROffset, tc)
      sId0:      Subtile row index (used for offset computation)
      subIterK:  subIterK index within the subtile (maps to mfmaC; subtileShape[0]=1 so mfmaR=0)
      dstTile:   RegisterTileInfo \u2014 destination vgpr tile for the load
      swizzled:  If True, LDS uses swizzled subtile layout; if False, contiguous K-row layout

  Returns a Module. For tiles with numRegs > 4 (e.g. FP8 8-VGPR tiles), emits
  multiple ds_read_b128 instructions (one per 4 VGPRs), each using the next
  sharedVgprLROffset entry.
  """
  REGS_PER_DS_READ = tileInfo.loadWidthLR // 4  # load width in bytes / 4 bytes per VGPR

  # du maps to mfmaC, mfmaR is always 0 (subtileShape[0]=1)
  mfmaId = tileInfo.getSubtileShapeLinearId(subIterK, 0)

  # TLU=1 bf16: ds_read_b64_tr_b16.  A lane holds 4 M-rows of one K column and
  # four lanes span a column, so the per-read address (which carries the swizzled
  # m-block) lives in its own VGPR and the ds immediate selects only the strip /
  # K-window.  Written before the fp4 arm because both match LRTag_TLU1.
  if _isTLU1B16(tileInfo):
    return _emitSingleDsReadTLU1B16(tileInfo, sId0, sId1, subIterK, dstTile)

  # TLU=1 (NT): transpose read. LDS is K-major (free-dim contiguous), so recover
  # the MFMA K-layout with ds_read_b64_tr_b4. Each transpose read returns 2 VGPRs
  # (a lane holds 16 fp4 K); two reads fill the 4-VGPR fp4 operand. Offsets follow
  # the verified thread map (format.md): the second read covers the next 16 K
  # cols (stride 16 * mStripBytes), and sId0 selects the instM-row M-tile block
  # (stride instM * bpe within the strip).
  if tileInfo.lr and isinstance(tileInfo.lr.config.tag, LRTag_TLU1):
    instM = int(tileInfo.mmaTileShape[0])
    bpe = tileInfo.bpe
    subtileM = int(tileInfo.subtileShape[0] * instM)
    mStripBytes = int(subtileM * bpe)     # LDS bytes per K row (free-dim strip width)
    mTileBytes = int(instM * bpe)         # sId0 M-tile block stride within the strip
    kReadStrideBytes = int(16 * mStripBytes)  # second read steps 16 K cols
    # Column-scatter (8x1+): the scattered LDS layout collapses the readIdx step
    # to a fixed byte stride (the two transpose reads land 16 K-columns apart,
    # which the bit-interleave maps to csc.readStrideBytes).  mTileBytes still
    # steps M-tiles within the strip.  See SubtileTLUSwizzle (TLUColScatter).
    csc = selectTLUColScatter(tileInfo)
    if csc is not None:
      kReadStrideBytes = int(csc.readStrideBytes)
    # sId0 is a global MMA-row index. Split it into which subtile strip and
    # which instM-row M-tile within that strip.  Adjacent strips are stripStride
    # bytes apart in LDS (pad-aware); within a strip, M-tiles step mTileBytes.
    stackM = int(tileInfo.subtileShape[0])
    subtileRow = sId0 // stackM
    mTileInStrip = sId0 % stackM
    stripStride = stripStrideBytes(tileInfo)
    # sId1 is the K-window index (DepthU / MatrixInstK windows per strip).  GR
    # writes window w at w * globalSubtileGrid[0] * stripStride in LDS
    # (emitSingleBufferLoad m0), so the transpose read must add the same term.
    kWindowStride = int(tileInfo.globalSubtileGrid[0]) * stripStride
    addrVgpr = tileInfo.sharedVgprLROffset[0]
    dstVgpr = dstTile.regList.indices[0]
    numRegs = len(dstTile.regList.indices)
    REGS_PER_TR = 2                       # ds_read_b64_tr_b4 returns 2 dwords
    numReads = numRegs // REGS_PER_TR
    module = Module()
    for readIdx in range(numReads):
      offset = (subtileRow * stripStride + sId1 * kWindowStride
                + mTileInStrip * mTileBytes + readIdx * kReadStrideBytes)
      module.add(DSLoadB64TrB4(
          dst=vgpr(dstVgpr + readIdx * REGS_PER_TR, REGS_PER_TR),
          src=vgpr(addrVgpr),
          ds=DSModifiers(offset=offset),
          comment="TrSubtile%s[%u, %u] subIterK=%u read=%u" % (tileInfo.tc, sId0, sId1, subIterK, readIdx)))
    return module

  if swizzled:
    # Swizzled: GR writes individual subtile K-groups into LDS.
    offsetStride = int(tileInfo.subtileSize)
    offset = sId0 * offsetStride + sId1 * int(tileInfo.globalSubtileGrid[0]) * offsetStride
  else:
    # Non-swizzled: full DepthU tile is contiguous in LDS with K as the fast
    # dimension.  Each M-row is depthUBytes wide.  A subtile row covers
    # subtileShape[0] * instM M-rows, so stride = that * depthUBytes.
    instM = int(tileInfo.mmaTileShape[0])
    instK = int(tileInfo.mmaTileShape[1])
    subtileShapeM = int(tileInfo.subtileShape[0])
    subtileShapeK = int(tileInfo.subtileShape[1])
    depthUBytes = int(tileInfo.depthUBytes)
    # Add padding
    rowPadBytes = tileInfo.ldsRowPadBytes
    rowStride = depthUBytes + rowPadBytes
    offsetStride = subtileShapeM * instM * rowStride
    offset = sId0 * offsetStride + sId1 * subtileShapeK * instK * int(tileInfo.bpe)

  dstVgpr = dstTile.regList.indices[0]
  numRegs = len(dstTile.regList.indices)
  numReadsForTile = numRegs // REGS_PER_DS_READ

  module = Module()
  for readIdx in range(numReadsForTile):
    addrVgpr = tileInfo.sharedVgprLROffset[mfmaId * numReadsForTile + readIdx]
    module.add(DSLoadB128(
        dst=vgpr(dstVgpr + readIdx * REGS_PER_DS_READ, REGS_PER_DS_READ),
        src=vgpr(addrVgpr),
        ds=DSModifiers(offset=offset),
        comment="Subtile%s[%u, %u] subIterK=%u read=%u" % (tileInfo.tc, sId0, sId1, subIterK, readIdx)))
  return module



def emitSubtileDsRead(writer, kernel, tileInfo, subtileId):

  module = Module()
  sId0 = subtileId[0]
  sId1 = subtileId[1]

  REGS_PER_DS_READ = tileInfo.loadWidthLR // 4  # load width in bytes / 4 bytes per VGPR
  offsetStride = int(tileInfo.subtileSize)
  offset = sId0 * offsetStride + sId1 * int(tileInfo.globalSubtileGrid[0]) * offsetStride

  lrOffsetIdx = 0
  for du in range(tileInfo.subtileShape[1]):
    mfmaId = tileInfo.getSubtileShapeLinearId(du, 0)
    tileIdx = tileInfo.lrTileIndexForSubtile(sId0, sId1, mfmaId)
    dstTile = tileInfo.vgprTiles[tileIdx]
    dstVgpr = dstTile.regList.indices[0]
    numRegs = len(dstTile.regList.indices)
    # Each tile may need multiple ds_read_b128 when numRegs > 4 (e.g. FP8 8-vgpr tiles).
    # Each read uses the next sharedVgprLROffset entry.
    numReadsForTile = numRegs // REGS_PER_DS_READ
    for readIdx in range(numReadsForTile):
      addrVgpr = tileInfo.sharedVgprLROffset[lrOffsetIdx]
      module.add(DSLoadB128(
          dst=vgpr(dstVgpr + readIdx * REGS_PER_DS_READ, REGS_PER_DS_READ),
          src=vgpr(addrVgpr),
          ds=DSModifiers(offset=offset),
          comment="Subtile%s[%u, %u] subIterK=%u read=%u" % (tileInfo.tc, sId0, sId1, du, readIdx)))
      lrOffsetIdx += 1

  return module

##################################################
# Subroutine to generate LR load code
# Initial idea: maybe store asm in modules in a separate obj?
#
def localReadDoSubtile(tc, writer, kernel):
  module = Module()

  tileInfo = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo

  for i in range(tileInfo.localSubtileGrid[0]):
    for j in range(tileInfo.localSubtileGrid[1]):
        module.add(emitSubtileDsRead(writer, kernel, tileInfo, [i, j]))

  return module


def localReadDTLInitCommonSwapVgpr(writer, kernel):
  module = Module()

  atile = writer.states.a.tileInfo
  btile = writer.states.b.tileInfo

  # One scratch SGPR, released at the end of this function.  The GR soffset
  # allocation ahead of it takes one register per subtile strip and can leave
  # the pool with nothing free while the architectural budget still has room,
  # so let the pool grow instead of failing to emit the kernel.
  stmp = writer.sgprPool.checkOut(1, tag="_localReadDTLInitCommonSwapVgpr_stmp",
                                  preventOverflow=False)
  module.add(SMovB32(dst=sgpr(stmp), src=writer.ldsTotalSize, comment="Store Total Lds Size for one buffer"))
  for i in range(len(atile.sharedVgprLROffset)):
    vgprId = atile.sharedVgprLROffset[i]
    vgprSwapId = atile.sharedVgprLROffsetSwap[i]
    module.add(VAddU32(dst=vgpr(vgprSwapId), src0=vgpr(vgprId), src1=sgpr(stmp), comment=""))
    module.add(VXorB32(dst=vgpr(vgprSwapId), src0=vgpr(vgprId), src1=vgpr(vgprSwapId), comment=""))

  for i in range(len(btile.sharedVgprLROffset)):
    vgprId = btile.sharedVgprLROffset[i]
    vgprSwapId = btile.sharedVgprLROffsetSwap[i]
    module.add(VAddU32(dst=vgpr(vgprSwapId), src0=vgpr(vgprId), src1=sgpr(stmp), comment=""))
    module.add(VXorB32(dst=vgpr(vgprSwapId), src0=vgpr(vgprId), src1=vgpr(vgprSwapId), comment=""))

  writer.sgprPool.checkIn(stmp)
  return module


##################################################
# Subroutine to generate DTL M0 LDS buffer swap
#
def localReadLDSBufferSwap(tc, writer, kernel):
  if kernel.get("1LDSBuffer", 0):
    # One buffer: there is no other half to flip to.  The swap mask is built
    # from writer.ldsTotalSize, which under 1LDSBuffer is the WHOLE allocation,
    # so XORing it would walk every LR base off the end of the kernel's LDS.
    return Module("LR LDS buffer swap (1LDSBuffer: no-op)")
  if tc in ['A', 'B']:
    ti_ = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
    return ti_.emitLRLDSBufferSwap(writer, kernel)
  else:
    ti_ = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
    return emitScaleLRLDSSwap(ti_, writer, kernel)
