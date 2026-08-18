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

from functools import singledispatch
from math import prod

from rocisa.code import Module
from rocisa.container import DSModifiers, EXEC, vgpr, sgpr
from rocisa.enum import RegisterType
from rocisa.instruction import (
    DSLoadB128,
    SMovB32, SMovB64,
    VAddU32, VAndB32, VMovB32, VXorB32,
    VLShiftLeftB32, VLShiftRightB32,
    VMulLOU32, VPermlane16SwapB32,
)

from .SubtileGeometry import (
    LRTag_1x1, LRTag_1x2, LRTag_TLU1,
)
from .SubtileScaleEmit import emitScaleLRLDSSwap


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
_allocLROffsetRegisters.register(LRTag_TLU1)(_stub)
_deallocLROffsetRegisters.register(LRTag_TLU1)(_stub)
_emitLRDTLInit.register(LRTag_TLU1)(_stub)
_emitLRLDSBufferSwap.register(LRTag_TLU1)(_stub)


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
  module = Module(f"LR Offset 1x2 ({ti.tc})")
  tc = ti.tc
  wavesize = kernel["WavefrontSize"]
  subIterKBytes = ti.subIterKBytes
  loadWidth = ti.loadWidthLR
  mi_m = ti.mmaTileShape[0]
  ldsRowBankSize = writer.states.archCaps["LDSBankCount"] * writer.states.archCaps["LDSBankWidth"]
  numRowsPerLDSBanks = ldsRowBankSize // subIterKBytes
  blockSize = subIterKBytes // loadWidth
  numMFMACols = int(ti.mmaTileShape[1] * ti.bpe) // loadWidth

  wg_m     = ti.waveGroupSize
  numWaves = ti.numWaves
  waves_coop = numWaves // wg_m

  tmpVgpr = writer.vgprPool.checkOut(5, tag="_emitLROffset_TLU0_tmpVgpr")
  lane16      = tmpVgpr
  lane16Group = tmpVgpr + 1
  rotation    = tmpVgpr + 2
  rowOffset   = tmpVgpr + 3
  colOffset   = tmpVgpr + 4

  # --- 1. lane16 and lane16Group from Serial ---
  module.add(VAndB32(dst=vgpr(lane16Group), src0=vgpr("Serial"), src1=wavesize-1,
             comment=f"{tc}: laneId"))
  module.add(VLShiftRightB32(dst=vgpr(lane16Group), shiftHex=hex(mi_m.bit_length()-1),
             src=vgpr(lane16Group), comment=f"{tc}: lane16Group = laneId // {mi_m}"))
  module.add(VAndB32(dst=vgpr(lane16), src0=vgpr("Serial"), src1=mi_m-1,
             comment=f"{tc}: lane16 = laneId %% {mi_m}"))

  # --- 2. Swizzling: rotation + permlane16 de-swizzle ---
  module.addComment0(f"{tc}: LR swizzling")
  # ldsRowId = lane16 // numRowsPerLDSBanks
  module.add(VLShiftRightB32(dst=vgpr(rotation), shiftHex=hex(numRowsPerLDSBanks.bit_length()-1),
             src=vgpr(lane16), comment=f"{tc}: lds_row_id"))
  # rotation = (ldsRowId // 2) * 2
  module.add(VLShiftRightB32(dst=vgpr(rotation), shiftHex=hex(1),
             src=vgpr(rotation), comment=f"{tc}: ldsRowId // 2"))
  module.add(VLShiftLeftB32(dst=vgpr(rotation), shiftHex=hex(1),
             src=vgpr(rotation), comment=f"{tc}: (ldsRowId // 2) * 2"))
  # colOffset = (rotation + lane16Group) % blockSize
  module.add(VAddU32(dst=vgpr(colOffset), src0=vgpr(rotation), src1=vgpr(lane16Group),
             comment=f"{tc}: rotation + lane16Group"))
  module.add(VAndB32(dst=vgpr(colOffset), src0=vgpr(colOffset), src1=hex(blockSize-1),
             comment=f"{tc}: %% blockSize"))
  # Permlane16 swap to match GR's quad_perm swizzle pattern
  _setExecMask(module, writer, 0x33333333, 0x33333333)
  module.add(VPermlane16SwapB32(dst=vgpr(colOffset), src=vgpr(colOffset),
             comment=f"{tc}: de-swizzle"))
  _setExecMask(module, writer, -1, -1)

  # --- 3. rowOffset = lane16 * subIterKBytes ---
  module.add(VLShiftLeftB32(dst=vgpr(rowOffset), shiftHex=hex(subIterKBytes.bit_length()-1),
             src=vgpr(lane16), comment=f"{tc}: row = lane16 * {subIterKBytes}"))

  # --- 4. Compute LR offsets for each ds_read within the subtile ---
  # offset[0] = colOffset * loadWidth + rowOffset
  # offset[i] = ((colOffset + i * numMFMACols) % blockSize) * loadWidth + rowOffset
  module.add(VMovB32(dst=vgpr(tile.sharedVgprLROffset[0]), src=vgpr(colOffset),
             comment=f"{tc}: LR offset 0 col"))
  for i in range(1, ti.numLRPerSubtile):
    module.add(VAddU32(dst=vgpr(tile.sharedVgprLROffset[i]),
               src0=vgpr(tile.sharedVgprLROffset[i-1]), src1=hex(numMFMACols),
               comment=f"{tc}: advance col for MFMA {i}"))
    module.add(VAndB32(dst=vgpr(tile.sharedVgprLROffset[i]),
               src0=vgpr(tile.sharedVgprLROffset[i]), src1=hex(blockSize-1),
               comment=f"{tc}: col %% blockSize"))

  for i in range(ti.numLRPerSubtile):
    module.add(VLShiftLeftB32(dst=vgpr(tile.sharedVgprLROffset[i]),
               shiftHex=hex(loadWidth.bit_length()-1), src=vgpr(tile.sharedVgprLROffset[i]),
               comment=f"{tc}: col * {loadWidth}"))
    module.add(VAddU32(dst=vgpr(tile.sharedVgprLROffset[i]),
               src0=vgpr(tile.sharedVgprLROffset[i]), src1=vgpr(rowOffset),
               comment=f"{tc}: row + col"))

  writer.vgprPool.checkIn(tmpVgpr)

  # --- 5. Wave partition: shift LR offsets by wave's LDS region ---
  # Each wave reads from a different partition of LDS along the tc's own wave-group axis.
  # Guard: wg_m > 1 ensures tc's own axis has multiple waves (for A: wg_m, for B: wg_n).
  # Without this guard, a 1x4 WG would wrongly treat A's 4 N-waves as M-partitions.
  if waves_coop > 1 and wg_m > 1:
    # Each wave reads from a different M partition. The A LDS region has size
    # MT * subIterKBytes, split into wg_m partitions (one per M-direction wave).
    # B uses the same stride since B partition also maps 1:1 to M-direction waves.
    MT = ti.globalMMATileGrid[0] * ti.mmaTileShape[0]
    sInterval = MT * subIterKBytes // wg_m

    waveId = writer.vgprPool.checkOut(1, tag="_emitLROffset_TLU0_waveId")
    module.add(VLShiftRightB32(dst=vgpr(waveId), shiftHex=hex(wavesize.bit_length()-1),
               src=vgpr("Serial"), comment=f"{tc}: waveId"))

    if tc == 'A':
      module.add(VAndB32(dst=vgpr(waveId), src0=hex(waves_coop - 1), src1=vgpr(waveId),
                 comment=f"{tc}: waveId %% {waves_coop}"))
    else:
      module.add(VLShiftRightB32(dst=vgpr(waveId),
                 shiftHex=hex(waves_coop.bit_length()-1), src=vgpr(waveId),
                 comment=f"{tc}: waveId // {waves_coop}"))

    tmpSgpr = writer.sgprPool.checkOut(1, tag="_emitLROffset_TLU0_tmpSgpr")
    module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(sInterval),
               comment=f"{tc}: LR partition stride"))
    module.add(VMulLOU32(dst=vgpr(waveId), src1=vgpr(waveId), src0=sgpr(tmpSgpr)))
    for i in range(ti.numLRPerSubtile):
      module.add(VAddU32(dst=vgpr(tile.sharedVgprLROffset[i]),
                 src0=vgpr(tile.sharedVgprLROffset[i]), src1=vgpr(waveId),
                 comment=f"{tc}: + wave partition"))
    writer.vgprPool.checkIn(waveId)
    writer.sgprPool.checkIn(tmpSgpr)
  elif wg_m > 1:
    # waves_coop == 1 but wg_m > 1: each wave owns a separate LDS region
    MT = ti.globalMMATileGrid[0] * ti.mmaTileShape[0]
    sInterval = MT * subIterKBytes // (numWaves)

    waveId = writer.vgprPool.checkOut(1, tag="_emitLROffset_TLU0_waveId")
    module.add(VLShiftRightB32(dst=vgpr(waveId), shiftHex=hex(wavesize.bit_length()-1),
               src=vgpr("Serial"), comment=f"{tc}: waveId"))

    tmpSgpr = writer.sgprPool.checkOut(1, tag="_emitLROffset_TLU0_tmpSgpr")
    module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(sInterval),
               comment=f"{tc}: LR partition stride"))
    module.add(VMulLOU32(dst=vgpr(waveId), src1=vgpr(waveId), src0=sgpr(tmpSgpr)))
    for i in range(ti.numLRPerSubtile):
      module.add(VAddU32(dst=vgpr(tile.sharedVgprLROffset[i]),
                 src0=vgpr(tile.sharedVgprLROffset[i]), src1=vgpr(waveId),
                 comment=f"{tc}: + wave partition"))
    writer.vgprPool.checkIn(waveId)
    writer.sgprPool.checkIn(tmpSgpr)

  # --- 6. Add global LDS start offset for B (B data follows A in LDS) ---
  ldsStartOffset = getattr(writer, f'ldsStartOffset{tc}', 0)
  if ldsStartOffset:
    stmp = writer.sgprPool.checkOut(1, tag="_emitLROffset_TLU0_stmp")
    module.add(SMovB32(dst=sgpr(stmp), src=ldsStartOffset,
               comment=f"{tc}: ldsStartOffset"))
    for i in range(ti.numLRPerSubtile):
      module.add(VAddU32(dst=vgpr(tile.sharedVgprLROffset[i]),
                 src0=vgpr(tile.sharedVgprLROffset[i]), src1=sgpr(stmp),
                 comment=f"{tc}: + LDS offset"))
    writer.sgprPool.checkIn(stmp)

  return module


# --- LR alloc/dealloc (LRTag_1x2) -------------------------------------------

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


# --- LR load emit (LRTag_1x2) -----------------------------------------------

@_emitLocalRead.register(LRTag_1x1)
@_emitLocalRead.register(LRTag_1x2)
def _emitLR_1x2(tag, tile, ti, writer, kernel):
  return Module(f"LR Load 1x2 ({ti.tc})")  # STUB
  """Emit ds_read_b128 for all subtiles in the local grid.

  For each subtile (sId0, sId1), for each MMA tile in K (subtileShape[1]):
    - addrVgpr = sharedVgprLROffset[mfmaId]  (per-lane LDS byte offset)
    - ds_offset = subtile position in LDS     (constant immediate)
    - dst = vgprTiles[tileIdx]                (destination register tile)

  The tile index mapping: for subtile at linearId with numLRPerSubtile reads,
    tileIdx = linearId * numLRPerSubtile + mfmaId
  This assumes non-interleaved layout (subtileShape[0]=1 for 1x2).
  """
  module = Module(f"LR Load 1x2 ({ti.tc})")
  tc = ti.tc
  # TODO: Remove legacy TileInfo dependency after full migration.
  # Uses legacy's grid/sizes/vgprTiles because TileInfo's expanded subtileShape
  # doesn't match the LDS layout computed from legacy values.
  legacyTi = getattr(writer.states, tc.lower()).tileInfo
  subtileSize = int(legacyTi.subtileSize)

  for i in range(int(legacyTi.localSubtileGrid[0])):
    for j in range(int(legacyTi.localSubtileGrid[1])):
      for du in range(int(legacyTi.subtileShape[1])):
        mfmaId = du
        addrVgpr = tile.sharedVgprLROffset[mfmaId]

        # DS offset: subtile position in LDS
        offset = i * subtileSize + j * int(legacyTi.globalSubtileGrid[0]) * subtileSize

        # Destination tile register
        tileIdx = ti.lrTileIndexForSubtile(i, j, mfmaId)
        dstTile = ti.vgprTiles[tileIdx]
        dstVgpr = dstTile.regList.indices[0]
        numRegs = len(dstTile.regList.indices)

        module.add(DSLoadB128(
            dst=vgpr(dstVgpr, numRegs),
            src=vgpr(addrVgpr),
            ds=DSModifiers(offset=offset),
            comment=f"LR {tc}[{i},{j}] k={du}")
        )

  return module


# --- LR DTL init (LRTag_1x2) ------------------------------------------------

@_emitLRDTLInit.register(LRTag_1x1)
@_emitLRDTLInit.register(LRTag_1x2)
def _emitLRDTLInit_1x2(tag, tile, ti, writer, kernel):
  return Module(f"LR DTL Init ({ti.tc})")  # STUB
  """Compute swap VGPRs for LR double-buffering.

  For each sharedVgprLROffset[i], computes the corresponding swap offset:
    swap[i] = XOR(offset[i], offset[i] + ldsTotalSize)
  This mask toggles the LR read between the two LDS buffer halves.
  """
  module = Module(f"LR DTL Init ({ti.tc})")
  stmp = writer.sgprPool.checkOut(1, tag="_emitLRDTLInit_1x2_stmp")
  module.add(SMovB32(dst=sgpr(stmp), src=writer.ldsTotalSize,
             comment=f"{ti.tc}: ldsTotalSize for swap"))

  for i in range(len(tile.sharedVgprLROffset)):
    vOff  = tile.sharedVgprLROffset[i]
    vSwap = tile.sharedVgprLROffsetSwap[i]
    module.add(VAddU32(dst=vgpr(vSwap), src0=vgpr(vOff), src1=sgpr(stmp),
               comment=f"{ti.tc}: offset + ldsTotalSize"))
    module.add(VXorB32(dst=vgpr(vSwap), src0=vgpr(vOff), src1=vgpr(vSwap),
               comment=f"{ti.tc}: swap mask = XOR"))

  writer.sgprPool.checkIn(stmp)
  return module


# --- LR LDS buffer swap (LRTag_1x2) -----------------------------------------

@_emitLRLDSBufferSwap.register(LRTag_1x1)
@_emitLRLDSBufferSwap.register(LRTag_1x2)
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
    rowBytes = int(du * bpe) + int(getattr(tileInfo, "ldsRowPadBytes", 0))
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


def _lraTileAssignment_legacy(writer, kernel):
  module = Module()
  module.addComment0("LR Offset Calculation for Subtile Based Tiling")
  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo
  if tileInfoA.bpe == 1:  # FP8: block-swap swizzle, no VPermlane16Swap
    return _lraTileAssignment_fp8_legacy(writer, kernel, module)
  subIterKBytes = tileInfoA.subIterKBytes
  wavesize = kernel["WavefrontSize"]
  mi_m = tileInfoA.mmaTileShape[0]
  loadWidth = tileInfoA.loadWidthLR
  ldsRowBankSize = writer.states.archCaps["LDSBankCount"] * writer.states.archCaps["LDSBankWidth"]
  # With LDS swizzling (gfx950), K-row is one subtile group; without, full DepthU.
  ldsKBytes = subIterKBytes if writer.states.subtileLdsSwizzle else tileInfoA.depthUBytes
  padBytes = int(getattr(tileInfoA, "ldsRowPadBytes", 0))
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
    rowPadBytes = getattr(tileInfo, "ldsRowPadBytes", 0)
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

  stmp = writer.sgprPool.checkOut(1, tag="_localReadDTLInitCommonSwapVgpr_stmp")
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
  if tc in ['A', 'B']:
    ti_ = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
    return ti_.emitLRLDSBufferSwap(writer, kernel)
  else:
    ti_ = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
    return emitScaleLRLDSSwap(ti_, writer, kernel)
