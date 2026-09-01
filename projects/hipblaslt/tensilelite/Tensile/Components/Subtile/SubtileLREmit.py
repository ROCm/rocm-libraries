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
    DSLoadB64TrB4,
    SMovB32, SMovB64,
    VAddU32, VAndB32, VMovB32, VOrB32, VXorB32,
    VLShiftLeftB32, VLShiftRightB32,
    VMulLOU32, VPermlane16SwapB32,
)

from .SubtileGeometry import (
    LRTag_1x1, LRTag_1x2, LRTag_TLU1,
)
from .SubtileScaleEmit import emitScaleLRLDSSwap
from .SubtileTLUSwizzle import selectTLUSwizzle, selectTLUColScatter, stripStrideBytes


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

@_allocLROffsetRegisters.register(LRTag_TLU1)
def _allocLROffsetRegs_tlu(tag, tile, ti, writer, kernel):
  """Allocate LR offset registers for the free-dim contiguous (TLU=1) shape.

  One base, not one per read.  The transpose read reaches every subtile of the
  strip from a single per-lane address through its immediate offset, so the
  extra bases the row-major shape needs would never be used as an address --
  and each one also costs a swap register and its double-buffer xor, in the
  main loop as well as at setup.
  """
  tile.sharedVgprLROffset = [writer.vgprPool.checkOut(1, tag="_allocLROffsetRegs_tlu_sharedVgprLROffset")]
  tile.sharedVgprLROffsetSwap = [writer.vgprPool.checkOut(1, tag="_allocLROffsetRegs_tlu_sharedVgprLROffsetSwap")]


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
@_emitLRDTLInit.register(LRTag_TLU1)
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


def _lraColScatterBase(writer, module, tc, base, csc):
  """Build the column-scatter LR base LDS byte offset in ``base``.

  On entry ``base`` holds the logical K-column (kGroup*groupKStride + frow).
  On exit it holds ``load*blkBytes + interleave(col_group)*16`` (m_chunk=0; the
  m_chunk and readIdx terms are added as ds_read immediates in emitSingleDsRead).

      load      = k_col & (N-1)
      col_group = k_col >> log2(N)
      t         = sum_i col_group[i] << cgThreadBits[i]      (bit-interleave)
      base      = load*blkBytes + t*16

  This mirrors the GR de-interleave (physical thread T holds col_group whose bit
  i sits at thread bit cgThreadBits[i]); interleave is its inverse, so GR write
  and LR read address the identical LDS chunk.  See SubtileTLUSwizzle.
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


def _lraTileAssignment_tlu(writer, kernel, module, tileInfo):
  """LR per-lane LDS base offset for TLU=1 (NT) transpose reads.

  GR wrote this operand into LDS free-dim (M/N) contiguous, K-major: one K row
  is ``mStripBytes = subtileM * bpe`` bytes wide (the whole free-dim strip), and
  consecutive K rows are that many bytes apart.

  ds_read_b64_tr_b4 reads a transposed 16(K) x 16(free) block. For the K-major
  LDS image our GR write produces (nibble(M,K) = K*subtileM + M), the per-lane
  base address of the first read (M-tile 0, instr 0) is

      kGroup = lane // instM          (0..numGroups-1)
      frow   = lane %  instM          (free-dim row within the 16-row block)
      base(lane) = (kGroup * groupKStride + frow) * mStripBytes

  where groupKStride = instK // numGroups is the K-row distance between adjacent
  lane groups (32 for fp4 16x16x128 with 4 groups).  The second transpose read
  within a tile (+K/2 of a group) and the M-tile selection are constant ds
  offsets applied by emitSingleDsRead.

  This map is verified on gfx950 hardware (benchmark-tools tr4_nt_readmap,
  formula `fmd`): reading LDS filled in the K-major layout above with these
  per-lane bases reconstructs A[M=lane%16, K=32*(lane//16)+rd*16+slot] exactly
  (2048/2048 slots).  The shipping non-subtile s+m+k formula is co-designed with
  a *padded* layout and does NOT match this unpadded K-major image.
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
  # Column-scatter LR base (8x1 and up): ``base`` now holds the logical K-column
  # (kGroup*groupKStride + frow).  Build the scattered LDS byte address from it
  # (load*blkBytes + interleave(col_group)*16) and skip the contiguous
  # *mStripBytes + single-bit XOR path used by 2x1/4x1.  The per-wave and LDS
  # start tails below still apply.
  csc = selectTLUColScatter(tileInfo)
  if csc is not None:
    _lraColScatterBase(writer, module, tc, base, csc)
  else:
    module.add(VLShiftLeftB32(dst=vgpr(base), shiftHex=hex(mStripBytes.bit_length() - 1),
               src=vgpr(base), comment="%s: * %u (mStripBytes)" % (tc, mStripBytes)))
  # Bank-conflict swizzle: apply the same chunk XOR + load-block pad the GR write
  # used, so the transpose read addresses the permuted physical chunk.  fswz is
  # an involution, so GR and LR apply the identical flip and A round-trips.
  #
  # ``base`` here holds the chunk's LDS byte address, and a b128 chunk is always
  # 16 bytes, so chunk bit b lives at byte bit b + 4 (log2 16), independent of
  # the strip width mStripBytes.  The swizzle bits are pure per-lane for every
  # wired stack (2x1: chunk[6]^=chunk[5]; 4x1: chunk[7]^=chunk[4]), but they do
  # not all live in the kGroup sub-field (4x1's chunk[4] comes from frow), so the
  # source bit is read straight from ``base`` rather than from kGroup -- this is
  # field-agnostic and stays correct as the stack grows.  See SubtileTLUSwizzle.
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
  # Multi-wave: LDS holds the full macro tile; each axis-wave reads the strips
  # it owns, at axisId * localSub0 * stripStride bytes.  Mirrors the per-wave GR
  # write base in _globalReadDTLInitCommonSgpr_tlu.
  axisWaves = kernel["MIWaveGroup"][0] if tc == 'A' else kernel["MIWaveGroup"][1]
  if axisWaves > 1:
    mWaves = kernel["MIWaveGroup"][0]
    localSub0 = int(tileInfo.localSubtileGrid[0])
    wavesPerStrip = int(getattr(tileInfo, "grWavesPerStrip", 1))
    if wavesPerStrip > 1:
      # Shared strip: it holds every axis-wave's M tiles side by side, so a wave
      # steps WITHIN the strip by its own M-tile window rather than by whole
      # strips.  The window is the wave's M extent (localMMATileGrid[0]), not the
      # strip height -- those differ exactly by wavesPerStrip.
      perWaveMTiles = int(tileInfo.localMMATileGrid[0])
      perWaveBytes = int(perWaveMTiles * tileInfo.mmaTileShape[0] * tileInfo.bpe)
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
    module.add(VAddU32(dst=vgpr(base), src0=vgpr(base), src1=vgpr(wv),
               comment="%s: + per-wave LDS strip offset" % tc))
    writer.vgprPool.checkIn(wv)
  ldsStartOffset = getattr(writer, "ldsStartOffset%s" % tc, 0)
  if ldsStartOffset:
    module.add(VAddU32(dst=vgpr(base), src0=hex(ldsStartOffset), src1=vgpr(base),
               comment="%s: + LDS start offset" % tc))
  writer.vgprPool.checkIn(tmp)
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
  padBytes = int(getattr(tileInfo, "ldsRowPadBytes", 0))
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
  if tc in ['A', 'B']:
    ti_ = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
    return ti_.emitLRLDSBufferSwap(writer, kernel)
  else:
    ti_ = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
    return emitScaleLRLDSSwap(ti_, writer, kernel)
