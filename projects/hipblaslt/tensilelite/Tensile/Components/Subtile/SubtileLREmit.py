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
    DSLoadB128, DSLoadB64TrB4,
    SMovB32, SMovB64,
    VAddU32, VAndB32, VMovB32, VXorB32,
    VLShiftLeftB32, VLShiftRightB32,
    VMulLOU32, VPermlane16SwapB32,
)

from .SubtileGeometry import (
    LRTag_1x1, LRTag_1x2, LRTag_TLU1,
)
from .SubtileScaleEmit import emitScaleLRLDSSwap
import math

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



# --- LR alloc/dealloc (LRTag_TLU1) -------------------------------------------
@_allocLROffsetRegisters.register(LRTag_TLU1)
def _allocLROffsetRegs_TLU1(tag, tile, ti, writer, kernel):
  """Allocate LR offset registers for column-major (TLU=1) FP4 transpose reads.
  ds_read_b64_tr_b4 reads 8 bytes/lane (64 bits). One subtile = 2048 B requires
  numLRPerSubtile = 4 reads (2048 / (8 * 64) = 4).
  Two register groups (mirroring the TLU=0 pattern):
    sharedVgprLROffset[]:     4 VGPRs - per-lane LDS byte offset for each ds_read.
    sharedVgprLROffsetSwap[]: 4 VGPRs - double-buffer swap masks (XOR toggles).
  """
  tile.sharedVgprLROffset = []
  tile.sharedVgprLROffsetSwap = []
  for _ in range(ti.numLRPerSubtile):
    tile.sharedVgprLROffset.append(writer.vgprPool.checkOut(1))
    tile.sharedVgprLROffsetSwap.append(writer.vgprPool.checkOut(1))

@_deallocLROffsetRegisters.register(LRTag_TLU1)
def _deallocLROffsetRegs_TLU1(tag, tile, ti, writer, kernel):
  """Free the TLU=1 LR offset and swap registers."""
  if isinstance(tile.sharedVgprLROffset, list):
    for voff in tile.sharedVgprLROffset:
      writer.vgprPool.checkIn(voff)
    tile.sharedVgprLROffset = []
  if isinstance(tile.sharedVgprLROffsetSwap, list):
    for voff in tile.sharedVgprLROffsetSwap:
      writer.vgprPool.checkIn(voff)
    tile.sharedVgprLROffsetSwap = []


# --- LR offset emit (LRTag_TLU1) ---------------------------------------------
@_emitLocalReadOffset.register(LRTag_TLU1)
def _emitLROffset_TLU1(tag, tile, ti, writer, kernel):
  """LR offset for column-major (TLU=1) FP4 transpose reads (ds_read_b64_tr_b4).
  Each lane computes its LDS byte address for 4 ds_read instructions per subtile.
  The transpose read requires lane j in group g to address K-column (g*k_per_group + k_half*16 + j),
  at the correct M-tile offset within that column.
  Formula per lane L:
    lane16       = L % instM
    lane16Group  = L // instM  (= kGroup index)
    base_offset  = (lane16Group * k_per_group + lane16) * col_stride
    offset[r]    = base_offset + tile_m*tile_m_stride + k_half*k_half_stride + wave_partition
  Where:
    col_stride     = subtileShape[0] * instM * bpe   (bytes per K-column in LDS)
    k_per_group    = instK // kGroups                (K-elements assigned to each kGroup)
    tile_m_stride  = instM * bpe                     (M-offset to next MMA tile within a column)
    k_half_stride  = 16 * col_stride                 (advance 16 K-columns for second half)
    wave_partition = mWave * localSubtileGrid[0] * subtileSize
  """
  module = Module(f"LR Offset TLU1 ({ti.tc})")
  tc = ti.tc
  waveSize = kernel["WavefrontSize"]
  instM = ti.mmaTileShape[0]
  instK = ti.mmaTileShape[1]
  bpe = ti.bpe
  kGroups = waveSize // instM
  k_per_group = instK // kGroups
  col_stride = int(ti.lrSubtileShape[0] * instM * bpe)
  k_half_stride = 16 * col_stride
  tile_m_stride = int(instM * bpe)
  reads_per_tile = k_per_group // 16
  # --- 1. Extract lane16 and lane16Group from Serial ---
  tmpVgpr = writer.vgprPool.checkOut(3)
  lane16      = tmpVgpr
  lane16Group = tmpVgpr + 1
  baseOffset  = tmpVgpr + 2
  module.add(VAndB32(dst=vgpr(lane16), src0=vgpr("Serial"), src1=instM - 1,
             comment=f"{tc}: lane16 = laneId % {instM}"))
  module.add(VAndB32(dst=vgpr(lane16Group), src0=vgpr("Serial"), src1=waveSize - 1,
             comment=f"{tc}: laneId = Serial % {waveSize}"))
  module.add(VLShiftRightB32(dst=vgpr(lane16Group),
             shiftHex=hex(instM.bit_length() - 1), src=vgpr(lane16Group),
             comment=f"{tc}: lane16Group = laneId // {instM}"))
  # --- 2. base_offset = (lane16Group * k_per_group + lane16) * col_stride ---
  module.add(VLShiftLeftB32(dst=vgpr(baseOffset),
             shiftHex=hex(k_per_group.bit_length() - 1), src=vgpr(lane16Group),
             comment=f"{tc}: lane16Group * {k_per_group}"))
  module.add(VAddU32(dst=vgpr(baseOffset), src0=vgpr(baseOffset), src1=vgpr(lane16),
             comment=f"{tc}: + lane16"))
  module.add(VLShiftLeftB32(dst=vgpr(baseOffset),
             shiftHex=hex(col_stride.bit_length() - 1), src=vgpr(baseOffset),
             comment=f"{tc}: * {col_stride} (col_stride)"))
  # --- 3. sharedVgprLROffset[r] = base_offset + read_constant[r] ---
  for r in range(ti.numLRPerSubtile):
    tile_m = r // reads_per_tile
    k_half = r % reads_per_tile
    read_const = tile_m * tile_m_stride + k_half * k_half_stride
    if read_const == 0:
      module.add(VMovB32(dst=vgpr(tile.sharedVgprLROffset[r]), src=vgpr(baseOffset),
                 comment=f"{tc}: LR offset[{r}] (tile_m={tile_m}, k_half={k_half})"))
    else:
      module.add(VAddU32(dst=vgpr(tile.sharedVgprLROffset[r]),
                 src0=vgpr(baseOffset), src1=hex(read_const),
                 comment=f"{tc}: LR offset[{r}] = base + {read_const} (tile_m={tile_m}, k_half={k_half})"))
  writer.vgprPool.checkIn(tmpVgpr)
  # --- 4. Wave partition offset ---
  mWaves = kernel["MIWaveGroup"][0]
  if mWaves > 1:
    partition_stride = int(ti.localSubtileGrid[0]) * int(ti.subtileSize)
    waveId = writer.vgprPool.checkOut(1)
    module.add(VLShiftRightB32(dst=vgpr(waveId),
               shiftHex=hex(waveSize.bit_length() - 1), src=vgpr("Serial"),
               comment=f"{tc}: waveId = Serial // {waveSize}"))
    module.add(VAndB32(dst=vgpr(waveId), src0=vgpr(waveId), src1=hex(mWaves - 1),
               comment=f"{tc}: mWave = waveId % {mWaves}"))
    tmpSgpr = writer.sgprPool.checkOut(1)
    module.add(SMovB32(dst=sgpr(tmpSgpr), src=hex(partition_stride),
               comment=f"{tc}: wave partition stride = {partition_stride}"))
    module.add(VMulLOU32(dst=vgpr(waveId), src1=vgpr(waveId), src0=sgpr(tmpSgpr)))
    for r in range(ti.numLRPerSubtile):
      module.add(VAddU32(dst=vgpr(tile.sharedVgprLROffset[r]),
                 src0=vgpr(tile.sharedVgprLROffset[r]), src1=vgpr(waveId),
                 comment=f"{tc}: + wave partition"))
    writer.vgprPool.checkIn(waveId)
    writer.sgprPool.checkIn(tmpSgpr)
  return module


# --- LR load emit (LRTag_TLU1) -----------------------------------------------
@_emitLocalRead.register(LRTag_TLU1)
def _emitLR_TLU1(tag, tile, ti, writer, kernel):
  """Emit ds_read_b64_tr_b4 for all subtiles in the local grid.
  For each subtile (sId0, sId1), emits numLRPerSubtile (=4) transpose reads.
  Each ds_read_b64_tr_b4 reads 8 bytes/lane (64 bits) = 2 destination VGPRs.
  Two reads fill one MMA tile (4 VGPRs). Two MMA tiles per subtile = 4 reads.
  Address: sharedVgprLROffset[r] + ds_offset
    - sharedVgprLROffset[r]: per-lane LDS byte offset (computed by _emitLROffset_TLU1,
      includes wave partition offset).
    - ds_offset: subtile position in LDS (constant immediate).
      Formula: sId0 * subtileSize + sId1 * globalSubtileGrid[0] * subtileSize
  Destination register mapping:
    reads_per_tile = numLRPerSubtile / tilesPerSubtile  (= 4/2 = 2)
    mfmaId     = r // reads_per_tile     (which MMA tile within the subtile)
    readInTile = r % reads_per_tile      (which 2-VGPR half of that tile)
    tileIdx    = lrTileIndexForSubtile(sId0, sId1, mfmaId)
    dstVgpr    = vgprTiles[tileIdx].start + readInTile * REGS_PER_READ
  """
  module = Module(f"LR Load TLU1 ({ti.tc})")
  tc = ti.tc
  subtileSize = int(ti.subtileSize)
  globalGrid0 = int(ti.globalSubtileGrid[0])
  tilesPerSubtile = int(ti.lrSubtileShape[0]) * int(ti.lrSubtileShape[1])
  reads_per_tile = ti.numLRPerSubtile // tilesPerSubtile
  REGS_PER_READ = 2  # ds_read_b64_tr_b4 -> 64 bits -> 2 VGPRs
  for i in range(int(ti.lrLocalSubtileGrid[0])):
    for j in range(int(ti.lrLocalSubtileGrid[1])):
      ds_offset = i * subtileSize + j * globalGrid0 * subtileSize
      for r in range(ti.numLRPerSubtile):
        mfmaId = r // reads_per_tile
        readInTile = r % reads_per_tile
        addrVgpr = tile.sharedVgprLROffset[r]
        tileIdx = ti.lrTileIndexForSubtile(i, j, mfmaId)
        dstTile = ti.vgprTiles[tileIdx]
        dstVgpr = dstTile.regList.indices[0] + readInTile * REGS_PER_READ
        module.add(DSLoadB64TrB4(
            dst=vgpr(dstVgpr, REGS_PER_READ),
            src=vgpr(addrVgpr),
            ds=DSModifiers(offset=ds_offset),
            comment=f"LR {tc}[{i},{j}] mfma={mfmaId} k_half={readInTile}"))
  return module


# --- LR DTL init (LRTag_TLU1) ------------------------------------------------
@_emitLRDTLInit.register(LRTag_TLU1)
def _emitLRDTLInit_TLU1(tag, tile, ti, writer, kernel):
  """Compute swap VGPRs for LR double-buffering (TLU=1 path).
  For each sharedVgprLROffset[i], computes the XOR swap mask:
    swap[i] = offset[i] XOR (offset[i] + ldsTotalSize)
  This mask, when XOR'd with an address in Buffer 0, produces the
  corresponding address in Buffer 1, and vice versa. The swap masks
  are computed once at kernel init and reused every loop iteration.
  """
  module = Module(f"LR DTL Init TLU1 ({ti.tc})")
  stmp = writer.sgprPool.checkOut(1)
  module.add(SMovB32(dst=sgpr(stmp), src=writer.ldsTotalSize,
             comment=f"{ti.tc}: ldsTotalSize for swap"))
  for i in range(len(tile.sharedVgprLROffset)):
    vOff  = tile.sharedVgprLROffset[i]
    vSwap = tile.sharedVgprLROffsetSwap[i]
    module.add(VAddU32(dst=vgpr(vSwap), src0=vgpr(vOff), src1=sgpr(stmp),
               comment=f"{ti.tc}: offset[{i}] + ldsTotalSize"))
    module.add(VXorB32(dst=vgpr(vSwap), src0=vgpr(vOff), src1=vgpr(vSwap),
               comment=f"{ti.tc}: swap[{i}] = XOR mask"))
  writer.sgprPool.checkIn(stmp)
  return module

# --- LR LDS buffer swap (LRTag_TLU1) -----------------------------------------
@_emitLRLDSBufferSwap.register(LRTag_TLU1)
def _emitLRLDSSwap_TLU1(tag, tile, ti, writer, kernel):
  """Toggle LR read offsets between double-buffer halves (TLU=1 path).
  XOR each sharedVgprLROffset with its precomputed swap mask.
  After this, all subsequent ds_read_b64_tr_b4 instructions will
  read from the other LDS buffer half.
  """
  module = Module(f"LR LDS Swap TLU1 ({ti.tc})")
  for i in range(len(tile.sharedVgprLROffset)):
    vOff  = tile.sharedVgprLROffset[i]
    vSwap = tile.sharedVgprLROffsetSwap[i]
    module.add(VXorB32(dst=vgpr(vOff), src0=vgpr(vOff), src1=vgpr(vSwap),
               comment=f"{ti.tc}: toggle buffer"))
  return module


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
    # LDS offset per wave = waveId_axis * (mt / numWavesThisAxis * du * bpe)
    ldsPerWave = int(mt // numWavesThisAxis * du * bpe)
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
  tileInfoA = writer.states.a.tileInfo
  # Detect TLU=1 for A: column-major data requires transpose LR addresses
  isTLU1_A = (hasattr(tileInfoA, 'gr') and tileInfoA.gr is not None
              and getattr(tileInfoA.gr.config, 'tlu', False))
  if isTLU1_A:
    return _lraTileAssignment_tlu1_a(writer, kernel)
  return _lraTileAssignment_legacy(writer, kernel)


def _lraTileAssignment_tlu1_a(writer, kernel):
  """LR offset computation when tensor A is TLU=1 (column-major, FP4).
  DTL hardware semantics: the per-lane VGPR `voff` only affects the GLOBAL
  fetch address - it does NOT affect LDS. Each lane Q's 16 bytes land at
  LDS[m0 + Q*16]  (offset12 does NOT affect LDS on GFX950). So the LDS layout is DENSE:
      LDS[LWB_A(wave) + sub_offset + Q*16]
  where LWB_A is per-wave (set in _globalReadDTLInitCommonSgpr_legacy):
      LWB_A(wave) = mGroup * GROUP + kCoop * COOP
  Layout per-slot (sId0, sId1) of subtileSize bytes:
      Bytes [0..1023]   : c=0 wave's data  (K=0..63 of M=M_start..M_start+31)
      Bytes [1024..2047]: c=1 wave's data  (K=64..127 of M=M_start..M_start+31)
      Within each K-col's 16 bytes: M=0..31 packed densely (col-major).
  LR formula (DENSE):
      base       = (lane16Group * k_per_group + lane16) * col_stride
      offset[r]  = base + tile_m * tile_m_stride + k_half * k_half_stride
  where:
      col_stride    = lrSubtileShape[0] * instM * bpe = 16
      k_half_stride = 16 * col_stride               = 256
      tile_m_stride = instM * bpe                   = 8
  Wave partition (M-group): adds mWave * GROUP_BYTES to each LR offset
  where mWave = waveId % mWaves (using VAndB32: waveId & (mWaves-1)).
  """
  module = Module()
  module.addComment0("LR Offset Calculation for Subtile Based Tiling")
  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo
  wavesize = kernel["WavefrontSize"]
  instM = tileInfoA.mmaTileShape[0]
  # ------------------------------------------------------------------
  # Part 1: TLU=1 offsets for A (transpose-read addresses)
  # ------------------------------------------------------------------
  instK = tileInfoA.mmaTileShape[1]
  bpe_a = tileInfoA.bpe
  kGroups = wavesize // instM
  k_per_group = instK // kGroups
  col_stride = int(tileInfoA.lrSubtileShape[0] * instM * bpe_a)  # 2*16*0.5 = 16
  k_half_stride = 16 * col_stride                                 # 256
  tile_m_stride = int(instM * bpe_a)                              # 8
  reads_per_tile = k_per_group // 16                              # 2
  tmpVgpr = writer.vgprPool.checkOut(3)
  lane16      = tmpVgpr
  lane16Group = tmpVgpr + 1
  baseOffset  = tmpVgpr + 2
  # Extract lane16 and lane16Group from Serial
  module.add(VAndB32(dst=vgpr(lane16), src0=vgpr("Serial"), src1=instM - 1,
             comment="A TLU1: lane16 = Serial %% %d" % instM))
  module.add(VAndB32(dst=vgpr(lane16Group), src0=vgpr("Serial"), src1=wavesize - 1,
             comment="A TLU1: laneId"))
  module.add(VLShiftRightB32(dst=vgpr(lane16Group),
             shiftHex=hex(int(math.log2(instM))), src=vgpr(lane16Group),
             comment="A TLU1: lane16Group = laneId // %d" % instM))
  # base = (lane16Group * k_per_group + lane16) * col_stride
  module.add(VLShiftLeftB32(dst=vgpr(baseOffset),
             shiftHex=hex(int(math.log2(k_per_group))), src=vgpr(lane16Group),
             comment="A TLU1: lane16Group * %d" % k_per_group))
  module.add(VAddU32(dst=vgpr(baseOffset), src0=vgpr(baseOffset), src1=vgpr(lane16),
             comment="A TLU1: + lane16"))
  module.add(VLShiftLeftB32(dst=vgpr(baseOffset),
             shiftHex=hex(int(math.log2(col_stride))), src=vgpr(baseOffset),
             comment="A TLU1: * %d (col_stride)" % col_stride))
  # Compute sharedVgprLROffset[r] = base + tile_m*tile_m_stride + k_half*k_half_stride
  # k_half_stride=256 and 256+8=264 are >64, so they go via SGPR (VOP3 inline limit).
  numLR_A = len(tileInfoA.sharedVgprLROffset)
  stmp = writer.sgprPool.checkOut(1)
  for r in range(numLR_A):
    tile_m = r // reads_per_tile
    k_half = r % reads_per_tile
    read_const = tile_m * tile_m_stride + k_half * k_half_stride
    dst = tileInfoA.sharedVgprLROffset[r]
    if read_const == 0:
      module.add(VMovB32(dst=vgpr(dst), src=vgpr(baseOffset),
                 comment="A TLU1: LR offset[%d] (tile_m=%d, k_half=%d)"
                 % (r, tile_m, k_half)))
    elif read_const <= 64:
      # Inline constant: safe to use directly in VOP3
      module.add(VAddU32(dst=vgpr(dst), src0=vgpr(baseOffset), src1=read_const,
                 comment="A TLU1: LR offset[%d] = base + %d (tile_m=%d, k_half=%d)"
                 % (r, read_const, tile_m, k_half)))
    else:
      # Load into SGPR first: literal constants > 64 not supported in VOP3
      module.add(SMovB32(dst=sgpr(stmp), src=hex(read_const),
                 comment="A TLU1: const %d for offset[%d]" % (read_const, r)))
      module.add(VAddU32(dst=vgpr(dst), src0=vgpr(baseOffset), src1=sgpr(stmp),
                 comment="A TLU1: LR offset[%d] = base + %d (tile_m=%d, k_half=%d)"
                 % (r, read_const, tile_m, k_half)))
  writer.sgprPool.checkIn(stmp)
  writer.vgprPool.checkIn(tmpVgpr)
  # Wave partition for A: mWave = waveId % mWaves; partition_stride = GROUP_BYTES
  # GROUP_BYTES = lsg0*lsg1*subtileSize = 8192 (NOT lsg0*subtileSize=4096, which is missing lsg1).
  mWaves = kernel["MIWaveGroup"][0]
  if mWaves > 1:
    kCoopWaves = kernel["MIWaveGroup"][1]
    GROUP_BYTES = (int(tileInfoA.localSubtileGrid[0])
                   * int(tileInfoA.localSubtileGrid[1])
                   * int(tileInfoA.subtileSize))
    waveId = writer.vgprPool.checkOut(1)
    module.add(VLShiftRightB32(dst=vgpr(waveId),
               shiftHex=hex(int(math.log2(wavesize))), src=vgpr("Serial"),
               comment="A TLU1: waveId"))
    # mWave = waveId % mWaves - matches output waveIdM and MXSA scale partition
    module.add(VAndB32(dst=vgpr(waveId),
               src0=vgpr(waveId), src1=mWaves-1,
               comment="A TLU1: mWave = waveId %% %d" % mWaves))
    stmp2 = writer.sgprPool.checkOut(1)
    module.add(SMovB32(dst=sgpr(stmp2), src=hex(GROUP_BYTES),
               comment="A TLU1: wave partition stride = %d" % GROUP_BYTES))
    module.add(VMulLOU32(dst=vgpr(waveId), src1=vgpr(waveId), src0=sgpr(stmp2),
               comment="A TLU1: mWave * GROUP_BYTES"))
    for r in range(numLR_A):
      module.add(VAddU32(dst=vgpr(tileInfoA.sharedVgprLROffset[r]),
                 src0=vgpr(tileInfoA.sharedVgprLROffset[r]), src1=vgpr(waveId),
                 comment="A TLU1: + wave partition"))
    writer.vgprPool.checkIn(waveId)
    writer.sgprPool.checkIn(stmp2)
  # ------------------------------------------------------------------
  # Part 2: TLU=0 offsets for B (standard row-major swizzle)
  # ------------------------------------------------------------------
  subIterKBytes_B = tileInfoB.subIterKBytes
  loadWidth_B = tileInfoB.loadWidthLR
  mi_m_B = tileInfoB.mmaTileShape[0]
  ldsRowBankSize = writer.states.archCaps["LDSBankCount"] * writer.states.archCaps["LDSBankWidth"]
  numRowsPerLDSBanks = ldsRowBankSize // subIterKBytes_B
  blockSize_B = subIterKBytes_B // loadWidth_B

  tmpVgpr2 = writer.vgprPool.checkOut(5)
  lane16_b      = tmpVgpr2
  lane16Group_b = tmpVgpr2 + 1
  rotation_b    = tmpVgpr2 + 2
  rowOffset_b   = tmpVgpr2 + 3
  colOffset_b   = tmpVgpr2 + 4
  module.add(VAndB32(dst=vgpr(lane16Group_b), src0=vgpr("Serial"), src1=wavesize - 1,
             comment="B: laneId"))
  module.add(VLShiftRightB32(dst=vgpr(lane16Group_b),
             shiftHex=hex(int(math.log2(mi_m_B))), src=vgpr(lane16Group_b),
             comment="B: lane16Group"))
  module.add(VAndB32(dst=vgpr(lane16_b), src0=vgpr("Serial"), src1=mi_m_B - 1,
             comment="B: lane16"))
  module.add(VLShiftRightB32(dst=vgpr(rotation_b),
             shiftHex=hex(int(math.log2(numRowsPerLDSBanks))), src=vgpr(lane16_b),
             comment="B: lds_row_id"))
  module.add(VLShiftRightB32(dst=vgpr(rotation_b), shiftHex=hex(1), src=vgpr(rotation_b),
             comment="B: (lds_row_id //2)"))
  module.add(VLShiftLeftB32(dst=vgpr(rotation_b), shiftHex=hex(1), src=vgpr(rotation_b),
             comment="B: rotation=(lds_row_id //2)*2"))
  module.add(VAddU32(dst=vgpr(colOffset_b), src0=vgpr(rotation_b), src1=vgpr(lane16Group_b),
             comment="B: colOffset"))
  module.add(VAndB32(dst=vgpr(colOffset_b), src0=vgpr(colOffset_b), src1=hex(blockSize_B - 1),
             comment="B: colOffset %% blockSize"))
  _setExecMask(module, writer, 0x33333333, 0x33333333)
  module.add(VPermlane16SwapB32(dst=vgpr(colOffset_b), src=vgpr(colOffset_b),
             comment="B: swizzle"))
  _setExecMask(module, writer, -1, -1)
  module.add(VLShiftLeftB32(dst=vgpr(rowOffset_b),
             shiftHex=hex(int(math.log2(subIterKBytes_B))), src=vgpr(lane16_b),
             comment="B: row = lane16 * %d" % subIterKBytes_B))
  _computeLROffset(module, tileInfoB, colOffset_b, rowOffset_b, writer.states.subtileLdsSwizzle)
  writer.vgprPool.checkIn(tmpVgpr2)
  # Wave partition for B
  _applyWavePartitionLROffset(module, writer, kernel, tileInfoB)
  # B LDS start offset
  for vgprId in range(len(tileInfoB.sharedVgprLROffset)):
    module.add(VAddU32(dst=vgpr(tileInfoB.sharedVgprLROffset[vgprId]),
               src0=writer.ldsStartOffsetB,
               src1=vgpr(tileInfoB.sharedVgprLROffset[vgprId]),
               comment="B matrix offset in LDS"))
  return module


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
  module.add(VLShiftLeftB32(dst=vgpr(rowOffset), shiftHex=hex(ldsKBytes.bit_length()-1), src=vgpr(lane16), comment="offsetRow = %d*lane16" % ldsKBytes))
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
    offsetStride = subtileShapeM * instM * depthUBytes
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



def emitSingleDsReadTLU1(tileInfo, sId0, sId1, subIterK, dstTile):
   """ds_read_b64_tr_b4 pair for ONE MMA-M tile (TLU=1, col-major FP4).
   Caller iterates `sId0 = tileId` over MMA-M tiles 0..localMMATileGrid[0]-1
   (stride lrGran.mn=1), and `sId1` over K-sub-tiles 0..localSubtileGrid[1]-1.
   GR (DTL) lays the per-mGroup LDS region as:
     LDS[ subId0*subtileSize + sId1*localSubtileGrid[0]*subtileSize + Q*16 + ... ]
   where Q is the lane id, and each lane's 16-byte cell holds two MMA-M tiles
   interleaved by 8 bytes (bytes 0..7 = tile_m=0, bytes 8..15 = tile_m=1).
   So a single `sId0` (MMA-M id) decomposes into:
       subId0        = sId0 // lrSubtileShape[0]        # which slot in M
       tile_m_within = sId0 %  lrSubtileShape[0]        # which 16-row block in the slot
   `sharedVgprLROffset` was prepared with index r = tile_m*reads_per_tile + k_half
   (see _lraTileAssignment_tlu1_a), so the right VGPR is index
       (tile_m_within * reads_per_tile) + readInTile.
   ds_offset selects ONLY the slot:
       offset = subId0*subtileSize + sId1*localSubtileGrid[0]*subtileSize
   which matches GR `_emitColMajorBufferLoad`'s subtileLDSOffset exactly.
   """
   module = Module()
   REGS_PER_READ = 2
   lrSubShape0     = int(tileInfo.lrSubtileShape[0])
   lrSubShape1     = int(tileInfo.lrSubtileShape[1])
   tilesPerSubtile = lrSubShape0 * lrSubShape1
   reads_per_tile  = tileInfo.numLRPerSubtile // tilesPerSubtile
   subId0          = sId0 // lrSubShape0
   tile_m_within   = sId0 %  lrSubShape0
   base_r          = tile_m_within * reads_per_tile
   offsetStride    = int(tileInfo.subtileSize)
   offset          = subId0 * offsetStride \
                     + sId1 * int(tileInfo.localSubtileGrid[0]) * offsetStride
   dstVgpr = dstTile.regList.indices[0]
   for readInTile in range(reads_per_tile):
     addrVgpr = tileInfo.sharedVgprLROffset[base_r + readInTile]
     module.add(DSLoadB64TrB4(
         dst=vgpr(dstVgpr + readInTile * REGS_PER_READ, REGS_PER_READ),
         src=vgpr(addrVgpr),
         ds=DSModifiers(offset=offset),
         comment="Subtile%s[%u,%u] subIterK=%u k_half=%u (subId0=%u, tile_m=%u)"
                 % (tileInfo.tc, sId0, sId1, subIterK,
                    readInTile, subId0, tile_m_within)))
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
