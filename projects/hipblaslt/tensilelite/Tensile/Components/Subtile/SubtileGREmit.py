# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

################################################################################
# GR (global read) emit and alloc dispatch.
#
# singledispatch over GR tag sentinels (GRTag_1x2, GRTag_TLU1, etc.).
# ABGRTile calls these via self.config.tag as the dispatch key.
#
# Structure:
#   1. Dispatch bases       — @singledispatch declarations
#   2. Implementations      — logic functions decorated with @register
#
# To add a new GR shape: define a new tag in SubtileGeometry.py, add geometry
# instances with that tag, and register a new implementation below.
# LR emit lives in a separate file (SubtileLREmit.py).
################################################################################

import math
from functools import singledispatch

from rocisa.code import Module
from rocisa.container import DPPModifiers, EXEC, MUBUFModifiers, VCC, vgpr, sgpr, mgpr
from rocisa.enum import RegisterType
from rocisa.instruction import (
    BufferLoadB128,
    SAddCU32, SAddU32, SAddU64, SAndB32, SMovB32, SMovB64, SMulI32, SNop, SOrB32, SXorB32,
    SCBranchSCC1, SCmpEQU32, SEndpgm,
    SLShiftLeftB64, SLShiftRightB32,
    VAddU32, VAndB32, VCmpXEqU32,
    VLShiftLeftB32, VLShiftRightB32, VMovB32,
    TensorLoadToLds,
    VMulLOU32, VReadfirstlaneB32, VSubU32, VXorB32,
)

from .SubtileGeometry import (
    RegList,
    GRTag_1x1, GRTag_1x2, GRTag_2x2, GRTag_TLU1,
)
from .SubtileScaleEmit import emitScaleGRLDSSwap

from math import ceil, log, log2, prod
from rocisa.code import Label
from ...Common import INDEX_CHARS
from ...Common.DataType import DataType


################################################################################
# 1. Dispatch bases
################################################################################

@singledispatch
def _emitGlobalReadOffset(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"emitGlobalReadOffset not implemented for {type(tag).__name__}")

@singledispatch
def _emitGlobalRead(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"emitGlobalRead not implemented for {type(tag).__name__}")

@singledispatch
def _emitLocalWrite(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"emitLocalWrite not implemented for {type(tag).__name__}")

@singledispatch
def _allocGROffsetRegisters(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"allocGROffsetRegisters not implemented for {type(tag).__name__}")

@singledispatch
def _deallocGROffsetRegisters(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"deallocGROffsetRegisters not implemented for {type(tag).__name__}")

@singledispatch
def _emitDTLInit(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"emitDTLInit not implemented for {type(tag).__name__}")

@singledispatch
def _emitGRLDSBufferSwap(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"emitGRLDSBufferSwap not implemented for {type(tag).__name__}")

@singledispatch
def _emitGRPtrUpdate(tag, tile, ti, writer, kernel):
  raise NotImplementedError(f"emitGRPtrUpdate not implemented for {type(tag).__name__}")

# Stubs for tags not yet implemented.
_stub = lambda tag, tile, ti, writer, kernel: None
_emitGlobalReadOffset.register(GRTag_TLU1)(_stub)
_allocGROffsetRegisters.register(GRTag_TLU1)(_stub)
_deallocGROffsetRegisters.register(GRTag_TLU1)(_stub)
_emitGlobalRead.register(GRTag_TLU1)(_stub)
_emitDTLInit.register(GRTag_TLU1)(_stub)
_emitGRLDSBufferSwap.register(GRTag_TLU1)(_stub)
_emitGRPtrUpdate.register(GRTag_TLU1)(_stub)
for _tag in (GRTag_1x1, GRTag_1x2, GRTag_2x2, GRTag_TLU1):
  _emitLocalWrite.register(_tag)(_stub)


################################################################################
# 2. Implementations — TLU=0 (shared by GRTag_1x2 and GRTag_2x2)
################################################################################

@_emitGlobalReadOffset.register(GRTag_1x1)
@_emitGlobalReadOffset.register(GRTag_1x2)
@_emitGlobalReadOffset.register(GRTag_2x2)
def _emitGROffset_TLU0(tag, tile, ti, writer, kernel):
  return Module(f"GR Offset TLU0 ({ti.tc})")  # STUB — legacy path in graTileAssignment
  """GR offset for row-major (TLU=0) geometry with swizzling and rotation.

  Ported from legacy graTileAssignment. Operates on a single tensor component.

  1. Compute waveId, laneId, colId, rowId from Serial (v0).
  2. Swizzle colId via DPP quad_perm to avoid LDS bank conflicts.
  3. Intra-wave rotation: shift colId based on LDS row parity.
  4. Inter-wave rotation: additional shift from waveId (when waves_coop > 1).
  5. Unified wave partition: localRow + partitionRow from waveId.
  6. Compute byte offsets for each GR load into sharedVgprGROffset[].
  7. Compute subtile perpendicular soffsets.
  """
  module = Module(f"GR Offset TLU0 ({ti.tc})")
  tc = ti.tc
  loadWidth = ti.loadWidthGR
  subIterKBytes = ti.subIterKBytes
  blockSize = subIterKBytes // loadWidth
  wavesize = kernel["WavefrontSize"]
  bpe = ti.bpe
  bpeBits = int(8 * bpe)
  strideRef = "StrideA0I" if tc == 'A' else "StrideB1J"
  ldsRowBankSize = writer.states.archCaps["LDSBankCount"] * writer.states.archCaps["LDSBankWidth"]

  wg_m       = ti.waveGroupSize
  numWaves   = ti.numWaves
  waves_coop = numWaves // wg_m
  numRowsPerWave    = wavesize // blockSize
  numRowsPerLDSBanks = ldsRowBankSize // subIterKBytes

  tmpVgpr = writer.vgprPool.checkOut(4, tag="_emitGROffset_TLU0_tmpVgpr")
  colId     = tmpVgpr
  rowId     = tmpVgpr + 1
  waveId    = tmpVgpr + 2
  localRow  = tmpVgpr + 3
  tmpSgpr   = writer.sgprPool.checkOut(1, tag="_emitGROffset_TLU0_tmpSgpr", preventOverflow=False)

  # --- 1. waveId, laneId, colId, rowId ---
  module.add(VLShiftRightB32(dst=vgpr(waveId), shiftHex=hex(wavesize.bit_length()-1),
             src=vgpr("Serial"), comment=f"{tc}: waveId"))
  module.add(VAndB32(dst=vgpr(localRow), src0=vgpr("Serial"), src1=wavesize-1,
             comment=f"{tc}: laneId"))
  module.add(VAndB32(dst=vgpr(colId), src0=vgpr("Serial"), src1=blockSize-1,
             comment=f"{tc}: colId for {loadWidth}B load"))
  module.add(VLShiftRightB32(dst=vgpr(rowId), shiftHex=hex(blockSize.bit_length()-1),
             src=vgpr(localRow), comment=f"{tc}: rowId within wave"))

  # --- 2. Swizzle: DPP quad_perm swap colId pairs on even LDS rows ---
  tmpSwz = writer.vgprPool.checkOut(2, tag="_emitGROffset_TLU0_tmpSwz")
  ldsRowId     = tmpSwz
  swzTmp       = tmpSwz + 1

  module.addComment0(f"{tc}: Swizzling")
  module.add(VLShiftRightB32(dst=vgpr(ldsRowId), shiftHex=hex(blockSize.bit_length()-1),
             src=vgpr(localRow), comment=f"{tc}: row id within wave"))
  module.add(VLShiftRightB32(dst=vgpr(ldsRowId), shiftHex=hex(numRowsPerLDSBanks.bit_length()-1),
             src=vgpr(ldsRowId), comment=f"{tc}: lds row id"))
  module.add(VAndB32(dst=vgpr(swzTmp), src0=vgpr(ldsRowId), src1=hex(1),
             comment=f"{tc}: lds row id %% 2"))
  module.add(VCmpXEqU32(dst=VCC(), src0=0, src1=vgpr(swzTmp),
             comment=f"{tc}: lds row id %% 2 == 0?"))
  module.add(VMovB32(dst=vgpr(colId), src=vgpr(colId), dpp=DPPModifiers(quad_perm=[1,0,3,2]),
             comment=f"{tc}: swap colId pairs"))
  module.add(SMovB64(dst=EXEC(), src=-1))

  # --- 3. Intra-wave rotation: blockSize - (ldsRowId // 2) * 2 ---
  module.addComment0(f"{tc}: Intra-wave rotation")
  module.add(VLShiftRightB32(dst=vgpr(swzTmp), shiftHex=hex(1), src=vgpr(ldsRowId)))
  module.add(VLShiftLeftB32(dst=vgpr(swzTmp), shiftHex=hex(1), src=vgpr(swzTmp),
             comment=f"{tc}: (ldsRowId // 2) * 2"))
  module.add(VSubU32(dst=vgpr(swzTmp), src0=hex(blockSize), src1=vgpr(swzTmp),
             comment=f"{tc}: rotation = blockSize - (ldsRowId//2)*2"))

  # --- 4. Inter-wave rotation (when waves cooperate on a subtile) ---
  if waves_coop > 1:
    waveRotation = writer.vgprPool.checkOut(1, tag="_emitGROffset_TLU0_waveRotation")
    module.addComment0(f"{tc}: Inter-wave rotation")
    module.add(VAndB32(dst=vgpr(waveRotation), src0=vgpr(waveId), src1=hex(1)))
    module.add(VLShiftLeftB32(dst=vgpr(waveRotation),
               shiftHex=hex((2*numRowsPerLDSBanks).bit_length() - 1), src=vgpr(waveRotation)))
    module.add(VSubU32(dst=vgpr(waveRotation), src0=vgpr(swzTmp), src1=vgpr(waveRotation)))
    module.add(VAddU32(dst=vgpr(colId), src0=vgpr(waveRotation), src1=vgpr(colId)))
    writer.vgprPool.checkIn(waveRotation)
  else:
    module.add(VAddU32(dst=vgpr(colId), src0=vgpr(swzTmp), src1=vgpr(colId)))

  module.add(VAndB32(dst=vgpr(colId), src0=vgpr(colId), src1=hex(blockSize-1),
             comment=f"{tc}: (col + rotation) %% blockSize"))
  writer.vgprPool.checkIn(tmpSwz)

  # --- 5. Unified wave partition ---
  rowOffset = writer.vgprPool.checkOut(1, tag="_emitGROffset_TLU0_rowOffset")
  partitionStride = ti.mmaTileShape[0] * int(ti.localSubtileGrid[0])
  waves_coop_shift = max(0, waves_coop.bit_length() - 1) if waves_coop > 0 else 0
  module.add(VAndB32(dst=vgpr(localRow), src0=hex(waves_coop - 1), src1=vgpr(waveId),
             comment=f"{tc}: waveId %% {waves_coop}"))
  module.add(VLShiftRightB32(dst=vgpr(rowOffset), shiftHex=hex(waves_coop_shift),
             src=vgpr(waveId), comment=f"{tc}: waveId // {waves_coop}"))
  module.add(VLShiftLeftB32(dst=vgpr(localRow), shiftHex=hex(numRowsPerWave.bit_length()-1),
             src=vgpr(localRow), comment=f"{tc}: local row * {numRowsPerWave}"))
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=partitionStride,
             comment=f"{tc}: partition stride"))
  module.add(VMulLOU32(dst=vgpr(rowOffset), src0=sgpr(tmpSgpr), src1=vgpr(rowOffset),
             comment=f"{tc}: partition row offset"))
  module.add(VAddU32(dst=vgpr(rowOffset), src0=vgpr(localRow), src1=vgpr(rowOffset),
             comment=f"{tc}: + local row"))
  module.add(VAddU32(dst=vgpr(rowOffset), src0=vgpr(rowId), src1=vgpr(rowOffset),
             comment=f"{tc}: + lane rowId"))

  # --- 6. Compute byte offsets for each GR load ---
  tmpVgpr2 = writer.vgprPool.checkOut(2, tag="_emitGROffset_TLU0_tmpVgpr2")
  colBytes = tmpVgpr2 + 1
  for i in range(ti.numGRPerSubtile):
    useColId = colId
    # For numGRPerSubtile > 1 with single-wave subtiles: rotate colId between loads
    if i > 0 and waves_coop == 1 and ti.numGRPerSubtile > 1:
      rotatedCol = writer.vgprPool.checkOut(1, tag="_emitGROffset_TLU0_rotatedCol")
      colRotation = blockSize // 2
      module.add(VAddU32(dst=vgpr(rotatedCol), src0=colRotation, src1=vgpr(colId),
                 comment=f"{tc}: rotate col for GR {i}"))
      module.add(VAndB32(dst=vgpr(rotatedCol), src0=vgpr(rotatedCol), src1=hex(blockSize-1),
                 comment=f"{tc}: (col + {colRotation}) %% blockSize"))
      useColId = rotatedCol

    module.add(VLShiftLeftB32(dst=vgpr(colBytes), shiftHex=hex(loadWidth.bit_length()-1),
               src=vgpr(useColId), comment=f"{tc}: colId * {loadWidth}"))
    module.add(VMulLOU32(dst=vgpr(tmpVgpr2), src0=sgpr(strideRef), src1=vgpr(rowOffset),
               comment=f"{tc}: rowOffset * stride"))
    module.add(VLShiftLeftB32(dst=vgpr(tmpVgpr2), shiftHex=hex(bpeBits.bit_length()-1),
               src=vgpr(tmpVgpr2), comment=f"{tc}: * bpe"))
    module.add(VLShiftRightB32(dst=vgpr(tmpVgpr2), shiftHex=hex(3), src=vgpr(tmpVgpr2),
               comment=f"{tc}: bits to bytes"))
    module.add(VAddU32(dst=vgpr(tile.sharedVgprGROffset[i]), src0=vgpr(colBytes), src1=vgpr(tmpVgpr2),
               comment=f"{tc}: GR offset {i}"))

    if i > 0 and waves_coop == 1 and ti.numGRPerSubtile > 1:
      writer.vgprPool.checkIn(rotatedCol)

    if i + 1 < ti.numGRPerSubtile:
      advance = ti.subtileShape[0] * ti.mmaTileShape[0] // ti.numGRPerSubtile
      module.add(VAddU32(dst=vgpr(rowOffset), src0=advance, src1=vgpr(rowOffset),
                 comment=f"{tc}: advance row for GR {i+1}"))
  writer.vgprPool.checkIn(tmpVgpr2)

  # --- 7. Subtile perpendicular soffsets ---
  subtileRowElements = ti.subtileShape[0] * ti.mmaTileShape[0]
  s_stride_bpe = int(subtileRowElements * bpe)
  for reg_idx in range(len(ti.localSubtilesRegister)):
    rl = ti.localSubtilesRegister[reg_idx]
    if len(rl) == 0:
      continue
    if rl.is_sgpr:
      module.add(SMulI32(dst=rl.ref(0), src0=hex(s_stride_bpe * reg_idx),
                 src1=sgpr(strideRef), comment=f"{tc}: subtile row {reg_idx} soffset"))
    else:
      stmp = writer.sgprPool.checkOut(1, tag="_emitGROffset_TLU0_stmp")
      for i, reg in enumerate(rl):
        module.add(SMulI32(dst=sgpr(stmp), src0=hex(s_stride_bpe * reg_idx),
                   src1=sgpr(strideRef), comment=f"{tc}: subtile row {reg_idx} soffset"))
        module.add(VAddU32(dst=vgpr(reg), src0=vgpr(tile.sharedVgprGROffset[i]), src1=sgpr(stmp),
                   comment=f"{tc}: bake soffset into vgpr"))
      writer.sgprPool.checkIn(stmp)

  writer.vgprPool.checkIn(rowOffset)
  writer.vgprPool.checkIn(tmpVgpr)
  writer.sgprPool.checkIn(tmpSgpr)
  return module


@_allocGROffsetRegisters.register(GRTag_1x1)
@_allocGROffsetRegisters.register(GRTag_1x2)
@_allocGROffsetRegisters.register(GRTag_2x2)
def _allocGROffsetRegs_TLU0(tag, tile, ti, writer, kernel):
  """Allocate GR offset registers for TLU=0 shapes.

  Two register groups are allocated:

  1. sharedVgprGROffset[]: one VGPR per GR load within a subtile.
     These hold per-lane byte offsets for buffer_load (colId * loadWidth +
     rowOffset * stride * bpe).  Shared across all subtile rows — only the
     soffset changes between rows.

  2. localSubtilesRegister[]: one RegList per perpendicular subtile row.
     Each entry holds the constant M-direction offset (soffset) that shifts
     the shared VGPR offset to the correct subtile row.

     Row 0 needs no offset (soffset=0), so its RegList is left empty.
     Row 1+ gets either:
       - 1 SGPR (preferred): used as the soffset field in buffer_load.
         The shared VGPR offset is reused as-is across rows.
       - numGRPerSubtile VGPRs (fallback when SGPRs exhausted): each VGPR
         has the shared offset + row offset baked in, replacing soffset.
  """
  # TDM handles global reads without per-lane VGPRs
  hasTDM = kernel.get("enableTDMA", False) and kernel.get("enableTDMB", False)
  if hasTDM:
    tile.sharedVgprGROffset = []
    ti.localSubtilesRegister = []
    return

  # Per-lane byte offsets: one VGPR per GR load within a subtile
  tile.sharedVgprGROffset = []
  for i in range(ti.numGRPerSubtile):
    tile.sharedVgprGROffset.append(writer.vgprPool.checkOut(1, tag="_allocGROffsetRegs_TLU0_sharedVgprGROffset"))

  # Per-subtile-row soffset registers.
  # perpDimSize = how many GR subtile shapes tile the perpendicular (M) dimension
  # per wave. Each position needs its own soffset register.
  ti.localSubtilesRegister = []
  # perpDimSize: distinct soffset positions in M = how many localSubtile rows
  # need their own soffset register.  localGRGranularity[0] tells how many
  # consecutive localSubtile rows one GR load covers (>1 only for bc==1 with
  # wave-cooperative expansion, i.e. loadRatioGR > 1).
  localSubtileRowCount = int(ti.localSubtileGrid[0])
  gran = tile.localGRGranularity(ti.numWaves)
  perpDimSize = math.ceil(localSubtileRowCount / gran[0])
  tmpSgprBuffer = 3
  sgprLimit = writer.states.regCaps["MaxSgpr"] - tmpSgprBuffer

  for reg_idx in range(perpDimSize):
    useSgpr = writer.sgprPool.size() < sgprLimit
    if useSgpr:
      rl = RegList(writer.sgprPool, RegisterType.Sgpr)
    else:
      rl = RegList(writer.vgprPool, RegisterType.Vgpr)
    ti.localSubtilesRegister.append(rl)
    # Row 0 is the base position — no soffset needed, RegList stays empty.
    if reg_idx == 0:
      continue
    if useSgpr:
      # SGPR path: 1 register for soffset, shared VGPR offset reused.
      rl.alloc(preventOverflow=False)
    else:
      # VGPR fallback: one VGPR per GR load, each with soffset baked in.
      for i in range(ti.numGRPerSubtile):
        rl.alloc(preventOverflow=False)


@_deallocGROffsetRegisters.register(GRTag_1x1)
@_deallocGROffsetRegisters.register(GRTag_1x2)
@_deallocGROffsetRegisters.register(GRTag_2x2)
def _deallocGROffsetRegs_TLU0(tag, tile, ti, writer, kernel):
  """Deallocate GR offset registers for TLU=0 shapes."""
  if isinstance(tile.sharedVgprGROffset, list):
    for voff in tile.sharedVgprGROffset:
      writer.vgprPool.checkIn(voff)
    tile.sharedVgprGROffset = []
  if isinstance(ti.localSubtilesRegister, list):
    for rl in ti.localSubtilesRegister:
      rl.dealloc()
    ti.localSubtilesRegister = []


# --- GR load emit (TLU=0) ---------------------------------------------------

@_emitGlobalRead.register(GRTag_1x1)
@_emitGlobalRead.register(GRTag_1x2)
@_emitGlobalRead.register(GRTag_2x2)
def _emitGR_TLU0(tag, tile, ti, writer, kernel):
  """Emit buffer_load_dwordx4 (DTL) for all subtiles in the local grid.

  For each subtile (sId0, sId1):
    - Computes LDS write address (m0) from LocalWriteBaseAddr + subtile offset.
    - Emits buffer_load_b128 with lds=True (direct-to-LDS).
    - Uses soffset (SGPR path) or baked VGPR offset for the subtile row.

  When loadRatioGR > 1, multiple subtiles share one GR load; only the first
  subtile in each group emits the load.
  """
  module = Module(f"GR Load TLU0 ({ti.tc})")
  tc = ti.tc
  isGlc = bool(kernel.get(f"NonTemporal{tc}", 0) & 0x1)
  isSlc = bool(kernel.get(f"NonTemporal{tc}", 0) & 0x2)
  isNT  = bool(kernel.get(f"NonTemporal{tc}", 0) & 0x4)

  perpDimSize = len(ti.localSubtilesRegister)

  # TODO: Remove legacy TileInfo dependency after full migration.
  # Currently uses legacy's grid/sizes because subtileShape expansion in for_kernel
  # changes subtileSize/localSubtileGrid/loadRatioGR, which must match the LDS
  # layout computed from legacy values.
  legacyTi = getattr(writer.states, tc.lower()).tileInfo
  localGrid0 = int(legacyTi.localSubtileGrid[0])
  localGrid1 = int(legacyTi.localSubtileGrid[1])
  legacyLoadRatio = legacyTi.loadRatioGR
  legacySubtileSize = int(legacyTi.subtileSize)

  for j in range(localGrid1):
    for i in range(localGrid0):
      slowId = i
      if legacyLoadRatio == 2.0:
        slowId = int(i // legacyLoadRatio)
      reg_idx = slowId

      # Skip duplicate loads when loadRatio > 1
      if legacyLoadRatio > 1:
        linearId = j * localGrid0 + i
        grBaseId = int(linearId // legacyLoadRatio)
        firstInGroup = int(grBaseId * legacyLoadRatio)
        if linearId != firstInGroup:
          continue

      rl = ti.localSubtilesRegister[min(reg_idx, perpDimSize - 1)]
      offsetK = j * int(ti.mmaTileShape[1] * ti.subtileShape[1] * ti.bpe)

      module.addComment0(f"GR load {tc} subtile [{i},{j}]")

      subtileOffset = int(math.ceil(legacyLoadRatio * legacySubtileSize)) if legacyLoadRatio else legacySubtileSize
      WriteBaseAddr = f"LocalWriteBaseAddr{tc}"

      for gr_idx in range(legacyTi.numGRPerSubtile):
        m0Offset = gr_idx * subtileOffset + (i + j * int(legacyTi.globalSubtileGrid[0])) * legacySubtileSize
        module.add(SAddU32(dst=mgpr(0), src0=sgpr(WriteBaseAddr), src1=(m0Offset - offsetK)))
        mubuf = MUBUFModifiers(offen=True, offset12=offsetK, glc=isGlc, slc=isSlc, nt=isNT, lds=True)

        use_sgpr = rl.is_sgpr if len(rl) > 0 else True
        soffset = rl.ref(0) if len(rl) > 0 and use_sgpr else 0
        voff = tile.sharedVgprGROffset[gr_idx] if use_sgpr or len(rl) == 0 else rl.indices[gr_idx]
        module.add(BufferLoadB128(dst=None, vaddr=vgpr(voff), saddr=sgpr(f"Srd{tc}", 4),
                   soffset=soffset, mubuf=mubuf, comment=f"GR{gr_idx} [{i},{j}]"))

  return module


# --- DTL init (TLU=0) -------------------------------------------------------

@_emitDTLInit.register(GRTag_1x1)
@_emitDTLInit.register(GRTag_1x2)
@_emitDTLInit.register(GRTag_2x2)
def _emitDTLInit_TLU0(tag, tile, ti, writer, kernel):
  return Module(f"DTL Init ({ti.tc})")  # STUB — legacy path in globalReadDTLInitCommonSgpr
  """Compute LocalWriteBaseAddr and Swap SGPR for one tensor component.

  The DTL (direct-to-LDS) buffer_load writes data at m0 = LocalWriteBaseAddr + subtile offset.
  LocalWriteBaseAddr is the wave's base LDS position, derived from the wave partition.
  Swap holds the XOR mask to toggle between double-buffer halves.

  For double-buffering: LocalWriteBaseAddr XOR Swap flips to the other buffer.

  Requires sgprs: LocalWriteBaseAddr{tc}, Swap{tc} (must be pre-allocated by caller).
  """
  module = Module(f"DTL Init ({ti.tc})")
  tc = ti.tc
  wavesize = kernel["WavefrontSize"]
  wg_m     = ti.waveGroupSize
  numWaves = ti.numWaves
  waves_coop = numWaves // wg_m

  vgprWaveId = writer.vgprPool.checkOut(1, tag="_emitDTLInit_TLU0_vgprWaveId")
  rowOffset  = writer.vgprPool.checkOut(1, tag="_emitDTLInit_TLU0_rowOffset")

  module.add(VLShiftRightB32(dst=vgpr(vgprWaveId), shiftHex=hex(wavesize.bit_length()-1),
             src=vgpr("Serial"), comment=f"{tc}: waveId"))

  # Wave partition: same unified formula as GR offset step 5
  numRowsPerWave  = wavesize // (ti.subIterKBytes // ti.loadWidthGR)
  partitionStride = ti.mmaTileShape[0] * int(ti.localSubtileGrid[0])
  waves_coop_shift = max(0, waves_coop.bit_length() - 1) if waves_coop > 0 else 0

  module.add(VLShiftRightB32(dst=vgpr(rowOffset), shiftHex=hex(waves_coop_shift),
             src=vgpr(vgprWaveId), comment=f"{tc}: partitionRow = waveId // {waves_coop}"))
  tmpSgpr = writer.sgprPool.checkOut(1, tag="_emitDTLInit_TLU0_tmpSgpr", preventOverflow=False)
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=partitionStride))
  module.add(VMulLOU32(dst=vgpr(rowOffset), src0=sgpr(tmpSgpr), src1=vgpr(rowOffset),
             comment=f"{tc}: partition row offset"))
  writer.sgprPool.checkIn(tmpSgpr)

  # Scale by subIterKBytes to get LDS byte offset
  module.add(VLShiftLeftB32(dst=vgpr(rowOffset),
             shiftHex=hex(ti.subIterKBytes.bit_length()-1), src=vgpr(rowOffset),
             comment=f"{tc}: * subIterKBytes"))

  # Move to SGPR via readfirstlane (uniform across wave)
  module.add(SNop(waitState=0, comment="wait for VGPR"))
  WriteBaseAddr = f"LocalWriteBaseAddr{tc}"
  Swap = f"Swap{tc}"
  module.add(VReadfirstlaneB32(dst=sgpr(WriteBaseAddr), src=vgpr(rowOffset),
             comment=f"{tc}: base LDS offset"))

  # Add global LDS start offset for B (B data follows A in LDS)
  ldsStartOffset = getattr(writer, f'ldsStartOffset{tc}', 0)
  if ldsStartOffset:
    module.add(SAddU32(dst=sgpr(WriteBaseAddr), src0=sgpr(WriteBaseAddr),
               src1=hex(ldsStartOffset), comment=f"{tc}: + ldsStartOffset"))

  # Swap mask: XOR(base, base + ldsTotalSize) toggles between buffer halves
  module.add(SAddU32(dst=sgpr(Swap), src0=sgpr(WriteBaseAddr), src1=writer.ldsTotalSize))
  module.add(SXorB32(dst=sgpr(Swap), src0=sgpr(WriteBaseAddr), src1=sgpr(Swap)))

  writer.vgprPool.checkIn(vgprWaveId)
  writer.vgprPool.checkIn(rowOffset)
  return module


# --- GR LDS buffer swap (TLU=0) ---------------------------------------------

@_emitGRLDSBufferSwap.register(GRTag_1x1)
@_emitGRLDSBufferSwap.register(GRTag_1x2)
@_emitGRLDSBufferSwap.register(GRTag_2x2)
def _emitGRLDSSwap_TLU0(tag, tile, ti, writer, kernel):
  """Toggle GR DTL write target between double-buffer halves.

  XOR LocalWriteBaseAddr with Swap to flip to the other LDS buffer.
  """
  module = Module()
  tc = ti.tc
  module.addComment0("Emit code to swap %s GR m0 offsets"%tc)
  module.add(SXorB32(dst=sgpr(f"LocalWriteBaseAddr{tc}"),
             src0=sgpr(f"LocalWriteBaseAddr{tc}"), src1=sgpr(f"Swap{tc}"),
             comment=""))
  return module


# --- GR pointer update (TLU=0) ----------------------------------------------

@_emitGRPtrUpdate.register(GRTag_1x1)
@_emitGRPtrUpdate.register(GRTag_1x2)
@_emitGRPtrUpdate.register(GRTag_2x2)
def _emitGRPtrUpdate_TLU0(tag, tile, ti, writer, kernel):
  """Advance SRD base pointer by one depthU iteration (depthU * bpe bytes)."""
  tc = ti.tc
  # TDM path: advance Address{tc} and sync the TDM descriptor instead of SRD.
  if kernel.get("enableTDM%s" % tc, False):
    module = Module(f"TDM GR Ptr Update ({tc})")
    inc = int(ti.depthUBytes)
    module.addComment0("TDM addr update: %s += %u" % (tc, inc))
    module.add(SAddU64(dst=sgpr("Address%s" % tc, 2), src0=sgpr("Address%s" % tc, 2), src1=inc))
    group0 = "tdm%sGroup0" % tc
    module.add(SMovB64(dst=sgpr("%s+2" % group0, 2), src=sgpr("Address%s" % tc, 2), comment="sync descriptor global addr"))
    module.add(SOrB32(dst=sgpr("%s+3" % group0), src0=sgpr("%s+3" % group0), src1=hex(2 << 30), comment="restore type field"))
    return module

  module = Module(f"GR Ptr Update ({tc})")
  inc = int(ti.depthUBytes)
  module.add(SAddU32(dst=sgpr(f"Srd{tc}"), src0=sgpr(f"Srd{tc}"), src1=inc,
             comment=f"{tc}: advance SRD by {inc} bytes"))
  module.add(SAddCU32(dst=sgpr(f"Srd{tc}+1"), src0=sgpr(f"Srd{tc}+1"), src1=0,
             comment=f"{tc}: carry"))
  return module


################################################################################
# Legacy GR emit functions (moved from SubtileBasedKernel.py)
################################################################################

##################################################
# Subroutine to generate GR offset calculation code
#
def graInitPointer(writer, kernel):
  module = Module()
  module.addComment0("REMOVE WHEN IMPLEMNTED: Placeholder for GR base pointer init")
  for i in range(8):
    module.addComment("")

  return module


##################################################
# Compute GR offset for a single matrix (A or B)
#
def _grComputeOffset(module, writer, tileInfo, colId, rowId, output):
  tc = tileInfo.tc
  bpeBits = int(8*tileInfo.bpe)

  tmpVgpr = writer.vgprPool.checkOut(2, tag="_grComputeOffset_tmpVgpr")
  colBytes = tmpVgpr + 1
  loadWidth = tileInfo.loadWidthGR

  module.add(VLShiftLeftB32(dst=vgpr(colBytes), shiftHex=hex(loadWidth.bit_length()-1), src=vgpr(colId), comment="scale col_id by load_width"))
  MT0 = tileInfo.globalMMATileGrid[0] * tileInfo.mmaTileShape[0]
  subtileSize = tileInfo.subtileShape[0]*tileInfo.mmaTileShape[0]
  strideRef = "StrideA0I" if tc == 'A' else "StrideB1J"
  module.add(VMulLOU32(dst=vgpr(tmpVgpr), src0=sgpr(strideRef), src1=vgpr(rowId), comment="%s: rowId * stride"%tc))
  module.add(VLShiftLeftB32(dst=vgpr(tmpVgpr), shiftHex=hex(bpeBits.bit_length()-1), src=vgpr(tmpVgpr), comment="%s: rowId*stride*bpe"%tc))
  module.add(VLShiftRightB32(dst=vgpr(tmpVgpr), shiftHex=hex(3), src=vgpr(tmpVgpr), comment="to bytes"))
  module.add(VAddU32(dst=vgpr(output), src0=vgpr(colBytes), src1=vgpr(tmpVgpr), comment="%s: GR row_offset"%tc))
  writer.vgprPool.checkIn(tmpVgpr)

##################################################
# Compute subtile perpendicular offsets for a single matrix
#
# TODO: need to generalize this to support TLU=1
def _grComputeSubtileOffsets(writer, module, tileInfo):
  tc = tileInfo.tc
  strideRef = "StrideA0I" if tc == 'A' else "StrideB1J"
  subtile_size = tileInfo.subtileShape[0]*tileInfo.mmaTileShape[0]
  # rowOffset between 2 subtiles offset, ie how many consecutive subtile covered by a single subtileOffset.
  # rowOffset = numGRPerSubtile * (local load ratio * subtile size)
  rowOffset = math.ceil(tileInfo.numGRPerSubtile*tileInfo.loadRatioGR*subtile_size)
  s_stride = int(rowOffset * tileInfo.bpe)

  for regId in range(len(tileInfo.localSubtilesRegister)):
    rl = tileInfo.localSubtilesRegister[regId]
    for i, reg in enumerate(rl):
      if rl.is_sgpr:
        module.add(SMulI32(dst=sgpr(reg), src0=hex(s_stride * regId), src1=sgpr(strideRef), comment="%s: %u rows offset, stride %u, %u"%(tc, rowOffset, s_stride, regId)))
      else:
        stmp = writer.sgprPool.checkOut(1, tag="_grComputeSubtileOffsets_stmp")
        module.add(SMulI32(dst=sgpr(stmp), src0=hex(s_stride * regId), src1=sgpr(strideRef), comment="%s: %u rows offset, stride %u, %u"%(tc, rowOffset, s_stride, regId)))
        module.add(VAddU32(dst=vgpr(reg), src0=vgpr(tileInfo.sharedVgprGROffset[i]), src1=sgpr(stmp)))
        writer.sgprPool.checkIn(stmp)

# Compute wave partition offset for a single tile (A or B)
#
def _grComputeRowPartition(module, kernel, writer, tileInfo, waveId, rowOffset):
  subIterKBytes = tileInfo.subIterKBytes
  wavesize = kernel["WavefrontSize"]
  loadWidth = tileInfo.loadWidthGR
  numRowsPerWave = wavesize // (subIterKBytes // loadWidth)
  tc = tileInfo.tc
  tmpVgpr = writer.vgprPool.checkOut(2, tag="_grComputeRowPartition_tmpVgpr")
  tmpSgpr = writer.sgprPool.checkOut(1, tag="_grComputeRowPartition_tmpSgpr", preventOverflow=False)
  localRow = tmpVgpr
  partitionRow = tmpVgpr+1
  partitionOffset = tileInfo.mmaTileShape[0]*tileInfo.localSubtileGrid[0]
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=partitionOffset, comment="%s: row offset"%tc))

  if tileInfo.loadRatioGR == 1.0:
    module.add(VAndB32(dst=vgpr(localRow), src0=hex(1), src1=vgpr(waveId), comment="%s: waveId %% 2"%tc))
    module.add(VLShiftRightB32(dst=vgpr(partitionRow), shiftHex=hex(1), src=vgpr(waveId), comment="%s: waveId / 2"%tc))
  elif tileInfo.loadRatioGR == 0.5:
    module.add(VMovB32(dst=vgpr(localRow), src=0, comment="%s"%tc))
    module.add(VMovB32(dst=vgpr(partitionRow), src=vgpr(waveId), comment="%s"%tc))
  elif tileInfo.loadRatioGR == 2.0:
    module.add(VMovB32(dst=vgpr(localRow), src=vgpr(waveId), comment="%s"%tc))
    module.add(VMovB32(dst=vgpr(partitionRow), src=0, comment="%s"%tc))
  else:
    raise NotImplementedError("Unsupported loadRatioGR for wave partition: %s"%str(tileInfo.loadRatioGR))

  module.add(VLShiftLeftB32(dst=vgpr(localRow), shiftHex=hex(numRowsPerWave.bit_length()-1), src=vgpr(localRow), comment="%s: local row offset"%tc))
  module.add(VMulLOU32(dst=vgpr(partitionRow), src0=sgpr(tmpSgpr), src1=vgpr(partitionRow), comment="%s: wave row offset"%tc))
  module.add(VAddU32(dst=vgpr(rowOffset), src0=vgpr(localRow), src1=vgpr(partitionRow), comment="%s: row offset"%tc))


  writer.vgprPool.checkIn(tmpVgpr)
  writer.sgprPool.checkIn(tmpSgpr)

##################################################
# Compute GR offsets for all subtiles of a single matrix (A or B)
#
def _grComputeAllOffsets(module, writer, tileInfo, colId, rowId, rowOffset):
  module.add(VAddU32(dst=vgpr(rowOffset), src0=vgpr(rowId), src1=vgpr(rowOffset), comment="%s: row offset"%tileInfo.tc))
  _grComputeOffset(module, writer, tileInfo, colId, rowOffset, tileInfo.sharedVgprGROffset[0])
  for i in range(1, len(tileInfo.sharedVgprGROffset)):
    subtileSize = tileInfo.subtileShape[0] * tileInfo.mmaTileShape[0]
    offset = math.ceil(subtileSize * tileInfo.loadRatioGR)
    module.add(VAddU32(dst=vgpr(rowOffset), src0=offset, src1=vgpr(rowOffset), comment="%s: advance row for GR offset %u"%(tileInfo.tc, i)))

    # Apply Rotation on entire wave. Only applies to 4x case as a subtile is loaded by a single wave in 2 steps. (waveId rotation not applied)
    rotatedcolId = writer.vgprPool.checkOut(1, tag="_grComputeAllOffsets_rotatedcolId")
    loadWidth = tileInfo.loadWidthGR
    if tileInfo.loadRatioGR == 0.5:
      blockSize = tileInfo.subIterKBytes // loadWidth
      colRotation = blockSize // 2
      module.add(VAddU32(dst=vgpr(rotatedcolId), src0=colRotation, src1=vgpr(colId), comment="%s: rotate col for GR offset %u"%(tileInfo.tc, i)))
      module.add(VAndB32(dst=vgpr(rotatedcolId), src0=vgpr(rotatedcolId), src1=hex(blockSize-1), comment="(col + %d) %% block_size"%colRotation))
    else:
      module.add(VMovB32(dst=vgpr(rotatedcolId), src=vgpr(colId), comment=""))

    _grComputeOffset(module, writer, tileInfo, rotatedcolId, rowOffset, tileInfo.sharedVgprGROffset[i])
    writer.vgprPool.checkIn(rotatedcolId)

##################################################
# Apply swizzling and rotation to col IDs for GR offset calculation.
#
# Swizzling reorders column indices to avoid LDS bank conflicts.
# Two levels of rotation are applied to the column IDs:
#   1. Intra-wave rotation: rotates colId based on the LDS row id within
#      a single wave. The rotation offset is: blockSize - (ldsRowId // 2) * 2.
#      This ensures consecutive rows access different LDS banks.
#   2. Inter-wave rotation: an additional per-wave offset derived from waveId
#      shifts the column further so that different waves also avoid bank
#      conflicts with each other. Only applied when loadRatioGR != 0.5
#      (i.e. when multiple waves share the same subtile region).
#
##################################################
# Subroutine to generate GR offset calculation code
#
def graTileAssignment(writer, kernel, useSwizzling=True):
  return _graTileAssignment_legacy(writer, kernel, useSwizzling)

# --- Legacy interleaved A+B GR offset (temporary, matches reference exactly) ---

def _grComputeOffset_legacy(module, writer, tileInfo, colId, rowId, output):
  tc = tileInfo.tc
  bpeBits = int(8*tileInfo.bpe)
  tmpVgpr = writer.vgprPool.checkOut(2, tag="_grComputeOffset_legacy_tmpVgpr")
  colBytes = tmpVgpr + 1
  loadWidth = tileInfo.loadWidthGR
  module.add(VLShiftLeftB32(dst=vgpr(colBytes), shiftHex=hex(loadWidth.bit_length()-1), src=vgpr(colId), comment="scale col_id by load_width"))
  strideRef = "StrideA0I" if tc == 'A' else "StrideB1J"
  module.add(VMulLOU32(dst=vgpr(tmpVgpr), src0=sgpr(strideRef), src1=vgpr(rowId), comment="%s: rowId * stride"%tc))
  module.add(VLShiftLeftB32(dst=vgpr(tmpVgpr), shiftHex=hex(bpeBits.bit_length()-1), src=vgpr(tmpVgpr), comment="%s: rowId*stride*bpe"%tc))
  module.add(VLShiftRightB32(dst=vgpr(tmpVgpr), shiftHex=hex(3), src=vgpr(tmpVgpr), comment="to bytes"))
  module.add(VAddU32(dst=vgpr(output), src0=vgpr(colBytes), src1=vgpr(tmpVgpr), comment="%s: GR row_offset"%tc))
  writer.vgprPool.checkIn(tmpVgpr)

def _grComputeSubtileOffsets_legacy(writer, module, tileInfo):
  tc = tileInfo.tc
  strideRef = "StrideA0I" if tc == 'A' else "StrideB1J"
  subtile_size = tileInfo.subtileShape[0]*tileInfo.mmaTileShape[0]
  rowOffset = math.ceil(tileInfo.numGRPerSubtile*tileInfo.loadRatioGR*subtile_size)
  s_stride = int(rowOffset * tileInfo.bpe)
  for regId in range(len(tileInfo.localSubtilesRegister)):
    rl = tileInfo.localSubtilesRegister[regId]
    for i, reg in enumerate(rl):
      if rl.is_sgpr:
        module.add(SMulI32(dst=sgpr(reg), src0=hex(s_stride * regId), src1=sgpr(strideRef), comment="%s: %u rows offset, stride %u, %u"%(tc, rowOffset, s_stride, regId)))
      else:
        stmp = writer.sgprPool.checkOut(1, tag="_grComputeSubtileOffsets_legacy_stmp")
        module.add(SMulI32(dst=sgpr(stmp), src0=hex(s_stride * regId), src1=sgpr(strideRef), comment="%s: %u rows offset, stride %u, %u"%(tc, rowOffset, s_stride, regId)))
        module.add(VAddU32(dst=vgpr(reg), src0=vgpr(tileInfo.sharedVgprGROffset[i]), src1=sgpr(stmp)))
        writer.sgprPool.checkIn(stmp)

def _grComputeRowPartition_legacy(module, kernel, writer, tileInfo, waveId, rowOffset):
  subIterKBytes = tileInfo.subIterKBytes
  wavesize = kernel["WavefrontSize"]
  loadWidth = tileInfo.loadWidthGR
  numRowsPerWave = wavesize // (subIterKBytes // loadWidth)
  tc = tileInfo.tc
  tmpVgpr = writer.vgprPool.checkOut(2, tag="_grComputeRowPartition_legacy_tmpVgpr")
  tmpSgpr = writer.sgprPool.checkOut(1, tag="_grComputeRowPartition_legacy_tmpSgpr", preventOverflow=False)
  localRow = tmpVgpr
  partitionRow = tmpVgpr+1
  partitionOffset = tileInfo.mmaTileShape[0]*tileInfo.localSubtileGrid[0]
  module.add(SMovB32(dst=sgpr(tmpSgpr), src=partitionOffset, comment="%s: row offset"%tc))
  if tileInfo.loadRatioGR == 1.0:
    module.add(VAndB32(dst=vgpr(localRow), src0=hex(1), src1=vgpr(waveId), comment="%s: waveId %% 2"%tc))
    module.add(VLShiftRightB32(dst=vgpr(partitionRow), shiftHex=hex(1), src=vgpr(waveId), comment="%s: waveId / 2"%tc))
  elif tileInfo.loadRatioGR == 0.5:
    module.add(VMovB32(dst=vgpr(localRow), src=0, comment="%s"%tc))
    module.add(VMovB32(dst=vgpr(partitionRow), src=vgpr(waveId), comment="%s"%tc))
  elif tileInfo.loadRatioGR == 2.0:
    module.add(VMovB32(dst=vgpr(localRow), src=vgpr(waveId), comment="%s"%tc))
    module.add(VMovB32(dst=vgpr(partitionRow), src=0, comment="%s"%tc))
  else:
    raise NotImplementedError("Unsupported loadRatioGR for wave partition: %s"%str(tileInfo.loadRatioGR))
  module.add(VLShiftLeftB32(dst=vgpr(localRow), shiftHex=hex(numRowsPerWave.bit_length()-1), src=vgpr(localRow), comment="%s: local row offset"%tc))
  module.add(VMulLOU32(dst=vgpr(partitionRow), src0=sgpr(tmpSgpr), src1=vgpr(partitionRow), comment="%s: wave row offset"%tc))
  module.add(VAddU32(dst=vgpr(rowOffset), src0=vgpr(localRow), src1=vgpr(partitionRow), comment="%s: row offset"%tc))
  writer.vgprPool.checkIn(tmpVgpr)
  writer.sgprPool.checkIn(tmpSgpr)

def _grComputeAllOffsets_legacy(module, writer, tileInfo, colId, rowId, rowOffset):
  module.add(VAddU32(dst=vgpr(rowOffset), src0=vgpr(rowId), src1=vgpr(rowOffset), comment="%s: row offset"%tileInfo.tc))
  _grComputeOffset_legacy(module, writer, tileInfo, colId, rowOffset, tileInfo.sharedVgprGROffset[0])
  for i in range(1, len(tileInfo.sharedVgprGROffset)):
    subtileSize = tileInfo.subtileShape[0] * tileInfo.mmaTileShape[0]
    offset = math.ceil(subtileSize * tileInfo.loadRatioGR)
    module.add(VAddU32(dst=vgpr(rowOffset), src0=offset, src1=vgpr(rowOffset), comment="%s: advance row for GR offset %u"%(tileInfo.tc, i)))
    rotatedcolId = writer.vgprPool.checkOut(1, tag="_grComputeAllOffsets_legacy_rotatedcolId")
    loadWidth = tileInfo.loadWidthGR
    if tileInfo.loadRatioGR == 0.5:
      if tileInfo.bpe == 1:  # FP8: intra-block K_group +2 rotation, preserving block bit
        tmpBlock = writer.vgprPool.checkOut(1, tag="_grComputeAllOffsets_legacy_tmpBlock")
        module.add(VAndB32(dst=vgpr(tmpBlock), src0=vgpr(colId), src1=hex(4), comment="%s: block_bit = colId & 4"%tileInfo.tc))
        module.add(VAndB32(dst=vgpr(rotatedcolId), src0=vgpr(colId), src1=hex(3), comment="%s: K_group = colId & 3"%tileInfo.tc))
        module.add(VAddU32(dst=vgpr(rotatedcolId), src0=vgpr(rotatedcolId), src1=hex(2), comment="%s: K_group + 2"%tileInfo.tc))
        module.add(VAndB32(dst=vgpr(rotatedcolId), src0=vgpr(rotatedcolId), src1=hex(3), comment="%s: (K_group+2) %% 4"%tileInfo.tc))
        module.add(VAddU32(dst=vgpr(rotatedcolId), src0=vgpr(rotatedcolId), src1=vgpr(tmpBlock), comment="%s: K_group_rot + block_bit"%tileInfo.tc))
        writer.vgprPool.checkIn(tmpBlock)
      else:  # FP4/FP16: half-block rotation
        blockSize = tileInfo.subIterKBytes // loadWidth
        colRotation = blockSize // 2
        module.add(VAddU32(dst=vgpr(rotatedcolId), src0=colRotation, src1=vgpr(colId), comment="%s: rotate col for GR offset %u"%(tileInfo.tc, i)))
        module.add(VAndB32(dst=vgpr(rotatedcolId), src0=vgpr(rotatedcolId), src1=hex(blockSize-1), comment="(col + %d) %% block_size"%colRotation))
    else:
      module.add(VMovB32(dst=vgpr(rotatedcolId), src=vgpr(colId), comment=""))
    _grComputeOffset_legacy(module, writer, tileInfo, rotatedcolId, rowOffset, tileInfo.sharedVgprGROffset[i])
    writer.vgprPool.checkIn(rotatedcolId)

def _grSwizzleColIds_legacy(module, writer, tileInfoA, tileInfoB, blockSize, numRowsPerLDSBanks,
                            laneId, colIdA, colIdB, waveId):
  tmpVgpr = writer.vgprPool.checkOut(3, tag="_grSwizzleColIds_legacy_tmpVgpr")
  ldsRowId = tmpVgpr
  tmp = tmpVgpr + 1
  waveRotation = tmpVgpr + 2
  half = blockSize // 2
  module.addComment0("Swizzling")
  module.add(VLShiftRightB32(dst=vgpr(ldsRowId), shiftHex=hex(blockSize.bit_length()-1), src=vgpr(laneId), comment="row id within wave"))
  module.add(VLShiftRightB32(dst=vgpr(ldsRowId), shiftHex=hex(numRowsPerLDSBanks.bit_length()-1), src=vgpr(ldsRowId), comment="lds row id"))
  module.add(VAndB32(dst=vgpr(tmp), src0=vgpr(ldsRowId), src1=hex(1), comment="swap_bit = ldsRowId & 1"))
  if tileInfoA.bpe == 1:  # FP8: step1=block-swap, step2=wave K_group rotation
    # Step 1: block-swap (XOR blockSize//2 for odd ldsRowId)
    module.add(VLShiftLeftB32(dst=vgpr(tmp), shiftHex=hex(int(math.log2(half))), src=vgpr(tmp),
               comment=f"swap_bit * {half}"))
    module.add(VXorB32(dst=vgpr(colIdA), src0=vgpr(colIdA), src1=vgpr(tmp),
               comment="FP8 step1: block-swap colIdA"))
    module.add(VMovB32(dst=vgpr(colIdB), src=vgpr(colIdA), comment="colIdB = colIdA"))
    # Step 2: K_group rotation = (waveId & 1) * 2 (only for loadRatioGR != 0.5)
    module.add(VAndB32(dst=vgpr(tmp), src0=vgpr(waveId), src1=hex(1), comment="wave_half = waveId & 1"))
    module.add(VLShiftLeftB32(dst=vgpr(tmp), shiftHex=hex(1), src=vgpr(tmp), comment="rotation = wave_half * 2"))
    for tInfo, cId in [(tileInfoA, colIdA), (tileInfoB, colIdB)]:
      if tInfo.loadRatioGR != 0.5:
        module.add(VAndB32(dst=vgpr(waveRotation), src0=vgpr(cId), src1=hex(4), comment="FP8 step2: block_bit = colId & 4"))
        module.add(VAndB32(dst=vgpr(cId), src0=vgpr(cId), src1=hex(3), comment="K_group = colId & 3"))
        module.add(VAddU32(dst=vgpr(cId), src0=vgpr(cId), src1=vgpr(tmp), comment="K_group + rotation"))
        module.add(VAndB32(dst=vgpr(cId), src0=vgpr(cId), src1=hex(3), comment="(K_group+rotation) % 4"))
        module.add(VAddU32(dst=vgpr(cId), src0=vgpr(cId), src1=vgpr(waveRotation), comment="K_group_rot + block_bit"))
  else:  # FP4/FP16: pair-swap (even ldsRowId) + intra/inter-wave rotation
    module.add(VCmpXEqU32(dst=VCC(), src0=0, src1=vgpr(tmp), comment="lds row id % 2 == 0 ?"))
    module.add(VMovB32(dst=vgpr(colIdA), src=vgpr(colIdA), dpp=DPPModifiers(quad_perm=[1,0,3,2]), comment="swap colId pairs for swizzling"))
    module.add(SMovB64(dst=EXEC(), src=-1))
    module.add(VMovB32(dst=vgpr(colIdB), src=vgpr(colIdA), comment=""))
    module.addComment0("Rotation within a single wave")
    module.add(VLShiftRightB32(dst=vgpr(tmp), shiftHex=hex(1), src=vgpr(ldsRowId), comment=""))
    module.add(VLShiftLeftB32(dst=vgpr(tmp), shiftHex=hex(1), src=vgpr(tmp), comment="(ldsRowId //2) * 2"))
    module.add(VSubU32(dst=vgpr(tmp), src0=hex(blockSize), src1=vgpr(tmp), comment="rotation offset : blockSize - (ldsRowId//2)*2"))
    for tInfo, cId in [(tileInfoA, colIdA), (tileInfoB, colIdB)]:
      if tInfo.loadRatioGR != 0.5:
        module.addComment0("Rotation per wave")
        module.add(VAndB32(dst=vgpr(waveRotation), src0=vgpr(waveId), src1=hex(1), comment=""))
        module.add(VLShiftLeftB32(dst=vgpr(waveRotation), shiftHex=hex((2*numRowsPerLDSBanks).bit_length() - 1), src=vgpr(waveRotation), comment=""))
        module.add(VSubU32(dst=vgpr(waveRotation), src0=vgpr(tmp), src1=vgpr(waveRotation), comment=""))
        module.add(VAddU32(dst=vgpr(cId), src0=vgpr(waveRotation), src1=vgpr(cId), comment=""))
      else:
        module.add(VAddU32(dst=vgpr(cId), src0=vgpr(tmp), src1=vgpr(cId), comment=""))
    module.add(VAndB32(dst=vgpr(colIdA), src0=vgpr(colIdA), src1=hex(blockSize-1), comment="(col + offset) % block_size"))
    module.add(VAndB32(dst=vgpr(colIdB), src0=vgpr(colIdB), src1=hex(blockSize-1), comment="(col + offset) % block_size"))
  writer.vgprPool.checkIn(tmpVgpr)

def _graTileAssignment_legacy(writer, kernel, useSwizzling=True):
  module = Module()
  module.addComment0("GR Offset Calculation for Subtile Based Tiling")
  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo
  subIterKBytes = tileInfoA.subIterKBytes
  wavesize = kernel["WavefrontSize"]
  ldsRowBankSize = writer.states.archCaps["LDSBankCount"] * writer.states.archCaps["LDSBankWidth"]
  loadWidth = tileInfoA.loadWidthGR
  assert subIterKBytes % loadWidth == 0
  assert subIterKBytes <= ldsRowBankSize
  blockSize = subIterKBytes // loadWidth
  numRowsPerLDSBanks = ldsRowBankSize // subIterKBytes
  tmpVgpr = writer.vgprPool.checkOut(7, tag="_graTileAssignment_legacy_tmpVgpr")
  colIdA = tmpVgpr
  colIdB = tmpVgpr + 1
  rowId = tmpVgpr + 2
  rowOffsetA = tmpVgpr + 3
  rowOffsetB = tmpVgpr + 4
  waveId = tmpVgpr + 5
  laneId = tmpVgpr + 6
  module.add(VLShiftRightB32(dst=vgpr(waveId), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="Wave Id"))
  module.add(VAndB32(dst=vgpr(laneId), src0=vgpr("Serial"), src1=wavesize-1, comment=""))
  module.add(VAndB32(dst=vgpr(colIdA), src0=vgpr("Serial"), src1=(blockSize-1), comment="get col_id in wave for %uB load"%loadWidth))
  module.add(VLShiftRightB32(dst=vgpr(rowId), shiftHex=hex(blockSize.bit_length()-1), src=vgpr(laneId), comment="row id within wave"))
  _grSwizzleColIds_legacy(module, writer, tileInfoA, tileInfoB, blockSize, numRowsPerLDSBanks,
                          laneId, colIdA, colIdB, waveId)
  _grComputeRowPartition_legacy(module, kernel, writer, tileInfoA, waveId, rowOffsetA)
  _grComputeRowPartition_legacy(module, kernel, writer, tileInfoB, waveId, rowOffsetB)
  _grComputeAllOffsets_legacy(module, writer, tileInfoA, colIdA, rowId, rowOffsetA)
  _grComputeAllOffsets_legacy(module, writer, tileInfoB, colIdB, rowId, rowOffsetB)
  writer.vgprPool.checkIn(tmpVgpr)
  _grComputeSubtileOffsets_legacy(writer, module, tileInfoA)
  _grComputeSubtileOffsets_legacy(writer, module, tileInfoB)
  return module

##################################################
# Subroutine to generate GR load code
#
def emitSingleBufferLoad(tileInfo, kernel, sId0, sId1):
  """Emit buffer_load instructions for a single subtile (sId0, sId1).

  When loadRatioGR > 1, multiple local subtiles share the same global read.
  Only the first subtile in each group emits the load; others return empty.

  Args:
      tileInfo: TileInfo or TileInfo for the tensor component
      sId0:     Subtile row index
      sId1:     Subtile column index (K-dimension)
  """
  module = Module()

  # TDM path: emit one tensor_load_to_lds per tensor, skip all per-subtile DTL loads
  if kernel.get("enableTDM%s" % tileInfo.tc[0], False):
    if sId0 == 0 and sId1 == 0:
      tc = tileInfo.tc
      group0 = "tdm%sGroup0" % tc
      group1 = "tdm%sGroup1" % tc
      module.add(TensorLoadToLds(sgpr(group0, 4), sgpr(group1, 8), None, None,
                                 comment="TDM: global->LDS for %s" % tc))
    return module

  linearId = tileInfo.getLocalSubtileLinearId(sId0, sId1)
  grBaseId = int(math.floor(linearId / tileInfo.loadRatioGR))

  if tileInfo.loadRatioGR > 1:
    firstInGroup = int(grBaseId * tileInfo.loadRatioGR)
    if linearId != firstInGroup:
      return module

  tc = tileInfo.tc
  isGlc = bool(kernel["NonTemporal%s"%tc] & 0x1)
  isSlc = bool(kernel["NonTemporal%s"%tc] & 0x2)
  isNT  = bool(kernel["NonTemporal%s"%tc] & 0x4)

  regListIdx = tileInfo.grRegGroupForSubtileRow(sId0)
  regList = tileInfo.localSubtilesRegister[regListIdx]
  useSgpr = regList.is_sgpr

  offsetK = sId1 * int(tileInfo.mmaTileShape[1] * tileInfo.subtileShape[1] * tileInfo.bpe)

  subtileOffset = int(math.ceil(tileInfo.loadRatioGR*tileInfo.subtileSize))
  WriteBaseAddr = "LocalWriteBaseAddr%s"%tc
  for i in range(tileInfo.numGRPerSubtile):
    m0Offset = int(i * subtileOffset + (sId0 + sId1 * tileInfo.globalSubtileGrid[0]) * tileInfo.subtileSize)
    module.add(SAddU32(dst=mgpr(0), src0=sgpr(WriteBaseAddr), src1=(m0Offset - offsetK)))
    mubuf = MUBUFModifiers(offen=True, offset12=offsetK, glc=isGlc, slc=isSlc, nt=isNT, lds=True)

    soffset = regList.ref(0) if len(regList) > 0 and useSgpr else 0
    voff = tileInfo.sharedVgprGROffset[i] if useSgpr or len(regList) == 0 else regList.indices[i]
    module.add(BufferLoadB128(dst=None, vaddr=vgpr(voff), saddr=sgpr("Srd%s"%tc, 4), soffset=soffset, mubuf=mubuf, comment="grBaseId = %u, i= %u"%(grBaseId , i)))

  return module


def emitSubtileBufferLoad(tc, writer, kernel, subtileId):
  tileInfo = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
  return emitSingleBufferLoad(tileInfo, kernel, subtileId[0], subtileId[1])

##################################################
# Subroutine to generate GR load code
# Initial idea: maybe store asm in modules in a separate obj?
#
def globalReadDoSubtile(tc, writer, kernel):
  module = Module()

  tileInfo = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo

  for j in range(tileInfo.localSubtileGrid[1]):
    for i in range(tileInfo.localSubtileGrid[0]):
      module.addComment0("Emit load for %s subtile: [%u, %u]"%(tc, i, j))
      module.add(emitSubtileBufferLoad(tc, writer, kernel, [i, j]))

  return module

##################################################
# Subroutine to generate DTL M0 LDS buffer swap
#
def globalReadDTLInitCommonSgpr(writer, kernel):
  return _globalReadDTLInitCommonSgpr_legacy(writer, kernel)

def _globalReadDTLInitCommonSgpr_legacy(writer, kernel):
  module = Module()
  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo
  wavesize = kernel["WavefrontSize"]
  vgprWaveId = writer.vgprPool.checkOut(1, tag="_globalReadDTLInitCommonSgpr_legacy_vgprWaveId")
  module.addComment0("Compute shared offsets used by m0 in DTL loads")
  module.add(VLShiftRightB32(dst=vgpr(vgprWaveId), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="Wave Id"))
  tmpVgpr = writer.vgprPool.checkOut(2, tag="_globalReadDTLInitCommonSgpr_legacy_tmpVgpr")
  rowOffsetA = tmpVgpr
  rowOffsetB = tmpVgpr + 1
  _grComputeRowPartition_legacy(module, kernel, writer, tileInfoA, vgprWaveId, rowOffsetA)
  _grComputeRowPartition_legacy(module, kernel, writer, tileInfoB, vgprWaveId, rowOffsetB)
  subIterKBytes = tileInfoA.subIterKBytes
  module.add(VLShiftLeftB32(dst=vgpr(rowOffsetA), shiftHex=hex((subIterKBytes).bit_length()-1), src=vgpr(rowOffsetA), comment="Apply wave-specific offset for A"))
  module.add(VLShiftLeftB32(dst=vgpr(rowOffsetB), shiftHex=hex((subIterKBytes).bit_length()-1), src=vgpr(rowOffsetB), comment="Apply wave-specific offset for B"))
  module.add(SNop(waitState=0, comment="Wait for VGPR to be ready"))
  module.add(VReadfirstlaneB32(dst=sgpr("LocalWriteBaseAddrA"), src=vgpr(rowOffsetA), comment="Store base LDS offset, will be modified"))
  module.add(VReadfirstlaneB32(dst=sgpr("LocalWriteBaseAddrB"), src=vgpr(rowOffsetB), comment="Store base LDS offset, will be modified"))
  module.add(SAddU32(dst=sgpr("LocalWriteBaseAddrB"), src0=sgpr("LocalWriteBaseAddrB"), src1=hex(writer.ldsStartOffsetB), comment=""))
  module.add(SAddU32(dst=sgpr("SwapA"), src0=sgpr("LocalWriteBaseAddrA"), src1=writer.ldsTotalSize, comment=""))
  module.add(SXorB32(dst=sgpr("SwapA"), src0=sgpr("LocalWriteBaseAddrA"), src1=sgpr("SwapA"), comment=""))
  module.add(SAddU32(dst=sgpr("SwapB"), src0=sgpr("LocalWriteBaseAddrB"), src1=writer.ldsTotalSize, comment=""))
  module.add(SXorB32(dst=sgpr("SwapB"), src0=sgpr("LocalWriteBaseAddrB"), src1=sgpr("SwapB"), comment=""))
  writer.vgprPool.checkIn(vgprWaveId)
  writer.vgprPool.checkIn(tmpVgpr)
  return module

##################################################
# Subroutine to generate DTL M0 LDS buffer swap
#
def globalReadLDSBufferSwap(tc, writer, kernel):
  if tc in ['A', 'B']:
    ti_ = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
    if kernel.get("enableTDM%s" % tc, False):
      ldsAddrSgpr = "tdmLdsAddr%s" % tc
      swapSgpr = "tdmLdsSwapMask%s" % tc
      module = Module()
      module.addComment0("TDM: swap %s LDS buffer (XOR with per-tensor swap mask)" % tc)
      module.add(SXorB32(dst=sgpr(ldsAddrSgpr), src0=sgpr(ldsAddrSgpr), src1=sgpr(swapSgpr), comment=""))
      group0 = "tdm%sGroup0" % tc
      module.add(SMovB32(dst=sgpr("%s+1" % group0), src=sgpr(ldsAddrSgpr), comment="sync descriptor LDS addr"))
      return module
    return ti_.emitGRLDSBufferSwap(writer, kernel)
  else:
    ti_ = writer.states.mxsa.tileInfo if tc == 'MXSA' else writer.states.mxsb.tileInfo
    return emitScaleGRLDSSwap(ti_, writer, kernel)



################################################################################
# TDM subtile functions (global offset, descriptor init, StreamK offset)
################################################################################

def tdmGlobalOffsetSubtile(writer, kernel, tP):
  """Per-wave global address for subtile TDM.

  All waves cooperatively load the tile: wave w covers M-rows
  [w*mt/numWaves, (w+1)*mt/numWaves) across the full wave count, rather
  than only this tensor's wave axis. Splitting over every wave avoids the
  duplicate loads the axis-only split issued for waves sharing an axis id.
  The LDS tile end-state (identity map global-row r -> LDS-row r) is
  unchanged; the barrier before local reads (WaitGROp has_sync) makes
  every wave's rows visible to all consumers.
  """
  tc = tP["tensorChar"]
  ti = tP["idx"]
  bpe = tP["bpeGR"]
  tlu = tP["tlu"]
  mt = kernel[f"MacroTile{ti}"]
  wavelen = kernel["WavefrontSize"]
  numWaves = prod(kernel["MIWaveGroup"])
  mod = Module(f"TDM Global Offset Subtile {tc}")

  with writer.allocTmpSgpr(3) as tmpSgprRes:
    tmp = tmpSgprRes.idx
    waveOff = tmpSgprRes.idx + 2

    tileStride = writer.strideRef(tc, ti)
    mod.add(SMulI32(dst=sgpr(tmp), src0=tileStride, src1=int(mt * bpe),
                     comment=f"stride * MT({mt}) * bpe({bpe})"))
    mod.add(SMulI32(dst=sgpr(tmp), src0=sgpr(tmp), src1=sgpr(f"WorkGroup{ti}"),
                     comment="*= wgId"))

    if numWaves > 1:
      mod.add(VReadfirstlaneB32(dst=sgpr(waveOff), src=vgpr("Serial"), comment="first tId"))
      mod.add(SLShiftRightB32(dst=sgpr(waveOff), src=sgpr(waveOff),
                               shiftHex=hex(int(ceil(log2(wavelen)))), comment=f"wId = tId / {wavelen}"))
      tileStrideSep = writer.strideRef(tc, 3) if tlu else writer.strideRef(tc, ti)
      mod.add(SMulI32(dst=sgpr(waveOff), src0=sgpr(waveOff), src1=int(mt // numWaves * bpe),
                       comment=f"waveOff = waveId * {mt // numWaves} * {bpe}"))
      mod.add(SMulI32(dst=sgpr(waveOff), src0=sgpr(waveOff), src1=tileStrideSep,
                       comment="waveOff *= stride"))
      mod.add(SAddU32(dst=sgpr(tmp), src0=sgpr(tmp), src1=sgpr(waveOff), comment="+= waveOff"))

    mod.add(SAddU32(dst=sgpr(f"Address{tc}"), src0=sgpr(f"Address{tc}"), src1=sgpr(tmp),
                     comment=f"+= offset(lo)"))
    mod.add(SAddCU32(dst=sgpr(f"Address{tc}+1"), src0=sgpr(f"Address{tc}+1"), src1=0,
                      comment=f"+= offset(hi)"))

    if kernel["ProblemType"]["Batched"] and kernel["ProblemType"]["StridedBatched"]:
      ia = tP["ia"]
      batchStrideName = f"Stride{tc}{writer.states.indexChars[ia[2]]}"
      mod.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(tmp), sgpr(tmp+1),
                                                     sgpr(batchStrideName), sgpr("WorkGroup2"),
                                                     comment="Batch: Stride*WG"))
      mod.add(SLShiftLeftB64(dst=sgpr(tmp, 2), src=sgpr(tmp, 2),
                              shiftHex=int(log2(bpe)), comment="scale by bpe"))
      mod.add(SAddU64(dst=sgpr(f"Address{tc}", 2), src0=sgpr(tmp, 2), src1=sgpr(f"Address{tc}", 2),
                       comment="+= batch"))

  return mod


def initTDMDescriptorSubtile(writer, kernel, tP):
  """Subtile variant of initTDMDescriptor()."""
  from ...Components.TensorDataMover import TensorDataMoverLoad
  comp = TensorDataMoverLoad.find(writer)
  tc = tP['tensorChar']
  ti = tP["idx"]
  tileChar = tP["tileChar"]
  mod = Module(f"Init TDM Descriptor Subtile {tc}")

  def descSgprName(idx):
    assert idx < 2
    return f"tdm{tc}Group{idx}"

  def strideRefName():
    return f"Stride{tc}{tileChar}"

  def sizeRefName(idx):
    idxChar = INDEX_CHARS[idx]
    return f"Size{idxChar}"

  dtype = kernel["ProblemType"][f"DataType{tc}"]
  mt = kernel[f"MacroTile{ti}"]
  du = kernel["DepthU"]
  bpe = tP["bpeGR"]
  numWaves = prod(kernel["MIWaveGroup"])
  wavelen = kernel["WavefrontSize"]

  # Use subtile LDS offsets from writer state (not kernel["LdsOffset{tc}"])
  ldsOffsetMap = {
    'A': writer.ldsStartOffsetA,
    'B': writer.ldsStartOffsetB,
  }
  ldsConstOffset = ldsOffsetMap.get(tc, 0)

  sizeTile0, sizeTile1 = du, mt
  # TDM D# Group1 pad fields
  #   padAmountBytes   -> pad_amount   [31:25], bytes inserted per pad event
  #   padIntervalBytes -> pad_interval [24:22], bytes written between pads
  # Sourced from TileInfo.ldsRowPadBytes so GR and LR
  # see the same value.
  tileInfoForTc = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
  padAmountBytes = int(getattr(tileInfoForTc, "ldsRowPadBytes", 0))
  padIntervalBytes = int(du * bpe) if padAmountBytes else 0

  mod.add(comp.initOperands(descSgprName(0), descSgprName(1), None, None))
  mod.add(comp.setDataType(dtype, descSgprName(1)))
  mod.add(comp.setGlobalAddr(descSgprName(0), f"Address{tc}"))

  with writer.allocTmpSgpr(1) as tmpSgprRes:
    waveOffsetSgprIdx = tmpSgprRes.idx
    mod.add(VReadfirstlaneB32(sgpr(waveOffsetSgprIdx), vgpr("Serial"), "first tId"))
    mod.add(SLShiftRightB32(sgpr(waveOffsetSgprIdx), ceil(log2(wavelen)), sgpr(waveOffsetSgprIdx), "wId=fTid // wavelen"))
    # Each wave writes its mt/numWaves rows to a distinct LDS region,
    # matching the cooperative full-wave global split in
    # tdmGlobalOffsetSubtile. The union over all waves covers the whole
    # mt-row tile (identity map global-row r -> LDS-row r).
    if padIntervalBytes != 0 and padAmountBytes != 0:
      tileBytes = round(mt // numWaves * du * bpe)
      padBytes = tileBytes // padIntervalBytes * padAmountBytes
      mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), tileBytes + padBytes,
              f"woffset = wId * ({tileBytes}+{padBytes})"))
    else:
      mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), round(mt // numWaves * du * bpe),
              "woffset = wId * (mt // numWaves * du * bpe)"))
    mod.add(SAddU32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), ldsConstOffset,
            f"ldsOffset = woffset + {ldsConstOffset} (subtile LDS offset for {tc})"))
    mod.add(comp.setLdsAddr(descSgprName(0), sgpr(waveOffsetSgprIdx)))
    # Save LDS offset to tracking SGPR for runtime double-buffer swap
    ldsTrackSgpr = f"tdmLdsAddr{tc}"
    mod.add(SMovB32(dst=sgpr(ldsTrackSgpr), src=sgpr(waveOffsetSgprIdx), comment=f"init {ldsTrackSgpr} for buffer tracking"))
    # Compute swap mask: swapMask = addr XOR (addr + ldsTotalSize)
    # Used by globalReadLDSBufferSwap to toggle between buffer 0 and buffer 1.
    swapMaskSgpr = f"tdmLdsSwapMask{tc}"
    ldsTotalSize = writer.ldsTotalSize
    mod.add(SAddU32(dst=sgpr(swapMaskSgpr), src0=sgpr(waveOffsetSgprIdx), src1=ldsTotalSize, comment=f"addr + ldsTotalSize({ldsTotalSize})"))
    mod.add(SXorB32(dst=sgpr(swapMaskSgpr), src0=sgpr(waveOffsetSgprIdx), src1=sgpr(swapMaskSgpr), comment=f"swapMask = addr XOR (addr + ldsTotalSize)"))
  sizeShifter = 1 if dtype.isFloat4() else 0
  sizeShifterDim = sizeShifter

  mod.add(comp.setIterationEnabled(descSgprName(1), False))
  mod.add(comp.setPadding(descSgprName(1), padIntervalBytes, padAmountBytes))
  mod.add(comp.setTensorDim0(descSgprName(1), sizeRefName(3), writer, sizeShifterDim))
  mod.add(comp.setTensorDim1(descSgprName(1), sizeRefName(ti), writer))

  sizeShifterTile = sizeShifter
  mod.add(comp.setTensorTile0(descSgprName(1), sizeTile0, writer, sizeShifterTile))
  mod.add(comp.setTensorTile1(descSgprName(1), sizeTile1 // numWaves, writer))
  mod.add(comp.setTensorStride0(descSgprName(1), strideRefName(), sizeShifterTile))
  return mod


def tdmApplyStreamKOffsetSubtile(writer, kernel, tP):
  """Assert StreamKLocalStart == 0 for subtile TDM path.

  StreamK=3 (Two-Tile) aligns WG iteration ranges to tile boundaries,
  so StreamKLocalStart is always 0.  The TDM descriptor is already
  initialized with the correct Address{tc} and does not need updating.

  If a future StreamK mode breaks this invariant, Address{tc} would need
  to be offset and the TDM descriptor synced (s_mov_b64 + s_or_b32).
  """
  tc = tP["tensorChar"]
  mod = Module(f"TDM StreamK K-offset subtile {tc}")
  # Assert StreamKLocalStart == 0 at runtime
  mod.addComment0(f"Assert: StreamKLocalStart == 0 (subtile TDM {tc})")
  mod.add(SCmpEQU32(src0=sgpr("StreamKLocalStart"), src1=0,
                    comment="subtile TDM requires tile-aligned WG starts"))
  assertLabel = Label(f"SK_Assert_OK_{tc}", "")
  mod.add(SCBranchSCC1(labelName=assertLabel.getLabelName(),
                       comment="OK: StreamKLocalStart == 0"))
  # Trap if invariant violated
  mod.add(SEndpgm(comment=f"FATAL: StreamKLocalStart != 0 for subtile TDM {tc}"))
  mod.add(assertLabel)
  return mod

##################################################
# Subroutine to update ptrs
#
def globalReadPtrUpdates(tc, writer, kernel):
  ti_ = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
  return ti_.emitGRPtrUpdate(writer, kernel)
