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
    SAddCU32, SAddU32, SAddU64, SAndB32, SMaxI32, SMinU32, SMovB32, SMovB64, SMulI32,
    SNop, SOrB32, SSubI32, SXorB32,
    SCBranchSCC1, SCmpEQU32, SCSelectB32, SEndpgm,
    SLShiftLeftB64, SLShiftRightB32,
    VAddU32, VAndB32, VCmpXEqU32,
    VLShiftLeftB32, VLShiftRightB32, VMovB32, VOrB32,
    TensorLoadToLds,
    VMulLOU32, VReadfirstlaneB32, VSubU32, VXorB32,
)

from .SubtileGeometry import (
    RegList,
    GRTag_1x1, GRTag_1x2, GRTag_2x2, GRTag_TLU1,
)
from .SubtileScaleEmit import emitScaleGRLDSSwap
from .SubtileTLUSwizzle import (selectTLUSwizzle, selectTLUColScatter, stripStrideBytes,
                                tluPadBytes, grLoadBlockBytes)

from math import ceil, log, log2, prod
from rocisa.code import Label
from rocisa.functions import vectorMultiplyBpe
from ...Common import INDEX_CHARS
from ...SolutionStructs.Utilities import isSubtileIterateMode as _isSubtileIterateMode
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
_emitGlobalRead.register(GRTag_TLU1)(_stub)
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
@_allocGROffsetRegisters.register(GRTag_TLU1)
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
  gran = tile.localGRGranularity(getattr(ti, "grLoadWaves", ti.numWaves))
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
@_deallocGROffsetRegisters.register(GRTag_TLU1)
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
@_emitDTLInit.register(GRTag_TLU1)
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
@_emitGRLDSBufferSwap.register(GRTag_TLU1)
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
@_emitGRPtrUpdate.register(GRTag_TLU1)
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
  # TLU=1 (NT / free-dim contiguous): the free dim is unit-stride and the K
  # (unroll) dim is strided, so one DepthU K-window advances the base by
  # DepthU * strideK * bpe bytes, not DepthU * bpe.  depthUBytes already folds
  # in DepthU * bpe (sub-byte-safe), so scale it by the runtime K stride.
  if isinstance(tag, GRTag_TLU1):
    unrollIdx = kernel["ProblemType"]["IndexUnroll"]
    strideK = writer.strideRef(tc, unrollIdx)
    if writer.isConstUnitStride(strideK):
      # K is unit-stride (no NT transpose on this operand): plain DepthU*bpe.
      module.add(SAddU32(dst=sgpr(f"Srd{tc}"), src0=sgpr(f"Srd{tc}"), src1=inc,
                 comment=f"{tc}: advance SRD by {inc} bytes"))
      module.add(SAddCU32(dst=sgpr(f"Srd{tc}+1"), src0=sgpr(f"Srd{tc}+1"), src1=0,
                 comment=f"{tc}: carry"))
      return module
    with writer.allocTmpSgpr(1) as tmpSgprRes:
      incSgpr = tmpSgprRes.idx
      module.add(SMulI32(dst=sgpr(incSgpr), src0=inc, src1=strideK,
                 comment=f"{tc}: DepthU*bpe({inc}) * strideK (NT K-window bytes)"))
      module.add(SAddU32(dst=sgpr(f"Srd{tc}"), src0=sgpr(f"Srd{tc}"), src1=sgpr(incSgpr),
                 comment=f"{tc}: advance SRD by DepthU K-window"))
      module.add(SAddCU32(dst=sgpr(f"Srd{tc}+1"), src0=sgpr(f"Srd{tc}+1"), src1=0,
                 comment=f"{tc}: carry"))
    return module

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

def _isGRTLU1(tileInfo):
  return bool(tileInfo.gr and isinstance(tileInfo.gr.config.tag, GRTag_TLU1))


def _grSwizzleColId_single(module, writer, tileInfo, blockSize, numRowsPerLDSBanks,
                           laneId, colId, waveId):
  """Row-major GR colId swizzle for a single tensor.

  Same rotation as _grSwizzleColIds_legacy, minus the colIdB copy: that path
  swizzles A and derives B from it, which only works when both operands share
  the row-major layout.
  """
  tc = tileInfo.tc
  tmpVgpr = writer.vgprPool.checkOut(3, tag="_grSwizzleColId_single_tmpVgpr")
  ldsRowId = tmpVgpr
  tmp = tmpVgpr + 1
  waveRotation = tmpVgpr + 2
  half = blockSize // 2
  module.addComment0("Swizzling (%s)" % tc)
  module.add(VLShiftRightB32(dst=vgpr(ldsRowId), shiftHex=hex(blockSize.bit_length()-1), src=vgpr(laneId), comment="row id within wave"))
  module.add(VLShiftRightB32(dst=vgpr(ldsRowId), shiftHex=hex(numRowsPerLDSBanks.bit_length()-1), src=vgpr(ldsRowId), comment="lds row id"))
  module.add(VAndB32(dst=vgpr(tmp), src0=vgpr(ldsRowId), src1=hex(1), comment="swap_bit = ldsRowId & 1"))
  if tileInfo.bpe == 1:  # FP8: step1=block-swap, step2=wave K_group rotation
    module.add(VLShiftLeftB32(dst=vgpr(tmp), shiftHex=hex(int(math.log2(half))), src=vgpr(tmp),
               comment=f"swap_bit * {half}"))
    module.add(VXorB32(dst=vgpr(colId), src0=vgpr(colId), src1=vgpr(tmp),
               comment="FP8 step1: block-swap colId"))
    module.add(VAndB32(dst=vgpr(tmp), src0=vgpr(waveId), src1=hex(1), comment="wave_half = waveId & 1"))
    module.add(VLShiftLeftB32(dst=vgpr(tmp), shiftHex=hex(1), src=vgpr(tmp), comment="rotation = wave_half * 2"))
    if tileInfo.loadRatioGR != 0.5:
      module.add(VAndB32(dst=vgpr(waveRotation), src0=vgpr(colId), src1=hex(4), comment="FP8 step2: block_bit = colId & 4"))
      module.add(VAndB32(dst=vgpr(colId), src0=vgpr(colId), src1=hex(3), comment="K_group = colId & 3"))
      module.add(VAddU32(dst=vgpr(colId), src0=vgpr(colId), src1=vgpr(tmp), comment="K_group + rotation"))
      module.add(VAndB32(dst=vgpr(colId), src0=vgpr(colId), src1=hex(3), comment="(K_group+rotation) % 4"))
      module.add(VAddU32(dst=vgpr(colId), src0=vgpr(colId), src1=vgpr(waveRotation), comment="K_group_rot + block_bit"))
  else:  # FP4/FP16: pair-swap (even ldsRowId) + intra/inter-wave rotation
    module.add(VCmpXEqU32(dst=VCC(), src0=0, src1=vgpr(tmp), comment="lds row id % 2 == 0 ?"))
    module.add(VMovB32(dst=vgpr(colId), src=vgpr(colId), dpp=DPPModifiers(quad_perm=[1,0,3,2]), comment="swap colId pairs for swizzling"))
    module.add(SMovB64(dst=EXEC(), src=-1))
    module.addComment0("Rotation within a single wave")
    module.add(VLShiftRightB32(dst=vgpr(tmp), shiftHex=hex(1), src=vgpr(ldsRowId), comment=""))
    module.add(VLShiftLeftB32(dst=vgpr(tmp), shiftHex=hex(1), src=vgpr(tmp), comment="(ldsRowId //2) * 2"))
    module.add(VSubU32(dst=vgpr(tmp), src0=hex(blockSize), src1=vgpr(tmp), comment="rotation offset : blockSize - (ldsRowId//2)*2"))
    if tileInfo.loadRatioGR != 0.5:
      module.addComment0("Rotation per wave")
      module.add(VAndB32(dst=vgpr(waveRotation), src0=vgpr(waveId), src1=hex(1), comment=""))
      module.add(VLShiftLeftB32(dst=vgpr(waveRotation), shiftHex=hex((2*numRowsPerLDSBanks).bit_length() - 1), src=vgpr(waveRotation), comment=""))
      module.add(VSubU32(dst=vgpr(waveRotation), src0=vgpr(tmp), src1=vgpr(waveRotation), comment=""))
      module.add(VAddU32(dst=vgpr(colId), src0=vgpr(waveRotation), src1=vgpr(colId), comment=""))
    else:
      module.add(VAddU32(dst=vgpr(colId), src0=vgpr(tmp), src1=vgpr(colId), comment=""))
    module.add(VAndB32(dst=vgpr(colId), src0=vgpr(colId), src1=hex(blockSize-1), comment="(col + offset) % block_size"))
  writer.vgprPool.checkIn(tmpVgpr)


def _graTileAssignment_rowMajorSingle(writer, kernel, module, tileInfo):
  """Row-major (TLU=0) GR offsets for a single tensor.

  Mirrors the interleaved A+B path, but every parameter comes from this
  tensor's own geometry so it can be paired with a TLU=1 operand (NN / TT).
  """
  subIterKBytes = tileInfo.subIterKBytes
  wavesize = kernel["WavefrontSize"]
  ldsRowBankSize = writer.states.archCaps["LDSBankCount"] * writer.states.archCaps["LDSBankWidth"]
  loadWidth = tileInfo.loadWidthGR
  assert subIterKBytes % loadWidth == 0
  assert subIterKBytes <= ldsRowBankSize
  blockSize = subIterKBytes // loadWidth
  numRowsPerLDSBanks = ldsRowBankSize // subIterKBytes
  tmpVgpr = writer.vgprPool.checkOut(5, tag="_graTileAssignment_rowMajorSingle_tmpVgpr")
  colId, rowId, rowOffset, waveId, laneId = range(tmpVgpr, tmpVgpr + 5)
  module.add(VLShiftRightB32(dst=vgpr(waveId), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="Wave Id"))
  module.add(VAndB32(dst=vgpr(laneId), src0=vgpr("Serial"), src1=wavesize-1, comment=""))
  module.add(VAndB32(dst=vgpr(colId), src0=vgpr("Serial"), src1=(blockSize-1), comment="get col_id in wave for %uB load"%loadWidth))
  module.add(VLShiftRightB32(dst=vgpr(rowId), shiftHex=hex(blockSize.bit_length()-1), src=vgpr(laneId), comment="row id within wave"))
  _grSwizzleColId_single(module, writer, tileInfo, blockSize, numRowsPerLDSBanks,
                         laneId, colId, waveId)
  _grComputeRowPartition_legacy(module, kernel, writer, tileInfo, waveId, rowOffset)
  _grComputeAllOffsets_legacy(module, writer, tileInfo, colId, rowId, rowOffset)
  writer.vgprPool.checkIn(tmpVgpr)


def _graTileAssignment_legacy(writer, kernel, useSwizzling=True):
  module = Module()
  module.addComment0("GR Offset Calculation for Subtile Based Tiling")
  tileInfoA = writer.states.a.tileInfo
  tileInfoB = writer.states.b.tileInfo
  aTLU1 = _isGRTLU1(tileInfoA)
  bTLU1 = _isGRTLU1(tileInfoB)
  # TLU=1 (NT / free-dim contiguous): each lane's 128-bit buffer_load grabs a
  # full free-dim strip (M for A, N for B) at one K row; lanes walk K. Offset
  # addressing is a pure K-stride ramp, so it does not use the row-major
  # colId/rowId/bank-swizzle machinery below.
  if aTLU1 and bTLU1:
    module.add(_graTileAssignment_tlu(writer, kernel, tileInfoA))
    module.add(_graTileAssignment_tlu(writer, kernel, tileInfoB))
    _grComputeSubtileOffsets_tlu(writer, module, tileInfoA)
    _grComputeSubtileOffsets_tlu(writer, module, tileInfoB)
    return module
  # NN / TT: exactly one operand is free-dim contiguous. The two emitters share
  # no state, so run each tensor through the one matching its own layout. The
  # row-major path below interleaves A and B (it derives colIdB from colIdA), so
  # the odd operand out gets the single-tensor variant instead.
  if aTLU1 or bTLU1:
    for ti, isTLU1 in ((tileInfoA, aTLU1), (tileInfoB, bTLU1)):
      if isTLU1:
        module.add(_graTileAssignment_tlu(writer, kernel, ti))
        _grComputeSubtileOffsets_tlu(writer, module, ti)
      else:
        _graTileAssignment_rowMajorSingle(writer, kernel, module, ti)
        _grComputeSubtileOffsets_legacy(writer, module, ti)
    return module
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


def _graTileAssignment_tlu_colScatter(writer, kernel, tileInfo, module, laneId,
                                      strideK, cs):
  """GR per-lane offsets for the column-scatter TLU layout (8x1 fp4 and up).

  Each DTL load i owns a scattered set of K-columns: physical thread T within a
  load holds logical (m_chunk, col_group) recovered by de-interleaving T (the
  inverse of the bit-interleave the LR read applies).  The global K-column is
  ``col = col_group * N + i`` and the free-dim (M/N) start is
  ``m_chunk * elemsPerChunk``, so the per-lane byte offset for load i is

      offset(T, i) = (col * strideK + m_chunk * elemsPerChunk) * bpe
                   = ((col_group*N + i) * strideK + m_chunk * elemsPerChunk) * bpe

  col_group and m_chunk are load-independent, so their contribution is computed
  once per lane; the loop only adds ``i * strideK`` in K.  See SubtileTLUSwizzle
  (TLUColScatter) and the verified bank model.
  """
  tc = tileInfo.tc
  tile = tileInfo.gr
  bpeBits = int(8 * tileInfo.bpe)
  elemsPerChunk = int(16 / tileInfo.bpe)
  N = cs.N

  module.addComment0("%s: TLU=1 GR offset (col_scatter, %ux1)" % (tc, N))
  # De-interleave laneId (= thread T within the load) into m_chunk and col_group.
  mc = writer.vgprPool.checkOut(1, tag="_graColScatter_mc")
  cg = writer.vgprPool.checkOut(1, tag="_graColScatter_cg")
  bit = writer.vgprPool.checkOut(1, tag="_graColScatter_bit")
  # m_chunk: gather thread bits cs.mChunkThreadBits[j] -> m_chunk bit j.
  # A stack of 2 has cBits == 0 (no m_chunk bits), so the loop below leaves mc
  # untouched and this zero is the value that is used.
  module.add(VMovB32(dst=vgpr(mc), src=0, comment="%s: m_chunk = 0" % tc))
  for j, tb in enumerate(cs.mChunkThreadBits):
    module.add(VLShiftRightB32(dst=vgpr(bit), shiftHex=hex(tb), src=vgpr(laneId),
               comment="%s: laneId >> %u" % (tc, tb)))
    module.add(VAndB32(dst=vgpr(bit), src0=vgpr(bit), src1=hex(1),
               comment="%s: thread bit %u" % (tc, tb)))
    if j == 0:
      module.add(VMovB32(dst=vgpr(mc), src=vgpr(bit), comment="%s: m_chunk bit 0" % tc))
    else:
      module.add(VLShiftLeftB32(dst=vgpr(bit), shiftHex=hex(j), src=vgpr(bit),
                 comment="%s: -> m_chunk bit %u" % (tc, j)))
      module.add(VOrB32(dst=vgpr(mc), src0=vgpr(mc), src1=vgpr(bit),
                 comment="%s: accumulate m_chunk" % tc))
  # col_group: gather thread bits cs.cgThreadBits[i] -> col_group bit i.
  # gBits (= 7 - log2(N)) is never 0, so i == 0 below always writes cg first.
  for i, tb in enumerate(cs.cgThreadBits):
    module.add(VLShiftRightB32(dst=vgpr(bit), shiftHex=hex(tb), src=vgpr(laneId),
               comment="%s: laneId >> %u" % (tc, tb)))
    module.add(VAndB32(dst=vgpr(bit), src0=vgpr(bit), src1=hex(1),
               comment="%s: thread bit %u" % (tc, tb)))
    if i == 0:
      module.add(VMovB32(dst=vgpr(cg), src=vgpr(bit), comment="%s: col_group bit 0" % tc))
    else:
      module.add(VLShiftLeftB32(dst=vgpr(bit), shiftHex=hex(i), src=vgpr(bit),
                 comment="%s: -> col_group bit %u" % (tc, i)))
      module.add(VOrB32(dst=vgpr(cg), src0=vgpr(cg), src1=vgpr(bit),
                 comment="%s: accumulate col_group" % tc))
  # col_group * N (load-independent K-column base).
  module.add(VLShiftLeftB32(dst=vgpr(cg), shiftHex=hex(N.bit_length() - 1), src=vgpr(cg),
             comment="%s: col_group * %u (= K-column at load 0)" % (tc, N)))
  # m_chunk * elemsPerChunk (free-dim element start, load-independent).
  module.add(VLShiftLeftB32(dst=vgpr(mc), shiftHex=hex(elemsPerChunk.bit_length() - 1),
             src=vgpr(mc), comment="%s: m_chunk * %u (free-dim start)" % (tc, elemsPerChunk)))

  waveAxisOffVgpr = _tluWaveAxisGlobalOffset(writer, kernel, module, tileInfo)
  colK = writer.vgprPool.checkOut(1, tag="_graColScatter_colK")
  for i in range(tileInfo.numGRPerSubtile):
    out = tile.sharedVgprGROffset[i]
    # K-column for load i = col_group*N + i.
    module.add(VAddU32(dst=vgpr(colK), src0=hex(i), src1=vgpr(cg),
               comment="%s: K-column = col_group*%u + %u" % (tc, N, i)))
    # colK * strideK (elements).
    module.add(VMulLOU32(dst=vgpr(colK), src0=strideK, src1=vgpr(colK),
               comment="%s: K-column * strideK" % tc))
    # + m_chunk*elemsPerChunk (free dim is unit stride).
    module.add(VAddU32(dst=vgpr(colK), src0=vgpr(colK), src1=vgpr(mc),
               comment="%s: + free-dim start" % tc))
    # * bpe (sub-byte safe).
    module.add(VLShiftLeftB32(dst=vgpr(colK), shiftHex=hex(bpeBits.bit_length() - 1),
               src=vgpr(colK), comment="%s: * bpe" % tc))
    module.add(VLShiftRightB32(dst=vgpr(out), shiftHex=hex(3), src=vgpr(colK),
               comment="%s: to bytes" % tc))
    if waveAxisOffVgpr is not None:
      module.add(VAddU32(dst=vgpr(out), src0=vgpr(out), src1=vgpr(waveAxisOffVgpr),
                 comment="%s: + per-wave free-dim (M/N) global offset" % tc))
  if waveAxisOffVgpr is not None:
    writer.vgprPool.checkIn(waveAxisOffVgpr)
  writer.vgprPool.checkIn(colK)
  writer.vgprPool.checkIn(bit)
  writer.vgprPool.checkIn(cg)
  writer.vgprPool.checkIn(mc)
  return module


def _graTileAssignment_tlu(writer, kernel, tileInfo):
  """GR per-lane offset for TLU=1 (NT / free-dim contiguous) subtile tiles.

  For NT the free dimension (M for A, N for B) is contiguous in global memory,
  so a single 128-bit buffer_load covers a full free-dim strip at one K row.
  The wave's 64 lanes span 64 K rows; numGRPerSubtile loads cover the remaining
  K.  The per-lane global byte offset is therefore a pure K ramp:

      offset(lane, i) = (laneId + i * wavesize) * strideK * bpe

  where strideK is the tensor's unroll (K) stride in elements.  One VGPR is
  produced per GR load into sharedVgprGROffset[]; the M/N position lives inside
  the load width, so no per-lane free-dim term and no bank-swizzle is needed.
  """
  module = Module()
  tc = tileInfo.tc
  tile = tileInfo.gr
  wavesize = kernel["WavefrontSize"]
  bpeBits = int(8 * tileInfo.bpe)
  unrollIdx = kernel["ProblemType"]["IndexUnroll"]
  # strideRef returns an sgpr(...) container, or a "constStride.." string when
  # the index is the packed unit-stride dim (not the case for the K index here).
  strideK = writer.strideRef(tc, unrollIdx)

  module.addComment0("%s: TLU=1 GR offset (K ramp)" % tc)
  laneId = writer.vgprPool.checkOut(1, tag="_graTileAssignment_tlu_laneId")
  module.add(VAndB32(dst=vgpr(laneId), src0=vgpr("Serial"), src1=wavesize - 1,
             comment="%s: laneId" % tc))

  # Column-scatter GR (8x1 and up): a single-bit XOR can no longer reach 1-way,
  # so each DTL load owns a scattered set of K-columns.  Handled in full by the
  # helper below (per-lane deinterleave -> global K/M offset per load).
  cs = selectTLUColScatter(tileInfo)
  if cs is not None:
    _graTileAssignment_tlu_colScatter(writer, kernel, tileInfo, module, laneId,
                                      strideK, cs)
    writer.vgprPool.checkIn(laneId)
    return module

  swz = selectTLUSwizzle(tileInfo)
  tmpVgpr = writer.vgprPool.checkOut(1, tag="_graTileAssignment_tlu_tmpVgpr")
  swzTmp = writer.vgprPool.checkOut(1, tag="_graTileAssignment_tlu_swzTmp") if swz else None

  # M-tiling across b128 loads (fp4 taller stacks).  Each lane's b128 covers
  # elemsPerChunk (16/bpe) contiguous free-dim (M/N) elements at one K row.  A
  # strip is mStripBytes wide, i.e. chunksPerK = mStripBytes/16 b128 chunks per
  # K row.  For the baseline 2x1 fp4 stack chunksPerK==1 (one b128 == one full
  # K row) and the per-lane offset is a pure K ramp; for taller fp4 stacks (4x1,
  # 8x1) a single b128 covers only part of a K row, so physical chunk
  # P = i*wavesize + laneId splits into K row (P // chunksPerK) plus an intra-row
  # M block (P % chunksPerK) of elemsPerChunk elements.  See the LDS image in
  # SubtileLREmit emitSingleDsRead.  Scoped to fp4 (bpe 0.5); other TLU dtypes
  # (e.g. bf16 AB_B16_TLU1) keep the original single-chunk-per-K-row ramp.
  instM = int(tileInfo.mmaTileShape[0])
  mStripBytes = int(tileInfo.subtileShape[0] * instM * tileInfo.bpe)
  isFp4 = float(tileInfo.bpe) == 0.5
  chunksPerK = max(1, mStripBytes // 16) if isFp4 else 1
  elemsPerChunk = int(16 / tileInfo.bpe)
  mTileTmp = writer.vgprPool.checkOut(1, tag="_graTileAssignment_tlu_mTileTmp") if chunksPerK > 1 else None

  # Multi-wave: waves split the free dim (M for A via MIWaveGroup[0], N for B
  # via MIWaveGroup[1]).  Each axis-wave owns localSubtileGrid[0] strips, so its
  # global read starts subtileM*localSub0 free-dim elements later.  The free dim
  # is unit-stride, so this is a flat byte offset added to every GR load.
  waveAxisOffVgpr = _tluWaveAxisGlobalOffset(writer, kernel, module, tileInfo)

  for i in range(tileInfo.numGRPerSubtile):
    out = tile.sharedVgprGROffset[i]
    # Physical chunk P handled by this lane in load i: laneId + i*wavesize
    module.add(VAddU32(dst=vgpr(tmpVgpr), src0=hex(i * wavesize), src1=vgpr(laneId),
               comment="%s: chunk P = laneId + %u" % (tc, i * wavesize)))
    # Bank-conflict swizzle (NT LDS layout): permute which global K-row this
    # lane loads so physical LDS chunk P holds logical K = fswz(P).  fswz is an
    # involution, so the LR read applies the same flip and A round-trips.  See
    # SubtileTLUSwizzle and format.md.
    if swz:
      module.add(VLShiftRightB32(dst=vgpr(swzTmp), shiftHex=hex(swz.xorFromBit),
                 src=vgpr(tmpVgpr), comment="%s: chunk >> %u" % (tc, swz.xorFromBit)))
      module.add(VAndB32(dst=vgpr(swzTmp), src0=vgpr(swzTmp), src1=hex(1),
                 comment="%s: chunk[%u]" % (tc, swz.xorFromBit)))
      module.add(VLShiftLeftB32(dst=vgpr(swzTmp), shiftHex=hex(swz.xorToBit),
                 src=vgpr(swzTmp), comment="%s: -> bit %u" % (tc, swz.xorToBit)))
      module.add(VXorB32(dst=vgpr(tmpVgpr), src0=vgpr(tmpVgpr), src1=vgpr(swzTmp),
                 comment="%s: chunk[%u] ^= chunk[%u]" % (tc, swz.xorToBit, swz.xorFromBit)))
    if chunksPerK > 1:
      # Split chunk P into K row (P // chunksPerK) and intra-row M block
      # (P % chunksPerK) of elemsPerChunk elements.  chunksPerK is a power of 2.
      cpkBits = chunksPerK.bit_length() - 1
      module.add(VAndB32(dst=vgpr(mTileTmp), src0=vgpr(tmpVgpr), src1=hex(chunksPerK - 1),
                 comment="%s: M block = P %% %u" % (tc, chunksPerK)))
      module.add(VLShiftRightB32(dst=vgpr(tmpVgpr), shiftHex=hex(cpkBits), src=vgpr(tmpVgpr),
                 comment="%s: K row = P // %u" % (tc, chunksPerK)))
      # K row * strideK (elements)
      module.add(VMulLOU32(dst=vgpr(tmpVgpr), src0=strideK,
                 src1=vgpr(tmpVgpr), comment="%s: K row * strideK" % tc))
      # + M block * elemsPerChunk (free dim is unit stride)
      module.add(VLShiftLeftB32(dst=vgpr(mTileTmp), shiftHex=hex(elemsPerChunk.bit_length() - 1),
                 src=vgpr(mTileTmp), comment="%s: M block * %u" % (tc, elemsPerChunk)))
      module.add(VAddU32(dst=vgpr(tmpVgpr), src0=vgpr(tmpVgpr), src1=vgpr(mTileTmp),
                 comment="%s: K*strideK + M block" % tc))
    else:
      # * strideK (elements)
      module.add(VMulLOU32(dst=vgpr(tmpVgpr), src0=strideK,
                 src1=vgpr(tmpVgpr), comment="%s: * strideK" % tc))
    # * bpe (sub-byte safe: <<(bpeBits.bit_length()-1) then >>3)
    module.add(VLShiftLeftB32(dst=vgpr(tmpVgpr), shiftHex=hex(bpeBits.bit_length() - 1),
               src=vgpr(tmpVgpr), comment="%s: K*stride*bpe" % tc))
    module.add(VLShiftRightB32(dst=vgpr(out), shiftHex=hex(3), src=vgpr(tmpVgpr),
               comment="%s: to bytes" % tc))
    if waveAxisOffVgpr is not None:
      module.add(VAddU32(dst=vgpr(out), src0=vgpr(out), src1=vgpr(waveAxisOffVgpr),
                 comment="%s: + per-wave free-dim (M/N) global offset" % tc))
  if swzTmp is not None:
    writer.vgprPool.checkIn(swzTmp)
  if mTileTmp is not None:
    writer.vgprPool.checkIn(mTileTmp)
  if waveAxisOffVgpr is not None:
    writer.vgprPool.checkIn(waveAxisOffVgpr)
  writer.vgprPool.checkIn(tmpVgpr)
  writer.vgprPool.checkIn(laneId)
  return module


def _tluWaveAxisId(writer, kernel, module, tc, dst):
  """Compute this wave's axis index for TLU multi-wave partitioning into dst.

  A splits along M (MIWaveGroup[0]): axisId = waveId % mWaves.
  B splits along N (MIWaveGroup[1]): axisId = waveId // mWaves.
  Returns True if the axis actually splits (axisWaves > 1), else leaves dst=0.
  """
  wavesize = kernel["WavefrontSize"]
  mWaves = kernel["MIWaveGroup"][0]
  axisWaves = kernel["MIWaveGroup"][0] if tc == 'A' else kernel["MIWaveGroup"][1]
  if axisWaves <= 1:
    module.add(VMovB32(dst=vgpr(dst), src=0, comment="%s: single axis-wave" % tc))
    return False
  module.add(VLShiftRightB32(dst=vgpr(dst), shiftHex=hex(wavesize.bit_length() - 1),
             src=vgpr("Serial"), comment="%s: waveId" % tc))
  if tc == 'A':
    module.add(VAndB32(dst=vgpr(dst), src0=vgpr(dst), src1=hex(mWaves - 1),
               comment="%s: waveIdM = waveId %% %d" % (tc, mWaves)))
  else:
    module.add(VLShiftRightB32(dst=vgpr(dst), shiftHex=hex(mWaves.bit_length() - 1),
               src=vgpr(dst), comment="%s: waveIdN = waveId / %d" % (tc, mWaves)))
  return True


def _tluCoopWaveId(writer, kernel, module, tileInfo, dst):
  """Index of this wave within the group cooperating on one strip's fetch.

  When only the axis-waves fetch (grCoopWaves == grWavesPerStrip) that index is
  the axis id.  When every wave fetches, it is the full waveId, so the whole
  wavefront covers the strip once instead of each other-axis wave refetching it.
  Returns True if the index can be non-zero.
  """
  tc = tileInfo.tc
  coop = int(getattr(tileInfo, "grCoopWaves", 1))
  if coop <= 1:
    module.add(VMovB32(dst=vgpr(dst), src=0, comment="%s: single fetching wave" % tc))
    return False
  perStrip = max(1, int(getattr(tileInfo, "grWavesPerStrip", 1)))
  kSplit = max(1, coop // perStrip)
  if kSplit <= 1:
    ok = _tluWaveAxisId(writer, kernel, module, tc, dst)
    # The index is a position inside one strip, so it wraps at perStrip.  Only
    # when a single strip spans every axis wave is that the axis id itself; with
    # more than one strip the axis waves past the first group repeat the same
    # positions in the next strip, which _tluStripIdx then steps to.
    if ok and perStrip > 1:
      module.add(VAndB32(dst=vgpr(dst), src0=vgpr(dst), src1=hex(perStrip - 1),
                 comment="%s: position within the strip (axisId %% %u)" % (tc, perStrip)))
    return ok
  # index = (axisId % perStrip) * kSplit + otherId, a bijection onto [0, coop).
  wavesize = kernel["WavefrontSize"]
  mWaves = kernel["MIWaveGroup"][0]
  wid = writer.vgprPool.checkOut(1, tag="_tluCoopWaveId_wid_%s" % tc)
  module.add(VLShiftRightB32(dst=vgpr(wid), shiftHex=hex(wavesize.bit_length() - 1),
             src=vgpr("Serial"), comment="%s: waveId" % tc))
  if tc == 'A':
    module.add(VLShiftRightB32(dst=vgpr(dst), shiftHex=hex(mWaves.bit_length() - 1),
               src=vgpr(wid), comment="%s: otherId = waveId / %d" % (tc, mWaves)))
  else:
    module.add(VAndB32(dst=vgpr(dst), src0=vgpr(wid), src1=hex(mWaves - 1),
               comment="%s: otherId = waveId %% %d" % (tc, mWaves)))
  if perStrip > 1:
    ax = writer.vgprPool.checkOut(1, tag="_tluCoopWaveId_ax_%s" % tc)
    if tc == 'A':
      module.add(VAndB32(dst=vgpr(ax), src0=vgpr(wid), src1=hex(mWaves - 1),
                 comment="%s: axisId = waveId %% %d" % (tc, mWaves)))
    else:
      module.add(VLShiftRightB32(dst=vgpr(ax), shiftHex=hex(mWaves.bit_length() - 1),
                 src=vgpr(wid), comment="%s: axisId = waveId / %d" % (tc, mWaves)))
    module.add(VAndB32(dst=vgpr(ax), src0=vgpr(ax), src1=hex(perStrip - 1),
               comment="%s: axisId %% %d" % (tc, perStrip)))
    module.add(VLShiftLeftB32(dst=vgpr(ax), shiftHex=hex(kSplit.bit_length() - 1),
               src=vgpr(ax), comment="%s: * %d K slices" % (tc, kSplit)))
    module.add(VAddU32(dst=vgpr(dst), src0=vgpr(dst), src1=vgpr(ax),
               comment="%s: fetch-group index" % tc))
    writer.vgprPool.checkIn(ax)
  writer.vgprPool.checkIn(wid)
  return True


def _tluStripIdx(writer, kernel, module, tc, ti, dst):
  """Which free-dim strip this wave's fetch group owns, into dst.

  grWavesPerStrip axis waves share a strip; the axis waves past that own the
  next strip along the free dim.  Returns False when the free dim is a single
  strip, so callers can skip the step.
  """
  strips = int(ti.globalSubtileGrid[0])
  perStrip = max(1, int(getattr(ti, "grWavesPerStrip", 1)))
  if strips <= 1 or perStrip <= 1:
    return False
  assert perStrip & (perStrip - 1) == 0, \
         "grWavesPerStrip must be a power of two to shift, got %u" % perStrip
  if not _tluWaveAxisId(writer, kernel, module, tc, dst):
    return False
  module.add(VLShiftRightB32(dst=vgpr(dst), shiftHex=hex(perStrip.bit_length() - 1),
             src=vgpr(dst), comment="%s: strip = axisId / %u" % (tc, perStrip)))
  return True


def _tluWaveAxisGlobalOffset(writer, kernel, module, tileInfo):
  """VGPR holding this wave's free-dim (M/N) global byte offset, or None.

  Each axis-wave owns localSubtileGrid[0] strips of subtileM free-dim elements;
  the free dim is unit-stride, so the byte offset is
  axisId * localSub0 * subtileM * bpe.  Returns None when the axis does not
  split (single wave) so callers can skip the add.
  """
  tc = tileInfo.tc
  axisWaves = kernel["MIWaveGroup"][0] if tc == 'A' else kernel["MIWaveGroup"][1]
  wavesPerStrip = int(getattr(tileInfo, "grWavesPerStrip", 1))
  coopWaves = int(getattr(tileInfo, "grCoopWaves", 1))
  winSplit = int(getattr(tileInfo, "grKWindowSplit", 1))
  if axisWaves <= 1 and coopWaves <= 1 and winSplit <= 1:
    return None
  dst = writer.vgprPool.checkOut(1, tag="_tluWaveAxisGlobalOffset_%s" % tc)
  # Shared strips split by K across the fetching waves; whole-strip ownership
  # steps along the free dim by axis id.
  idOk = (_tluCoopWaveId(writer, kernel, module, tileInfo, dst) if wavesPerStrip > 1
           else _tluWaveAxisId(writer, kernel, module, tc, dst))
  if not idOk:
    # No free-dim step, but the other-axis waves may still be taking K slices of
    # the single strip this wave owns.
    writer.vgprPool.checkIn(dst)
    return _tluKSliceGlobalOffset(writer, kernel, module, tileInfo) if wavesPerStrip <= 1 else None
  if wavesPerStrip > 1:
    # Shared strip: the waves sharing one split its K rows, so wave a starts at
    # K row a*kRowsPerWave.  K is the strided dim for NT, so this needs the
    # runtime K stride.  A strip does not span the whole free dim when the tile
    # takes more than one, so the fetch groups past the first also step along it
    # -- that step is added after the K term, which is in bytes by then.
    # Units must match what the per-lane GR offset walks:
    #  - col_scatter: the load index IS the K column (col = col_group*N + L), so
    #    a wave owning numGRPerSubtile consecutive loads starts that many K
    #    columns in.
    #  - K ramp: the lane walks a chunk ramp at chunksPerK chunks per K row, so
    #    the wave's chunk base converts to whole K rows.
    if selectTLUColScatter(tileInfo) is not None:
      kRowsPerWave = int(tileInfo.numGRPerSubtile)
    else:
      kRows = int(tileInfo.mmaTileShape[1] * tileInfo.subtileShape[1])
      kRowsPerWave = kRows // coopWaves
    unrollIdx = kernel["ProblemType"]["IndexUnroll"]
    strideK = writer.strideRef(tc, unrollIdx)
    tmpS = writer.sgprPool.checkOut(1, tag="_tluWaveAxisKOffset_s_%s" % tc, preventOverflow=False)
    module.add(SMovB32(dst=sgpr(tmpS), src=hex(kRowsPerWave), comment="%s: K rows per wave" % tc))
    module.add(VMulLOU32(dst=vgpr(dst), src0=sgpr(tmpS), src1=vgpr(dst),
          comment="%s: wave K-row base = axisId*%d" % (tc, kRowsPerWave)))
    module.add(VMulLOU32(dst=vgpr(dst), src0=strideK, src1=vgpr(dst),
          comment="%s: K row * strideK (elements)" % tc))
    # bpe is per-operand: 0.5 for fp4 shifts right, but the bf16 TLU=1 geometry
    # has a bpe of 2 and has to shift left, so let the helper pick.
    module.add(vectorMultiplyBpe(dst, dst, float(tileInfo.bpe),
          comment="%s: elements -> bytes" % tc))
    writer.sgprPool.checkIn(tmpS)
    stripVgpr = writer.vgprPool.checkOut(1, tag="_tluWaveAxisStrip_%s" % tc)
    if _tluStripIdx(writer, kernel, module, tc, tileInfo, stripVgpr):
      subtileM = int(tileInfo.subtileShape[0] * tileInfo.mmaTileShape[0])
      stripBytes = int(subtileM * tileInfo.bpe)
      tmpS2 = writer.sgprPool.checkOut(1, tag="_tluWaveAxisStrip_s_%s" % tc,
                                       preventOverflow=False)
      module.add(SMovB32(dst=sgpr(tmpS2), src=hex(stripBytes),
            comment="%s: free-dim bytes per strip" % tc))
      module.add(VMulLOU32(dst=vgpr(stripVgpr), src0=sgpr(tmpS2), src1=vgpr(stripVgpr),
            comment="%s: strip * %u" % (tc, stripBytes)))
      module.add(VAddU32(dst=vgpr(dst), src0=vgpr(dst), src1=vgpr(stripVgpr),
            comment="%s: + free-dim strip offset" % tc))
      writer.sgprPool.checkIn(tmpS2)
    writer.vgprPool.checkIn(stripVgpr)
    return dst
  localSub0 = int(tileInfo.localSubtileGrid[0])
  subtileM = int(tileInfo.subtileShape[0] * tileInfo.mmaTileShape[0])
  strideBytes = int(localSub0 * subtileM * tileInfo.bpe)
  tmpS = writer.sgprPool.checkOut(1, tag="_tluWaveAxisGlobalOffset_s_%s" % tc, preventOverflow=False)
  module.add(SMovB32(dst=sgpr(tmpS), src=hex(strideBytes), comment="%s: free-dim wave stride" % tc))
  module.add(VMulLOU32(dst=vgpr(dst), src0=sgpr(tmpS), src1=vgpr(dst),
        comment="%s: wave free-dim global offset = axisId*%d" % (tc, strideBytes)))
  writer.sgprPool.checkIn(tmpS)
  # Whole-strip ownership plus a K split: the other-axis waves take a slice of
  # this strip's K rows on top of the free-dim step.
  kOff = _tluKSliceGlobalOffset(writer, kernel, module, tileInfo)
  if kOff is not None:
    module.add(VAddU32(dst=vgpr(dst), src0=vgpr(dst), src1=vgpr(kOff),
          comment="%s: + K-slice offset" % tc))
    writer.vgprPool.checkIn(kOff)
  return dst


def _tluKWaveSlots(tileInfo):
  """(kSplit, winSplit, rowsPerSlice, rowsPerWindowRun) for one tile's fetch group.

  A strip column is cut kSplit ways inside a single K window and winSplit ways
  across whole K windows.  Fetch-group index g decomposes as
  ``slice = g % kSplit`` and ``run = (g // kSplit) % winSplit``; the two row
  counts convert each of those to K rows.  The sId1 an emit sees is the FIRST
  window of a winSplit-sized group -- both the scheduler's grA.k and the
  globalReadDoSubtile loop step by winSplit -- so one run is one window.
  """
  coop = int(getattr(tileInfo, "grCoopWaves", 1))
  perStrip = max(1, int(getattr(tileInfo, "grWavesPerStrip", 1)))
  kSplit = max(1, coop // perStrip)
  winSplit = max(1, int(getattr(tileInfo, "grKWindowSplit", 1)))
  kRowsPerWindow = int(tileInfo.mmaTileShape[1] * tileInfo.subtileShape[1])
  if selectTLUColScatter(tileInfo) is not None:
    # col_scatter: the load index IS the K column (col = col_group*N + L), so a
    # wave owning numGRPerSubtile consecutive loads starts that many K rows in.
    rowsPerSlice = int(tileInfo.numGRPerSubtile)
  else:
    rowsPerSlice = kRowsPerWindow // kSplit
  return kSplit, winSplit, rowsPerSlice, kRowsPerWindow


def _tluOtherAxisId(writer, kernel, module, tc, dst):
  """Emit dst = this wave's index along the axis the operand does NOT depend on."""
  mWaves = kernel["MIWaveGroup"][0]
  wavesize = kernel["WavefrontSize"]
  module.add(VLShiftRightB32(dst=vgpr(dst), shiftHex=hex(wavesize.bit_length() - 1),
             src=vgpr("Serial"), comment="%s: waveId" % tc))
  if tc == 'A':
    module.add(VLShiftRightB32(dst=vgpr(dst), shiftHex=hex(mWaves.bit_length() - 1),
               src=vgpr(dst), comment="%s: otherId = waveId / %d" % (tc, mWaves)))
  else:
    module.add(VAndB32(dst=vgpr(dst), src0=vgpr(dst), src1=hex(mWaves - 1),
               comment="%s: otherId = waveId %% %d" % (tc, mWaves)))


def _tluKSliceTerms(writer, kernel, module, tileInfo, src, dst, sliceUnit, runUnit, tag):
  """dst = (src % kSplit)*sliceUnit + ((src // kSplit) % winSplit)*runUnit.

  ``src`` holds the fetch-group index (this wave's other-axis id).  Driven with
  K-row units for the global address and byte units for the LDS write address
  so the two stay in lock step.
  """
  tc = tileInfo.tc
  kSplit, winSplit, _, _ = _tluKWaveSlots(tileInfo)
  otherWaves = int(getattr(tileInfo, "grOtherAxisWaves", 1))
  tmpS = writer.sgprPool.checkOut(1, tag="%s_s_%s" % (tag, tc), preventOverflow=False)
  if kSplit > 1:
    if kSplit < otherWaves:
      # The group is narrower than the other-axis wave count, so several waves
      # land on the same slice -- a residual refetch the strip cannot avoid.
      module.add(VAndB32(dst=vgpr(dst), src0=vgpr(src), src1=hex(kSplit - 1),
                 comment="%s: K slice = index %% %d" % (tc, kSplit)))
    else:
      module.add(VMovB32(dst=vgpr(dst), src=vgpr(src), comment="%s: K slice" % tc))
    module.add(SMovB32(dst=sgpr(tmpS), src=hex(sliceUnit), comment="%s: per K slice" % tc))
    module.add(VMulLOU32(dst=vgpr(dst), src0=sgpr(tmpS), src1=vgpr(dst),
               comment="%s: K-slice base = slice*%d" % (tc, sliceUnit)))
  else:
    module.add(VMovB32(dst=vgpr(dst), src=0, comment="%s: no K slice within a window" % tc))
  if winSplit > 1:
    run = writer.vgprPool.checkOut(1, tag="%s_run_%s" % (tag, tc))
    module.add(VLShiftRightB32(dst=vgpr(run), shiftHex=hex(kSplit.bit_length() - 1),
               src=vgpr(src), comment="%s: drop the K-slice bits" % tc))
    module.add(VAndB32(dst=vgpr(run), src0=vgpr(run), src1=hex(winSplit - 1),
               comment="%s: K-window run = index / %d %% %d" % (tc, kSplit, winSplit)))
    module.add(SMovB32(dst=sgpr(tmpS), src=hex(runUnit), comment="%s: per K-window run" % tc))
    module.add(VMulLOU32(dst=vgpr(run), src0=sgpr(tmpS), src1=vgpr(run),
               comment="%s: K-window base = run*%d" % (tc, runUnit)))
    module.add(VAddU32(dst=vgpr(dst), src0=vgpr(dst), src1=vgpr(run),
               comment="%s: + K-window run" % tc))
    writer.vgprPool.checkIn(run)
  writer.sgprPool.checkIn(tmpS)


def _tluKSliceGlobalOffset(writer, kernel, module, tileInfo):
  """Global byte offset of this wave's K slice within the strip column it fetches.

  Returns None when the column is not K-split across the other-axis waves.
  """
  tc = tileInfo.tc
  kSplit, winSplit, rowsPerSlice, rowsPerRun = _tluKWaveSlots(tileInfo)
  if kSplit <= 1 and winSplit <= 1:
    return None
  dst = writer.vgprPool.checkOut(1, tag="_tluKSlice_%s" % tc)
  other = writer.vgprPool.checkOut(1, tag="_tluKSliceOther_%s" % tc)
  _tluOtherAxisId(writer, kernel, module, tc, other)
  _tluKSliceTerms(writer, kernel, module, tileInfo, other, dst,
                  rowsPerSlice, rowsPerRun, "_tluKSlice")
  writer.vgprPool.checkIn(other)
  unrollIdx = kernel["ProblemType"]["IndexUnroll"]
  strideK = writer.strideRef(tc, unrollIdx)
  module.add(VMulLOU32(dst=vgpr(dst), src0=strideK, src1=vgpr(dst),
        comment="%s: K row * strideK (elements)" % tc))
  module.add(vectorMultiplyBpe(dst, dst, float(tileInfo.bpe),
        comment="%s: elements -> bytes" % tc))
  return dst


def _grComputeSubtileOffsets_tlu(writer, module, tileInfo):
  """Fill per-subtile-row global-M soffset registers for TLU=1 (NT).

  Each subtile strip covers ``subtileM = subtileShape[0]*instM`` free-dim rows.
  For NT the free dim (M for A, N for B) is unit-stride, so the global byte
  offset between adjacent strips is ``subtileM * bpe``.  Row 0 needs no soffset
  (RegList empty); row r>=1 gets r*subtileM*bpe.  These feed the ``soffset``
  field of the per-strip buffer_load in emitSingleBufferLoad.
  """
  tc = tileInfo.tc
  subtileM = int(tileInfo.subtileShape[0] * tileInfo.mmaTileShape[0])
  strideBytes = int(subtileM * tileInfo.bpe)
  for regId in range(len(tileInfo.localSubtilesRegister)):
    rl = tileInfo.localSubtilesRegister[regId]
    if len(rl) == 0:
      continue  # row 0: soffset stays 0
    off = strideBytes * regId
    if rl.is_sgpr:
      module.add(SMovB32(dst=rl.ref(0), src=hex(off),
                 comment="%s: subtile row %u soffset = %u*subtileM*bpe" % (tc, regId, regId)))
    else:
      # VGPR fallback: bake soffset into each per-load offset VGPR.
      for i, reg in enumerate(rl):
        module.add(VAddU32(dst=vgpr(reg), src0=vgpr(tileInfo.sharedVgprGROffset[i]),
                   src1=hex(off), comment="%s: subtile row %u offset (vgpr)" % (tc, regId)))
  return module
##################################################
# Subroutine to generate GR load code
#
def emitSingleBufferLoad(tileInfo, kernel, sId0, sId1, writer=None):
  """Emit buffer_load instructions for a single subtile (sId0, sId1).

  When loadRatioGR > 1, multiple local subtiles share the same global read.
  Only the first subtile in each group emits the load; others return empty.

  Args:
      tileInfo: TileInfo or TileInfo for the tensor component
      sId0:     Subtile row index
      sId1:     Subtile column index (K-dimension)
      writer:   KernelWriter (needed on the TLU=1 path to look up the runtime
                K stride for K-window (sId1>0) global-address advances).
  """
  module = Module()

  # TDM path: emit one tensor_load_to_lds per tensor, skip all per-subtile DTL loads
  if kernel.get("enableTDM%s" % tileInfo.tc[0], False):
    if sId0 == 0 and sId1 == 0:
      tc = tileInfo.tc
      group0 = "tdm%sGroup0" % tc
      group1 = "tdm%sGroup1" % tc
      isSubtileIter = _isSubtileIterateMode(kernel, tc)
      group2 = sgpr("tdm%sGroup2" % tc, 4) if isSubtileIter else None
      group3 = sgpr("tdm%sGroup3" % tc, 4) if isSubtileIter else None
      module.add(TensorLoadToLds(sgpr(group0, 4), sgpr(group1, 8), group2, group3,
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

  # For TLU=1 the scheduler yields sId0 in MMA-tile units (steps by
  # subtileShape[0]); convert to a subtile-row index so multi-strip macro tiles
  # (MT > one subtile) address the right LDS strip and soffset group.
  isTLU1 = bool(tileInfo.gr and isinstance(tileInfo.gr.config.tag, GRTag_TLU1))
  stackM = int(tileInfo.subtileShape[0])
  subtileRow = (sId0 // stackM) if isTLU1 else sId0

  regListIdx = tileInfo.grRegGroupForSubtileRow(subtileRow)
  regList = tileInfo.localSubtilesRegister[regListIdx]
  useSgpr = regList.is_sgpr

  offsetK = sId1 * int(tileInfo.mmaTileShape[1] * tileInfo.subtileShape[1] * tileInfo.bpe)

  subtileOffset = int(math.ceil(tileInfo.loadRatioGR*tileInfo.subtileSize))
  # loadRatioGR folds in grLoadWaves, so subtileOffset is the bytes one
  # *cooperative* load round covers.  Waves that share a strip are separated
  # within it by whole load-blocks (each takes a slice of the strip's
  # K rows) rather than interleaved within one block, so a wave still advances
  # m0 by its own single-wave block, not by the cooperative total.
  coopWaves = int(getattr(tileInfo, "grCoopWaves", 1))
  if isTLU1 and coopWaves > 1:
    # TLU=1 only: loadRatioGR folds in every cooperating wave, so undo it to get
    # the bytes a single wave's own load block covers.  TLU=0 keeps the
    # cooperative stride, which is what its m0 walk expects.
    subtileOffset = int(subtileOffset // coopWaves)
  WriteBaseAddr = "LocalWriteBaseAddr%s"%tc
  padBytes = tluPadBytes(tileInfo)
  # Pad-aware LDS stride between subtile strips (M/N direction).
  stripStride = stripStrideBytes(tileInfo) if isTLU1 else int(tileInfo.subtileSize)
  # K-window (sId1) global-address advance for TLU=1.
  #
  # For NT the free (M/N) dim is unit-stride and the K (unroll) dim is strided,
  # so stepping to K-window sId1 must move the GLOBAL read address by
  # sId1*instK*stackK*strideK*bpe bytes.  The unit-stride byte count for that
  # step is exactly offsetK; scaling it by the runtime K stride gives the
  # strided global offset.  This CANNOT be encoded via the DTL offset12
  # immediate (that shifts the LDS m0 write, not the global fetch), so it is
  # folded into soffset instead.  The LDS placement of the window is already
  # carried by the (sId1 * globalSubtileGrid[0]) * stripStride term in m0Offset.
  baseSoffset = regList.ref(0) if len(regList) > 0 and useSgpr else 0
  kWindowSoffset = None
  if isTLU1 and sId1 > 0:
    assert writer is not None, "TLU=1 K-window GR load needs writer for strideK"
    unrollIdx = kernel["ProblemType"]["IndexUnroll"]
    strideK = writer.strideRef(tc, unrollIdx)
    kWindowSoffset = writer.sgprPool.checkOut(1, tag="grKWindowSoffset%s" % tc,
                                              preventOverflow=False)
    if writer.isConstUnitStride(strideK):
      module.add(SMovB32(dst=sgpr(kWindowSoffset), src=hex(offsetK),
                 comment="%s: K-window %u global byte offset (unit stride)" % (tc, sId1)))
    else:
      module.add(SMulI32(dst=sgpr(kWindowSoffset), src0=offsetK, src1=strideK,
                 comment="%s: K-window %u global offset = offsetK*strideK" % (tc, sId1)))
    if baseSoffset != 0:
      module.add(SAddU32(dst=sgpr(kWindowSoffset), src0=sgpr(kWindowSoffset),
                 src1=baseSoffset, comment="%s: + subtile-row M soffset" % tc))
  for i in range(tileInfo.numGRPerSubtile):
    if isTLU1:
      m0Offset = int(i * subtileOffset
                     + (subtileRow + sId1 * tileInfo.globalSubtileGrid[0]) * stripStride)
    else:
      m0Offset = int(i * subtileOffset + (sId0 + sId1 * tileInfo.globalSubtileGrid[0]) * tileInfo.subtileSize)
    # Bank-conflict swizzle / col_scatter: insert padBytes between load-blocks so
    # each GR load i (one wavesize-chunk block) shifts by i*padBytes.  The LR
    # read applies the same pad. See SubtileTLUSwizzle.
    if padBytes:
      m0Offset += i * padBytes
    if isTLU1:
      # LDS m0 already carries the window placement; the K-window global step
      # rides on soffset, so no offset12 immediate is needed here.
      module.add(SAddU32(dst=mgpr(0), src0=sgpr(WriteBaseAddr), src1=m0Offset))
      mubuf = MUBUFModifiers(offen=True, glc=isGlc, slc=isSlc, nt=isNT, lds=True)
      soffset = sgpr(kWindowSoffset) if kWindowSoffset is not None else baseSoffset
    else:
      module.add(SAddU32(dst=mgpr(0), src0=sgpr(WriteBaseAddr), src1=(m0Offset - offsetK)))
      mubuf = MUBUFModifiers(offen=True, offset12=offsetK, glc=isGlc, slc=isSlc, nt=isNT, lds=True)
      soffset = baseSoffset
    voff = tileInfo.sharedVgprGROffset[i] if useSgpr or len(regList) == 0 else regList.indices[i]
    module.add(BufferLoadB128(dst=None, vaddr=vgpr(voff), saddr=sgpr("Srd%s"%tc, 4), soffset=soffset, mubuf=mubuf, comment="grBaseId = %u, i= %u"%(grBaseId , i)))

  if kWindowSoffset is not None:
    writer.sgprPool.checkIn(kWindowSoffset)

  return module


def emitSubtileBufferLoad(tc, writer, kernel, subtileId):
  tileInfo = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
  return emitSingleBufferLoad(tileInfo, kernel, subtileId[0], subtileId[1], writer=writer)

##################################################
# Subroutine to generate GR load code
# Initial idea: maybe store asm in modules in a separate obj?
#
def globalReadDoSubtile(tc, writer, kernel):
  module = Module()

  tileInfo = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo

  # A K-window split hands each wave every grKWindowSplit'th run of windows, so
  # the loop issues one window per run and the runtime base picks the run.
  winSplit = int(getattr(tileInfo, "grKWindowSplit", 1))
  for j in range(0, int(tileInfo.localSubtileGrid[1]), winSplit):
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
  aTLU1 = _isGRTLU1(tileInfoA)
  bTLU1 = _isGRTLU1(tileInfoB)
  if aTLU1 and bTLU1:
    return _globalReadDTLInitCommonSgpr_tlu(writer, kernel, module, tileInfoA, tileInfoB)
  if aTLU1 or bTLU1:
    # NN / TT: each tensor's DTL write base comes from its own layout.
    module.addComment0("Mixed TLU: per-operand DTL write base")
    for tc, ti, isTLU1 in (("A", tileInfoA, aTLU1), ("B", tileInfoB, bTLU1)):
      if isTLU1:
        _grDTLInitBase_tlu(writer, kernel, module, tc, ti)
      else:
        _grDTLInitBase_rowMajor(writer, kernel, module, tc, ti)
      _grDTLInitSwap(writer, module, tc)
    return module
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


def _globalReadDTLInitCommonSgpr_tlu(writer, kernel, module, tileInfoA, tileInfoB):
  """LocalWriteBaseAddr + Swap for TLU=1 (NT), including multi-wave partition.

  LDS holds the full macro tile (all subtile strips).  Each axis-wave owns
  localSubtileGrid[0] consecutive strips, so its DTL write base is
  axisId * localSub0 * stripStride bytes into the tensor's LDS region.  Single
  axis-wave -> base 0 (+ ldsStartOffset for B).
  """
  module.addComment0("TLU: per-wave DTL write base (M/N strip partition)")
  for tc, ti in (("A", tileInfoA), ("B", tileInfoB)):
    _grDTLInitBase_tlu(writer, kernel, module, tc, ti)
    _grDTLInitSwap(writer, module, tc)
  return module


def _grDTLInitBase_tlu(writer, kernel, module, tc, ti):
  """LocalWriteBaseAddr for one TLU=1 tensor (multi-wave strip partition)."""
  base = "LocalWriteBaseAddr%s" % tc
  axisWaves = kernel["MIWaveGroup"][0] if tc == 'A' else kernel["MIWaveGroup"][1]
  stripStride = stripStrideBytes(ti)
  localSub0 = int(ti.localSubtileGrid[0])
  wavesPerStrip = int(getattr(ti, "grWavesPerStrip", 1))
  coopWaves = int(getattr(ti, "grCoopWaves", 1))
  if wavesPerStrip > 1:
    # Shared strip: the waves sharing one write into it at a contiguous run of
    # DTL load-blocks each (its share of the strip's K rows).  numGRPerSubtile is
    # already the per-wave load count, and one load block occupies
    # wavesize*loadWidth bytes plus the swizzle pad.  Groups owning a later strip
    # add its stride below, matching the free-dim step on the global side.
    blkBytes = grLoadBlockBytes(kernel["WavefrontSize"], ti)
    perWaveBytes = int(ti.numGRPerSubtile * blkBytes)
  else:
    perWaveBytes = int(localSub0 * stripStride)
  winSplit = int(getattr(ti, "grKWindowSplit", 1))
  # The write base must be keyed the same way as the global offset: by the
  # fetching-wave index for shared strips, by the axis id otherwise.
  if (coopWaves > 1) if wavesPerStrip > 1 else (axisWaves > 1 or coopWaves > 1 or winSplit > 1):
    wv = writer.vgprPool.checkOut(1, tag="_grDTLInit_tlu_wave_%s" % tc)
    if wavesPerStrip > 1:
      _tluCoopWaveId(writer, kernel, module, ti, wv)
    elif not _tluWaveAxisId(writer, kernel, module, tc, wv):
      module.add(VMovB32(dst=vgpr(wv), src=0, comment="%s: single axis-wave" % tc))
    tmpS = writer.sgprPool.checkOut(1, tag="_grDTLInit_tlu_s_%s" % tc, preventOverflow=False)
    module.add(SMovB32(dst=sgpr(tmpS), src=hex(perWaveBytes), comment="%s: LDS wave stride" % tc))
    module.add(VMulLOU32(dst=vgpr(wv), src0=sgpr(tmpS), src1=vgpr(wv),
               comment="%s: wave LDS base = axisId*%d" % (tc, perWaveBytes)))
    writer.sgprPool.checkIn(tmpS)
    if wavesPerStrip > 1:
      st = writer.vgprPool.checkOut(1, tag="_grDTLInit_tlu_strip_%s" % tc)
      if _tluStripIdx(writer, kernel, module, tc, ti, st):
        tmpS2 = writer.sgprPool.checkOut(1, tag="_grDTLInit_tlu_strip_s_%s" % tc,
                                         preventOverflow=False)
        module.add(SMovB32(dst=sgpr(tmpS2), src=hex(int(stripStride)),
                   comment="%s: LDS bytes per strip" % tc))
        module.add(VMulLOU32(dst=vgpr(st), src0=sgpr(tmpS2), src1=vgpr(st),
                   comment="%s: strip * %u" % (tc, int(stripStride))))
        module.add(VAddU32(dst=vgpr(wv), src0=vgpr(wv), src1=vgpr(st),
                   comment="%s: + strip LDS offset" % tc))
        writer.sgprPool.checkIn(tmpS2)
      writer.vgprPool.checkIn(st)
    else:
      _grDTLAddKSlice(writer, kernel, module, tc, ti, wv)
    module.add(SNop(waitState=0, comment="wait for VGPR"))
    module.add(VReadfirstlaneB32(dst=sgpr(base), src=vgpr(wv),
               comment="%s: per-wave DTL write base" % tc))
    writer.vgprPool.checkIn(wv)
  else:
    module.add(SMovB32(dst=sgpr(base), src=0, comment="%s: single axis-wave, base 0" % tc))


def _grDTLInitBase_rowMajor(writer, kernel, module, tc, ti):
  """LocalWriteBaseAddr for one TLU=0 tensor."""
  base = "LocalWriteBaseAddr%s" % tc
  wavesize = kernel["WavefrontSize"]
  vgprWaveId = writer.vgprPool.checkOut(1, tag="_grDTLInit_rm_wave_%s" % tc)
  rowOffset = writer.vgprPool.checkOut(1, tag="_grDTLInit_rm_row_%s" % tc)
  module.add(VLShiftRightB32(dst=vgpr(vgprWaveId), shiftHex=hex(wavesize.bit_length()-1),
             src=vgpr("Serial"), comment="Wave Id"))
  _grComputeRowPartition_legacy(module, kernel, writer, ti, vgprWaveId, rowOffset)
  module.add(VLShiftLeftB32(dst=vgpr(rowOffset), shiftHex=hex(ti.subIterKBytes.bit_length()-1),
             src=vgpr(rowOffset), comment="Apply wave-specific offset for %s" % tc))
  module.add(SNop(waitState=0, comment="Wait for VGPR to be ready"))
  module.add(VReadfirstlaneB32(dst=sgpr(base), src=vgpr(rowOffset),
             comment="Store base LDS offset, will be modified"))
  writer.vgprPool.checkIn(vgprWaveId)
  writer.vgprPool.checkIn(rowOffset)


def _grDTLInitSwap(writer, module, tc):
  """Fold in the tensor's LDS start offset and build its double-buffer Swap mask."""
  base = "LocalWriteBaseAddr%s" % tc
  if tc == 'B' and writer.ldsStartOffsetB:
    module.add(SAddU32(dst=sgpr(base), src0=sgpr(base), src1=hex(writer.ldsStartOffsetB),
               comment="B: + ldsStartOffset"))
  swap = "Swap%s" % tc
  module.add(SAddU32(dst=sgpr(swap), src0=sgpr(base), src1=writer.ldsTotalSize, comment=""))
  module.add(SXorB32(dst=sgpr(swap), src0=sgpr(base), src1=sgpr(swap), comment=""))

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

  with writer.allocTmpSgpr(max(3, writer.states.laneSGPRCount)) as tmpSgprRes:
    tmp = tmpSgprRes.idx
    waveOff = tmpSgprRes.idx + 2

    # Load tc[batch] into Address{tc} before tile and strided-batch offsets.
    mod.add(writer._resolveTDMGlobalAddr(
        kernel, tc, f"Address{tc}", tmpSgprRes))

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
      batchIdx = sgpr("WorkGroup2")
      if kernel["ProblemType"]["SupportUserArgs"] and tc in ("A", "B"):
        writer.cmpNamedArgTypeEq(
            mod, 3, "ArgType == 3 for General Batched GEMM")
        mod.add(SCSelectB32(
            dst=sgpr(waveOff),
            src0=0,
            src1=batchIdx,
            comment="general batch uses an already-dereferenced matrix base"))
        batchIdx = sgpr(waveOff)
      mod.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(tmp), sgpr(tmp+1),
                                                     sgpr(batchStrideName), batchIdx,
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
    maxIdx = 4 if isSubtileIter else 2
    assert idx < maxIdx, f"descSgprName({idx}) out of range (iterate={isSubtileIter})"
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
  isSubtileIter = _isSubtileIterateMode(kernel, tc)

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
  # OR the per-tensor broadcast mask into the descriptor for TDM multicast.
  # Subtile loads both A and B on every wave, so it uses split masks
  # (MulticastMask{tc}), not the non-subtile single parity mask.
  from ...Components.ClusterLoad import ClusterLoadTDM
  clusterComp = ClusterLoadTDM.find(writer)
  if clusterComp:
    mod.add(clusterComp.applyToDescriptor(writer, kernel, descSgprName(1), tc, subtile=True))

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
  if isSubtileIter:
    mod.add(comp.setPadding(descSgprName(1), 0, 0))
  else:
    mod.add(comp.setPadding(descSgprName(1), padIntervalBytes, padAmountBytes))
  mod.add(comp.setTensorDim0(descSgprName(1), sizeRefName(3), writer, sizeShifterDim))
  mod.add(comp.setTensorDim1(descSgprName(1), sizeRefName(ti), writer))

  sizeShifterTile = sizeShifter
  mod.add(comp.setTensorTile0(descSgprName(1), sizeTile0, writer, sizeShifterTile))

  if isSubtileIter:
    # Iterate mode: one row per iteration.
    mod.add(comp.setTensorTile1(descSgprName(1), 1, writer))
  else:
    # Clamp each wave's Tile1 (free-dim-1) load extent to the valid remainder.
    # tdmGlobalOffsetSubtile bases wave w at row w*(mt//numWaves), but the
    # descriptor Dim1 does not bound the walk, so an edge tile (free dim < mt)
    # reads past the tensor. setTensorTile1 takes a compile-time int, so write its
    # field (+4[15:0]) with a runtime clamp. No-op when the tile fits.
    perWaveRows = sizeTile1 // numWaves
    if numWaves > 1:
      with writer.allocTmpSgpr(2) as tileClampRes:
        validRows = tileClampRes.idx
        waveRowStart = tileClampRes.idx + 1
        mod.add(VReadfirstlaneB32(sgpr(waveRowStart), vgpr("Serial"), "first tId"))
        mod.add(SLShiftRightB32(sgpr(waveRowStart), ceil(log2(wavelen)), sgpr(waveRowStart),
                "wId = fTid // wavelen"))
        mod.add(SMulI32(sgpr(waveRowStart), sgpr(waveRowStart), perWaveRows,
                f"waveGlobalRowStart = wId * {perWaveRows}"))
        mod.add(SSubI32(dst=sgpr(validRows), src0=sgpr(sizeRefName(ti)), src1=sgpr(waveRowStart),
                comment="Size_free - waveGlobalRowStart"))
        mod.add(SMaxI32(dst=sgpr(validRows), src0=sgpr(validRows), src1=0,
                comment="saturate negative remainder to 0"))
        mod.add(SMinU32(dst=sgpr(validRows), src0=sgpr(validRows), src1=perWaveRows,
                comment=f"clamp to per-wave rows ({perWaveRows})"))
        mod.add(SAndB32(sgpr(f"{descSgprName(1)}+4"), sgpr(f"{descSgprName(1)}+4"),
                hex(0xFFFF0000), "clear tile1 field"))
        mod.add(SOrB32(sgpr(f"{descSgprName(1)}+4"), sgpr(f"{descSgprName(1)}+4"),
                sgpr(validRows), "set tile1 = clamped validRows"))
    else:
      mod.add(comp.setTensorTile1(descSgprName(1), perWaveRows, writer))
  mod.add(comp.setTensorStride0(descSgprName(1), strideRefName(), sizeShifterTile))

  if isSubtileIter:
    dss = comp.dataSizeShift(dtype)
    lds_inc = (padIntervalBytes + padAmountBytes) >> dss
    iter_count = sizeTile1 // numWaves
    mod.add(comp.setIterationEnabled(descSgprName(1), True))
    with writer.allocTmpSgpr(2) as tmp:
      sIter, sGInc = tmp.idx, tmp.idx + 1
      mod.add(SMovB32(sgpr(sGInc), sgpr(strideRefName()), "global_inc = stride"))
      if dtype.isFloat4():
        mod.add(SLShiftRightB32(sgpr(sGInc), 1, sgpr(sGInc),
                                "fp4 sub-byte: global_inc bytes = elements / 2"))
      mod.add(comp.setIterationIncrements(descSgprName(2), lds_inc, sGInc))
      mod.add(SMovB32(sgpr(sIter), hex(iter_count - 1), f"iter_count={iter_count}-1"))
      mod.add(comp.setIterations(descSgprName(2), sIter))

  return mod


def tdmApplyStreamKOffsetSubtile(writer, kernel, tP):
  """Apply the StreamK K-offset to the subtile TDM descriptor.

  StreamK=3 DP-partial work items have a nonzero StreamKLocalStart and must read
  their own K-slice. Advance Address{tc} by StreamKLocalStart unroll iterations
  (inc = ti.depthUBytes, matching _emitGRPtrUpdate_TLU0's per-iteration advance)
  and re-sync the descriptor. No-op when StreamKLocalStart == 0.
  """
  tc = tP["tensorChar"]
  ti = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
  inc = int(ti.depthUBytes)  # per-unroll TDM advance; same source as _emitGRPtrUpdate_TLU0
  group0 = f"tdm{tc}Group0"
  mod = Module(f"TDM StreamK K-offset subtile {tc}")
  # DP-only: StreamKLocalStart == 0, so the K-start offset is 0 and this is a
  # no-op. StreamKLocalStart is not allocated in DP-only mode.
  if kernel["StreamKForceDPOnly"]:
    return mod
  with writer.allocTmpSgpr(2, alignment=2, tag="tdmSkOffset") as tmpSgprRes:
    o = tmpSgprRes.idx
    mod.add(SMulI32(dst=sgpr(o), src0=sgpr("StreamKLocalStart"), src1=inc,
                    comment=f"SK K-start * depthU*bpe ({inc})"))
    mod.add(SMovB32(dst=sgpr(o + 1), src=0, comment="SK K-start offset hi = 0"))
    mod.add(SAddU32(dst=sgpr(f"Address{tc}+0"), src0=sgpr(f"Address{tc}+0"), src1=sgpr(o),
                    comment="Address += SK K-start offset (lo)"))
    mod.add(SAddCU32(dst=sgpr(f"Address{tc}+1"), src0=sgpr(f"Address{tc}+1"), src1=sgpr(o + 1),
                     comment="Address += SK K-start offset (hi, carry)"))
  mod.add(SMovB64(dst=sgpr(f"{group0}+2", 2), src=sgpr(f"Address{tc}", 2),
                  comment="sync descriptor global addr"))
  mod.add(SOrB32(dst=sgpr(f"{group0}+3"), src0=sgpr(f"{group0}+3"), src1=hex(2 << 30),
                 comment="restore descriptor type field"))
  return mod

##################################################
# Subroutine to update ptrs
#
def globalReadPtrUpdates(tc, writer, kernel):
  ti_ = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
  return ti_.emitGRPtrUpdate(writer, kernel)


def _grDTLAddKSlice(writer, kernel, module, tc, ti, dstVgpr):
  """Add this wave's K-slice LDS byte offset within its strip column to dstVgpr.

  Mirrors the K term _tluKSliceGlobalOffset adds to the global address, so the
  DTL write lands where the read expects it.
  """
  kSplit, winSplit, _, _ = _tluKWaveSlots(ti)
  if kSplit <= 1 and winSplit <= 1:
    return
  blkBytes = grLoadBlockBytes(kernel["WavefrontSize"], ti)
  sliceBytes = int(ti.numGRPerSubtile * blkBytes)
  # A K window sits one whole strip column of the macro tile further into LDS
  # (the same term emitSingleBufferLoad applies statically for sId1 > 0).
  runBytes = int(int(ti.globalSubtileGrid[0]) * stripStrideBytes(ti))
  o = writer.vgprPool.checkOut(1, tag="_grDTLKSlice_%s" % tc)
  other = writer.vgprPool.checkOut(1, tag="_grDTLKSliceOther_%s" % tc)
  _tluOtherAxisId(writer, kernel, module, tc, other)
  _tluKSliceTerms(writer, kernel, module, ti, other, o, sliceBytes, runBytes, "_grDTLKSlice")
  writer.vgprPool.checkIn(other)
  module.add(VAddU32(dst=vgpr(dstVgpr), src0=vgpr(dstVgpr), src1=vgpr(o),
             comment="%s: + K slice within the strip column" % tc))
  writer.vgprPool.checkIn(o)
