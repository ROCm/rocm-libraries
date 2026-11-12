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

import contextlib
from ...ExecutionPolicy import isPersistentDataParallel
from ...Common.Utilities import SWIZZLE_DTL_OFFSET12_MAX, swizzleTensorMaxDepthU
import math
from functools import singledispatch

from rocisa.code import Module, Label
from rocisa.container import DPPModifiers, EXEC, MUBUFModifiers, VCC, vgpr, sgpr, mgpr
from rocisa.enum import RegisterType
from rocisa.instruction import (
    BufferLoadB128,
    SAddCU32, SAddU32, SAddU64, SAndB32, SMaxI32, SMinU32, SMovB32, SMovB64, SMulI32,
    SNop, SOrB32, SSubI32, SSubU32, SXorB32,
    SCBranchSCC0, SCBranchSCC1, SCmpEQU32, SCmpGeU32, SCmpLeU32, SCSelectB32, SEndpgm,
    SLShiftLeftB32, SLShiftLeftB64, SLShiftRightB32,
    VAddU32, VAndB32, VCmpLtI32, VCmpXEqU32, VCndMaskB32,
    VLShiftLeftB32, VLShiftRightB32, VMovB32, VOrB32,
    TensorLoadToLds,
    VMulLOU32, VReadfirstlaneB32, VSubU32, VXorB32,
)
from .LdsV7Layout import (
    HALF_ROWS, ISSUER_A, ISSUER_B, ISSUER_MXSA, ISSUER_MXSB,
)

from .SubtileGeometry import (
    RegList,
    GRTag_1x1, GRTag_1x2, GRTag_2x2, GRTag_TLU1,
)
from .SubtileScaleEmit import emitScaleGRLDSSwap
from .SubtileTLUSwizzle import (selectTLUSwizzle, selectTLUColScatter, stripStrideBytes,
                                tluPadBytes, grLoadBlockBytes)

from math import ceil, log, log2, prod
from rocisa.functions import vectorMultiplyBpe
from ...Common import INDEX_CHARS
from ...SolutionStructs.Utilities import isSubtileIterateMode as _isSubtileIterateMode
from ...Common.Utilities import isMxf4SubtilePath
from ...Common.DataType import DataType


def _ldsRowBankSize(writer) -> int:
  """Bytes spanned by one full row of LDS banks."""
  archCaps = writer.states.archCaps
  return archCaps["LDSBankCount"] * archCaps["LDSBankWidth"]


def _grRowTiling(writer, tileInfo):
  """(loads per subIterK row, subIterK rows per LDS bank row) for a row-major operand.

  A subIterK row has to divide into whole loads and fit inside one bank row.
  Both hold for every geometry Solution.py admits.
  """
  subIterKBytes = tileInfo.subIterKBytes
  ldsRowBankSize = _ldsRowBankSize(writer)
  assert subIterKBytes % tileInfo.loadWidthGR == 0
  assert subIterKBytes <= ldsRowBankSize
  return subIterKBytes // tileInfo.loadWidthGR, ldsRowBankSize // subIterKBytes


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
def _emitGRPtrUpdate(tag, tile, ti, writer, kernel, holdOnLastIter=False):
  raise NotImplementedError(f"emitGRPtrUpdate not implemented for {type(tag).__name__}")

# Stubs for tags not yet implemented.
_stub = lambda *args, **kwargs: None


def emitSrdAdvance(module, tc, inc, writer, kernel=None, holdOnLastIter=False):
  """Advance Srd{tc}'s base pointer by inc bytes.

  holdOnLastIter suppresses the advance when the K-block that the PGR=2 pre-loop's
  second prefetch cluster would read does not exist at all. It is set only for the
  pre-loop advance that sits between the two clusters. When the whole summation
  fits in one DepthU the pre-loop branches straight to the NLL, which reads only
  cluster 1's LDS buffer, so cluster 2 prefetches a K-block past the end of the
  tensor. Its loads still have to issue -- the pre-loop's vmcnt(N) counts on them
  -- so instead of skipping them we leave the SRD in place and let them re-read
  cluster 1's K-block. Srd+2 is a fixed tile-boundary limit rather than a distance
  to the end of the tensor, so nothing else would clip them and they run off the
  end of the allocation (hipErrorIllegalAddress on the MX scale tensors, whose
  allocations are small enough to end on a mapping boundary).

  The hold must not fire when a K-tail follows a single unrolled iteration. There
  cluster 2's K-block is the tail's own block: it exists (partially), and the tail
  loop reads it through this same SRD, so holding makes the tail re-MAC K-block 0.
  The condition is therefore "hold iff sizeL <= DepthU", expressed as
  LoopCounterL <= (0 if the tail runs else 1). The tail test mirrors
  calculateLoopNumIter's own SizesSum % DepthU.
  """
  if not holdOnLastIter or not isMxf4SubtilePath(kernel):
    module.add(SAddU32(dst=sgpr(f"Srd{tc}"), src0=sgpr(f"Srd{tc}"), src1=inc,
                       comment=f"{tc}: advance SRD by {inc} bytes"))
    module.add(SAddCU32(dst=sgpr(f"Srd{tc}+1"), src0=sgpr(f"Srd{tc}+1"), src1=0,
                        comment=f"{tc}: carry"))
    return

  depthU = kernel["DepthU"]
  assert (depthU & (depthU - 1)) == 0, "PGR=2 pre-loop SRD hold assumes DepthU is a power of two"
  tmp = writer.sgprPool.checkOut(1, tag=f"emitSrdAdvance_{tc}", preventOverflow=False)
  module.add(SAndB32(dst=sgpr(tmp), src0=sgpr("SizesSum+%u" % writer.states.unrollIdx),
                     src1=depthU - 1,
                     comment="tail = SizesSum %% DepthU (0 => NLL is terminal)"))
  module.add(SCmpEQU32(src0=sgpr(tmp), src1=0, comment="no tail loop?"))
  module.add(SCSelectB32(dst=sgpr(tmp), src0=1, src1=0,
                         comment="hold threshold: 1 without a tail, 0 with one"))
  module.add(SCmpLeU32(src0=sgpr("LoopCounterL"), src1=sgpr(tmp),
                       comment="sizeL <= DepthU? (cluster 2 has no K-block to read)"))
  module.add(SCSelectB32(dst=sgpr(tmp), src0=0, src1=inc,
                         comment=f"{tc}: hold SRD rather than read past the tensor"))
  module.add(SAddU32(dst=sgpr(f"Srd{tc}"), src0=sgpr(f"Srd{tc}"), src1=sgpr(tmp),
                     comment=f"{tc}: advance SRD by {inc} bytes"))
  module.add(SAddCU32(dst=sgpr(f"Srd{tc}+1"), src0=sgpr(f"Srd{tc}+1"), src1=0,
                      comment=f"{tc}: carry"))
  writer.sgprPool.checkIn(tmp)
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
  gran = tile.localGRGranularity(ti.grLoadWaves)
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
      if ti.isPreShuffled:
        offsetK = _preShuffledKStepBytes(ti, offsetK)

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
def _emitGRPtrUpdate_TLU0(tag, tile, ti, writer, kernel, holdOnLastIter=False):
  """Advance SRD base pointer by one depthU iteration (depthU * bpe bytes)."""
  tc = ti.tc
  # TDM path: advance Address{tc} and sync the TDM descriptor instead of SRD.
  if kernel.get("enableTDM%s" % tc, False):
    # Address{tc} is copied into the descriptor by refresh/patch before each load.
    module = Module(f"TDM GR Ptr Update ({tc})")
    inc = int(ti.depthUBytes)
    module.addComment0("TDM addr update: %s += %u" % (tc, inc))
    module.add(SAddU64(dst=sgpr("Address%s" % tc, 2), src0=sgpr("Address%s" % tc, 2), src1=inc))
    return module

  module = Module(f"GR Ptr Update ({tc})")
  # Host-pre-shuffled data interleaves one K tile across all rows of an MFMA
  # tile, so advancing one DepthU consumes 16 contiguous row fragments.
  rowMultiplier = ti.mmaTileShape[0] if ti.isPreShuffled else 1
  inc = int(ti.depthUBytes * rowMultiplier)
  emitSrdAdvance(module, tc, inc, writer, kernel, holdOnLastIter)
  return module


@_emitGRPtrUpdate.register(GRTag_TLU1)
def _emitGRPtrUpdate_TLU1(tag, tile, ti, writer, kernel, holdOnLastIter=False):
  """Advance the SRD base pointer by one DepthU K-window.

  holdOnLastIter is accepted so this shares the dispatcher's signature, but it is
  deliberately not honoured here. The hold is a TLU=0 (TN) optimization: it relies
  on cluster 2 being able to harmlessly re-read cluster 1's K-block, which only
  holds for the pre-loop shape the TN path builds. GRTag_TLU1 covers the NN/NT/TT
  layouts, where applying it miscompares. Leaving the advance unconditional keeps
  those kernels byte-identical to the unoptimized path.

  The free dim is unit-stride here and K is strided, so a DepthU window spans
  DepthU * bpe * strideK bytes.  depthUBytes is already DepthU * bpe, so it only
  needs scaling by the runtime K stride.  When K is itself unit-stride the window
  is the plain TLU=0 advance, which that case and the TDM path both defer to.

  The window lands in the persistent GlobalReadIncs{tc} rather than a pool temp.
  PostLoopStoreInNll weaves the fused store's coordinate VALU into this block's
  MFMA gaps (LogicalScheduler._emitNGLL), and that splice happens AFTER both
  streams are generated -- so a temp whose allocTmpSgpr scope has already closed
  gets handed out again to the coord stream and the two live ranges overlap. A
  pinned register cannot be re-handed. The pin is guaranteed here: this branch
  writes Srd{tc}, so BufferLoad is on (globalReadIncsUseVgpr False), and TLU=1
  keeps useConstSgprGlobalReadIncs off, so numSgprGlobalReadIncs{tc} >= 1 and
  removeGROffsetsVariableSgprsFromPool holds it out of the pool for the whole
  loop. The value is also exactly what GlobalReadIncs{tc} is defined to hold.
  """
  tc = ti.tc
  strideK = writer.strideRef(tc, kernel["ProblemType"]["IndexUnroll"])
  if kernel.get("enableTDM%s" % tc, False) or writer.isConstUnitStride(strideK):
    return _emitGRPtrUpdate_TLU0(tag, tile, ti, writer, kernel)

  module = Module(f"GR Ptr Update ({tc})")
  inc = int(ti.depthUBytes)
  incSgpr = writer.globalReadIncsOperand(tc, writer.states.unrollIdx)
  assert not isinstance(incSgpr, str), \
    f"GR Ptr Update ({tc}): TLU=1 must hold its K-window in a GlobalReadIncs SGPR"
  module.add(SMulI32(dst=incSgpr, src0=inc, src1=strideK,
             comment=f"{tc}: DepthU*bpe({inc}) * strideK (NT K-window bytes)"))
  module.add(SAddU32(dst=sgpr(f"Srd{tc}"), src0=sgpr(f"Srd{tc}"), src1=incSgpr,
             comment=f"{tc}: advance SRD by DepthU K-window"))
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

  wavesPerPartition = int(2 * tileInfo.loadRatioGR)
  if tileInfo.loadRatioGR not in (0.5, 1.0, 2.0):
    raise NotImplementedError("Unsupported loadRatioGR for wave partition: %s"%str(tileInfo.loadRatioGR))
  # loadRatioGR determines how many waves cooperate inside one free-dimension
  # partition: 0.5 -> 1, 1.0 -> 2, 2.0 -> 4.  Express the old three special
  # cases as a single quotient/remainder mapping so eight-wave asymmetric
  # groups do not send the upper waves past the end of LDS.
  if wavesPerPartition == 1:
    module.add(VMovB32(dst=vgpr(localRow), src=0, comment="%s: one wave per partition"%tc))
    module.add(VMovB32(dst=vgpr(partitionRow), src=vgpr(waveId), comment="%s: partition = waveId"%tc))
  else:
    shift = wavesPerPartition.bit_length() - 1
    module.add(VAndB32(dst=vgpr(localRow), src0=hex(wavesPerPartition - 1), src1=vgpr(waveId),
                      comment="%s: waveId %% %u"%(tc, wavesPerPartition)))
    module.add(VLShiftRightB32(dst=vgpr(partitionRow), shiftHex=hex(shift), src=vgpr(waveId),
                              comment="%s: waveId / %u"%(tc, wavesPerPartition)))

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

def _preShuffledKStepBytes(tileInfo, offsetK):
  """Scale a K-subtile step for pre-shuffled data, refusing steps that cannot be encoded.

  A K step consumes a K-extent on every row of the block, so the global step is
  ``mmaTileShape[0]`` times the unit-stride byte count.  That product rides in the
  DTL immediate, so it has a hard ceiling; past it the step has to move into
  soffset the way the TLU=1 K-window path already does.
  """
  rowsPerBlock = int(tileInfo.mmaTileShape[0])
  depthU = int(tileInfo.depthUBytes / tileInfo.bpe)
  maxDepthU = swizzleTensorMaxDepthU(rowsPerBlock, tileInfo.bpe)
  if depthU > maxDepthU:
    raise RuntimeError(
        "SwizzleTensor%s with DepthU %u is not supported: the pre-shuffled global "
        "read is only validated up to DepthU %u.  Emitting it would read the wrong "
        "K positions instead of failing."
        % (tileInfo.tc, depthU, maxDepthU))
  scaled = offsetK * rowsPerBlock
  if scaled > SWIZZLE_DTL_OFFSET12_MAX:
    raise RuntimeError(
        "SwizzleTensor%s: pre-shuffled K-window step is %u bytes (DepthU %u, %u "
        "rows per block), which does not fit the %u byte DTL immediate.  Fold the "
        "step into soffset instead."
        % (tileInfo.tc, scaled, depthU, rowsPerBlock, SWIZZLE_DTL_OFFSET12_MAX))
  return scaled


def _grComputeOffset_legacy(module, writer, tileInfo, colId, rowId, output):
  tc = tileInfo.tc
  bpeBits = int(8*tileInfo.bpe)
  tmpVgpr = writer.vgprPool.checkOut(2, tag="_grComputeOffset_legacy_tmpVgpr")
  colBytes = tmpVgpr + 1
  loadWidth = tileInfo.loadWidthGR
  module.add(VLShiftLeftB32(dst=vgpr(colBytes), shiftHex=hex(loadWidth.bit_length()-1), src=vgpr(colId), comment="scale col_id by load_width"))
  if tileInfo.isPreShuffled:
    # The GR enumeration covers one K-subtile at a time, so a row step is that
    # subtile's K-extent, not the whole DepthU window.  They are equal only when
    # localSubtileGrid[1] == 1.
    subIterKBytes = tileInfo.subIterKBytes
    module.add(VLShiftLeftB32(dst=vgpr(tmpVgpr), shiftHex=hex(subIterKBytes.bit_length()-1),
                              src=vgpr(rowId), comment="%s: rowId * subIterKBytes"%tc))
  else:
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
  wavesPerPartition = int(2 * tileInfo.loadRatioGR)
  if tileInfo.loadRatioGR not in (0.5, 1.0, 2.0):
    raise NotImplementedError("Unsupported loadRatioGR for wave partition: %s"%str(tileInfo.loadRatioGR))
  if wavesPerPartition == 1:
    module.add(VMovB32(dst=vgpr(localRow), src=0, comment="%s: one wave per partition"%tc))
    module.add(VMovB32(dst=vgpr(partitionRow), src=vgpr(waveId), comment="%s: partition = waveId"%tc))
  else:
    shift = wavesPerPartition.bit_length() - 1
    module.add(VAndB32(dst=vgpr(localRow), src0=hex(wavesPerPartition - 1), src1=vgpr(waveId),
                      comment="%s: waveId %% %u"%(tc, wavesPerPartition)))
    module.add(VLShiftRightB32(dst=vgpr(partitionRow), shiftHex=hex(shift), src=vgpr(waveId),
                              comment="%s: waveId / %u"%(tc, wavesPerPartition)))
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
    if tileInfo.loadRatioGR == 0.5 and not tileInfo.isPreShuffled:
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

def _grSwizzleColIds(module, writer, pairs, blockSize, numRowsPerLDSBanks,
                     laneId, waveId):
  """Row-major GR colId swizzle for the tensors sharing this layout.

  `pairs` is [(tileInfo, colId), ...].  The base swizzle does not depend on the
  tensor, so it is computed on the first colId and copied to the rest; only the
  loadRatioGR rotation is per tensor.  NN and TT pass a single pair, since there
  only one operand is row-major.
  """
  tmpVgpr = writer.vgprPool.checkOut(3, tag="_grSwizzleColIds_tmpVgpr")
  ldsRowId = tmpVgpr
  tmp = tmpVgpr + 1
  waveRotation = tmpVgpr + 2
  half = blockSize // 2
  baseInfo, baseColId = pairs[0]
  copies = pairs[1:]
  baseName = "colIdA" if copies else "colId"
  module.addComment0("Swizzling" if copies else "Swizzling (%s)" % baseInfo.tc)
  module.add(VLShiftRightB32(dst=vgpr(ldsRowId), shiftHex=hex(blockSize.bit_length()-1), src=vgpr(laneId), comment="row id within wave"))
  module.add(VLShiftRightB32(dst=vgpr(ldsRowId), shiftHex=hex(numRowsPerLDSBanks.bit_length()-1), src=vgpr(ldsRowId), comment="lds row id"))
  module.add(VAndB32(dst=vgpr(tmp), src0=vgpr(ldsRowId), src1=hex(1), comment="swap_bit = ldsRowId & 1"))
  if baseInfo.bpe == 1:  # FP8: step1=block-swap, step2=wave K_group rotation
    module.add(VLShiftLeftB32(dst=vgpr(tmp), shiftHex=hex(int(math.log2(half))), src=vgpr(tmp),
               comment=f"swap_bit * {half}"))
    module.add(VXorB32(dst=vgpr(baseColId), src0=vgpr(baseColId), src1=vgpr(tmp),
               comment="FP8 step1: block-swap %s" % baseName))
    for _, cId in copies:
      module.add(VMovB32(dst=vgpr(cId), src=vgpr(baseColId), comment="colIdB = colIdA"))
    module.add(VAndB32(dst=vgpr(tmp), src0=vgpr(waveId), src1=hex(1), comment="wave_half = waveId & 1"))
    module.add(VLShiftLeftB32(dst=vgpr(tmp), shiftHex=hex(1), src=vgpr(tmp), comment="rotation = wave_half * 2"))
    for tInfo, cId in pairs:
      if tInfo.loadRatioGR != 0.5:
        module.add(VAndB32(dst=vgpr(waveRotation), src0=vgpr(cId), src1=hex(4), comment="FP8 step2: block_bit = colId & 4"))
        module.add(VAndB32(dst=vgpr(cId), src0=vgpr(cId), src1=hex(3), comment="K_group = colId & 3"))
        module.add(VAddU32(dst=vgpr(cId), src0=vgpr(cId), src1=vgpr(tmp), comment="K_group + rotation"))
        module.add(VAndB32(dst=vgpr(cId), src0=vgpr(cId), src1=hex(3), comment="(K_group+rotation) % 4"))
        module.add(VAddU32(dst=vgpr(cId), src0=vgpr(cId), src1=vgpr(waveRotation), comment="K_group_rot + block_bit"))
  else:  # FP4/FP16: pair-swap (even ldsRowId) + intra/inter-wave rotation
    module.add(VCmpXEqU32(dst=VCC(), src0=0, src1=vgpr(tmp), comment="lds row id % 2 == 0 ?"))
    module.add(VMovB32(dst=vgpr(baseColId), src=vgpr(baseColId), dpp=DPPModifiers(quad_perm=[1,0,3,2]), comment="swap colId pairs for swizzling"))
    module.add(SMovB64(dst=EXEC(), src=-1))
    for _, cId in copies:
      module.add(VMovB32(dst=vgpr(cId), src=vgpr(baseColId), comment=""))
    module.addComment0("Rotation within a single wave")
    module.add(VLShiftRightB32(dst=vgpr(tmp), shiftHex=hex(1), src=vgpr(ldsRowId), comment=""))
    module.add(VLShiftLeftB32(dst=vgpr(tmp), shiftHex=hex(1), src=vgpr(tmp), comment="(ldsRowId //2) * 2"))
    module.add(VSubU32(dst=vgpr(tmp), src0=hex(blockSize), src1=vgpr(tmp), comment="rotation offset : blockSize - (ldsRowId//2)*2"))
    for tInfo, cId in pairs:
      if tInfo.loadRatioGR != 0.5:
        module.addComment0("Rotation per wave")
        module.add(VAndB32(dst=vgpr(waveRotation), src0=vgpr(waveId), src1=hex(1), comment=""))
        module.add(VLShiftLeftB32(dst=vgpr(waveRotation), shiftHex=hex((2*numRowsPerLDSBanks).bit_length() - 1), src=vgpr(waveRotation), comment=""))
        module.add(VSubU32(dst=vgpr(waveRotation), src0=vgpr(tmp), src1=vgpr(waveRotation), comment=""))
        module.add(VAddU32(dst=vgpr(cId), src0=vgpr(waveRotation), src1=vgpr(cId), comment=""))
      else:
        module.add(VAddU32(dst=vgpr(cId), src0=vgpr(tmp), src1=vgpr(cId), comment=""))
    for _, cId in pairs:
      module.add(VAndB32(dst=vgpr(cId), src0=vgpr(cId), src1=hex(blockSize-1), comment="(col + offset) % block_size"))
  writer.vgprPool.checkIn(tmpVgpr)

def _isGRTLU1(tileInfo):
  return bool(tileInfo.gr and isinstance(tileInfo.gr.config.tag, GRTag_TLU1))


def _graTileAssignment_rowMajorSingle(writer, kernel, module, tileInfo):
  """Row-major (TLU=0) GR offsets for a single tensor.

  Mirrors the interleaved A+B path, but every parameter comes from this
  tensor's own geometry so it can be paired with a TLU=1 operand (NN / TT).
  """
  wavesize = kernel["WavefrontSize"]
  loadWidth = tileInfo.loadWidthGR
  blockSize, numRowsPerLDSBanks = _grRowTiling(writer, tileInfo)
  tmpVgpr = writer.vgprPool.checkOut(5, tag="_graTileAssignment_rowMajorSingle_tmpVgpr")
  colId, rowId, rowOffset, waveId, laneId = range(tmpVgpr, tmpVgpr + 5)
  module.add(VLShiftRightB32(dst=vgpr(waveId), shiftHex=hex(wavesize.bit_length()-1), src=vgpr("Serial"), comment="Wave Id"))
  module.add(VAndB32(dst=vgpr(laneId), src0=vgpr("Serial"), src1=wavesize-1, comment=""))
  module.add(VAndB32(dst=vgpr(colId), src0=vgpr("Serial"), src1=(blockSize-1), comment="get col_id in wave for %uB load"%loadWidth))
  module.add(VLShiftRightB32(dst=vgpr(rowId), shiftHex=hex(blockSize.bit_length()-1), src=vgpr(laneId), comment="row id within wave"))
  _grSwizzleColIds(module, writer, [(tileInfo, colId)], blockSize, numRowsPerLDSBanks,
                   laneId, waveId)
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
  wavesize = kernel["WavefrontSize"]
  loadWidth = tileInfoA.loadWidthGR
  blockSize, numRowsPerLDSBanks = _grRowTiling(writer, tileInfoA)
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
  # Without a pre-shuffled tensor the swizzle derives colIdB from colIdA, so only
  # seed it separately when one of the two skips the swizzle.
  if tileInfoA.isPreShuffled or tileInfoB.isPreShuffled:
    module.add(VAndB32(dst=vgpr(colIdB), src0=vgpr("Serial"), src1=(blockSize-1), comment="colIdB base"))
  module.add(VLShiftRightB32(dst=vgpr(rowId), shiftHex=hex(blockSize.bit_length()-1), src=vgpr(laneId), comment="row id within wave"))
  # A pre-shuffled tensor already carries the host's layout, so it must not be
  # swizzled again; pass only the tensors that still need it.
  swizzlePairs = [(t, c) for t, c in ((tileInfoA, colIdA), (tileInfoB, colIdB))
                  if not t.isPreShuffled]
  if swizzlePairs:
    _grSwizzleColIds(module, writer, swizzlePairs,
                     blockSize, numRowsPerLDSBanks, laneId, waveId)
  _grComputeRowPartition_legacy(module, kernel, writer, tileInfoA, waveId, rowOffsetA)
  _grComputeRowPartition_legacy(module, kernel, writer, tileInfoB, waveId, rowOffsetB)
  _grComputeAllOffsets_legacy(module, writer, tileInfoA, colIdA, rowId, rowOffsetA)
  _grComputeAllOffsets_legacy(module, writer, tileInfoB, colIdB, rowId, rowOffsetB)

  for tile, rowOff in ((tileInfoA, rowOffsetA), (tileInfoB, rowOffsetB)):
    if tile.isPreShuffled:
      _addPreShuffleCorrectionToGROffsets(module, writer, kernel, tile, rowOff)

  writer.vgprPool.checkIn(tmpVgpr)
  _grComputeSubtileOffsets_legacy(writer, module, tileInfoA)
  _grComputeSubtileOffsets_legacy(writer, module, tileInfoB)
  return module


def _tluPadFreeExtent(kernel, tileInfo):
  """Real free-dim element extent when the GR strip is padded, else None.

  A padded stack fetches tiles that are never read, and the SRD limit cannot
  exclude them (one linear bound over the K window), so the caller uses this
  extent to push the pad lanes out of range.  Only a single strip pads, and
  only at wave group 1 is m_chunk the absolute free-dim position.
  """
  tc = tileInfo.tc
  wgIdx = 0 if tc == 'A' else 1
  if int(kernel["MIWaveGroup"][wgIdx]) != 1:
    return None
  freeElems = int(kernel["MacroTile0" if tc == 'A' else "MacroTile1"])
  mtTiles = freeElems // int(tileInfo.mmaTileShape[0])
  stack = int(tileInfo.subtileShape[0])
  if stack <= 0 or mtTiles % stack == 0:
    return None
  return freeElems


def _graTileAssignment_tlu_colScatter(writer, kernel, tileInfo, module, laneId,
                                      strideK, cs):
  """GR per-lane offsets for the column-scatter TLU layout (8x1 fp4 and up).

  Thread T within load i holds logical (m_chunk, col_group), de-interleaved from
  T as the inverse of the LR read's bit-interleave, giving::

      offset(T, i) = ((col_group*N + i) * strideK + m_chunk * elemsPerChunk) * bpe

  col_group and m_chunk are load-independent, so they are computed once per lane
  and the loop only adds ``i * strideK``.  See SubtileTLUSwizzle.
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
  # A padded strip carries lanes whose free-dim chunk starts past the operand's
  # real tiles.  Compare while mc is still a chunk index: the bound is then small
  # enough to be an inline constant, where the element extent (192, 224) is not.
  # m_chunk is load-independent, so one compare covers every load; the select has
  # to come last though, after the per-wave offset is folded in, because the steps
  # between would mangle an out-of-range value back into a valid address.
  padFreeElems = _tluPadFreeExtent(kernel, tileInfo)
  laneSGPRCount = writer.states.laneSGPRCount
  with contextlib.ExitStack() as padStack:
    padMaskSgpr = None
    oobVgpr = None
    if padFreeElems is not None:
      padChunks = -(-padFreeElems // elemsPerChunk)
      oobVgpr = writer.vgprPool.checkOut(1, tag="_graColScatter_oob")
      module.add(VMovB32(dst=vgpr(oobVgpr), src="BufferOOB",
                 comment="%s: offset that the buffer bound rejects" % tc))
      padMaskSgpr = padStack.enter_context(
          writer.allocTmpSgpr(laneSGPRCount, alignment=laneSGPRCount,
                              tag="_graColScatter_padMask")).idx
      module.add(VCmpLtI32(dst=sgpr(padMaskSgpr, laneSGPRCount), src0=vgpr(mc),
                 src1=padChunks,
                 comment="%s: m_chunk < %u, i.e. lane inside the real %u elements?"
                         % (tc, padChunks, padFreeElems)))

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
      if padMaskSgpr is not None:
        module.add(VCndMaskB32(dst=vgpr(out), src0=vgpr(oobVgpr), src1=vgpr(out),
                   src2=sgpr(padMaskSgpr, laneSGPRCount),
                   comment="%s: pad lane -> BufferOOB, dropping its global read" % tc))

    if oobVgpr is not None:
      writer.vgprPool.checkIn(oobVgpr)

  if waveAxisOffVgpr is not None:
    writer.vgprPool.checkIn(waveAxisOffVgpr)
  writer.vgprPool.checkIn(colK)
  writer.vgprPool.checkIn(bit)
  writer.vgprPool.checkIn(cg)
  writer.vgprPool.checkIn(mc)
  return module


def _b128ChunkTiling(tileInfo, mStripBytes):
  """How one lane's b128 divides a strip: (chunks per K row, elements per chunk).

  A b128 covers 16/bpe contiguous free-dim elements at one K row.  Only fp4
  subdivides a row; every other TLU dtype keeps one chunk per K row, making the
  per-lane offset a pure K ramp.
  """
  chunksPerK = max(1, mStripBytes // 16) if float(tileInfo.bpe) == 0.5 else 1
  return chunksPerK, int(16 / tileInfo.bpe)


def _graTileAssignment_tlu(writer, kernel, tileInfo):
  """GR per-lane offset for TLU=1 (NT / free-dim contiguous) subtile tiles.

  The free dim is contiguous, so when one 128-bit buffer_load covers a whole
  free-dim strip at one K row the per-lane byte offset is a pure K ramp::

      offset(lane, i) = (laneId + i * wavesize) * strideK * bpe

  One VGPR per GR load into sharedVgprGROffset[]; the M/N position lives inside
  the load width, so there is no free-dim term and no bank swizzle.
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

  instM = int(tileInfo.mmaTileShape[0])
  mStripBytes = int(tileInfo.subtileShape[0] * instM * tileInfo.bpe)
  chunksPerK, elemsPerChunk = _b128ChunkTiling(tileInfo, mStripBytes)
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


def _tluKRowsPerWave(tileInfo, coopWaves):
  """K rows one wave owns when several share a strip.

  The unit matches what the per-lane GR offset walks: under col_scatter the load
  index is itself the K column, so a wave owning numGRPerSubtile loads starts
  that many columns in; on the plain K ramp the strip's K rows divide by the
  group.
  """
  if selectTLUColScatter(tileInfo) is not None:
    return int(tileInfo.numGRPerSubtile)
  kRows = int(tileInfo.mmaTileShape[1] * tileInfo.subtileShape[1])
  return kRows // coopWaves


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
  coop = int(tileInfo.grCoopWaves)
  if coop <= 1:
    module.add(VMovB32(dst=vgpr(dst), src=0, comment="%s: single fetching wave" % tc))
    return False
  perStrip = max(1, int(tileInfo.grWavesPerStrip))
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
  perStrip = max(1, int(ti.grWavesPerStrip))
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
  wavesPerStrip = int(tileInfo.grWavesPerStrip)
  coopWaves = int(tileInfo.grCoopWaves)
  winSplit = int(tileInfo.grKWindowSplit)
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
    # Waves sharing a strip split its K rows, so wave a starts at K row
    # a * kRowsPerWave.  K is the strided dim on TLU=1, hence the runtime stride.
    kRowsPerWave = _tluKRowsPerWave(tileInfo, coopWaves)
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
  counts convert each of those to K rows.  The sId1 an emit sees is the first
  window of a winSplit-sized group, so one run is one window.
  """
  coop = int(tileInfo.grCoopWaves)
  perStrip = max(1, int(tileInfo.grWavesPerStrip))
  kSplit = max(1, coop // perStrip)
  winSplit = max(1, int(tileInfo.grKWindowSplit))
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
  otherWaves = int(tileInfo.grOtherAxisWaves)
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
def _addPreShuffleCorrectionToGROffsets(module, writer, kernel, tile, rowOffset):
  """Fold aligned 16-row HBM padding into pre-shuffled GR offsets."""
  tc = tile.tc
  strideRef = "StrideA0I" if tc == 'A' else "StrideB1J"
  depthU = int(tile.subIterKBytes / tile.bpe)
  nTileRows = tile.mmaTileShape[0]
  corrVgpr = writer.vgprPool.checkOut(
      1, tag="_addPreShuffleCorrectionToGROffsets_corrVgpr")
  tmpSgpr = writer.sgprPool.checkOut(
      1, tag="_addPreShuffleCorrectionToGROffsets_tmpSgpr", preventOverflow=False)

  module.add(VAndB32(dst=vgpr(corrVgpr),
                     src0=hex(~(nTileRows - 1) & 0xFFFFFFFF),
                     src1=vgpr(rowOffset), comment="aligned pre-shuffled row"))
  if tile.bpe == 0.5:
    # Halve the stride rather than the product: a K shorter than one DepthU
    # window makes this step legitimately negative, and shifting the product
    # right is a logical shift, which would turn it into roughly 2**31.
    module.add(SLShiftRightB32(dst=sgpr(tmpSgpr), shiftHex=hex(1),
                               src=sgpr(strideRef), comment="FP4 elements to bytes"))
    module.add(SSubU32(dst=sgpr(tmpSgpr), src0=sgpr(tmpSgpr), src1=hex(depthU // 2),
                       comment="stride - DepthU"))
  else:
    module.add(SSubU32(dst=sgpr(tmpSgpr), src0=sgpr(strideRef), src1=hex(depthU),
                       comment="stride - DepthU"))
  module.add(VMulLOU32(dst=vgpr(corrVgpr), src0=sgpr(tmpSgpr),
                       src1=vgpr(corrVgpr), comment="row padding correction"))

  for i, grOffset in enumerate(tile.sharedVgprGROffset):
    module.add(VAddU32(dst=vgpr(grOffset), src0=vgpr(grOffset),
                       src1=vgpr(corrVgpr),
                       comment="pre-shuffled %s GR[%u] correction" % (tc, i)))
  writer.vgprPool.checkIn(corrVgpr)
  writer.sgprPool.checkIn(tmpSgpr)


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
                K stride for K-window (sId1>0) global-address advances, and for
                unique TDMSingleIssuer skip labels).
  """
  module = Module()

  # TDM path: emit one tensor_load_to_lds per tensor, skip all per-subtile DTL loads
  if kernel.get("enableTDM%s" % tileInfo.tc[0], False):
    if sId0 == 0 and sId1 == 0:
      tc = tileInfo.tc
      singleIssuer = bool(kernel.get("TDMSingleIssuer"))
      # TDMSingleIssuer (gemm_wt): A/B on 0/2, MXSA/MXSB on PF 1/3. Advances
      # sit inside each issuer gate. Each GR site emits its own tensor's load.
      if singleIssuer:
        if writer is None:
          return module
        module.add(emitTDMSingleIssuerLoad(writer, kernel, tc))
        return module
      group0 = "tdm%sGroup0" % tc
      group1 = "tdm%sGroup1" % tc
      isSubtileIter = _isSubtileIterateMode(kernel, tc)
      group2 = sgpr("tdm%sGroup2" % tc, 4) if isSubtileIter else None
      group3 = sgpr("tdm%sGroup3" % tc, 4) if isSubtileIter else None
      tensor = writer.tdmLoadModifiers(kernel, tc) if writer is not None else None
      module.add(TensorLoadToLds(sgpr(group0, 4), sgpr(group1, 8), group2, group3,
                                 comment="TDM: global->LDS for %s" % tc, tensor=tensor))
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
  if tileInfo.isPreShuffled:
    offsetK = _preShuffledKStepBytes(tileInfo, offsetK)

  subtileOffset = int(math.ceil(tileInfo.loadRatioGR*tileInfo.subtileSize))
  # loadRatioGR folds in grLoadWaves, so subtileOffset is the bytes one
  # *cooperative* load round covers.  Waves that share a strip are separated
  # within it by whole load-blocks (each takes a slice of the strip's
  # K rows) rather than interleaved within one block, so a wave still advances
  # m0 by its own single-wave block, not by the cooperative total.
  coopWaves = int(tileInfo.grCoopWaves)
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
  winSplit = int(tileInfo.grKWindowSplit)
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
  wavesPerStrip = int(ti.grWavesPerStrip)
  coopWaves = int(ti.grCoopWaves)
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
  winSplit = int(ti.grKWindowSplit)
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
def _subtileEnableTDM(kernel, tc):
  """Scales ride the matching data TDM enable; there is no enableTDMMXSA."""
  if tc in ('A', 'MXSA'):
    return bool(kernel.get("enableTDMA", False))
  if tc in ('B', 'MXSB'):
    return bool(kernel.get("enableTDMB", False))
  return False


def _subtileTileInfo(writer, tc):
  if tc == 'A':
    return writer.states.a.tileInfo
  if tc == 'B':
    return writer.states.b.tileInfo
  if tc == 'MXSA':
    return writer.states.mxsa.tileInfo
  if tc == 'MXSB':
    return writer.states.mxsb.tileInfo
  raise ValueError(f"unknown subtile tensor {tc}")


def _tdmSwapParams(tc, writer):
  """(nBuffers, strideBytes) for this tensor's LDS buffer advance."""
  lay = getattr(writer, "ldsV7", None)
  if lay is not None and tc.startswith("MX"):
    return lay.sc_nbuf, lay.sc_buf
  if lay is not None and tc in ("A", "B"):
    return 2, lay.buf
  return writer.states.numLDSBlk, writer.ldsTotalSize


def _tdmSwapLdsBuffer(tc, writer, kernel):
  module = Module()
  ldsAddrSgpr = "tdmLdsAddr%s" % tc
  swapSgpr = "tdmLdsSwapMask%s" % tc
  # TDMSingleIssuer patches tdmLdsAddr into the descriptor after each load.
  n, stride = _tdmSwapParams(tc, writer)
  useRotate = n >= 3 or bool(kernel.get("TDMSingleIssuer"))
  if not useRotate:
    module.addComment0("TDM: swap %s LDS buffer (XOR with per-tensor swap mask)" % tc)
    module.add(SXorB32(dst=sgpr(ldsAddrSgpr), src0=sgpr(ldsAddrSgpr), src1=sgpr(swapSgpr), comment=""))
    return module
  # Rotate write address by one buffer and wrap past the last one.
  # swapSgpr holds the per-tensor ceiling (base + n*stride).
  module.addComment0("TDM: rotate %s LDS write buffer (+1 of %d, wrap at ceiling)" % (tc, n))
  with writer.allocTmpSgpr(1) as tmpRes:
    tmp = tmpRes.idx
    module.add(SAddU32(dst=sgpr(ldsAddrSgpr), src0=sgpr(ldsAddrSgpr), src1=stride,
                       comment="%s TDM -> next LDS buffer" % tc))
    module.add(SSubU32(dst=sgpr(tmp), src0=sgpr(ldsAddrSgpr), src1=n * stride,
                       comment="wrapped-to-buffer0 candidate"))
    module.add(SCmpGeU32(src0=sgpr(ldsAddrSgpr), src1=sgpr(swapSgpr),
                         comment="past the last LDS buffer?"))
    module.add(SCSelectB32(dst=sgpr(ldsAddrSgpr), src0=sgpr(tmp), src1=sgpr(ldsAddrSgpr),
                           comment="wrap TDM write addr to buffer 0"))
  return module


def globalReadLDSBufferSwap(tc, writer, kernel):
  if _subtileEnableTDM(kernel, tc):
    return _tdmSwapLdsBuffer(tc, writer, kernel)
  if tc in ['A', 'B']:
    ti_ = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
    return ti_.emitGRLDSBufferSwap(writer, kernel)
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

  MXSA/MXSB use the gfx1250 InMemorySwizzle packing (mxUnit * MT * bpe),
  not the linear HostPreSwizzle stride used on gfx950.
  """
  tc = tP["tensorChar"]
  if tc.startswith("MX"):
    return _tdmGlobalOffsetSubtileMX(writer, kernel, tP)
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

    # TDMSingleIssuer: issuer wave owns the whole tile from row 0; skip per-wave split.
    if numWaves > 1 and not kernel.get("TDMSingleIssuer"):
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


class _MxScaleTdmShape:
  """Per-wave split of one DepthU of MX scales for subtile TDM.

  Global layout is InMemorySwizzle {K group, M/N, mxUnit} (K group row stride
  Size * mxUnit * bpe). LDS layout is [M/N chunk][K group][chunk bytes], what the
  scale LR expects (InstructionEmitter, ldsChunkBytes). Wave w loads K group
  w // spansPerKGroup, span w % spansPerKGroup of the WG's rowBytes: spanBytes
  contiguous global bytes as rows of `chunk` bytes, which TDM LDS padding of
  chunk * (numKGroups - 1) bytes spreads to the chunk * numKGroups pitch.
  """
  def __init__(self, writer, kernel, tc):
    subTc = tc[-1]
    ti = _subtileTileInfo(writer, tc)
    mxBlock = kernel["ProblemType"][f"MXBlock{subTc}"]
    self.mxUnit = kernel["MatrixInstK"] // mxBlock
    self.bpe = int(ti.bpe)
    self.mt = kernel["MacroTile0"] if subTc == "A" else kernel["MacroTile1"]
    self.numWaves = prod(kernel["MIWaveGroup"])
    self.numKGroups = (kernel["DepthU"] // mxBlock) // self.mxUnit
    self.chunk = int(ti.ldsChunkBytes)
    self.kGroupRowBytes = self.mxUnit * self.bpe   # times Size = global K-group row stride
    self.rowBytes = self.mt * self.mxUnit * self.bpe
    assert self.numWaves % self.numKGroups == 0, \
        f"{tc}: {self.numWaves} waves cannot split {self.numKGroups} scale K groups"
    self.spansPerKGroup = self.numWaves // self.numKGroups
    self.spanBytes = self.rowBytes // self.spansPerKGroup
    partitionBytes = self.rowBytes // kernel["MIWaveGroup"][0 if subTc == "A" else 1]
    assert self.chunk % int(ti.lrSubtileSize) == 0 and partitionBytes % self.chunk == 0 \
        and self.spanBytes % self.chunk == 0, \
        f"{tc}: scale chunk {self.chunk} B does not tile LR group {ti.lrSubtileSize} B, " \
        f"wave partition {partitionBytes} B and TDM span {self.spanBytes} B"
    self.rowsPerSpan = self.spanBytes // self.chunk

  def emitWaveSplit(self, mod, wavelen, kGroupSgpr, spanSgpr):
    """kGroupSgpr = wId // spansPerKGroup, spanSgpr = wId % spansPerKGroup."""
    mod.add(VReadfirstlaneB32(sgpr(spanSgpr), vgpr("Serial"), "first tId"))
    mod.add(SLShiftRightB32(sgpr(spanSgpr), ceil(log2(wavelen)), sgpr(spanSgpr),
            "wId = fTid // wavelen"))
    mod.add(SLShiftRightB32(sgpr(kGroupSgpr), int(log2(self.spansPerKGroup)), sgpr(spanSgpr),
            f"scale K group = wId // {self.spansPerKGroup}"))
    mod.add(SAndB32(sgpr(spanSgpr), sgpr(spanSgpr), self.spansPerKGroup - 1,
            f"scale span = wId % {self.spansPerKGroup}"))


def _tdmGlobalOffsetSubtileMX(writer, kernel, tP):
  """InMemorySwizzle start address for subtile MX TDM.

  WG offset is mxUnit * MT * bpe along M/N. Wave offset is its K group row
  (kGroup * Size * mxUnit * bpe) plus its span along M/N (_MxScaleTdmShape).
  """
  tc = tP["tensorChar"]
  tIdx = tP["idx"]
  bpe = tP["bpeGR"]
  subTc = tc[-1]
  mxBlock = kernel["ProblemType"][f"MXBlock{subTc}"]
  mxUnit = kernel["MatrixInstK"] // mxBlock
  mt = kernel["MacroTile0"] if subTc == "A" else kernel["MacroTile1"]
  numWaves = prod(kernel["MIWaveGroup"])
  shape = _MxScaleTdmShape(writer, kernel, tc)
  mod = Module(f"TDM Global Offset Subtile {tc}")

  with writer.allocTmpSgpr(max(3, writer.states.laneSGPRCount)) as tmpSgprRes:
    tmp = tmpSgprRes.idx
    waveOff = tmpSgprRes.idx + 2

    mod.add(writer._resolveTDMGlobalAddr(
        kernel, tc, f"Address{tc}", tmpSgprRes))

    # WG tile in packed MX layout is mxUnit * MT * bpe, not stride * MT * bpe.
    mod.add(SMulI32(dst=sgpr(tmp), src0=sgpr(f"WorkGroup{tIdx}"),
                     src1=int(mxUnit * mt * bpe),
                     comment=f"wgId * mxUnit({mxUnit}) * MT({mt}) * bpe({bpe})"))

    # TDMSingleIssuer: scale issuer starts at K group 0 / span 0 and ITERATEs.
    if numWaves > 1 and not kernel.get("TDMSingleIssuer"):
      kGroup = tmp + 1
      shape.emitWaveSplit(mod, kernel["WavefrontSize"], kGroup, waveOff)
      mod.add(SMulI32(dst=sgpr(waveOff), src0=sgpr(waveOff), src1=shape.spanBytes,
                       comment=f"waveOff = span * {shape.spanBytes}"))
      mod.add(SAddU32(dst=sgpr(tmp), src0=sgpr(tmp), src1=sgpr(waveOff), comment="+= waveOff"))
      mod.add(SMulI32(dst=sgpr(kGroup), src0=sgpr(kGroup), src1=sgpr(f"Size{INDEX_CHARS[tIdx]}"),
                       comment=f"kGroup * Size{INDEX_CHARS[tIdx]}"))
      mod.add(SMulI32(dst=sgpr(kGroup), src0=sgpr(kGroup), src1=shape.kGroupRowBytes,
                       comment=f"*= mxUnit * bpe ({shape.kGroupRowBytes})"))
      mod.add(SAddU32(dst=sgpr(tmp), src0=sgpr(tmp), src1=sgpr(kGroup), comment="+= K group row"))

    mod.add(SAddU32(dst=sgpr(f"Address{tc}"), src0=sgpr(f"Address{tc}"), src1=sgpr(tmp),
                     comment=f"+= offset(lo)"))
    mod.add(SAddCU32(dst=sgpr(f"Address{tc}+1"), src0=sgpr(f"Address{tc}+1"), src1=0,
                      comment=f"+= offset(hi)"))

    if kernel["ProblemType"]["Batched"] and kernel["ProblemType"]["StridedBatched"]:
      ia = tP["ia"]
      batchStrideName = f"Stride{tc}{writer.states.indexChars[ia[2]]}"
      batchIdx = sgpr("WorkGroup2")
      mod.addModuleAsFlatItems(writer.s_mul_u64_u32(sgpr(tmp), sgpr(tmp+1),
                                                     sgpr(batchStrideName), batchIdx,
                                                     comment="Batch: Stride*WG"))
      mod.add(SLShiftLeftB64(dst=sgpr(tmp, 2), src=sgpr(tmp, 2),
                              shiftHex=int(log2(bpe)) if bpe >= 1 else 0, comment="scale by bpe"))
      mod.add(SAddU64(dst=sgpr(f"Address{tc}", 2), src0=sgpr(tmp, 2), src1=sgpr(f"Address{tc}", 2),
                       comment="+= batch"))

  return mod


def _commitSubtileLdsTracking(mod, tc, waveOffsetSgprIdx, ldsTotalSize, writer):
  """Save the per-wave LDS write address and the buffer-swap helper sgpr.

  n == 2 : tdmLdsSwapMask = addr ^ (addr + stride)   (xor toggle)
  n >= 3 : tdmLdsSwapMask = addr + n*stride          (rotate ceiling)

  TDMSingleIssuer (ldsV7): only track LDS addr — advance uses tdmLdsXorMask /
  tdmGlobalInc (SwapMask SGPRs are not allocated).
  """
  ldsTrackSgpr = f"tdmLdsAddr{tc}"
  n, stride = _tdmSwapParams(tc, writer)
  # ldsTotalSize arg kept for call-site compatibility; v7 overrides via _tdmSwapParams.
  if n < 3 and getattr(writer, "ldsV7", None) is None:
    stride = ldsTotalSize
  mod.add(SMovB32(dst=sgpr(ldsTrackSgpr), src=sgpr(waveOffsetSgprIdx),
                  comment=f"init {ldsTrackSgpr} for buffer tracking"))
  if getattr(writer, "ldsV7", None) is not None:
    return
  swapMaskSgpr = f"tdmLdsSwapMask{tc}"
  if n >= 3:
    mod.add(SAddU32(dst=sgpr(swapMaskSgpr), src0=sgpr(waveOffsetSgprIdx), src1=n * stride,
                    comment=f"ceiling = addr + {n}*stride({stride})"))
    return
  mod.add(SAddU32(dst=sgpr(swapMaskSgpr), src0=sgpr(waveOffsetSgprIdx), src1=stride,
                  comment=f"addr + stride({stride})"))
  mod.add(SXorB32(dst=sgpr(swapMaskSgpr), src0=sgpr(waveOffsetSgprIdx), src1=sgpr(swapMaskSgpr),
                  comment="swapMask = addr XOR (addr + stride)"))


def _initTDMSingleIssuerAdvParams(mod, writer, kernel, tc):
  """Load wave-private advance SGPRs for this issuer's tensor.

  Must run inside the issuer-wave gate. Early (ungated) LDS tracking writes
  every tensor's ceiling into the aliased tdmLdsSwapMaskA on every wave, so
  the final early value is always MXSB's ceiling — rewrite it here from this
  wave's tdmLdsAddr{tc}.

  Advance bodies are split (proto-style): A/B uses xor + global bump only;
  MX uses rotate/wrap + global bump. A/B still zero unused rotate params so
  leftover readers stay safe.
  """
  if not kernel.get("TDMSingleIssuer"):
    return
  n, stride = _tdmSwapParams(tc, writer)
  addr = sgpr("tdmLdsAddr%s" % tc)
  if n < 3:
    # xorMask = addr ^ (addr + stride); SwapMask/AdvStride not allocated under SI.
    mod.add(SAddU32(dst=sgpr("tdmLdsXorMask"), src0=addr, src1=stride,
                    comment="TDMSingleIssuer: %s addr+stride" % tc))
    mod.add(SXorB32(dst=sgpr("tdmLdsXorMask"), src0=addr, src1=sgpr("tdmLdsXorMask"),
                    comment="TDMSingleIssuer: %s xor mask" % tc))
  else:
    # n>=3 rotate needs SwapMask/AdvStride/Span — not allocated under ldsV7.
    raise RuntimeError(
        "TDMSingleIssuer n>=3 advance needs rotate SGPRs; ldsV7 uses n=2")
  if tc.startswith("MX"):
    from .SubtileScaleEmit import emitScaleTdmStepBytes, scaleRowPitch
    ti = _subtileTileInfo(writer, tc)
    rows = scaleRowPitch(mod, kernel, tc, "tdmGlobalInc")
    mod.add(emitScaleTdmStepBytes(ti, kernel, sgpr("tdmGlobalInc"), rows,
                                  "TDMSingleIssuer: %s global inc" % tc))
  else:
    ti = _subtileTileInfo(writer, tc)
    mod.add(SMovB32(dst=sgpr("tdmGlobalInc"), src=int(ti.depthUBytes),
                    comment="TDMSingleIssuer: %s global inc=%u" % (tc, int(ti.depthUBytes))))


def _emitSubtileDataLdsTracking(writer, tc, mt, du, bpe, numWaves, wavelen,
                                ldsConstOffset, padIntervalBytes, padAmountBytes):
  mod = Module(f"TDM LDS tracking {tc}")
  lay = getattr(writer, "ldsV7", None)
  if lay is not None:
    # Single issuer owns the whole operand: LDS base is A0/B0 (buf0 of half0).
    base = lay.a0 if tc == "A" else lay.b0
    with writer.allocTmpSgpr(1) as tmpSgprRes:
      waveOffsetSgprIdx = tmpSgprRes.idx
      mod.add(SMovB32(sgpr(waveOffsetSgprIdx), base,
              f"TDMSingleIssuer: {tc} LDS base {'A0' if tc == 'A' else 'B0'}={base}"))
      _commitSubtileLdsTracking(mod, tc, waveOffsetSgprIdx, lay.buf, writer)
    return mod
  compStride = writer.ldsSegCompStride
  with writer.allocTmpSgpr(2 if compStride else 1) as tmpSgprRes:
    waveOffsetSgprIdx = tmpSgprRes.idx
    compSgprIdx = tmpSgprRes.idx + 1
    mod.add(VReadfirstlaneB32(sgpr(waveOffsetSgprIdx), vgpr("Serial"), "first tId"))
    mod.add(SLShiftRightB32(sgpr(waveOffsetSgprIdx), ceil(log2(wavelen)), sgpr(waveOffsetSgprIdx),
            "wId=fTid // wavelen"))
    # Each wave writes its mt/numWaves rows to a distinct LDS region,
    # matching the cooperative full-wave global split in
    # tdmGlobalOffsetSubtile. The union over all waves covers the whole
    # mt-row tile (identity map global-row r -> LDS-row r).
    if compStride:
      # Segment interleave: the tile's two halves (waves [0, n/2) and [n/2, n)) start
      # compStride apart so each half lands in its own LDS segment.
      wavesPerComp = numWaves // 2
      perWaveBytes = round(mt // numWaves * du * bpe)
      if padIntervalBytes != 0 and padAmountBytes != 0:
        perWaveBytes += perWaveBytes // padIntervalBytes * padAmountBytes
      mod.add(SLShiftRightB32(sgpr(compSgprIdx), int(log2(wavesPerComp)), sgpr(waveOffsetSgprIdx),
              f"compId = wId // {wavesPerComp}"))
      mod.add(SMulI32(sgpr(compSgprIdx), sgpr(compSgprIdx), compStride,
              f"compOffset = compId * {compStride}"))
      mod.add(SAndB32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), wavesPerComp - 1,
              "wId within component"))
      mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), perWaveBytes,
              f"woffset = wIdInComp * {perWaveBytes}"))
      mod.add(SAddU32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), sgpr(compSgprIdx),
              "woffset += compOffset"))
    elif padIntervalBytes != 0 and padAmountBytes != 0:
      tileBytes = round(mt // numWaves * du * bpe)
      padBytes = tileBytes // padIntervalBytes * padAmountBytes
      mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), tileBytes + padBytes,
              f"woffset = wId * ({tileBytes}+{padBytes})"))
    else:
      mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), round(mt // numWaves * du * bpe),
              "woffset = wId * (mt // numWaves * du * bpe)"))
    mod.add(SAddU32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), ldsConstOffset,
            f"ldsOffset = woffset + {ldsConstOffset} (subtile LDS offset for {tc})"))
    _commitSubtileLdsTracking(mod, tc, waveOffsetSgprIdx, writer.ldsTotalSize, writer)
  return mod


def _emitSubtileMxLdsTracking(writer, kernel, tc, ldsConstOffset):
  shape = _MxScaleTdmShape(writer, kernel, tc)
  mod = Module(f"TDM LDS tracking {tc}")
  lay = getattr(writer, "ldsV7", None)
  if lay is not None:
    # Single issuer covers all K groups via ITERATE from ring base (column 0).
    with writer.allocTmpSgpr(1) as tmpSgprRes:
      waveOffsetSgprIdx = tmpSgprRes.idx
      mod.add(SMovB32(sgpr(waveOffsetSgprIdx), ldsConstOffset,
              f"TDMSingleIssuer: {tc} scale ring base {ldsConstOffset}"))
      _commitSubtileLdsTracking(mod, tc, waveOffsetSgprIdx, lay.sc_buf, writer)
    return mod
  with writer.allocTmpSgpr(2) as tmpSgprRes:
    waveOffsetSgprIdx = tmpSgprRes.idx
    kGroup = tmpSgprRes.idx + 1
    shape.emitWaveSplit(mod, kernel["WavefrontSize"], kGroup, waveOffsetSgprIdx)
    mod.add(SMulI32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx),
            shape.spanBytes * shape.numKGroups,
            f"woffset = span * {shape.spanBytes} * {shape.numKGroups} K groups"))
    mod.add(SMulI32(sgpr(kGroup), sgpr(kGroup), shape.chunk, f"kGroup * chunk ({shape.chunk})"))
    mod.add(SAddU32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), sgpr(kGroup),
            "woffset += K group column"))
    mod.add(SAddU32(sgpr(waveOffsetSgprIdx), sgpr(waveOffsetSgprIdx), ldsConstOffset,
            f"ldsOffset = woffset + {ldsConstOffset} (subtile LDS offset for {tc})"))
    _commitSubtileLdsTracking(mod, tc, waveOffsetSgprIdx, writer.ldsTotalSize, writer)
  return mod


def initTDMLdsTrackingSubtile(writer, kernel, tP):
  """Prologue: record the per-wave LDS base and double-buffer swap mask.

  Without TDMSingleIssuer, refresh rebuilds the descriptor before every load
  (scale aliases A/B), so a descriptor built here is never consumed.
  TDMSingleIssuer inits once via initTDMDescriptorsSubtilePersistent and then
  patches LDS + global from these trackers. Address{tc} stays with
  tdmGlobalOffsetSubtile and the StreamK offset.
  """
  tc = tP["tensorChar"]
  bpe = tP["bpeGR"]
  numWaves = prod(kernel["MIWaveGroup"])
  wavelen = kernel["WavefrontSize"]
  if tc.startswith("MX"):
    return _emitSubtileMxLdsTracking(writer, kernel, tc, getattr(writer, f"ldsStartOffset{tc}", 0))
  ti = tP["idx"]
  mt = kernel[f"MacroTile{ti}"]
  du = kernel["DepthU"]
  tileInfo = writer.states.a.tileInfo if tc == "A" else writer.states.b.tileInfo
  padAmountBytes = int(getattr(tileInfo, "ldsRowPadBytes", 0))
  padIntervalBytes = int(getattr(tileInfo, "ldsBlockSizePerPadBytes", 0)) if padAmountBytes else 0
  ldsConstOffset = writer.ldsStartOffsetA if tc == "A" else writer.ldsStartOffsetB
  return _emitSubtileDataLdsTracking(
      writer, tc, mt, du, bpe, numWaves, wavelen,
      ldsConstOffset, padIntervalBytes, padAmountBytes)


def _emitSubtileGroup1Base(mod, comp, writer, kernel, desc0, desc1, tc, dtype,
                           padInterval, padAmount, iterateEnabled):
  """Write Group1+0 in one mov, then attach a multicast mask if this kernel has one.

  The mask is shifted by workgroup id, so it can land on the iterate bit.
  The previous sequence cleared that bit after the OR when iterate is off, and
  set it after the OR when iterate is on.
  """
  mod.add(comp.initOperands(desc0, desc1, None, None, clearGroup1=False))
  mod.add(comp.setGroup1Control(
      desc1, comp.group1ControlWord(dtype, padInterval, padAmount, False)))
  mod.add(comp.setGlobalAddr(desc0, f"Address{tc}"))
  # Subtile loads both A and B on every wave, so the mask is MulticastMask{tc},
  # not the non-subtile single parity mask.
  from ...Components.ClusterLoad import ClusterLoadTDM
  clusterComp = ClusterLoadTDM.find(writer)
  clusterMod = Module()
  if clusterComp:
    clusterMod = clusterComp.applyToDescriptor(writer, kernel, desc1, tc, subtile=True)
  mod.add(clusterMod)
  if iterateEnabled:
    mod.add(comp.setIterationEnabled(desc1, True))
  elif len(clusterMod.items()) > 0:
    mod.add(comp.setIterationEnabled(desc1, False))


def initTDMDescriptorSubtile(writer, kernel, tP, refresh=False):
  """Subtile variant of initTDMDescriptor()."""
  from ...Components.TensorDataMover import TensorDataMoverLoad
  comp = TensorDataMoverLoad.find(writer)
  tc = tP['tensorChar']
  if tc.startswith("MX"):
    return _initTDMDescriptorSubtileMX(writer, kernel, tP, refresh)
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
  singleIssuer = bool(kernel.get("TDMSingleIssuer"))
  # Pad-overflow iterate (row-by-row) is separate from single-issuer half-tile iterate.
  isPadIter = _isSubtileIterateMode(kernel, tc) and not singleIssuer
  isSubtileIter = isPadIter or singleIssuer

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
  padIntervalBytes = int(getattr(tileInfoForTc, "ldsBlockSizePerPadBytes", 0)) if padAmountBytes else 0

  # Pad-overflow iterate stores no padding in Group1+0; half-tile iterate keeps pad.
  padInterval = 0 if isPadIter else padIntervalBytes
  padAmount = 0 if isPadIter else padAmountBytes
  _emitSubtileGroup1Base(
      mod, comp, writer, kernel, descSgprName(0), descSgprName(1), tc, dtype,
      padInterval, padAmount, isSubtileIter)

  ldsTrackSgpr = f"tdmLdsAddr{tc}"
  if refresh:
    mod.add(comp.setLdsAddr(descSgprName(0), sgpr(ldsTrackSgpr)))
  else:
    mod.add(_emitSubtileDataLdsTracking(
        writer, tc, mt, du, bpe, numWaves, wavelen,
        ldsConstOffset, padIntervalBytes, padAmountBytes))
    mod.add(comp.setLdsAddr(descSgprName(0), sgpr(ldsTrackSgpr)))
  sizeShifter = 1 if dtype.isFloat4() else 0
  sizeShifterTile = sizeShifter

  # Group1 +1[31:16] dim0 lo, +2[15:0] dim0 hi, +2[31:16] dim1 lo, +3[15:0] dim1 hi,
  # +3[31:16] tile0. dim0 = SizeL >> sizeShifter: the shift into +1 drops SizeL's low
  # bit into bit 15, which is zero because an fp4 SizeL is even.
  g1 = descSgprName(1)
  with writer.allocTmpSgpr(1) as tmpSgprRes:
    tmp = sgpr(tmpSgprRes.idx)
    mod.add(SLShiftLeftB32(sgpr(f"{g1}+1"), 16 - sizeShifter, sgpr(sizeRefName(3)), "dim0 lo"))
    mod.add(SLShiftRightB32(tmp, 16 + sizeShifter, sgpr(sizeRefName(3)), "dim0 hi"))
    mod.add(SLShiftLeftB32(sgpr(f"{g1}+2"), 16, sgpr(sizeRefName(ti)), "dim1 lo"))
    mod.add(SOrB32(sgpr(f"{g1}+2"), sgpr(f"{g1}+2"), tmp, "| dim0 hi"))
  mod.add(SLShiftRightB32(sgpr(f"{g1}+3"), 16, sgpr(sizeRefName(ti)), "dim1 hi"))
  mod.add(SOrB32(sgpr(f"{g1}+3"), sgpr(f"{g1}+3"), hex(((sizeTile0 >> sizeShifterTile) & 0xFFFF) << 16),
                 f"tile0 = {sizeTile0 >> sizeShifterTile}"))

  if singleIssuer:
    # Half-tile iterate: 128 rows into A0/B0, then +inc into A1/B1.
    mod.add(comp.setTensorTile1(descSgprName(1), HALF_ROWS, writer, overwrite=True))
  elif isPadIter:
    # Iterate mode: one row per iteration.
    mod.add(comp.setTensorTile1(descSgprName(1), 1, writer, overwrite=True))
  else:
    # Clamp each wave's Tile1 (free-dim-1) load extent to the valid remainder.
    # tdmGlobalOffsetSubtile bases wave w at row w*(mt//numWaves), but the
    # descriptor Dim1 does not bound the walk, so an edge tile (free dim < mt)
    # reads past the tensor. setTensorTile1 takes a compile-time int, so write its
    # field (+4[15:0]) with a runtime clamp. No-op when the tile fits.
    perWaveRows = sizeTile1 // numWaves
    if numWaves > 1:
      mod.add(SMovB32(sgpr(f"{descSgprName(1)}+4"), sgpr(f"tdmClamp{tc}"),
              comment="set tile1 = clamped validRows (initTDMClampSubtile)"))
    else:
      mod.add(comp.setTensorTile1(descSgprName(1), perWaveRows, writer, overwrite=True))
  mod.add(comp.setTensorStride0(descSgprName(1), strideRefName(), sizeShifterTile, zeroHigh=False))
  mod.add(SMovB64(sgpr(f"{descSgprName(1)}+6", 2), 0, comment="Group1+6/+7 reserved"))

  if singleIssuer:
    lay = writer.ldsV7
    lds_inc = lay.inc_a if tc == "A" else lay.inc_b
    with writer.allocTmpSgpr(2) as tmp:
      sIter, sGInc = tmp.idx, tmp.idx + 1
      # 128 rows * bpe; fp4 packed -> bytes = 128 * stride * 0.5 = 64 * stride elems.
      mod.add(SMulI32(sgpr(sGInc), sgpr(strideRefName()), HALF_ROWS // (2 if dtype.isFloat4() else 1),
                      f"global_inc = {HALF_ROWS} rows * bpe"))
      mod.add(SMovB32(sgpr(f"{descSgprName(2)}+0"), 1, "tensor_dim2 = 1"))
      mod.add(comp.setIterationIncrements(descSgprName(2), lds_inc, sGInc))
      mod.add(SMovB32(sgpr(sIter), 1, "iter_count=1 (2 halves)"))
      mod.add(comp.setIterations(descSgprName(2), sIter))
  elif isPadIter:
    dss = comp.dataSizeShift(dtype)
    lds_inc = (padIntervalBytes + padAmountBytes) >> dss
    iter_count = sizeTile1 // numWaves
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


def _initTDMDescriptorSubtileMX(writer, kernel, tP, refresh=False):
  """Subtile TDM descriptor for MXSA/MXSB (InMemorySwizzle, cooperative waves)."""
  from ...Components.TensorDataMover import TensorDataMoverLoad
  comp = TensorDataMoverLoad.find(writer)
  tc = tP['tensorChar']
  tIdx = tP["idx"]
  dtype = kernel["ProblemType"][f"DataType{tc}"]
  shape = _MxScaleTdmShape(writer, kernel, tc)
  assert shape.bpe == 1, f"{tc}: subtile scale TDM expects 1-byte scales, got {shape.bpe}"
  chunk = shape.chunk
  ldsConstOffset = getattr(writer, f"ldsStartOffset{tc}", 0)
  sizeName = f"Size{INDEX_CHARS[tIdx]}"
  singleIssuer = bool(kernel.get("TDMSingleIssuer"))
  mod = Module(f"Init TDM Descriptor Subtile {tc}")

  def descSgprName(idx):
    return f"tdm{tc}Group{idx}"

  if kernel.get("MXScaleLRB128"):
    mod.add(_initTDMDescriptorSubtileMXB128(
        writer, kernel, tc, comp, dtype, shape, sizeName, descSgprName, ldsConstOffset, refresh))
    return mod

  # The wave's span is read as a dense (chunk, rowsPerSpan) tensor; LDS padding
  # leaves room for the other K groups' chunks between rows. Single-issuer
  # ITERATE keeps that pad and steps lds_inc=chunk across K-group columns.
  padAmountBytes = chunk * (shape.numKGroups - 1)
  _emitSubtileGroup1Base(
      mod, comp, writer, kernel, descSgprName(0), descSgprName(1), tc, dtype,
      chunk if padAmountBytes else 0, padAmountBytes, singleIssuer)

  ldsTrackSgpr = f"tdmLdsAddr{tc}"
  if not refresh:
    mod.add(_emitSubtileMxLdsTracking(writer, kernel, tc, ldsConstOffset))
  mod.add(comp.setLdsAddr(descSgprName(0), sgpr(ldsTrackSgpr)))

  mod.add(SMovB32(sgpr(f"{descSgprName(1)}+1"), hex(chunk << 16), f"dim0 = {chunk}"))
  if singleIssuer:
    # Full MT span (issuer covers all K groups); dim1 = rows of one span.
    rows = shape.rowBytes // chunk
    mod.add(SMovB32(sgpr(f"{descSgprName(1)}+2"), hex(rows << 16),
            f"dim1 = {rows} (full MT scale rows)"))
  else:
    mod.add(SMovB32(sgpr(f"{descSgprName(1)}+2"), sgpr(f"tdmClamp{tc}"),
            "dim1 = clamped rows (initTDMClampSubtile)"))
  mod.add(SMovB32(sgpr(f"{descSgprName(1)}+3"), hex(chunk << 16), f"tile0 = {chunk}"))
  mod.add(SMovB32(sgpr(f"{descSgprName(1)}+4"), shape.rowsPerSpan if not singleIssuer else (shape.rowBytes // chunk),
          f"tile1 = scale rows"))
  mod.add(SMovB32(sgpr(f"{descSgprName(1)}+5"), chunk, f"row stride = {chunk}"))
  mod.add(SMovB64(sgpr(f"{descSgprName(1)}+6", 2), 0, comment="Group1+6/+7 reserved"))

  if singleIssuer:
    # ITERATE over K groups: lds_inc=chunk, global_inc=Size*kGroupRowBytes.
    # MX aliases A Group2 (wave-private on PF waves 1/3).
    g2 = f"tdm{tc}Group2"
    with writer.allocTmpSgpr(2) as tmp:
      sIter, sGInc = tmp.idx, tmp.idx + 1
      mod.add(SMulI32(sgpr(sGInc), sgpr(sizeName), shape.kGroupRowBytes,
                      f"global_inc = Size * mxUnit*bpe ({shape.kGroupRowBytes})"))
      mod.add(SMovB32(sgpr(f"{g2}+0"), 1, "tensor_dim2 = 1"))
      mod.add(comp.setIterationIncrements(g2, chunk, sGInc))
      mod.add(SMovB32(sgpr(sIter), shape.numKGroups - 1,
                      f"iter_count={shape.numKGroups}-1"))
      mod.add(comp.setIterations(g2, sIter))
  return mod


def _initTDMDescriptorSubtileMXB128(writer, kernel, tc, comp, dtype, shape, sizeName,
                                    descSgprName, ldsConstOffset, refresh):
  """MXScaleLRB128 scale descriptor: copy each K group's MT rows to LDS unpadded.

  The host layout already holds the [lane][block] dword order the b128 LR reads
  (scaleLRB128Layout), so one K group is MT * 4 contiguous bytes, read as rows of
  `row` bytes. ITERATE steps the K groups: lds_inc = MT * 4, global_inc = padded
  rows * 4 (scaleRowPitch).
  """
  from .SubtileScaleEmit import scaleLRB128Layout, scaleRowPitch
  lay = scaleLRB128Layout(kernel, tc)
  assert shape.kGroupRowBytes == 4 and lay["kGroupBytes"] == shape.rowBytes, \
      f"{tc}: MXScaleLRB128 expects 4 B scale rows covering the MT, got " \
      f"{shape.kGroupRowBytes} B / {shape.rowBytes} B"
  assert kernel.get("TDMSingleIssuer"), f"{tc}: MXScaleLRB128 requires TDMSingleIssuer"
  row = 128
  rows = shape.rowBytes // row
  mod = Module(f"Init TDM Descriptor Subtile {tc} (b128 LR layout)")
  g1 = descSgprName(1)
  _emitSubtileGroup1Base(
      mod, comp, writer, kernel, descSgprName(0), g1, tc, dtype, 0, 0, True)
  ldsTrackSgpr = f"tdmLdsAddr{tc}"
  if not refresh:
    mod.add(_emitSubtileMxLdsTracking(writer, kernel, tc, ldsConstOffset))
  mod.add(comp.setLdsAddr(descSgprName(0), sgpr(ldsTrackSgpr)))
  mod.add(SMovB32(sgpr(f"{g1}+1"), hex(row << 16), f"dim0 = {row}"))
  mod.add(SMovB32(sgpr(f"{g1}+2"), hex(rows << 16), f"dim1 = {rows} (MT scale rows)"))
  mod.add(SMovB32(sgpr(f"{g1}+3"), hex(row << 16), f"tile0 = {row}"))
  mod.add(SMovB32(sgpr(f"{g1}+4"), rows, "tile1 = scale rows"))
  mod.add(SMovB32(sgpr(f"{g1}+5"), row, f"row stride = {row}"))
  mod.add(SMovB64(sgpr(f"{g1}+6", 2), 0, comment="Group1+6/+7 reserved"))
  g2 = f"tdm{tc}Group2"
  with writer.allocTmpSgpr(2) as tmp:
    sIter, sGInc = tmp.idx, tmp.idx + 1
    pitch = scaleRowPitch(mod, kernel, tc, sGInc)
    mod.add(SLShiftLeftB32(sgpr(sGInc), 2, pitch, "global_inc = scale rows * 4"))
    mod.add(SMovB32(sgpr(f"{g2}+0"), 1, "tensor_dim2 = 1"))
    mod.add(comp.setIterationIncrements(g2, lay["kGroupBytes"], sGInc))
    mod.add(SMovB32(sgpr(sIter), shape.numKGroups - 1, f"iter_count={shape.numKGroups}-1"))
    mod.add(comp.setIterations(g2, sIter))
  return mod


def initTDMClampSubtile(writer, kernel, tc):
  """Once per tile: tdmClamp{tc} = this wave's runtime descriptor bound.

  A/B: tile1 = clamp(Size - wId * mt/numWaves, 0, mt/numWaves). Scales: the
  Group1+2 word, dim1 = rows of this wave's span inside the tensor << 16, so an
  edge tile zero-fills whole chunks past Size. A partial last chunk still reads
  up to chunk - mxUnit bytes past Size (into the next K group row, or past the
  end for the last one); those scales only reach out-of-range rows of D.
  """
  mod = Module(f"TDM descriptor clamp {tc}")
  clamp = sgpr(f"tdmClamp{tc}")
  wavelen = kernel["WavefrontSize"]
  idx = 0 if tc.endswith("A") else 1
  sizeName = f"Size{INDEX_CHARS[idx]}"
  with writer.allocTmpSgpr(1) as tmpSgprRes:
    tmp = sgpr(tmpSgprRes.idx)
    mod.add(VReadfirstlaneB32(tmp, vgpr("Serial"), "first tId"))
    mod.add(SLShiftRightB32(tmp, ceil(log2(wavelen)), tmp, "wId = fTid // wavelen"))
    if not tc.startswith("MX"):
      perWaveRows = kernel[f"MacroTile{idx}"] // prod(kernel["MIWaveGroup"])
      mod.add(SMulI32(tmp, tmp, perWaveRows, f"waveGlobalRowStart = wId * {perWaveRows}"))
      mod.add(SSubI32(clamp, sgpr(sizeName), tmp, "Size_free - waveGlobalRowStart"))
      mod.add(SMaxI32(clamp, clamp, 0, "saturate negative remainder to 0"))
      mod.add(SMinU32(clamp, clamp, perWaveRows, f"clamp to per-wave rows ({perWaveRows})"))
      return mod
    shape = _MxScaleTdmShape(writer, kernel, tc)
    chunk = shape.chunk
    mod.add(SAndB32(tmp, tmp, shape.spansPerKGroup - 1, f"scale span = wId % {shape.spansPerKGroup}"))
    mod.add(SMulI32(tmp, tmp, shape.spanBytes, f"span * {shape.spanBytes}"))
    mod.add(SMulI32(clamp, shape.mt, sgpr(f"WorkGroup{idx}"), f"MT({shape.mt}) * wgId"))
    mod.add(SSubI32(clamp, sgpr(sizeName), clamp, f"remaining M/N = {sizeName} - MT*wgId"))
    mod.add(SMulI32(clamp, clamp, shape.kGroupRowBytes, f"remaining bytes (* {shape.kGroupRowBytes})"))
    mod.add(SSubI32(clamp, clamp, tmp, "- span start"))
    mod.add(SMaxI32(clamp, clamp, 0, "saturate negative remainder to 0"))
    mod.add(SMinU32(clamp, clamp, shape.spanBytes, f"clamp to span bytes ({shape.spanBytes})"))
    mod.add(SAddU32(clamp, clamp, chunk - 1, "round up to whole chunks"))
    mod.add(SLShiftRightB32(clamp, int(log2(chunk)), clamp, f"rows = ceil(bytes / {chunk})"))
    mod.add(SLShiftLeftB32(clamp, 16, clamp, "dim1 = rows (Group1+2 high half)"))
  return mod


def patchTDMDescriptorSubtile(writer, kernel, tc):
  """Copy live Address/LDS trackers into a persistent subtile TDM descriptor.

  TDMSingleIssuer keeps one descriptor per issuer wave in wave-private SGPRs
  (MXSA/MXSB still alias the A/B names). Only LDS + global change each
  iteration — matching the mxf4 prototype — so the mainloop avoids the full
  dim/tile/iterate rebuild that was dominating SALU latency.
  """
  from ...Components.TensorDataMover import TensorDataMoverLoad
  comp = TensorDataMoverLoad.find(writer)
  group0 = f"tdm{tc}Group0"
  mod = Module(f"Patch TDM Descriptor Subtile {tc}")
  mod.add(comp.setLdsAddr(group0, sgpr(f"tdmLdsAddr{tc}")))
  mod.add(comp.setGlobalAddr(group0, f"Address{tc}"))
  return mod


def advanceTDMDescriptorSubtileAfterLoad(writer, kernel, tc):
  """Proto-style post-TDM prepare: mutate LDS + global in the live descriptor.

  Issued immediately after tensor_load_to_lds so the next iteration's
  barrier→load path has no descriptor SALU. Matches mxf4_proto:
    tensor_load
    s_mov/xor LDS in Group0+1
    s_add global in Group0+2
  Shadow Address/tdmLdsAddr trackers are not updated here; the next tile
  re-bases them in setupNewTile. Replaces pre-load gr_inc for TDMSingleIssuer.

  Under TDMSingleIssuer all tensors share the A SGPR names (wave-private
  values), so patches always target tdmAGroup0.
  """
  group0 = "tdmAGroup0" if kernel.get("TDMSingleIssuer") else f"tdm{tc}Group0"
  mod = Module(f"Advance TDM after load {tc}")
  n, stride = _tdmSwapParams(tc, writer)
  # TDMSingleIssuer stores a rotate ceiling in SwapMask (even for A/B n=2).
  useRotate = n >= 3 or bool(kernel.get("TDMSingleIssuer"))
  if not useRotate:
    mod.addComment0("TDM: xor %s LDS in descriptor (next buffer)" % tc)
    mod.add(SXorB32(dst=sgpr(f"{group0}+1"), src0=sgpr(f"{group0}+1"),
                    src1=sgpr(f"tdmLdsSwapMask{tc}"), comment=""))
  else:
    # Rotate LDS write addr in the descriptor and wrap at ceiling.
    mod.addComment0("TDM: rotate %s LDS in descriptor (+1 of %d)" % (tc, n))
    with writer.allocTmpSgpr(1) as tmpRes:
      tmp = tmpRes.idx
      mod.add(SAddU32(dst=sgpr(f"{group0}+1"), src0=sgpr(f"{group0}+1"), src1=stride,
                      comment="%s TDM desc LDS -> next buffer" % tc))
      mod.add(SSubU32(dst=sgpr(tmp), src0=sgpr(f"{group0}+1"), src1=n * stride,
                      comment="wrapped-to-buffer0 candidate"))
      mod.add(SCmpGeU32(src0=sgpr(f"{group0}+1"), src1=sgpr(f"tdmLdsSwapMask{tc}"),
                        comment="past the last LDS buffer?"))
      mod.add(SCSelectB32(dst=sgpr(f"{group0}+1"), src0=sgpr(tmp), src1=sgpr(f"{group0}+1"),
                          comment="wrap TDM desc LDS to buffer 0"))
  if tc.startswith("MX"):
    from .SubtileScaleEmit import emitScaleTdmStepBytes, scaleRowPitch
    ti = _subtileTileInfo(writer, tc)
    with writer.allocTmpSgpr(1, tag="scaleTdmStepPost") as tmpSgprRes:
      inc = tmpSgprRes.idx
      rows = scaleRowPitch(mod, kernel, tc, inc)
      mod.add(emitScaleTdmStepBytes(ti, kernel, sgpr(inc), rows, "scale step"))
      mod.add(SAddU32(dst=sgpr(f"{group0}+2"), src0=sgpr(f"{group0}+2"), src1=sgpr(inc),
                      comment="%s TDM desc global +=" % tc))
      mod.add(SAddCU32(dst=sgpr(f"{group0}+3"), src0=sgpr(f"{group0}+3"), src1=0,
                       comment="%s TDM desc global carry" % tc))
  else:
    ti = _subtileTileInfo(writer, tc)
    inc = int(ti.depthUBytes)
    mod.addComment0("TDM: bump %s global in descriptor += %u" % (tc, inc))
    mod.add(SAddU64(dst=sgpr(f"{group0}+2", 2), src0=sgpr(f"{group0}+2", 2), src1=inc))
  return mod


def _tdmSIGroup0(tc):
  """Live descriptor Group0 name for SI (A/B share A; MX aliases MXSA→A)."""
  if tc in ("A", "B"):
    return "tdmAGroup0"
  if tc in ("MXSA", "MXSB"):
    return "tdmMXSAGroup0"
  return f"tdm{tc}Group0"


def _tdmSIGroup1(tc):
  if tc in ("A", "B"):
    return "tdmAGroup1"
  if tc in ("MXSA", "MXSB"):
    return "tdmMXSAGroup1"
  return f"tdm{tc}Group1"


def _emitTDMSingleIssuerGlobalBump(tc=None):
  """``S[2:3] += tdmGlobalInc`` on the live desc only (proto TJ_ADDR).

  Shadow ``Address`` / ``tdmLdsAddr`` stay at the tile base — ``setupNewTile``
  re-bases them. Data and MX keep separate Group0/1 so dual-issue does not
  clobber the advanced data descriptor.
  """
  g0 = _tdmSIGroup0(tc) if tc else "tdmAGroup0"
  mod = Module("TDMSingleIssuer global bump")
  mod.add(SAddU32(dst=sgpr(f"{g0}+2"), src0=sgpr(f"{g0}+2"),
                  src1=sgpr("tdmGlobalInc"),
                  comment="desc global += inc"))
  mod.add(SAddCU32(dst=sgpr(f"{g0}+3"), src0=sgpr(f"{g0}+3"), src1=0,
                   comment="desc global carry"))
  return mod


def _emitTDMSingleIssuerAdvanceAB(tc=None):
  """A/B (and MX depth-2) advance: xor LDS toggle + global bump on live desc."""
  g0 = _tdmSIGroup0(tc) if tc else "tdmAGroup0"
  mod = Module("TDMSingleIssuer advance A/B")
  mod.addComment0("TDMSingleIssuer: xor toggle + global bump (proto TOGGLE+ADDR)")
  mod.add(SXorB32(dst=sgpr(f"{g0}+1"), src0=sgpr(f"{g0}+1"),
                  src1=sgpr("tdmLdsXorMask"),
                  comment="desc LDS ^= xorMask (buf0^buf1)"))
  mod.add(_emitTDMSingleIssuerGlobalBump(tc))
  return mod


def _emitTDMSingleIssuerAdvanceMX(writer, tc):
  """MX depth>=3 advance: ring rotate + wrap, then global bump on live desc."""
  g0 = _tdmSIGroup0(tc)
  mod = Module("TDMSingleIssuer advance MX")
  mod.addComment0("TDMSingleIssuer: MX rotate/wrap + global bump")
  with writer.allocTmpSgpr(1) as tmpRes:
    tmp = tmpRes.idx
    mod.add(SAddU32(dst=sgpr(f"{g0}+1"), src0=sgpr(f"{g0}+1"),
                    src1=sgpr("tdmLdsAdvStride"),
                    comment="desc LDS += stride"))
    mod.add(SSubU32(dst=sgpr(tmp), src0=sgpr(f"{g0}+1"),
                    src1=sgpr("tdmLdsAdvSpan"),
                    comment="wrapped-to-buffer0 candidate"))
    mod.add(SCmpGeU32(src0=sgpr(f"{g0}+1"), src1=sgpr("tdmLdsSwapMaskA"),
                      comment="past last LDS buffer?"))
    mod.add(SCSelectB32(dst=sgpr(f"{g0}+1"), src0=sgpr(tmp),
                        src1=sgpr(f"{g0}+1"),
                        comment="wrap desc LDS to buffer 0"))
  mod.add(_emitTDMSingleIssuerGlobalBump(tc))
  return mod


def _emitTDMSingleIssuerAdvanceBody(writer, tc):
  """Proto-style per-tensor advance body (no shared xor+rotate template).

  MX with sc_nbuf==2 (gemm_wt default) uses the same xor toggle as A/B;
  rotate/wrap is only for rings deeper than 2.
  """
  n, _ = _tdmSwapParams(tc, writer)
  if tc.startswith("MX") and n >= 3:
    return _emitTDMSingleIssuerAdvanceMX(writer, tc)
  return _emitTDMSingleIssuerAdvanceAB(tc)


def _emitTDMSingleIssuerAdvance(writer, kernel, tc):
  """Wave-gated post-load advance using wave-private advance params."""
  issuer = {"A": ISSUER_A, "B": ISSUER_B, "MXSA": ISSUER_MXSA, "MXSB": ISSUER_MXSB}[tc]
  mod = Module(f"TDMSingleIssuer advance {tc}")
  name = writer.labels.getUniqueNamePrefix("tdm_si_adv_%s" % tc)
  skip = Label(name, "")
  mod.add(SCmpEQU32(src0=sgpr("WaveIdx"), src1=issuer,
                    comment="TDMSingleIssuer: %s advance = wave %d" % (tc, issuer)))
  mod.add(SCBranchSCC0(labelName=skip.getLabelName(),
                       comment="other waves skip %s advance" % tc))
  mod.add(_emitTDMSingleIssuerAdvanceBody(writer, tc))
  mod.add(skip)
  return mod


def _tdmSingleIssuerTensorParams(writer, tc):
  ti = _subtileTileInfo(writer, tc)
  idx = 0 if tc.endswith("A") else 1
  return {
      "tensorChar": tc,
      "idx": idx,
      "tileChar": f"{idx}{INDEX_CHARS[idx]}",
      "bpeGR": ti.bpe,
  }


def _emitTDMSingleIssuerWaveLoads(writer, kernel, wave, tc):
  """One tensor_load + advance on its issuer wave; other waves skip.

  gemm_wt: A/B on 0/2, MX on PF 1/3 (each wave owns one descriptor in the
  shared A/MX SGPR names). Adv params are wave-private.
  """
  g0, g1 = _tdmSIGroup0(tc), _tdmSIGroup1(tc)
  g2 = "tdm%sGroup2" % tc
  mod = Module("TDMSingleIssuer loads %s wave %d" % (tc, wave))
  skip = Label(writer.labels.getUniqueNamePrefix("tdm_si_ld_%s" % tc), "")
  mod.add(SCmpEQU32(src0=sgpr("WaveIdx"), src1=wave,
                    comment="TDMSingleIssuer: %s issue = wave %d" % (tc, wave)))
  mod.add(SCBranchSCC0(labelName=skip.getLabelName(),
                       comment="other waves skip %s TDM issue" % tc))
  mod.add(TensorLoadToLds(
      sgpr(g0, 4), sgpr(g1, 8),
      sgpr(g2, 4), sgpr("tdm%sGroup3" % tc, 4),
      comment="TDMSingleIssuer: %s tensor_load (wave %d)" % (tc, wave)))
  mod.add(_emitTDMSingleIssuerAdvanceBody(writer, tc))
  mod.add(skip)
  return mod


_TDM_SI_ISSUER = {"A": ISSUER_A, "B": ISSUER_B, "MXSA": ISSUER_MXSA, "MXSB": ISSUER_MXSB}


def emitTDMSingleIssuerLoad(writer, kernel, tc):
  """TDMSingleIssuer GR site for ``tc`` (A, B, MXSA or MXSB): its issuer-gated load."""
  return _emitTDMSingleIssuerWaveLoads(writer, kernel, _TDM_SI_ISSUER[tc], tc)


def refreshTDMDescriptorSubtile(writer, kernel, tc):
  """Sync a subtile descriptor immediately before its load.

  TDMSingleIssuer: patch LDS + global only (descriptor inited once per tile).
  Otherwise rebuild from constants / Size / Stride / Address / tdmLdsAddr /
  tdmClamp (no VALU dependency) because scale aliases overwrite A/B.
  """
  if kernel.get("TDMSingleIssuer"):
    return patchTDMDescriptorSubtile(writer, kernel, tc)
  ti = _subtileTileInfo(writer, tc)
  idx = 0 if tc.endswith("A") else 1
  tP = {
    "tensorChar": tc,
    "idx": idx,
    "tileChar": f"{idx}{INDEX_CHARS[idx]}",
    "bpeGR": ti.bpe,
  }
  return initTDMDescriptorSubtile(writer, kernel, tP, refresh=True)


def initTDMDescriptorsSubtilePersistent(writer, kernel):
  """Wave-gated one-time full TDM init for TDMSingleIssuer.

  Proto roles: A+MXSA on w0, B+MXSB on w2 (dedicated MX Group0/1). Prefetch
  waves 1/3 leave descriptors untouched.
  """
  mod = Module("Init TDM Descriptors Subtile Persistent")
  if not kernel.get("TDMSingleIssuer"):
    return mod
  # One issuer wave per tensor; each ends with its own adv params live.
  issuers = (("MXSA", ISSUER_MXSA), ("MXSB", ISSUER_MXSB),
             ("A", ISSUER_A), ("B", ISSUER_B))
  for tc, issuer in issuers:
    if tc.startswith("MX"):
      if not (kernel["ProblemType"].get("MXBlock%s" % tc[-1], 0)
              and kernel.get("enableTDM%s" % tc[-1], False)):
        continue
    elif not kernel.get("enableTDM%s" % tc[-1], False):
      continue
    tP = _tdmSingleIssuerTensorParams(writer, tc)
    name = writer.labels.getUniqueNamePrefix("tdm_si_init_%s" % tc)
    skip = Label(name, "")
    mod.add(SCmpEQU32(src0=sgpr("WaveIdx"), src1=issuer,
                      comment="TDMSingleIssuer: init %s on wave %d" % (tc, issuer)))
    mod.add(SCBranchSCC0(labelName=skip.getLabelName(),
                         comment="other waves skip %s TDM init" % tc))
    mod.add(initTDMDescriptorSubtile(writer, kernel, tP, refresh=True))
    # Load xor mask + globalInc for the issuer advance (SwapMask not used).
    _initTDMSingleIssuerAdvParams(mod, writer, kernel, tc)
    mod.add(skip)
  return mod


def tdmApplyTileKOffsetSubtile(writer, kernel, tP):
  """Apply the StreamK K-offset to the subtile TDM descriptor.

  StreamK=3 DP-partial work items have a nonzero StreamKLocalStart and must read
  their own K-slice. Advance Address{tc} by StreamKLocalStart unroll iterations
  (inc = ti.depthUBytes, matching _emitGRPtrUpdate_TLU0's per-iteration advance)
  and re-sync the descriptor. No-op when StreamKLocalStart == 0.
  """
  tc = tP["tensorChar"]
  ti = _subtileTileInfo(writer, tc)
  group0 = f"tdm{tc}Group0"
  mod = Module(f"TDM StreamK K-offset subtile {tc}")
  # DP-only: StreamKLocalStart == 0, so the K-start offset is 0 and this is a
  # no-op. StreamKLocalStart is not allocated in DP-only mode.
  if isPersistentDataParallel(kernel):
    return mod
  with writer.allocTmpSgpr(2, alignment=2, tag="tdmSkOffset") as tmpSgprRes:
    o = tmpSgprRes.idx
    if tc.startswith("MX"):
      from .SubtileScaleEmit import emitScaleTdmStepBytes, scaleRowPitch
      rows = scaleRowPitch(mod, kernel, tc, o)
      mod.add(SMulI32(dst=sgpr(o), src0=sgpr("StreamKLocalStart"), src1=rows,
                      comment="SK K-start * scale rows"))
      mod.add(emitScaleTdmStepBytes(ti, kernel, sgpr(o), sgpr(o), "SK K-start * scale step"))
    else:
      inc = int(ti.depthUBytes)  # per-unroll TDM advance; same source as _emitGRPtrUpdate_TLU0
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
def globalReadPtrUpdates(tc, writer, kernel, holdOnLastIter=False):
  ti_ = writer.states.a.tileInfo if tc == 'A' else writer.states.b.tileInfo
  return ti_.emitGRPtrUpdate(writer, kernel, holdOnLastIter)


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
