# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""TDMStore: store D through LDS with tensor_store_from_lds in place of buffer_store (gfx1250).

The store blocks a TDMStoreInst level picks store through the TDM block, which stages the converted D
tile in LDS over the main-loop buffers and stores it with one tensor_store_from_lds per wave,
bounded to D by tensor_dim. With beta != 0 it first loads the C tile into the staged tile with
tensor_load_to_lds, bounded the same way. Each level adds one path to the level below:

  =====  =======================================================================
  Level  Paths stored through the TDM block
  =====  =======================================================================
  0      none
  1      EdgeM: the masked paths narrower than the store vector width, and the narrowest one
  2      EdgeN too: the masked path of the store vector width, so every edge tile
  3      NonEdge too: the unmasked path, so every tile, the OptNLL store included
  =====  =======================================================================

Level 3 stores every tile through the TDM block, whatever the alignment of its columns. Where the
columns of D do not start on a 16-byte boundary the TDM store is slower than buffer_store, so tuning
chooses between levels 2 and 3.

A path is a store block of the store-width cascade generateBetaModules builds, one per vector width,
and the level only picks the blocks that store through the TDM block: no level adds, drops or moves a
block, label or branch. With AdaptiveGemm=0 the narrowest masked block is the only one narrower than
the store vector width, which every M-edge workgroup takes. With AdaptiveGemm=1 an M-edge workgroup
takes the widest block whose vector width divides SizeI, so the masked block of the store vector
width also stores the M-edge tiles of a SizeI it divides. At store vector width 1 the narrowest masked
block is the one of the store vector width, so levels 1 and 2 store that one block through the TDM block.

The block runs in the phases of GlobalWriteBatchWriter. With beta != 0, _prolog of the first batch
calls emitTDMStoreSetup and emitTDMStoreLoadC, so the C tile is in LDS before any element reads it,
and every element reads its C with emitTDMStoreLoadCElement in place of the buffer load. With
beta == 0, _emitNonatomicAdd calls emitTDMStoreSetup for the first batch. _emitNonatomicAdd stages
the D of every batch with emitTDMStoreStaging and calls emitTDMStoreFlush after the last batch.
generateBetaModules picks the blocks with tdmStoreBlocks; Solution checks a configuration
with tdmStoreRejectReason and reserves the staged tile with tdmStoreLdsBytes, at _TDMStoreLdsOffset after
the epilogue vectors, over the main-loop buffers when _TDMStoreOverMainLoop.
"""

from math import log2

from rocisa.code import Label, Module
from rocisa.container import DSModifiers, MemTokenData, sgpr, vgpr
from rocisa.enum import NonVolatile, TemporalHint
from rocisa.instruction import DSLoadD16HIU16, DSLoadU8, SAddCU32, SAddU32, SCBranchSCC1, SCmpLeU32, SLShiftLeftB64, \
  SLShiftRightB32, SMinU32, SMulHIU32, SMulI32, SSubU32, SWaitCnt, SWaitTensorcnt, VAddU32, VLShiftLeftB32, \
  VMulLOU32, VReadfirstlaneB32, VSubU32

from ..AsmMemoryHelpers import dsLoad, dsStore
from ..ExecutionPolicy import isPersistent
from .NonTemporal import tdmCacheModifiers

# PackData and TensorDataMover are imported where they are used: they load every component
# (Component.py), and GlobalWriteBatch imports Solution, which imports this module.

TDM_STORE_PATHS = {
  0: (),
  1: ("EdgeM",),
  2: ("EdgeM", "EdgeN"),
  3: ("EdgeM", "EdgeN", "NonEdge"),
}

# StinkyRemoveWaitCntPass never removes a wait whose comment contains this tag. The TDM block tags
# the waits that WaitCntInsertion does not re-derive.
STINKY_KEEP = "[stinky-keep]"


def hasEpilogueSideEffects(pt):
  """Whether the epilogue writes or reduces per-element outputs beyond C and D (E, Gradient, gate residual,
  AmaxD). The TDM blocks of the edge paths of such a kernel keep the masks, the vector width and the C loads of
  the edge block, so the elements outside D reach neither those outputs nor their reductions."""
  return pt["UseE"] or pt["Gradient"] or pt["UseGateResidual"] or pt["OutputAmaxD"]


def tdmStoreRejectReason(state):
  """Return why a TDMStoreInst solution is outside the configurations the TDM block is built for, or None.

  Solution rejects TDMStoreInst without the HasTDM assembler capability before it asks this. The TDM
  block stages the converted D tile in LDS and stores it once, after the last store batch, with one
  tensor_store_from_lds per wave. That needs fp32 D, fp64 or complex float D from accumulators of their type, int32 or
  int8 D from int32 accumulators, or bf16, fp16, fp8 or bf8 D converted from fp32 accumulators, written directly to D, by a GlobalSplitU=1 kernel, by
  the workgroup that reduces a MultipleBufferSingleKernel tile (below), or by a persistent kernel,
  which stores each tile it completes, and no LDS data kept from one persistent tile to the next.
  With PrefetchAcrossPersistent the next tile's loads write the main-loop buffers during the epilogue,
  so the staged tile lies after them (_TDMStoreOverMainLoop is False) and its LDS accesses carry the
  epilogue's memory token. ReuseAcrossPersistent keeps A in VGPRs from one tile to the next, so the
  staged tile overwrites nothing it reuses.
  The block also computes the elements outside D and bounds only the C loads and the D store, so
  besides C and D the epilogue may only read the epilogue vectors, bias, ScaleAlphaVec and ScaleAB along
  M or N, which the staged tile follows in LDS, and scalars such as ScaleC and ScaleD. With per-element
  outputs beyond C and D (hasEpilogueSideEffects) the TDM blocks of the edge paths are generated as the edge
  block, with its masks, vector width and C loads, and only stage D in place of its store. GroupedGemm sets the sizes, strides and addresses
  the block reads for each group.
  A GlobalSplitU=-1 kernel, with any GlobalSplitUAlgorithm and AdaptiveGemmGSUA, uses the block when
  the runtime GSU is 1; the partials and atomics of its GSU>1 branch keep buffer_store. The reduced tile of a
  MultipleBufferSingleKernel GSU>1 branch goes to SrdTD, which addresses D while SrdD addresses the workspace, at
  the size of D. The block stores a tile that every wave of the workgroup staged, so the reduced tile takes it
  only where one workgroup reduces the tile with all its waves: with MbskPrefetchMethod 0 and a runtime GSU of at
  most gsuThreshold (2), where the workgroup of GSUSumIdx GSU-1 waits for the others. For a larger GSU, and with
  MbskPrefetchMethod 1, each wave takes the last decrement of its own synchronizer slot and reduces on its own.
  With MbskPrefetchMethod 0 globalWriteElements therefore generates the reduced tile twice, through the block and
  through buffer_store, and the runtime GSU picks the copy; with MbskPrefetchMethod 1 the reduced tile keeps
  buffer_store. A static GlobalSplitU > 1 shares its kernel with GlobalSplitU=-1 and a split-K setting overrides
  it, so the copy is picked at run time for it as well.
  StoreRemapVectorWidth applies to the blocks that keep buffer_store; a TDM block stages D itself and
  skips the remap. In a block whose stores run in the CompactLoopStore loop, the block sets up the staged tile and loads C
  before the loop, steps the staged tile's base by each iteration's rows and stores the tile after the loop.
  """
  if not state["TDMStoreInst"]:
    return None

  pt = state["ProblemType"]
  sixteenBit = pt["DestDataType"].isBFloat16() or pt["DestDataType"].isHalf()
  converted = sixteenBit or pt["DestDataType"].is8bitFloat()
  integer = (pt["DestDataType"].isInt32() or pt["DestDataType"].isInt8()) and pt["ComputeDataType"].isInt32()
  accumType = (pt["DestDataType"].isDouble() and pt["ComputeDataType"].isDouble()) \
    or (pt["DestDataType"].isSingleComplex() and pt["ComputeDataType"].isSingleComplex()) \
    or (pt["DestDataType"].isDoubleComplex() and pt["ComputeDataType"].isDoubleComplex())
  if not (integer or accumType) and (not pt["ComputeDataType"].isSingle()
                                    or not (pt["DestDataType"].isSingle() or (converted and pt["HighPrecisionAccumulate"]))):
    return "TDMStoreInst supports fp32 D, fp64, complex float and complex double D from accumulators of their type, " \
           "int32 and int8 D from int32 accumulators, and bf16, fp16, fp8 and bf8 D converted from fp32 accumulators"
  if not isPersistent(state) and state["GlobalSplitU"] not in (1, -1) \
     and state["GlobalSplitUAlgorithm"] != "MultipleBufferSingleKernel":
    return "TDMStoreInst requires GlobalSplitU=1 or -1, or MultipleBufferSingleKernel"
  if state["UseSubtileImpl"]:
    return "TDMStoreInst does not support UseSubtileImpl"
  if pt["FusedGemmA2A"]:
    return "TDMStoreInst does not support FusedGemmA2A"
  return None


def _waveColumns(kernel):
  """The columns of the tile each wave stores: MacroTile1 over the waves, rounded up."""
  return -(-kernel["MacroTile1"] // (kernel["NumThreads"] // kernel["WavefrontSize"]))


def _unevenColumns(kernel):
  """Whether MacroTile1 is not a multiple of the number of waves, so the columns of the last waves reach past
  the tile."""
  return kernel["MacroTile1"] % (kernel["NumThreads"] // kernel["WavefrontSize"]) != 0


def tdmStoreLdsBytes(state):
  """Return the LDS bytes of the D tile the TDM block stages in LDS (0 without TDMStoreInst).

  The reservation follows the level, not the store blocks tdmStoreBlocks picks for the kernel, and covers
  the columns of every wave, MacroTile1 rounded up to a multiple of the number of waves.
  Reserving too much costs LDS only; reserving too little would stage over live LDS.
  """
  if not state["TDMStoreInst"]:
    return 0
  waves = state["NumThreads"] // state["WavefrontSize"]
  return state["MacroTile0"] * _waveColumns(state) * waves * int(state["ProblemType"]["DestDataType"].numBytes())


def storesFinalD(kernel, atomic):
  """Whether a store block writes D's final elements: not atomically and not to a GSU workspace. The
  MultipleBufferSingleKernel store writes the reduced tile to D; it counts with MbskPrefetchMethod 0, whose kernel
  also keeps a buffer_store copy of it (tdmStoreRejectReason)."""
  if kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":
    return not atomic and kernel.get("MbskPrefetchMethod") == 0
  return not atomic and kernel["_GlobalAccumulation"] in (None, "PartialsBuffer")


def tdmStoreBlocks(kernel, edge, atomic, bpe, vectorWidths):
  """Return the vector widths of the store blocks of one beta that store through the TDM block.

  vectorWidths are the store blocks generateBetaModules builds, the store-width cascade: the widest,
  of the store vector width, holds the NonEdge path and, with edge, the masked EdgeN path; the
  narrower ones are masked. EdgeM is every masked block narrower than the store vector width and the
  narrowest one; without edge the one block is unmasked, NonEdge, and generateBetaModules picks the
  NonEdge path of a store with edge the same way, with edge=False. In a picked block the TDM block
  replaces the store; it is generated like the NonEdge path (widest vector width, no per-element
  mask). Only a block that storesFinalD uses it, not a GSU or StreamK partial-tile store, and it stages
  D at the size of D (bpe).
  """
  paths = TDM_STORE_PATHS[kernel["TDMStoreInst"]] if storesFinalD(kernel, atomic) else ()
  assert not paths or "ProblemType" not in kernel or bpe == int(kernel["ProblemType"]["DestDataType"].numBytes()), \
    "a TDM store block stages D at the size of D"
  storeVectorWidth = max(vectorWidths)
  blocks = {"EdgeM": {w for w in vectorWidths if w < storeVectorWidth} | {min(vectorWidths)},
            "EdgeN": {storeVectorWidth}} if edge else {"NonEdge": {storeVectorWidth}}
  return set().union(*(blocks.get(path, ()) for path in paths))


def tdmStoreModifiers(asmCaps, kernel, tc="D"):
  """Return the TENSORModifiers of the TDM block's transfer of tensor tc, from its hints as addStore and
  addLoad decode them: the tensor_store_from_lds of D, or the tensor_load_to_lds of C."""
  hints = [(kernel["NonTemporal%s" % tc], TemporalHint(kernel.get("TemporalHint%s" % tc, 0)),
            NonVolatile(kernel.get("NonVolatile%s" % tc, 0)))]
  return tdmCacheModifiers(asmCaps, hints, is_store=tc == "D")


def emitTDMStoreSetup(gwb) -> Module:
  """Fence the main-loop LDS when the staged tile lies over it, and compute this thread's byte address
  in the staged D tile.

  The tile is staged M-contiguous from LDS offset _TDMStoreLdsOffset, after the epilogue vectors. When it
  lies over the main-loop buffers (_TDMStoreOverMainLoop), every wave must be done with them, including
  in-flight tensor_load_to_lds, before the staged tile is written.
  """
  module = Module("TDMStoreSetup")
  kw = gwb.parentWriter
  if gwb.kernel["_TDMStoreOverMainLoop"]:
    if gwb.kernel["enableTDMA"] or gwb.kernel["enableTDMB"]:
      module.add(SWaitTensorcnt(tensorcnt=0, comment="drain tensor_load_to_lds before reusing its LDS"))
    module.add(kw._syncThreads(gwb.kernel, "all waves are done with the main-loop LDS"))
  base = gwb.tdmStoreLdsBase
  mt0 = gwb.kernel["MacroTile0"]
  module.add(SMulI32(dst=sgpr(gwb.tmpS01), src0=sgpr("WorkGroup1"), src1=gwb.kernel["MacroTile1"], comment="wg1*MT1"))
  module.add(VSubU32(dst=vgpr(base), src0=vgpr(kw.vgprs.coord1), src1=sgpr(gwb.tmpS01), comment="tile column = coord1 - wg1*MT1"))
  module.add(VMulLOU32(dst=vgpr(base), src0=vgpr(base), src1=mt0, comment="* MT0"))
  module.add(VAddU32(dst=vgpr(base), src0=vgpr(base), src1=vgpr(kw.vgprs.coord0), comment="+ coord0"))
  module.add(SMulI32(dst=sgpr(gwb.tmpS01), src0=sgpr("WorkGroup0"), src1=mt0, comment="wg0*MT0"))
  module.add(VSubU32(dst=vgpr(base), src0=vgpr(base), src1=sgpr(gwb.tmpS01), comment="- wg0*MT0"))
  module.add(VLShiftLeftB32(dst=vgpr(base), shiftHex=int(log2(kw.states.bpeCexternalGSU1)), src=vgpr(base), comment="* bpe"))
  if gwb.kernel["_TDMStoreLdsOffset"]:
    module.add(VAddU32(dst=vgpr(base), src0=gwb.kernel["_TDMStoreLdsOffset"], src1=vgpr(base), comment="+ LDS offset of the staged tile"))
  return module


def _tdmStoreLdsAddress(gwb, module: Module, coordOffset0, coordOffset1):
  """Return the VGPR and DS offset of element (coordOffset0, coordOffset1) of the staged tile."""
  offset = (coordOffset1 * gwb.kernel["MacroTile0"] + coordOffset0) * gwb.parentWriter.states.bpeCexternalGSU1
  if offset <= 0xFFFF:
    return gwb.tdmStoreLdsBase, offset
  # The DS instruction offset has 16 bits.
  module.add(VAddU32(dst=vgpr(gwb.tmpVgpr), src0=vgpr(gwb.tdmStoreLdsBase), src1=hex(offset & ~0xFFFF), comment="LDS offset above 64 KiB"))
  return gwb.tmpVgpr, offset & 0xFFFF


def _stagedTileToken(gwb):
  """The memory token of the staged tile's LDS accesses: that of the main-loop buffer the tile lies over or,
  when it lies after the main-loop buffers (PrefetchAcrossPersistent), the epilogue's token, so StinkyTofu does
  not order them after the next tile's loads into those buffers."""
  states = gwb.parentWriter.states
  return states.memTokenLdsBuffer0 if gwb.kernel["_TDMStoreOverMainLoop"] else states.memTokenEpilogue


def emitTDMStoreLoadCElement(gwb, addrCalc, data, vc0) -> Module:
  """Read this element's C vector from the staged tile, where its D is staged later.

  An odd 2-byte fp16 element goes to the high half of its own VGPR, where readInput loads it, with
  ds_load_u16_d16_hi. A 1-byte element, fp8, bf8 or int8 C at vector width 1, goes to the low byte of its VGPR with
  ds_load_u8, which zero-fills the rest as readInput's byte load does on gfx1250. Only that half is read afterwards, so it does not matter that a d16 load writes
  all 32 bits of its VGPR on gfx1250, where ECC zero-fills the other half. A DS load reads at most
  16 bytes, so a wider vector, such as bf16 at store vector width 16, is read 16 bytes at a time.
  """
  module = Module("TDMStoreLoadCElement")
  kw = gwb.parentWriter
  bpl = kw.states.bpeCexternalGSU1 * gwb.gwvw
  addr, offset = _tdmStoreLdsAddress(gwb, module, addrCalc.coordOffset0, addrCalc.coordOffset1)
  comment = "load C element (%u, %u)" % (addrCalc.coordOffset0, addrCalc.coordOffset1)
  memToken = MemTokenData([_stagedTileToken(gwb)])
  if bpl == 2 and gwb.kernel["ProblemType"]["DestDataType"].isHalf() and not kw.states.HHH_WMMA and vc0 % 2:
    inst = DSLoadD16HIU16(dst=vgpr(data), src=vgpr(addr), ds=DSModifiers(offset=offset), comment=comment)
    inst.setMemToken(memToken)
    module.add(inst)
    return module
  if bpl == 1:
    inst = DSLoadU8(dst=vgpr(data), src=vgpr(addr), ds=DSModifiers(offset=offset), comment=comment)
    inst.setMemToken(memToken)
    module.add(inst)
    return module
  width = min(bpl, 16)
  for piece in range(bpl // width):
    module.add(dsLoad(width, dst=vgpr(data + piece * width // kw.states.bpr, max(1, width // kw.states.bpr)),
                      src=vgpr(addr), ds=DSModifiers(offset=offset + piece * width), comment=comment, memToken=memToken))
  return module


def emitTDMStoreStaging(gwb, stagedElements) -> Module:
  """Write this batch's packed D elements to their (M, N) position in the staged tile.

  An element spans the registers of its type from ValuC + sumIdx times their number, as addStore stores it: two
  for fp64 and complex float. dsStore writes at most 32 bytes, as two 16-byte stores, so a wider vector, such as fp32 at store vector
  width 16, is staged 32 bytes at a time.
  """
  from .PackData import formatting
  module = Module("TDMStoreStaging")
  kw = gwb.parentWriter
  bps = kw.states.bpeCexternalGSU1 * gwb.gwvw
  width = min(bps, 32)
  numRegs = max(1, width // kw.states.bpr)
  regsPerElement = max(1, kw.states.bpeCexternalGSU1 // kw.states.bpr)
  for sumIdx, coordOffset0, coordOffset1 in stagedElements:
    addr, offset = _tdmStoreLdsAddress(gwb, module, coordOffset0, coordOffset1)
    for piece in range(bps // width):
      src = vgpr(formatting(sumIdx * regsPerElement + piece * numRegs, "ValuC+", kw.states.c.startVgprValu), numRegs)
      module.add(dsStore(width, dstAddr=vgpr(addr), src=src, ds=DSModifiers(offset=offset + piece * width),
                         comment="stage D element (%u, %u)" % (coordOffset0, coordOffset1),
                         memToken=MemTokenData([_stagedTileToken(gwb)])))
  return module


def _tdmStoreDescriptor(gwb, module: Module, tdm, g0, g1, tmp, tc):
  """Fill the descriptor of this wave's columns of the tile of tensor tc (C or D) and of the staged tile.

  Each wave covers ceil(MacroTile1/numWaves) consecutive columns: tile MT0 x those columns, M-contiguous
  from the LDS address of its first column, global address Srd<tc> (SrdTD for the reduced tile of MultipleBufferSingleKernel) + (column*Stride<tc> + wg0*MT0)*bpe,
  and tensor_dim0/1 the part of the tile inside SizeI and SizeJ. The first of the four tmp SGPRs is left
  at the first column of the wave. When MacroTile1 is not a multiple of the number of waves, the columns
  of the last waves reach past the tile, so tensor_dim1 ends at the column bound min(SizeJ, (wg1+1)*MT1),
  which the second tmp SGPR keeps. The block assumes one free index per operand, as every GEMM problem
  type has, so PackedC0IndicesX and PackedC1IndicesX hold one index each. TDM data_size is at most 8 bytes, so a
  complex double element is two 8-byte units, in which tensor_dim0, tile0 and the stride count; the byte offsets
  and the staged tile do not change.
  """
  assert len(gwb.kernel["PackedC0IndicesX"]) == 1 and len(gwb.kernel["PackedC1IndicesX"]) == 1, \
    "the TDM block assumes one free index per operand"
  kw = gwb.parentWriter
  bpe = kw.states.bpeCexternalGSU1
  units = bpe // min(bpe, 8)
  mt0 = gwb.kernel["MacroTile0"]
  ws = gwb.kernel["WavefrontSize"]
  colsPerWave = _waveColumns(gwb.kernel)
  stride1 = "Stride%s%s" % (tc, kw.states.indexChars[gwb.kernel["PackedC1IndicesX"][0]])
  srd = "SrdTD" if tc == "D" and gwb.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel" else "Srd" + tc
  sizeI = kw.sizeRef(gwb.kernel["ProblemType"]["Index0"])
  sizeJ = kw.sizeRef(gwb.kernel["ProblemType"]["Index1"])
  col, dim, offLo, offHi = tmp.idx, tmp.idx + 1, tmp.idx + 2, tmp.idx + 3
  module.add(tdm.initOperands(g0, g1))
  module.add(tdm.setDataSize(bpe // units, g1))
  module.add(VReadfirstlaneB32(dst=sgpr(col), src=vgpr("Serial"), comment="first thread id"))
  module.add(SLShiftRightB32(dst=sgpr(col), shiftHex=int(log2(ws)), src=sgpr(col), comment="waveId"))
  module.add(SMulI32(dst=sgpr(col), src0=sgpr(col), src1=colsPerWave, comment="first tile column of this wave"))
  module.add(SMulI32(dst=sgpr(offLo), src0=sgpr(col), src1=mt0 * bpe, comment="its LDS address"))
  if gwb.kernel["_TDMStoreLdsOffset"]:
    module.add(SAddU32(dst=sgpr(offLo), src0=sgpr(offLo), src1=gwb.kernel["_TDMStoreLdsOffset"], comment="+ LDS offset of the staged tile"))
  module.add(tdm.setLdsAddr(g0, sgpr(offLo)))
  module.add(SMulHIU32(dst=sgpr(offHi), src0=sgpr(col), src1=sgpr(stride1), comment="hi(column*Stride%s)" % tc))
  module.add(SMulI32(dst=sgpr(offLo), src0=sgpr(col), src1=sgpr(stride1), comment="lo(column*Stride%s)" % tc))
  module.add(SMulI32(dst=sgpr(dim), src0=sgpr("WorkGroup0"), src1=mt0, comment="wg0*MT0"))
  module.add(SAddU32(dst=sgpr(offLo), src0=sgpr(offLo), src1=sgpr(dim), comment="+ wg0*MT0"))
  module.add(SAddCU32(dst=sgpr(offHi), src0=sgpr(offHi), src1=0, comment="carry"))
  module.add(SLShiftLeftB64(dst=sgpr(offLo, 2), src=sgpr(offLo, 2), shiftHex=int(log2(bpe)), comment="64-bit byte offset from %s" % srd))
  module.add(tdm.setGlobalAddr(g0, srd))
  module.add(SAddU32(dst=sgpr(g0 + 2), src0=sgpr(g0 + 2), src1=sgpr(offLo), comment="global address lo"))
  module.add(SAddCU32(dst=sgpr(g0 + 3), src0=sgpr(g0 + 3), src1=sgpr(offHi), comment="global address hi"))
  module.add(SSubU32(dst=sgpr(dim), src0=sizeI, src1=sgpr(dim), comment="tensor_dim0 = SizeI - wg0*MT0"))
  if units > 1:
    module.add(SAddU32(dst=sgpr(dim), src0=sgpr(dim), src1=sgpr(dim), comment="tensor_dim0 in 8-byte units"))
  module.add(tdm.setTensorDim0(g1, dim, kw))
  module.add(SMulI32(dst=sgpr(dim), src0=sgpr("WorkGroup1"), src1=gwb.kernel["MacroTile1"], comment="wg1*MT1"))
  module.add(SAddU32(dst=sgpr(col), src0=sgpr(col), src1=sgpr(dim), comment="first %s column of this wave" % tc))
  if _unevenColumns(gwb.kernel):
    module.add(SAddU32(dst=sgpr(dim), src0=sgpr(dim), src1=gwb.kernel["MacroTile1"], comment="end of the tile"))
    module.add(SMinU32(dst=sgpr(dim), src0=sizeJ, src1=sgpr(dim), comment="column bound = min(SizeJ, end of the tile)"))
    module.add(SSubU32(dst=sgpr(offLo), src0=sgpr(dim), src1=sgpr(col), comment="tensor_dim1 = column bound - first column"))
    module.add(tdm.setTensorDim1(g1, offLo, kw))
  else:
    module.add(SSubU32(dst=sgpr(dim), src0=sizeJ, src1=sgpr(col), comment="tensor_dim1 = SizeJ - first column"))
    module.add(tdm.setTensorDim1(g1, dim, kw))
  module.add(tdm.setTensorTile0(g1, mt0 * units))
  module.add(tdm.setTensorTile1(g1, colsPerWave))
  if units > 1:
    module.add(SAddU32(dst=sgpr(offHi), src0=sgpr(stride1), src1=sgpr(stride1), comment="Stride%s in 8-byte units" % tc))
    module.add(tdm.setTensorStride0(g1, offHi))
  else:
    module.add(tdm.setTensorStride0(g1, stride1))


def _skipTest(gwb, sizeJ, tmp):
  """The compare whose SCC skips this wave's transfer: none of its columns is before the column bound, SizeJ or,
  when MacroTile1 is not a multiple of the number of waves, min(SizeJ, end of the tile)."""
  if _unevenColumns(gwb.kernel):
    return SCmpLeU32(src0=sgpr(tmp.idx + 1), src1=sgpr(tmp.idx), comment="all columns of this wave at or past the column bound?")
  return SCmpLeU32(src0=sizeJ, src1=sgpr(tmp.idx), comment="all columns of this wave at or past SizeJ?")


def emitTDMStoreLoadC(gwb) -> Module:
  """Load the C tile into the staged tile with one tensor_load_to_lds per wave.

  Each wave loads the columns it stores. tensor_dim0/1 clip the load at the end of C, so no load
  reads past it; a wave whose columns all lie at or past the column bound skips its load. Each thread then
  reads its C elements where it stages their D, so only the barrier after the loads is needed. The
  load carries C's cache hints, as the buffer loads of C would.
  """
  from .TensorDataMover import TensorDataMoverStore
  module = Module("TDMStoreLoadC")
  kw = gwb.parentWriter
  sizeJ = kw.sizeRef(gwb.kernel["ProblemType"]["Index1"])
  skipLoad = Label(kw.labels.getNameInc("TDMStoreLoadCSkip"), "no column of this wave is inside C")
  tdm = TensorDataMoverStore([_stagedTileToken(gwb)])
  with kw.allocTmpSgpr(8, alignment=4, tag="tdmLoadCDescGroup1") as group1, \
      kw.allocTmpSgpr(4, alignment=4, tag="tdmLoadCDescGroup0") as group0:
    with kw.allocTmpSgpr(4, alignment=2, tag="tdmLoadCTmp") as tmp:
      _tdmStoreDescriptor(gwb, module, tdm, group0.idx, group1.idx, tmp, "C")
      module.add(_skipTest(gwb, sizeJ, tmp))
      module.add(SCBranchSCC1(labelName=skipLoad.getLabelName(), comment="skip the load"))
      module.add(tdm.issueLoad(group0.idx, group1.idx, tensor=tdmStoreModifiers(kw.states.asmCaps, gwb.kernel, "C")))
      # StinkyRemoveWaitCntPass keeps a wait marked stinky-keep; nothing re-derives this one.
      module.add(SWaitTensorcnt(tensorcnt=0, comment="wait for the TDM load of C " + STINKY_KEEP))
      module.add(skipLoad)
  module.add(kw._syncThreads(gwb.kernel, "every wave has loaded its part of the C tile"))
  return module


def emitTDMStoreFlush(gwb) -> Module:
  """Store the staged D tile with one tensor_store_from_lds per wave.

  Each wave stores ceil(MacroTile1/numWaves) consecutive columns of the tile. In a persistent kernel the next tile
  must not write LDS before every wave's store has read the staged tile: the barrier that ends the tile
  handoff orders that when the handoff has one (WorkAssignment.handoffFencesLds), and otherwise the flush
  ends with its own. tensor_dim0/1 bound
  the store to the part of the tile inside D; a wave whose columns all lie at or past the column
  bound skips its store instead of relying on a zero tensor_dim.
  """
  from .TensorDataMover import TensorDataMoverStore
  module = Module("TDMStoreFlush")
  kw = gwb.parentWriter
  sizeJ = kw.sizeRef(gwb.kernel["ProblemType"]["Index1"])
  skipStore = Label(kw.labels.getNameInc("TDMStoreSkip"), "no column of this wave is inside D")
  tdm = TensorDataMoverStore([_stagedTileToken(gwb)])
  with kw.allocTmpSgpr(8, alignment=4, tag="tdmStoreDescGroup1") as group1, \
      kw.allocTmpSgpr(4, alignment=4, tag="tdmStoreDescGroup0") as group0:
    with kw.allocTmpSgpr(4, alignment=2, tag="tdmStoreTmp") as tmp:
      _tdmStoreDescriptor(gwb, module, tdm, group0.idx, group1.idx, tmp, "D")
      # StinkyRemoveWaitCntPass keeps a wait marked stinky-keep; nothing re-derives this one.
      module.add(SWaitCnt(dscnt=0, comment="staging stores done " + STINKY_KEEP))
      module.add(kw._syncThreads(gwb.kernel, "every wave has staged its part of the tile"))
      module.add(_skipTest(gwb, sizeJ, tmp))
      module.add(SCBranchSCC1(labelName=skipStore.getLabelName(), comment="skip the store"))
      module.add(tdm.issueStore(group0.idx, group1.idx, tensor=tdmStoreModifiers(kw.states.asmCaps, gwb.kernel)))
      # Nor this one: after the OptNLL store, nothing re-derives it before s_endpgm.
      module.add(SWaitTensorcnt(tensorcnt=0, comment="wait for the TDM store " + STINKY_KEEP))
      module.add(skipStore)
  from ..Component import Component
  if isPersistent(gwb.kernel) and not Component.WorkAssignment.find(kw).handoffFencesLds(gwb.kernel):
    module.add(kw._syncThreads(gwb.kernel, "every TDM store is done with the staged tile before the next tile writes LDS"))
  return module
