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
the epilogue vectors.
"""

from math import log2

from rocisa.code import Label, Module
from rocisa.container import DSModifiers, MemTokenData, sgpr, vgpr
from rocisa.enum import NonVolatile, TemporalHint
from rocisa.instruction import DSLoadD16HIU16, SAddCU32, SAddU32, SCBranchSCC1, SCmpLeU32, SLShiftLeftB64, \
  SLShiftRightB32, SMulHIU32, SMulI32, SSubU32, SWaitCnt, SWaitTensorcnt, VAddU32, VLShiftLeftB32, VMulLOU32, \
  VReadfirstlaneB32, VSubU32

from ..AsmMemoryHelpers import dsLoad, dsStore
from ..ExecutionPolicy import hasStaticAssignment, isPersistent, isStreamKSpatialCluster, \
  requiresPartialReduction
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


def tdmStoreRejectReason(state):
  """Return why a TDMStoreInst solution is outside the configurations the TDM block is built for, or None.

  The TDM block stages the converted D tile in LDS over the main-loop buffers and stores it once,
  after the last store batch, with one tensor_store_from_lds per wave. That needs a 16-bit HPA
  output written directly to D, by a GlobalSplitU=1 kernel or as the final store of a non-atomic
  StreamK=3 tile, no LDS data kept from one persistent tile to the next, and MacroTile1 split
  evenly across the waves. The block also computes the elements outside D and bounds only the C
  loads and the D store, so besides C and D the epilogue may only read the bias and ScaleAlphaVec
  vectors along M and the ScaleAB vectors, which the staged tile follows in LDS.
  A GlobalSplitU=-1 kernel uses the block only when the runtime GSU is 1, and its GSU>1
  partials keep buffer_store; it is supported with MultipleBuffer and AdaptiveGemmGSUA=0.
  """
  if not state["TDMStoreInst"]:
    return None

  pt = state["ProblemType"]
  if tuple(state["ISA"])[:2] != (12, 5):
    return "TDMStoreInst requires gfx1250"
  if not (pt["DestDataType"].isBFloat16() or pt["DestDataType"].isHalf()):
    return "TDMStoreInst supports only bf16 and fp16 DestDataType"
  if not pt["HighPrecisionAccumulate"] or not pt["ComputeDataType"].isSingle():
    return "TDMStoreInst requires HighPrecisionAccumulate with a single-precision ComputeDataType"
  if not state["BufferStore"]:
    return "TDMStoreInst requires BufferStore"
  if isPersistent(state) and not (requiresPartialReduction(state) and hasStaticAssignment(state)):
    return "TDMStoreInst supports persistent kernels only as non-atomic StreamK=3"
  if state["PrefetchAcrossPersistent"] or state["ReuseAcrossPersistent"] or isStreamKSpatialCluster(state):
    return "TDMStoreInst does not support PrefetchAcrossPersistent, ReuseAcrossPersistent or StreamK spatial clusters"
  if not isPersistent(state) and state["GlobalSplitU"] not in (1, -1):
    return "TDMStoreInst requires GlobalSplitU=1 or -1"
  if state["GlobalSplitU"] == -1 and (state["GlobalSplitUAlgorithm"] != "MultipleBuffer" or state["AdaptiveGemmGSUA"]):
    return "TDMStoreInst supports GlobalSplitU=-1 only with MultipleBuffer and AdaptiveGemmGSUA=0"
  if state["UseSubtileImpl"]:
    return "TDMStoreInst does not support UseSubtileImpl"
  # -1 (auto) is resolved later, where Solution rejects TDMStoreInst if it resolves to a remap width.
  if state["StoreRemapVectorWidth"] > 0:
    return "TDMStoreInst does not support StoreRemapVectorWidth"
  if state["CompactLoopStore"]:
    return "TDMStoreInst does not support CompactLoopStore"
  if pt["FusedGemmA2A"]:
    return "TDMStoreInst does not support FusedGemmA2A"
  if pt["UseE"] or pt["Gradient"] or pt["UseGateResidual"] or pt["UseScaleCD"] or pt["OutputAmaxD"] \
      or pt["GroupedGemm"]:
    return "TDMStoreInst does not support E, Gradient, gate residual, ScaleCD, AmaxD or GroupedGemm"
  if pt["UseBias"] > 1 or pt["UseScaleAlphaVec"] > 1:
    return "TDMStoreInst supports bias and ScaleAlphaVec only along M"
  if state["MacroTile1"] % (state["NumThreads"] // state["WavefrontSize"]) != 0:
    return "TDMStoreInst requires MacroTile1 to be a multiple of the number of waves"
  return None


def tdmStoreLdsBytes(state):
  """Return the LDS bytes of the D tile the TDM block stages over the main-loop LDS (0 without TDMStoreInst).

  The reservation follows the level, not the store blocks tdmStoreBlocks picks for the kernel.
  Reserving too much costs LDS only; reserving too little would stage over live LDS.
  """
  if not state["TDMStoreInst"]:
    return 0
  return state["MacroTile0"] * state["MacroTile1"] * int(state["ProblemType"]["DestDataType"].numBytes())


def tdmStoreBlocks(kernel, edge, atomic, bpe, vectorWidths):
  """Return the vector widths of the store blocks of one beta that store through the TDM block.

  vectorWidths are the store blocks generateBetaModules builds, the store-width cascade: the widest,
  of the store vector width, holds the NonEdge path and, with edge, the masked EdgeN path; the
  narrower ones are masked. EdgeM is every masked block narrower than the store vector width and the
  narrowest one; without edge the one block is unmasked, NonEdge, and generateBetaModules picks the
  NonEdge path of a store with edge the same way, with edge=False. In a picked block the TDM block
  replaces the store; it is generated like the NonEdge path (widest vector width, no per-element
  mask). Only a direct, non-atomic store to D of elements narrower than 4 bytes (bpe) uses it: not a
  GSU or StreamK partial-tile store, and not fp32, whose staged tile would be twice the 16-bit one.
  """
  paths = TDM_STORE_PATHS[kernel["TDMStoreInst"]] \
    if not atomic and bpe < 4 and kernel["_GlobalAccumulation"] in (None, "PartialsBuffer") else ()
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
  """Fence the main-loop LDS and compute this thread's byte address in the staged D tile.

  The tile is staged M-contiguous from LDS offset _TDMStoreLdsOffset, after the epilogue vectors and
  over the main-loop buffers, so every wave must be done with them, including in-flight
  tensor_load_to_lds, before the staged tile is written.
  """
  module = Module("TDMStoreSetup")
  kw = gwb.parentWriter
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
  module.add(VLShiftLeftB32(dst=vgpr(base), shiftHex=int(log2(kw.states.bpeCexternal)), src=vgpr(base), comment="* bpe"))
  if gwb.kernel["_TDMStoreLdsOffset"]:
    module.add(VAddU32(dst=vgpr(base), src0=gwb.kernel["_TDMStoreLdsOffset"], src1=vgpr(base), comment="+ LDS offset of the staged tile"))
  return module


def _tdmStoreLdsAddress(gwb, module: Module, coordOffset0, coordOffset1):
  """Return the VGPR and DS offset of element (coordOffset0, coordOffset1) of the staged tile."""
  offset = (coordOffset1 * gwb.kernel["MacroTile0"] + coordOffset0) * gwb.parentWriter.states.bpeCexternal
  if offset <= 0xFFFF:
    return gwb.tdmStoreLdsBase, offset
  # The DS instruction offset has 16 bits.
  module.add(VAddU32(dst=vgpr(gwb.tmpVgpr), src0=vgpr(gwb.tdmStoreLdsBase), src1=hex(offset & ~0xFFFF), comment="LDS offset above 64 KiB"))
  return gwb.tmpVgpr, offset & 0xFFFF


def emitTDMStoreLoadCElement(gwb, addrCalc, data, vc0) -> Module:
  """Read this element's C vector from the staged tile, where its D is staged later.

  An odd 2-byte fp16 element goes to the high half of its own VGPR, where readInput loads it, with
  ds_load_u16_d16_hi. Only that half is read afterwards, so it does not matter that a d16 load writes
  all 32 bits of its VGPR on gfx1250, where ECC zero-fills the other half.
  """
  module = Module("TDMStoreLoadCElement")
  kw = gwb.parentWriter
  bpl = kw.states.bpeCexternal * gwb.gwvw
  addr, offset = _tdmStoreLdsAddress(gwb, module, addrCalc.coordOffset0, addrCalc.coordOffset1)
  comment = "load C element (%u, %u)" % (addrCalc.coordOffset0, addrCalc.coordOffset1)
  memToken = MemTokenData([kw.states.memTokenLdsBuffer0])
  if bpl == 2 and gwb.kernel["ProblemType"]["DestDataType"].isHalf() and not kw.states.HHH_WMMA and vc0 % 2:
    inst = DSLoadD16HIU16(dst=vgpr(data), src=vgpr(addr), ds=DSModifiers(offset=offset), comment=comment)
    inst.setMemToken(memToken)
  else:
    inst = dsLoad(bpl, dst=vgpr(data, max(1, bpl // kw.states.bpr)), src=vgpr(addr), ds=DSModifiers(offset=offset),
                  comment=comment, memToken=memToken)
  module.add(inst)
  return module


def emitTDMStoreStaging(gwb, stagedElements) -> Module:
  """Write this batch's packed D elements to their (M, N) position in the staged tile."""
  from .PackData import formatting
  module = Module("TDMStoreStaging")
  kw = gwb.parentWriter
  bps = kw.states.bpeCexternal * gwb.gwvw
  numRegs = max(1, bps // kw.states.bpr)
  for sumIdx, coordOffset0, coordOffset1 in stagedElements:
    addr, offset = _tdmStoreLdsAddress(gwb, module, coordOffset0, coordOffset1)
    module.add(dsStore(bps, dstAddr=vgpr(addr), src=vgpr(formatting(sumIdx, "ValuC+", kw.states.c.startVgprValu), numRegs),
                       ds=DSModifiers(offset=offset), comment="stage D element (%u, %u)" % (coordOffset0, coordOffset1),
                       memToken=MemTokenData([kw.states.memTokenLdsBuffer0])))
  return module


def _tdmStoreDescriptor(gwb, module: Module, tdm, g0, g1, tmp, tc):
  """Fill the descriptor of this wave's columns of the tile of tensor tc (C or D) and of the staged tile.

  Each wave covers MacroTile1/numWaves consecutive columns: tile MT0 x those columns, M-contiguous from
  the LDS address of its first column, global address Srd<tc> + (column*Stride<tc> + wg0*MT0)*bpe, and
  tensor_dim0/1 the part of the tile inside SizeI and SizeJ. The first of the four tmp SGPRs is left at
  the first column of the wave. The block assumes one free index per operand, as every GEMM problem
  type has, so PackedC0IndicesX and PackedC1IndicesX hold one index each.
  """
  assert len(gwb.kernel["PackedC0IndicesX"]) == 1 and len(gwb.kernel["PackedC1IndicesX"]) == 1, \
    "the TDM block assumes one free index per operand"
  kw = gwb.parentWriter
  bpe = kw.states.bpeCexternal
  mt0 = gwb.kernel["MacroTile0"]
  ws = gwb.kernel["WavefrontSize"]
  colsPerWave = gwb.kernel["MacroTile1"] // (gwb.kernel["NumThreads"] // ws)
  stride1 = "Stride%s%s" % (tc, kw.states.indexChars[gwb.kernel["PackedC1IndicesX"][0]])
  sizeI = kw.sizeRef(gwb.kernel["ProblemType"]["Index0"])
  sizeJ = kw.sizeRef(gwb.kernel["ProblemType"]["Index1"])
  col, dim, offLo, offHi = tmp.idx, tmp.idx + 1, tmp.idx + 2, tmp.idx + 3
  module.add(tdm.initOperands(g0, g1))
  module.add(tdm.setDataSize(bpe, g1))
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
  module.add(SLShiftLeftB64(dst=sgpr(offLo, 2), src=sgpr(offLo, 2), shiftHex=int(log2(bpe)), comment="64-bit byte offset from Srd%s" % tc))
  module.add(tdm.setGlobalAddr(g0, "Srd%s" % tc))
  module.add(SAddU32(dst=sgpr(g0 + 2), src0=sgpr(g0 + 2), src1=sgpr(offLo), comment="global address lo"))
  module.add(SAddCU32(dst=sgpr(g0 + 3), src0=sgpr(g0 + 3), src1=sgpr(offHi), comment="global address hi"))
  module.add(SSubU32(dst=sgpr(dim), src0=sizeI, src1=sgpr(dim), comment="tensor_dim0 = SizeI - wg0*MT0"))
  module.add(tdm.setTensorDim0(g1, dim, kw))
  module.add(SMulI32(dst=sgpr(dim), src0=sgpr("WorkGroup1"), src1=gwb.kernel["MacroTile1"], comment="wg1*MT1"))
  module.add(SAddU32(dst=sgpr(col), src0=sgpr(col), src1=sgpr(dim), comment="first %s column of this wave" % tc))
  module.add(SSubU32(dst=sgpr(dim), src0=sizeJ, src1=sgpr(col), comment="tensor_dim1 = SizeJ - first column"))
  module.add(tdm.setTensorDim1(g1, dim, kw))
  module.add(tdm.setTensorTile0(g1, mt0))
  module.add(tdm.setTensorTile1(g1, colsPerWave))
  module.add(tdm.setTensorStride0(g1, stride1))


def emitTDMStoreLoadC(gwb) -> Module:
  """Load the C tile into the staged tile with one tensor_load_to_lds per wave.

  Each wave loads the columns it stores. tensor_dim0/1 clip the load at the end of C, so no load
  reads past it; a wave whose columns all lie at or past SizeJ skips its load. Each thread then
  reads its C elements where it stages their D, so only the barrier after the loads is needed. The
  load carries C's cache hints, as the buffer loads of C would.
  """
  from .TensorDataMover import TensorDataMoverStore
  module = Module("TDMStoreLoadC")
  kw = gwb.parentWriter
  sizeJ = kw.sizeRef(gwb.kernel["ProblemType"]["Index1"])
  skipLoad = Label(kw.labels.getNameInc("TDMStoreLoadCSkip"), "no column of this wave is inside C")
  tdm = TensorDataMoverStore([kw.states.memTokenLdsBuffer0])
  with kw.allocTmpSgpr(12, alignment=8, tag="tdmLoadCDesc") as desc:
    with kw.allocTmpSgpr(4, alignment=2, tag="tdmLoadCTmp") as tmp:
      _tdmStoreDescriptor(gwb, module, tdm, desc.idx + 8, desc.idx, tmp, "C")
      module.add(SCmpLeU32(src0=sizeJ, src1=sgpr(tmp.idx), comment="all columns of this wave at or past SizeJ?"))
      module.add(SCBranchSCC1(labelName=skipLoad.getLabelName(), comment="skip the load"))
      module.add(tdm.issueLoad(desc.idx + 8, desc.idx, tensor=tdmStoreModifiers(kw.states.asmCaps, gwb.kernel, "C")))
      # StinkyRemoveWaitCntPass keeps a wait marked stinky-keep; nothing re-derives this one.
      module.add(SWaitTensorcnt(tensorcnt=0, comment="wait for the TDM load of C " + STINKY_KEEP))
      module.add(skipLoad)
  module.add(kw._syncThreads(gwb.kernel, "every wave has loaded its part of the C tile"))
  return module


def emitTDMStoreFlush(gwb) -> Module:
  """Store the staged D tile with one tensor_store_from_lds per wave.

  Each wave stores MacroTile1/numWaves consecutive columns of the tile. tensor_dim0/1 bound
  the store to the part of the tile inside D; a wave whose columns all lie at or past SizeJ
  skips its store instead of relying on a zero tensor_dim.
  """
  from .TensorDataMover import TensorDataMoverStore
  module = Module("TDMStoreFlush")
  kw = gwb.parentWriter
  sizeJ = kw.sizeRef(gwb.kernel["ProblemType"]["Index1"])
  skipStore = Label(kw.labels.getNameInc("TDMStoreSkip"), "no column of this wave is inside D")
  tdm = TensorDataMoverStore([kw.states.memTokenLdsBuffer0])
  # group1 (8 SGPRs) must be 8-aligned, so it goes first; group0 (4 SGPRs) follows.
  with kw.allocTmpSgpr(12, alignment=8, tag="tdmStoreDesc") as desc:
    with kw.allocTmpSgpr(4, alignment=2, tag="tdmStoreTmp") as tmp:
      _tdmStoreDescriptor(gwb, module, tdm, desc.idx + 8, desc.idx, tmp, "D")
      # StinkyRemoveWaitCntPass keeps a wait marked stinky-keep; nothing re-derives this one.
      module.add(SWaitCnt(dscnt=0, comment="staging stores done " + STINKY_KEEP))
      module.add(kw._syncThreads(gwb.kernel, "every wave has staged its part of the tile"))
      module.add(SCmpLeU32(src0=sizeJ, src1=sgpr(tmp.idx), comment="all columns of this wave at or past SizeJ?"))
      module.add(SCBranchSCC1(labelName=skipStore.getLabelName(), comment="skip the store"))
      module.add(tdm.issueStore(desc.idx + 8, desc.idx, tensor=tdmStoreModifiers(kw.states.asmCaps, gwb.kernel)))
      # Nor this one: after the OptNLL store, nothing re-derives it before s_endpgm.
      module.add(SWaitTensorcnt(tensorcnt=0, comment="wait for the TDM store " + STINKY_KEEP))
      module.add(skipStore)
  if isPersistent(gwb.kernel):
    module.add(kw._syncThreads(gwb.kernel, "every TDM store is done with the staged tile before the next tile writes LDS"))
  return module
