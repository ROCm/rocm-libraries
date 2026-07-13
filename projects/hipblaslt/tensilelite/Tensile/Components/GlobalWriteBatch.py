################################################################################
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
# ies of the Software, and to permit persons to whom the Software is furnished
# to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
# PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
# CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
################################################################################

from rocisa.code import Label, Module, RegSet, TextBlock
from rocisa.container import SMEMModifiers, VOP3PModifiers, MUBUFModifiers, GLOBALModifiers, \
  SDWAModifiers, replaceHolder, EXEC, VCC, vgpr, sgpr, ContinuousRegister, mgpr
from rocisa.enum import CvtType, HighBitSel, RoundType, SaturateCastType, SelectBit, CacheScope
from rocisa.instruction import BufferAtomicAddF32, BufferAtomicCmpswapB32, BufferWbl2, \
  GlobalAtomicAddU32, GlobalLoadB32, GlobalStoreB32, \
  BufferAtomicCmpswapB64, BufferStoreB16, BufferStoreB32, BufferStoreB64, BufferStoreB128, DSBPermuteB32, FlatAtomicCmpswapB32, \
  SAddCU32, SAddU32, SAndB32, \
  SAndB64, SAtomicDec, SBarrier, SBranch, SCBranchExecNZ, SCBranchExecZ, \
  SCBranchSCC0, SCBranchSCC1, SCmpGtU32, SCmpKGtU32, SCSelectB32, SCmpEQI32, SCmpEQU32, SCmpGtI32, SCmpLeI32, SMinU32, SEndpgm, \
  SLShiftLeftB32, SLShiftLeftB64, SLShiftRightB32, SLShiftRightB64, SMovB32, SMovB64, SMulI32, \
  SNop, SOrB32, SOrB64, SOrSaveExecB32, SOrSaveExecB64, SSleep, SSubI32, SSubU32, \
  SSwapPCB64, SWaitCnt, SWaitAlu, VAShiftRightI32, VAddCCOU32, VAddCOU32, VAddF32, VAddF64, \
  VAddI32, VAddPKF16, VAddPKF32, VAddU32, VBfeI32, VCmpEQU32, VCmpGEI32, VCmpGtU32, \
  VCmpNeU32, VCmpNeU64, VCndMaskB32, VCvtBF8toF32, VCvtF16toF32, VCvtF32toF16, VCvtF32toI32, \
  VCvtFP8toF32, VCvtI32toF32, VCvtPkBF8toF32, VCvtPkF32toBF16, VCvtPkF32toFP16, VCvtPkFP8toF32, \
  VFmaF32, VFmaF64, VFmaPKF32, VFmaMixF32, VAndB32, VLShiftLeftB32, VPermlane16SwapB32, VPermlane32SwapB32, \
  VLShiftRightB32, VMacF32, VMadMixF32, VMaxF32, VMovB32, VMovB64, VMulF32, VMulF64, \
  VMulLOU32, VMulPKF16, VMulPKF32, VPackF16toB32, VReadfirstlaneB32, VRndneF32, VCvtBF16toFP32, VSubU32
from rocisa.functions import vectorStaticMultiply

from ..Common import DataDirection, SemanticVersion, isSubtileMultiDU
from ..Common.DataType import DataType
from ..Component import GlobalWriteComponents
from ..Component import Component
from ..SolutionStructs import Solution
from ..Activation import ActivationModule
from ..AsmStoreState import StoreState
from ..AsmAddressCalculation import AddrCalculation
from ..Components.PackData import formatting, PackData_F16, PackData_BF16, PackData_FLOAT8, PackData_FLOAT8_fnuz
from rocisa.instruction import ECvtF16toF32, ECvtPkFP8toF32, ECvtPkBF8toF32
from ..KernelWriterModules import hasSequentialValuC
# NOTE: fusedA2AKernArgLayout / FUSED_A2A_MAX_RANKS are imported lazily inside the
# fused-A2A methods (from .Signature) to avoid a circular import at module load time
# (Component.py does `from .Components import *`, which re-enters this module).

from math import ceil, log2


def _scmpGtU32(writer, src, imm, comment=""):
    """ISA-aware scalar compare: s_cmpk_gt_u32 when available, else s_cmp_gt_u32 via temp SGPR."""
    if writer.states.asmCaps["HasSCMPK"]:
        return SCmpKGtU32(src=src, simm16=imm, comment=comment)
    else:
        module = Module("scmpGtU32")
        tmpSgpr = writer.sgprPool.checkOut(1, preventOverflow=False)
        module.add(SMovB32(dst=sgpr(tmpSgpr), src=imm))
        module.add(SCmpGtU32(src0=src, src1=sgpr(tmpSgpr), comment=comment))
        writer.sgprPool.checkIn(tmpSgpr)
        return module

class GlobalWriteBatchComponent(GlobalWriteComponents):
  kernel = {"ProblemType": {"OperationType": "GEMM" }}
  def __call__(self, kernel: Solution, tPA, tPB, activation: ActivationModule, ss: StoreState, \
    batchIdx, applyAlpha, beta, edge, atomic, gwvw, atomicW, \
    batchElements, addrE, addrD, addrC, addrBias, addrScaleAVec, addrScaleBVec, addrScaleAlphaVec, isLocalBarrierInit: bool, \
    tmpVgpr, tmpVgprDynamic, cvtVgprStruct, activationSetPCStruct, activationTypeStr, batchElementSgprs, tmpSgpr, codeAccVgprRead, \
    codeMulAlpha, packdata, parentWriter, factorDim, amdClangVersion: SemanticVersion, numBatches: int,
    inter_iter_rowInc: int = 0, direct_next_rowInc: int = 0) -> Module:
    return GlobalWriteBatchWriter(kernel, tPA, tPB, activation, ss, batchIdx, applyAlpha, \
      beta, edge, atomic, gwvw, atomicW, \
      batchElements, addrE, addrD, addrC, addrBias, addrScaleAVec, addrScaleBVec, addrScaleAlphaVec, isLocalBarrierInit, \
      tmpVgpr, tmpVgprDynamic, cvtVgprStruct, activationSetPCStruct, activationTypeStr, batchElementSgprs, tmpSgpr, \
      codeAccVgprRead, codeMulAlpha, packdata, parentWriter, factorDim, amdClangVersion, numBatches,
      inter_iter_rowInc, direct_next_rowInc).emit()

class GlobalWriteBatchWriter:
  def __init__(self, kernel: Solution, tPA, tPB, activation: ActivationModule, ss: StoreState, \
    batchIdx, applyAlpha, beta, edge, atomic, gwvw, atomicW, \
    batchElements, addrE, addrD, addrC, addrBias, addrScaleAVec, addrScaleBVec, addrScaleAlphaVec, isLocalBarrierInit: bool, \
    tmpVgpr, tmpVgprDynamic, cvtVgprStruct, activationSetPCStruct, activationTypeStr, batchElementSgprs, tmpSgpr, codeAccVgprRead, \
    codeMulAlpha, packdata, parentWriter, factorDim, amdClangVersion: SemanticVersion, numBatches: int,
    inter_iter_rowInc: int = 0, direct_next_rowInc: int = 0):
    self.kernel = kernel
    self.tPA    = tPA
    self.tPB    = tPB
    self.activation = activation
    self.ss = ss
    self.batchIdx = batchIdx
    self.applyAlpha = applyAlpha
    self.beta = beta
    self.edge = edge
    self.atomic = atomic
    self.gwvw = gwvw
    self.atomicW = atomicW
    self.batchElements = batchElements
    self.addrE    = addrE
    self.addrD    = addrD
    self.addrC    = addrC
    self.addrBias = addrBias
    self.addrScaleAVec = addrScaleAVec
    self.addrScaleBVec = addrScaleBVec
    self.addrScaleAlphaVec = addrScaleAlphaVec
    self.isLocalBarrierInit  = isLocalBarrierInit
    self.activationSetPCStruct = activationSetPCStruct
    self.activationTypeStr     = activationTypeStr
    self.tmpVgpr = tmpVgpr.idx
    self.tmpVgprSize = tmpVgpr.size
    self.tmpVgprDynamic = None
    if tmpVgprDynamic:
      self.tmpVgprDynamic = tmpVgprDynamic.idx
      self.tmpVgprDynamicSize = tmpVgprDynamic.size
    self.cvtVgprStruct = cvtVgprStruct
    self.batchElementSgprs = batchElementSgprs
    self.tmpSgpr = tmpSgpr
    self.codeAccVgprRead = codeAccVgprRead
    self.codeMulAlpha = codeMulAlpha
    self.packdata     = packdata
    self.parentWriter = parentWriter
    self.storesIssued = 0
    self.factorDim = factorDim
    self.amdClangVersion = amdClangVersion

    # Stateful tracking for N-group OOB guard deduplication (_emitSubtileOobGuard).
    # The outer loop iterates N-outer / M-inner, so all M elements within a fixed N
    # group share the same N guard result.  We emit the N s_cmp/s_cbranch only once
    # per N group and skip it for subsequent M elements in the same group.
    self._subtilePrevBlockIdxN = -1       # sentinel: no group seen yet
    self._subtileNGroupSkipLabel = None   # end-of-N-group label (M cbranch target)
    self._subtileAllStoresEndLabel = None # end-of-all-stores label (N cbranch target)
    self._subtileCloadPrevD1 = -1         # sentinel: last d1 group seen in C load guard
    self._subtilePendingSrdDInc = None    # deferred SrdD incToNextRow (emitted after N-group label)
    self._align8NMaskBlockIdxN = -1       # last blockIdxN for which N mask was computed
    self.numBatches = numBatches

    # CompactLoopStore: stash the "next batch's first elt rowInc" passed in via
    # `inter_iter_rowInc`. Used as the look-ahead fall-through value at the
    # LAST emitting elt of the batch (cross-batch transition), where the
    # intra-batch forward scan finds no further emitting elt.
    # 0 means no override (non-CLS / last batch / atomic).
    self.inter_iter_rowInc = inter_iter_rowInc
    self.direct_next_rowInc = direct_next_rowInc

    # Internal state for GlobalWriteBatch
    # 0 for None, 1 for WorkGroupReduction = False, 2 for WorkGroupReduction = True
    self.storeBiasD = 0
    if self.parentWriter.states.useBias == DataDirection.WRITE and \
      (not self.kernel["WorkGroupReduction"]) and \
      self.kernel["ProblemType"]["BiasSrc"] == "D":
      self.storeBiasD = 1

  @property
  def needsAccumToDestConversion(self) -> bool:
    """
    Check if accumulation values need to be converted to destination type:
    1. HighPrecisionAccumulate is enabled (accumulator precision > output precision)
       e.g., F32 accumulator -> FP16/BF16/FP8/BF8/I32/I8 output
    2. _GlobalAccumulation is not 'MultipleBuffer'

    When True, the pack/convert module will be generated to perform:
    - F32 -> FP16/BF16 packing
    - F32 -> FP8/BF8 conversion (with optional stochastic rounding)
    - F32 -> I32/I8 conversion and packing
    """
    return self.kernel["ProblemType"]["HighPrecisionAccumulate"] and \
           (self.kernel["_GlobalAccumulation"] != 'MultipleBuffer')

  @property
  def skipRearrangement(self) -> bool:
    """
    Check if we can skip v_mov_b32 rearrangement and use WMMA output registers directly.

    skipRearrangement changes store to read from (elementSumIdx - elementSumIdx[0]),
    but other operations (bias, activation, alpha) work on (elementSumIdx - startVgprValu).
    These positions are only the same when no operations modify the data.

    Safe to skip rearrangement only when:
    1. MIArchVgpr is True (MFMA/WMMA writes directly to arch VGPRs)
    2. hasSequentialValuC is True (WMMA output is already sequential)
    3. needsAccumToDestConversion is False (no pack module to do the rearrangement)
    4. Not a Beta path (Beta paths need separate alpha multiply + beta*C)
    5. Not complex (real/imag are interleaved in acc VGPRs, so the relative
       offset elementSumIdx[i]-elementSumIdx[0] does not locate the imag half)
    6. Single output tile (MIWaveTile == [1,1] and VectorWidth == 1). The
       i*gwvw offset only matches the arch-VGPR layout for one contiguous block;
       with multiple wave-tiles the acc registers are grouped per tile (e.g. TN +
       2x2 read the wrong registers).
    7. CompactLoopStore is off. CLS-loop store reads acc VGPRs via
       v_movrelsd_2_b32 indexing, which relies on the rearrangement layout.
    """
    if self.parentWriter.states.useBias == DataDirection.READ or \
       self.kernel.get("CompactLoopStore", False) or \
       self.kernel.get("ActivationFuncCall", False) or \
       self.applyAlpha or \
       self.kernel["ProblemType"].get("UseScaleAlphaVec", 0) or \
       self.kernel["ProblemType"].get("UseScaleAB", "") == "Vector" or \
       self.kernel["ProblemType"].get("UseScaleCD", False) or \
       self.kernel["ProblemType"]["DataType"].isComplex():
      return False

    miWaveTile = self.kernel.get("MIWaveTile", [1, 1])
    if miWaveTile[0] != 1 or miWaveTile[1] != 1 or \
       self.kernel.get("VectorWidthA", 1) != 1 or self.kernel.get("VectorWidthB", 1) != 1:
      return False

    return self.kernel["MIArchVgpr"] and \
           hasSequentialValuC(self.kernel) and \
           not self.needsAccumToDestConversion and \
           not self.beta

  @property
  def wavelen(self) -> int:
    return self.kernel["WavefrontSize"]

  @property
  def laneSGPRC(self) -> int:
    return self.parentWriter.states.laneSGPRCount

  @property
  def tmpS01(self):
    return self.tmpSgpr

  @property
  def tmpS23(self):
    return self.tmpS01 + self.laneSGPRC

  @property
  def debugConfig(self):
    return self.parentWriter.db

  @property
  def computeDataType(self) -> DataType:
    return self.kernel["ProblemType"]["ComputeDataType"]

  @property
  def destDataType(self) -> DataType:
    return self.kernel["ProblemType"]["DestDataType"]

  @property
  def moduleName(self):
    return "globalWriteBatch (Atomic)" if self.atomic else "globalWriteBatch (Non atomic)"

  def getEdgeMovInstType(self):
    return SMovB32 if self.wavelen == 32 else SMovB64

  def getEdgeOrInstType(self):
    return SOrB32 if self.wavelen == 32 else SOrB64

  def getEdgeAndInstType(self):
    return SAndB32 if self.wavelen == 32 else SAndB64

  def getSOrSaveExecType(self):
    return SOrSaveExecB32 if self.wavelen == 32 else SOrSaveExecB64

  @staticmethod
  def computeCLSLayout(kernel, numBatches: int):
    """Single source of truth for the CLS loop layout math.

    Returns (batchesPerCLSBody, iterCount, m0Step), all derived from ONE shared
    read of the MI output layout so the loop trip count and the M0 src stride
    can never drift apart:
      - batchesPerCLSBody: how many batches one CLS loop body covers. The body
        must align to the SrdD-advance period, else the loop would run s_add
        SrdD too many times.
      - iterCount: numBatches / batchesPerCLSBody (>= 1).
      - m0Step: stride of the CLS iter dim in the v_movrelsd_2_b32 src formula
        (added to M0 each iteration).
    Which dim is the CLS iter dim depends on the output layout:
      (a) outerTT1  > 1, VW1 == 1            : iter = wgIdx1, step = OPM*BM*BN*VW0*outerTT0*VW1
      (b) outerTT1 == 1, VW1 > 1, SS=False   : iter = vw1,    step = OPM*BM*BN*VW0*outerTT0
      (c) outerTT1 == 1, VW1 == 1, SS=True   : iter = tIdx,   step = NEPBS / inner_dims
    Store paths not covered by the CLS loop (non-MI, StoreRemap, StreamK) and
    non-divisible / non-regular layouts fall back to a single iteration
    (batchesPerCLSBody = numBatches), where m0Step is dead.
    """
    batchesPerBody = numBatches
    m0Step = 1
    if not kernel["EnableMatrixInstruction"]:
      return batchesPerBody, 1, m0Step

    VW0 = kernel["VectorWidthA"]
    VW1 = kernel["VectorWidthB"]
    outerTT0 = kernel["MIWaveTile"][0] // VW0
    outerTT1 = kernel["MIWaveTile"][1] // VW1
    miT  = min(kernel["MatrixInstM"], kernel["MatrixInstN"])
    miM_ = (kernel["MatrixInstM"] * kernel["MatrixInstBM"]) if (kernel["MatrixInstM"] == 4) else miT
    miN_ = (kernel["MatrixInstN"] * kernel["MatrixInstBN"]) if (kernel["MatrixInstN"] == 4) else miT
    matrixInstBM = 1 if (kernel["MatrixInstM"] == 4) else kernel["MatrixInstBM"]
    matrixInstBN = 1 if (kernel["MatrixInstN"] == 4) else kernel["MatrixInstBN"]
    OPM   = miM_ * miN_ // kernel["WavefrontSize"]
    NEPBS = kernel["NumElementsPerBatchStore"]

    if outerTT1 > 1 and VW1 == 1:
      m0Step = OPM * matrixInstBM * matrixInstBN * VW0 * outerTT0 * VW1
      if numBatches % outerTT1 == 0:
        batchesPerBody = max(1, numBatches // outerTT1)
    elif outerTT1 == 1 and VW1 > 1 and not kernel["SourceSwap"]:
      m0Step = OPM * matrixInstBM * matrixInstBN * VW0 * outerTT0
      if numBatches % VW1 == 0:
        batchesPerBody = max(1, numBatches // VW1)
    elif outerTT1 == 1 and VW1 == 1 and kernel["SourceSwap"]:
      inner_dims = VW0 * outerTT0 * matrixInstBM * matrixInstBN
      if NEPBS % inner_dims == 0:
        m0Step = max(1, NEPBS // inner_dims)
        batchesPerBody = max(1, ceil(numBatches / NEPBS))

    # StoreRemap / StreamK store paths are not covered by the CLS loop: force a
    # single iteration (body = all batches). m0Step is left as derived above to
    # preserve the original emitted SrdD step (it is dead when iterCount == 1).
    if kernel.get("StoreRemapVectorWidth", 0) != 0 or kernel.get("StreamK", 0) != 0:
      batchesPerBody = numBatches

    iterCount = max(1, numBatches // batchesPerBody)
    return batchesPerBody, iterCount, m0Step

  @staticmethod
  def computeBatchesPerCLSBody(kernel, numBatches: int) -> int:
    return GlobalWriteBatchWriter.computeCLSLayout(kernel, numBatches)[0]

  def _computeBatchesPerCLSBody(self) -> int:
    return GlobalWriteBatchWriter.computeCLSLayout(self.kernel, self.numBatches)[0]

  @staticmethod
  def computeCLSIterCount(kernel, numBatches: int) -> int:
    """CLS loop iter count = numBatches / batchesPerCLSBody. Minimum 1."""
    return GlobalWriteBatchWriter.computeCLSLayout(kernel, numBatches)[1]

  def _computeCLSIterCount(self) -> int:
    return GlobalWriteBatchWriter.computeCLSLayout(self.kernel, self.numBatches)[1]

  def _computeCLSLayout(self):
    return GlobalWriteBatchWriter.computeCLSLayout(self.kernel, self.numBatches)

  def emit(self) -> Module:
    assert self._checkAtomicPreconditions()
    module = Module(self.moduleName)
    self._prolog(module)
    # The bias/SAV drain barrier ordering is a multi-DU-only hardening that
    # prevents cross-wave LDS corruption from ds_bpermute. Multi-DU emits the
    # drain+barrier before the _emitAdd subtile stores; non-multi-DU emits
    # _emitAdd first.
    isMultiDU = isSubtileMultiDU(self.kernel)
    drainBiasSav = self.kernel.get("UseSubtileImpl") and \
       (self.parentWriter.states.useBias != DataDirection.NONE or \
        self.kernel["ProblemType"].get("UseScaleAlphaVec", 0))
    if not isMultiDU:
      self._emitAdd(module)
    if drainBiasSav:
      module.add(SWaitCnt(dscnt=0, comment="drain bias/SAV LDS reads"))
      module.add(SBarrier(comment="sync waves before subtile paired stores"))
    if isMultiDU:
      self._emitAdd(module)
    self._epilog(module)
    # fused-A2A cross-card handshake (design spec section 2.3): emitted ONCE per
    # store path, at the LAST batch, after all PUSH stores of this WG are issued.
    # A WG computes its whole tile across all batches, so "after the tile is done"
    # == after the last batch. The tail is runtime-gated to PUSH WGs and elects a
    # single last WG per dst_rank to release (wbl2 + system flag). See
    # _emitFusedA2AHandshake.
    if self.kernel["FusedGemmA2A"] and self.batchIdx == self.numBatches - 1:
      self._emitFusedA2AHandshake(module)
    # CompactLoopStore CLS countdown tail: emit countdown + branch + s_endpgm at
    # END of the CLS-loop body (= last batch of batchesPerCLSBody). Gated by
    # CompactLoopStore so non-CLS .s matches baseline (no CLS tail emit).
    if self.kernel["CompactLoopStore"]:
      #     updateCoord1 = (edge or multi-packed)
      if self.direct_next_rowInc != 0 and self.ss.elementAddr:
        kw = self.parentWriter
        rowInc = self.direct_next_rowInc
        bufferStore = self.kernel["BufferStore"]
        updateCoord1 = self.edge or len(self.kernel["PackedC1IndicesX"]) > 1
        emit_coord1 = (not bufferStore) or updateCoord1
        emit_rowptr = (not self.ss.optSrdIncForRow) and bufferStore
        if emit_coord1 or emit_rowptr:
          module.addComment0("CLS look-ahead: next batch's row advance at end of this batch")
          # Reuse the canonical per-element advance helpers (emitCoord1Advance +
          # emitRowPtrAdvance) on a representative AddrCalculation instead of
          # re-implementing the coord1 / rowPtr advance inline. The look-ahead is just
          # the next batch's emitAddressSetupCode row-advance, produced from the same
          # source. Same spirit as the delayed-primer reuse of incrementToNextRow.
          addrCalc = self.ss.elementAddr[0]
        if emit_coord1:
          module.add(addrCalc.emitCoord1Advance(rowInc, self.tmpS01,
                     comment="coord1.la: coord1Vgpr += rowInc (look-ahead)",
                     scomment="rowInc look-ahead"))
        if emit_rowptr:
          # Reuse the canonical row-pointer advance (the SAME helper the per-element
          # advance uses in emitAddressSetupCode) instead of re-implementing the
          # cinRowPtr/coutRowPtrD adds inline. This also makes the look-ahead cover
          # coutRowPtrE / coutRowPtrBias / packed-C1 for free (same conditions).
          module.add(addrCalc.emitRowPtrAdvance(self.kernel, self.ss, self.tmpS01, rowInc, lookahead=True))

      clsLabel = getattr(self.ss, "_clsLoopLabel", None)
      if clsLabel is not None and (self._computeBatchesPerCLSBody() - 1 == self.batchIdx) and self.ss.elementAddr:
        module.add(SSubI32(dst=sgpr("CLSLoopCounter"), src0=sgpr("CLSLoopCounter"), src1=1))
        module.add(SCmpEQU32(src0=sgpr("CLSLoopCounter"), src1=0))
        # 32-bit-offset backward branch: the replay hazard protection 7bfd03f
        # can overflow simm16 ("branch size exceeds simm16"). longBranchScc0 emits a
        # getpc/setpc sequence that handles the full 32-bit distance.
        module.add(self.parentWriter.longBranchScc0(clsLabel, posNeg=-1, comment="loop while counter != 0"))
        # if not self.kernel["StreamK"] == 3:
        #   module.add(SEndpgm(comment="stop here after CLS loop"))
        self.ss._clsLoopLabel = None
    return module

  def globalStoreWait(self, elementIdx, waitCnter, vlcntTotalIssued, dscntTotalIssued, interleaveStoreVmcnt: bool):
    vlcnt = -1
    dscnt = -1
    vscnt = -1
    isSingleKernel = ((self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1) or self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel") or self.kernel["StreamK"] > 0
    if interleaveStoreVmcnt:
      waitLocalLoadCnt = 0
      waitLocalLoadCntStrList = []
      waitLoadCnt = 0
      waitLoadCntStrList = []
      # Calculate global loads
      if self.beta:
        waitLoadCnt += self.betaLoadIssued[elementIdx]
        waitLoadCntStrList.append("%d (beta)"%self.betaLoadIssued[elementIdx])
      if self.loadE:
        waitLoadCnt += self.eLoadIssued[elementIdx]
        waitLoadCntStrList.append("%d (load E)"%self.eLoadIssued[elementIdx])
      if self.parentWriter.states.useGateResidual and elementIdx < len(self.gateLoadIssued):
        waitLoadCnt += self.gateLoadIssued[elementIdx]
        waitLoadCntStrList.append("%d (load Gate)"%self.gateLoadIssued[elementIdx])
      # Calculate local loads
      # UseSubtileImpl with bias/SAV: skip bias/SAV LDS loads from interleaved
      # waitcnt and rely on the batch-start barrier for LDS synchronization.
      subtileBarrierDrains = self.kernel.get("UseSubtileImpl") and \
        (self.parentWriter.states.useBias != DataDirection.NONE or \
         self.kernel["ProblemType"].get("UseScaleAlphaVec", 0))
      if self.parentWriter.states.useBias == DataDirection.READ and not subtileBarrierDrains:
        waitLocalLoadCnt += self.biasLoadIssued[elementIdx]
        waitLocalLoadCntStrList.append("%d (bias)"%self.biasLoadIssued[elementIdx])
      if (self.kernel["ProblemType"]["UseScaleAB"] == "Vector") and isSingleKernel:
        waitLocalLoadCnt += self.scaleAVecLoadIssued[elementIdx]
        waitLocalLoadCntStrList.append("%d (scaleAVec)"%self.scaleAVecLoadIssued[elementIdx])
        waitLocalLoadCnt += self.scaleBVecLoadIssued[elementIdx]
        waitLocalLoadCntStrList.append("%d (scaleBVec)"%self.scaleBVecLoadIssued[elementIdx])
      # Skip scaleAlphaVec when subtileBarrierDrains
      if self.kernel["ProblemType"]["UseScaleAlphaVec"] and isSingleKernel and not subtileBarrierDrains:
        waitLocalLoadCnt += self.scaleAlphaVecLoadIssued[elementIdx]
        waitLocalLoadCntStrList.append("%d (scaleAlphaVec)"%self.scaleAlphaVecLoadIssued[elementIdx])
      # Get vlcnt and dscnt
      vlcnt = vlcntTotalIssued - waitLoadCnt
      if waitCnter[0] > 0  or vlcnt != waitCnter[0] : # Check if global load issued > 0
        if waitCnter[0] == vlcnt: # No need to wait if the global load cnt doesn't change
          vlcnt = -1
        else:
          waitCnter[0] = vlcnt
      else:
        vlcnt = -1

      dscnt = dscntTotalIssued - waitLocalLoadCnt
      if waitCnter[1] > 0 or dscnt != waitCnter[1]: # Check if local load issued > 0
        if waitCnter[1] == dscnt: # No need to wait if the local load cnt doesn't change
          dscnt = -1
        else:
          waitCnter[1] = dscnt
      else:
        dscnt = -1
      # Get vscnt
      if vlcnt != -1 and not (self.parentWriter.states.asmCaps["SeparateVscnt"] or self.parentWriter.states.asmCaps["SeparateVMcnt"]):
          if self.kernel.get("UseSubtileImpl") and not self.kernel["GroupLoadStore"]:
            vscnt = 0
          else:
            vscnt = self.storesIssued if not self.kernel["GroupLoadStore"] else 0
      if (vlcnt != -1) or (vscnt != -1) or (dscnt != -1):
        # Get comment
        comment = ""
        if vlcnt != -1:
          tmp = ""
          for cntStr in waitLoadCntStrList:
            tmp += " - %s"%cntStr
          comment = "vlcnt(%s) = %d%s"%(vlcnt, vlcntTotalIssued, tmp)
        if vscnt != -1:
          comment = comment + (" " if comment else "") + "vscnt(%s)"%(vscnt)
        if dscnt != -1:
          tmp = ""
          for cntStr in waitLocalLoadCntStrList:
            tmp += " - %s"%cntStr
          comment = comment + (" " if comment else "") + "dscnt(%d) = %d%s"%(dscnt, dscntTotalIssued, tmp)
        # if not self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":
        return SWaitCnt(dscnt=dscnt, vlcnt=vlcnt, vscnt=vscnt, comment="%s (interleaved)"%comment)
    else:
      commentList = []
      # Global read wait
      if self.beta:
        vlcnt = 0
        commentList.append("Beta")
      if self.loadE:
        vlcnt = 0
        commentList.append("E")
      if self.parentWriter.states.useGateResidual:
        vlcnt = 0
        commentList.append("Gate")
      # Local read wait
      if self.parentWriter.states.useBias == DataDirection.READ:
        dscnt = 0
        commentList.append("Bias LDS")
      if (self.kernel["ProblemType"]["UseScaleAB"] == "Vector") and isSingleKernel:
        dscnt = 0
        commentList.append("ScaleABVec")
      if self.kernel["ProblemType"]["UseScaleAlphaVec"] and isSingleKernel:
        dscnt = 0
        commentList.append("ScaleAlphaVec")
      if (vlcnt != -1) or (dscnt != -1):
        # Get comment
        comment = "wait for " + commentList[0]
        for c in commentList[1:]:
          comment += ", %s"%c
        return SWaitCnt(dscnt=dscnt, vlcnt=vlcnt, vscnt=vscnt, comment=comment)
    return None

  ##############################################################################
  # choose the ADD instruction for combining external C with internal C
  # used in atomic=1 case to compute expected external data
  ##############################################################################
  def _chooseAddForAtomic(self, kernel, dst, src0, src1, comment):
    module = Module("chooseAddForAtomic")
    if kernel["ProblemType"]["DataType"].isBFloat16():
      if kernel["_GlobalAccumulation"]:
        module.add(VAddF32(dst, src0, src1, comment=comment))
    elif kernel["ProblemType"]["DataType"].isHalf():
      if kernel["_GlobalAccumulation"]:
        module.add(VAddF32(dst, src0, src1, comment=comment))
      elif kernel["ProblemType"]["HighPrecisionAccumulate"]:
        if self.parentWriter.states.asmCaps["v_fma_mix_f32"]:
          module.add(VFmaMixF32(dst, src0, 1, src1, comment=comment))
        elif self.parentWriter.states.asmCaps["v_mad_mix_f32"]:
          module.add(VMadMixF32(dst, src0, 1, src1, comment=comment))
        else:
          assert False, "No valid v_mad_mix_f32 equivalent"
      else:
        module.add(VAddPKF16(dst, src0, src1, comment))
    elif kernel["ProblemType"]["DataType"].isInt8x4() or kernel["ProblemType"]["DataType"].isInt8():
      # assume v_add_i32 can be used in place of v_add_f32
      # need to add saturation directive to v_add_i32 instruction to clamp integer arithmetic
      module.add(VAddI32(dst, src0, src1, comment=comment))
    elif kernel["ProblemType"]["DataType"].isSingle():
      module.add(VAddF32(dst, src0, src1, comment=comment))
    else:
       #support for double
      module.add(VAddF64(dst, src0, src1, comment=comment))

    return module

  def _emitLdsBarrierIfNeeded(self, targetModule: Module, isSingleKernel: bool):
    """Emit the LDS write barrier once per batch; idempotent via
    self.isLocalBarrierInit so repeat callers (preamble/body) are no-ops.
    """
    if isSingleKernel and (not self.isLocalBarrierInit):
      targetModule.add(SWaitCnt(dscnt=0, comment="Wait for LDS write"))
      targetModule.add(SBarrier(comment="LDS write barrier"))
      self.isLocalBarrierInit = True

  def _emitElt0EpilogueLoads(self, module: Module, addrCalc: 'AddrCalculation',
                             mask, elementIdx: int, bufferOOB,
                             loadInputCode: Module, factor_gwvw: int,
                             preamble: bool = False):
    """Emit one element's epilogue LDS loads (Bias -> ScaleAlphaVec ->
    ScaleAVec/ScaleBVec, then reorder). Shared by the CLS preamble
    (preamble=True: address compute only) and the per-element loop
    (preamble=False: also ds_load). xxxLoadIssued lists are appended in the
    body only, so the preamble adds no extra entry.
    """
    addrBiasVgpr          = addrCalc.addrBiasVgpr
    addrScaleAVecVgpr     = addrCalc.addrScaleAVecVgpr
    addrScaleBVecVgpr     = addrCalc.addrScaleBVecVgpr
    addrScaleAlphaVecVgpr = addrCalc.addrScaleAlphaVecVgpr
    dataBias              = self.ss.elementDataBias[elementIdx]
    dataScaleAVec         = self.ss.elementDataScaleAVec[elementIdx]
    dataScaleBVec         = self.ss.elementDataScaleBVec[elementIdx]
    dataScaleAlphaVec     = self.ss.elementDataScaleAlphaVec[elementIdx]
    skipLoad = True if self.factorDim else False
    isSingleKernel = ((self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1) or self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel") or self.kernel["StreamK"] > 0

    def addEpilogueLoad(modGwvw, ldName: str, addrVecVgpr, addrVec, dataVec, loadedDataVec,
                        vecOffset, gwvw, referenceVgpr, dim, referenceDim,
                        skipLoad: bool = False, comment: str = "") -> int:
      """One vector's epilogue load: emitLdChange (address compute) always runs;
      captured `preamble` gates ONLY the ds_load (preamble=True = address only).
      """
      loadsIssued = 0
      tmpInrSgpr = self._epilogScratchSgpr(1)
      module.add(addrCalc.emitLdChange(self.kernel, self.ss, ldName, self.edge, self.beta, mask, bufferOOB, (elementIdx == 0), self.tmpVgpr, tmpInrSgpr, addrVecVgpr, addrVec, dim))
      self._epilogScratchFree(tmpInrSgpr)
      ldsAddrVgpr = referenceVgpr if (referenceVgpr and (dim == referenceDim)) else addrVecVgpr
      if dataVec not in loadedDataVec:
        # GroupLoadStore routes barrier+ds_load into `loadInputCode` so they
        # are grouped with the C input; otherwise both go into `module`.
        targetModule = loadInputCode if self.kernel["GroupLoadStore"] else module
        self._emitLdsBarrierIfNeeded(targetModule, isSingleKernel)
        if not preamble:
          targetModule.add(self.parentWriter.addLdsLoad(self.kernel["ProblemType"]["ComputeDataType"], dataVec, ldsAddrVgpr, vecOffset, gwvw, comment=comment))
          loadedDataVec[dataVec] = ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * gwvw / 16)
          loadsIssued = ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * gwvw / 16)
          if (self.ss.cfg.gwvw != gwvw) and (not skipLoad):
            remain_load = self.ss.cfg.gwvw - 1
            #For below ds_read instruction do not add bias issued , because of all ds_load instructions need to be completed at the same time in this batch.
            for r in range(remain_load):
              modGwvw.add(self.parentWriter.addLdsLoad(self.kernel["ProblemType"]["ComputeDataType"], dataVec, ldsAddrVgpr, vecOffset, factor_gwvw, comment=comment))
      return loadsIssued

    modGwvwScale = []
    localReferenceVgpr = None
    if self.parentWriter.states.useBias == DataDirection.READ:
      modGwvwBias = Module("GwvwBias")
      self.localLoadsBiasIssued += addEpilogueLoad(modGwvwBias, 'Bias', addrBiasVgpr, self.addrBias, dataBias, self.loadedDataBias, addrCalc.biasOffset[self.factorDim], factor_gwvw, localReferenceVgpr, self.factorDim, self.factorDim, skipLoad=skipLoad, comment="load Bias")
      localReferenceVgpr = addrBiasVgpr
      modGwvwScale.append(modGwvwBias)
    if not preamble:
      self.biasLoadIssued.append(len(self.loadedDataBias) * ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * factor_gwvw / 16))

    if self.kernel["ProblemType"]["UseScaleAlphaVec"] and isSingleKernel:
      modGwvwScaleAlpha = Module("GwvwScaleAlpha")
      # For multi-DU, the subtile ScaleAlphaVec epilogue load passes None as the
      # LDS reference vgpr; non-multi-DU uses localReferenceVgpr.
      savIsMultiDU = isSubtileMultiDU(self.kernel)
      if savIsMultiDU:
        savLdsRefVgpr = None if (self.kernel.get("UseSubtileImpl") and addrScaleAlphaVecVgpr is not None) else localReferenceVgpr
      else:
        savLdsRefVgpr = localReferenceVgpr
      self.loadsScaleAlphaVecIssued += addEpilogueLoad(modGwvwScaleAlpha, "ScaleAlphaVec", addrScaleAlphaVecVgpr, self.addrScaleAlphaVec, dataScaleAlphaVec, self.loadedDataScaleAlphaVec, addrCalc.scaleAlphaVecOffset[self.factorDim], factor_gwvw, savLdsRefVgpr, self.factorDim, self.factorDim, skipLoad=skipLoad, comment="load scaleAlpha")
      if localReferenceVgpr == None:
        localReferenceVgpr = addrScaleAlphaVecVgpr
      modGwvwScale.append(modGwvwScaleAlpha)
    if not preamble:
      self.scaleAlphaVecLoadIssued.append(len(self.loadedDataScaleAlphaVec) if self.factorDim else len(self.loadedDataScaleAlphaVec) * ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * factor_gwvw / 16))

    if (self.kernel["ProblemType"]["UseScaleAB"] == "Vector") and isSingleKernel:
      modGwvwScaleA = Module("GwvwScaleA")
      modGwvwScaleB = Module("GwvwScaleB")
      self.loadsScaleAVecIssued += addEpilogueLoad(modGwvwScaleA, "ScaleAVec", addrScaleAVecVgpr, self.addrScaleAVec, dataScaleAVec, self.loadedDataScaleAVec, addrCalc.scaleAVecOffset, self.ss.cfg.gwvw, localReferenceVgpr, 0, self.factorDim, comment="load scaleA")
      self.loadsScaleBVecIssued += addEpilogueLoad(modGwvwScaleB, "ScaleBVec", addrScaleBVecVgpr, self.addrScaleBVec, dataScaleBVec, self.loadedDataScaleBVec, addrCalc.scaleBVecOffset, 1, localReferenceVgpr, 1, self.factorDim, skipLoad=True, comment="load scaleB")
      if localReferenceVgpr == None:
        localReferenceVgpr = addrScaleAVecVgpr if self.factorDim == 0 else addrScaleBVecVgpr
      modGwvwScale.append(modGwvwScaleA)
      modGwvwScale.append(modGwvwScaleB)
    if not preamble:
      self.scaleAVecLoadIssued.append(len(self.loadedDataScaleAVec) * ceil(self.kernel["ProblemType"]["ComputeDataType"].numBytes() * self.ss.cfg.gwvw / 16))
      self.scaleBVecLoadIssued.append(len(self.loadedDataScaleBVec))

    # Reorder scale
    length = 0
    for mod in modGwvwScale:
      length = max(length, len(mod.items()))

    for index in range(0, length):
      for mod in modGwvwScale:
        if len(mod.items()) > index:
          module.add(mod.items()[index])

  def _emitElt0LdsPreambleBeforeBanner(self, module: Module, bufferOOB,
                                       loadInputCode: Module, factor_gwvw: int):
    """CompactLoopStore preamble (batch 0): hoist elt-0 LDS setup out of the
    per-element loop so the CLS countdown loop need not re-emit it each iter. The
    body dedups via state flags (ss.singleCol*AddrUpdated, isLocalBarrierInit).
    Per-section gating (addr compute only if optSingleColVgpr; D scaleToBpe /
    sgpr-offset init / MSB prewarm NonEdge-only) is documented inline below.
    """
    elementIdx = 0
    addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
    addrBiasVgpr          = addrCalc.addrBiasVgpr
    addrCVgpr             = addrCalc.addrCVgpr
    addrDVgpr             = addrCalc.addrDVgpr
    addrScaleAlphaVecVgpr = addrCalc.addrScaleAlphaVecVgpr
    addrScaleAVecVgpr     = addrCalc.addrScaleAVecVgpr
    addrScaleBVecVgpr     = addrCalc.addrScaleBVecVgpr
    mask                  = self.ss.elementMask[elementIdx]
    bufferOOB             = None

    isSingleKernel = ((self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1) or \
                      self.kernel["GlobalSplitUAlgorithm"] == "MultipleBufferSingleKernel") or \
                     self.kernel["StreamK"] > 0

    # emitAddressSetupCode for elt 0 is safe: coordOffset0=0 and rowInc=0, so it
    # only emits the (d1,vc1,d0,vc0) comment + sets coord0Vgpr state -- no real
    # instructions. The body re-runs it for elt 0 (just re-emits the comment).
    tmpInrSgpr = self._epilogScratchSgpr(1)
    module.add(addrCalc.emitAddressSetupCode(self.kernel, self.tPB, self.ss, self.tmpVgpr, \
        tmpInrSgpr, self.edge, self.beta, self.atomic, elementIdx, addrDVgpr))
    self._epilogScratchFree(tmpInrSgpr)

    # elt-0 epilogue hoist: optSingleColVgpr -> hoist the address compute
    # (preamble=True; body folds it via ss.singleCol*AddrUpdated). Else (Edge /
    # non-optSingleCol) -> hoist only the LDS barrier.
    if self.ss.optSingleColVgpr:
      self._emitElt0EpilogueLoads(module, addrCalc, mask, elementIdx, bufferOOB,
                                   loadInputCode, factor_gwvw,
                                   True)
    else:
      targetModule = loadInputCode if self.kernel["GroupLoadStore"] else module
      self._emitLdsBarrierIfNeeded(targetModule, isSingleKernel)

    # D scaleToBpe -- NonEdge only (Edge has its own per-elt path that is not
    # safe to hoist).
    if not self.edge:
      tmpInrSgpr = self._epilogScratchSgpr(1)
      if self.kernel["GlobalSplitU"] == 1 or (self.kernel["GlobalSplitUAlgorithm"] != "MultipleBufferSingleKernel"):
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'D', self.edge, self.beta, \
            mask, bufferOOB, True, self.tmpVgpr, tmpInrSgpr, addrDVgpr, self.addrD, 0))
      if self.beta:
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'C', self.edge, self.beta, \
            mask, bufferOOB, True, self.tmpVgpr, tmpInrSgpr, addrCVgpr, self.addrC, 0))
      self._epilogScratchFree(tmpInrSgpr)

    # Init sgpr offset -- NonEdge only. Needed for the delayed-pattern
    # incrementToNextRow when optSrdIncForRow=1 (NonEdge). Edge path
    # (optSrdIncForRow=0) can skips the delayed pattern entirely so these inits
    # might not needed.

    module.add(SMovB32(dst=sgpr(self.tmpS01),   src=0, comment="Init sgpr offset"))
    module.add(SMovB32(dst=sgpr(self.tmpS01+1), src=0, comment="Init sgpr offset"))

    # Prewarm VGPR MSB bank -- NonEdge only. Forces the upper-bank toggle
    # for the upcoming ds_load to settle ahead of time so the loop body
    # avoids a stall on first use.
    prewarmVgpr = None
    if self.parentWriter.states.useBias == DataDirection.READ and addrBiasVgpr is not None:
      prewarmVgpr = addrBiasVgpr
    elif self.kernel["ProblemType"]["UseScaleAlphaVec"] and addrScaleAlphaVecVgpr is not None:
      prewarmVgpr = addrScaleAlphaVecVgpr
    elif (self.kernel["ProblemType"]["UseScaleAB"] == "Vector") and addrScaleAVecVgpr is not None:
      prewarmVgpr = addrScaleAVecVgpr
    if prewarmVgpr is not None:
      module.add(VMovB32(dst=vgpr(prewarmVgpr), src=vgpr(prewarmVgpr),
                          comment="prewarm VGPR MSB bank for upcoming ds_load"))

  def _lookaheadRowInc(self, elementIdx: int) -> int:
    """CompactLoopStore look-ahead: the rowInc that the NEXT EMITTING elt's
    s_add will consume, so this elt's delayed AFTER-primer can encode it
    (removes the off-by-one in the CLS primer chain). Forward-scan from
    elementIdx+1 for the first elt with rowInc != 0; if none remain in this
    batch, fall through to self.inter_iter_rowInc (the cross-batch advance,
    precomputed by KernelWriterAssembly; 0 on the last batch). Returns 0 when
    there is no override (non-CLS, no elementAddr, or no further advance).
    """
    if not (self.kernel["CompactLoopStore"] and self.ss.elementAddr):
      return 0
    for _j in range(elementIdx + 1, len(self.batchElements)):
      _ri = self.ss.elementAddr[_j].rowInc
      if _ri != 0:
        return _ri
    return self.inter_iter_rowInc

  def _epilogScratchSgpr(self, n: int = 1):
    """Scratch sgpr for epilogue address math.
    CompactLoopStore: borrow `n` aligned sgpr(s) from the pool so the address
    math does NOT clobber the row-increment primer chain (which lives in
    tmpS01 / tmpS01+1). Non-CLS: reuse self.tmpSgpr -- this is the baseline, so
    CLS=0 codegen is bit-for-bit unchanged. Pair with _epilogScratchFree.
    """
    if self.kernel["CompactLoopStore"]:
      return self.parentWriter.sgprPool.checkOutAligned(n, 1)
    return self.tmpSgpr

  def _epilogScratchFree(self, sgprIdx):
    """Release a scratch sgpr from _epilogScratchSgpr (no-op for non-CLS, which
    reused self.tmpSgpr and owns nothing).
    """
    if self.kernel["CompactLoopStore"]:
      self.parentWriter.sgprPool.checkIn(sgprIdx)

  def _prolog(self, module: Module):
    module.addComment0("optSingleColVgpr=%u optSharedColVgpr=%u optSGPRUsage=%s optSrdIncForRow=%u factorDim=%u" % \
              (self.ss.optSingleColVgpr, self.ss.optSharedColVgpr, self.ss.optSGPRUsage, self.ss.optSrdIncForRow, self.factorDim))

    if self.kernel["StoreSyncOpt"]:
      self._storeSyncOpt(module)

    # comment tt1, tt0, vc1, vc0
    # tt = thread tile, vc=vector component
    commentStr = "Global Write%s%s Batch #%u (d1,d0,vc1,vc0) =\n   " \
        % (" Beta" if self.beta else "", " Edge" if self.edge else "", self.batchIdx)

    commentStr = ''.join([commentStr] \
                            + ["(%u,%u,%u,%u:vw%u%s)%s" % \
                               (element[0], element[1], element[2], element[3], self.gwvw,
                               ":vaw:%u"%self.atomicW if self.atomic else "",
                               "" if idx == len(self.batchElements) -1 else "; ")
                               for idx, element in enumerate(self.batchElements)])
    # setupStoreElementsForBatch must run before the CLS preamble call so
    # ss.elementAddr[0] is populated. For non-CLS the batch banner is emitted
    # right after in the else branch (matches baseline ordering); for CLS the
    # banner is moved into the CLS header block below (after sgpr setup, before
    # the CLS label) so the per-iter store body is visually a single labelled
    # unit.
    if self.kernel["_GlobalAccumulation"] != "MultipleBufferSingleKernel":
      self.ss.setupStoreElementsForBatch(self.kernel, self.gwvw, self.batchElements, self.batchElementSgprs, isOptNLL=False, factorDim=self.factorDim)
    else:
      self.ss.setupStoreElementsForBatch(self.kernel, self.gwvw, self.batchElements, self.batchElementSgprs, isOptNLL=True, factorDim=self.factorDim)

    self.localLoadsBiasIssued = 0
    self.storesIssued    = 0
    self.loadsBetaIssued   = 0
    self.loadsEIssued      = 0
    self.loadsGateIssued   = 0
    self.loadsScaleAVecIssued = 0
    self.loadsScaleBVecIssued = 0
    self.loadsScaleAlphaVecIssued     = 0

    ########################################
    # calculate addr and masks
    # On input, coord0 and coord1 are VGPRs computed in the pre-batch code, based
    # on the thread and tid number.  These are ELEMENT offsets from start of tensor C
    # for the top-left corner this thread will write.  These are not changed
    # across all the store loop iters.
    if self.debugConfig["ConservativeWaitCnt"] & 0x10:
      module.add(SBarrier(comment="debug"))
      module.add(SWaitCnt(vlcnt=0, vscnt=0, comment="ConservativeWaitCnt"))
      module.add(SBarrier(comment="debug"))

    if not self.edge and self.debugConfig["ForceEdgeStores"] >= 2:
      module.add(self.parentWriter.getBomb()) # should not get here
    if self.edge and self.debugConfig["AssertNoEdge"]:
      module.add(self.parentWriter.getBomb()) # should not get here

    ########################################
    # rC *= alpha
    if not self.kernel["InterleaveAlpha"] and self.applyAlpha and self.parentWriter.alphaBeforeLoadC:
      module.addComment1("rC *= alpha batchElements=%s"%self.batchElements)
      if self.codeMulAlpha is None:
        elementIdx = 0
        while elementIdx < len(self.batchElements):
          isEnd = (elementIdx == len(self.batchElements) - 1)
          if not isEnd and (self.ss.elementSumIdx[elementIdx] + 1 == self.ss.elementSumIdx[elementIdx + 1]) and (self.ss.elementSumIdx[elementIdx] % 2 == 0):
            module.add(self._applyAlpha(self.kernel, self.gwvw, self.ss.elementSumIdx, elementIdx, self.tmpS01, usePK=True))
            elementIdx += 2
          else:
            module.add(self._applyAlpha(self.kernel, self.gwvw, self.ss.elementSumIdx, elementIdx, self.tmpS01))
            elementIdx += 1
      else:
          regsPerScalar = self.parentWriter.states.bpeCinternal // self.parentWriter.states.bpr # register per scalar
          for elementIdx in range(len(self.batchElements)):
            for vi in range(self.gwvw):
              rh = replaceHolder(self.codeMulAlpha.popFirstItem(), self.ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi)
              if (self.kernel["GlobalSplitU"] == 1) and (self.kernel["ProblemType"]["ComputeDataType"].isSingle() and self.kernel["ProblemType"]["DataType"].isInt8()):
                srcRegName = rh.getParams()[2].getCompleteRegName()
                module.add(VCvtI32toF32(dst=vgpr(srcRegName), src=vgpr(srcRegName), comment="Convert MI out reg to fp32"))
              module.add(rh)


    loadInputCode    = Module("loadInputCode")

    self.betaLoadIssued = []
    self.eLoadIssued = []
    self.gateLoadIssued = []
    self.biasLoadIssued = []
    self.scaleAVecLoadIssued = []
    self.scaleBVecLoadIssued = []
    self.scaleAlphaVecLoadIssued = []

    self.loadedDataBeta = {}
    self.loadedDataE = {}
    self.loadedDataGate = {}
    self.loadedDataBias = {}
    self.loadedDataScaleAVec = {}
    self.loadedDataScaleBVec = {}
    self.loadedDataScaleAlphaVec = {}

    #when factorDim = 1 the bias's gwvw is alwasy be 1.
    factor_gwvw = 1 if self.factorDim else self.ss.cfg.gwvw

    # CompactLoopStore CLS header (batch 0 only): preamble + sgpr setup + CLS
    # label + M0 assignment + M0 step. Wrapped in `if CompactLoopStore` as a
    # unit so non-CLS .s emits ONLY the original module.addComment2(commentStr)
    # in the else branch (matches baseline)
    if self.kernel["CompactLoopStore"] and self.batchIdx == 0:
      self._emitElt0LdsPreambleBeforeBanner(module, None, #bufferOOB,
                                             loadInputCode,
                                             factor_gwvw)

      module.add(SMovB32(dst=sgpr("CLSm0Base"), src=hex(0x0), comment="CLS M0 base = 0"))
      module.add(SMovB32(dst=sgpr("CLSLoopCounter"), src=hex(self._computeCLSIterCount()), comment="CLS loop iter count"))
      module.addComment2(commentStr)
      self.ss._clsLoopLabel = Label(self.parentWriter.labels.getNameInc("CLS"), "")
      module.add(self.ss._clsLoopLabel)
      module.add(SMovB32(dst=mgpr(0), src=sgpr("CLSm0Base"),
          comment="LDS clamp at sgpr(CLSm0Base)"))
      # CLS M0 step: M0 indirectly offsets the src VGPR index of v_movrelsd_2_b32.
      # Step = the stride of the CLS iter dim in the SRC formula; taken from the
      # same layout derivation as the loop iter count (computeCLSLayout) so the
      # two can never drift apart.
      _, _, cls_m0_step = self._computeCLSLayout()
      module.add(SAddU32(dst=sgpr("CLSm0Base"), src0=sgpr("CLSm0Base"), src1=cls_m0_step,
                         comment="CLS M0 step (src coef of CLS iter dim)"))
    else:
      module.addComment2(commentStr)

    module.addComment1("calc coords, apply mask, and issue loads (if necessary)")

    if self.kernel["BufferStore"] and (self.edge or (self.kernel["NumWaveSplitK"] > 1)):
      bufferOOB = self.tmpVgpr + self.tmpVgprSize - 1
      module.add(VMovB32(dst=vgpr(bufferOOB), src="BufferOOB"))
    else:
      bufferOOB = None

    for elementIdx, element in enumerate(self.batchElements):
      addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
      addrCVgpr    = addrCalc.addrCVgpr
      addrDVgpr    = addrCalc.addrDVgpr
      addrEVgpr    = addrCalc.addrEVgpr
      addrBiasVgpr = addrCalc.addrBiasVgpr
      addrScaleAVecVgpr = addrCalc.addrScaleAVecVgpr
      addrScaleBVecVgpr = addrCalc.addrScaleBVecVgpr
      addrScaleAlphaVecVgpr = addrCalc.addrScaleAlphaVecVgpr
      data     = self.ss.elementData[elementIdx]
      dataBeta = self.ss.elementData[elementIdx]
      dataE    = self.ss.elementDataE[elementIdx]
      dataGate = self.ss.elementDataGate[elementIdx] if self.parentWriter.states.useGateResidual else 0
      dataBias = self.ss.elementDataBias[elementIdx]
      dataScaleAVec = self.ss.elementDataScaleAVec[elementIdx]
      dataScaleBVec = self.ss.elementDataScaleBVec[elementIdx]
      dataScaleAlphaVec = self.ss.elementDataScaleAlphaVec[elementIdx]
      mask     = self.ss.elementMask[elementIdx]
      vc0 = element[3]
      sumIdxGSUSYNC = self.ss.elementSumIdx[elementIdx]

      # CompactLoopStore look-ahead override: this elt's delayed AFTER-primer
      # encodes the NEXT EMITTING elt's rowInc (see _lookaheadRowInc).
      _emitOverrideRows = self._lookaheadRowInc(elementIdx)

      tmpInrSgpr = self._epilogScratchSgpr(1)
      _skipCrossBatchAdv = (self.kernel["CompactLoopStore"] and elementIdx == 0 and self.batchIdx > 0)
      module.add(addrCalc.emitAddressSetupCode(self.kernel, self.tPB, self.ss, self.tmpVgpr, tmpInrSgpr, self.edge, self.beta, self.atomic, elementIdx, addrDVgpr,
                                               skipCrossBatchAdvance=_skipCrossBatchAdv))
      self._epilogScratchFree(tmpInrSgpr)

      if self.edge:
        tmpInrSgpr = self._epilogScratchSgpr(2)
        module.add(addrCalc.edgeProtectCode(self.kernel, self.edge, self.beta, self.atomic, mask, tmpInrSgpr))
        self._epilogScratchFree(tmpInrSgpr)
        if self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":
          module.addComment1("edge Protect")
      # create code Module to push mov vgpr,acc instructions
      if self.beta:
        tmpInrSgpr = self._epilogScratchSgpr(1)
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'C', self.edge, self.beta, mask, bufferOOB, (elementIdx == 0), self.tmpVgpr, tmpInrSgpr, addrCVgpr, self.addrC, 0))
        self._epilogScratchFree(tmpInrSgpr)
        if dataBeta not in self.loadedDataBeta:
          # In the UseSubtileImpl NonEdge path the workgroup-level edge check is relaxed
          # (subtile-aligned remainder is allowed into NonEdge), so individual waves may
          # own rows/columns beyond the valid output region.  Gate each C load by writing
          # SrdC+2 (num_records): BufferOOB → normal load, 0 → hardware returns zero.
          #
          # element loop is N(d1)-outer / M(d0)-inner.
          # d1 (N) check: emitted once per d1 group — sets SrdC+2 = BufferOOB if N valid, else 0.
          # d0 (M) check: emitted per element — overwrites SrdC+2 = SrdC+2 if M valid, else 0.
          #   (AND semantics: SrdC+2 = BufferOOB only when both M and N are valid.)
          # d0 is monotone within each d1 group: once OOB, remaining d0s are also OOB.
          mGuardSgpr = self.parentWriter.states.subtileM32ValidBlocksSgpr
          nGuardSgpr = self.parentWriter.states.subtileN16ValidBlocksSgpr
          if not self.edge and (mGuardSgpr is not None or nGuardSgpr is not None):
            d1, d0 = element[0], element[1]
            # N guard: emit once per d1 group.
            if nGuardSgpr is not None and d1 != self._subtileCloadPrevD1:
              d1Cmp = d1 * 16 if self.parentWriter.states.storeAlign8 else d1
              module.add(_scmpGtU32(self.parentWriter, sgpr("SubtileNGuard"), d1Cmp,
                                    comment="subtile C load: clamped > %d?" % d1Cmp))
              module.add(SCSelectB32(dst=sgpr("SrdC+2"), src0="BufferOOB", src1=0,
                                     comment="SrdC+2 = BufferOOB if N valid, else 0"))
              self._subtileCloadPrevD1 = d1
            # M guard: emit per element, AND into SrdC+2.
            if mGuardSgpr is not None:
              module.add(_scmpGtU32(self.parentWriter, sgpr("SubtileMGuard"), d0,
                                    comment="subtile C load: numMBlocks > d0=%d?" % d0))
              if nGuardSgpr is not None:
                module.add(SCSelectB32(dst=sgpr("SrdC+2"), src0=sgpr("SrdC+2"), src1=0,
                                       comment="SrdC+2 = SrdC+2 if M valid, else 0 (AND with N result)"))
              else:
                module.add(SCSelectB32(dst=sgpr("SrdC+2"), src0="BufferOOB", src1=0,
                                       comment="SrdC+2 = BufferOOB if M valid, else 0"))
          # Pass `elementIdx` so the readInput's incrementToNextRow gate can
          # open the CompactLoopStore chain-seed emit for elt-0. Pass
          # `overrideAfterPrimerRows=_emitOverrideRows` so the AFTER-primer
          # encodes the NEXT EMITTING elt's rowInc (look-ahead). Non-CLS:
          # `elementIdx` is unused and `_emitOverrideRows`==0 means no override.
          if self.kernel["GroupLoadStore"]:
            loadInputCode.add(self.parentWriter.readInput(self.kernel, self.ss, 'C', self.kernel["ProblemType"]["DestDataType"], addrCalc, vc0, data, self.gwvw, addrCVgpr, self.tmpS01, elementIdx, self.batchIdx,
                                                          overrideAfterPrimerRows=_emitOverrideRows))
          else:
            module.add(self.parentWriter.readInput(self.kernel, self.ss, 'C', self.kernel["ProblemType"]["DestDataType"], addrCalc, vc0, data, self.gwvw, addrCVgpr, self.tmpS01, elementIdx, self.batchIdx,
                                                  overrideAfterPrimerRows=_emitOverrideRows))
          self.loadedDataBeta[dataBeta] = ceil(self.kernel["ProblemType"]["DestDataType"].numBytes() * self.ss.cfg.gwvw / 16)
          self.loadsBetaIssued += ceil(self.kernel["ProblemType"]["DestDataType"].numBytes() * self.gwvw / 16)
      self.betaLoadIssued.append(len(self.loadedDataBeta) * ceil(self.kernel["ProblemType"]["DestDataType"].numBytes() * self.ss.cfg.gwvw / 16))

      if (self.kernel["ProblemType"]["UseE"] and self.kernel["ProblemType"]["Gradient"] and self.kernel["ProblemType"]["ActivationType"] != 'none') and ((self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1) or self.kernel["StreamK"] > 0):
        tmpInrSgpr = self._epilogScratchSgpr(1)
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'E', self.edge, self.beta, mask, bufferOOB, (elementIdx == 0), self.tmpVgpr, tmpInrSgpr, addrEVgpr, self.addrE, 0))
        self._epilogScratchFree(tmpInrSgpr)
        if dataE not in self.loadedDataE:
          loadOffset = int((self.kernel["ProblemType"]["ComputeDataType"].numRegisters() - self.kernel["ProblemType"]["DataTypeE"].numRegisters()) * self.ss.cfg.gwvw)
          if self.kernel["GroupLoadStore"]:
            loadInputCode.add(self.parentWriter.readInput(self.kernel, self.ss, 'E', self.kernel["ProblemType"]["DataTypeE"], addrCalc, vc0, dataE + loadOffset, self.gwvw, addrEVgpr, self.tmpS01))
          else:
            module.add(self.parentWriter.readInput(self.kernel, self.ss, 'E', self.kernel["ProblemType"]["DataTypeE"], addrCalc, vc0, dataE + loadOffset, self.gwvw, addrEVgpr, self.tmpS01))
          self.loadedDataE[dataE] = ceil(self.kernel["ProblemType"]["DataTypeE"].numBytes() * self.ss.cfg.gwvw / 16)
          self.loadsEIssued += ceil(self.kernel["ProblemType"]["DataTypeE"].numBytes() * self.gwvw / 16)
        self.loadE = True
      else:
        self.loadE = False
      self.eLoadIssued.append(len(self.loadedDataE) * ceil(self.kernel["ProblemType"]["DataTypeE"].numBytes() * self.ss.cfg.gwvw / 16))

      # Per-element epilogue LDS loads (same helper the CLS preamble used for
      # elt 0); if the preamble already loaded it, the ds_load is skipped here.
      self._emitElt0EpilogueLoads(module, addrCalc, mask, elementIdx, bufferOOB,
                                   loadInputCode, factor_gwvw,
                                   False)
      tmpInrSgpr = self._epilogScratchSgpr(1)
      if (self.kernel["ProblemType"]["UseE"] and not self.kernel["ProblemType"]["Gradient"]) and ((self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1) or self.kernel["StreamK"] > 0):
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'E', self.edge, self.beta, mask, bufferOOB, (elementIdx == len(self.batchElements) - 1), self.tmpVgpr, tmpInrSgpr, addrEVgpr, self.addrE, 0))
      if self.storeBiasD == 1:
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'Bias', self.edge, self.beta, mask, bufferOOB, (elementIdx == len(self.batchElements) - 1), self.tmpVgpr, tmpInrSgpr, addrBiasVgpr, self.addrBias, self.factorDim))

      if self.parentWriter.states.useGateResidual and \
         (self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1 or self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel"):
        _gateList = self.kernel["ProblemType"].get("GateResidualDataTypeList", [])
        _destDtype = self.kernel["ProblemType"]["DestDataType"]
        if not _gateList:
          _prologLoadDtype = self.kernel["ProblemType"]["DestDataType"]
        elif len(_gateList) == 1:
          _prologLoadDtype = _gateList[0]
        else:
          _prologLoadDtype = _destDtype if _destDtype in _gateList else _gateList[0]

        _multiDtypeGate = (_gateList and len(_gateList) > 1)

        addrGateVgpr = addrCalc.addrGateVgpr if addrCalc.addrGateVgpr != None else addrDVgpr
        if not _multiDtypeGate and not self.ss.optSingleColVgpr:
          module.add(addrCalc.emitLdChange(
              self.kernel, self.ss, 'Gate', self.edge, self.beta, mask, bufferOOB,
              (elementIdx == 0), self.tmpVgpr, tmpInrSgpr, addrGateVgpr, self.addrD, 0))
        if dataGate not in self.loadedDataGate:
          if self.ss.optSingleColVgpr:
            # opt (single + multi dtype): hoisted to ONE branch after the prolog loop
            # (see _emitHoistedGateLoadPhase).
            pass
          elif not _multiDtypeGate:
            # no-opt single-dtype: unconditional load.
            # Null gate -> SRD num_records==0 -> buffer_load returns 0;
            _glTgt = loadInputCode if self.kernel["GroupLoadStore"] else module
            gateLoadMod = self.parentWriter.readInput(
                self.kernel, self.ss, 'Gate',
                _prologLoadDtype,
                addrCalc, vc0, dataGate, self.gwvw, addrGateVgpr, self.tmpS01)
            _glTgt.add(gateLoadMod)
          else:
            # no-opt (edge) multi-dtype: per-dtype dispatcher per element (gate
            # borrows D's per-element addr, so it must stay interleaved here).
            # all using this element's fresh mask. The dispatcher tail-branches
            # to a common "GateLoad_End" label per element.
            labels = self.parentWriter.labels
            gateLoadEndLabel = Label(
                labels.getNameInc("GateLoad_End_%u"%elementIdx), "")
            gateLoadTypeLabels = [
                Label(labels.getNameInc(
                    "GateLoad_%s_%u"%(g.toNameAbbrev(), elementIdx)), "")
                for g in _gateList]
            gateLoadTypeLabels.append(gateLoadEndLabel)

            # Snapshot mutables we'll restore after the dispatcher chain.
            _savedBpeGate = self.parentWriter.states.bpeGate
            _savedGlobalOffsetGate = addrCalc.globalOffsetGate

            # Per-element skip when gate pointer is null (no-gate problem): jump past
            # the whole dtype dispatcher straight to GateLoad_End for this element.
            module.add(self.parentWriter.getSCMPKInstruction(
                "EQU32", "SrdGate+2", 0, comment="gate disabled? (SrdGate num_records==0)"))
            module.add(SCBranchSCC1(gateLoadEndLabel.getLabelName(), "skip gate load if disabled (null gate)"))

            for i, nextLabel in enumerate(gateLoadTypeLabels[1:]):
              typeValue = _gateList[i].value
              gDtype = _gateList[i]
              module.add(gateLoadTypeLabels[i])
              module.add(self.parentWriter.getSCMPKInstruction(
                  "LGU32", "GateType", typeValue,
                  comment="GateType != %u"%typeValue))
              module.add(SCBranchSCC1(nextLabel.getLabelName(),
                                      "Branch if true (try next gate dtype)"))
              # per-dtype bpe. no-opt path only (opt is handled by the hoisted phase),
              # so gate uses per-element runtime addr -> globalOffsetGate fixed 0.
              perBranchBpe = int(self.parentWriter.states.bpr * gDtype.numRegisters())
              perBranchBpe = max(1, perBranchBpe)
              self.parentWriter.states.bpeGate = perBranchBpe
              addrCalc.globalOffsetGate = 0
              # Force fresh per-dtype gate address setup.
              self.ss.singleColGateAddrUpdated = False
              module.add(addrCalc.emitLdChange(
                  self.kernel, self.ss, 'Gate', self.edge, self.beta, mask, bufferOOB,
                  (elementIdx == 0), self.tmpVgpr, tmpInrSgpr, addrGateVgpr, self.addrD, 0))
              module.add(self.parentWriter.readInput(
                  self.kernel, self.ss, 'Gate', gDtype,
                  addrCalc, vc0, dataGate, self.gwvw, addrGateVgpr, self.tmpS01))
              # Restore bpe/offset for the next branch in this elem.
              self.parentWriter.states.bpeGate = _savedBpeGate
              addrCalc.globalOffsetGate = _savedGlobalOffsetGate
              module.add(SBranch(labelName=gateLoadEndLabel.getLabelName(),
                                 comment="Branch to GateLoad end"))
            module.add(gateLoadEndLabel)
          gateBytes = ceil(_prologLoadDtype.numBytes() * self.ss.cfg.gwvw / 16)
          self.loadedDataGate[dataGate] = gateBytes
          self.loadsGateIssued += gateBytes
        self.gateLoadIssued.append(len(self.loadedDataGate) * ceil(_prologLoadDtype.numBytes() * self.ss.cfg.gwvw / 16))
      else:
        self.gateLoadIssued.append(0)

      if self.kernel["GlobalSplitU"] == 1 or (self.kernel["_GlobalAccumulation"] != "MultipleBufferSingleKernel"): # "SingleBuffer" or "MultipleBuffer"
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'D', self.edge, self.beta, mask, bufferOOB, (elementIdx == len(self.batchElements) - 1), self.tmpVgpr, tmpInrSgpr, addrDVgpr, self.addrD, 0))
      if self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":
        module.add(addrCalc.emitLdChange(self.kernel, self.ss, 'TD', self.edge, self.beta, mask, bufferOOB, (elementIdx == len(self.batchElements) - 1), self.tmpVgpr, tmpInrSgpr, addrCalc.addrGSUSyncVgprs, self.addrD, 0))
      self._epilogScratchFree(tmpInrSgpr)
      if self.atomic and (not self.parentWriter.states.useAtomicAdd):
        # load c into data+1 because of CAS structure
        # TODO - Fix for double here, would need bigger load
        # FIXME
        # gwvw is the number of elements in the batch
        # iterate over number of atomic operations to perform, each of width atomicW
        for avi in range(self.gwvw // self.atomicW):
          dataV = self.ss.elementData[elementIdx] + int(avi*self.ss.cfg.numVgprsPerDataPerVI)
          bpm = self.parentWriter.states.bpeCexternal * self.atomicW
          useBuffer = self.kernel["BufferStore"]
          if self.kernel["BufferStore"]: # yes, BufferStore here - use same addressing regs for this load
            addr0 = vgpr(addrDVgpr)
            addr1 = sgpr("SrdD", 4)
          else:
            addr0 = vgpr(addrDVgpr, 2)
            addr1 = ""
          # Calculate vgpr Index for 32-bit/64-bit instruction
          # DGEMM use SRCS[2] register
          vgprIdx = bpm // 4
          module.add(self.parentWriter.chooseGlobalRead(useBuffer, bpm, dataV + vgprIdx, \
                    addr0, addr1, soffset=0, offset=addrCalc.globalOffset,
                    comment="load D (atomic) bpm=%u vaw=%u"%(bpm,self.atomicW)))

      if self.kernel["InterleaveAlpha"] and self.applyAlpha:
        module.add(self._applyAlpha(self.kernel, self.gwvw, self.ss.elementSumIdx, elementIdx, self.tmpS01))

      if not self.kernel["BufferStore"]:
        # emitLdChange for 'D' (above) already computed addrDVgpr = addrD + offset
        # Only need to build the address here if emitLdChange for 'D' was NOT called
        if not (self.kernel["GlobalSplitU"] == 1 or (self.kernel["GlobalSplitUAlgorithm"] != "MultipleBufferSingleKernel")):
          offsetSrc = (self.tmpVgpr + 2) if self.beta else addrDVgpr

          module.add(VAddCOU32(vgpr(addrDVgpr+0), VCC(), vgpr(self.addrD+0), \
              vgpr(offsetSrc+0), "addrDVgpr = D + index*bytes (lo)"))
          module.add(VAddCCOU32(vgpr(addrDVgpr+1), VCC(), vgpr(self.addrD+1), \
              vgpr(offsetSrc+1), VCC(), "addrDVgpr = D + index*bytes (hi)"))

        # restore full exec mask for calculating addr of next element
        if self.edge and (self.beta or self.loadE or self.atomic):
          module.add(self.getEdgeMovInstType()(EXEC(), -1, "full mask -1 -> exec"))

      if self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":
        if self.ss.optSrdIncForRow and (addrCalc.rowInc or (self.kernel["CompactLoopStore"] and elementIdx == 0 and self.batchIdx == 0)) and self.kernel["StoreRemapVectorWidth"] > 0:
          module.addComment1("StoreRemap: shift coord1 address MultipleBufferSingleKernel")
          if self.kernel["ProblemType"]["UseE"] and (self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1):
            # TODO Check if works with StreamK
            printExit("Use E does not support StoreRemapVectorWidth if GSU == 1.")
            # module.add(addrCalc.incrementToNextRow(self.kernel, "E", self.ss, self.tmpS01, isCompute=True))
          module.add(addrCalc.incrementToNextRow(self.kernel, "D", self.ss, self.tmpS01, forceinitrow0=1, overrideAfterPrimerRows=_emitOverrideRows))
          module.add(VMovB32(vgpr(self.tmpVgpr), addrCalc.rowInc, comment="set shift rows"))
          module.add(VAddU32(vgpr(self.parentWriter.vgprs.storeRemapCoord1), vgpr(self.parentWriter.vgprs.storeRemapCoord1), vgpr(self.tmpVgpr), "shift storeRemap coord1"))

    module.add(loadInputCode)

    # opt + multi-dtype gate: emit the gate prolog load as ONE GateType dispatch
    # (the per-element loop above skipped it on the opt path).
    self._emitHoistedGateLoadPhase(module, bufferOOB)

    # Restore SrdC+2 = BufferOOB after subtile NonEdge C-load OOB gating (which may have set it to 0).
    if self.beta and not self.edge:
      mGuardSgpr = self.parentWriter.states.subtileM32ValidBlocksSgpr
      nGuardSgpr = self.parentWriter.states.subtileN16ValidBlocksSgpr
      if mGuardSgpr is not None or nGuardSgpr is not None:
        module.add(SMovB32(dst=sgpr("SrdC+2"), src="BufferOOB",
                           comment="restore SrdC+2 after subtile NonEdge C-load OOB gating"))

    if self.beta and self.kernel["StoreSyncOpt"]:
      self._storeSyncOpt(module)

    ########################################
    # AccVgpr read
    if self.codeAccVgprRead is not None and (self.kernel["LocalSplitU"] == 1 or self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel"):
      regsPerScalar = self.parentWriter.states.bpeCinternal // self.parentWriter.states.bpr # register per scalar
      #TODOBS: Need to change this, for LSU>1 + subtile impl case
      if self.kernel["MIArchVgpr"] and self.kernel["LocalSplitU"] > 1:
        tmpStartVgprValuC = self.parentWriter.states.c.startVgprValu
        self.parentWriter.states.c.startVgprValu = 0
        module.add(RegSet("v", "vgprValuC", 0))
      # loop over store instructions within one batch
      for elementIdx in range(len(self.batchElements)):
        # loop over scalars within one store instruction
        for vi in range(self.gwvw):
          # loop over registers within one scalar
          for rIdx in range(0, regsPerScalar):
            module.add(replaceHolder(self.codeAccVgprRead.popFirstItem(), self.ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx - self.parentWriter.states.c.startVgprValu))

      if self.kernel["MIArchVgpr"] and self.kernel["LocalSplitU"] > 1:
        self.parentWriter.states.c.startVgprValu = tmpStartVgprValuC
        module.add(RegSet("v", "vgprValuC", tmpStartVgprValuC))

    elif self.kernel["LocalSplitU"] > 1:
      # read from LSU VGPRs
      regsPerScalar = self.parentWriter.states.bpeCinternal // self.parentWriter.states.bpr # register per scalar
      if self.ss.lsuStartVgprOffset > 0:
        for elementIdx in range(len(self.batchElements)):
          for vi in range(self.gwvw):
            for rIdx in range(0, regsPerScalar):
              idx = self.ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi + rIdx - self.parentWriter.states.c.startVgprValu
              module.add(VMovB32(vgpr("ValuC+%u"%(idx)), vgpr("ValuC+%u"%(idx + self.ss.lsuStartVgprOffset)), comment="load from "+str(idx + self.ss.lsuStartVgprOffset)+" to "+str(idx) ))
      self.ss.lsuStartVgprOffset += len(self.batchElements) * self.gwvw * regsPerScalar

      if not self.kernel["MIArchVgpr"]:
        module.add(SNop(1, "2 wait states required before reading vgpr"))

    if self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":
      module.addComment1("store after Acc, "+"GSU: "+str(self.kernel["GlobalSplitU"]))

    storeCodeGSUSK = Module("GroupLoadStore")
    if self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":#GSUGSU
      storeWidth = self.kernel["StoreVectorWidth"]
      for elementIdx in range(0, len(self.batchElements)):
        addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
        if self.batchIdx == 0 and elementIdx == 0:
          addrDVgpr = addrCalc.addrDVgpr
          storeCodeGSUSK.add(vectorStaticMultiply(vgpr(addrDVgpr), vgpr("Serial"), storeWidth * self.parentWriter.states.bpeCinternal, ContinuousRegister(self.tmpS01, 1)))
          storeCodeGSUSK.add(SMovB32(dst=sgpr(self.tmpS01), src=0, comment="Init sgpr offset"))
          storeCodeGSUSK.addSpaceLine()
        if (self.kernel["ProblemType"]["UseE"] and not self.kernel["ProblemType"]["Gradient"]) and ((self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1) or self.kernel["StreamK"] > 0):
          vgprIdx = self.ss.elementSumIdx[elementIdx] - self.parentWriter.states.c.startVgprValu
          vgprDst = self.activationSetPCStruct.vgprActCopy if mergeActFuncCall else "ValuC+%d"%vgprIdx
          module.add(self.parentWriter.addStore(self.kernel, self.ss, 'E', addrCalc, vgprDst, self.tmpS01, self.edge, comment="store E"))

        sumIdx = self.ss.elementSumIdx[elementIdx]
        if self.kernel["StoreRemapVectorWidth"]:
          rpe = self.parentWriter.states.bpeCinternal // self.parentWriter.states.bpr
          module.add(self.parentWriter.storeRemapAddLocalWrite(self.kernel, self.ss, addrCalc, sumIdx*rpe))
          # Column Block Shape has been written to LDS
          # Now read back and write out to global memory
      # module.add(storeCodeGSUSK)

    if self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel" and self.kernel["StoreRemapVectorWidth"]:
      if self.parentWriter.StoreRemapLastBatch == 1:
        module.addComment1("Handle local read and global write")
        tmpInrSgpr = self._epilogScratchSgpr(2)
        storeModule, numNewStores = self.parentWriter.storeRemapAddStore(self.kernel, self.tmpVgpr, tmpInrSgpr, self.edge, self.parentWriter.StoreRemapLastBatch)
        self._epilogScratchFree(tmpInrSgpr)
        module.add(storeModule)
        self.storesIssued += numNewStores

    gsuComponent = Component.GSU.find(self.parentWriter)
    module.add(gsuComponent.globalWriteBatchProlog(self.parentWriter, self.kernel, self.tmpVgpr, self.tmpVgprSize, self.tmpVgprDynamic, \
                                                   self.batchIdx, self.ss, self.gwvw, self.batchElements, \
                                                   self.beta, self.edge, sumIdxGSUSYNC, addrCalc))

    # rC *= alpha
    if not self.kernel["InterleaveAlpha"] and self.applyAlpha and not self.parentWriter.alphaBeforeLoadC:
      module.addComment1("rC *= alpha batchElements=%s"%self.batchElements)
      if self.codeMulAlpha is None:
        elementIdx = 0
        while elementIdx < len(self.batchElements):
          isEnd = (elementIdx == len(self.batchElements) - 1)
          if not isEnd and (self.ss.elementSumIdx[elementIdx] + 1 == self.ss.elementSumIdx[elementIdx + 1]) and (self.ss.elementSumIdx[elementIdx] % 2 == 0):
            module.add(self._applyAlpha(self.kernel, self.gwvw, self.ss.elementSumIdx, elementIdx, self.tmpS01, usePK=True))
            elementIdx += 2
          else:
            module.add(self._applyAlpha(self.kernel, self.gwvw, self.ss.elementSumIdx, elementIdx, self.tmpS01))
            elementIdx += 1
      else:
          regsPerScalar = self.parentWriter.states.bpeCinternal // self.parentWriter.states.bpr # register per scalar
          for elementIdx in range(len(self.batchElements)):
            for vi in range(self.gwvw):
              rh = replaceHolder(self.codeMulAlpha.popFirstItem(), self.ss.elementSumIdx[elementIdx]*regsPerScalar + regsPerScalar*vi - self.parentWriter.states.c.startVgprValu)
              if (self.kernel["GlobalSplitU"] == 1) and (self.kernel["ProblemType"]["ComputeDataType"].isSingle() and self.kernel["ProblemType"]["DataType"].isInt8()):
                srcRegName = rh.getParams()[2].getCompleteRegName()
                module.add(VCvtI32toF32(dst=vgpr(srcRegName), src=vgpr(srcRegName), comment="Convert MI out reg to fp32"))
              module.add(rh)

  def _epilog(self, module: Module):
    # return registers to pool:
    lastDataD       = -1
    lastDataE       = -1
    checkedDataBias = {}
    checkedDataGate = {}
    checkedDataScaleAVec = {}
    checkedDataScaleBVec = {}
    checkedDataScaleAlphaVec = {}
    for elementIdx in range(len(self.batchElements)):
      sumIdxGSUSYNC = self.ss.elementSumIdx[elementIdx]
      if not self.ss.sharedColDVgprs:
        addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
        addrEVgpr    = addrCalc.addrEVgpr
        addrDVgpr    = addrCalc.addrDVgpr
        addrGSUSyncVgprs    = addrCalc.addrGSUSyncVgprs
        addrCVgpr    = addrCalc.addrCVgpr
        addrBiasVgpr = addrCalc.addrBiasVgpr
        addrGateVgpr = addrCalc.addrGateVgpr
        addrScaleAVecVgpr = addrCalc.addrScaleAVecVgpr
        addrScaleBVecVgpr = addrCalc.addrScaleBVecVgpr
        addrScaleAlphaVecVgpr = addrCalc.addrScaleAlphaVecVgpr
        if addrEVgpr != None:
          self.parentWriter.vgprPool.checkIn(addrEVgpr)
        self.parentWriter.vgprPool.checkIn(addrDVgpr)
        if addrCVgpr != addrDVgpr:
          self.parentWriter.vgprPool.checkIn(addrCVgpr)
        if addrGSUSyncVgprs != None:
          self.parentWriter.vgprPool.checkIn(addrGSUSyncVgprs)
        if addrBiasVgpr != None:
          self.parentWriter.vgprPool.checkIn(addrBiasVgpr)
        # No-opt path: addrGateVgpr aliases addrDVgpr
        if addrGateVgpr != None and addrGateVgpr != addrDVgpr:
          self.parentWriter.vgprPool.checkIn(addrGateVgpr)
        if addrScaleAVecVgpr != None:
          self.parentWriter.vgprPool.checkIn(addrScaleAVecVgpr)
        if addrScaleBVecVgpr != None:
          self.parentWriter.vgprPool.checkIn(addrScaleBVecVgpr)
        if addrScaleAlphaVecVgpr != None:
          self.parentWriter.vgprPool.checkIn(addrScaleAlphaVecVgpr)

      data = self.ss.elementData[elementIdx]
      if data != 0:
        if data != lastDataD:
          self.parentWriter.vgprPool.checkIn(data)
        lastDataD = data

      dataBias = self.ss.elementDataBias[elementIdx]
      if dataBias != 0:
        if dataBias not in checkedDataBias:
          self.parentWriter.vgprPool.checkIn(dataBias)
        checkedDataBias[dataBias] = 1

      dataE = self.ss.elementDataE[elementIdx]
      if dataE != 0:
        if dataE != lastDataE:
          self.parentWriter.vgprPool.checkIn(dataE)
        lastDataE = dataE

      if self.parentWriter.states.useGateResidual:
        dataGate = self.ss.elementDataGate[elementIdx]
        if dataGate != 0:
          if dataGate not in checkedDataGate:
            self.parentWriter.vgprPool.checkIn(dataGate)
          checkedDataGate[dataGate] = 1

      def checkScaleVec(dataScaleVec, checkedDataScaleVec):
        if dataScaleVec != 0:
          if dataScaleVec not in checkedDataScaleVec:
            self.parentWriter.vgprPool.checkIn(dataScaleVec)
          checkedDataScaleVec[dataScaleVec] = 1

      checkScaleVec(self.ss.elementDataScaleAVec[elementIdx], checkedDataScaleAVec)
      checkScaleVec(self.ss.elementDataScaleBVec[elementIdx], checkedDataScaleBVec)
      checkScaleVec(self.ss.elementDataScaleAlphaVec[elementIdx], checkedDataScaleAlphaVec)

    self.ss.firstBatch = False
    self.ss.checkInTempVgprC()
    if self.kernel["_GlobalAccumulation"] != "MultipleBufferSingleKernel" and self.kernel["StoreRemapVectorWidth"]:
      if self.parentWriter.StoreRemapLastBatch == 1:
        module.addComment1("Handle local read and global write")
        # this seems buggy? it's possible to issue more than one stores for SR
        # module.add(self.storeRemapAddStore(kernel, tmpVgpr, tmpS01, edge))
        # storesIssued += 1
        tmpInrSgpr = self._epilogScratchSgpr(2)
        storeModule, numNewStores = self.parentWriter.storeRemapAddStore(self.kernel, self.tmpVgpr, tmpInrSgpr, self.edge, self.parentWriter.StoreRemapLastBatch)
        self._epilogScratchFree(tmpInrSgpr)
        module.add(storeModule)
        self.storesIssued += numNewStores

    if self.parentWriter.states.serializedStore:
      module.add(SNop(0, "1 wait state required when next inst writes vgprs held by previous dwordx4 store inst"))

    if self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":
      module.addCommentAlign("GW end") #GSUSYNC

  def _emitAdd(self, module: Module):
    if self.atomic:
      del self.tmpVgpr # catch bugs
      if self.parentWriter.states.useAtomicAdd:
        self._emitAtomicAdd(module)
      else:
        self._emitCasAdd(module)
    else:
      self._emitNonatomicAdd(module)

  def _emitGateCvt(self, dataGate, gateDtype):
    cvtMod = Module("GateCvt_%s" % gateDtype.toNameAbbrev())
    if gateDtype.isSingle():
      pass
    elif gateDtype.isHalf():
      for vi in range(self.gwvw - 1, -1, -1):
        sel = HighBitSel.LOW if (vi % 2) == 0 else HighBitSel.HIGH
        cvtMod.add(ECvtF16toF32(dst=vgpr(dataGate + vi), src=vgpr(dataGate + vi // 2), sel=sel,
          comment="GateCvt[h]: f16 -> f32 (vi=%d)"%vi))
    elif gateDtype.isBFloat16():
      # bf16 = top 16 bits of an f32; lo half shifts up, hi half masks.
      for vi in range(self.gwvw - 1, -1, -1):
        if (vi % 2) == 0:
          cvtMod.add(VLShiftLeftB32(dst=vgpr(dataGate + vi), shiftHex=16, src=vgpr(dataGate + vi // 2),
            comment="GateCvt[bf16]: lo bf16 -> f32 (vi=%d)"%vi))
        else:
          cvtMod.add(VAndB32(dst=vgpr(dataGate + vi), src0=hex(0xffff0000), src1=vgpr(dataGate + vi // 2),
            comment="GateCvt[bf16]: hi bf16 -> f32 (vi=%d)"%vi))
    elif gateDtype.isAnyFloat8() or gateDtype.isAnyBFloat8():
      # PK-cvt 2 elements per instruction into a tmp, then move to dataGate.
      isFP8 = gateDtype.isAnyFloat8()
      CvtPk = ECvtPkFP8toF32 if isFP8 else ECvtPkBF8toF32
      cvtTmp = self.tmpVgpr
      for vi in range(((self.gwvw - 1) // 2) * 2, -1, -2):
        sel = HighBitSel.LOW if (vi % 4) == 0 else HighBitSel.HIGH
        cvtMod.add(CvtPk(dst=vgpr(cvtTmp, 2), src=vgpr(dataGate + vi // 4), sel=sel,
          comment="GateCvt[%s]: 2x8bit -> 2xf32 (vi=%d)"%(gateDtype.toNameAbbrev(), vi)))
        for j in range(2):
          if (vi + j) < self.gwvw:
            cvtMod.add(VMovB32(dst=vgpr(dataGate + vi + j), src=vgpr(cvtTmp + j),
              comment="GateCvt[%s]: store f32 (vi=%d)"%(gateDtype.toNameAbbrev(), vi + j)))
    elif gateDtype.isInt8():
      # int8 = 4 signed bytes/dword; extract+sign-extend byte (vi%4) to i32, then -> f32.
      for vi in range(self.gwvw - 1, -1, -1):
        dst = dataGate + vi
        if (vi % 4) != 3:
          cvtMod.add(VMovB32(dst=vgpr(self.tmpVgpr), src=hex((vi % 4) * 8), comment="byte offset"))
          cvtMod.add(VBfeI32(dst=vgpr(dst), src0=vgpr(dataGate + vi // 4), src1=vgpr(self.tmpVgpr), src2=8,
            comment="GateCvt[i8]: int8 -> int32 (vi=%d)"%vi))
        else:
          cvtMod.add(VAShiftRightI32(dst=vgpr(dst), shiftHex=24, src=vgpr(dataGate + vi // 4),
            comment="GateCvt[i8]: int8(byte3) -> int32 (vi=%d)"%vi))
        cvtMod.add(VCvtI32toF32(dst=vgpr(dst), src=vgpr(dst), comment="GateCvt[i8]: int32 -> f32 (vi=%d)"%vi))
    elif gateDtype.isInt32():
      # int32 = 1 value/dword; cvt in place to f32.
      for vi in range(self.gwvw - 1, -1, -1):
        cvtMod.add(VCvtI32toF32(dst=vgpr(dataGate + vi), src=vgpr(dataGate + vi),
          comment="GateCvt[i32]: int32 -> f32 (vi=%d)"%vi))
    else:
      raise RuntimeError(
          "GateResidual: unsupported gate dtype %s" % str(gateDtype))
    return cvtMod

  def _emitHoistedGateLoadPhase(self, module: Module, bufferOOB):
    """opt path: emit the gate prolog load wrapped in ONE null-gate skip branch."""
    if not (self.parentWriter.states.useGateResidual and
            (self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1 or self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel")):
      return
    if not self.ss.optSingleColVgpr:
      return  # no-opt path keeps the per-element load in the prolog loop
    gateList = list(self.kernel["ProblemType"].get("GateResidualDataTypeList", []))
    if not gateList:
      return
    multi = len(gateList) > 1

    labels = self.parentWriter.labels
    endLabel = Label(labels.getNameInc("GateLoadAll_End"), "")
    savedBpe = self.parentWriter.states.bpeGate

    def _emitLoadsFor(gDtype):
      bpe = max(1, int(self.parentWriter.states.bpr * gDtype.numRegisters()))
      self.parentWriter.states.bpeGate = bpe
      self.ss.singleColGateAddrUpdated = False
      seen = set()
      for ei, element in enumerate(self.batchElements):
        addrCalc = self.ss.elementAddr[ei]
        dataGate = self.ss.elementDataGate[ei]
        if dataGate == 0 or dataGate in seen:
          continue
        seen.add(dataGate)
        addrGateVgpr = addrCalc.addrGateVgpr  # opt: gate's own sharedColGateVgprs
        addrCalc.globalOffsetGate = addrCalc.coordOffset0 * bpe
        module.add(addrCalc.emitLdChange(
            self.kernel, self.ss, 'Gate', self.edge, self.beta, self.ss.elementMask[ei],
            bufferOOB, (ei == 0), self.tmpVgpr, self.tmpSgpr, addrGateVgpr, self.addrD, 0))
        module.add(self.parentWriter.readInput(
            self.kernel, self.ss, 'Gate', gDtype, addrCalc, element[3], dataGate,
            self.gwvw, addrGateVgpr, self.tmpS01))

    if not multi:
      # single-dtype: no GateType dispatch needed, just the loads.
      _emitLoadsFor(gateList[0])
    else:
      typeLabels = [Label(labels.getNameInc("GateLoadAll_%s"%g.toNameAbbrev()), "") for g in gateList]
      typeLabels.append(endLabel)
      for i, nextLabel in enumerate(typeLabels[1:]):
        gDtype = gateList[i]
        module.add(typeLabels[i])
        module.add(self.parentWriter.getSCMPKInstruction(
            "LGU32", "GateType", gDtype.value, comment="GateType != %u"%gDtype.value))
        module.add(SCBranchSCC1(nextLabel.getLabelName(), "Branch if true (try next gate dtype)"))
        _emitLoadsFor(gDtype)
        module.add(SBranch(labelName=endLabel.getLabelName(), comment="Branch to GateLoadAll end"))
    module.add(endLabel)
    self.parentWriter.states.bpeGate = savedBpe

  def _emitGateCvtAllPhase(self, module: Module):
    """Multi-dtype Gate: convert ALL loaded gate values to f32 in one GateType
    dispatch."""
    if not (self.parentWriter.states.useGateResidual and
            (self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1 or self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel")):
      return
    gateList = list(self.kernel["ProblemType"].get("GateResidualDataTypeList", []))
    if len(gateList) <= 1:
      if self.ss.optSingleColVgpr:
        module.add(SWaitCnt(vlcnt=0, comment="GateResidual: wait for hoisted single-dtype gate loads"))
      return

    labels = self.parentWriter.labels
    cvtEndLabel = Label(labels.getNameInc("GateCvtAll_End"), "")
    cvtTypeLabels = [Label(labels.getNameInc("GateCvtAll_%s"%g.toNameAbbrev()), "") for g in gateList]
    cvtTypeLabels.append(cvtEndLabel)

    # Skip the whole gate cvt phase when the gate pointer is null (no-gate problem).
    module.add(self.parentWriter.getSCMPKInstruction(
        "EQU32", "SrdGate+2", 0, comment="gate disabled? (SrdGate num_records==0)"))
    module.add(SCBranchSCC1(cvtEndLabel.getLabelName(), "skip gate cvt if disabled (null gate)"))

    # One wait for all prolog gate loads (multi-dtype already paid a blunt wait
    # per element previously; this hoists it to a single wait).
    module.add(SWaitCnt(vlcnt=0, comment="GateResidual: wait for prolog gate loads (branch-once cvt)"))
    for i, nextLabel in enumerate(cvtTypeLabels[1:]):
      gDtype = gateList[i]
      module.add(cvtTypeLabels[i])
      module.add(self.parentWriter.getSCMPKInstruction(
          "LGU32", "GateType", gDtype.value, comment="GateType != %u"%gDtype.value))
      module.add(SCBranchSCC1(nextLabel.getLabelName(), "Branch if true (try next gate dtype)"))
      seen = set()
      for ei in range(len(self.batchElements)):
        dg = self.ss.elementDataGate[ei]
        if dg != 0 and dg not in seen:
          seen.add(dg)
          module.add(self._emitGateCvt(dg, gDtype))
      module.add(SBranch(labelName=cvtEndLabel.getLabelName(), comment="Branch to GateCvtAll end"))
    module.add(cvtEndLabel)

  def _emitNonatomicAdd(self, module: Module):
    ########################################
    # Not Atomic
    ########################################
    # edge has v_cndmask so loads or stores may not issue, hard to track vmcnt:
    interleaveStoreVmcnt = self.parentWriter.states.interleaveStoreVmcnt and not self.edge

    for elementIdx in range(len(self.batchElements)):
      for vi in range(self.gwvw):
        sumIdxV = self.ss.elementSumIdx[elementIdx] + vi
        newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
        # covers sgemm, gemm_ex(HHS/HSS/BBS/BSS (HPA=T)), int8 (int8x4?)
        if self.kernel["ProblemType"]["ComputeDataType"].isInt32() or \
            self.kernel["ProblemType"]["ComputeDataType"].isSingle(): # covers sgemm/gemm_ex(HHS/HSS/BBS/BSS)
            if self.debugConfig["ForceExpectedValue"]:
              module.add(VMovB32(vgpr("ValuC+%u"%newSumIdxV), self.debugConfig["ValueCExpectedValue"], "force expected value" ))
            if self.parentWriter.db["ForceVSerial"]:
              module.add(VMovB32(vgpr("ValuC+%u"%newSumIdxV), vgpr("Serial"), "force expected value to serial" ))
            if self.parentWriter.db["CheckValueC"]:
              module.add(SMovB32(sgpr(self.tmpS01), self.debugConfig["ValueCExpectedValue"], "Move expected value"))
              module.add(self.parentWriter.getCmpAssert(self.parentWriter.asmAssert.eq, vgpr("ValuC+%u"%newSumIdxV), sgpr(self.tmpS01)))

    ########################################
    # wait for batched load
    # Here we wait all
    if not interleaveStoreVmcnt:
      waitcntInst = self.globalStoreWait(0, [], 0, 0, False)
      if waitcntInst:
        module.add(waitcntInst)

    if self.kernel["ProblemType"]["StochasticRounding"]:
      if self.parentWriter.states.asmCaps["v_prng_b32"]:
        vgprRND = self.parentWriter.vgprPool.checkOut(1, tag="_emitNonatomicAdd_vgprRND")
      else:
        # legacy PRNG approach needs extra 2 VGPRs
        # Ref.: Module("StochasticRoundingCvt")
        vgprRND = self.parentWriter.vgprPool.checkOut(3, tag="_emitNonatomicAdd_vgprRND2")

    # Multi-dtype Gate: branch-once cvt of all loaded gate values to f32.
    self._emitGateCvtAllPhase(module)

    module.addComment1("apply mask, calc new C and issue writes")
    # module.add(self.getBomb()) # can see store addresses just before the store inst

    activationCDataType = self.kernel["ProblemType"]["ActivationComputeDataType"]

    if self.kernel["_GlobalAccumulation"] != 'MultipleBuffer':
      if self.kernel["ProblemType"]["DestDataType"].isBFloat16() and self.kernel["ProblemType"]["HighPrecisionAccumulate"]:
        module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprBf16Mask), "0xffff0000", comment="mask for pack two bfloat16 element to 32bit" ))
        module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprFp32Nan), "0x7fff0000", comment="fp32 Nan" ))
        module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprBf16Inc), "0x7fff", comment="rounding bias for bfloat16" ))
    # is16bitSubtile: controls partner-lane address setup for dwordx4 paired-subtile stores.
    # Must match is16bitSubtilePaired (per-element store dispatch) exactly.
    # Excluded for "MultipleBufferSingleKernel" and "MultipleBuffer" (StreamK partial-tile
    # workspace path) — both write float32 to workspace, not 16bit to D output.
    isSubtileNonEdge = (
      self.kernel.get("UseSubtileImpl") and not self.edge
      and self.kernel["_GlobalAccumulation"] not in ("MultipleBufferSingleKernel", "MultipleBuffer")
    )
    is16bitSubtile = (
      isSubtileNonEdge
      and (self.kernel["ProblemType"]["DestDataType"].isBFloat16() or
           self.kernel["ProblemType"]["DestDataType"].isHalf())
      and self.kernel["ProblemType"]["HighPrecisionAccumulate"]
      and self.kernel["WavefrontSize"] != 32  # wave32: skip permute-based packed store (uses wave64-only ops)
    )
    if is16bitSubtile:
      assert self.kernel["BufferStore"], \
        "UseSubtileImpl 16bit optimized store requires BufferStore=1"
      # Compute ds_permute partner-lane address for 16bit dwordx4 paired-subtile stores.
      # vtmp1 = lane_id, vtmp2 = partner_lane_id (for ds_permute forward scatter)
      # After v_permlane32_swap + v_permlane16_swap + exec masking:
      #   each lane ends up with the lane_id of the partner that will scatter data to it.
      # vPermAddr = partner_lane_id * 4  (byte address for ds_permute_b32)
      vPermAddr = self.cvtVgprStruct.vgprPermAddr
      vTmp = self.cvtVgprStruct.vgprBf16Temp  # reuse scratch temp before it's used for mask init
      module.addComment1("16bit dwordx4 UseSubtileImpl: compute ds_permute partner-lane address")
      module.add(VAndB32(dst=vgpr(vTmp),     src0=self.kernel["WavefrontSize"]-1, src1=vgpr("Serial"), comment="lane_id & (WS-1)"))
      module.add(VAndB32(dst=vgpr(vPermAddr), src0=self.kernel["WavefrontSize"]-1, src1=vgpr("Serial"), comment="copy of lane_id"))
      module.add(VPermlane32SwapB32(dst=vgpr(vTmp), src=vgpr(vTmp), comment="lane XOR 32 swap"))
      module.add(SNop(waitState=0, comment="delay after v_permlane32_swap"))
      module.add(VPermlane16SwapB32(dst=vgpr(vTmp), src=vgpr(vTmp), comment="lane XOR 16 swap"))
      # Exec mask: lanes where both XOR swaps changed the value (i.e., the 'first' half of each pair)
      # selects lanes 0-15 and 32-47 within the wave.
      stmp = self.parentWriter.sgprPool.checkOutAligned(2,2, tag="_emitNonatomicAdd_stmp")
      module.add(SMovB32(dst=sgpr(stmp), src="0x0000ffff", comment="select lanes 0-15, 32-47"))
      module.add(SMovB32(dst=sgpr(stmp+1), src="0xffff0000"))
      module.add(VCndMaskB32(dst=vgpr(vTmp), src0=vgpr(vTmp), src1=vgpr(vPermAddr), src2=sgpr(stmp,2), comment="restore original lane_id for selected lanes"))
      self.parentWriter.sgprPool.checkIn(stmp)
      module.add(VLShiftLeftB32(dst=vgpr(vPermAddr), shiftHex=2, src=vgpr(vTmp), comment="partner_lane * 4 = ds_permute byte addr"))
      # Pre-compute lane_group*8 once; reused as the row-byte address correction in every
      # paired dwordx4 store (addrDVgpr encodes lane_group*8 but we need lane_group*16).
      vLGDelta = self.cvtVgprStruct.vgprLaneGroupDelta
      module.addComment1("16bit dwordx4: pre-compute lane_group*8 row-byte correction")
      module.add(VAndB32(dst=vgpr(vLGDelta), src0=self.kernel["WavefrontSize"]-1, src1=vgpr("Serial"),
                         comment="lane_id = Serial & (WS-1)"))
      module.add(VLShiftRightB32(dst=vgpr(vLGDelta), shiftHex=4, src=vgpr(vLGDelta),
                                 comment="lane_group = lane_id >> 4"))
      module.add(VLShiftLeftB32(dst=vgpr(vLGDelta), shiftHex=3, src=vgpr(vLGDelta),
                                comment="vgprLaneGroupDelta = lane_group * 8"))
      # Compute bpe scale shift once (compile-time constant); used inside
      # _emit16bitSubtilePairedStore to adjust addrDVgpr inline without
      # modifying it in place, so no restore loop is needed after the stores.
    elif self.kernel["_GlobalAccumulation"] != 'MultipleBuffer':
      if self.kernel["ProblemType"]["DestDataType"].isFloat8_fnuz() and self.kernel["ProblemType"]["HighPrecisionAccumulate"]:
        module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprFp8NanInf), "0x207", comment="Nan and +/- inf" ))
        module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprFp8Max), "0x43700000", comment="Fp8 Max value 240 as float32" ))
        module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprFp8Min), "0xc3700000", comment="Fp8 Min value -240 as float32" ))
      elif self.kernel["ProblemType"]["DestDataType"].isFloat8() and self.kernel["ProblemType"]["HighPrecisionAccumulate"]:
        module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprFp8NanInf), "0x207", comment="Nan and +/- inf" ))
        module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprFp8Max), "0x43E00000", comment="Fp8 Max value 448 as float32" ))
        module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprFp8Min), "0xc3E00000", comment="Fp8 Min value -448 as float32" ))
      elif self.kernel["ProblemType"]["DestDataType"].isAnyBFloat8() and self.kernel["ProblemType"]["HighPrecisionAccumulate"]:
        module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprBF8NanInf), "0x207", comment="Nan and +/- inf" ))
        module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprBF8Max), "0x47600000", comment="BF8 Max value 57344 as float32" ))
        module.add(VMovB32(vgpr(self.cvtVgprStruct.vgprBF8Min), "0xc7600000", comment="BF8 Min value -57344 as float32" ))

    storeCode = Module("GroupLoadStore")
    vlcntTotalIssued = self.loadsBetaIssued + self.loadsEIssued + self.loadsGateIssued
    dscntTotalIssued = self.localLoadsBiasIssued + self.loadsScaleAVecIssued + self.loadsScaleBVecIssued + self.loadsScaleAlphaVecIssued
    waitCnter = [vlcntTotalIssued, dscntTotalIssued]
    for elementIdx in range(0, len(self.batchElements)):
      element = self.batchElements[elementIdx]
      addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
      addr = addrCalc.addrDVgpr
      dataE = self.ss.elementDataE[elementIdx]
      dataGate = self.ss.elementDataGate[elementIdx] if self.parentWriter.states.useGateResidual else 0
      dataBias = self.ss.elementDataBias[elementIdx]
      dataScaleAVec = self.ss.elementDataScaleAVec[elementIdx]
      dataScaleBVec = self.ss.elementDataScaleBVec[elementIdx]
      dataScaleAlphaVec = self.ss.elementDataScaleAlphaVec[elementIdx]
      mask = self.ss.elementMask[elementIdx]
      vc0 = element[3]

      # When skipRearrangement is True:
      # - Data is at WMMA output (v[0:N]), not at elementSumIdx (v[144:N])
      # - Store should read from WMMA output directly
      # Note: Beta paths need the rearrangement because they load C and compute rC = alpha*rC + beta*C
      #
      # skipRearrangement requires elementSumIdx to be contiguous so that
      # (elementSumIdx[i] - elementSumIdx[0]) == i.  When checkOutAligned
      # returns non-contiguous blocks (gaps in the pool), the relative offset
      # no longer matches the sequential MFMA output layout, causing stores
      # to read from wrong VGPRs.
      esIdx = self.ss.elementSumIdx
      contiguous = all(esIdx[i] - esIdx[0] == i for i in range(len(esIdx)))
      if self.skipRearrangement and contiguous:
        sumIdx = esIdx[elementIdx] - esIdx[0]
      else:
        sumIdx = self.ss.elementSumIdx[elementIdx]

      # CompactLoopStore look-ahead override for this store loop (see
      # _lookaheadRowInc). Separate for-loop pass from the _prolog one, so it is
      # recomputed here.
      _emitOverrideRows = self._lookaheadRowInc(elementIdx)

      # print(str(element)+" rowInc="+str(addrCalc.rowInc))
      # Already write wave column block into LDS
      # Now read lds data back to registers and write to global memroy
      if self.kernel["_GlobalAccumulation"] != "MultipleBufferSingleKernel":
        if self.ss.optSrdIncForRow and (addrCalc.rowInc or (self.kernel["CompactLoopStore"] and elementIdx == 0 and self.batchIdx == 0)) and self.kernel["StoreRemapVectorWidth"] > 0:
          module.addComment1("StoreRemap: shift coord1 address")
          if self.kernel["ProblemType"]["UseE"] and (self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1):
            # TODO Check if works with StreamK
            printExit("Use E does not support StoreRemapVectorWidth if GSU == 1.")
            # module.add(addrCalc.incrementToNextRow(self.kernel, "E", self.ss, self.tmpS01, isCompute=True))
          module.add(addrCalc.incrementToNextRow(self.kernel, "D", self.ss, self.tmpS01, forceinitrow0=1, overrideAfterPrimerRows=_emitOverrideRows))
          module.add(VMovB32(vgpr(self.tmpVgpr), addrCalc.rowInc, comment="set shift rows"))
          module.add(VAddU32(vgpr(self.parentWriter.vgprs.storeRemapCoord1), vgpr(self.parentWriter.vgprs.storeRemapCoord1), vgpr(self.tmpVgpr), "shift storeRemap coord1"))

      # When stores are interleaved (GLS=0) with subtile NonEdge guards, the
      # M-guard branch for the last store in N-group K targets the N-group end
      # label.  That label must be placed BEFORE the beta*C fmacs for N-group K+1,
      # otherwise the M-guard branch skips the fmacs and the next N-group stores zeros.
      if isSubtileNonEdge and not self.kernel["GroupLoadStore"]:
        blockIdxN = element[0]
        if blockIdxN != self._subtilePrevBlockIdxN and self._subtileNGroupSkipLabel is not None:
          module.add(self._subtileNGroupSkipLabel)
          self._subtileNGroupSkipLabel = None
          if self._subtilePendingSrdDInc is not None:
            module.add(self._subtilePendingSrdDInc)
            self._subtilePendingSrdDInc = None

      # apply in-bounds exec mask
      if self.edge and not self.kernel["BufferStore"]:
        module.add(self.getEdgeMovInstType()(EXEC(), sgpr(mask, self.laneSGPRC), "sgprs -> exec"))

      if interleaveStoreVmcnt:
        waitcntInst = self.globalStoreWait(elementIdx, waitCnter, vlcntTotalIssued, dscntTotalIssued, True)
        if waitcntInst:
          module.addSpaceLine()
          module.add(waitcntInst)

      def applyScaleVec(vecModule, addressStr, dataScaleVec, factorDim, isGlobal=True):
        if not self.beta and not self.applyAlpha: # case for beta-0 and alpha == 1,(OptNLL)
          if (self.kernel["ProblemType"]["DestDataType"].isInt8() or self.kernel["ProblemType"]["DestDataType"].isInt32() or \
              (self.kernel["ProblemType"]["DataType"].isInt8() and self.kernel["ProblemType"]["DestDataType"].isHalf()) or \
              (self.kernel["ProblemType"]["DataType"].isInt8() and self.kernel["ProblemType"]["DestDataType"].isBFloat16())) and \
            self.kernel["ProblemType"]["ComputeDataType"].isSingle():
            module.add(convertData(self.gwvw, self.ss.elementSumIdx[elementIdx], cvtType=CvtType.CVT_I32_to_F32, \
                                        inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu))

        if self.kernel["ProblemType"]["ComputeDataType"].isSingle():
          maskConst = 1.0
        elif self.kernel["ProblemType"]["ComputeDataType"].isInt32():
          maskConst = 1

        gwvw = 1 if factorDim else self.gwvw
        if isGlobal:
          vecModule.add(VCmpGtU32(dst=sgpr("Address%s"%addressStr, self.parentWriter.states.laneSGPRCount), src0=sgpr("Srd%s+2"%addressStr), src1=0, comment=" == 0 ?"))
          for vi2 in range(0, gwvw):
            vecModule.add(VCndMaskB32(
              dst=vgpr(dataScaleVec + vi2), \
              src1=vgpr(dataScaleVec + vi2), \
              src0=maskConst, \
              src2=sgpr("Address%s"%addressStr, self.parentWriter.states.laneSGPRCount), \
              comment="1. mul 1 if 0"))
        if factorDim and self.gwvw > 1:
          vecModule.add(VMovB32(dst=vgpr(dataScaleVec+1), src=vgpr(dataScaleVec), comment="copy data%s to data%s+1"%(addressStr, addressStr)))

        for vi in range(0, self.gwvw):
          inputScaleVecVgpr = dataScaleVec + (0 if factorDim else vi)
          sumIdxV   = self.ss.elementSumIdx[elementIdx] + vi
          if self.kernel["ProblemType"]["ComputeDataType"].isSingle():
            vgprIdx = sumIdxV - self.parentWriter.states.c.startVgprValu
            # Generate single f32 code if edge is detected.
            if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):
              vecModule.add(VMulF32(dst=vgpr("ValuC+%d"%vgprIdx), src0=vgpr(inputScaleVecVgpr), src1=vgpr("ValuC+%d"%vgprIdx), comment="*= %sVMul"%addressStr ))
            # Original packed route
            elif vi%2 == 1:
              assert (self.gwvw % 2 == 0)
            else:
              vecModule.add(VMulPKF32(dst=vgpr("ValuC+%d"%vgprIdx, 2), src0=vgpr(inputScaleVecVgpr, 2), src1=vgpr("ValuC+%d"%vgprIdx, 2), comment="*= %sVMulPK(%d)(%d)"%(addressStr, dataScaleVec,vi)))
          elif self.kernel["ProblemType"]["ComputeDataType"].isInt32():
            vgprIdx = sumIdxV - self.parentWriter.states.c.startVgprValu
            # Generate single i32 code if edge is detected.
            if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):
              vecModule.add(VMulLOU32(dst=vgpr("ValuC+%d"%vgprIdx), src0=vgpr(inputScaleVecVgpr), src1=vgpr("ValuC+%d"%vgprIdx), comment="*= %sVMul"%addressStr ))
            elif vi%2 == 1:
              assert (self.gwvw % 2 == 0)
            else:
              vecModule.add(VMulLOU32(dst=vgpr("ValuC+%d"%vgprIdx), src0=vgpr(inputScaleVecVgpr), src1=vgpr("ValuC+%d"%vgprIdx), comment="*= %sVMulPK(%d)(%d)"%(addressStr, dataScaleAlphaVec,vi)))
              vecModule.add(VMulLOU32(dst=vgpr("ValuC+%d"%(vgprIdx+1)), src0=vgpr(inputScaleVecVgpr+1), src1=vgpr("ValuC+%d"%(vgprIdx+1)), comment="*= %sVMulPK(%d)(%d)"%(addressStr, dataScaleAlphaVec,vi)))
          else:
            raise RuntimeError("Unsupported %s compute data type %s."%(addressStr, str(self.kernel["ProblemType"]["ComputeDataType"])))

      isSingleKernel = ((self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1) or self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel") or self.kernel["StreamK"] > 0

      scaleAVecModule = Module("ScaleAVecModule")
      scaleBVecModule = Module("ScaleBVecModule")
      if (self.kernel["ProblemType"]["UseScaleAB"] == "Vector") and isSingleKernel:
        applyScaleVec(scaleAVecModule, "ScaleA", dataScaleAVec, 0, isGlobal=False)
        applyScaleVec(scaleBVecModule, "ScaleB", dataScaleBVec, 1, isGlobal=False)
      module.add(scaleAVecModule)
      module.add(scaleBVecModule)

      scaleAlphaVecModule = Module("scaleAlphaVecModule")
      if self.kernel["ProblemType"]["UseScaleAlphaVec"] and isSingleKernel:
        applyScaleVec(scaleAlphaVecModule, "ScaleAlphaVec", dataScaleAlphaVec, self.factorDim, isGlobal=False)
      module.add(scaleAlphaVecModule)

      if self.beta:
        module.add(self._addSumAlphaWithCBeta(self.kernel, self.ss, self.gwvw, elementIdx, vc0, self.tmpVgpr, self.cvtVgprStruct))
      elif ((self.parentWriter.states.useBias == DataDirection.READ) or self.kernel["ActivationFuncCall"]) and not self.applyAlpha \
        and not ( self.kernel["ProblemType"]["UseScaleAlphaVec"] and isSingleKernel): # case of alpha=1 and beta=0
        if (self.kernel["ProblemType"]["DestDataType"].isInt8() or self.kernel["ProblemType"]["DestDataType"].isInt32() or \
            (self.kernel["ProblemType"]["DataType"].isInt8() and self.kernel["ProblemType"]["DestDataType"].isHalf()) or \
            (self.kernel["ProblemType"]["DataType"].isInt8() and self.kernel["ProblemType"]["DestDataType"].isBFloat16())) and \
           self.kernel["ProblemType"]["ComputeDataType"].isSingle():
          module.add(convertData(self.gwvw, self.ss.elementSumIdx[elementIdx], cvtType=CvtType.CVT_I32_to_F32, \
                                      inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu))

      # Add bias
      mergeActFuncCall = False
      if self.parentWriter.states.useBias == DataDirection.READ:
        if activationCDataType == self.kernel["ProblemType"]["ComputeDataType"] and self.kernel["ActivationFuncCall"]:
          mergeActFuncCall = True
        if (self.kernel["ProblemType"]["Gradient"] and self.kernel["ProblemType"]["ActivationType"] != 'none' and self.kernel["ProblemType"]["UseE"]) and ((self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1) or self.kernel["StreamK"] > 0):
          mergeActFuncCall = False

        if self.factorDim and self.gwvw > 1:
          module.add(VMovB32(dst=vgpr(dataBias+1), src=vgpr(dataBias), comment="copy dataBias to dataBIas+1"))

        for vi in range(0, self.gwvw):
          inputVgpr = dataBias + + (0 if self.factorDim else vi)
          sumIdxV   = self.ss.elementSumIdx[elementIdx] + vi
          if self.kernel["ProblemType"]["ComputeDataType"].isSingle():
            vgprIdx = sumIdxV - self.parentWriter.states.c.startVgprValu
            vgprDst = (self.activationSetPCStruct.vgprActCopy + vi) if mergeActFuncCall else "ValuC+%d"%vgprIdx
            # Generate single f32 code if edge is detected.
            if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):
              module.add(VAddF32(dst=vgpr(vgprDst), src0=vgpr(inputVgpr), src1=vgpr("ValuC+%d"%vgprIdx), \
                                 comment="C += bias"))

            # Original packed route
            elif vi%2 == 1:
              assert (self.gwvw % 2 == 0)
            else:
              module.add(VAddPKF32(dst=vgpr(vgprDst, 2), src0=vgpr(inputVgpr, 2), \
                                   src1=vgpr("ValuC+%d"%vgprIdx, 2), comment="C += bias"))
          else:
            raise RuntimeError("Unsupported bias compute data type %s."%str(self.kernel["ProblemType"]["ComputeDataType"]))

      # Gate Residual: acc = gate*acc + gate. Built here but DEFERRED — added after
      # activation+scaleD (see module.add(gateModule) below), on the f32 ValuC.
      gateModule = Module("Empty GateResidual")
      if self.parentWriter.states.useGateResidual and \
         (self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1 or self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel"):
        # TODO: I8 input (int32 compute) — needs int32 acc -> cvt f32 -> FMA ->
        # saturate-cast. Not supported yet; reject below.
        if not self.kernel["ProblemType"]["ComputeDataType"].isSingle():
          raise RuntimeError(
              "GateResidual currently requires ComputeDataType=f32 (HPA); got %s" %
              str(self.kernel["ProblemType"]["ComputeDataType"]))

        gateList = list(self.kernel["ProblemType"].get("GateResidualDataTypeList", []))
        if not gateList:
          # Default fallback: use DestDataType as the gate dtype.
          gateList = [self.kernel["ProblemType"]["DestDataType"]]

        def _emit_gate_fma():
          """Identity (branchless null): ValuC = (gate+s)*ValuC + gate.
          GateNullOne (s) = 1.0 if gate null else 0.0 -> null:(0+1)*acc+0=acc; real:gate*acc+gate.
          gwvw>1 packs 2 elements/op (v_pk_add_f32 + v_pk_fma_f32). s is broadcast from its sgpr
          via op_sel_hi."""
          fmaMod = Module("GateFMA")
          assert self.tmpVgprSize >= 2, "gate packed FMA needs a 2-vgpr store-tmps scratch"
          tmpPk = self.tmpVgpr
          for vi in range(0, self.gwvw):
            sumIdxV = self.ss.elementSumIdx[elementIdx] + vi
            vgprIdx = sumIdxV - self.parentWriter.states.c.startVgprValu
            if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):
              # scalar (gwvw==1 or odd tail): sgpr GateNullOne broadcasts naturally.
              fmaMod.add(VAddF32(dst=vgpr(self.tmpVgpr), src0=vgpr(dataGate + vi),
                src1=sgpr("GateNullOne"), comment="gate + s (vi=%d)"%vi))
              fmaMod.add(VFmaF32(dst=vgpr("ValuC+%d"%vgprIdx), src0=vgpr(self.tmpVgpr),
                src1=vgpr("ValuC+%d"%vgprIdx), src2=vgpr(dataGate + vi),
                comment="GateResidual identity: acc = (gate+s)*acc + gate (vi=%d)"%vi))
            elif vi % 2 == 1:
              assert (self.gwvw % 2 == 0)
            else:
              # packed pair: op_sel_hi=[1,0,1] broadcasts GateNullOne (src1) low element to both lanes.
              fmaMod.add(VAddPKF32(dst=vgpr(tmpPk, 2), src0=vgpr(dataGate + vi, 2),
                src1=sgpr("GateNullOne", 2), vop3=VOP3PModifiers(op_sel_hi=[1,0,1]),
                comment="gate + s (vi=%d,%d)"%(vi, vi+1)))
              fmaMod.add(VFmaPKF32(dst=vgpr("ValuC+%d"%vgprIdx, 2), src0=vgpr(tmpPk, 2),
                src1=vgpr("ValuC+%d"%vgprIdx, 2), src2=vgpr(dataGate + vi, 2),
                comment="GateResidual identity packed: acc = (gate+s)*acc + gate (vi=%d,%d)"%(vi, vi+1)))
          return fmaMod

        if len(gateList) == 1:
          gateModule.add(self._emitGateCvt(dataGate, gateList[0]))
          gateModule.add(_emit_gate_fma())
        else:
          gateModule.add(_emit_gate_fma())

      if (self.kernel["ProblemType"]["UseE"] and not self.kernel["ProblemType"]["Gradient"]) and ((self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1) or self.kernel["StreamK"] > 0):
        vgprIdx   = self.ss.elementSumIdx[elementIdx] - self.parentWriter.states.c.startVgprValu
        vgprDst   = self.activationSetPCStruct.vgprActCopy if mergeActFuncCall else vgprIdx
        prefixStr = "" if mergeActFuncCall else "ValuC+"
        prefixOffset = 0 if mergeActFuncCall else self.parentWriter.states.c.startVgprValu
        # Packdata if needed
        tmpVgpr = self.tmpVgpr
        if mergeActFuncCall:
          tmpVgpr += self.gwvw * self.kernel["ProblemType"]["ComputeDataType"].numRegisters()
        if self.kernel["ProblemType"]["ComputeDataType"].isSingle():
          if self.kernel["ProblemType"]["DataTypeE"].isHalf():
            packdata = PackData_F16()
            module.add(packdata(self.gwvw, tmpVgpr, vgprDst, tmpVgpr=tmpVgpr, inputPrefix=prefixStr, prefixOffset=prefixOffset))
            vgprDst = tmpVgpr
          elif self.kernel["ProblemType"]["DataTypeE"].isBFloat16():
            packdata = PackData_BF16()
            module.add(packdata(self.gwvw, tmpVgpr, vgprDst, self.cvtVgprStruct, self.tmpS01, self.laneSGPRC,
                                tmpVgpr=tmpVgpr, inputPrefix=prefixStr, prefixOffset=prefixOffset))
            vgprDst = tmpVgpr
          elif self.kernel["ProblemType"]["DataTypeE"].isSingle():
            if not mergeActFuncCall:
              vgprDst = "ValuC+%d" % vgprDst
          elif self.kernel["ProblemType"]["DataTypeE"].isFloat8():
            packdata = PackData_FLOAT8()
            module.add(packdata(self.gwvw, tmpVgpr, vgprDst, self.cvtVgprStruct, self.tmpS01, self.laneSGPRC,
                                inputPrefix=prefixStr, prefixOffset=prefixOffset))
            vgprDst = tmpVgpr
          elif self.kernel["ProblemType"]["DataTypeE"].isFloat8_fnuz():
            packdata = PackData_FLOAT8_fnuz()
            module.add(packdata(self.gwvw, tmpVgpr, vgprDst, self.cvtVgprStruct, self.tmpS01, self.laneSGPRC,
                                inputPrefix=prefixStr, prefixOffset=prefixOffset))
            vgprDst = tmpVgpr
          else:
            printExit("Unsupport type for E output. (%s)"%self.kernel["ProblemType"]["DataTypeE"].toEnum())
        else:
          printExit("Unsupport compute type for E output. (%s)"%self.kernel["ProblemType"]["ComputeDataType"].toEnum())

        module.add(self.parentWriter.addStore(self.kernel, self.ss, 'E', addrCalc, vgprDst, self.tmpS01, self.edge, comment="store E"))

      SaturateTypeInt8 = SaturateCastType.NORMAL

      gradientCvtModule = Module("gradientCvtModule")
      if (self.kernel["ProblemType"]["UseE"] and self.kernel["ProblemType"]["Gradient"]) and ((self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1) or self.kernel["StreamK"] > 0):
        loadOffset = int((self.kernel["ProblemType"]["ComputeDataType"].numRegisters() - self.kernel["ProblemType"]["DataTypeE"].numRegisters()) * self.ss.cfg.gwvw)
        if activationCDataType != self.kernel["ProblemType"]["DataTypeE"]:
          if activationCDataType.isSingle() and self.kernel["ProblemType"]["DataTypeE"].isHalf():
            for vi in range(0, self.gwvw):
              dataEV  = dataE + vi
              dataEV2 = dataE + vi // 2
              selectbit = HighBitSel.LOW if (self.gwvw != 1 and vi % 2 == 0) or (self.gwvw == 1 and elementIdx % 2 == 0) else HighBitSel.HIGH
              gradientCvtModule.add(ECvtF16toF32(dst=vgpr(dataEV), src=vgpr(dataEV2+loadOffset), sel=selectbit, comment="gwvw %d, elementIdx %d"%(self.gwvw, elementIdx)))
          elif activationCDataType.isSingle() and self.kernel["ProblemType"]["DataTypeE"].isBFloat16():
            for vi in range(0, self.gwvw):
              dataEV  = dataE + vi
              dataEV2 = dataE + vi // 2
              # Consider bf16 without packing (gwvw==1)
              # TODO: check correctness for gfx950
              selectWord = 0 if (self.gwvw != 1 and vi % 2 == 0) or (self.gwvw == 1) else 1
              module.add(VCvtBF16toFP32(dst=vgpr(dataEV), src=vgpr(dataEV2+loadOffset), vgprMask=vgpr(self.cvtVgprStruct.vgprBf16Mask), vi=(selectWord), comment="gwvw %d, elementIdx %d"%(self.gwvw, elementIdx)))
          else:
            printExit("[Gradient input] Unsupported conversion.")

      # Activation
      activationModule = None
      isActivationInsertAfter = False
      if self.kernel["ProblemType"]["Gradient"] and ((self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1) or self.kernel["StreamK"] > 0):
        gradientInput = dataE
        enableValuC   = False
      else:
        gradientInput = self.ss.elementSumIdx[elementIdx]
        enableValuC   = True
        if self.kernel["LocalSplitU"] > 1:
          # When LSU > 1, the VGPRs are from LSU output.
          # the elementSumIdx has indicated the VGPRs from LSU.
          # Don't use the ValuC prefix here.
          enableValuC = False
        # gradientInput is an absolute vgpr index (= elementSumIdx). The
        # ActivationFuncCall/copyData and getActivationDestDataType consumers below
        # address it directly as an absolute vgpr, so keep it absolute here. Only
        # getActivationActivationComputeType() uses the "ValuC+N" relative namespace
        # (when enableValuC is set); that startVgprValu offset is applied at its
        # call site below.
      if self.kernel["ActivationFuncCall"]:
        if (activationCDataType == self.kernel["ProblemType"]["DestDataType"]) and \
          (activationCDataType != self.kernel["ProblemType"]["ComputeDataType"]) and ((self.kernel["ProblemType"]["UseScaleCD"] == False) or (self.kernel["ProblemType"]["UseScaleAlphaVec"] == False)):
          isActivationInsertAfter = True
        activationModule = Module("ActivationFuncCall")
        if (not mergeActFuncCall) and (not isActivationInsertAfter):
          activationModule.appendModule (copyData(activationCDataType, gradientInput, self.gwvw, \
            self.activationSetPCStruct.vgprActCopy))
        swappc = SSwapPCB64(dst=sgpr(self.activationSetPCStruct.sgprOffsetBack, 2), \
          src=sgpr(self.activationSetPCStruct.sgprOffsetActivation, 2))
        calleeLabelsByGwvw = getattr(self.activationSetPCStruct, "calleeLabelsByGwvw", None)
        if calleeLabelsByGwvw:
          calleeFuncs = list(calleeLabelsByGwvw.get(self.gwvw, ()))
          if calleeFuncs:
            swappc.calleeFuncs = calleeFuncs
        activationModule.add(swappc)
        activationModule.appendModule (copyData(activationCDataType, gradientInput, self.gwvw, \
          self.activationSetPCStruct.vgprActCopy, 1))
      elif self.parentWriter.insertActivationAfterPacked(self.kernel, self.activationTypeStr) and (self.kernel["ProblemType"]["UseScaleAlphaVec"] == False):
        isActivationInsertAfter = True
        activationModule = self.parentWriter.getActivationDestDataType(self.kernel, self.activation, \
          self.activationTypeStr, self.gwvw, gradientInput , gradientInput, self.tmpVgpr, self.tmpSgpr)
      else:
        satInt8 = False
        if self.kernel["ProblemType"]["DestDataType"].isInt8():
          if (self.activationTypeStr == 'abs') or (self.activationTypeStr == 'relu'):
            SaturateTypeInt8 = SaturateCastType.DO_NOTHING
            satInt8 = True
        # getActivationActivationComputeType emits on the "ValuC+N" relative
        # register namespace when enableValuC is set, so make its input index
        # relative to startVgprValu. The offset is scoped to this consumer only;
        # the absolute-vgpr consumers above are left untouched.
        actComputeInput = gradientInput
        if enableValuC:
          actComputeInput = gradientInput - self.parentWriter.states.c.startVgprValu
        activationModule = self.parentWriter.getActivationActivationComputeType(self.kernel, self.activation, \
          self.activationTypeStr, self.gwvw, actComputeInput, actComputeInput, self.tmpVgpr, self.tmpSgpr, satInt8, enableValuC)
      # Add C *= GradientAct
      if self.kernel["ProblemType"]["ActivationType"] != 'none' and self.kernel["ProblemType"]["Gradient"] and ((self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1) or self.kernel["StreamK"] > 0):
        if isActivationInsertAfter:
          assert 0, "Gradient does not support isActivationInsertAfter."
        for vi in range(0, self.gwvw):
          sumIdxV = self.ss.elementSumIdx[elementIdx] + vi
          dataEV  = dataE + vi
          if self.kernel["ProblemType"]["ComputeDataType"].isSingle():
            vgprIdx = sumIdxV - self.parentWriter.states.c.startVgprValu
            # Generate single f32 code if edge is detected.
            if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):
              activationModule.add(VMulF32(dst=vgpr("ValuC+%d"%vgprIdx), src0=vgpr("ValuC+%d"%vgprIdx), src1=vgpr(dataEV), comment="C *= GradAct"))
            # Original packed route
            elif vi%2 == 1:
              assert (self.gwvw % 2 == 0)
            else:
              activationModule.add(VMulPKF32(dst=vgpr("ValuC+%d"%vgprIdx, 2), src0=vgpr("ValuC+%d"%vgprIdx, 2), src1=vgpr(dataEV, 2), comment="C *= GradAct"))
          else:
            assert 0, "Unsupported gradient type"

      scaleDModule = Module("Empty scaleDModule")
      if self.kernel["ProblemType"]["UseScaleCD"] and ((self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1) or self.kernel["StreamK"] > 0):
        for vi in range(0, self.gwvw):
          sumIdxV = self.ss.elementSumIdx[elementIdx] + vi
          if self.kernel["ProblemType"]["ComputeDataType"].isSingle():
            vgprIdx = sumIdxV - self.parentWriter.states.c.startVgprValu
            # Generate single f32 code if edge is detected.
            if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):
              if self.kernel["ProblemType"]["OutputAmaxD"]:
                if self.edge:
                  activationModule.add(VCmpEQU32(dst=VCC(), src0="BufferOOB", src1=(vgpr(addrCalc.addrDVgpr)), comment =""))
                  activationModule.add(VCndMaskB32(dst=vgpr("AmaxOutB"), src0=vgpr("ValuC+%d"%vgprIdx), src1=0, src2=VCC(), comment="Check If OOB, put zero if OOB"))
                  activationModule.add(VMaxF32(dst=vgpr("AmaxOut"), src0=vgpr("AmaxOut"), src1=vgpr("AmaxOutB", isAbs=True), comment="absmax"))
                else:
                  activationModule.add(VMaxF32(dst=vgpr("AmaxOut"), src0=vgpr("AmaxOut"), src1=vgpr("ValuC+%d"%vgprIdx, isAbs=True), comment="absmax"))
              activationModule.add(VMulF32(dst=vgpr("ValuC+%d"%vgprIdx), src0=vgpr("ValuC+%d"%vgprIdx), src1=sgpr("ScaleD"), comment="result *= ScaleD"))
            # Original packed route
            elif vi%2 == 1:
              assert (self.gwvw % 2 == 0)
            else:
              activationModule.add(VMulPKF32(dst=vgpr("ValuC+%d"%vgprIdx, 2), src0=vgpr("ValuC+%d"%vgprIdx, 2), src1=sgpr("ScaleD", 2), vop3=VOP3PModifiers(op_sel_hi=[1,0,1]), comment="result *= ScaleD"))
          else:
            assert 0, "Unsupported scaleD type"


      # pack stores, beta and non-beta reach here:
      packModule = Module("Empty pack module")
      convertModule = Module("Empty convert module")
      if self.needsAccumToDestConversion:
        if self.kernel["ActivationFuncCall"] and activationCDataType == self.kernel["ProblemType"]["DestDataType"]:
          destIdx = self.activationSetPCStruct.vgprActCopy
        else:
          destIdx = self.ss.elementSumIdx[elementIdx]
        packTmpS01 = self._epilogScratchSgpr(self.laneSGPRC)
        if self.kernel["ProblemType"]["DestDataType"].isHalf():
          # For UseSubtileImpl non-edge: paired dwordx4 path handles packing in _emit16bitSubtilePairedStore.
          if not is16bitSubtile:
            packModule = self.packdata(self.gwvw, destIdx, self.ss.elementSumIdx[elementIdx], inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
        elif self.kernel["ProblemType"]["DestDataType"].isBFloat16():
          # For UseSubtileImpl non-edge: paired dwordx4 path handles packing in _emit16bitSubtilePairedStore.
          if not is16bitSubtile:
            packModule = self.packdata(self.gwvw, destIdx, self.ss.elementSumIdx[elementIdx], bf16CVTVgprStruct=self.cvtVgprStruct,
                                       tmpS01=packTmpS01, laneSGPRC=self.laneSGPRC, inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
        elif self.kernel["ProblemType"]["DestDataType"].isAnyFloat8():
          if self.kernel["ProblemType"]["StochasticRounding"]:
            packModule = self.packdata(self.gwvw, destIdx, self.ss.elementSumIdx[elementIdx], fp8CVTVgprStruct=self.cvtVgprStruct, \
                                       tmpS01=packTmpS01, laneSGPRC=self.laneSGPRC, vgprTmp=vgprRND, inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu, alphaScale=1.0)
          else:
            packModule = self.packdata(self.gwvw, destIdx, self.ss.elementSumIdx[elementIdx], fp8CVTVgprStruct=self.cvtVgprStruct, \
                                       tmpS01=packTmpS01, laneSGPRC=self.laneSGPRC, inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
        elif self.kernel["ProblemType"]["DestDataType"].isAnyBFloat8():
          # TODO: BF8 stochastic rounding is not yet supported here.
          #       VCvtSRF32toBF8 instruction exists but stochasticRoundingCvt() only emits VCvtSRF32toFP8.
          #       To support BF8 SR: add SR branch here, generalize stochasticRoundingCvt() to accept bf8CVTVgprStruct,
          #       and select VCvtSRF32toBF8 based on DestDataType.
          packModule = self.packdata(self.gwvw, destIdx, self.ss.elementSumIdx[elementIdx], bf8CVTVgprStruct=self.cvtVgprStruct, \
                                     tmpS01=packTmpS01, laneSGPRC=self.laneSGPRC, inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
        elif self.kernel["ProblemType"]["DestDataType"].isInt32():
          if self.kernel["ProblemType"]["ComputeDataType"].isSingle() and ((self.parentWriter.states.useBias == DataDirection.READ) or self.kernel["ActivationFuncCall"] or self.applyAlpha or self.beta):
            convertModule = convertData(self.gwvw, self.ss.elementSumIdx[elementIdx], cvtType=CvtType.CVT_F32_to_I32, \
                                        inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
        elif self.kernel["ProblemType"]["DestDataType"].isInt8():
          if self.kernel["ProblemType"]["ComputeDataType"].isSingle() and ((self.parentWriter.states.useBias == DataDirection.READ) or self.kernel["ActivationFuncCall"] or self.applyAlpha or self.beta):
            convertModule = convertData(self.gwvw, self.ss.elementSumIdx[elementIdx], cvtType=CvtType.CVT_F32_to_I32, roundType=RoundType.ROUND_TO_NEAREST_EVEN, \
                                        inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
          packModule = self.packdata(self.gwvw, destIdx, self.ss.elementSumIdx[elementIdx], self.cvtVgprStruct, self.tmpS01,
                                     SaturateTypeInt8=SaturateTypeInt8, inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
        self._epilogScratchFree(packTmpS01)

      if self.parentWriter.states.asmCaps["HasWMMA_V1"] and self.kernel["EnableMatrixInstruction"] and self.kernel["ProblemType"]["DestDataType"].isHalf() and (not self.kernel["ProblemType"]["HighPrecisionAccumulate"]):
        for vi in range(0, self.gwvw):
          sumIdxV = self.ss.elementSumIdx[elementIdx] + vi
          if vi%2 == 1:
            formatVgpr = formatting(sumIdxV, "ValuC+", self.parentWriter.states.c.startVgprValu)
            d = self.ss.elementSumIdx[elementIdx] + vi//2
            dVgpr = formatting(d, "ValuC+", self.parentWriter.states.c.startVgprValu)
            packModule.add(VPackF16toB32(dst=vgpr(dVgpr), src0=vgpr(formatting(sumIdxV-1, "ValuC+", self.parentWriter.states.c.startVgprValu)), src1=vgpr(formatVgpr), \
                          comment="Pack with neighbor"))

      if self.kernel["ExpertSchedulingMode"] > 0:
        packModule.add(SWaitAlu(va_vdst=0, comment="wait for writes to complete"))

      biasReductionModule = Module("biasReductionModule")
      if self.storeBiasD == 1:
        vgprIdx = self.ss.elementSumIdx[elementIdx] - self.parentWriter.states.c.startVgprValu
        biasReductionModule.add(self.parentWriter.addStore(self.kernel, self.ss, 'Bias', addrCalc, "ValuC+%d"%vgprIdx, self.tmpS01, self.edge, comment="store Bias"))

      if isActivationInsertAfter:
        if self.parentWriter.states.useGateResidual and \
           (self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1):
          raise RuntimeError(
              "GateResidual is not supported with isActivationInsertAfter "
              "(activation applied after pack on dest-dtype value).")
        module.add(convertModule)
        module.add(packModule)
        module.add(gradientCvtModule)
        module.add(activationModule)
      else:
        module.add(gradientCvtModule)
        module.add(activationModule)
        module.add(scaleDModule)
        module.add(biasReductionModule)
        module.add(gateModule)
        module.add(convertModule)
        module.add(packModule)

      if not self.kernel["StoreRemapVectorWidth"]:
        # 16bit UseSubtileImpl non-edge: emit paired dwordx4 stores combining sba=0
        # with sba=1 subtile data into one buffer_store_dwordx4.  Works for both
        # bf16 and fp16 HPA output types.
        #
        # UseSubtileImpl splits MIWaveTile[0] into two subtile groups:
        #   sba=0 owns even tt0 values (0, 2, 4, ...)
        #   sba=1 owns odd  tt0 values (1, 3, 5, ...)
        # The element list interleaves them as consecutive (even, odd) tt0 pairs:
        #   element 0: tt0=0 (sba=0)
        #   element 1: tt0=1 (sba=1)   <- pair with element 0
        #   element 2: tt0=2 (sba=0)   (if MIWaveTile[0]>2)
        #   ...
        # Pairing key: tt0 % 2 — even tt0 is sba=0, odd tt0 is sba=1.
        storeCodeModule = storeCode if self.kernel["GroupLoadStore"] else module
        # SourceSwap + row-major (UseInitialStridesCD): under SourceSwap a lane's 4
        # accumulators lie along N (consecutive N-cols) at a fixed M-row, and row-major D
        # has StrideD1J==1 so those 4 N-cols are contiguous in memory.  The 4 N-cols are
        # 4 SEPARATE store elements (vc1=0,1,2,3) sharing the same (d1, d0) — element[i]'s
        # sumIdx holds exactly the value for its (M-row, N-col).  Rather than the column-major
        # per-M-row b16 scatter (which is wrong under SourceSwap — the 4 vc registers are 4
        # different N-cols, not 4 M-rows of one lane), aggregate the 4 N-col elements' sumIdx
        # and emit one dwordx2 wide store along N.  This branch REPLACES the paired/orphan
        # M-pairing dispatch for the SourceSwap row-major case; column-major / non-SS is
        # untouched below.
        strideD0DispatchRef = self.parentWriter.strideRef('D', 0)
        ssRowMajorNWide = (is16bitSubtile and self.kernel["SourceSwap"]
                           and not self.parentWriter.isConstUnitStride(strideD0DispatchRef))
        if ssRowMajorNWide:
          d1 = element[0]
          d0 = element[1]
          vc1 = element[2]
          # Gather all N-col partners (same d1, d0) in this batch, ordered by vc1.
          ncolPartners = sorted(
            [ei for ei in range(len(self.batchElements))
             if self.batchElements[ei][0] == d1 and self.batchElements[ei][1] == d0],
            key=lambda ei: self.batchElements[ei][2])
          # Emit the wide store only once, on the highest-vc1 element (all epilogues done).
          isLastNCol = (elementIdx == ncolPartners[-1])
          if isLastNCol:
            blockIdxM = d0
            blockIdxN = d1
            skipLabel = self._emitSubtileOobGuard(storeCodeModule, blockIdxM, blockIdxN,
                                                  labelPrefix="subtile_skip_nwide")
            sumIdxList = [self.ss.elementSumIdx[ei] for ei in ncolPartners]
            # vc1 (N-col index within the 4-wide group) for each partner, in the same
            # order as sumIdxList.  The full group is vc1 = 0,1,2,3; a batch boundary can
            # split it so only a subset lands in this batch.
            ncolVc1List = [self.batchElements[ei][2] for ei in ncolPartners]
            coord0 = self.ss.elementCoord0[ncolPartners[0]]
            # coord1 (N-col base of the group) must come from the vc1=0 partner: when the
            # group is split, ncolPartners[0] may be vc1>0, so subtract its vc1 to recover
            # the group base (matches the full-group coord1, which is the vc1=0 element's).
            coord1 = self.ss.elementCoord1[ncolPartners[0]] - ncolVc1List[0]
            prefixOffset = self.parentWriter.states.c.startVgprValu
            # N-col group must be complete (4 partners = MIOutputVectorWidth).
            # Subtile batch sizes are always multiples of the N-col group width
            # (MIWaveTile[0] * vectorWidth1) because the tile's N dimension is
            # at least MatrixInstN=16, well above MIOutputVectorWidth=4.
            assert len(ncolPartners) == 4, \
              f"SS row-major N-wide store requires 4 N-col partners, got {len(ncolPartners)}"
            tmpStoreCode = self._emit16bitSubtileNWideStore(
              sumIdxList, prefixOffset, coord0, coord1,
              blockIdxM=blockIdxM, blockIdxN=blockIdxN)
            # Fused-A2A recv PUSH branch (Task 4): the N-wide recv store packs the same 4 N-col
            # sumIdx as the local store and writes one dwordx2 into recv[W,M,n_shard] (n_shard
            # contiguous), matching the local N-wide write shape.  _fusedA2APushBuilder's M-row
            # packing is the wrong axis under SourceSwap, so use the N-wide builder instead.
            self._addSubtileStore(storeCodeModule, blockIdxN, tmpStoreCode,
                                  self._fusedA2APushNWideBuilder(ncolPartners))
            if skipLabel is not None:
              storeCodeModule.add(skipLabel)
            self.storesIssued += 1
          # Non-last N-col elements: epilogue already applied above; no store here.
        elif is16bitSubtile:
          tt0 = element[1]  # d0: thread-tile index along M
          # Epilogue (bias/activation) is applied per-element in iteration order.
          # The paired store must be emitted AFTER both sba=0 and sba=1 elements have
          # had their epilogue applied, so we defer it to the sba=1 (odd tt0) iteration.
          if tt0 % 2 == 1:
            # sba=1 element (odd tt0): both sba=0 and sba=1 epilogues are done — emit paired store.
            # Find the sba=0 partner: the immediately preceding element with tt0-1.
            partnerElementIdx = elementIdx - 1
            partnerExists = (partnerElementIdx >= 0 and
                             self.batchElements[partnerElementIdx][1] == tt0 - 1)
            if partnerExists:
              # Paired dwordx4 store for (sba=0 at tt0-1, sba=1 at tt0).
              partnerAddrCalc: AddrCalculation = self.ss.elementAddr[partnerElementIdx]
              sumIdx0 = self.ss.elementSumIdx[partnerElementIdx]
              sumIdx1 = self.ss.elementSumIdx[elementIdx]
              prefixOffset = self.parentWriter.states.c.startVgprValu
              blockIdxN = element[0]
              # Guard with tt0-1 (lower block): skip if even the lower M-block is OOB.
              # This also handles N-group transitions.
              blockIdxM = tt0 - 1
              skipLabel = self._emitSubtileOobGuard(storeCodeModule, blockIdxM, blockIdxN,
                                                    labelPrefix="subtile_skip_store")
              # Additional check: paired store needs BOTH blocks valid (MGuard > tt0).
              # When only the lower block is valid, fall through to a scalar fallback.
              guardMSgpr = self.parentWriter.states.subtileM32ValidBlocksSgpr
              if guardMSgpr is not None:
                afterPairedLabel = Label(self.parentWriter.labels.getNameInc("subtile_after_paired"),
                                        f"after paired/fallback store tt0={tt0}")
                fallbackLabelName = self.parentWriter.labels.getNameInc("subtile_scalar_fallback")
                fallbackLabel = Label(fallbackLabelName,
                                      f"scalar fallback for d0={tt0-1} when d0={tt0} is OOB")
                storeCodeModule.add(_scmpGtU32(self.parentWriter, sgpr("SubtileMGuard"), tt0,
                                               comment=f"paired store: both M-blocks valid? (MGuard > {tt0})"))
                storeCodeModule.add(SCBranchSCC0(labelName=fallbackLabel.getLabelName(),
                                                 comment=f"only d0={tt0-1} valid -> scalar fallback"))
                tmpStoreCode = self._emit16bitSubtilePairedStore(partnerAddrCalc, sumIdx0, sumIdx1, prefixOffset, tt0 - 1, blockIdxM=blockIdxM, blockIdxN=blockIdxN)
                self._addSubtileStore(storeCodeModule, blockIdxN, tmpStoreCode,
                                      self._fusedA2APushBuilder([partnerElementIdx, elementIdx]))
                storeCodeModule.add(SBranch(labelName=afterPairedLabel.getLabelName(),
                                            comment="skip scalar fallback"))
                storeCodeModule.add(fallbackLabel)
                tmpFallbackCode = self._emit16bitSubtileScalarStore(partnerAddrCalc, sumIdx0, prefixOffset, tt0 - 1, blockIdxM=blockIdxM, blockIdxN=blockIdxN)
                self._addSubtileStore(storeCodeModule, blockIdxN, tmpFallbackCode,
                                      self._fusedA2APushBuilder([partnerElementIdx]))
                storeCodeModule.add(afterPairedLabel)
              else:
                tmpStoreCode = self._emit16bitSubtilePairedStore(partnerAddrCalc, sumIdx0, sumIdx1, prefixOffset, tt0 - 1, blockIdxM=blockIdxM, blockIdxN=blockIdxN)
                self._addSubtileStore(storeCodeModule, blockIdxN, tmpStoreCode,
                                      self._fusedA2APushBuilder([partnerElementIdx, elementIdx]))
              if skipLabel is not None:
                storeCodeModule.add(skipLabel)
              self.storesIssued += 1
            else:
              # sba=1 orphan (no sba=0 partner in this batch — split by batch boundary).
              blockIdxM = tt0
              blockIdxN = element[0]
              orphanSkipLabel = self._emitSubtileOobGuard(storeCodeModule, blockIdxM, blockIdxN,
                                                          labelPrefix="subtile_skip_orphan")
              sumIdx0 = self.ss.elementSumIdx[elementIdx]
              prefixOffset = self.parentWriter.states.c.startVgprValu
              tmpStoreCode = self._emit16bitSubtileScalarStore(addrCalc, sumIdx0, prefixOffset, tt0, blockIdxM=blockIdxM, blockIdxN=blockIdxN)
              self._addSubtileStore(storeCodeModule, blockIdxN, tmpStoreCode,
                                    self._fusedA2APushBuilder([elementIdx]))
              if orphanSkipLabel is not None:
                storeCodeModule.add(orphanSkipLabel)
              self.storesIssued += 1
          else:
            # sba=0 element (even tt0): defer SRD row increment until after N-group label.
            if self.ss.optSrdIncForRow and addrCalc.rowInc:
              self._subtilePendingSrdDInc = addrCalc.incrementToNextRow(self.kernel, "D", self.ss, self.tmpS01)
            partnerElementIdx = elementIdx + 1
            partnerExists = (partnerElementIdx < len(self.batchElements) and
                             self.batchElements[partnerElementIdx][1] == tt0 + 1)
            if not partnerExists:
              # Orphan element (no sba=1 partner in this batch): scalar 16bit store now.
              # Guard against OOB wave groups (same as paired store path).
              blockIdxM = tt0
              blockIdxN = element[0]
              # Early exit: skip this orphan scalar store if the wave group is outside the valid M/N tile bounds.
              orphanSkipLabel = self._emitSubtileOobGuard(storeCodeModule, blockIdxM, blockIdxN,
                                                          labelPrefix="subtile_skip_orphan")
              sumIdx0 = self.ss.elementSumIdx[elementIdx]
              prefixOffset = self.parentWriter.states.c.startVgprValu
              tmpStoreCode = self._emit16bitSubtileScalarStore(addrCalc, sumIdx0, prefixOffset, tt0, blockIdxM=blockIdxM, blockIdxN=blockIdxN)
              self._addSubtileStore(storeCodeModule, blockIdxN, tmpStoreCode,
                                    self._fusedA2APushBuilder([elementIdx]))
              if orphanSkipLabel is not None:
                storeCodeModule.add(orphanSkipLabel)
              self.storesIssued += 1
        elif self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":#GSUGSU
          tmpStoreCode = self.parentWriter.addStore(self.kernel, self.ss, 'TD', addrCalc, sumIdx, self.tmpS01, self.edge, comment="store TD not StoreRemapVectorWidth")
          storeCodeModule.add(tmpStoreCode)
          self.storesIssued += 1
        else:
          # Regular store path. If UseSubtileImpl NonEdge, guard against OOB wave groups.
          skipLabel = None
          if isSubtileNonEdge:
            tt0 = element[1]
            blockIdxM = tt0  # each tt0 maps to one mBlockSize-row block
            blockIdxN = element[0]
            # Early exit: skip this store if the wave group is outside the valid M/N tile bounds.
            skipLabel = self._emitSubtileOobGuard(storeCodeModule, blockIdxM, blockIdxN,
                                                  labelPrefix="subtile_skip_store")
          # Apply exec mask for partial M/N blocks (regular fp32 store path)
          if self.parentWriter.states.storeAlign8 and isSubtileNonEdge:
            # wave32: 2 LGs x 8 rows/LG -> shift=1; wave64: 4 LGs x 4 rows/LG -> shift=2
            rowScaleShift = 1 if self.wavelen == 32 else 2
            self._emitAlign8ExecMask(storeCodeModule, self.tmpS01, self.tmpS23, blockIdxM, blockIdxN,
                                     mGuardOffset=1, rowScaleShift=rowScaleShift)
            storeCodeModule.add(self.getEdgeMovInstType()(EXEC(), sgpr(self.tmpS01, self.laneSGPRC), "apply exec mask"))
          # _emitOverrideRows reused from the top of this store loop (see _lookaheadRowInc).
          # TODO Task 7: also route regular/TD store paths through _fusedA2ADispatch.
          tmpStoreCode = self.parentWriter.addStore(self.kernel, self.ss, 'D', addrCalc, sumIdx, self.tmpS01, self.edge, elementIdx, self.batchIdx,
                                                   overrideAfterPrimerRows=_emitOverrideRows, comment="store D")
          storeCodeModule.add(tmpStoreCode)
          if self.parentWriter.states.storeAlign8 and isSubtileNonEdge:
            storeCodeModule.add(self.getEdgeMovInstType()(EXEC(), -1, "restore exec"))
          if skipLabel is not None:
            storeCodeModule.add(skipLabel)
          self.storesIssued += 1

        if (self.kernel["ProblemType"]["UseE"] and not self.kernel["ProblemType"]["Gradient"]) and ((self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1) or self.kernel["StreamK"] > 0):
          self.storesIssued += 1
        if self.storeBiasD == 1:
          self.storesIssued += 1

      else:
        if not self.kernel["_GlobalAccumulation"] == "MultipleBufferSingleKernel":#GSUGSU
          rpe = self.parentWriter.states.bpeCinternal // self.parentWriter.states.bpr
          module.add(self.parentWriter.storeRemapAddLocalWrite(self.kernel, self.ss, addrCalc, sumIdx*rpe))
          # Column Block Shape has been written to LDS
          # Now read back and write out to global memory
        else:
          tmpStoreCode = self.parentWriter.addStore(self.kernel, self.ss, 'TD', addrCalc, sumIdx, self.tmpS01, self.edge, comment="store TD StoreRemapVectorWidth")

          if self.kernel["GroupLoadStore"]:
            storeCode.add(tmpStoreCode)

          module.add(tmpStoreCode)

          self.storesIssued += 1
          if (self.kernel["ProblemType"]["UseE"] and not self.kernel["ProblemType"]["Gradient"]) and ((self.kernel["GlobalSplitU"] == 1 or self.kernel["GlobalSplitU"] == -1) or self.kernel["StreamK"] > 0):
            self.storesIssued += 1
          if self.storeBiasD == 1:
            self.storesIssued += 1

    # Close the last N-group OOB skip label (if any) opened by _emitSubtileOobGuard.
    self._finalizeSubtileOobGuards(storeCode if self.kernel["GroupLoadStore"] else module)
    if self.kernel["ProblemType"]["StochasticRounding"]:
      self.parentWriter.vgprPool.checkIn(vgprRND)

    module.add(storeCode)

    if self.parentWriter.db["CheckStoreC"]>=0:
      useBuffer = self.kernel["BufferStore"]
      # Note - CheckStoreC won't work for EDGE store cases since they load 0 for OOB, would need more sophisticated check
      # Note - TODO- CheckStoreC also won't work for StoreRemap
      module.add(SWaitCnt(vscnt=0, comment="CheckStoreC, wait for stores to complete"))
      for elementIdx in range(0, len(self.batchElements)):
        addr = self.ss.elementAddr[elementIdx].addrDVgpr
        sumIdx = self.ss.elementSumIdx[elementIdx]

        bps = self.kernel["ProblemType"]["DestDataType"].numBytes() * self.gwvw
        if self.kernel["BufferStore"]:
          addr0 = vgpr(addr)
          addr1 = sgpr("SrdC", 4)
        else:
          addr0 = vgpr(addr,2)
          addr1 = ""

        if self.kernel["ProblemType"]["DestDataType"].isHalf() or self.kernel["ProblemType"]["DestDataType"].isBFloat16():
          if not self.kernel["ProblemType"]["HighPrecisionAccumulate"]:
            module.add(self.parentWriter.chooseGlobalRead(useBuffer, bps, sumIdx//2, \
                                  addr0, addr1, soffset=0, offset=0, hi16=sumIdx%2))
          else:
            module.add(self.parentWriter.chooseGlobalRead(useBuffer, bps, sumIdx, \
                                  addr0, addr1, soffset=0, offset=0, hi16=0))
        elif self.kernel["ProblemType"]["DestDataType"].isInt32() or self.kernel["ProblemType"]["DestDataType"].isSingle():
          module.add(self.parentWriter.chooseGlobalRead(useBuffer, bps, sumIdx, \
                                addr0, addr1, soffset=0, offset=0))
        elif self.kernel["ProblemType"]["DestDataType"].isDouble() or self.kernel["ProblemType"]["DestDataType"].isSingleComplex() :
          module.add(self.parentWriter.chooseGlobalRead(useBuffer, bps, sumIdx*2, \
                                addr0, addr1, soffset=0, offset=0))
        elif self.kernel["ProblemType"]["DestDataType"].isDoubleComplex():
          module.add(self.parentWriter.chooseGlobalRead(useBuffer, bps, sumIdx*4, \
                                addr0, addr1, soffset=0, offset=0))
      module.add(SWaitCnt(vscnt=0, comment="CheckStoreC, wait for stores to complete"))
      # Add checks for expected values:
      module.add(SMovB32(sgpr(self.tmpS01), self.parentWriter.db["CheckStoreC"], "expected value"))
      for elementIdx in range(0, len(self.batchElements)):
        sumIdx = self.ss.elementSumIdx[elementIdx]
        # Need to fix for other types:
        assert (self.kernel["ProblemType"]["DestDataType"].isSingle() or self.kernel["ProblemType"]["DestDataType"].isInt32())
        module.add(self.parentWriter.getCmpAssert(self.parentWriter.asmAssert.eq, vgpr(sumIdx), sgpr(self.tmpS01)))


    if self.edge and (self.atomic or not self.kernel["BufferStore"]):
      # subsequent batch must start with full exec mask
      # BufferStore doesn't need exec since it used buffer range checking when
      # possible
      module.add(self.getEdgeMovInstType()(EXEC(), -1, "full mask -> exec"))

    if self.parentWriter.db["ConservativeWaitCnt"] & 0x40:
      module.add(SBarrier(comment="debug"))
      module.add(SWaitCnt(vscnt=0, comment="ConservativeWaitCnt"))
      module.add(SBarrier(comment="debug"))

  def _emitSubtilePackedPermute(self, vPack: int, vPermAddr: int, addrWhilePermuting=None) -> Module:
    """Shuffle four packed dwords across wave halves for a subtile dwordx4 store.

    After the caller packs 8 f32 accumulator values into four 16bit dwords
    (vPack+0..+3), this routine performs the two-step permute that assembles
    eight consecutive M-rows owned by a pair of lane-groups into a contiguous
    dwordx4 payload:

      Step 1 — ds_bpermute (in-place, 4×): each lane fetches vPack+k from its
               partner lane l' (= the lane at LG±1 distance, pre-encoded as a
               byte address in vPermAddr).  The LDS pipe latches vPermAddr at
               issue time, so vPermAddr can be repurposed as soon as all four
               ds_bpermute instructions are issued.

      Step 2 — v_permlane32_swap_b32 (2×): exchange (vPack+0 ↔ vPack+2) and
               (vPack+1 ↔ vPack+3) across the 32-lane boundary so that lanes
               0-31 end up with rows LG*8+0..LG*8+7 in ascending order.

    The caller may supply an optional `addrWhilePermuting` callable that adds
    address-preparation instructions to the same module *between* the four
    ds_bpermute issues and the SWaitCnt.  This overlaps address arithmetic
    with the LDS round-trip latency at no extra cost.

    Args:
      vPack:              Base VGPR index of the four packed dwords (must be
                          2-aligned to satisfy dwordx4 store alignment).
      vPermAddr:          VGPR holding the partner-lane byte address (pre-computed
                          once per batch in the vgprPermAddr slot).
      addrWhilePermuting: Optional callable() that appends address instructions
                          to `module` while the ds_bpermute results are in-flight.

    Returns:
      Module containing ds_bpermute × 4, optional address code, SWaitCnt,
      and v_permlane32_swap_b32 × 2.  Leaves vPack+0..+3 holding the
      correctly ordered dwords ready for buffer_store_dwordx4.
    """
    module = Module("SubtilePackedPermute")

    module.addComment1("ds_bpermute in-place: gather packed dwords from partner lane-group")
    for k in range(4):
      module.add(DSBPermuteB32(dst=vgpr(vPack+k), src0=vgpr(vPermAddr), src1=vgpr(vPack+k),
                               comment=f"perm dword {k}"))

    if addrWhilePermuting is not None:
      addrWhilePermuting(module)

    module.add(SWaitCnt(dscnt=0, comment="wait for ds_bpermute (lgkmcnt=0)"))

    module.addComment1("v_permlane32_swap_b32: swap across lane-32 boundary")
    module.add(VPermlane32SwapB32(dst=vgpr(vPack+0), src=vgpr(vPack+2), comment="swap dwords 0↔2"))
    module.add(VPermlane32SwapB32(dst=vgpr(vPack+1), src=vgpr(vPack+3), comment="swap dwords 1↔3"))

    return module

  def _addSubtileStore(self, targetModule, blockIdxN: int, storeModule: Module, pushBuilder=None):
    """Add a 16bit subtile store, routing through the fused-A2A dispatch when enabled.

    When FusedGemmA2A is off the store Module is added verbatim (output byte-identical
    to the non-fused path); when on, a RUNTIME dispatch on WorkGroup1 (the owning N-tile)
    decides between the local D store (storeModule) and the remote all-to-all PUSH store
    (built by pushBuilder).

    pushBuilder: optional zero-arg callable returning a Module that stores this element
                 to the remote recv[W,M,n_shard] layout.  Required for the PUSH branch;
                 when None the local storeModule is used for both branches (skeleton).
    """
    if self.kernel["FusedGemmA2A"]:
      self._fusedA2ADispatch(targetModule, blockIdxN, storeModule, pushBuilder)
    else:
      targetModule.add(storeModule)

  def _fusedA2APushBuilder(self, elementIdxList):
    """Return a zero-arg callable building the PUSH store for the given store elements.

    elementIdxList: the element indices whose (sumIdx, coordOffset0, coordOffset1) form the
                    subtile M-blocks packed into this store (1 for scalar/orphan, 2 for paired).
    Returns None when FusedGemmA2A is off so callers can pass it unconditionally.
    """
    if not self.kernel["FusedGemmA2A"]:
      return None
    groups = [(self.ss.elementSumIdx[ei], self.ss.elementCoord0[ei], self.ss.elementCoord1[ei])
              for ei in elementIdxList]
    return lambda: self._emitFusedA2APushStore(groups)

  def _fusedA2ADispatch(self, targetModule, blockIdxN: int, storeModule: Module, pushBuilder=None):
    """Route a D-store by owning N-tile for FusedGemmA2A (RUNTIME decision).

    A workgroup owns one MacroTile1-wide N column-block, identified at RUNTIME by
    WorkGroup1 (the global N-tile index).  Workgroups whose N-tile is in the first AN
    columns (WorkGroup1 < AN_tiles) are the all-to-all "PUSH" workgroups: their output
    is redirected to the remote recv[W,M,n_shard] buffer (built by pushBuilder) instead
    of the local D output.  Workgroups at/above are locally owned and use the regular
    local store.

    AN_tiles is derived at RUNTIME from the FusedAN kernarg (the A2A column count):
    AN_tiles = FusedAN / MacroTile1.  MacroTile1 is the compile-time constant
    self.kernel["MacroTile1"] (256 for the champion config) and is a power of two, so
    the divide is a right shift by log2(MacroTile1).  The design guarantees AN is a
    whole number of macro-tiles (AN % 256 == 0), so the shift is exact.

    Both code paths are emitted and guarded by a runtime SCC compare/branch on
    WorkGroup1 vs AN_tiles, so each WG executes exactly one path at run time.
    (blockIdxN = element[0] is only the N wave-block WITHIN a macro-tile (0..15) and must
    NOT gate this decision; the owning N-tile is a per-WG runtime value in WorkGroup1.)

    When pushBuilder is None (dispatch-only skeleton) the PUSH branch falls back to the
    local store so GEMM numerics stay intact.
    """
    from .Signature import fusedA2AKernArgLayout
    kw = self.parentWriter
    mt1 = self.kernel["MacroTile1"]
    log2mt1 = int(log2(mt1))
    localLabel = Label(kw.labels.getNameInc("fusedA2A_dispatch_local"),
                       "fused-A2A: WorkGroup1 >= AN_tiles -> local store")
    afterLabel = Label(kw.labels.getNameInc("fusedA2A_dispatch_after"),
                       "fused-A2A: after PUSH/local dispatch")
    # Runtime gate: PUSH when WorkGroup1 < AN_tiles, else local store.
    # AN_tiles = FusedAN >> log2(MacroTile1); FusedAN is read on demand from the
    # fused kernarg segment (metadata-only, per the Task 5 contract).
    layout = fusedA2AKernArgLayout()
    fusedBase = kw.states.fusedA2AKernArgBase
    tmpS = kw.sgprPool.checkOut(1, tag="fusedA2A_dispatchGate", preventOverflow=False)
    targetModule.addComment0("fused-A2A dispatch: runtime gate WorkGroup1 < (FusedAN/MT1) ? PUSH : local")
    targetModule.add(kw.argLoader.loadKernArg(tmpS, "KernArgAddress",
      sgprOffset=hex(fusedBase + layout["FusedAN"]), dword=1))
    targetModule.add(SWaitCnt(kmcnt=0, comment="wait FusedAN"))
    targetModule.add(SLShiftRightB32(dst=sgpr(tmpS), shiftHex=log2mt1, src=sgpr(tmpS),
                                     comment=f"AN_tiles = FusedAN >> log2(MT1={mt1})"))
    targetModule.add(SCmpGtU32(src0=sgpr(tmpS), src1=sgpr("WorkGroup1"),
                               comment="AN_tiles > WorkGroup1? (this WG's N-tile in PUSH region)"))
    targetModule.add(SCBranchSCC0(labelName=localLabel.getLabelName(),
                                  comment="WorkGroup1 >= AN_tiles -> local store"))
    kw.sgprPool.checkIn(tmpS)
    # PUSH branch (WorkGroup1 < AN_tiles): remote all-to-all store.
    targetModule.addComment0("fused-A2A dispatch: PUSH branch (remote recv[W,M,n_shard])")
    if pushBuilder is not None:
      targetModule.add(pushBuilder())
    else:
      targetModule.add(storeModule)
    targetModule.add(SBranch(labelName=afterLabel.getLabelName(), comment="skip local store"))
    # Local branch (WorkGroup1 >= AN_tiles): regular local D store.
    targetModule.add(localLabel)
    targetModule.addComment0("fused-A2A dispatch: local branch (local D store)")
    targetModule.add(storeModule)
    targetModule.add(afterLabel)

  def _fusedA2ALoadRecvBase(self, module, recvBaseSgpr, shardBaseSgpr, nShardSgpr, tmpSgpr):
    """Switch-load recv_ptr[dst_rank] into recvBaseSgpr and dst_rank*n_shard into shardBaseSgpr.

    recv_ptr[dst_rank] is NOT in an SGPR (Task 5 registered it as kernarg metadata only),
    so per the Task 5 contract this reads exactly one recv base pointer on demand via a
    switch over the compile-time-constant kernarg slots.  dst_rank is a per-WG runtime
    constant: since n_shard is a multiple of MacroTile1 (design guarantees (AN/W)%256==0),
    every 256-column macro-tile lies entirely within one destination rank's shard, so
    dst_rank = (WorkGroup1 * MacroTile1) / n_shard is the same for all lanes in the WG.

    Rather than divide by the runtime n_shard, the switch scans candidate ranks j
    (0..MAX-1): rank j is the destination when this WG's first N column
    (n_col_base_wg = WorkGroup1*MT1) is >= j*n_shard.  Ranks are contiguous, so the
    HIGHEST matching j wins; the forward scan keeps overwriting recvBase/shardBase with
    each qualifying rank.  Ranks j >= FusedW never qualify (n_col_base_wg < AN = W*n_shard),
    so unused slots are harmless even though all MAX slots are visited at compile time.

    Args:
      module:        Module to append instructions to.
      recvBaseSgpr:  2-SGPR pair (aligned) to receive recv_ptr[dst_rank].
      shardBaseSgpr: 1 SGPR to receive dst_rank*n_shard (element units).
      nShardSgpr:    1 SGPR pre-loaded with FusedNShard (n_shard, element units).
      tmpSgpr:       2 scratch SGPRs; tmpSgpr+0 holds n_col_base_wg, tmpSgpr+1 shard_lo.
    """
    from .Signature import fusedA2AKernArgLayout, FUSED_A2A_MAX_RANKS
    layout = fusedA2AKernArgLayout()
    fusedBase = self.parentWriter.states.fusedA2AKernArgBase

    # n_col_base_wg = WorkGroup1 * MacroTile1  (this WG's first output N column).
    module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr("WorkGroup1"), src1=self.kernel["MacroTile1"],
                       comment="n_col_base_wg = WorkGroup1 * MT1"))

    # Rank 0 default: recv base = recv_ptr_0, shard_base = 0 (winner unless a higher rank matches).
    module.add(self.parentWriter.argLoader.loadKernArg(recvBaseSgpr, "KernArgAddress",
      sgprOffset=hex(fusedBase + layout["recv_ptr_0"]), dword=2))
    module.add(SMovB32(dst=sgpr(shardBaseSgpr), src=0, comment="shard_base = 0 (rank 0)"))

    for j in range(1, FUSED_A2A_MAX_RANKS):
      skipLabel = Label(self.parentWriter.labels.getNameInc("fusedA2A_recv_skip%u" % j),
                        f"n_col_base_wg below rank {j}")
      # shard_lo = j * n_shard; if n_col_base_wg < shard_lo this rank (and all higher) lose.
      module.add(SMulI32(dst=sgpr(tmpSgpr + 1), src0=sgpr(nShardSgpr), src1=j,
                         comment=f"cand shard_lo = {j} * n_shard"))
      module.add(SCmpGtU32(src0=sgpr(tmpSgpr + 1), src1=sgpr(tmpSgpr),
                           comment=f"shard_lo > n_col_base_wg? (WG below rank {j})"))
      module.add(SCBranchSCC1(labelName=skipLabel.getLabelName(),
                              comment=f"below rank {j}: keep current winner"))
      # Rank j qualifies (highest so far): overwrite winner recv base + shard_base.
      module.add(self.parentWriter.argLoader.loadKernArg(recvBaseSgpr, "KernArgAddress",
        sgprOffset=hex(fusedBase + layout["recv_ptr_%u" % j]), dword=2))
      module.add(SMovB32(dst=sgpr(shardBaseSgpr), src=sgpr(tmpSgpr + 1),
                         comment=f"shard_base = {j} * n_shard"))
      module.add(skipLabel)
    module.add(SWaitCnt(kmcnt=0, comment="wait recv_ptr[dst_rank] load"))

  # Number of consecutive M-rows a lane owns per subtile output group (MFMA vector width).
  _FUSED_A2A_MROWS = 4

  def _emitFusedA2APushStore(self, groups) -> Module:
    """Emit the remote all-to-all PUSH store for one subtile store element.

    Redirects the PUSH-column output from the local D buffer (column-major [M, GN], M
    contiguous) to the remote recv buffer laid out row-major [W, M, n_shard] (n_shard
    contiguous) -- the byte-identical layout the baseline all_to_all produces on the
    destination rank (design spec section 1 / section 3.3 repack rule
    out[:, :AN].view(M, W, n_shard).transpose -> [W, M, n_shard]).

    recv element offset (elements, before *bpe), on destination card recv_ptr[dst_rank]:
        my_rank * (M * n_shard)      slot for the source rank (recv[my_rank])
      + m_global * n_shard           row within the slot
      + n_local                      column within the shard, n_local = n_global - dst_rank*n_shard
    where dst_rank = n_global // n_shard (per-WG constant), m/n_global are per-lane.

    Because recv is transposed relative to D (M is the slow/strided dim here, not the fast
    dim), the lane's 4 consecutive M-rows are n_shard*bpe apart -- so this emits one
    buffer_store_b16 per M-row (a scatter) rather than the contiguous dwordx2 the local
    store uses.

    Device scope only (plain buffer_store): this validates the L2 layout.  The
    system-scope fence + counter/flag handshake are deferred to Task 8.

    Args:
      groups: list of (sumIdx, coordOffset0, coordOffset1) tuples, one per subtile M-block
              packed into this store (1 for a scalar/orphan element, 2 for a paired element).
              coordOffset0/1 are the compile-time element M/N offsets from coord0/coord1.
    """
    kw = self.parentWriter
    module = Module("fusedA2APushStore")
    isFp16 = self.kernel["ProblemType"]["DestDataType"].isHalf()
    VCvtPkF32to16 = VCvtPkF32toFP16 if isFp16 else VCvtPkF32toBF16
    typeStr = "fp16" if isFp16 else "bf16"
    bpe = kw.states.bpeCexternalGSU1  # 16bit dest == 2
    prefixOffset = kw.states.c.startVgprValu

    ntd = self.kernel["NonTemporalD"]
    isGlc = bool(ntd & 0x1)
    isSlc = bool(ntd & 0x2)
    isNT  = bool(ntd & 0x4)

    module.addComment1(f"fused-A2A PUSH store -> recv[W,M,n_shard] ({typeStr}, device scope)")

    # --- kernarg reads (on demand, per Task 5 contract): my_rank + n_shard ---
    myRankSgpr = kw.sgprPool.checkOut(1, tag="fusedA2A_myRank", preventOverflow=False)
    nShardSgpr = kw.sgprPool.checkOut(1, tag="fusedA2A_nShard", preventOverflow=False)
    from .Signature import fusedA2AKernArgLayout
    layout = fusedA2AKernArgLayout()
    fusedBase = kw.states.fusedA2AKernArgBase
    module.add(kw.argLoader.loadKernArg(myRankSgpr, "KernArgAddress",
      sgprOffset=hex(fusedBase + layout["FusedMyRank"]), dword=1))
    module.add(kw.argLoader.loadKernArg(nShardSgpr, "KernArgAddress",
      sgprOffset=hex(fusedBase + layout["FusedNShard"]), dword=1))
    module.add(SWaitCnt(kmcnt=0, comment="wait FusedMyRank / FusedNShard"))

    # --- switch-load recv_ptr[dst_rank] + shard_base (dst_rank*n_shard) ---
    recvSrd = kw.sgprPool.checkOutAligned(4, 4, tag="fusedA2A_recvSrd", preventOverflow=False)
    shardBaseSgpr = kw.sgprPool.checkOut(1, tag="fusedA2A_shardBase", preventOverflow=False)
    tmpSgpr2 = kw.sgprPool.checkOut(2, tag="fusedA2A_recvSwitchTmp", preventOverflow=False)
    # recv base pointer lands in recvSrd+0/+1 (SRD base words).
    self._fusedA2ALoadRecvBase(module, recvSrd, shardBaseSgpr, nShardSgpr, tmpSgpr2)

    # slotElem = my_rank * M * n_shard  (recv[my_rank] slot base, in elements).
    slotSgpr = kw.sgprPool.checkOut(1, tag="fusedA2A_slot", preventOverflow=False)
    module.add(SMulI32(dst=sgpr(slotSgpr), src0=sgpr(myRankSgpr), src1=sgpr("SizeI"),
                       comment="my_rank * M"))
    module.add(SMulI32(dst=sgpr(slotSgpr), src0=sgpr(slotSgpr), src1=sgpr(nShardSgpr),
                       comment="slotElem = my_rank * M * n_shard"))

    # Finish the recv SRD: limit word (BufferOOB) + config word (Srd127_96).
    module.add(SMovB32(dst=sgpr(recvSrd + 2), src="BufferOOB", comment="recv SRD num_records"))
    module.add(SMovB32(dst=sgpr(recvSrd + 3), src="Srd127_96", comment="recv SRD config"))

    # --- per-lane address VGPRs ---
    # nShardBpe (per M-row byte step) into a VGPR for the scatter.
    vRecvAddr  = kw.vgprPool.checkOut(1, tag="fusedA2A_recvAddr")
    vNLocal    = kw.vgprPool.checkOut(1, tag="fusedA2A_nLocal")
    vTmp       = kw.vgprPool.checkOut(1, tag="fusedA2A_tmp")
    vPack      = kw.vgprPool.checkOut(2, tag="fusedA2A_pack")  # 2 dwords = 4 16bit values
    vNShardBpe = kw.vgprPool.checkOut(1, tag="fusedA2A_nShardBpe")

    coord0 = kw.vgprs.coord0
    coord1 = kw.vgprs.coord1

    # nShardBpe = n_shard * bpe  (byte stride between adjacent M-rows in recv).
    module.add(SMulI32(dst=sgpr(tmpSgpr2), src0=sgpr(nShardSgpr), src1=bpe,
                       comment="n_shard * bpe (M-row byte stride)"))
    module.add(VMovB32(dst=vgpr(vNShardBpe), src=sgpr(tmpSgpr2), comment="n_shard*bpe -> vgpr"))

    def vc(sumIdx, vi):
      idx = sumIdx + vi - prefixOffset
      return vgpr("ValuC+" + str(idx))

    for (sumIdx, coordOffset0, coordOffset1) in groups:
      module.addComment1(f"PUSH group coordOff0={coordOffset0} coordOff1={coordOffset1}: pack 4 M-rows, scatter b16")
      # Pack the 4 f32 accumulators (this lane's 4 M-rows at fixed N-col) into 2 dwords.
      module.add(VCvtPkF32to16(dst=vgpr(vPack + 0), src0=vc(sumIdx, 0), src1=vc(sumIdx, 1),
                               comment=f"M-row+0/+1 -> {typeStr}"))
      module.add(VCvtPkF32to16(dst=vgpr(vPack + 1), src0=vc(sumIdx, 2), src1=vc(sumIdx, 3),
                               comment=f"M-row+2/+3 -> {typeStr}"))
      module.add(SNop(waitState=0, comment=f"delay after pk_{typeStr}"))

      # n_local = coord1 + coordOffset1 - shard_base   (per-lane N within the shard).
      # coordOffset0/1 are compile-time ints; move to an SGPR first because v_add_u32
      # rejects large inline literals on gfx950 (tmpSgpr2 is free after the recv switch).
      if coordOffset1:
        module.add(SMovB32(dst=sgpr(tmpSgpr2), src=coordOffset1, comment=f"coordOffset1={coordOffset1}"))
        module.add(VAddU32(dst=vgpr(vNLocal), src0=vgpr(coord1), src1=sgpr(tmpSgpr2),
                           comment=f"n_global = coord1 + {coordOffset1}"))
        module.add(VSubU32(dst=vgpr(vNLocal), src0=vgpr(vNLocal), src1=sgpr(shardBaseSgpr),
                           comment="n_local = n_global - shard_base"))
      else:
        module.add(VSubU32(dst=vgpr(vNLocal), src0=vgpr(coord1), src1=sgpr(shardBaseSgpr),
                           comment="n_local = coord1 - shard_base"))

      # m_base = coord0 + coordOffset0  (per-lane global M row of the group's first value).
      # recvElemBase = (slotElem + m_base * n_shard) + n_local
      #             --> byte = recvElemBase * bpe
      if coordOffset0:
        module.add(SMovB32(dst=sgpr(tmpSgpr2), src=coordOffset0, comment=f"coordOffset0={coordOffset0}"))
        module.add(VAddU32(dst=vgpr(vTmp), src0=vgpr(coord0), src1=sgpr(tmpSgpr2),
                           comment=f"m_base = coord0 + {coordOffset0}"))
        module.add(VMulLOU32(dst=vgpr(vTmp), src0=vgpr(vTmp), src1=sgpr(nShardSgpr),
                             comment="m_base * n_shard"))
      else:
        module.add(VMulLOU32(dst=vgpr(vTmp), src0=vgpr(coord0), src1=sgpr(nShardSgpr),
                             comment="m_base * n_shard"))
      module.add(VAddU32(dst=vgpr(vTmp), src0=vgpr(vTmp), src1=sgpr(slotSgpr),
                         comment="+ slotElem (my_rank*M*n_shard)"))
      module.add(VAddU32(dst=vgpr(vRecvAddr), src0=vgpr(vTmp), src1=vgpr(vNLocal),
                         comment="recvElemBase = slot + m_base*n_shard + n_local"))
      module.add(VLShiftLeftB32(dst=vgpr(vRecvAddr), shiftHex=int(log2(bpe)), src=vgpr(vRecvAddr),
                                comment="recv byte offset = recvElemBase * bpe"))

      # Scatter: one buffer_store_b16 per M-row (rows are n_shard*bpe apart in recv).
      for row in range(self._FUSED_A2A_MROWS):
        # 16bit value for this row: lo/hi of vPack dword row//2.
        srcVgpr = vPack + (row // 2)
        # Extract the correct 16bit half into vTmp when row is odd (hi half).
        if row % 2 == 1:
          module.add(VLShiftRightB32(dst=vgpr(vTmp), shiftHex=16, src=vgpr(srcVgpr),
                                     comment=f"M-row+{row}: hi 16bit -> lo"))
          storeSrc = vTmp
        else:
          storeSrc = srcVgpr
        module.add(BufferStoreB16(
          src=vgpr(storeSrc),
          vaddr=vgpr(vRecvAddr),
          saddr=sgpr(recvSrd, 4),
          soffset=0,
          mubuf=MUBUFModifiers(offen=True, offset12=0, glc=isGlc, slc=isSlc, nt=isNT),
          comment=f"recv scatter M-row+{row} (device scope)"))
        if row != self._FUSED_A2A_MROWS - 1:
          module.add(VAddU32(dst=vgpr(vRecvAddr), src0=vgpr(vRecvAddr), src1=vgpr(vNShardBpe),
                             comment="advance recv addr by n_shard*bpe (next M-row)"))

    module.add(SNop(waitState=0, comment="WAR: latch store src before next batch overwrites pack"))

    # Release temporaries.
    kw.vgprPool.checkIn(vNShardBpe)
    kw.vgprPool.checkIn(vPack)
    kw.vgprPool.checkIn(vTmp)
    kw.vgprPool.checkIn(vNLocal)
    kw.vgprPool.checkIn(vRecvAddr)
    kw.sgprPool.checkIn(slotSgpr)
    kw.sgprPool.checkIn(tmpSgpr2)
    kw.sgprPool.checkIn(shardBaseSgpr)
    kw.sgprPool.checkIn(recvSrd)
    kw.sgprPool.checkIn(nShardSgpr)
    kw.sgprPool.checkIn(myRankSgpr)
    return module

  def _fusedA2APushNWideBuilder(self, ncolPartners):
    """Return a zero-arg callable building the N-wide recv PUSH store for a 4-N-col group.

    Sibling of _fusedA2APushBuilder for the SourceSwap row-major layout: instead of packing
    an element's sumIdx+0..3 as 4 M-rows (the wrong axis under SourceSwap), it takes the 4
    N-col store elements aggregated in the dispatch (vc1=0..3 sharing one (d1,d0)) and packs
    their sumIdx as 4 consecutive N-cols.  recv[W,M,n_shard] has n_shard contiguous, so the 4
    N-cols land contiguously -> one buffer_store_dwordx2 (matching the local N-wide store), not
    a per-M-row b16 scatter.

    ncolPartners: the 4 element indices for the N-col group, in vc1-ascending order.
    Returns None when FusedGemmA2A is off so callers can pass it unconditionally.
    """
    if not self.kernel["FusedGemmA2A"]:
      return None
    sumIdxList = [self.ss.elementSumIdx[ei] for ei in ncolPartners]
    coord0 = self.ss.elementCoord0[ncolPartners[0]]
    # coord1 of the vc1=0 group base: elementCoord1 already folds in vc1 (AsmStoreState
    # coordOffset1 = base + vc1), so subtract the first partner's vc1 to recover the base
    # when a batch boundary makes ncolPartners[0] start at vc1>0.  Mirrors the dispatch's
    # coord1 recovery for _emit16bitSubtileNWideStore.
    coord1Base = self.ss.elementCoord1[ncolPartners[0]] - self.batchElements[ncolPartners[0]][2]
    return lambda: self._emitFusedA2APushNWideStore(sumIdxList, coord0, coord1Base)

  def _emitFusedA2APushNWideStore(self, sumIdxList, coordOffset0, coordOffset1) -> Module:
    """Emit the N-wide recv all-to-all PUSH store for one SourceSwap row-major N-col group.

    N-wide sibling of _emitFusedA2APushStore.  Under SourceSwap a lane's 4 accumulators lie
    along N (4 consecutive N-cols at a fixed M-row), so the 4 values passed here are 4 N-cols
    (vc1=0..3), NOT 4 M-rows.  recv is laid out row-major [W, M, n_shard] with n_shard
    contiguous, so the 4 N-cols land contiguously in recv -> one buffer_store_dwordx2, the same
    write shape the local N-wide store uses (Task 3), instead of the per-M-row b16 scatter the
    column-major _emitFusedA2APushStore emits.

    recv element offset (elements, before *bpe), on destination card recv_ptr[dst_rank]:
        my_rank * (M * n_shard)      slot for the source rank (recv[my_rank])
      + m_base   * n_shard           M-row within the slot (fixed for all 4 N-cols)
      + n_local                      first N-col within the shard, n_local = n_global - shard_base
    with the 4 N-cols at n_local+0..3.  The kernarg reads, recv-SRD switch-load, slotElem, and
    n_local/m_base math are byte-identical to _emitFusedA2APushStore (they depend only on
    coord0/coord1, not the acc axis); only the pack + store shape differs.

    Args:
      sumIdxList:   4 elementSumIdx values in N-col order (vc1=0..3).
      coordOffset0: compile-time M offset of the group (elementCoord0 of the vc1=0 element).
      coordOffset1: compile-time N-col base of the group (elementCoord1 with vc1 removed).
    """
    kw = self.parentWriter
    module = Module("fusedA2APushNWideStore")
    isFp16 = self.kernel["ProblemType"]["DestDataType"].isHalf()
    VCvtPkF32to16 = VCvtPkF32toFP16 if isFp16 else VCvtPkF32toBF16
    typeStr = "fp16" if isFp16 else "bf16"
    bpe = kw.states.bpeCexternalGSU1  # 16bit dest == 2
    prefixOffset = kw.states.c.startVgprValu

    ntd = self.kernel["NonTemporalD"]
    isGlc = bool(ntd & 0x1)
    isSlc = bool(ntd & 0x2)
    isNT  = bool(ntd & 0x4)

    assert len(sumIdxList) == 4, \
      f"N-wide recv PUSH store requires 4 N-col sumIdx, got {len(sumIdxList)}"

    module.addComment1(f"fused-A2A PUSH store -> recv[W,M,n_shard] N-wide ({typeStr}, device scope)")

    # --- kernarg reads (on demand, per Task 5 contract): my_rank + n_shard ---
    myRankSgpr = kw.sgprPool.checkOut(1, tag="fusedA2A_myRank", preventOverflow=False)
    nShardSgpr = kw.sgprPool.checkOut(1, tag="fusedA2A_nShard", preventOverflow=False)
    from .Signature import fusedA2AKernArgLayout
    layout = fusedA2AKernArgLayout()
    fusedBase = kw.states.fusedA2AKernArgBase
    module.add(kw.argLoader.loadKernArg(myRankSgpr, "KernArgAddress",
      sgprOffset=hex(fusedBase + layout["FusedMyRank"]), dword=1))
    module.add(kw.argLoader.loadKernArg(nShardSgpr, "KernArgAddress",
      sgprOffset=hex(fusedBase + layout["FusedNShard"]), dword=1))
    module.add(SWaitCnt(kmcnt=0, comment="wait FusedMyRank / FusedNShard"))

    # --- switch-load recv_ptr[dst_rank] + shard_base (dst_rank*n_shard) ---
    recvSrd = kw.sgprPool.checkOutAligned(4, 4, tag="fusedA2A_recvSrd", preventOverflow=False)
    shardBaseSgpr = kw.sgprPool.checkOut(1, tag="fusedA2A_shardBase", preventOverflow=False)
    tmpSgpr2 = kw.sgprPool.checkOut(2, tag="fusedA2A_recvSwitchTmp", preventOverflow=False)
    self._fusedA2ALoadRecvBase(module, recvSrd, shardBaseSgpr, nShardSgpr, tmpSgpr2)

    # slotElem = my_rank * M * n_shard  (recv[my_rank] slot base, in elements).
    slotSgpr = kw.sgprPool.checkOut(1, tag="fusedA2A_slot", preventOverflow=False)
    module.add(SMulI32(dst=sgpr(slotSgpr), src0=sgpr(myRankSgpr), src1=sgpr("SizeI"),
                       comment="my_rank * M"))
    module.add(SMulI32(dst=sgpr(slotSgpr), src0=sgpr(slotSgpr), src1=sgpr(nShardSgpr),
                       comment="slotElem = my_rank * M * n_shard"))

    # Finish the recv SRD: limit word (BufferOOB) + config word (Srd127_96).
    module.add(SMovB32(dst=sgpr(recvSrd + 2), src="BufferOOB", comment="recv SRD num_records"))
    module.add(SMovB32(dst=sgpr(recvSrd + 3), src="Srd127_96", comment="recv SRD config"))

    # --- per-lane address VGPRs ---
    vRecvAddr = kw.vgprPool.checkOut(1, tag="fusedA2A_recvAddr")
    vNLocal   = kw.vgprPool.checkOut(1, tag="fusedA2A_nLocal")
    vTmp      = kw.vgprPool.checkOut(1, tag="fusedA2A_tmp")
    vPack     = kw.vgprPool.checkOutAligned(2, 2, tag="fusedA2A_pack")  # 2 dwords, 64-bit aligned for dwordx2

    coord0 = kw.vgprs.coord0
    coord1 = kw.vgprs.coord1

    def vc(sumIdx):
      idx = sumIdx - prefixOffset
      return vgpr("ValuC+" + str(idx))

    module.addComment1(f"PUSH N-wide coordOff0={coordOffset0} coordOff1={coordOffset1}: pack 4 N-cols, dwordx2")
    # Pack the 4 N-col accumulators into 2 dwords (N-col 0,1 then 2,3) -- same order as the
    # local N-wide store, so recv contiguity (n_shard) matches D contiguity (StrideD1J==1).
    module.add(VCvtPkF32to16(dst=vgpr(vPack + 0), src0=vc(sumIdxList[0]), src1=vc(sumIdxList[1]),
                             comment=f"N-col 0,1 -> {typeStr}"))
    module.add(VCvtPkF32to16(dst=vgpr(vPack + 1), src0=vc(sumIdxList[2]), src1=vc(sumIdxList[3]),
                             comment=f"N-col 2,3 -> {typeStr}"))
    module.add(SNop(waitState=0, comment=f"delay after pk_{typeStr}"))

    # n_local = coord1 + coordOffset1 - shard_base  (first N-col within the shard; +1,+2,+3
    # contiguous in recv follow from n_shard being the fast dim).
    if coordOffset1:
      module.add(SMovB32(dst=sgpr(tmpSgpr2), src=coordOffset1, comment=f"coordOffset1={coordOffset1}"))
      module.add(VAddU32(dst=vgpr(vNLocal), src0=vgpr(coord1), src1=sgpr(tmpSgpr2),
                         comment=f"n_global = coord1 + {coordOffset1}"))
      module.add(VSubU32(dst=vgpr(vNLocal), src0=vgpr(vNLocal), src1=sgpr(shardBaseSgpr),
                         comment="n_local = n_global - shard_base"))
    else:
      module.add(VSubU32(dst=vgpr(vNLocal), src0=vgpr(coord1), src1=sgpr(shardBaseSgpr),
                         comment="n_local = coord1 - shard_base"))

    # m_base = coord0 + coordOffset0 (fixed M-row for all 4 N-cols under SourceSwap).
    # recvElemBase = slotElem + m_base * n_shard + n_local  -->  byte = recvElemBase * bpe
    if coordOffset0:
      module.add(SMovB32(dst=sgpr(tmpSgpr2), src=coordOffset0, comment=f"coordOffset0={coordOffset0}"))
      module.add(VAddU32(dst=vgpr(vTmp), src0=vgpr(coord0), src1=sgpr(tmpSgpr2),
                         comment=f"m_base = coord0 + {coordOffset0}"))
      module.add(VMulLOU32(dst=vgpr(vTmp), src0=vgpr(vTmp), src1=sgpr(nShardSgpr),
                           comment="m_base * n_shard"))
    else:
      module.add(VMulLOU32(dst=vgpr(vTmp), src0=vgpr(coord0), src1=sgpr(nShardSgpr),
                           comment="m_base * n_shard"))
    module.add(VAddU32(dst=vgpr(vTmp), src0=vgpr(vTmp), src1=sgpr(slotSgpr),
                       comment="+ slotElem (my_rank*M*n_shard)"))
    module.add(VAddU32(dst=vgpr(vRecvAddr), src0=vgpr(vTmp), src1=vgpr(vNLocal),
                       comment="recvElemBase = slot + m_base*n_shard + n_local"))
    module.add(VLShiftLeftB32(dst=vgpr(vRecvAddr), shiftHex=int(log2(bpe)), src=vgpr(vRecvAddr),
                              comment="recv byte offset = recvElemBase * bpe"))

    # 4 N-cols are contiguous in recv (n_shard fast dim) -> one dwordx2 store.
    module.addComment1(f"buffer_store_dwordx2: write 4 {typeStr} N-cols at fixed M-row (recv N-wide)")
    module.add(BufferStoreB64(
      src=vgpr(vPack + 0, 2),
      vaddr=vgpr(vRecvAddr),
      saddr=sgpr(recvSrd, 4),
      soffset=0,
      mubuf=MUBUFModifiers(offen=True, offset12=0, glc=isGlc, slc=isSlc, nt=isNT),
      comment="recv N-wide store: 4 consecutive N-cols at fixed M-row (device scope)"))

    module.add(SNop(waitState=0, comment="WAR: latch store src before next batch overwrites pack"))

    # Release temporaries.
    kw.vgprPool.checkIn(vPack)
    kw.vgprPool.checkIn(vTmp)
    kw.vgprPool.checkIn(vNLocal)
    kw.vgprPool.checkIn(vRecvAddr)
    kw.sgprPool.checkIn(slotSgpr)
    kw.sgprPool.checkIn(tmpSgpr2)
    kw.sgprPool.checkIn(shardBaseSgpr)
    kw.sgprPool.checkIn(recvSrd)
    kw.sgprPool.checkIn(nShardSgpr)
    kw.sgprPool.checkIn(myRankSgpr)
    return module

  def _fusedA2ALoadFlagBaseAndRank(self, module, flagBaseSgpr, dstRankSgpr, nShardSgpr, tmpSgpr):
    """Switch-load flag_ptr[dst_rank] into flagBaseSgpr and the integer dst_rank into dstRankSgpr.

    Same highest-matching-rank scan as _fusedA2ALoadRecvBase (dst_rank is the
    largest j with j*n_shard <= n_col_base_wg = WorkGroup1*MT1), but captures the
    numeric dst_rank (for counter[dst_rank]) and reads flag_ptr[dst_rank] (Task 5
    kernarg metadata, not in prologue SGPR).

    Args:
      module:       Module to append instructions to.
      flagBaseSgpr: 2-SGPR pair (aligned) to receive flag_ptr[dst_rank].
      dstRankSgpr:  1 SGPR to receive dst_rank (integer rank index).
      nShardSgpr:   1 SGPR pre-loaded with FusedNShard (n_shard, element units).
      tmpSgpr:      2 scratch SGPRs; tmpSgpr+0 holds n_col_base_wg, tmpSgpr+1 shard_lo.
    """
    from .Signature import fusedA2AKernArgLayout, FUSED_A2A_MAX_RANKS
    layout = fusedA2AKernArgLayout()
    fusedBase = self.parentWriter.states.fusedA2AKernArgBase

    # n_col_base_wg = WorkGroup1 * MacroTile1 (this WG's first output N column).
    module.add(SMulI32(dst=sgpr(tmpSgpr), src0=sgpr("WorkGroup1"), src1=self.kernel["MacroTile1"],
                       comment="n_col_base_wg = WorkGroup1 * MT1"))

    # Rank 0 default: flag base = flag_ptr_0, dst_rank = 0 (winner unless higher rank matches).
    module.add(self.parentWriter.argLoader.loadKernArg(flagBaseSgpr, "KernArgAddress",
      sgprOffset=hex(fusedBase + layout["flag_ptr_0"]), dword=2))
    module.add(SMovB32(dst=sgpr(dstRankSgpr), src=0, comment="dst_rank = 0 (default)"))

    for j in range(1, FUSED_A2A_MAX_RANKS):
      skipLabel = Label(self.parentWriter.labels.getNameInc("fusedA2A_flag_skip%u" % j),
                        f"n_col_base_wg below rank {j}")
      module.add(SMulI32(dst=sgpr(tmpSgpr + 1), src0=sgpr(nShardSgpr), src1=j,
                         comment=f"cand shard_lo = {j} * n_shard"))
      module.add(SCmpGtU32(src0=sgpr(tmpSgpr + 1), src1=sgpr(tmpSgpr),
                           comment=f"shard_lo > n_col_base_wg? (WG below rank {j})"))
      module.add(SCBranchSCC1(labelName=skipLabel.getLabelName(),
                              comment=f"below rank {j}: keep current winner"))
      module.add(self.parentWriter.argLoader.loadKernArg(flagBaseSgpr, "KernArgAddress",
        sgprOffset=hex(fusedBase + layout["flag_ptr_%u" % j]), dword=2))
      module.add(SMovB32(dst=sgpr(dstRankSgpr), src=j, comment=f"dst_rank = {j}"))
      module.add(skipLabel)
    module.add(SWaitCnt(kmcnt=0, comment="wait flag_ptr[dst_rank] load"))

  def _fusedA2ALoadFlagBaseByRank(self, module, flagBaseSgpr, rankSgpr):
    """Switch-load flag_ptr[rankSgpr] into flagBaseSgpr, selecting on a runtime rank SGPR.

    Sibling of _fusedA2ALoadFlagBaseAndRank, but the rank is an explicit runtime value
    (used by DRAIN: the elected last WG polls THIS card's own flag buffer flag_ptr[my_rank]).
    flag_ptr[j] is kernarg metadata only (Task 5), so this reads exactly one flag base on
    demand via a compile-time switch: rank 0 is the default; for each candidate j the load
    fires only when rankSgpr == j.  Slots j >= FusedW are never selected at run time.

    Args:
      module:       Module to append instructions to.
      flagBaseSgpr: 2-SGPR pair (aligned) to receive flag_ptr[rankSgpr].
      rankSgpr:     1 SGPR holding the rank index to select.
    """
    from .Signature import fusedA2AKernArgLayout, FUSED_A2A_MAX_RANKS
    layout = fusedA2AKernArgLayout()
    fusedBase = self.parentWriter.states.fusedA2AKernArgBase

    # Rank 0 default: flag base = flag_ptr_0 (overwritten if rankSgpr matches a higher rank).
    module.add(self.parentWriter.argLoader.loadKernArg(flagBaseSgpr, "KernArgAddress",
      sgprOffset=hex(fusedBase + layout["flag_ptr_0"]), dword=2))

    for j in range(1, FUSED_A2A_MAX_RANKS):
      skipLabel = Label(self.parentWriter.labels.getNameInc("fusedA2A_selfflag_skip%u" % j),
                        f"rank != {j}")
      module.add(SCmpEQU32(src0=sgpr(rankSgpr), src1=j,
                           comment=f"rank == {j}?"))
      module.add(SCBranchSCC0(labelName=skipLabel.getLabelName(),
                              comment=f"rank != {j}: keep current base"))
      module.add(self.parentWriter.argLoader.loadKernArg(flagBaseSgpr, "KernArgAddress",
        sgprOffset=hex(fusedBase + layout["flag_ptr_%u" % j]), dword=2))
      module.add(skipLabel)
    module.add(SWaitCnt(kmcnt=0, comment="wait flag_ptr[my_rank] load"))

  # READY value written into the destination card's flag slot (design spec 2.3).
  _FUSED_A2A_FLAG_READY = 1

  def _emitFusedA2AHandshake(self, module: Module):
    """Emit the cross-card handshake for PUSH workgroups (design spec section 2.3).

    Runs ONCE per WG (last batch of the store path), gated at RUNTIME to PUSH WGs
    (WorkGroup1 < AN_tiles, same gate as the PUSH store dispatch).  Sequence, per
    the design spec 2.3 timing (rank_0 supplying rank_1 example):
      (1) s_waitcnt vscnt(0)         -- this WG's scatter stores reached L2 before
                                        it decrements the counter (correctness key:
                                        else the elected last WG wbl2's incomplete data).
      (2) s_barrier + wave-0 gate    -- the counter atomic + election fire ONCE per WG,
                                        not per lane (StreamK single-writer blueprint).
      (3) old = atomic_add(counter[dst_rank], 1)   device scope, RETURN pre-op (sc0).
      (4) if old+1 != FusedTarget -> not the last WG -> skip the release.
      (5) elected last WG: buffer_wbl2 sc0 sc1 (flush this card's data for dst_rank
          across xGMI) + s_waitcnt vscnt(0) (wait wbl2) + store flag_ptr[dst_rank]
          [my_rank] = READY  system scope (sc0 sc1).

    Only the elected last WG does wbl2+flag, so there are exactly W wbl2 and W flag
    writes per card (granularity aligned, spec 2.3).  DRAIN-side poll (self flag) and
    the consumer-side acquire (buffer_inv) are later tasks (Task 9 / Task 13).
    """
    kw = self.parentWriter
    module.addComment2("fused-A2A cross-card handshake (design spec 2.3): counter election + wbl2 + system flag")

    afterLabel = Label(kw.labels.getNameInc("fusedA2A_handshake_after"),
                       "fused-A2A: after handshake (non-PUSH WGs skip)")
    skipReleaseLabel = Label(kw.labels.getNameInc("fusedA2A_handshake_notlast"),
                             "fused-A2A: not the last WG for dst_rank -> skip release")

    # --- runtime PUSH gate (same as the store dispatch): PUSH iff WorkGroup1 < AN_tiles,
    #     with AN_tiles = FusedAN >> log2(MacroTile1) read on demand from kernarg. ---
    from .Signature import fusedA2AKernArgLayout
    layout = fusedA2AKernArgLayout()
    fusedBase = kw.states.fusedA2AKernArgBase
    mt1 = self.kernel["MacroTile1"]
    log2mt1 = int(log2(mt1))
    gateSgpr = kw.sgprPool.checkOut(1, tag="fusedA2A_hsGate", preventOverflow=False)
    module.add(kw.argLoader.loadKernArg(gateSgpr, "KernArgAddress",
      sgprOffset=hex(fusedBase + layout["FusedAN"]), dword=1))
    module.add(SWaitCnt(kmcnt=0, comment="wait FusedAN"))
    module.add(SLShiftRightB32(dst=sgpr(gateSgpr), shiftHex=log2mt1, src=sgpr(gateSgpr),
                               comment=f"AN_tiles = FusedAN >> log2(MT1={mt1})"))
    module.add(SCmpGtU32(src0=sgpr(gateSgpr), src1=sgpr("WorkGroup1"),
                         comment="AN_tiles > WorkGroup1? (this WG in PUSH region)"))
    kw.sgprPool.checkIn(gateSgpr)
    module.add(SCBranchSCC0(labelName=afterLabel.getLabelName(),
                            comment="WorkGroup1 >= AN_tiles -> not a PUSH WG, skip handshake"))

    # Restore full EXEC: the store loop may leave a partial edge mask, but the
    # wave-0 election reads VReadfirstlaneB32(Serial) which needs lane 0 active.
    module.add(self.getEdgeMovInstType()(EXEC(), -1, "fused-A2A: full exec before wave-0 election"))

    # (1) ensure this WG's scatter stores are in L2 before touching the counter.
    module.add(SWaitCnt(vscnt=0, comment="fused-A2A: my PUSH stores reached L2 (spec 2.3 step 2)"))

    # --- kernarg reads (on demand, Task 5 contract): my_rank, target, n_shard.
    #     layout / fusedBase already bound above for the PUSH gate. ---
    myRankSgpr = kw.sgprPool.checkOut(1, tag="fusedA2A_hsMyRank", preventOverflow=False)
    targetSgpr = kw.sgprPool.checkOut(1, tag="fusedA2A_hsTarget", preventOverflow=False)
    nShardSgpr = kw.sgprPool.checkOut(1, tag="fusedA2A_hsNShard", preventOverflow=False)
    module.add(kw.argLoader.loadKernArg(myRankSgpr, "KernArgAddress",
      sgprOffset=hex(fusedBase + layout["FusedMyRank"]), dword=1))
    module.add(kw.argLoader.loadKernArg(targetSgpr, "KernArgAddress",
      sgprOffset=hex(fusedBase + layout["FusedTarget"]), dword=1))
    module.add(kw.argLoader.loadKernArg(nShardSgpr, "KernArgAddress",
      sgprOffset=hex(fusedBase + layout["FusedNShard"]), dword=1))

    # counter_ptr (dword=2) into an aligned pair.
    counterPtrSgpr = kw.sgprPool.checkOutAligned(2, 2, tag="fusedA2A_hsCounterPtr", preventOverflow=False)
    module.add(kw.argLoader.loadKernArg(counterPtrSgpr, "KernArgAddress",
      sgprOffset=hex(fusedBase + layout["counter_ptr"]), dword=2))
    module.add(SWaitCnt(kmcnt=0, comment="wait FusedMyRank/Target/NShard/counter_ptr"))

    # --- switch-load flag_ptr[dst_rank] + numeric dst_rank. ---
    flagBaseSgpr = kw.sgprPool.checkOutAligned(2, 2, tag="fusedA2A_hsFlagBase", preventOverflow=False)
    dstRankSgpr  = kw.sgprPool.checkOut(1, tag="fusedA2A_hsDstRank", preventOverflow=False)
    tmpSgpr2     = kw.sgprPool.checkOut(2, tag="fusedA2A_hsSwitchTmp", preventOverflow=False)
    self._fusedA2ALoadFlagBaseAndRank(module, flagBaseSgpr, dstRankSgpr, nShardSgpr, tmpSgpr2)

    # (2) all waves in this WG finished their PUSH stores + the L2 wait; elect a
    # single writer (wave 0) so the counter atomic fires once per WG, not per lane.
    module.add(SBarrier(comment="fused-A2A: all waves done before counter election"))
    serialSgpr = kw.sgprPool.checkOut(1, tag="fusedA2A_hsSerial", preventOverflow=False)
    module.add(VReadfirstlaneB32(dst=sgpr(serialSgpr), src=vgpr("Serial"),
                                 comment="wave 0 elects the WG's single counter writer"))
    module.add(SCmpEQU32(src0=sgpr(serialSgpr), src1=0, comment="wave 0?"))
    kw.sgprPool.checkIn(serialSgpr)
    module.add(SCBranchSCC0(labelName=afterLabel.getLabelName(),
                            comment="non-wave-0 -> skip (single writer per WG)"))

    # Wave 0 still has EXEC=-1 (set above so VReadfirstlaneB32(Serial) saw lane 0).
    # The counter increment + flag store below are VECTOR memory ops on a lane-
    # uniform address; under all-ones EXEC every active lane would issue them, so
    # the counter would jump by wavefrontSize per WG and the old+1==target election
    # never fires. Narrow EXEC to a single lane (thread 0, the same lane whose
    # Serial==0 passed the gate) so the atomic + flag store issue exactly once per
    # WG. Mirrors the file's mask-EXEC-before-atomic idiom (see lines 3112, 3160).
    module.add(self.getEdgeMovInstType()(EXEC(), 1, "fused-A2A: isolate lane 0 for the once-per-WG counter atomic + flag store"))

    # (3) counter[dst_rank] address = counter_ptr + dst_rank*4; atomic_add returns pre-op.
    module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr2), src=sgpr(dstRankSgpr), shiftHex=2,
                              comment="dst_rank * 4 (u32 counter byte offset)"))
    module.add(SAddU32(dst=sgpr(counterPtrSgpr), src0=sgpr(counterPtrSgpr), src1=sgpr(tmpSgpr2),
                       comment="counter[dst_rank] lo = counter_ptr + dst_rank*4"))
    module.add(SAddCU32(dst=sgpr(counterPtrSgpr + 1), src0=sgpr(counterPtrSgpr + 1), src1=0,
                        comment="counter[dst_rank] hi (carry)"))
    vCntAddr = kw.vgprPool.checkOutAligned(2, 2, tag="fusedA2A_hsCntAddr")
    vOne     = kw.vgprPool.checkOut(1, tag="fusedA2A_hsOne")
    vOld     = kw.vgprPool.checkOut(1, tag="fusedA2A_hsOld")
    module.add(VMovB32(dst=vgpr(vCntAddr + 0), src=sgpr(counterPtrSgpr + 0), comment="counter addr lo -> vgpr"))
    module.add(VMovB32(dst=vgpr(vCntAddr + 1), src=sgpr(counterPtrSgpr + 1), comment="counter addr hi -> vgpr"))
    module.add(VMovB32(dst=vgpr(vOne), src=1, comment="counter increment = 1"))
    offSaddr = vgpr("off", 1, False, False, True)
    module.add(GlobalAtomicAddU32(
      dst=vgpr(vOld), vaddr=vgpr(vCntAddr, 2), data=vgpr(vOne), saddr=offSaddr,
      modifier=GLOBALModifiers(glc=True, slc=False, scope=CacheScope.SCOPE_NONE),
      comment="old = atomic_add(counter[dst_rank], 1) device scope, return pre-op (sc0)"))
    module.add(SWaitCnt(vlcnt=0, comment="fused-A2A: wait counter atomic return (load counter)"))

    # (4) last WG for dst_rank iff old+1 == FusedTarget; else skip the release.
    module.add(VReadfirstlaneB32(dst=sgpr(tmpSgpr2), src=vgpr(vOld), comment="old -> sgpr"))
    module.add(SAddU32(dst=sgpr(tmpSgpr2), src0=sgpr(tmpSgpr2), src1=1, comment="old + 1"))
    module.add(SCmpEQU32(src0=sgpr(tmpSgpr2), src1=sgpr(targetSgpr),
                         comment="old+1 == FusedTarget? (this WG is the last for dst_rank)"))
    module.add(SCBranchSCC0(labelName=skipReleaseLabel.getLabelName(),
                            comment="not the last WG -> skip wbl2 + flag"))

    # (5) elected last WG: release this card's data for dst_rank + set remote flag.
    module.add(BufferWbl2(scope=CacheScope.SCOPE_SYS,
                          comment="fused-A2A: flush data for dst_rank across xGMI (system scope, sc0 sc1)"))
    module.add(SWaitCnt(vscnt=0, comment="fused-A2A: wait wbl2 complete (spec 2.3 step 4)"))
    # remote flag address = flag_ptr[dst_rank] + my_rank*4 (flag[my_rank] slot on dest).
    module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr2), src=sgpr(myRankSgpr), shiftHex=2,
                              comment="my_rank * 4 (dest flag slot byte offset)"))
    module.add(SAddU32(dst=sgpr(flagBaseSgpr), src0=sgpr(flagBaseSgpr), src1=sgpr(tmpSgpr2),
                       comment="flag[my_rank] lo = flag_ptr[dst_rank] + my_rank*4"))
    module.add(SAddCU32(dst=sgpr(flagBaseSgpr + 1), src0=sgpr(flagBaseSgpr + 1), src1=0,
                        comment="flag[my_rank] hi (carry)"))
    vFlagAddr = kw.vgprPool.checkOutAligned(2, 2, tag="fusedA2A_hsFlagAddr")
    vReady    = kw.vgprPool.checkOut(1, tag="fusedA2A_hsReady")
    module.add(VMovB32(dst=vgpr(vFlagAddr + 0), src=sgpr(flagBaseSgpr + 0), comment="flag addr lo -> vgpr"))
    module.add(VMovB32(dst=vgpr(vFlagAddr + 1), src=sgpr(flagBaseSgpr + 1), comment="flag addr hi -> vgpr"))
    module.add(VMovB32(dst=vgpr(vReady), src=self._FUSED_A2A_FLAG_READY, comment="READY value"))
    module.add(GlobalStoreB32(
      vaddr=vgpr(vFlagAddr, 2), src=vgpr(vReady), saddr=offSaddr,
      modifier=GLOBALModifiers(glc=True, slc=True, scope=CacheScope.SCOPE_NONE, isStore=True),
      comment="flag_ptr[dst_rank][my_rank] = READY (system scope, sc0 sc1)"))
    module.add(SWaitCnt(vscnt=0, comment="fused-A2A: wait flag store issued"))

    # --- DRAIN barrier (design spec 2.4): make kernel-exit == this card received ---
    # all its incoming data.  The WG that elected counter[dst_rank] (=j), after setting
    # rank j's remote flag above, confirms THIS card's recv[j] slot arrived here by
    # polling THIS card's own flag buffer at flag_ptr[my_rank] + j*4.  j==my_rank has no
    # remote producer (recv[my_rank] is written locally), so it self-sets that slot
    # instead of polling (else it would spin on a signal that never comes).  Gated at
    # RUNTIME by the FusedDrain kernarg (not a compile-time gate: a compile-time gate
    # would fork the fused kernel into drain-on/off variants).  Still single-lane EXEC.
    skipDrainLabel = Label(kw.labels.getNameInc("fusedA2A_drain_skip"),
                           "fused-A2A: FusedDrain==0 -> no drain barrier")
    drainPollLabel = Label(kw.labels.getNameInc("fusedA2A_drain_poll"),
                           "fused-A2A: DRAIN poll self flag[j] until READY")
    drainSelfSetLabel = Label(kw.labels.getNameInc("fusedA2A_drain_selfset"),
                              "fused-A2A: DRAIN j==my_rank -> self-set flag[my_rank]")

    # runtime gate: FusedDrain == 0 -> skip the whole barrier.
    drainSgpr = kw.sgprPool.checkOut(1, tag="fusedA2A_drainFlag", preventOverflow=False)
    module.add(kw.argLoader.loadKernArg(drainSgpr, "KernArgAddress",
      sgprOffset=hex(fusedBase + layout["FusedDrain"]), dword=1))
    module.add(SWaitCnt(kmcnt=0, comment="wait FusedDrain"))
    module.add(SCmpEQU32(src0=sgpr(drainSgpr), src1=0, comment="FusedDrain == 0?"))
    kw.sgprPool.checkIn(drainSgpr)
    module.add(SCBranchSCC1(labelName=skipDrainLabel.getLabelName(),
                            comment="FusedDrain==0 -> skip drain barrier"))

    # self flag base = flag_ptr[my_rank] (THIS card's own flag buffer). Reuses flagBaseSgpr
    # (its previous remote value flag_ptr[dst_rank] is no longer needed after the store above).
    self._fusedA2ALoadFlagBaseByRank(module, flagBaseSgpr, myRankSgpr)

    # j == my_rank -> self-set (no remote producer); else poll.
    module.add(SCmpEQU32(src0=sgpr(dstRankSgpr), src1=sgpr(myRankSgpr),
                         comment="dst_rank == my_rank? (recv[my_rank] is local, no remote producer)"))
    module.add(SCBranchSCC1(labelName=drainSelfSetLabel.getLabelName(),
                            comment="j==my_rank -> self-set flag[my_rank], skip poll"))

    # poll path: flag[j] address = flag_ptr[my_rank] + j*4 (source rank j's slot on this card).
    module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr2), src=sgpr(dstRankSgpr), shiftHex=2,
                              comment="j * 4 (self flag slot byte offset)"))
    module.add(SAddU32(dst=sgpr(flagBaseSgpr), src0=sgpr(flagBaseSgpr), src1=sgpr(tmpSgpr2),
                       comment="self flag[j] lo = flag_ptr[my_rank] + j*4"))
    module.add(SAddCU32(dst=sgpr(flagBaseSgpr + 1), src0=sgpr(flagBaseSgpr + 1), src1=0,
                        comment="self flag[j] hi (carry)"))
    module.add(VMovB32(dst=vgpr(vFlagAddr + 0), src=sgpr(flagBaseSgpr + 0), comment="self flag addr lo -> vgpr"))
    module.add(VMovB32(dst=vgpr(vFlagAddr + 1), src=sgpr(flagBaseSgpr + 1), comment="self flag addr hi -> vgpr"))
    # spin: system-scope load (sc0 sc1) bypasses this card's stale L2 (0) to read the HBM
    # truth written by the remote producer; loop until == READY.
    module.add(drainPollLabel)
    module.add(GlobalLoadB32(
      dst=vgpr(vReady), vaddr=vgpr(vFlagAddr, 2), saddr=offSaddr,
      modifier=GLOBALModifiers(glc=True, slc=True, scope=CacheScope.SCOPE_NONE, isStore=False),
      comment="poll self flag[j] (system scope, sc0 sc1)"))
    module.add(SWaitCnt(vlcnt=0, comment="fused-A2A: wait poll load"))
    module.add(VReadfirstlaneB32(dst=sgpr(tmpSgpr2), src=vgpr(vReady), comment="flag[j] -> sgpr"))
    module.add(SCmpEQU32(src0=sgpr(tmpSgpr2), src1=self._FUSED_A2A_FLAG_READY,
                         comment="flag[j] == READY?"))
    module.add(SCBranchSCC0(labelName=drainPollLabel.getLabelName(),
                            comment="not READY yet -> spin (poll again)"))
    module.add(SBranch(labelName=skipDrainLabel.getLabelName(),
                       comment="flag[j] READY -> drain done"))

    # self-set path (j==my_rank): system-scope store of READY to flag_ptr[my_rank] + my_rank*4
    # (matches the remote-flag store + poll load scope so the system-scope downstream reader sees it).
    module.add(drainSelfSetLabel)
    module.add(SLShiftLeftB32(dst=sgpr(tmpSgpr2), src=sgpr(myRankSgpr), shiftHex=2,
                              comment="my_rank * 4 (self flag slot byte offset)"))
    module.add(SAddU32(dst=sgpr(flagBaseSgpr), src0=sgpr(flagBaseSgpr), src1=sgpr(tmpSgpr2),
                       comment="self flag[my_rank] lo = flag_ptr[my_rank] + my_rank*4"))
    module.add(SAddCU32(dst=sgpr(flagBaseSgpr + 1), src0=sgpr(flagBaseSgpr + 1), src1=0,
                        comment="self flag[my_rank] hi (carry)"))
    module.add(VMovB32(dst=vgpr(vFlagAddr + 0), src=sgpr(flagBaseSgpr + 0), comment="self flag addr lo -> vgpr"))
    module.add(VMovB32(dst=vgpr(vFlagAddr + 1), src=sgpr(flagBaseSgpr + 1), comment="self flag addr hi -> vgpr"))
    module.add(VMovB32(dst=vgpr(vReady), src=self._FUSED_A2A_FLAG_READY, comment="READY value"))
    module.add(GlobalStoreB32(
      vaddr=vgpr(vFlagAddr, 2), src=vgpr(vReady), saddr=offSaddr,
      modifier=GLOBALModifiers(glc=True, slc=True, scope=CacheScope.SCOPE_NONE, isStore=True),
      comment="flag_ptr[my_rank][my_rank] = READY (system scope, sc0 sc1)"))
    module.add(SWaitCnt(vscnt=0, comment="fused-A2A: wait self-set flag store issued"))
    module.add(skipDrainLabel)
    # No buffer_inv here: this kernel does not read recv; acquire is the recv-reader's job.
    # TODO Task 13: multi-card DRAIN validation (poll path across real xGMI producers).

    kw.vgprPool.checkIn(vReady)
    kw.vgprPool.checkIn(vFlagAddr)

    module.add(skipReleaseLabel)
    kw.vgprPool.checkIn(vOld)
    kw.vgprPool.checkIn(vOne)
    kw.vgprPool.checkIn(vCntAddr)
    kw.sgprPool.checkIn(tmpSgpr2)
    kw.sgprPool.checkIn(dstRankSgpr)
    kw.sgprPool.checkIn(flagBaseSgpr)
    kw.sgprPool.checkIn(counterPtrSgpr)
    kw.sgprPool.checkIn(nShardSgpr)
    kw.sgprPool.checkIn(targetSgpr)
    kw.sgprPool.checkIn(myRankSgpr)
    # Restore full EXEC before falling through to afterLabel: wave 0 narrowed EXEC
    # to a single lane for the counter atomic + flag store, but the CLS look-ahead
    # emitted after the handshake (emit(): emitCoord1Advance) issues a VECTOR
    # VAddCOU32 on coord1 that needs all lanes active. The two early gate branches
    # (PUSH gate, wave-0 gate) jump straight to afterLabel and bypass this restore,
    # which is correct -- neither of those paths narrowed EXEC.
    module.add(self.getEdgeMovInstType()(EXEC(), -1, "fused-A2A: restore full exec after single-lane handshake"))
    module.add(afterLabel)

  def _emitSubtileOobGuard(self, targetModule, blockIdxM: int, blockIdxN: int, labelPrefix: str = "subtile_skip_store"):
    """Emit M/N OOB guard branches for UseSubtileImpl NonEdge stores.

    Background
    ----------
    UseSubtileImpl assigns each wave a fixed subtile region of the output matrix.
    In the NonEdge path the macro-tile fits entirely within the output bounds, but
    individual wave groups within the macro-tile may still be out-of-bounds when
    the problem size is not a multiple of the macro-tile.  The SGPRs
    subtileM32ValidBlocksSgpr and subtileN16ValidBlocksSgpr count how many M/N
    blocks (in units of mBlockSize rows / nBlockSize columns) belong to valid
    output for this wave, and are set to None when no guard is needed (edge path
    or problem is tile-aligned).

    Logic
    -----
    For a store at (blockIdxM, blockIdxN):
      - If numValidNBlocks <= blockIdxN → N OOB: jump past ALL remaining stores.
        Valid because N is monotone: subsequent N groups (blockIdxN+1, ...) are also OOB.
      - If numValidMBlocks <= blockIdxM → M OOB: jump to the end of the current N group.
        Valid because M is monotone: remaining M elements in this N group are also OOB.

    The N guard is emitted ONCE per N group (when blockIdxN changes).  It branches to
    _subtileAllStoresEndLabel (past all stores).  _subtileNGroupSkipLabel marks the
    boundary between N groups; M guards branch there to skip the rest of the current
    N group without re-testing the remaining M elements.
    Both labels are placed by _finalizeSubtileOobGuards (called after the element loop).

    Returns a per-element skip Label only when there is no N guard (M-only case); the
    caller must add it after the store.  Returns None in all other cases.
    """
    guardMSgpr = self.parentWriter.states.subtileM32ValidBlocksSgpr
    guardNSgpr = self.parentWriter.states.subtileN16ValidBlocksSgpr
    # No guard SGPRs means the store is always in-bounds for this path; nothing to emit.
    if guardMSgpr is None and guardNSgpr is None:
      return None

    # --- N-group guard (emitted once per unique blockIdxN) ---
    # Branches to _subtileAllStoresEndLabel when N OOB, skipping all remaining stores.
    # Because N is monotone (blockIdxN increases each group), if this group is OOB
    # then every subsequent group is also OOB — no need to test them.
    if guardNSgpr is not None and blockIdxN != self._subtilePrevBlockIdxN:
      # Place the previous N group's end label before starting a new group.
      if self._subtileNGroupSkipLabel is not None:
        targetModule.add(self._subtileNGroupSkipLabel)
        self._subtileNGroupSkipLabel = None
      if self._subtilePendingSrdDInc is not None:
        targetModule.add(self._subtilePendingSrdDInc)
        self._subtilePendingSrdDInc = None
      # Create the single end-of-all-stores label on the first N group.
      if self._subtileAllStoresEndLabel is None:
        endLabelName = self.parentWriter.labels.getNameInc("subtile_all_stores_end")
        self._subtileAllStoresEndLabel = Label(endLabelName, "end of all subtile NonEdge D stores")
      nGroupEndLabelName = self.parentWriter.labels.getNameInc(
        f"{labelPrefix}_N{blockIdxN}_end")
      nGroupEndLabel = Label(nGroupEndLabelName,
                             f"end of N group blockIdxN={blockIdxN} (M cbranch target)")
      nGuardCmp = blockIdxN * 16 if self.parentWriter.states.storeAlign8 else blockIdxN
      targetModule.add(_scmpGtU32(self.parentWriter, sgpr("SubtileNGuard"), nGuardCmp,
                                   comment=f"quick-exit: clamped > {nGuardCmp}? (OOB -> skip all stores)"))
      targetModule.add(SCBranchSCC0(labelName=self._subtileAllStoresEndLabel.getLabelName(),
                                     comment=f"quick-exit: N OOB at blockIdxN={blockIdxN}, skip all remaining stores"))
      self._subtileNGroupSkipLabel = nGroupEndLabel
      self._subtilePrevBlockIdxN = blockIdxN

    # --- M guard (emitted per element) ---
    # Branches to end of current N group when M OOB, skipping remaining M elements.
    # Because M is monotone (blockIdxM increases within the N group), if this element
    # is OOB then all subsequent M elements in this N group are also OOB.
    if guardMSgpr is None:
      return None
    targetModule.add(_scmpGtU32(self.parentWriter, sgpr("SubtileMGuard"), blockIdxM,
                                 comment=f"quick-exit: numValidMBlocks > {blockIdxM}? (OOB -> skip N group)"))
    if guardNSgpr is not None and self._subtileNGroupSkipLabel is not None:
      # M OOB → jump to end of this N group (no per-element label needed).
      targetModule.add(SCBranchSCC0(labelName=self._subtileNGroupSkipLabel.getLabelName(),
                                     comment=f"quick-exit: M OOB at blockIdxM={blockIdxM}, skip rest of N group"))
      return None
    else:
      # No N guard → fall back to a per-element skip label (caller places it after the store).
      skipLabelName = self.parentWriter.labels.getNameInc(
        f"{labelPrefix}_M{blockIdxM}_N{blockIdxN}")
      skipLabel = Label(skipLabelName,
                        f"skip OOB store blockIdxM={blockIdxM} blockIdxN={blockIdxN}")
      targetModule.add(SCBranchSCC0(labelName=skipLabel.getLabelName(),
                                     comment=f"quick-exit: M OOB at blockIdxM={blockIdxM}, skip store"))
      return skipLabel

  def _finalizeSubtileOobGuards(self, targetModule):
    """Place the pending N-group end label and end-of-all-stores label after the element loop.

    Must be called once after all elements have been emitted to close out the last
    N group and anchor the N-cbranch target past all stores.
    """
    if self._subtileNGroupSkipLabel is not None:
      targetModule.add(self._subtileNGroupSkipLabel)
      self._subtileNGroupSkipLabel = None
    if self._subtilePendingSrdDInc is not None:
      targetModule.add(self._subtilePendingSrdDInc)
      self._subtilePendingSrdDInc = None
    if self._subtileAllStoresEndLabel is not None:
      targetModule.add(self._subtileAllStoresEndLabel)
      self._subtileAllStoresEndLabel = None

  def _emitAlign8ExecMask(self, module, tmpS, tmpS2, blockIdxM, blockIdxN, mGuardOffset, rowScaleShift):
    """Emit exec mask for partial M/N blocks in the NonEdge store path.

    Computes a 64-bit exec mask that disables OOB lanes for partial M and N
    blocks at the tile boundary.  The mask is the AND of an M component and
    an N component, each computed independently.

    MMA output layout (MI16x16, wavefront=64):
      64 lanes = 4 lane-groups (LGs) of 16 lanes each.
      Each LG owns consecutive M-rows:
        - Paired bf16 store: 8 M-rows per LG (32 rows total, 2 MMA tiles)
        - Scalar/fp32 store: 4 M-rows per LG (16 rows total, 1 MMA tile)
      Within each LG, lane_id % 16 selects the N-column (0..15).

    M mask algorithm:
      Given validRows (number of valid M-rows in this block), the number of
      active lane-groups is validRows / rowsPerLG.  Each LG occupies 16
      consecutive lanes, so the mask is the bottom (numValidLGs * 16) bits of
      a 64-bit word.  This is computed as:
        shiftAmt = 64 - validRows * (16 / rowsPerLG)
        mask = (uint64_t)-1 >> shiftAmt
      validM_wave is precomputed once per tile in _emitSubtileGuards.

    N mask algorithm:
      SubtileNGuard holds the clamped valid-N-column count for this wave.
      partialN = SubtileNGuard % 16 gives the number of valid columns within
      the last 16-column MMA tile.  The N mask disables columns >= partialN:
        nMask = (1 << partialN) - 1        e.g. partialN=5 -> 0x001F
        nMask = nMask | (nMask << 16)      replicate across both LG halves
      This 32-bit word is ANDed into both lo and hi halves of the exec mask.

    This sequence is designed to fit within the ds_bpermute latency window
    (~88 cycles at 4 cyc/issue) for zero-overhead execution on interior blocks.

    Args:
        module:         Target module to emit instructions into.
        tmpS:           SGPR index for mask result (2 consecutive, even-aligned).
        tmpS2:          SGPR index for scratch (2 consecutive).
        blockIdxM:      M block index for this store element.
        blockIdxN:      N block index for this store element.
        mGuardOffset:   Number of MMA tiles this store spans (2=paired, 1=single).
        rowScaleShift:  Left-shift to convert validRows to lane-count
                        (1 for paired/8-rows-per-LG, 2 for scalar-fp32/4-rows-per-LG).
    """
    miM = self.kernel["MatrixInstM"]
    blockStartRow = blockIdxM * miM
    validMWaveSgpr = self.parentWriter.states.subtileTotalMOffsetSgpr
    mMaskDone = Label(self.parentWriter.labels.getNameInc("align8_m_done"), "")

    # --- M mask: per-lane-group selection via right-shift ---
    isWave32 = self.wavelen == 32
    if isWave32:
      module.add(SMovB32(dst=sgpr(tmpS), src=-1, comment="mask = full"))
    else:
      module.add(SMovB64(dst=sgpr(tmpS, 2), src=-1, comment="mask = full"))
    module.add(_scmpGtU32(self.parentWriter, sgpr("SubtileMGuard"), blockIdxM + mGuardOffset,
                          comment=f"SubtileMGuard > {blockIdxM + mGuardOffset}? (block fully interior)"))
    module.add(SCBranchSCC1(labelName=mMaskDone.getLabelName(),
                            comment="interior M block -> mask stays -1"))
    if blockStartRow > 0:
      module.add(SSubU32(dst=sgpr(tmpS2), src0=sgpr(validMWaveSgpr), src1=blockStartRow,
                         comment=f"validRows = validM_wave - {blockStartRow}"))
      module.add(SCSelectB32(dst=sgpr(tmpS2), src0=0, src1=sgpr(tmpS2), comment="clamp to 0"))
      module.add(SLShiftLeftB32(dst=sgpr(tmpS2), src=sgpr(tmpS2), shiftHex=rowScaleShift,
                                comment=f"validRows * {1 << rowScaleShift}"))
    else:
      module.add(SLShiftLeftB32(dst=sgpr(tmpS2), src=sgpr(validMWaveSgpr), shiftHex=rowScaleShift,
                                comment=f"validM_wave * {1 << rowScaleShift}"))
    module.add(SSubU32(dst=sgpr(tmpS2), src0=self.wavelen, src1=sgpr(tmpS2),
                       comment=f"shiftAmt = {self.wavelen} - validRows * scale"))
    if isWave32:
      module.add(SLShiftRightB32(dst=sgpr(tmpS), src=-1, shiftHex=sgpr(tmpS2),
                                 comment="M mask = -1 >> shiftAmt"))
    else:
      module.add(SLShiftRightB64(dst=sgpr(tmpS, 2), src=-1, shiftHex=sgpr(tmpS2),
                                 comment="M mask = -1 >> shiftAmt"))
    module.add(mMaskDone)

    # --- N mask: per-column bit-mask for arbitrary partial N ---
    nCmpVal = (blockIdxN + 1) * 16
    nMaskDone = Label(self.parentWriter.labels.getNameInc("align8_n_done"), "")
    module.add(_scmpGtU32(self.parentWriter, sgpr("SubtileNGuard"), nCmpVal,
                          comment=f"clamped > {nCmpVal}? (not last N block)"))
    module.add(SCBranchSCC1(labelName=nMaskDone.getLabelName(),
                            comment="interior N block -> mask unchanged"))
    module.add(SAndB32(dst=sgpr(tmpS2), src0=sgpr("SubtileNGuard"), src1=0xF,
                       comment="partialN = clamped %% 16, SCC=1 if non-zero"))
    nFullLabel = Label(self.parentWriter.labels.getNameInc("align8_n_full"), "")
    module.add(SCBranchSCC0(labelName=nFullLabel.getLabelName(),
                            comment="partialN==0 -> full 16-col block"))
    module.add(SLShiftLeftB32(dst=sgpr(tmpS2), src=1, shiftHex=sgpr(tmpS2),
                              comment="1 << partialN"))
    module.add(SSubU32(dst=sgpr(tmpS2), src0=sgpr(tmpS2), src1=1,
                       comment="(1 << partialN) - 1"))
    if isWave32:
      # wave32: replicate lo16 to both halves without extra scratch SGPR
      module.add(SMulI32(dst=sgpr(tmpS2), src0=sgpr(tmpS2), src1=hex(0x10001),
                         comment="replicate lo16 to both halves"))
      module.add(SAndB32(dst=sgpr(tmpS), src0=sgpr(tmpS), src1=sgpr(tmpS2),
                         comment="mask &= N mask"))
    else:
      module.add(SLShiftLeftB32(dst=sgpr(tmpS2+1), src=sgpr(tmpS2), shiftHex=16,
                                comment="replicate to hi16"))
      module.add(SOrB32(dst=sgpr(tmpS2), src0=sgpr(tmpS2), src1=sgpr(tmpS2+1),
                        comment="N mask word = lo16 | hi16"))
      module.add(SAndB32(dst=sgpr(tmpS), src0=sgpr(tmpS), src1=sgpr(tmpS2),
                         comment="mask_lo &= N mask"))
      module.add(SAndB32(dst=sgpr(tmpS+1), src0=sgpr(tmpS+1), src1=sgpr(tmpS2),
                         comment="mask_hi &= N mask"))
    module.add(nFullLabel)
    module.add(nMaskDone)

  def _emitSubtileNColVaddr(self, module: Module, coord0: int, coord1: int) -> int:
    """Compute the per-lane vc1=0 N-col base vaddr for the SourceSwap row-major layout.

    Shared address math for both the full 4-wide store (_emit16bitSubtileNWideStore)
    and the partial per-partner fallback (_emit16bitSubtileNColScalarStore).  Emits into
    `module` and returns the vgpr index holding the vaddr (== vPack+2).  The N-col base
    corresponds to vc1=0; partner vc1=i lands at this vaddr + i*bpe (N is contiguous in
    row-major D, StrideD1J==1).  See _emit16bitSubtileNWideStore's docstring for the
    per-lane address derivation.
    """
    vPack = self.cvtVgprStruct.vgprBf16Temp
    bpe = self.parentWriter.states.bpeCexternalGSU1  # always 2 for 16bit dest
    strideD0 = self.parentWriter.strideRef('D', 0)
    ws     = self.kernel["WavefrontSize"]
    wg0    = self.kernel["MIWaveGroup"][0]
    wg1    = self.kernel["MIWaveGroup"][1]
    mt0    = self.kernel["MacroTile0"]
    mt1    = self.kernel["MacroTile1"]
    wave_rows = mt0 // wg0
    wave_cols = mt1 // wg1
    wsLog2 = int(log2(ws))
    tmpS = self.tmpS01

    # --- N-col element count into vPack+3 ---
    #   N_col = wave_id1 * wave_cols + lane_group*4 + coord1
    module.addComment1("compute per-lane N-col element count")
    module.add(VLShiftRightB32(dst=vgpr(vPack+3), shiftHex=4, src=vgpr("Serial"),
                               comment="lane_id >> 4"))
    module.add(VAndB32(dst=vgpr(vPack+3), src0=3, src1=vgpr(vPack+3),
                       comment="lane_group = (lane_id >> 4) & 3"))
    module.add(VLShiftLeftB32(dst=vgpr(vPack+3), shiftHex=2, src=vgpr(vPack+3),
                              comment="lane_group * 4 (N-col base within lane)"))
    if coord1:
      module.add(SMovB32(dst=sgpr(tmpS), src=coord1, comment=f"coord1={coord1}"))
      module.add(VAddU32(dst=vgpr(vPack+3), src0=vgpr(vPack+3), src1=sgpr(tmpS),
                         comment="+ coord1 (N-col subtile/MFMA-tile base)"))
    if wg1 > 1:
      module.add(VLShiftRightB32(dst=vgpr(vPack+2), shiftHex=wsLog2, src=vgpr("Serial"),
                                 comment=f"waveId = Serial >> {wsLog2}"))
      if wg0 & (wg0 - 1) == 0:
        module.add(VLShiftRightB32(dst=vgpr(vPack+2), shiftHex=int(log2(wg0)), src=vgpr(vPack+2),
                                   comment=f"wave_id1 = waveId >> log2(wg0={wg0})"))
      else:
        raise NotImplementedError(f"Non-power-of-2 MIWaveGroup[0]={wg0} not supported in N-wide store")
      module.add(SMovB32(dst=sgpr(tmpS), src=wave_cols, comment=f"wave_cols={wave_cols}"))
      module.add(VMulLOU32(dst=vgpr(vPack+2), src0=vgpr(vPack+2), src1=sgpr(tmpS),
                           comment="wave_id1 * wave_cols"))
      module.add(VAddU32(dst=vgpr(vPack+3), src0=vgpr(vPack+3), src1=vgpr(vPack+2),
                         comment="N_col += wave_id1 * wave_cols"))
    module.add(VLShiftLeftB32(dst=vgpr(vPack+2), shiftHex=int(log2(bpe)), src=vgpr(vPack+3),
                              comment=f"N byte offset = N_col * {bpe}"))

    # --- M-row element count into vPack+3, then scale by StrideD0*bpe ---
    #   M_row = wave_id0 * wave_rows + wg0*MT0 + coord0 + (lane_id & 15)
    module.addComment1("compute per-lane M-row element count and scale by StrideD0*bpe")
    module.add(VAndB32(dst=vgpr(vPack+3), src0=15, src1=vgpr("Serial"),
                       comment="row_in_wave = lane_id & 15 (SourceSwap: lane -> M-row)"))
    if coord0:
      module.add(SMovB32(dst=sgpr(tmpS), src=coord0, comment=f"coord0={coord0}"))
      module.add(VAddU32(dst=vgpr(vPack+3), src0=vgpr(vPack+3), src1=sgpr(tmpS),
                         comment="+ coord0 (M-tile position)"))
    module.add(SMulI32(dst=sgpr(tmpS), src0=sgpr("WorkGroup0"), src1=mt0,
                       comment="wg0_M_elems = WorkGroup0 * MT0"))
    module.add(VAddU32(dst=vgpr(vPack+3), src0=vgpr(vPack+3), src1=sgpr(tmpS),
                       comment="M_row += wg0_M_elems"))
    if wg0 > 1:
      module.add(VLShiftRightB32(dst=vgpr(vPack+0), shiftHex=wsLog2, src=vgpr("Serial"),
                                 comment=f"waveId = Serial >> {wsLog2}"))
      if wg0 & (wg0 - 1) == 0:
        module.add(VAndB32(dst=vgpr(vPack+0), src0=wg0 - 1, src1=vgpr(vPack+0),
                           comment=f"wave_id0 = waveId & {wg0-1}"))
      else:
        raise NotImplementedError(f"Non-power-of-2 MIWaveGroup[0]={wg0} not supported in N-wide store")
      module.add(SMovB32(dst=sgpr(tmpS), src=wave_rows, comment=f"wave_rows={wave_rows}"))
      module.add(VMulLOU32(dst=vgpr(vPack+0), src0=vgpr(vPack+0), src1=sgpr(tmpS),
                           comment="wave_id0 * wave_rows"))
      module.add(VAddU32(dst=vgpr(vPack+3), src0=vgpr(vPack+3), src1=vgpr(vPack+0),
                         comment="M_row += wave_id0 * wave_rows"))
    module.add(SMulI32(dst=sgpr(tmpS), src0=strideD0, src1=bpe,
                       comment="StrideD0*bpe (row-major M-row byte stride)"))
    module.add(VMulLOU32(dst=vgpr(vPack+3), src0=vgpr(vPack+3), src1=sgpr(tmpS),
                         comment="M byte offset = M_row * StrideD0 * bpe"))
    module.add(VAddU32(dst=vgpr(vPack+2), src0=vgpr(vPack+2), src1=vgpr(vPack+3),
                       comment="vaddr = N byte offset + M byte offset"))
    return vPack + 2

  def _emit16bitSubtileNWideStore(self, sumIdxList, prefixOffset: int,
                                  coord0: int, coord1: int,
                                  blockIdxM: int = 0, blockIdxN: int = 0) -> Module:
    """Emit a wide 16bit store along N for the SourceSwap row-major subtile layout.

    Under SourceSwap the MFMA A/B operands are swapped, so a lane's 4 accumulators
    (reg_k 0..3 within a subtile) lie along N (consecutive N-cols) at a FIXED M-row --
    the transpose of the non-SourceSwap layout (4 consecutive M-rows at a fixed N-col).
    Row-major D (UseInitialStridesCD) has StrideD1J == 1, so those 4 N-cols are contiguous
    in memory: 4 bf16 = 8 bytes = one buffer_store_dwordx2.

    The 4 N-cols are 4 SEPARATE store elements (vc1 = 0,1,2,3) sharing the same
    (d1, d0); sumIdxList holds their elementSumIdx in N-col order (vc1 ascending).  Each
    sumIdx already carries the correct value for its (M-row, N-col) position (accVgprRead
    mapped element[i] -> sumIdx[i] in arch order), so the store simply packs the 4 values
    and writes them contiguously along N.

    Per-lane address (elements), matching _build_accvgpr_init_matrix_transposed_asm:
      M_row  = wave_id0 * wave_rows + wg0*MT0 + coord0 + (lane_id & 15)
      N_col  = wave_id1 * wave_cols + lane_group*4 + coord1   (coord1 == 0 for the vc1=0 elem)
      lane_group = (lane_id >> 4) & 3         (SourceSwap: 4-col groups in N)
      vaddr  = M_row * StrideD0 * bpe + N_col * bpe
    The SRD base encodes only wg1*MT1*StrideD1J*bpe (N workgroup offset); the M workgroup
    offset (wg0*MT0) and wave/lane offsets are folded into vaddr here.

    Args:
      sumIdxList:   4 elementSumIdx values (N-col order vc1=0..3).
      prefixOffset: parentWriter.states.c.startVgprValu (offset into ValuC).
      coord0:       elementCoord0 (M-tile position, in elements) of the vc1=0 element.
      coord1:       elementCoord1 (N-col base, == 0 for the vc1=0 element).
    """
    module = Module("16bitSubtileNWideStore")
    isFp16 = self.kernel["ProblemType"]["DestDataType"].isHalf()

    ntd = self.kernel["NonTemporalD"]
    isGlc = bool(ntd & 0x1)
    isSlc = bool(ntd & 0x2)
    isNT  = bool(ntd & 0x4)

    # Scratch vgprs from the cvtVgprStruct block (overwritten each call):
    #   vPack+0 : packed dword (N-col 0,1)   vPack+1 : packed dword (N-col 2,3)
    #   vPack+2 : per-lane vaddr             vPack+3 : temp for offset arithmetic
    vPack = self.cvtVgprStruct.vgprBf16Temp

    def vc(sumIdx):
      idx = sumIdx - prefixOffset
      return vgpr("ValuC+" + str(idx))

    typeStr = "fp16" if isFp16 else "bf16"
    VCvtPkF32to16 = VCvtPkF32toFP16 if isFp16 else VCvtPkF32toBF16
    module.addComment1(f"{typeStr} SourceSwap N-wide subtile: pack 4 N-cols at fixed M-row, store dwordx2 along N (row-major contiguous)")

    # Per-lane vc1=0 N-col base vaddr -> vPack+2 (shared with the split fallback).
    self._emitSubtileNColVaddr(module, coord0, coord1)

    # --- pack the 4 N-cols into 2 dwords ---
    # vc(sumIdxList[0]) -> N-col 0 (lo16 of dword0), vc(sumIdxList[1]) -> N-col 1 (hi16 of dword0)
    # vc(sumIdxList[2]) -> N-col 2 (lo16 of dword1), vc(sumIdxList[3]) -> N-col 3 (hi16 of dword1)
    module.add(VCvtPkF32to16(dst=vgpr(vPack+0), src0=vc(sumIdxList[0]), src1=vc(sumIdxList[1]),
                             comment=f"N-col 0,1 -> {typeStr}"))
    module.add(VCvtPkF32to16(dst=vgpr(vPack+1), src0=vc(sumIdxList[2]), src1=vc(sumIdxList[3]),
                             comment=f"N-col 2,3 -> {typeStr}"))
    module.add(SNop(waitState=0, comment=f"delay after pk_{typeStr}"))

    # 4 N-cols are contiguous in row-major D (StrideD1J==1) -> one dwordx2 store.
    module.addComment1(f"buffer_store_dwordx2: write 4 {typeStr} N-cols at fixed M-row (SourceSwap row-major)")
    module.add(BufferStoreB64(
      src=vgpr(vPack+0, 2),
      vaddr=vgpr(vPack+2),
      saddr=sgpr("SrdD", 4),
      soffset=0,
      mubuf=MUBUFModifiers(offen=True, offset12=0, glc=isGlc, slc=isSlc, nt=isNT),
      comment=f"N-wide store d0={blockIdxM} d1={blockIdxN}: 4 consecutive N-cols at fixed M-row"
    ))
    return module

  def _emit16bitSubtilePairedStore(self, addrCalc, sumIdx0: int, sumIdx1: int, prefixOffset: int, tt0: int = 0, blockIdxM: int = 0, blockIdxN: int = 0) -> Module:
    """Emit a paired 16bit store combining sba=0 and sba=1 subtile data.

    Works for both bf16 and fp16 HPA output types.

    sba = subtile block index along A (M dimension).  UseSubtileImpl iterates over
    two subtile groups (sba=0, sba=1) that share the same (tt1, tt0) element
    coordinates but draw from different accumulator registers.  The element list
    therefore contains consecutive pairs with identical (tt1, tt0): sba=0 first
    (even elementIdx), sba=1 second (odd elementIdx).

    Converts 8 f32 accvgprs (4 from sba=0, 4 from sba=1) to 16bit, shuffles them
    across wave halves via ds_bpermute + v_permlane32_swap_b32, then issues
    1 × buffer_store_dwordx4 at the sba=0 element's address.  The cvtVgpr block
    is 2-aligned (64-bit) in KWA so vgprBf16Temp satisfies the dwordx4 alignment.

    Args:
      addrCalc:     AddrCalculation for the sba=0 element.
      sumIdx0:      elementSumIdx for the sba=0 element.
      sumIdx1:      elementSumIdx for the sba=1 element.
      prefixOffset: parentWriter.states.c.startVgprValu (offset into ValuC).
      tt0:          thread-tile M index (same for both sba=0 and sba=1).
    """
    module = Module("16bitSubtilePairedStore")
    isFp16 = self.kernel["ProblemType"]["DestDataType"].isHalf()

    ntd = self.kernel["NonTemporalD"]
    isGlc = bool(ntd & 0x1)
    isSlc = bool(ntd & 0x2)
    isNT  = bool(ntd & 0x4)

    # Reuse cvtVgprStruct.vgprBf16Temp..vgprBf16Inc (+0..+3) as 4 scratch vgprs.
    # The cvtVgpr block is allocated with 2-alignment (64-bit aligned) in KWA so that
    # vgprBf16Temp is at an even VGPR index, satisfying buffer_store_dwordx4's
    # alignment requirement.  The +0..+3 slots are safely overwritten here as pack/perm
    # staging for each pair.
    vPack = self.cvtVgprStruct.vgprBf16Temp  # +0..3: packed 16bit dwords, 2-aligned

    vPermAddr    = self.cvtVgprStruct.vgprPermAddr
    vLGDelta     = self.cvtVgprStruct.vgprLaneGroupDelta
    vAddrScratch = self.cvtVgprStruct.vgprAddrScratch
    addrDVgpr    = addrCalc.addrDVgpr

    typeStr = "fp16" if isFp16 else "bf16"
    VCvtPkF32to16 = VCvtPkF32toFP16 if isFp16 else VCvtPkF32toBF16
    module.addComment1(f"{typeStr} paired dwordx4 store tt0={tt0} (sba=0+sba=1): pack 8 f32 accvgprs -> 4 {typeStr} dwords")

    # Pack sba=0 subtile: ValuC+sumIdx0+{0,1} → vPack+0; ValuC+sumIdx0+{2,3} → vPack+1
    # Pack sba=1 subtile: ValuC+sumIdx1+{0,1} → vPack+2; ValuC+sumIdx1+{2,3} → vPack+3
    def vc(sumIdx, vi):
      idx = sumIdx + vi - prefixOffset
      return vgpr("ValuC+" + str(idx))

    def packF32pair(dst, src0, src1, comment):
      """Pack two f32 VGPRs into one dword of two 16bit values."""
      module.add(VCvtPkF32to16(dst=vgpr(dst), src0=src0, src1=src1, comment=f"{comment} -> {typeStr}"))

    packF32pair(vPack+0, vc(sumIdx0, 0), vc(sumIdx0, 1), f"sba=0 tt0={tt0}[0:1]")
    packF32pair(vPack+1, vc(sumIdx0, 2), vc(sumIdx0, 3), f"sba=0 tt0={tt0}[2:3]")
    packF32pair(vPack+2, vc(sumIdx1, 0), vc(sumIdx1, 1), f"sba=1 tt0={tt0}[0:1]")
    packF32pair(vPack+3, vc(sumIdx1, 2), vc(sumIdx1, 3), f"sba=1 tt0={tt0}[2:3]")

    # Compute adjusted D address into vgprAddrScratch while ds_bpermute results are in-flight.
    # addrDVgpr holds the M-byte offset in bpeCexternal units; scale to bpeCexternalGSU1
    # (16bit=2 bytes) then add lane_group*8 so the dwordx4 store lands at the correct row.
    # addrDVgpr and vgprPermAddr are left unchanged — vgprAddrScratch is dedicated scratch
    # for this purpose so no restore is needed.
    bpeCurr = self.parentWriter.states.bpeCexternal
    bpeDest = self.parentWriter.states.bpeCexternalGSU1
    globalOffset = addrCalc.globalOffset * bpeDest // bpeCurr
    addrScaleShift = int(log2(bpeCurr // bpeDest)) if bpeCurr > bpeDest else 0

    useAlign8 = self.parentWriter.states.storeAlign8

    # Row-major D (UseInitialStridesCD): M (index0) is the strided dim.  addrDVgpr already
    # carries coord0*StrideD0 (folded by the generic AsmAddressCalculation), but the
    # lane-group M offset (vLGDelta = LG*4*bpe) is column-major and the 8 packed M-rows are
    # no longer contiguous -- they are StrideD0*bpe apart -- so the wide dwordx4 degrades to
    # a per-M-row b16 scatter (same shape as _emitFusedA2APushStore's recv scatter).
    strideD0 = self.parentWriter.strideRef('D', 0)
    rowMajorD = not self.parentWriter.isConstUnitStride(strideD0)

    # vLGDelta is precomputed once per batch and read by every store in the batch, so it must
    # stay intact.  The row-major path needs a scratch VGPR for the lane-group scaling and the
    # b16 hi-half extraction; take one from the pool (freed at the end of this store).
    vRMTmp = self.parentWriter.vgprPool.checkOut(1, tag="subtilePairedRowMajorTmp") if rowMajorD else -1

    def emitAddrWhilePermuting(module):
      """Callback emitted between ds_bpermute and s_waitcnt lgkmcnt(0).

      The ds_bpermute has ~88 cycles of LDS latency.  We overlap SALU/VALU
      work in this window to hide the cost:
        1. Compute the adjusted D store address (VALU).
        2. Compute the exec mask for partial M/N blocks (SALU) — this
           suppresses OOB lanes at tile boundaries without adding any
           latency to the critical path.
      `module` is the permute Module, so instructions emitted here land
      between the ds_bpermute issue and the s_waitcnt that consumes results."""
      if not rowMajorD:
        if addrScaleShift:
          module.add(VLShiftRightB32(dst=vgpr(vAddrScratch), shiftHex=addrScaleShift,
                                     src=vgpr(addrDVgpr), comment=f"scale addrDVgpr bpe {bpeCurr}->{bpeDest}"))
          module.add(VAddU32(dst=vgpr(vAddrScratch), src0=vgpr(vAddrScratch), src1=vgpr(vLGDelta),
                             comment="adjusted D addr = scaled addrDVgpr + lane_group*8"))
        else:
          module.add(VAddU32(dst=vgpr(vAddrScratch), src0=vgpr(addrDVgpr), src1=vgpr(vLGDelta),
                             comment="adjusted D addr = addrDVgpr + lane_group*8"))
      else:
        # Row-major: base = scaled addrDVgpr (already carries coord0*StrideD0) plus the
        # lane-group M offset = LG*4 * StrideD0 * bpe (the column-major vLGDelta term, which
        # is LG*4*bpe, rescaled from bpe to StrideD0*bpe).
        if addrScaleShift:
          module.add(VLShiftRightB32(dst=vgpr(vAddrScratch), shiftHex=addrScaleShift,
                                     src=vgpr(addrDVgpr), comment=f"scale addrDVgpr bpe {bpeCurr}->{bpeDest}"))
        else:
          module.add(VMovB32(dst=vgpr(vAddrScratch), src=vgpr(addrDVgpr), comment="base D addr (row-major)"))
        # lane_group element count = (vLGDelta = LG*4*bpe) >> log2(bpe) = LG*4; keep vLGDelta
        # intact (shared per batch) by staging into vRMTmp.  tmpS23 is used here for the
        # StrideD0*bpe product because the align8 mask below reuses tmpS01/tmpS23 last, so
        # the scatter loop recomputes the step into tmpS01 after align8 rather than relying
        # on a value produced before it.
        module.add(VLShiftRightB32(dst=vgpr(vRMTmp), shiftHex=int(log2(bpeDest)), src=vgpr(vLGDelta),
                                   comment="lane_group*4 (M-row element count) = vLGDelta >> log2(bpe)"))
        module.add(SMulI32(dst=sgpr(self.tmpS23), src0=strideD0, src1=bpeDest,
                           comment="StrideD0*bpe (row-major M-row byte stride)"))
        module.add(VMulLOU32(dst=vgpr(vRMTmp), src0=vgpr(vRMTmp), src1=sgpr(self.tmpS23),
                             comment="lane_group_M_off = LG*4 * StrideD0 * bpe"))
        module.add(VAddU32(dst=vgpr(vAddrScratch), src0=vgpr(vAddrScratch), src1=vgpr(vRMTmp),
                           comment="row-major D addr = scaled addrDVgpr + lane_group_M_off"))
      if useAlign8:
        self._emitAlign8ExecMask(module, self.tmpS01, self.tmpS23, blockIdxM, blockIdxN,
                                 mGuardOffset=2, rowScaleShift=1)

    module.add(self._emitSubtilePackedPermute(vPack, vPermAddr, addrWhilePermuting=emitAddrWhilePermuting))

    if useAlign8:
      tmpS = self.tmpS01
      module.add(self.getEdgeMovInstType()(EXEC(), sgpr(tmpS, self.laneSGPRC), "apply exec mask"))

    if not rowMajorD:
      # Column-major: 8 M-rows contiguous -> one wide dwordx4 store.
      module.addComment1("buffer_store_dwordx4: write 8 16bit values (4 dwords, 2-aligned src)")
      module.add(BufferStoreB128(
        src=vgpr(vPack, 4),
        vaddr=vgpr(vAddrScratch),
        saddr=sgpr("SrdD", 4),
        soffset=0,
        mubuf=MUBUFModifiers(offen=True, offset12=globalOffset, glc=isGlc, slc=isSlc, nt=isNT),
        comment=f"16bit paired dwordx4 store tt0={tt0},{tt0+1}"
      ))
    else:
      # Row-major: 8 M-rows StrideD0*bpe apart -> per-M-row b16 scatter.  vPack holds the
      # 8 packed 16bit values as 4 dwords: dword d = M-rows (2d, 2d+1) lo/hi.  Rows 0..3 are
      # sba=0, rows 4..7 are sba=1 (the wide store wrote them as 8 contiguous M-rows, so the
      # sba=1 block sits 4 M-rows above sba=0).  globalOffset (coordOffset0) is already in
      # addrDVgpr, so offset12=0.  Recompute StrideD0*bpe into tmpS01 here because align8
      # reuses tmpS01/tmpS23 during the permute callback.
      module.addComment1(f"buffer_store_b16 x8: scatter 8 {typeStr} M-rows StrideD0*bpe apart (row-major paired)")
      module.add(SMulI32(dst=sgpr(self.tmpS01), src0=strideD0, src1=bpeDest,
                         comment="StrideD0*bpe (M-row byte step)"))
      for row in range(8):
        srcVgpr = vPack + (row // 2)
        if row % 2 == 1:  # hi 16bit half of the packed dword
          module.add(VLShiftRightB32(dst=vgpr(vRMTmp), shiftHex=16, src=vgpr(srcVgpr),
                                     comment=f"M-row+{row}: hi 16bit -> lo"))
          storeSrc = vRMTmp
        else:
          storeSrc = srcVgpr
        module.add(BufferStoreB16(
          src=vgpr(storeSrc),
          vaddr=vgpr(vAddrScratch),
          saddr=sgpr("SrdD", 4),
          soffset=0,
          mubuf=MUBUFModifiers(offen=True, offset12=0, glc=isGlc, slc=isSlc, nt=isNT),
          comment=f"paired tt0={tt0},{tt0+1} M-row+{row} (row-major scatter)"
        ))
        if row != 7:
          module.add(VAddU32(dst=vgpr(vAddrScratch), src0=vgpr(vAddrScratch), src1=sgpr(self.tmpS01),
                             comment="advance vaddr by StrideD0*bpe (next M-row)"))

    if useAlign8:
      module.add(self.getEdgeMovInstType()(EXEC(), -1, "restore exec"))

    # WAR hazard: the store reads vPack[0:3] as source operands.
    # The next paired store's v_cvt_pk_bf16_f32 will overwrite vPack.
    # Insert nop to ensure the store has latched its source VGPRs.
    module.add(SNop(waitState=0, comment="1 wait state: WAR hazard between store src and next pack dst"))

    if rowMajorD:
      self.parentWriter.vgprPool.checkIn(vRMTmp)

    return module

  def _emit16bitSubtileScalarStore(self, addrCalc, sumIdx0: int, prefixOffset: int, tt0: int = 0, blockIdxM: int = 0, blockIdxN: int = 0) -> Module:
    """Emit a 16bit store for an orphan subtile element with no partner.

    Used when MIWaveTile[0] is odd and the last sba=0 element has no sba=1
    partner, or when batch boundaries split an (sba=0, sba=1) pair.

    The layout below is specific to the mfma instruction used here: lane l = LG*16 + r
    owns 4 output values at M-rows (LG*4 + 0..3) and a single N-column
    (l % 16 = r = lane_id & 15).  In column-major (row-first in memory) layout
    these 4 values ARE contiguous
    in memory (consecutive M-rows at fixed N-col), so we use 2x buffer_store_dwordx2
    after packing all 4 16bit values into 2 dwords.

    The per-lane vaddr encodes:
      vaddr = (lane_id & 15) * StrideD1J * bpe   [N-col byte offset within wave tile]
            + vLGDelta                            [LG*4 M-rows * bpe = LG*8 bytes]
            + wg0*MT0*bpe                         [workgroup M byte base]
            + waveId0 * waveM_stride * bpe        [M-wave byte offset within WG]
            + waveId1 * waveN_stride * StrideD1J * bpe  [N-wave byte offset]
    and a constant offset12 = globalOffset (encodes d0 M-tile position within wave).

    The SRD base encodes only wg1*MT1*StrideD1J*bpe (N workgroup offset).
    The M workgroup offset (wg0*MT0*bpe) and wave-within-WG offsets must be
    included in the vaddr explicitly.

    Args:
      addrCalc:     AddrCalculation for the element.
      sumIdx0:      elementSumIdx for the element.
      prefixOffset: parentWriter.states.c.startVgprValu (offset into ValuC).
    """
    module = Module("16bitSubtileScalarStore")
    isFp16 = self.kernel["ProblemType"]["DestDataType"].isHalf()

    ntd = self.kernel["NonTemporalD"]
    isGlc = bool(ntd & 0x1)
    isSlc = bool(ntd & 0x2)
    isNT  = bool(ntd & 0x4)

    # Scratch vgprs from the cvtVgprStruct block (overwritten each call):
    #   vPack+0  : 16bit packed dword (vc=0,1)
    #   vPack+1  : wave ID scratch / 16bit packed dword (vc=2,3)
    #   vPack+2  : per-lane vaddr (N-col byte offset + M offsets)
    #   vPack+3  : temp for N-col byte offset computation
    vPack    = self.cvtVgprStruct.vgprBf16Temp
    vLGDelta = self.cvtVgprStruct.vgprLaneGroupDelta  # LG*4*bpe = LG*8 bytes (pre-computed)

    # addrCalc.globalOffset was computed with bpeCexternal (may be 4 for _GlobalAccumulation kernels),
    # but the 16bit orphan store always targets the final 16bit output (bpeCexternalGSU1=2).
    bpeCurr = self.parentWriter.states.bpeCexternal
    bpe     = self.parentWriter.states.bpeCexternalGSU1  # always 2 for 16bit dest
    globalOffset = addrCalc.globalOffset * bpe // bpeCurr

    def vc(vi):
      idx = sumIdx0 + vi - prefixOffset
      return vgpr("ValuC+" + str(idx))

    # Derive the D-stride sgpr name (e.g. "StrideDJ") the same way incrementToNextRow does.
    packedC1  = self.kernel["PackedC1IndicesX"]
    indexChar = self.parentWriter.states.indexChars[packedC1[0]]
    strideD1J = "StrideD%s" % indexChar

    # Row-major D (UseInitialStridesCD): M (index0) is the strided dim, so every M-row
    # advance is StrideD0*bpe apart instead of the column-major bpe.  strideRef('D',0)
    # returns the StrideD0 sgpr when UseInitialStridesCD is on and the const string
    # "constStrideD0<idx>" (unit stride) otherwise, so isConstUnitStride is the exact
    # column-major/row-major switch: column-major keeps the original wide store, row-major
    # scales every M offset by StrideD0 and scatters the wide store into per-M-row b16.
    strideD0 = self.parentWriter.strideRef('D', 0)
    rowMajorD = not self.parentWriter.isConstUnitStride(strideD0)

    ws     = self.kernel["WavefrontSize"]
    miwg0  = self.kernel["MIWaveGroup"][0]
    miwg1  = self.kernel["MIWaveGroup"][1]
    matM   = self.kernel["MatrixInstM"]
    matN   = self.kernel["MatrixInstN"]

    typeStr = "fp16" if isFp16 else "bf16"
    VCvtPkF32to16 = VCvtPkF32toFP16 if isFp16 else VCvtPkF32toBF16
    module.addComment1(f"{typeStr} orphan subtile tt0={tt0}: pack 4 M-rows (vc=0..3) at fixed N-col, store as 2x dwordx2")

    # Build per-lane vaddr:
    #   vaddr = (lane_id & 15) * StrideD1J * bpe   [N-col]
    #         + vLGDelta                            [LG*4 M-rows = LG*8 bytes]
    #         + wg0*MT0*bpe                         [M-WG base]
    #         + waveId0*waveM_stride*bpe            [M-wave offset, if miwg0>1]
    #         + waveId1*waveN_stride*StrideD1J*bpe  [N-wave offset, if miwg1>1]
    # The SRD already encodes wg1*MT1*StrideD1J*bpe (N-WG offset).
    tmpS = self.tmpS01

    module.addComment1("compute per-lane orphan vaddr = N_col_off + LG_M_off + wg0_M_off [+ wave offsets]")

    # N-col byte offset: (lane_id & 15) * StrideD1J * bpe.  Unchanged for both layouts:
    # row-major D has StrideD1J==1 (N is the contiguous dim) so this naturally collapses
    # to (lane&15)*bpe; column-major keeps StrideD1J==LDD.
    module.add(VAndB32(dst=vgpr(vPack+2), src0=15, src1=vgpr("Serial"),
                       comment="col_in_wave = lane_id & 15  (N-column index)"))
    module.add(VMulLOU32(dst=vgpr(vPack+3), src0=vgpr(vPack+2), src1=sgpr(strideD1J),
                         comment="col_in_wave * StrideD1J"))
    if bpe == 2:
      module.add(VLShiftLeftB32(dst=vgpr(vPack+2), shiftHex=1, src=vgpr(vPack+3),
                                comment="N_col_off = col_in_wave * StrideD1J * 2"))
    else:
      module.add(VMulLOU32(dst=vgpr(vPack+2), src0=vgpr(vPack+3), src1=bpe,
                           comment="N_col_off = col_in_wave * StrideD1J * bpe"))

    # M byte offsets.  Column-major (row-first): a row is bpe apart, so MT0/vLGDelta/wave
    # terms are pre-scaled by bpe.  Row-major (UseInitialStridesCD): a row is StrideD0*bpe
    # apart, so each M-element count is scaled by StrideD0*bpe instead.  The coordOffset0
    # (globalOffset/offset12) M-tile position must also fold into the vaddr under row-major
    # because offset12 is an immediate and cannot carry the StrideD0 sgpr product.
    if not rowMajorD:
      mt0bpe = self.kernel["MacroTile0"] * bpe

      # Add LG M-row offset: vLGDelta = LG*4*bpe = LG*8 bytes (pre-computed at batch start)
      module.add(VAddU32(dst=vgpr(vPack+2), src0=vgpr(vPack+2), src1=vgpr(vLGDelta),
                         comment="vaddr += LG_M_off (= vLGDelta = LG*4*bpe)"))

      # Add M-WG offset: wg0 * MT0 * bpe
      module.add(SMulI32(dst=sgpr(tmpS), src0=sgpr("WorkGroup0"), src1=mt0bpe,
                         comment="wg0_M_off = WorkGroup0 * MT0 * bpe"))
      module.add(VAddU32(dst=vgpr(vPack+2), src0=vgpr(vPack+2), src1=sgpr(tmpS),
                         comment="vaddr += wg0_M_off"))

      # Add M-wave offset: waveId0 * MIWaveTile[0] * matM * bpe.
      if miwg0 > 1:
        wsLog2 = int(log2(ws))
        waveM_stride_bpe = self.kernel["MIWaveTile"][0] * matM * bpe
        module.add(VLShiftRightB32(dst=vgpr(vPack+3), shiftHex=wsLog2, src=vgpr("Serial"),
                                   comment=f"waveId = Serial >> {wsLog2}"))
        if miwg0 & (miwg0 - 1) == 0:  # power of 2 — use AND mask
          module.add(VAndB32(dst=vgpr(vPack+3), src0=miwg0 - 1, src1=vgpr(vPack+3),
                             comment=f"waveId0 = waveId & {miwg0-1}"))
        else:
          raise NotImplementedError(f"Non-power-of-2 MIWaveGroup[0]={miwg0} not supported in orphan store")
        module.add(SMovB32(dst=sgpr(tmpS), src=waveM_stride_bpe,
                           comment=f"waveM_stride_bpe={waveM_stride_bpe}"))
        module.add(VMulLOU32(dst=vgpr(vPack+3), src0=vgpr(vPack+3), src1=sgpr(tmpS),
                             comment=f"wave_M_off = waveId0 * {waveM_stride_bpe}"))
        module.add(VAddU32(dst=vgpr(vPack+2), src0=vgpr(vPack+2), src1=vgpr(vPack+3),
                           comment="vaddr += wave_M_off"))
    else:
      # Row-major: accumulate the full M element count (LG*4 + wg0*MT0 + waveId0*MIWT0*matM
      # + coordOffset0) into vPack+3, then scale once by StrideD0*bpe and add to vaddr.
      # LG*4 = vLGDelta>>1 (vLGDelta = LG*4*bpe = LG*8, bpe==2 for 16bit dest).
      module.add(VLShiftRightB32(dst=vgpr(vPack+3), shiftHex=int(log2(bpe)), src=vgpr(vLGDelta),
                                 comment="LG*4 (M-row element count for this lane group)"))
      # + wg0*MT0 (M workgroup base in elements)
      module.add(SMulI32(dst=sgpr(tmpS), src0=sgpr("WorkGroup0"), src1=self.kernel["MacroTile0"],
                         comment="wg0_M_elems = WorkGroup0 * MT0"))
      module.add(VAddU32(dst=vgpr(vPack+3), src0=vgpr(vPack+3), src1=sgpr(tmpS),
                         comment="M elems += wg0_M_elems"))
      # + coordOffset0 (this element's M-tile position; folded in from offset12)
      if addrCalc.coordOffset0:
        module.add(SMovB32(dst=sgpr(tmpS), src=addrCalc.coordOffset0,
                           comment=f"coordOffset0={addrCalc.coordOffset0}"))
        module.add(VAddU32(dst=vgpr(vPack+3), src0=vgpr(vPack+3), src1=sgpr(tmpS),
                           comment="M elems += coordOffset0"))
      # + waveId0*MIWaveTile[0]*matM (M-wave offset in elements)
      if miwg0 > 1:
        wsLog2 = int(log2(ws))
        waveM_stride = self.kernel["MIWaveTile"][0] * matM
        module.add(VLShiftRightB32(dst=vgpr(vPack+0), shiftHex=wsLog2, src=vgpr("Serial"),
                                   comment=f"waveId = Serial >> {wsLog2}"))
        if miwg0 & (miwg0 - 1) == 0:  # power of 2 — use AND mask
          module.add(VAndB32(dst=vgpr(vPack+0), src0=miwg0 - 1, src1=vgpr(vPack+0),
                             comment=f"waveId0 = waveId & {miwg0-1}"))
        else:
          raise NotImplementedError(f"Non-power-of-2 MIWaveGroup[0]={miwg0} not supported in orphan store")
        module.add(SMovB32(dst=sgpr(tmpS), src=waveM_stride,
                           comment=f"waveM_stride(elems)={waveM_stride}"))
        module.add(VMulLOU32(dst=vgpr(vPack+0), src0=vgpr(vPack+0), src1=sgpr(tmpS),
                             comment=f"wave_M_elems = waveId0 * {waveM_stride}"))
        module.add(VAddU32(dst=vgpr(vPack+3), src0=vgpr(vPack+3), src1=vgpr(vPack+0),
                           comment="M elems += wave_M_elems"))
      # Scale total M element count by StrideD0*bpe and add to the N-col byte offset.
      module.add(SMulI32(dst=sgpr(tmpS), src0=strideD0, src1=bpe,
                         comment="StrideD0*bpe (row-major M-row byte stride)"))
      module.add(VMulLOU32(dst=vgpr(vPack+3), src0=vgpr(vPack+3), src1=sgpr(tmpS),
                           comment="M_byte_off = M_elems * StrideD0 * bpe"))
      module.add(VAddU32(dst=vgpr(vPack+2), src0=vgpr(vPack+2), src1=vgpr(vPack+3),
                         comment="vaddr += M_byte_off (row-major)"))

    # Add N-wave offset: waveId1 * MIWaveTile[1] * matN * StrideD1J * bpe.
    if miwg1 > 1:
      wsLog2 = int(log2(ws))
      waveN_stride_bpe = self.kernel["MIWaveTile"][1] * matN * bpe
      module.add(VLShiftRightB32(dst=vgpr(vPack+3), shiftHex=wsLog2, src=vgpr("Serial"),
                                 comment=f"waveId = Serial >> {wsLog2}"))
      if miwg0 & (miwg0 - 1) == 0:  # miwg0 is power of 2
        module.add(VLShiftRightB32(dst=vgpr(vPack+3), shiftHex=int(log2(miwg0)),
                                   src=vgpr(vPack+3),
                                   comment=f"waveId1 = waveId / {miwg0}"))
      else:
        raise NotImplementedError(f"Non-power-of-2 MIWaveGroup[0]={miwg0} not supported in orphan store")
      module.add(SMovB32(dst=sgpr(tmpS), src=waveN_stride_bpe,
                         comment=f"waveN_stride_bpe={waveN_stride_bpe}"))
      module.add(VMulLOU32(dst=vgpr(vPack+3), src0=vgpr(vPack+3), src1=sgpr(tmpS),
                           comment=f"waveId1 * {waveN_stride_bpe}"))
      module.add(VMulLOU32(dst=vgpr(vPack+3), src0=vgpr(vPack+3), src1=sgpr(strideD1J),
                           comment=f"wave_N_off = waveId1 * {waveN_stride_bpe} * StrideD1J"))
      module.add(VAddU32(dst=vgpr(vPack+2), src0=vgpr(vPack+2), src1=vgpr(vPack+3),
                         comment="vaddr += wave_N_off"))

    # Pack all 4 16bit values (consecutive M-rows at fixed N-col) into 2 dwords.
    # vc=0 → M-row+0 (lo16 of dword0), vc=1 → M-row+1 (hi16 of dword0)
    # vc=2 → M-row+2 (lo16 of dword1), vc=3 → M-row+3 (hi16 of dword1)
    #
    module.add(VCvtPkF32to16(dst=vgpr(vPack+0), src0=vc(0), src1=vc(1), comment=f"M-row+0/+1 -> {typeStr}"))
    module.add(VCvtPkF32to16(dst=vgpr(vPack+1), src0=vc(2), src1=vc(3), comment=f"M-row+2/+3 -> {typeStr}"))
    module.add(SNop(waitState=0, comment=f"delay after pk_{typeStr}"))

    useAlign8 = self.parentWriter.states.storeAlign8
    if useAlign8:
      self._emitAlign8ExecMask(module, self.tmpS01, self.tmpS23, blockIdxM, blockIdxN,
                               mGuardOffset=1, rowScaleShift=2)
      module.add(self.getEdgeMovInstType()(EXEC(), sgpr(self.tmpS01, self.laneSGPRC), "apply exec mask"))

    if not rowMajorD:
      # Column-major: the 4 M-rows are bpe apart (contiguous in memory) -> one wide store.
      module.addComment1(f"buffer_store_b64: write 4 {typeStr} M-rows at fixed N-col (orphan subtile)")
      module.add(BufferStoreB64(
        src=vgpr(vPack+0, 2),
        vaddr=vgpr(vPack+2),
        saddr=sgpr("SrdD", 4),
        soffset=0,
        mubuf=MUBUFModifiers(offen=True, offset12=globalOffset, glc=isGlc, slc=isSlc, nt=isNT),
        comment=f"orphan tt0={tt0} vc=0..3: 4 consecutive M-rows at fixed N-col"
      ))
    else:
      # Row-major: the 4 M-rows are StrideD0*bpe apart, so the wide store degrades to a
      # per-M-row b16 scatter (same shape as _emitFusedA2APushStore's recv scatter, only
      # the step is StrideD0*bpe instead of n_shard*bpe).  offset12 is 0 because the
      # coordOffset0 M-tile position was already folded into the vaddr above.  vPack+2
      # holds M-row 0's byte offset; step by StrideD0*bpe after each store.
      module.addComment1(f"buffer_store_b16 x4: scatter 4 {typeStr} M-rows StrideD0*bpe apart (row-major orphan)")
      module.add(SMulI32(dst=sgpr(tmpS), src0=strideD0, src1=bpe,
                         comment="StrideD0*bpe (M-row byte step)"))
      for row in range(4):
        srcVgpr = vPack + (row // 2)
        if row % 2 == 1:  # hi 16bit half of the packed dword
          module.add(VLShiftRightB32(dst=vgpr(vPack+3), shiftHex=16, src=vgpr(srcVgpr),
                                     comment=f"M-row+{row}: hi 16bit -> lo"))
          storeSrc = vPack + 3
        else:
          storeSrc = srcVgpr
        module.add(BufferStoreB16(
          src=vgpr(storeSrc),
          vaddr=vgpr(vPack+2),
          saddr=sgpr("SrdD", 4),
          soffset=0,
          mubuf=MUBUFModifiers(offen=True, offset12=0, glc=isGlc, slc=isSlc, nt=isNT),
          comment=f"orphan tt0={tt0} M-row+{row} (row-major scatter)"
        ))
        if row != 3:
          module.add(VAddU32(dst=vgpr(vPack+2), src0=vgpr(vPack+2), src1=sgpr(tmpS),
                             comment="advance vaddr by StrideD0*bpe (next M-row)"))

    if useAlign8:
      module.add(self.getEdgeMovInstType()(EXEC(), -1, "restore exec"))

    return module

  def _emitAtomicAdd(self, module: Module):
    ########################################
    # first attempt write
    module.addComment1("issue first atomic writes")
    for elementIdx in range(len(self.batchElements)):
      addrCalc = self.ss.elementAddr[elementIdx]
      mask     = self.ss.elementMask[elementIdx]

      # apply in-bounds exec mask
      if self.edge:
        module.add(self.getEdgeMovInstType()(EXEC(), sgpr(mask, self.laneSGPRC), "sgprs -> exec (before atomic)"))

      for avi in range(0, self.gwvw // self.atomicW):
        sumIdxV = self.ss.elementSumIdx[elementIdx] + avi
        newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
        if self.parentWriter.do["GlobalWrite"]:
          if self.kernel["BufferStore"]:
            module.add(BufferAtomicAddF32(vgpr("ValuC+%u"%newSumIdxV), \
                         vgpr(addrCalc.addrDVgpr,1), \
                         sgpr("SrdD", 4), \
                         0,
                         MUBUFModifiers(offen=True, offset12=addrCalc.globalOffset),
                         "attempt write avi=%u" % (avi)))
          else:
            pass # TODO:

    if self.edge:
      module.add(self.getEdgeMovInstType()(EXEC(), -1, "full mask -> exec"))

  def _emitCasAdd(self, module: Module):
    # TODO for atomic GWVW:
    #  - Use vi to compute addresses, sumIdx.
    #  - Need a solution for the mask.  Can move to all buffer or can fix?
    element = self.batchElements[0]
    d1 = element[0]
    d0 = element[1]
    vc1 = element[2]
    vc0 = element[3]
    labels = self.parentWriter.labels
    labelString = "Global_Write%s%s_%u_%u_%u_%u" % ("_Beta" if self.beta else "", "_Edge" if self.edge else "", vc0, vc1, d0, d1 )
    labelComment = "Global_Write (Beta) (Edge) vc0 vc1 d0 d1"
    label = Label(labels.getName(labelString), labelComment)
    labelString += "_EarlyExit"
    labelAfterAtomicLoop = Label(labels.getName(labelString), labelComment)

    ########################################
    # wait for batched load
    # TODO - we are always atomic here?
    module.add(SWaitCnt(vlcnt=0, vscnt=0, comment="wait C (atomic)"))
    ########################################
    # first attempt write
    module.addComment1("issue first atomic writes")
    for elementIdx, element in enumerate(self.batchElements):
      addrCalc = self.ss.elementAddr[elementIdx]
      mask = self.ss.elementMask[elementIdx]

      # apply in-bounds exec mask
      if self.edge:
        module.add(self.getEdgeMovInstType()(EXEC(), sgpr(mask, self.parentWriter.states.laneSGPRCount), "sgprs -> exec (before atomic)"))

      for avi in range(0, self.gwvw//self.atomicW):
        dataV = self.ss.elementData[elementIdx] + int(avi*self.ss.cfg.numVgprsPerDataPerVI)
        sumIdxV = self.ss.elementSumIdx[elementIdx] + avi
        ## number of src[s]/dst[s] register for DGEMM / SGEMM HGEMM
        vgprCnt = 2 if self.kernel["ProblemType"]["DestDataType"].isDouble() else 1
        if self.kernel["ProblemType"]["DestDataType"].numRegisters() < 1 and not self.kernel["_GlobalAccumulation"]:
          sumIdxV //= 2
        if self.kernel["ProblemType"]["DestDataType"].isDouble(): sumIdxV = sumIdxV * 2
        newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
        bpm = self.parentWriter.states.bpeCexternal * self.atomicW
        # Calculate vgpr Index for 32-bit/64-bit instruction
        # DGEMM use SRCS[2] register
        vgprIdx = 1*(bpm//4)
        # for atomic, data[1] = original c, data[0] = new c
        module.add(self._chooseAddForAtomic(self.kernel, \
                  vgpr(dataV+0,vgprCnt), vgpr(dataV+1*vgprIdx,vgprCnt), vgpr("ValuC+%u"%newSumIdxV,vgprCnt), \
                  "desired value avi=%u"%avi))

        # attempt write
        atomicDestVgpr = dataV if self.kernel["BufferStore"] else dataV+2
        if self.parentWriter.do["GlobalWrite"]:
          if self.kernel["BufferStore"]:
            # use cmpswap_x2 for DGEMM in CAS loop
            if self.kernel["ProblemType"]["DestDataType"].isDouble():
              module.add(BufferAtomicCmpswapB64(vgpr(dataV,4), \
                              vgpr(addrCalc.addrDVgpr,1), \
                              sgpr("SrdD", 4),  \
                              0,
                              MUBUFModifiers(offen=True, offset12=addrCalc.globalOffset, glc=True),
                              "attempt write avi=%u"%(avi)))
            else:
            # use cmpswap for SGEMM in CAS loop
              module.add(BufferAtomicCmpswapB32(vgpr(dataV,2), \
                           vgpr(addrCalc.addrDVgpr,1), \
                           sgpr("SrdD", 4), \
                           0, \
                           MUBUFModifiers(offen=True, offset12=addrCalc.globalOffset, glc=True), \
                           "attempt write avi=%u"%(avi)))
          else:
            module.add(FlatAtomicCmpswapB32(vgpr(atomicDestVgpr), \
                                            vgpr(addrCalc.addrDVgpr,2), \
                                            vgpr(dataV,2),
                                            FLATModifiers(glc=True),
                                            "attempt write"))
        else:
            # Fake successful CAS swap
            module.add(VMovB32(vgpr(atomicDestVgpr), vgpr(dataV+1), "Fake successful CAS" ))

    ########################################
    # wait for first attempt write
    module.add(SWaitCnt(vlcnt=0, vscnt=0, comment="wait for atomic writes"))
    ########################################
    # check first attempt
    module.addComment1("check success of writes, update masks")
    for elementIdx, element in enumerate(self.batchElements):
      mask = self.ss.elementMask[elementIdx]

      # calculate new masks
      if self.edge:
        module.add(self.getEdgeMovInstType()(EXEC(), sgpr(mask, self.laneSGPRC), "sgprs -> exec"))
        for avi in range(0, self.gwvw // self.atomicW):
          dataV = self.ss.elementData[elementIdx] + int(avi * self.ss.cfg.numVgprsPerDataPerVI)
          atomicDestVgpr = dataV if self.kernel["BufferStore"] else dataV+2
          # need to apply element mask before comparison
          # so that all valid lanes are doing the cmp
          if avi == 0:
            # use u64 for DGEMM
            if self.kernel["ProblemType"]["DestDataType"].isDouble():
              module.add(VCmpNeU64(sgpr(self.tmpS01, self.laneSGPRC), vgpr(atomicDestVgpr,2), \
                  vgpr(dataV+2,2), comment="c read during atomic == c read during prior load (avi=%u, first)"%avi))
            else:
              module.add(VCmpNeU32(sgpr(self.tmpS01, self.laneSGPRC), vgpr(atomicDestVgpr), \
                  vgpr(dataV+1), comment="c read during atomic == c read during prior load (avi=%u, first)"%avi))
          else:
            if self.kernel["ProblemType"]["DestDataType"].isDouble():
              module.add(VCmpNeU64(sgpr(self.tmpS23, self.laneSGPRC), vgpr(atomicDestVgpr,2), \
                  vgpr(dataV+2,2), comment="c read during atomic != c read during prior load"))
            else:
              module.add(VCmpNeU32(sgpr(self.tmpS23, self.laneSGPRC), vgpr(atomicDestVgpr), \
                  vgpr(dataV+1), comment="c read during atomic == c read during prior load (avi=%u)"%avi))
            module.add(self.getEdgeOrInstType()(sgpr(self.tmpS01, self.laneSGPRC), \
                  sgpr(self.tmpS01, self.laneSGPRC), sgpr(self.tmpS23, self.laneSGPRC), "combine with tmp mask"))

        module.add(self.getEdgeAndInstType()(sgpr(mask, self.laneSGPRC), sgpr(self.tmpS01, self.laneSGPRC), sgpr(mask,self.laneSGPRC), "inBounds & must try again"))

      else:
        for avi in range(0, self.gwvw//self.atomicW):
          dataV = self.ss.elementData[elementIdx] + int(avi*self.ss.cfg.numVgprsPerDataPerVI)
          atomicDestVgpr = dataV if self.kernel["BufferStore"] else dataV+2
          if self.kernel["ProblemType"]["DestDataType"].isDouble():
            module.add(VCmpNeU64(sgpr(mask, self.laneSGPRC), vgpr(atomicDestVgpr,2), \
                vgpr(dataV+2,2), comment="c read during atomic != c read during prior load"))
          else:
            module.add(VCmpNeU32(sgpr(mask, self.laneSGPRC), vgpr(atomicDestVgpr), \
                vgpr(dataV+1), comment="c read during atomic != c read during prior load"))

    # or masks together to check early exit
    module.addComment1("or masks to check for exit")
    module.add(self.getEdgeMovInstType()(sgpr(self.tmpS01, self.laneSGPRC), 0, "empty mask"))
    for elementIdx in range(0, len(self.batchElements)):
      mask = self.ss.elementMask[elementIdx]
      module.add(self.getEdgeOrInstType()(sgpr(self.tmpS01, self.laneSGPRC), sgpr(mask, self.laneSGPRC), sgpr(self.tmpS01, self.laneSGPRC), "or to add threads"))
    module.add(self.getSOrSaveExecType()(sgpr(self.tmpS23,self.laneSGPRC), sgpr(self.tmpS01,self.laneSGPRC), "apply combined mask"))
    module.add(SCBranchExecZ(labelAfterAtomicLoop.getLabelName(), "if exec is zero skip loop"))

    # begin atomic loop
    module.addComment1("atomic CAS loop")
    module.add(label)

    module.addComment1("apply updated masks and issue writes again")
    for elementIdx in range(0, len(self.batchElements)):
      addrCalc = self.ss.elementAddr[elementIdx]
      addr = addrCalc.addrDVgpr
      mask = self.ss.elementMask[elementIdx]
      vgprCnt = 2 if self.kernel["ProblemType"]["DestDataType"].isDouble() else 1   # number of registers for f32/f64
      bpm = self.parentWriter.states.bpeCexternal * self.atomicW
      vgprIdx = 1*(bpm//4)   # index register

      for avi in range(0, self.gwvw//self.atomicW):
        dataV = self.ss.elementData[elementIdx] + int(avi*self.ss.cfg.numVgprsPerDataPerVI)
        atomicDestVgpr = dataV if self.kernel["BufferStore"] else dataV+2
        sumIdxV = self.ss.elementSumIdx[elementIdx] + avi
        if self.kernel["ProblemType"]["DestDataType"].numRegisters() < 1 and not self.kernel["_GlobalAccumulation"]:
          sumIdxV //= 2
        if self.kernel["ProblemType"]["DestDataType"].isDouble():
          sumIdxV =  sumIdxV * 2
        newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu

        # apply mask for element
        module.add(self.getEdgeMovInstType()(EXEC(), sgpr(mask,self.laneSGPRC), "must try again"))
        if self.kernel["ProblemType"]["DestDataType"].isDouble():
          #64-bit C val move by 2 32-bit instructions
          module.add(VMovB32(vgpr(dataV+2), vgpr(atomicDestVgpr), "dataV+2 = tmp (new original C)" ))
          module.add(VMovB32(vgpr(dataV+3), vgpr(atomicDestVgpr+1), "dataV+3 = tmp (new original C)" ))
        else:
          module.add(VMovB32(dst=vgpr(dataV+1), src=vgpr(atomicDestVgpr), comment="dataV+1 = tmp (new original C)" ))
        module.add(self._chooseAddForAtomic(self.kernel, \
                        vgpr(dataV+0,vgprCnt), vgpr(dataV+1*vgprIdx,vgprCnt), vgpr("ValuC+%u"%newSumIdxV,vgprCnt), \
                        "newC = rC + originalC"))
        if self.parentWriter.do["GlobalWrite"]:
          if self.kernel["BufferStore"]:
            # Using no-ret version here?
            # cmpswap_x2 for DGEMM
            if self.kernel["ProblemType"]["DestDataType"].isDouble():
              module.add(BufferAtomicCmpswapB64(vgpr(dataV,4), \
                          vgpr(addr,1), \
                          sgpr("SrdD", 4), \
                          0,
                          MUBUFModifiers(offen=True, offset12=addrCalc.globalOffset, glc=True,),
                          "try again"))
            else:
              module.add(BufferAtomicCmpswapB32(
                          vgpr(dataV,2), \
                          vgpr(addr,1), \
                          sgpr("SrdD", 4), \
                          0,
                          MUBUFModifiers(offen=True, offset12=addrCalc.globalOffset, glc=True),
                          "try again"))
          else:
            module.add(FlatAtomicCmpswapB32(vgpr(atomicDestVgpr), \
                                            vgpr(addr,2), \
                                            vgpr(dataV,2), \
                                            FLATModifiers(glc=True), \
                                            "try again"))

    # wait for batched write
    module.add(SWaitCnt(vlcnt=0, vscnt=0, comment="wait for atomic writes"))
    # check batched write success
    module.addComment1("apply masks and check for success")
    for elementIdx in range(0, len(self.batchElements)):
      data = self.ss.elementData[elementIdx]
      mask = self.ss.elementMask[elementIdx]
      for avi in range(0, self.gwvw//self.atomicW):
        dataV = self.ss.elementData[elementIdx] + int(avi*self.ss.cfg.numVgprsPerDataPerVI)
        atomicDestVgpr = dataV if self.kernel["BufferStore"] else dataV+2

        # apply mask for element
        module.add(self.getEdgeMovInstType()(EXEC(), sgpr(mask,self.laneSGPRC), "must try again"))

        # compare success
        if self.kernel["ProblemType"]["DestDataType"].isDouble():
          module.add(VCmpNeU64(sgpr(self.tmpS01,self.laneSGPRC), vgpr(data+2,2), vgpr(atomicDestVgpr,2), \
              comment="c read during atomic != c read during prior load"))
        else:
          module.add(VCmpNeU32(sgpr(self.tmpS01,self.laneSGPRC), vgpr(data+1), vgpr(atomicDestVgpr), \
              comment="c read during atomic == c read during prior load"))
        # update element mask
        module.add(self.getEdgeAndInstType()(sgpr(mask,self.laneSGPRC), sgpr(self.tmpS01,self.laneSGPRC), sgpr(mask,self.laneSGPRC), "inBounds & must try again"))

    # or masks together
    module.addComment1("or masks to check for exit")
    module.add(self.getEdgeMovInstType()(sgpr(self.tmpS01,self.laneSGPRC), 0, "empty mask"))
    for elementIdx in range(0, len(self.batchElements)):
      mask = self.ss.elementMask[elementIdx]
      module.add(self.getEdgeOrInstType()(sgpr(self.tmpS01,self.laneSGPRC), sgpr(mask,self.laneSGPRC), sgpr(self.tmpS01,self.laneSGPRC), "or to add threads"))

    # apply combined masks and exit
    module.add(self.getSOrSaveExecType()(sgpr(self.tmpS23, self.laneSGPRC), sgpr(self.tmpS01,self.laneSGPRC), "apply combined mask"))
    module.add(SCBranchExecNZ(label.getLabelName(), "try again if not complete"))
    module.add(labelAfterAtomicLoop)
    module.add(self.getEdgeMovInstType()(EXEC(), -1, "full mask -> exec"))

  def _checkAtomicPreconditions(self) -> bool:
    if self.atomic:
      # all kinds of code relies on this assumption:
      if self.atomicW > self.gwvw:
        return False

      if (self.kernel["ProblemType"]["DataType"].isHalf() or self.kernel["ProblemType"]["DataType"].isBFloat16()) \
        and not self.kernel["_GlobalAccumulation"]:
        return self.atomicW >= 2
    return True

  def _storeSyncOpt(self, module: Module):
    module.add(SSleep(self.kernel["StoreSyncOpt"] - 1, "optimization: sync and wait"))
    module.add(SBarrier())

  def _applyAlpha(self, kernel, gwvw, elementSumIdx, elementIdx, tmpS01, usePK=False):
    module = Module("applyAlpha")

    if kernel["_GlobalAccumulation"] == 'MultipleBuffer':
      return module

    if self.parentWriter.do["ApplyAlpha"]:
      for vi in range(0, gwvw):
        sumIdxV = elementSumIdx[elementIdx] + vi

        if kernel["ProblemType"]["ComputeDataType"].isHalf() and not kernel["ProblemType"]["HighPrecisionAccumulate"]:
          # (h,h,h,h,h,h), internal alpha is f16 (2-16bits)
          if sumIdxV%2:
            newSumIdx = sumIdxV // 2 - self.parentWriter.states.c.startVgprValu
            module.add(VMulPKF16(dst=vgpr("ValuC+%u"%(newSumIdx)), src0=sgpr("Alpha"), src1=vgpr("ValuC+%u"%(newSumIdx)), comment="*= alpha sumIdx=%u vi=%u"%(elementSumIdx[elementIdx], vi)))

        # Int8 (TODO- Int8x4 not checked, but should be OK)
        elif kernel["ProblemType"]["ComputeDataType"].isInt32():
          newSumIdx = sumIdxV - self.parentWriter.states.c.startVgprValu
          # below assume we use v_mul_lo_u32. Could also use v_mul_i32_i24.
          # module.add(VMulI32I24(dst=vgpr("ValuC+%u"%newSumIdx), src0=sgpr("Alpha"), src1=vgpr("ValuC+%u"%newSumIdx), comment="*= alpha" )_
          module.add(VMulLOU32(dst=vgpr("ValuC+%u"%newSumIdx), src0=sgpr("Alpha"), src1=vgpr("ValuC+%u"%newSumIdx), comment="*= alpha" ))
          if usePK:
            module.add(VMulLOU32(dst=vgpr("ValuC+%u"%(newSumIdx+1)), src0=sgpr("Alpha"), src1=vgpr("ValuC+%u"%(newSumIdx+1)), comment="*= alpha" ))
          if self.parentWriter.db["ForceExpectedValue"]:
            module.add(VMovB32(dst=vgpr("ValuC+%u"%newSumIdx), src=self.parentWriter.db["ValueCExpectedValue"], comment="force expected value" ))
          if self.parentWriter.db["CheckValueC"]:
            module.add(SMovB32(dst=sgpr(tmpS01), src=self.parentWriter.db["ValueCExpectedValue"], comment="Move expected value"))
            module.add(self.parentWriter.getCmpAssert(self.parentWriter.asmAssert.eq, vgpr("ValuC+%u"%newSumIdx), sgpr(tmpS01)))

        # sgemm, HPA-bfgemm(b,b,b,b,s,s), and HPA-hgemm(h,h,h,h,s,s)
        # (h,h,h,h,h,h) + HPA (will be converted to (h,h,h,h,s,s)), internal alpha is single
        elif kernel["ProblemType"]["ComputeDataType"].isSingle() or (kernel["ProblemType"]["ComputeDataType"].isHalf() and kernel["ProblemType"]["HighPrecisionAccumulate"]):

          if kernel["ProblemType"]["DataType"].isInt8() and kernel["ProblemType"]["HighPrecisionAccumulate"]:
            if usePK or gwvw > 1:
              if vi % 2 == 0:
                module.add(VCvtI32toF32(dst=vgpr("ValuC+%u"%sumIdxV), src=vgpr("ValuC+%u"%sumIdxV), comment="convert to fp32" ))
                module.add(VCvtI32toF32(dst=vgpr("ValuC+%u"%(sumIdxV+1)), src=vgpr("ValuC+%u"%(sumIdxV+1)), comment="convert to fp32" ))
            else:
              module.add(VCvtI32toF32(dst=vgpr("ValuC+%u"%sumIdxV), src=vgpr("ValuC+%u"%sumIdxV), comment="convert to fp32" ))

          newSumIdx = sumIdxV - self.parentWriter.states.c.startVgprValu
          # Use pk if possible
          if usePK or gwvw > 1:
            if newSumIdx % 2 == 0:
              module.add(VMulPKF32(dst=vgpr("ValuC+%u"%newSumIdx, 2), src0=sgpr("Alpha",2), src1=vgpr("ValuC+%u"%newSumIdx,2), vop3=VOP3PModifiers(op_sel_hi=[0,1,1]), comment="*= alpha (pk)"))
          else:
            module.add(VMulF32(dst=vgpr("ValuC+%u"%newSumIdx), src0=sgpr("Alpha"), src1=vgpr("ValuC+%u"%newSumIdx), comment="*= alpha" ))
          if self.parentWriter.db["ForceExpectedValue"]:
            module.add(VMovB32(dst=vgpr("ValuC+%u"%newSumIdx), src=self.parentWriter.db["ValueCExpectedValue"], comment="force expected value" ))
          if self.parentWriter.db["ForceVSerial"]:
            module.add(VMovB32(dst=vgpr("ValuC+%u"%newSumIdx), src=vgpr("Serial"), comment="force expected value to serial" ))
          if self.parentWriter.db["CheckValueC"]:
            module.add(SMovB32(dst=sgpr(tmpS01), src=self.parentWriter.db["ValueCExpectedValue"], comment="Move expected value"))
            module.add(self.parentWriter.getCmpAssert(self.parentWriter.asmAssert.eq, vgpr("ValuC+%u"%newSumIdx), sgpr(tmpS01)))

        # dgemm
        elif kernel["ProblemType"]["ComputeDataType"].isDouble():
          newSumIdx = sumIdxV * 2 - self.parentWriter.states.c.startVgprValu
          module.add(VMulF64(dst=vgpr("ValuC+%u"%(newSumIdx),2), src0=sgpr("Alpha",2), src1=vgpr("ValuC+%u"%(newSumIdx),2), comment="*= alpha"))
          if usePK:
            module.add(VMulF64(dst=vgpr("ValuC+%u"%(newSumIdx+2),2), src0=sgpr("Alpha",2), src1=vgpr("ValuC+%u"%(newSumIdx+2),2), comment="*= alpha"))

        # single precision complex
        elif kernel["ProblemType"]["ComputeDataType"].isSingleComplex():
          newSumIdx = sumIdxV * 2 - self.parentWriter.states.c.startVgprValu
          tmpVgpr = self.parentWriter.vgprPool.checkOut(1, tag="_applyAlpha_tmpVgpr")
          module.add(VMovB32(dst=vgpr(tmpVgpr), src=vgpr("ValuC+%u"%(newSumIdx)), comment="store Cr"))
          module.add(VMulF32(dst=vgpr("ValuC+%u"%(newSumIdx)), src0=sgpr("Alpha"), src1=vgpr("ValuC+%u"%(newSumIdx)), comment="*= alpha ( Cr = Ar * Cr)"))
          module.add(VMacF32(dst=vgpr("ValuC+%u"%(newSumIdx)), src0=(sgpr("Alpha+1").getMinus()), src1=vgpr("ValuC+%u"%(newSumIdx+1)), comment="*= alpha ( Cr += -Ai * Ci )"))
          module.add(VMulF32(dst=vgpr("ValuC+%u"%(newSumIdx+1)), src0=sgpr("Alpha"), src1=vgpr("ValuC+%u"%(newSumIdx+1)), comment="*= alpha ( Ci = Ar * Ci)"))
          module.add(VMacF32(dst=vgpr("ValuC+%u"%(newSumIdx+1)), src0=sgpr("Alpha+1"), src1=vgpr(tmpVgpr), comment="*= alpha ( Ci += Ai * Cr_backup )"))
          if usePK:
            newSumIdx2 = newSumIdx + 2
            module.add(VMovB32(dst=vgpr(tmpVgpr), src=vgpr("ValuC+%u"%(newSumIdx2)), comment="store Cr"))
            module.add(VMulF32(dst=vgpr("ValuC+%u"%(newSumIdx2)), src0=sgpr("Alpha"), src1=vgpr("ValuC+%u"%(newSumIdx2)), comment="*= alpha ( Cr = Ar * Cr)"))
            module.add(VMacF32(dst=vgpr("ValuC+%u"%(newSumIdx2)), src0=(sgpr("Alpha+1").getMinus()), src1=vgpr("ValuC+%u"%(newSumIdx2+1)), comment="*= alpha ( Cr += -Ai * Ci )"))
            module.add(VMulF32(dst=vgpr("ValuC+%u"%(newSumIdx2+1)), src0=sgpr("Alpha"), src1=vgpr("ValuC+%u"%(newSumIdx2+1)), comment="*= alpha ( Ci = Ar * Ci)"))
            module.add(VMacF32(dst=vgpr("ValuC+%u"%(newSumIdx2+1)), src0=sgpr("Alpha+1"), src1=vgpr(tmpVgpr), comment="*= alpha ( Ci += Ai * Cr_backup )"))
          self.parentWriter.vgprPool.checkIn(tmpVgpr)

        # double precision complex
        elif kernel["ProblemType"]["ComputeDataType"].isDoubleComplex():
          newSumIdx = sumIdxV * 4 - self.parentWriter.states.c.startVgprValu
          vtmp1 = self.parentWriter.vgprPool.checkOutAligned(2, 2, tag="_applyAlpha_vtmp1")
          vtmp2 = self.parentWriter.vgprPool.checkOutAligned(2, 2, tag="_applyAlpha_vtmp2")
          # tmp1 = a.real * b.real (t1 = Ar*Cr)
          module.add(VMulF64(dst=vgpr(vtmp1,2), src0=sgpr("Alpha+0",2), src1=vgpr("ValuC+%u"%(newSumIdx+0),2)))
          # tmp2 = a.imag * b.real (t2 = Ai*Cr)
          module.add(VMulF64(dst=vgpr(vtmp2,2), src0=sgpr("Alpha+2",2), src1=vgpr("ValuC+%u"%(newSumIdx+0),2)))
          # c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag (Cr = -Ai*Ci + t1).
          module.add(VFmaF64(dst=vgpr("ValuC+%u"%(newSumIdx+0),2), src0=sgpr("Alpha+2",2), src1=vgpr("ValuC+%u"%(newSumIdx+2),2).getMinus(), src2=vgpr(vtmp1,2)))
          # c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2 (Ci = Ar*Ci + t2).
          module.add(VFmaF64(dst=vgpr("ValuC+%u"%(newSumIdx+2),2), src0=sgpr("Alpha+0",2), src1=vgpr("ValuC+%u"%(newSumIdx+2),2), src2=vgpr(vtmp2,2)))
          if usePK:
            newSumIdx2 = newSumIdx + 4
            # tmp1 = a.real * b.real
            module.add(VMulF64(dst=vgpr(vtmp1,2), src0=sgpr("Alpha+0",2), src1=vgpr("ValuC+%u"%(newSumIdx2+0),2)))
            # tmp2 = a.imag * b.real
            module.add(VMulF64(dst=vgpr(vtmp2,2), src0=sgpr("Alpha+2",2), src1=vgpr("ValuC+%u"%(newSumIdx2+0),2)))
            # c.real = a.real * b.real - a.imag * b.imag = tmp1 - a.imag * b.imag
            module.add(VFmaF64(dst=vgpr("ValuC+%u"%(newSumIdx2+0),2), src0=sgpr("Alpha+2",2), src1=vgpr("ValuC+%u"%(newSumIdx2+2),2).getMinus(), src2=vgpr(vtmp1,2)))
            # c.imag = a.real * b.imag + a.imag * b.real = a.real * b.imag + tmp2
            module.add(VFmaF64(dst=vgpr("ValuC+%u"%(newSumIdx2+2),2), src0=sgpr("Alpha+0",2), src1=vgpr("ValuC+%u"%(newSumIdx2+2),2), src2=vgpr(vtmp2,2)))
          self.parentWriter.vgprPool.checkIn(vtmp1)
          self.parentWriter.vgprPool.checkIn(vtmp2)
    return module

  def _addSumAlphaWithCBeta(self, kernel, ss, gwvw, elementIdx, vc0, tmpVgpr, cvtVgprStruct):
    module = Module("addSumAlphaWithCBeta #elementIdx%u, vc0 %u"%(elementIdx, vc0))
    for vi in range(0, gwvw):
      dataV = ss.elementData[elementIdx] + int(vi*ss.cfg.numVgprsPerDataPerVI)
      sumIdxV = ss.elementSumIdx[elementIdx] + vi
      if kernel["ProblemType"]["DestDataType"].isHalf():
        if not kernel["ProblemType"]["HighPrecisionAccumulate"]:
          if self.parentWriter.states.asmCaps["HasWMMA_V1"] and kernel["EnableMatrixInstruction"]:
            dataV = ss.elementData[elementIdx] + int(vi / 2 * ss.cfg.numVgprsPerDataPerVI)
            if (vi % 2) == 0:
              module.add(VMulPKF16(dst=vgpr(dataV), src0=sgpr("Beta"), src1=vgpr(dataV+0), \
                    comment="%s = C*beta ei=%u vi=%u"%(vgpr(dataV),elementIdx, vi)))
            else:
              module.add(VLShiftRightB32(dst=vgpr(dataV), shiftHex=16, src=vgpr(dataV), \
                    comment="shift 16bit to get next half of packed ValueC"))
            # dataV+0 = new c = old c*beta + rC
            module.add(VAddPKF16(dst=vgpr("ValuC+%u"%(sumIdxV)), src0=vgpr(dataV), src1=vgpr("ValuC+%u"%(sumIdxV)), \
                comment="sum*alpha + C*beta"))
          elif sumIdxV%2==0 or (not ss.cfg.halfDataRegPerVI and gwvw==1):
            newSumIdxV = sumIdxV // 2 - self.parentWriter.states.c.startVgprValu
            # dataV+0 = new c = old c*beta
            module.add(VMulPKF16(dst=vgpr(dataV), src0=sgpr("Beta"), src1=vgpr(dataV+0), \
                comment="%s = C*beta ei=%u vi=%u"%(vgpr(dataV),elementIdx, vi)))
            # dataV+0 = new c = old c*beta + rC
            module.add(VAddPKF16(dst=vgpr("ValuC+%u"%(newSumIdxV)), src0=vgpr(dataV), src1=vgpr("ValuC+%u"%(newSumIdxV)), \
                comment="sum*alpha + C*beta"))
          else:
            pass # add will have been done previously
        else: # HPA
          newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
          # dataV+0 = new c = old c*beta + rC
          # src0 = beta = f32 = opsel 00
          # src1 = dataV = f16.lo = opsel 10 or 11 depending on even/odd
          # src2 = sumIdxV = f32 = opsel 00
          dataCExternal = ss.elementData[elementIdx] + vi//2
          hi16 = (vi + gwvw*vc0) % 2
          module.add(self.parentWriter.states.mixinst(dst=vgpr("ValuC+%u"%newSumIdxV), src0=sgpr("Beta"), \
              src1=vgpr(dataCExternal), src2=vgpr("ValuC+%u"%newSumIdxV), \
              vop3=VOP3PModifiers(op_sel=[0,hi16,0], op_sel_hi=[0,1,0]),
              comment="//C*=beta"))

      elif kernel["ProblemType"]["DestDataType"].isBFloat16():
        if kernel["ProblemType"]["HighPrecisionAccumulate"]:
          # dataV+0 = new c = old c*beta + rC
          # src0 = beta = f32 = opsel 00
          # src1 = dataV = f16.lo = opsel 10 or 11 depending on even/odd
          # src2 = sumIdxV = f32 = opsel 00
          dataCExternal = ss.elementData[elementIdx] + vi//2
          module.add(VCvtBF16toFP32(dst=vgpr(tmpVgpr), src=vgpr(dataCExternal), vgprMask=vgpr(cvtVgprStruct.vgprBf16Mask), vi=(vi)))
          newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
          module.add(VMacF32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr(tmpVgpr), src1=sgpr("Beta"), \
              comment="finalSum = sum*alpha + C*beta"))
      elif kernel["ProblemType"]["DestDataType"].isSingle():
        newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
        module.add(VMacF32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr(dataV+0), src1=sgpr("Beta"), \
            comment="finalSum = sum*alpha + C*beta"))

      elif kernel["ProblemType"]["DestDataType"].isInt8():
        if kernel["ProblemType"]["HighPrecisionAccumulate"]:
          dataCExternal   = ss.elementData[elementIdx] + (vi // 4)
          byteIdx       = vi %  4
          if (vi%4) != 3:
            module.add(VMovB32(dst=vgpr(tmpVgpr+1), src=hex(byteIdx * 8), comment="value = %u"%(byteIdx * 8)))
            module.add(VBfeI32(dst=vgpr(tmpVgpr), src0=vgpr(dataCExternal), src1=vgpr(tmpVgpr+1), src2=8, comment="int8 to int32"))
          else:
            module.add(VAShiftRightI32(dst=vgpr(tmpVgpr), shiftHex=24, src=vgpr(dataCExternal), comment="int8 to int32"))

          newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
          if kernel["ProblemType"]["ComputeDataType"].isSingle():
            module.add(VCvtI32toF32(dst=vgpr(tmpVgpr), src=vgpr(tmpVgpr), comment="convert to fp32" ))
            module.add(VMacF32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr(tmpVgpr), src1=sgpr("Beta"), \
                               comment="finalSum = sum*alpha + C*beta"))
          else:
            module.add(VMulLOU32(dst=vgpr(tmpVgpr), src0=sgpr("Beta"), src1=vgpr(tmpVgpr), comment="C = C*beta"))
            module.add(VAddU32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr(tmpVgpr), src1=vgpr("ValuC+%u"%newSumIdxV), comment="finalSum = sum*alpha + C*beta"))

      elif kernel["ProblemType"]["DestDataType"].isInt32():
        newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
        if kernel["ProblemType"]["ComputeDataType"].isSingle():
          module.add(VCvtI32toF32(dst=vgpr(dataV+0), src=vgpr(dataV+0), comment="convert to fp32" ))
          module.add(VMacF32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr(dataV+0), src1=sgpr("Beta"), comment="finalSum = sum*alpha + C*beta"))
        else:
          # assume we will need to replace v_mac_f32 with v_add_u32 and s_mul_lo_i32
          # v_mad_i32_i24
          # module.add(VMadI32I24(dst=vgpr("ValuC+%u"%sumIdxV), src0=vgpr(dataV+0), src1=sgpr("Beta"), src2=vgpr("ValuC+%u"%sumIdxV), \
          #     comment="finalSum = sum*alpha + C*beta"))
          module.add(VMulLOU32(dst=vgpr(dataV+0), src0=sgpr("Beta"), src1=vgpr(dataV+0), comment="C = C*beta"))
          module.add(VAddU32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr(dataV+0), src1=vgpr("ValuC+%u"%newSumIdxV), comment="finalSum = sum*alpha + C*beta"))

      elif kernel["ProblemType"]["DestDataType"].isDouble():
        newSumIdxV = sumIdxV * 2 - self.parentWriter.states.c.startVgprValu
        # dataV+0 = new c = old c*beta
        module.add(VFmaF64(dst=vgpr("ValuC+%u"%(newSumIdxV),2), src0=vgpr(dataV+0,2), src1=sgpr("Beta",2), src2=vgpr("ValuC+%u"%(newSumIdxV),2), \
            comment="finalSum = sum*alpha + C*beta"))

      # single precision complex
      elif kernel["ProblemType"]["DestDataType"].isSingleComplex():
        newSumIdxV = sumIdxV * 2 - self.parentWriter.states.c.startVgprValu
        module.add(VMacF32(dst=vgpr("ValuC+%u"%(newSumIdxV)), src0=vgpr(dataV+0), src1=sgpr("Beta"), comment="finalSum Cr += old Cr * Br"))
        module.add(VMacF32(dst=vgpr("ValuC+%u"%(newSumIdxV)), src0=vgpr(dataV+1), src1=sgpr("Beta+1").getMinus(), comment="finalSum Cr += old Ci * -Bi"))
        module.add(VMacF32(dst=vgpr("ValuC+%u"%(newSumIdxV+1)), src0=vgpr(dataV+1), src1=sgpr("Beta"), comment="finalSum Ci += old Ci * Br"))
        module.add(VMacF32(dst=vgpr("ValuC+%u"%(newSumIdxV+1)), src0=vgpr(dataV+0), src1=sgpr("Beta+1"), comment="finalSum Ci += old Cr * Bi"))

      # double precision complex
      elif kernel["ProblemType"]["DestDataType"].isDoubleComplex():
        newSumIdxV = sumIdxV * 4 - self.parentWriter.states.c.startVgprValu
        module.add(VFmaF64(dst=vgpr("ValuC+%u"%(newSumIdxV+0),2), src0=vgpr(dataV+0,2), src1=sgpr("Beta+0",2), src2=vgpr("ValuC+%u"%(newSumIdxV+0),2), comment="c.real += a.real * b.real"))
        module.add(VFmaF64(dst=vgpr("ValuC+%u"%(newSumIdxV+0),2), src0=vgpr(dataV+2,2), src1=sgpr("Beta+2",2).getMinus(), src2=vgpr("ValuC+%u"%(newSumIdxV+0),2), comment="c.real -= a.imag * b.imag"))
        module.add(VFmaF64(dst=vgpr("ValuC+%u"%(newSumIdxV+2),2), src0=vgpr(dataV+0,2), src1=sgpr("Beta+2",2), src2=vgpr("ValuC+%u"%(newSumIdxV+2),2), comment="c.imag += a.real * b.imag"))
        module.add(VFmaF64(dst=vgpr("ValuC+%u"%(newSumIdxV+2),2), src0=vgpr(dataV+2,2), src1=sgpr("Beta+0",2), src2=vgpr("ValuC+%u"%(newSumIdxV+2),2), comment="c.imag += a.imag * b.real"))

      # float8 precision
      elif kernel["ProblemType"]["DestDataType"].isAnyFloat8():
        if kernel["ProblemType"]["HighPrecisionAccumulate"]:
          dataCExternal   = ss.elementData[elementIdx] + (vi // 4)
          newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
          # Generate single f32 code if edge is detected.
          isPK = False
          if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):
            if self.parentWriter.states.archCaps["VOP3ByteSel"]:
              sb = 0 if self.gwvw == 1 else 1
              if not self.amdClangVersion.major >= 19:
                module.add(VCvtFP8toF32(dst=vgpr(tmpVgpr), src=vgpr(dataCExternal), vop3=VOP3PModifiers(op_sel=[0,sb])))
              else:
                module.add(VCvtFP8toF32(dst=vgpr(tmpVgpr), src=vgpr(dataCExternal), vop3=VOP3PModifiers(byte_sel=[sb])))
            else:
              sb = SelectBit.BYTE_0 if self.gwvw == 1 else SelectBit.BYTE_2
              module.add(VCvtFP8toF32(dst=vgpr(tmpVgpr), src=vgpr(dataCExternal), sdwa=SDWAModifiers(src0_sel=sb)))
          # Original packed route
          elif vi%2 == 1:
            continue
          else:
            isPK = True
            module.add(ECvtPkFP8toF32(dst=vgpr(tmpVgpr, 2), src=vgpr(dataCExternal), sel=HighBitSel.LOW if vi%4 == 0 else HighBitSel.HIGH))
          module.add(SNop(waitState=0))
          if kernel["ProblemType"]["ComputeDataType"].isSingle():
            module.add(VMacF32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr(tmpVgpr), src1=sgpr("Beta"), comment="finalSum = sum*alpha + C*beta"))
            if isPK:
              module.add(VMacF32(dst=vgpr("ValuC+%u"%(newSumIdxV+1)), src0=vgpr(tmpVgpr+1), src1=sgpr("Beta"), comment="finalSum = sum*alpha + C*beta (PK)"))
      # bfloat8 precision
      elif kernel["ProblemType"]["DestDataType"].isAnyBFloat8():
        if kernel["ProblemType"]["HighPrecisionAccumulate"]:
          dataCExternal   = ss.elementData[elementIdx] + (vi // 4)
          newSumIdxV = sumIdxV - self.parentWriter.states.c.startVgprValu
          # Generate single f32 code if edge is detected.
          isPK = False
          if ((vi + 1) == self.gwvw) and ((self.gwvw % 2) == 1):
            if self.parentWriter.states.archCaps["VOP3ByteSel"]:
              sb = 0 if self.gwvw == 1 else 1
              if not self.amdClangVersion.major >= 19:
                module.add(VCvtBF8toF32(dst=vgpr(tmpVgpr), src=vgpr(dataCExternal), vop3=VOP3PModifiers(op_sel=[0,sb])))
              else:
                module.add(VCvtBF8toF32(dst=vgpr(tmpVgpr), src=vgpr(dataCExternal), vop3=VOP3PModifiers(byte_sel=[sb])))
            else:
              sb = SelectBit.BYTE_0 if self.gwvw == 1 else SelectBit.BYTE_2
              module.add(VCvtBF8toF32(dst=vgpr(tmpVgpr), src=vgpr(dataCExternal), sdwa=SDWAModifiers(src0_sel=sb)))
          # Original packed route
          elif vi%2 == 1:
            continue
          else:
            isPK = True
            module.add(ECvtPkBF8toF32(dst=vgpr(tmpVgpr, 2), src=vgpr(dataCExternal), sel=HighBitSel.LOW if vi%4 == 0 else HighBitSel.HIGH))
          module.add(SNop(waitState=0))
          if kernel["ProblemType"]["ComputeDataType"].isSingle():
            module.add(VMacF32(dst=vgpr("ValuC+%u"%newSumIdxV), src0=vgpr(tmpVgpr), src1=sgpr("Beta"), comment="finalSum = sum*alpha + C*beta"))
            if isPK:
              module.add(VMacF32(dst=vgpr("ValuC+%u"%(newSumIdxV+1)), src0=vgpr(tmpVgpr+1), src1=sgpr("Beta"), comment="finalSum = sum*alpha + C*beta (PK)"))
    return module

def copyData(computeDataType, elementSumIdx, gwvw, vgprStart, direction=0):
  module = Module("Copy Data")
  vi = 0
  while vi < gwvw:
    sumIdxV = elementSumIdx + vi
    if computeDataType.isHalf() or computeDataType.isBFloat16():
      if (sumIdxV % 2 != 0):
        vi += 1
        continue
      vgprIdx = elementSumIdx + vi // 2
      if (vi + 1 < gwvw) and ((vgprStart + (vi // 2)) % 2 == 0) and (vgprIdx % 2 == 0):
        module.add(VMovB64(dst=vgpr(vgprStart + (vi // 2), 2), src=vgpr(vgprIdx, 2)))
        vi += 2
      else:
        module.add(VMovB32(dst=vgpr(vgprStart + (vi // 2)), src=vgpr(vgprIdx)))
        vi += 1
    elif computeDataType.isSingle() or computeDataType.isInt32():
      vgprIdx = sumIdxV
      if (vi + 1 < gwvw) and ((vgprStart + vi) % 2 == 0) and (vgprIdx % 2 == 0):
        module.add(VMovB64(dst=vgpr(vgprStart + vi, 2), src=vgpr(vgprIdx, 2)))
        vi += 2
      else:
        module.add(VMovB32(dst=vgpr(vgprStart + vi), src=vgpr(vgprIdx)))
        vi += 1
    elif computeDataType.isDouble():
      vgprIdx = elementSumIdx + vi * 2
      module.add(VMovB64(dst=vgpr(vgprStart + vi * 2, 2), src=vgpr(vgprIdx, 2)))
      vi += 1
    else:
      assert 0

  if direction == 1:
    for i in module.items():
      srcs = i.srcs
      tmp = srcs[0]
      srcs[0] = i.dst
      i.dst = tmp
      i.srcs = srcs
  return module

def convertData(gwvw, elementSumIdx, cvtType: CvtType, roundType: RoundType = RoundType.ROUND_UP, inputPrefix="", prefixOffset=0):
  module = Module("ConvertData")
  for vi in range(0, gwvw):
    sumIdxV = elementSumIdx + vi
    formatVgpr = formatting(sumIdxV, inputPrefix, prefixOffset)
    if cvtType == CvtType.CVT_F32_to_I32:
        if roundType == RoundType.ROUND_TO_NEAREST_EVEN:
          module.add(VRndneF32(dst=vgpr(formatVgpr), src=vgpr(formatVgpr), comment=" round to even"))
        module.add(VCvtF32toI32(dst=vgpr(formatVgpr), src=vgpr(formatVgpr), comment=" convert fp32 to i32"))
    elif cvtType == CvtType.CVT_I32_to_F32:
        module.add(VCvtI32toF32(dst=vgpr(formatVgpr), src=vgpr(formatVgpr), comment=" convert to fp32"))
    else:
      #TODO add other convert types here.
      assert 0
  return module
