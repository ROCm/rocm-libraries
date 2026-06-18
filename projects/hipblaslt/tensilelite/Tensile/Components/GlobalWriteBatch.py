################################################################################
#
# Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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
from rocisa.container import SMEMModifiers, VOP3PModifiers, MUBUFModifiers, \
  SDWAModifiers, replaceHolder, EXEC, VCC, vgpr, sgpr, ContinuousRegister, mgpr
from rocisa.enum import CvtType, HighBitSel, RoundType, SaturateCastType, SelectBit
from rocisa.instruction import BufferAtomicAddF32, BufferAtomicCmpswapB32, \
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
  VFmaF64, VFmaMixF32, VAndB32, VLShiftLeftB32, VPermlane16SwapB32, VPermlane32SwapB32, \
  VLShiftRightB32, VMacF32, VMadMixF32, VMaxF32, VMovB32, VMovB64, VMulF32, VMulF64, \
  VMulLOU32, VMulPKF16, VMulPKF32, VPackF16toB32, VReadfirstlaneB32, VRndneF32, VCvtBF16toFP32, \
  VCmpClassF32, VMed3F32, VPrngB32, VCvtSRF32toFP8, MacroInstruction
from rocisa.functions import vectorStaticMultiply
from rocisa.macro import PseudoRandomGeneratorModule

from ..Common import DataDirection, SemanticVersion
from ..Common.DataType import DataType
from ..Component import GlobalWriteComponents
from ..Component import Component
from ..SolutionStructs import Solution
from ..Activation import ActivationModule
from ..AsmStoreState import StoreState
from ..AsmAddressCalculation import AddrCalculation
from ..Components.PackData import formatting, PackData_F16, PackData_BF16, PackData_FLOAT8, PackData_FLOAT8_fnuz
from rocisa.instruction import ECvtF16toF32, ECvtPkFP8toF32, ECvtPkBF8toF32

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
    self._emitAdd(module)
    # UseSubtileImpl with bias/SAV: drain LDS reads and sync waves after alpha
    # multiply to prevent cross-wave LDS corruption from ds_bpermute.
    if self.kernel.get("UseSubtileImpl") and \
       (self.parentWriter.states.useBias != DataDirection.NONE or \
        self.kernel["ProblemType"].get("UseScaleAlphaVec", 0)):
      module.add(SWaitCnt(dscnt=0, comment="drain bias/SAV LDS reads"))
      module.add(SBarrier(comment="sync waves before subtile paired stores"))
    self._epilog(module)
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
        module.add(SCBranchSCC0(clsLabel.getLabelName(), "loop while counter != 0"))
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
      self.loadsScaleAlphaVecIssued += addEpilogueLoad(modGwvwScaleAlpha, "ScaleAlphaVec", addrScaleAlphaVecVgpr, self.addrScaleAlphaVec, dataScaleAlphaVec, self.loadedDataScaleAlphaVec, addrCalc.scaleAlphaVecOffset[self.factorDim], factor_gwvw, localReferenceVgpr, self.factorDim, self.factorDim, skipLoad=skipLoad, comment="load scaleAlpha")
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
    self.biasLoadIssued = []
    self.scaleAVecLoadIssued = []
    self.scaleBVecLoadIssued = []
    self.scaleAlphaVecLoadIssued = []

    self.loadedDataBeta = {}
    self.loadedDataE = {}
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
    vlcntTotalIssued = self.loadsBetaIssued + self.loadsEIssued
    dscntTotalIssued = self.localLoadsBiasIssued + self.loadsScaleAVecIssued + self.loadsScaleBVecIssued + self.loadsScaleAlphaVecIssued
    waitCnter = [vlcntTotalIssued, dscntTotalIssued]
    for elementIdx in range(0, len(self.batchElements)):
      element = self.batchElements[elementIdx]
      addrCalc: AddrCalculation = self.ss.elementAddr[elementIdx]
      addr = addrCalc.addrDVgpr
      dataE = self.ss.elementDataE[elementIdx]
      dataBias = self.ss.elementDataBias[elementIdx]
      dataScaleAVec = self.ss.elementDataScaleAVec[elementIdx]
      dataScaleBVec = self.ss.elementDataScaleBVec[elementIdx]
      dataScaleAlphaVec = self.ss.elementDataScaleAlphaVec[elementIdx]
      mask = self.ss.elementMask[elementIdx]
      vc0 = element[3]
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
      if self.kernel["ActivationFuncCall"]:
        if (activationCDataType == self.kernel["ProblemType"]["DestDataType"]) and \
          (activationCDataType != self.kernel["ProblemType"]["ComputeDataType"]) and ((self.kernel["ProblemType"]["UseScaleCD"] == False) or (self.kernel["ProblemType"]["UseScaleAlphaVec"] == False)):
          isActivationInsertAfter = True
        activationModule = Module("ActivationFuncCall")
        if (not mergeActFuncCall) and (not isActivationInsertAfter):
          activationModule.appendModule (copyData(activationCDataType, gradientInput, self.gwvw, \
            self.activationSetPCStruct.vgprActCopy))
        activationModule.add(SSwapPCB64(dst=sgpr(self.activationSetPCStruct.sgprOffsetBack, 2), \
          src=sgpr(self.activationSetPCStruct.sgprOffsetActivation, 2)))
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
        activationModule = self.parentWriter.getActivationActivationComputeType(self.kernel, self.activation, \
          self.activationTypeStr, self.gwvw, gradientInput, gradientInput, self.tmpVgpr, self.tmpSgpr, satInt8, enableValuC)
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
      if self.kernel["ProblemType"]["HighPrecisionAccumulate"] and (self.kernel["_GlobalAccumulation"] != 'MultipleBuffer'):
        if self.kernel["ActivationFuncCall"] and activationCDataType == self.kernel["ProblemType"]["DestDataType"]:
          destIdx = self.activationSetPCStruct.vgprActCopy
        else:
          destIdx = self.ss.elementSumIdx[elementIdx]
        if self.kernel["ProblemType"]["DestDataType"].isHalf():
          # For UseSubtileImpl non-edge: paired dwordx4 path handles packing in _emit16bitSubtilePairedStore.
          if not is16bitSubtile:
            packModule = self.packdata(self.gwvw, destIdx, self.ss.elementSumIdx[elementIdx], inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
        elif self.kernel["ProblemType"]["DestDataType"].isBFloat16():
          # For UseSubtileImpl non-edge: paired dwordx4 path handles packing in _emit16bitSubtilePairedStore.
          if not is16bitSubtile:
            packModule = self.packdata(self.gwvw, destIdx, self.ss.elementSumIdx[elementIdx], bf16CVTVgprStruct=self.cvtVgprStruct,
                                       tmpS01=self.tmpS01, laneSGPRC=self.laneSGPRC, inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
        elif self.kernel["ProblemType"]["DestDataType"].isAnyFloat8():
          if self.kernel["ProblemType"]["StochasticRounding"]:
            # Note: Current stochastic rounding FP8 converter does not support pack version
            convertModule = stochasticRoundingCvt(self, gwvw=self.gwvw, destIdx=destIdx, elementSumIdx=self.ss.elementSumIdx[elementIdx], fp8CVTVgprStruct=self.cvtVgprStruct, \
                                                  tmpS01=self.tmpS01, laneSGPRC=self.laneSGPRC, vgprTmp=vgprRND, inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
          else:
            packModule = self.packdata(self.gwvw, destIdx, self.ss.elementSumIdx[elementIdx], fp8CVTVgprStruct=self.cvtVgprStruct, \
                                       tmpS01=self.tmpS01, laneSGPRC=self.laneSGPRC, inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
        elif self.kernel["ProblemType"]["DestDataType"].isAnyBFloat8():
          # TODO: BF8 stochastic rounding is not yet supported here.
          #       VCvtSRF32toBF8 instruction exists but stochasticRoundingCvt() only emits VCvtSRF32toFP8.
          #       To support BF8 SR: add SR branch here, generalize stochasticRoundingCvt() to accept bf8CVTVgprStruct,
          #       and select VCvtSRF32toBF8 based on DestDataType.
          packModule = self.packdata(self.gwvw, destIdx, self.ss.elementSumIdx[elementIdx], bf8CVTVgprStruct=self.cvtVgprStruct, \
                                     tmpS01=self.tmpS01, laneSGPRC=self.laneSGPRC, inputPrefix="ValuC+", prefixOffset=self.parentWriter.states.c.startVgprValu)
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
        module.add(convertModule)
        module.add(packModule)
        module.add(gradientCvtModule)
        module.add(activationModule)
      else:
        module.add(gradientCvtModule)
        module.add(activationModule)
        module.add(scaleDModule)
        module.add(biasReductionModule)
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
        if is16bitSubtile:
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
                storeCodeModule.add(tmpStoreCode)
                storeCodeModule.add(SBranch(labelName=afterPairedLabel.getLabelName(),
                                            comment="skip scalar fallback"))
                storeCodeModule.add(fallbackLabel)
                tmpFallbackCode = self._emit16bitSubtileScalarStore(partnerAddrCalc, sumIdx0, prefixOffset, tt0 - 1, blockIdxM=blockIdxM, blockIdxN=blockIdxN)
                storeCodeModule.add(tmpFallbackCode)
                storeCodeModule.add(afterPairedLabel)
              else:
                tmpStoreCode = self._emit16bitSubtilePairedStore(partnerAddrCalc, sumIdx0, sumIdx1, prefixOffset, tt0 - 1, blockIdxM=blockIdxM, blockIdxN=blockIdxN)
                storeCodeModule.add(tmpStoreCode)
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
              storeCodeModule.add(tmpStoreCode)
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
              storeCodeModule.add(tmpStoreCode)
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
            self._emitAlign8ExecMask(storeCodeModule, self.tmpS01, self.tmpS23, blockIdxM, blockIdxN,
                                     mGuardOffset=1, rowScaleShift=2)
            storeCodeModule.add(self.getEdgeMovInstType()(EXEC(), sgpr(self.tmpS01, self.laneSGPRC), "apply exec mask"))
          # _emitOverrideRows reused from the top of this store loop (see _lookaheadRowInc).
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
      if addrScaleShift:
        module.add(VLShiftRightB32(dst=vgpr(vAddrScratch), shiftHex=addrScaleShift,
                                   src=vgpr(addrDVgpr), comment=f"scale addrDVgpr bpe {bpeCurr}->{bpeDest}"))
        module.add(VAddU32(dst=vgpr(vAddrScratch), src0=vgpr(vAddrScratch), src1=vgpr(vLGDelta),
                           comment="adjusted D addr = scaled addrDVgpr + lane_group*8"))
      else:
        module.add(VAddU32(dst=vgpr(vAddrScratch), src0=vgpr(addrDVgpr), src1=vgpr(vLGDelta),
                           comment="adjusted D addr = addrDVgpr + lane_group*8"))
      if useAlign8:
        self._emitAlign8ExecMask(module, self.tmpS01, self.tmpS23, blockIdxM, blockIdxN,
                                 mGuardOffset=2, rowScaleShift=1)

    module.add(self._emitSubtilePackedPermute(vPack, vPermAddr, addrWhilePermuting=emitAddrWhilePermuting))

    if useAlign8:
      tmpS = self.tmpS01
      module.add(self.getEdgeMovInstType()(EXEC(), sgpr(tmpS, self.laneSGPRC), "apply exec mask"))

    module.addComment1("buffer_store_dwordx4: write 8 16bit values (4 dwords, 2-aligned src)")
    module.add(BufferStoreB128(
      src=vgpr(vPack, 4),
      vaddr=vgpr(vAddrScratch),
      saddr=sgpr("SrdD", 4),
      soffset=0,
      mubuf=MUBUFModifiers(offen=True, offset12=globalOffset, glc=isGlc, slc=isSlc, nt=isNT),
      comment=f"16bit paired dwordx4 store tt0={tt0},{tt0+1}"
    ))

    if useAlign8:
      module.add(self.getEdgeMovInstType()(EXEC(), -1, "restore exec"))

    # WAR hazard: buffer_store_dwordx4 reads vPack[0:3] as source operands.
    # The next paired store's v_cvt_pk_bf16_f32 will overwrite vPack.
    # Insert nop to ensure the store has latched its source VGPRs.
    module.add(SNop(waitState=0, comment="1 wait state: WAR hazard between store src and next pack dst"))

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
    mt0bpe = self.kernel["MacroTile0"] * bpe

    module.addComment1("compute per-lane orphan vaddr = N_col_off + LG_M_off + wg0_M_off [+ wave offsets]")

    # N-col byte offset: (lane_id & 15) * StrideD1J * bpe
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

    module.addComment1(f"buffer_store_b64: write 4 {typeStr} M-rows at fixed N-col (orphan subtile)")
    module.add(BufferStoreB64(
      src=vgpr(vPack+0, 2),
      vaddr=vgpr(vPack+2),
      saddr=sgpr("SrdD", 4),
      soffset=0,
      mubuf=MUBUFModifiers(offen=True, offset12=globalOffset, glc=isGlc, slc=isSlc, nt=isNT),
      comment=f"orphan tt0={tt0} vc=0..3: 4 consecutive M-rows at fixed N-col"
    ))

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

# F32 to FP8 stochastic rounding conversion
def stochasticRoundingCvt(self, gwvw, destIdx, elementSumIdx, fp8CVTVgprStruct, tmpS01, laneSGPRC, vgprTmp, inputPrefix="", prefixOffset=0):
  vgprFp8NanInf = fp8CVTVgprStruct.vgprFp8NanInf
  vgprFp8Temp   = fp8CVTVgprStruct.vgprFp8Temp
  vgprFp8Min    = fp8CVTVgprStruct.vgprFp8Min
  vgprFp8Max    = fp8CVTVgprStruct.vgprFp8Max
  vRand = vgprTmp #seed
  if not self.parentWriter.states.asmCaps["v_prng_b32"]:
    vTemp0 = vgprTmp+1
    vTemp1 = vgprTmp+2

  module = Module("StochasticRoundingCvt")

  for vi in range(0, gwvw):
    sumIdxV = elementSumIdx + vi
    formatVgpr = formatting(sumIdxV, inputPrefix, prefixOffset)
    d = destIdx + vi//4

    module.add(VCmpClassF32(dst=sgpr(tmpS01,laneSGPRC), src0=vgpr(formatVgpr), src1=vgpr(vgprFp8NanInf), comment="Nan and +/- inf"))
    module.add(VMed3F32(dst=vgpr(vgprFp8Temp), src0=vgpr(formatVgpr), src1= vgpr(vgprFp8Min), src2=vgpr(vgprFp8Max)))
    module.add(VCndMaskB32(dst=vgpr(formatVgpr), src0=vgpr(vgprFp8Temp), src1=vgpr(formatVgpr), src2=sgpr(tmpS01,laneSGPRC)))

    if self.parentWriter.states.asmCaps["v_prng_b32"]:
      # NOTE: Current PRNG seed implementation simply uses the value to be converted directly as seed.
      # For thread ID-based seed design, see the legacy PRND_GENERATOR approach in tensilelite/rocisa/rocisa/include/macro.hpp
      module.add(VPrngB32(dst=vgpr(vRand),src=vgpr(formatVgpr),comment="Pseudo Random Number Generator"))
    else:
      if self.parentWriter.states.asmCaps["HasVgprMSB"]:
        module.add(PseudoRandomGeneratorModule(vRand, vgprFp8Temp, vTemp0, vTemp1))
      else:
        module.add(MacroInstruction(name="PRND_GENERATOR", args=[vRand, vgprFp8Temp, vTemp0, vTemp1]))
    # sels=[vi%4] selects which byte within the packed VGPR to write the FP8 value to
    module.add(VCvtSRF32toFP8(dst=vgpr(d), src0=vgpr(formatVgpr), src1=vgpr(vRand), sels=[vi%4]))

  return module
